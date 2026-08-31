"""
CO Rovibrational Flux Extraction Module

Fits multiple CO lines and contaminating lines simultaneously from IFU spectra,
extracting and summing the total CO flux to create integrated flux maps.
"""

import sys
import numpy as np
import pandas as pd
import time
import warnings
import traceback
from threading import Lock
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy import units as u
from concurrent.futures import ThreadPoolExecutor, as_completed

# RANSAC for continuum fitting (from Pascucci)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*All-NaN slice encountered.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrame concatenation with empty or all-NA entries.*')

from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline

# REUSE: Import helper functions from gaussian_fitter
from .gaussian_fitter import (
    compute_noise_std,
    GaussianFitResult, gaussian,
)
from .utils import get_spectral_resolution
from .unit_converter import (
    integrated_flux_to_cgs, get_wave_unit, _resolve_flux_unit, convert_area_to_cgs,
    extract_units, extract_value_from_quantity,
)

# Derived from fit_glines' own bounds (center +/-1sigma, fitted sigma in [0.5,2]x instrumental):
# 3*(2x instrumental sigma) + 1sigma center shift = 7sigma is as far as real flux can ever land.
DEFAULT_SIGMA_MULTIPLIER = 7.0

# Display/diagnostic-only reduced constant (does not affect curve_fit)
DISPLAY_CROP_SIGMA_MULTIPLIER = 3.0

# ================== Unit Utilities ==================

def fit_auto_continuum(wavelength, flux, window_size=51, sigma_threshold=2.0,
                                peak_width=1, peak_distance=3, peak_exclusion=3,
                                iterations=2, smoothing_factor=5):
    """
    Fit continuum using peak detection and median filter method: finds emission peaks,
    excludes them, and fits a smooth continuum through the rest via median filtering and
    spline smoothing. Same method used by the GUI's Automatic Continuum button.

    Args:
        wavelength (array): Wavelength array
        flux (array): Flux array
        window_size (int): Size of median filter window (must be odd)
        sigma_threshold (float): Prominence threshold for peak detection, in units of sigma
        peak_width (int): Minimum width of peaks in pixels
        peak_distance (int): Minimum distance between peaks in pixels
        peak_exclusion (int): Number of pixels to exclude on each side of a peak
        iterations (int): Number of iterative sigma clipping iterations
        smoothing_factor (float): Spline smoothing parameter divisor

    Returns:
        continuum (array): Fitted continuum matching wavelength, or all-NaN if fitting fails
    """
    try:
        # Remove NaN and infinite values
        mask = np.isfinite(wavelength) & np.isfinite(flux)
        wavelength_clean = wavelength[mask]
        flux_clean = flux[mask]
        
        if len(wavelength_clean) < 10:
            return np.full_like(wavelength, np.nan)
        
        # STEP 1: Initial median filter to smooth noise
        flux_smoothed = median_filter(flux_clean, size=window_size, mode='reflect')
        
        # STEP 2: Find peaks (emission lines) in the spectrum
        peaks, properties = find_peaks(
            flux_clean,
            prominence=np.std(flux_clean) * sigma_threshold,
            width=peak_width,
            distance=peak_distance
        )
        
        # STEP 3: Create mask to exclude peaks and surrounding region
        exclude_mask = np.zeros(len(flux_clean), dtype=bool)
        for peak_idx in peaks:
            left_edge = max(0, peak_idx - peak_exclusion)
            right_edge = min(len(flux_clean), peak_idx + peak_exclusion + 1)
            exclude_mask[left_edge:right_edge] = True
        
        # STEP 4: Fit continuum using only non-peak regions
        good_indices = np.where(~exclude_mask)[0]
        if len(good_indices) < 10:
            # Fallback if too many peaks excluded
            exclude_mask = np.zeros(len(flux_clean), dtype=bool)
        
        # Apply median filter to good (non-peak) regions
        flux_good = flux_clean.copy()
        flux_good[exclude_mask] = np.nan
        
        # Interpolate over NaN (peak) regions
        if np.any(exclude_mask):
            flux_good[exclude_mask] = np.interp(
                wavelength_clean[exclude_mask],
                wavelength_clean[~exclude_mask],
                flux_clean[~exclude_mask]
            )
        
        # Apply median filter to the interpolated data
        continuum = median_filter(flux_good, size=window_size, mode='reflect')
        
        # STEP 5: Iterative sigma clipping for fine-tuning
        for iteration in range(iterations):
            residuals = flux_clean - continuum
            sigma = np.std(residuals[~exclude_mask])
            
            # Only clip points ABOVE continuum (emission) that weren't already excluded
            outliers = (residuals > 2.0 * sigma) & (~exclude_mask)
            
            if np.sum(outliers) == 0:
                break  # No more outliers found
            
            # Update mask
            exclude_mask |= outliers
            
            # Re-interpolate and filter
            flux_good = flux_clean.copy()
            flux_good[exclude_mask] = np.interp(
                wavelength_clean[exclude_mask],
                wavelength_clean[~exclude_mask],
                flux_clean[~exclude_mask]
            )
            continuum = median_filter(flux_good, size=window_size, mode='reflect')
        
        # STEP 6: Final smoothing with spline
        try:
            spline = UnivariateSpline(
                wavelength_clean[~exclude_mask], 
                continuum[~exclude_mask], 
                s=len(wavelength_clean) / smoothing_factor,
                k=3
            )
            continuum_smooth = spline(wavelength_clean)
        except Exception:
            continuum_smooth = continuum
        
        # Create full continuum array matching original wavelength
        continuum_values = np.full_like(wavelength, np.nan)
        continuum_values[mask] = continuum_smooth
        
        return continuum_values
        
    except Exception as e:
        # Return NaN array if anything fails
        return np.full_like(wavelength, np.nan)


# ================== Gaussian Functions ==================

def many_gaussians_list(xl, a, b, *params):
    """
    Sum of multiple Gaussians plus a linear continuum, vectorized over points and lines at
    once (broadcast, not a per-point/per-line Python loop) since curve_fit's Jacobian
    re-evaluates this model on every iteration. Same math, just how it's computed.

    Args:
        xl (array): Wavelength values to evaluate at
        a (float): Continuum baseline (a in a + b*x)
        b (float): Continuum slope (b in a + b*x)
        *params (float): Gaussian params, grouped as [amp0, cen0, sig0, amp1, cen1, sig1, ...]

    Returns:
        model (array): Sum of all Gaussians plus continuum, evaluated at xl
    """
    N = 3  # Parameters per Gaussian
    xl = np.asarray(xl)
    params = np.asarray(params)
    min_freq = np.min(xl)
    amps, cens, sigs = params[0::N], params[1::N], params[2::N]
    # A per-point, per-line Python loop here was profiled at 81% of one chunk's fit time --
    # the broadcast below computes the same (n_points, n_lines) grid without it.
    gaussians = amps[np.newaxis, :] * np.exp(
        -0.5 * ((xl[:, np.newaxis] - cens[np.newaxis, :]) / sigs[np.newaxis, :]) ** 2
    )
    return a + b * (xl - min_freq) + gaussians.sum(axis=1)


# ================== Line Detection ==================

def find_lines(xwave, yflux, lines_to_search, contaminating_lines=None, remove_duplicate=False, snr_threshold=3.0, noise_std=None):
    """
    Identify which spectral lines are detected in a spectrum and assign a starting max_amplitude.

    Performs peak detection on a fine wavelength grid around each line and compares
    to local noise. Handles duplicate/close lines and contaminating line avoidance.

    Args:
        xwave (array): Wavelength array
        yflux (array): Flux array
        lines_to_search (pd.DataFrame): Columns 'line_wave_um', 'species', others optional
        contaminating_lines (pd.DataFrame): Optional column 'line_wave_um' for lines to avoid
        remove_duplicate (bool): True to remove duplicate detections (keep the strongest per group)
        snr_threshold (float): Signal-to-noise threshold for line detection, in units of sigma
        noise_std (float): Precomputed noise estimate, to avoid recomputing it on repeat calls
            over the same (xwave, yflux) -- e.g. extract_co_flux_from_spectrum's separate
            CO/contaminant passes. Computed fresh if None.

    Returns:
        lines (pd.DataFrame): Columns line, species, disposition (0/1), max_amplitude, line_flux
    """
    # Suppress repetitive output (called for every pixel)
    # print('>>> Finding lines and removing duplicates...')
    xres = np.nanmean(np.diff(xwave))

    # Estimate noise level for significance testing
    if noise_std is None:
        try:
            noise_std = compute_noise_std(xwave, yflux, pdeg=1)
            if noise_std is None or not np.isfinite(noise_std) or noise_std <= 0:
                # Fallback: use median absolute deviation as noise estimate
                noise_std = np.nanmedian(np.abs(yflux - np.nanmedian(yflux))) * 1.4826
        except Exception:
            # Fallback: use median absolute deviation as noise estimate
            noise_std = np.nanmedian(np.abs(yflux - np.nanmedian(yflux))) * 1.4826

    print(f"    [FIND_LINES DEBUG] noise_std={noise_std:.6e}, SNR threshold={snr_threshold * noise_std:.6e}")
    
    # at start all lines are assumed to be non-detections and assigned NAN for max_amplitude
    data = {
        'line': lines_to_search['line_wave_um'],
        'species': lines_to_search['species'],
        'disposition': [0] * len(lines_to_search['line_wave_um']),
        'max_amplitude': [None] * len(lines_to_search['line_wave_um']),
        'line_flux': [None] * len(lines_to_search['line_wave_um'])
    }
    lines_dispositions = pd.DataFrame(data)
    potential_lines = []
    
    for ii in range(len(lines_dispositions['line'])):
        cline = lines_dispositions['line'][ii]  # current line
        idx_cline = np.argmin(np.abs(xwave - cline))  # index of the wave closest to the line
        cwave = xwave[max(0, idx_cline - 2):min(len(xwave), idx_cline + 3)]
        cflux = yflux[max(0, idx_cline - 2):min(len(yflux), idx_cline + 3)]
        
        cflux_min, cflux_max = np.nanmin(cflux), np.nanmax(cflux)
        cflux_idx_max = np.argmax(cflux)
        peak_amplitude = cflux_max - cflux_min
        
        # Check if peak exists, is at right location, and is statistically significant
        if (cflux_max > cflux_min) and (cwave[cflux_idx_max] <= cline + xres) and (cwave[cflux_idx_max] >= cline - xres):
            # Apply SNR threshold: peak must be significant above noise
            if peak_amplitude > snr_threshold * noise_std:
                print(f"    [LINE {ii}] {cline:.6f} um: DETECTED! amplitude={peak_amplitude:.6e} (threshold={snr_threshold * noise_std:.6e})")
                if isinstance(contaminating_lines, pd.DataFrame):
                    # Check if peak is away from contaminating lines
                    if all(np.abs(cwave[cflux_idx_max] - iline) > xres for iline in contaminating_lines['line_wave_um']):
                        lines_dispositions.loc[ii, 'disposition'] = 1
                        lines_dispositions.loc[ii, 'max_amplitude'] = peak_amplitude
                        lines_dispositions.loc[ii, 'line_flux'] = yflux[idx_cline]
                        potential_lines.append((ii, yflux[idx_cline]))
                else:
                    # No need to check for contaminating lines
                    lines_dispositions.loc[ii, 'disposition'] = 1
                    lines_dispositions.loc[ii, 'max_amplitude'] = peak_amplitude
                    lines_dispositions.loc[ii, 'line_flux'] = yflux[idx_cline]
                    potential_lines.append((ii, yflux[idx_cline]))
            else:
                # Line NOT detected - show why
                reason = "UNKNOWN"
                if cflux_max <= cflux_min:
                    reason = "No flux variation"
                elif not (cwave[cflux_idx_max] <= cline + xres and cwave[cflux_idx_max] >= cline - xres):
                    reason = f"Peak at {cwave[cflux_idx_max]:.6f} um, outside range {cline - xres:.6f}-{cline + xres:.6f}"
                elif peak_amplitude <= snr_threshold * noise_std:
                    reason = f"amplitude={peak_amplitude:.6e} < threshold={snr_threshold * noise_std:.6e}"
                print(f"    [LINE {ii}] {cline:.6f} um: REJECTED - {reason}")
    
    # Remove duplicates: keep strongest line in each close group
    if remove_duplicate:
        # Step 1: Create groups of lines that are within `2 * xres` of each other
        groups = []
        for i, (idx1, _) in enumerate(potential_lines):
            group = [idx1]
            for j, (idx2, _) in enumerate(potential_lines[i + 1:], start=i + 1):
                if abs(lines_dispositions.loc[idx1, 'line'] - lines_dispositions.loc[idx2, 'line']) <= 2 * xres:
                    group.append(idx2)
                    groups.append(group)
        
         # Step 2 & 3: Find the strongest line in each group and set the disposition of other lines to 0
        for group in groups:
            if len(group) > 1:
                # Prefer main isotopologue (12C16O or CO) if present
                valid_indices = [idx for idx in group if lines_dispositions.loc[idx, 'species'] in ['CO', '12C16O']]
                # If there are no valid lines, consider all lines in the group
                if not valid_indices:
                    valid_indices = group
                
                # Find strongest line
                strongest_line_idx = max(valid_indices, key=lambda idx: lines_dispositions.loc[idx, 'line_flux'])
                
                # Mark others as not detected
                for idx in group:
                    if idx != strongest_line_idx:
                        lines_dispositions.loc[idx, 'disposition'] = 0
    
    return lines_dispositions


# ================== Continuum Fitting (RANSAC) ==================

def cont_fit_ransac(x, y, pdeg, return_params=False, exclude_wavelength_ranges=None):
    """
    Fit continuum using RANSAC polynomial regression (from Pascucci), robust to outliers.

    Args:
        x (array): Wavelength values
        y (array): Flux values
        pdeg (int): Polynomial degree (1=linear, 2=quadratic, etc.)
        return_params (bool): If True, return polynomial coefficients instead of the continuum
        exclude_wavelength_ranges (list of tuples): Wavelength ranges to exclude from fitting

    Returns:
        result (tuple): (continuum or coefficients, std_residuals)
    """
    # Combine x and y into a single array
    data = np.column_stack((x, y))
    # Remove rows with NaN values for fitting
    valid_data = data[~np.isnan(data).any(axis=1)]
    # Separate the input variables and target variable for fitting
    xfit = valid_data[:, 0]
    yfit = valid_data[:, 1]
    if exclude_wavelength_ranges is not None:
        # Filter data to exclude points within the specified wavelength ranges
        for exclude_range in exclude_wavelength_ranges:
            mask = ~((xfit >= exclude_range[0]) & (xfit <= exclude_range[1]))
            xfit = xfit[mask]
            yfit = yfit[mask]

    # Create a pipeline with polynomial feature transformation and RANSACRegressor
    ransac_pipeline = make_pipeline(
        PolynomialFeatures(degree=pdeg),
        RANSACRegressor()
    )
    # Fit the pipeline on the valid data and calculate the stddev of the residuals
    ransac_pipeline.fit(xfit[:, np.newaxis], yfit)
    std_residuals = np.std(yfit - ransac_pipeline.predict(xfit[:, np.newaxis]))
    # Predict y-values for all x-values (including NaNs)
    yfit_all = ransac_pipeline.predict(x[:, np.newaxis])
    # Replace predicted values corresponding to NaN x-values with NaN
    yfit_all[np.isnan(x)] = np.nan

    if return_params:
        best_fit_params = ransac_pipeline.named_steps['ransacregressor'].estimator_.coef_
        return best_fit_params, std_residuals
    else:
        return yfit_all, std_residuals


# ================== Multi-Gaussian Fitting ====================

def fit_glines(xwave, yflux, list_wboundaries, detco_lines, detcontaminating_lines, sigma,
                progress_callback=None, flux_unit_label=None, wave_unit_label=None, header=None,
                list_wboundaries_extent=None, half_width_sigma=DEFAULT_SIGMA_MULTIPLIER):
    """
    Fit simultaneously many Gaussians and the underlying continuum in each wavelength chunk
    (given in list_wboundaries), via curve_fit on many_gaussians_list.

    Args:
        xwave (array): Wavelength array
        yflux (array): Flux array (continuum still present -- fit alongside the lines)
        list_wboundaries (array): Flat, sorted chunk boundaries; consecutive pairs = one chunk.
            Controls which lines belong to which chunk (a partition -- every line assigned once)
        detco_lines (pd.DataFrame): Detected CO lines with 'line'/'disposition'/'max_amplitude',
            plus an optional per-line 'sigma' (falls back to the shared `sigma` if absent)
        detcontaminating_lines (pd.DataFrame): Same columns, for contaminating lines
        sigma (float): Shared fallback Gaussian sigma, used only where a line has no own 'sigma'
        progress_callback (callable): Optional progress_callback(current_chunk, total_chunks),
            called after each chunk finishes (converged or not)
        flux_unit_label (str): Cosmetic unit label for the debug prints only -- every returned
            value stays in xwave/yflux's own native units; no CGS conversion happens here
        wave_unit_label (str): Cosmetic wavelength unit label, same debug-print-only role
        header (astropy.io.fits.Header): Optional, only for an approximate erg/s/cm**2 preview
            alongside the native value in the debug prints -- not a second source of truth
        list_wboundaries_extent (array): Same shape as list_wboundaries, optionally wider per
            chunk (see _merged_wave_boundaries) -- what curve_fit/plots actually see, so
            capping ownership never truncates a member's own window. None falls back to list_wboundaries.
        half_width_sigma (float): Must match whatever built list_wboundaries -- used only to
            find "borrowed" lines (see below), not to rebuild the boundaries.

    Returns:
        result (tuple): (sum_CO_area, sum_contaminating_area, mean_cont, df_residual,
            df_continuum, per_line_fits, per_contam_fits), all in native units. per_line_fits/
            per_contam_fits are per-line (amplitude, center, sigma, area, x/y/fit data) dicts for
            callers needing per-line detail (e.g. fit_hitran_lines_collective); aggregate-only
            callers (e.g. extract_co_flux_from_spectrum) can ignore them.
    """
    sum_CO_area, sum_contaminating_area, mean_cont = 0, 0, 0
    df_residual = pd.DataFrame({
        'wave': xwave,
        'residual': np.zeros(len(xwave))
    })
    df_continuum = pd.DataFrame({
        'wave': xwave,
        'continuum': np.full(len(xwave), np.nan)
    })
    per_line_fits = []
    per_contam_fits = []

    # Every known line, used below to find lines a chunk should "borrow" (not owned, but
    # close enough to bias its continuum if left unfit). Built once, re-filtered per chunk.
    all_lines_pool = pd.concat([detco_lines, detcontaminating_lines], ignore_index=True)
    pool_wave = all_lines_pool['line'].astype(float).values
    pool_disp = all_lines_pool['disposition'].values
    pool_sigma = (all_lines_pool['sigma'].astype(float).values if 'sigma' in all_lines_pool.columns
                  else np.full(len(all_lines_pool), sigma))
    pool_nat_lo = pool_wave - half_width_sigma * pool_sigma
    pool_nat_hi = pool_wave + half_width_sigma * pool_sigma

    print('===== Start fitting...')
    start_time = time.time()

    for ii in range(len(list_wboundaries) - 1):
        xwave_min, xwave_max = list_wboundaries[ii], list_wboundaries[ii + 1]
        # Ownership (xwave_min/xwave_max) decides which lines belong to this chunk; the extent
        # (possibly wider) decides what data the chunk actually fits/plots -- so capping
        # ownership to keep chunks small never truncates a member line's own window.
        if list_wboundaries_extent is not None:
            ext_min, ext_max = list_wboundaries_extent[ii], list_wboundaries_extent[ii + 1]
        else:
            ext_min, ext_max = xwave_min, xwave_max
        # Excludes NaN/inf up front (real IFU spaxels are often partially covered) so a few bad
        # samples don't throw out the whole chunk; everything below only ever sees finite data.
        idx_xwave = (xwave >= ext_min) & (xwave <= ext_max) & np.isfinite(xwave) & np.isfinite(yflux)
        print('> Fitting wave range ', np.around(ext_min, decimals=4), np.around(ext_max, decimals=4))

        # Filtering the DataFrame based on the disposition and the wavelength range: CO and contaminating lines
        filtered_colines = detco_lines[
            (detco_lines['disposition'] == 1) &
            (detco_lines['line'] >= xwave_min) &
            (detco_lines['line'] <= xwave_max)
        ]
        filtered_contaminatinglines = detcontaminating_lines[
            (detcontaminating_lines['disposition'] == 1) &
            (detcontaminating_lines['line'] >= xwave_min) &
            (detcontaminating_lines['line'] <= xwave_max)
        ]

        # Find number of contaminating lines for Gauss fit and store
        ncont_lines = filtered_contaminatinglines.shape[0]
        n_co = len(filtered_colines)

        # Borrowed: not owned by this chunk, but its window still reaches into what this
        # chunk fits -- fit it here too, but never sum its flux (its own chunk already does).
        not_owned = (pool_wave < xwave_min) | (pool_wave > xwave_max)
        overlaps_extent = (pool_nat_lo < ext_max) & (pool_nat_hi > ext_min)
        borrowed_lines = all_lines_pool[not_owned & overlaps_extent & (pool_disp == 1)]
        if len(borrowed_lines) > 0:
            print(f'  [BORROW DEBUG] Chunk {ii}: borrowing {len(borrowed_lines)} nearby line(s) '
                  f'{list(np.round(borrowed_lines["line"].astype(float).values, 6))} to fit alongside '
                  f'(not counted in this chunk\'s flux)')

        # Append contaminating lines, then any borrowed lines, at the end of the CO lines
        filtered_lines = pd.concat([filtered_colines, filtered_contaminatinglines, borrowed_lines], ignore_index=True)

        if len(filtered_lines) == 0:
            print('  No lines to fit in this range')
            continue

        # curve_fit needs >= as many finite points as free params (2 + 3/line) to be determined
        # at all; fewer than that gets marked failed here rather than handed an arbitrary "answer".
        n_finite = np.count_nonzero(idx_xwave)
        n_params_needed = 2 + 3 * len(filtered_lines)
        if n_finite <= n_params_needed:
            print(f'  [DATA WARN] Only {n_finite} finite points for {n_params_needed} parameters '
                  f'in this chunk -- skipping (not enough data to fit)')
            for _, coline in filtered_colines.iterrows():
                per_line_fits.append({
                    'wavelength': coline['line'], 'amplitude': 0.0, 'center': coline['line'],
                    'sigma': 0.0, 'area': 0.0, 'amplitude_error': None, 'sigma_error': None,
                    'catalog_sigma': coline['sigma'] if 'sigma' in coline else None, 'noise_std': None,
                    'x_data': xwave[idx_xwave], 'y_data': yflux[idx_xwave],
                    'y_fit': np.zeros(n_finite),
                    'success': False,
                    'message': f"Not enough finite data points ({n_finite}) for {n_params_needed} parameters",
                })
            for _, contam_line in filtered_contaminatinglines.iterrows():
                per_contam_fits.append({
                    'wavelength': contam_line['line'], 'amplitude': 0.0, 'center': contam_line['line'],
                    'sigma': 0.0, 'area': 0.0, 'amplitude_error': None, 'sigma_error': None,
                    'catalog_sigma': contam_line['sigma'] if 'sigma' in contam_line else None, 'noise_std': None,
                    'x_data': xwave[idx_xwave], 'y_data': yflux[idx_xwave],
                    'y_fit': np.zeros(n_finite),
                    'success': False,
                    'message': f"Not enough finite data points ({n_finite}) for {n_params_needed} parameters",
                })
            continue

        # Per-line sigma when supplied (real per-line width), else broadcast the shared `sigma`
        # -- one shared width across a chunk under-resolves any line that doesn't share it.
        if 'sigma' in filtered_lines.columns:
            line_sigmas = filtered_lines['sigma'].astype(float).values
        else:
            line_sigmas = np.full(len(filtered_lines), sigma)

        # Create the NumPy array with input params
        p0_2d = np.column_stack((
            filtered_lines['max_amplitude'].astype(float),
            filtered_lines['line'],
            line_sigmas
        ))
        p0 = np.ravel(p0_2d)  # 1d array
        # NaN-safe and scoped to this chunk (matching p0slope) -- the whole-spectrum
        # np.nanmin(yflux) can sit far from this chunk's real flux level, making the bounds infeasible.
        p0base, p0slope = np.nanmin(yflux[idx_xwave]), (yflux[idx_xwave][-1] - yflux[idx_xwave][0]) / (xwave[idx_xwave][-1] - xwave[idx_xwave][0])
        p0 = np.insert(p0, [0, 0], [p0base, p0slope])

        # Create NumPy arrays with lower and upper bounds
        lower_bounds_2d = np.column_stack((
            [0.0] * len(filtered_lines),
            filtered_lines['line'] - line_sigmas,
            0.5 * line_sigmas
        ))
        lower_bounds = np.ravel(lower_bounds_2d)
        lower_bounds = np.insert(lower_bounds, [0, 0], [p0base - 2 * np.abs(p0slope), -2 * np.abs(p0slope)])

        upper_bounds_2d = np.column_stack((
            filtered_lines['max_amplitude'].astype(float) * 1.1,
            filtered_lines['line'] + line_sigmas,
            2 * line_sigmas
        ))
        upper_bounds = np.ravel(upper_bounds_2d)
        upper_bounds = np.insert(upper_bounds, [0, 0], [p0base + 2 * np.abs(p0slope), 2 * np.abs(p0slope)])

        # Perform the fit (Pascucci fit_Glines) (maxfev from 10k to 1k)
        par, cov = curve_fit(many_gaussians_list, xwave[idx_xwave], yflux[idx_xwave], p0=p0,
                             bounds=(lower_bounds, upper_bounds), method='trf', ftol=1e-5, maxfev=1000)
        print(f'  [FIT DEBUG] curve_fit succeeded, returned {len(par)} parameters')

        a, b = par[0], par[1]  # Extracts which parameters relate to the continuum
        par = par[2:]  # Extracts which parameters relate to the Gaussians
        print(f'  [FIT DEBUG] continuum: a={a:.6e} {flux_unit_label}, b={b:.6e} {flux_unit_label}/{wave_unit_label}; '
              f'{len(par)//3} Gaussian components')
        if len(par) >= 3:
            print(f'  [FIT DEBUG] First line: amp={par[0]:.6e} {flux_unit_label}, center={par[1]:.6f} {wave_unit_label}, '
                  f'sigma={par[2]:.6e} {wave_unit_label}')

        try:
            bfit = many_gaussians_list(xwave[idx_xwave], a, b, *par)
            df_residual.loc[idx_xwave, 'residual'] = yflux[idx_xwave] - bfit
            # many_gaussians_list's continuum is a + b*(x - min_freq), not a + b*x -- must match
            # exactly or the recorded curve is off by b*min_freq. Only affects the plotted curve.
            df_continuum.loc[idx_xwave, 'continuum'] = a + b * (xwave[idx_xwave] - xwave[idx_xwave].min())

            # Noise from this chunk's own residuals, with the real dof of the full model (the
            # same quantity curve_fit itself uses to scale `cov`) -- consistent with amp errors below.
            n_data_chunk = np.count_nonzero(idx_xwave)
            dof_chunk = n_data_chunk - (len(par) + 2)
            if dof_chunk > 0:
                noise_std_chunk = float(np.sqrt(np.sum((yflux[idx_xwave] - bfit) ** 2) / dof_chunk))
            else:
                print(f'  [NOISE WARN] Chunk {ii}: dof={dof_chunk} <= 0, cannot estimate noise from residuals')
                noise_std_chunk = None

            # Owned params only -- borrowed lines are still fitted (subtracted from the
            # residual) but never summed here, so n_owned_params excludes them, not len(par).
            n_owned_params = 3 * (n_co + ncont_lines)
            if ncont_lines > 0:
                current_sum_CO = sum(par[jj] * par[jj + 2] * np.sqrt(2 * np.pi)
                                    for jj in range(0, 3 * n_co, 3))
                current_sum_contaminating = sum(par[jj] * par[jj + 2] * np.sqrt(2 * np.pi)
                                               for jj in range(3 * n_co, n_owned_params, 3))
            else:
                current_sum_contaminating = 0  # there are no contaminating lines in this wave range!
                current_sum_CO = sum(par[jj] * par[jj + 2] * np.sqrt(2 * np.pi)
                                    for jj in range(0, n_owned_params, 3))

            area_unit_label = f'{flux_unit_label}·{wave_unit_label}'
            print(f'  [FLUX DEBUG] Range {ii}: {len(par)//3} fitted lines, '
                  f'CO flux segment={current_sum_CO:.6e} {area_unit_label}, contam={current_sum_contaminating:.6e} {area_unit_label}')
            if current_sum_CO < 0:
                print(f'  [FLUX WARN] Negative CO flux detected! intensities ({flux_unit_label}): '
                      f'{[par[jj] for jj in range(0, len(par) - 3 * ncont_lines, 3)]}')

            # filtered_lines is CO lines then contaminating lines; `par` follows the same order
            # in 3-param groups -- filtered_colines.iloc[i] <-> par[3*i:3*i+3], contam offset by n_co.
            min_freq_chunk = np.nanmin(xwave[idx_xwave])

            def _local_fit_record(cen_i, amp_i, sig_i, cat_wave, amp_err_i, sig_err_i, catalog_sigma_i):
                """
                Build one line's per_line_fits/per_contam_fits dict from this chunk's fit.

                Args:
                    cen_i (float): This line's fitted center
                    amp_i (float): This line's fitted amplitude
                    sig_i (float): This line's fitted sigma
                    cat_wave (float): Catalog (rest) wavelength for this line
                    amp_err_i (float): This line's amplitude uncertainty
                    sig_err_i (float): This line's sigma uncertainty
                    catalog_sigma_i (float): The fixed instrumental sigma this line was given

                Returns:
                    record (dict): Per-line fit dict, see fit_glines' own Returns for its keys
                """
                # The chunk's own fit window, not a re-derived crop -- plots from this record
                # show exactly what curve_fit saw, and DEFAULT_SIGMA_MULTIPLIER changes what's drawn.
                x_local = xwave[idx_xwave]
                y_local = yflux[idx_xwave]
                # Full chunk model: local continuum + every Gaussian in the chunk (not just this
                # line's own) -- honestly reflects the data, including overlap with a neighbor.
                y_fit_local = a + b * (x_local - min_freq_chunk)
                for jj in range(0, len(par), 3):
                    y_fit_local = y_fit_local + gaussian(x_local, par[jj], par[jj + 1], par[jj + 2])
                # Same local linear term, but only this line's own Gaussian -- for a
                # per-line plot that shouldn't draw a neighbor sharing this chunk.
                y_fit_own = a + b * (x_local - min_freq_chunk) + gaussian(x_local, amp_i, cen_i, sig_i)
                return {
                    'wavelength': cat_wave, 'amplitude': amp_i, 'center': cen_i, 'sigma': sig_i,
                    'area': amp_i * sig_i * np.sqrt(2 * np.pi),
                    # amp_err_i: from curve_fit's joint covariance matrix, not a re-fit local
                    # model -- a line degenerate with a close neighbor gets a larger error.
                    'amplitude_error': amp_err_i,
                    # sig_err_i: same joint covariance matrix -- with amp_err_i, feeds Pascucci's
                    # area_err = sqrt(2pi*(sigma**2*amp_err**2 + amp**2*sig_err**2)).
                    'sigma_error': sig_err_i,
                    # catalog_sigma_i: the fixed instrumental sigma given as input, not fitted
                    # `sig_i` -- used for the upper-limit Gaussian when this line's fit isn't real.
                    'catalog_sigma': catalog_sigma_i,
                    'noise_std': noise_std_chunk,
                    'x_data': x_local, 'y_data': y_local, 'y_fit': y_fit_local,
                    'y_fit_own': y_fit_own,
                    'success': True, 'message': 'Success',
                }

            for i, (_, coline) in enumerate(filtered_colines.iterrows()):
                amp_err_i = float(np.sqrt(np.abs(cov[2 + 3 * i, 2 + 3 * i])))
                sig_err_i = float(np.sqrt(np.abs(cov[2 + 3 * i + 2, 2 + 3 * i + 2])))
                per_line_fits.append(_local_fit_record(
                    par[3 * i + 1], par[3 * i], par[3 * i + 2], coline['line'], amp_err_i, sig_err_i, line_sigmas[i]
                ))
            for j, (_, contam_line) in enumerate(filtered_contaminatinglines.iterrows()):
                jj = 3 * (n_co + j)
                amp_err_j = float(np.sqrt(np.abs(cov[2 + jj, 2 + jj])))
                sig_err_j = float(np.sqrt(np.abs(cov[2 + jj + 2, 2 + jj + 2])))
                per_contam_fits.append(_local_fit_record(
                    par[jj + 1], par[jj], par[jj + 2], contam_line['line'], amp_err_j, sig_err_j, line_sigmas[n_co + j]
                ))

        except RuntimeError as e:
            print("curve_fit did not converge! Fit a straight line and assign 3sigma upper limit", e)
            continuum, std_residuals = cont_fit_ransac(xwave[idx_xwave], yflux[idx_xwave], pdeg=1)
            # median(line_sigmas), not the scalar sigma -- same value when broadcast (CO Flux
            # Map), but reflects real per-line widths when they vary (fit_hitran_lines_collective).
            fallback_sigma = np.median(line_sigmas)
            current_sum_contaminating = 3 * np.median(std_residuals) * fallback_sigma * np.sqrt(2 * np.pi)
            current_sum_CO = 3 * np.median(std_residuals) * fallback_sigma * np.sqrt(2 * np.pi)
            a = np.nanmedian(continuum)
            df_continuum.loc[idx_xwave, 'continuum'] = continuum
            for i, (_, coline) in enumerate(filtered_colines.iterrows()):
                per_line_fits.append({
                    'wavelength': coline['line'], 'amplitude': 0.0, 'center': coline['line'],
                    'sigma': 0.0, 'area': 0.0, 'amplitude_error': None, 'sigma_error': None,
                    'catalog_sigma': line_sigmas[i], 'noise_std': None,
                    'x_data': xwave[idx_xwave], 'y_data': yflux[idx_xwave],
                    'y_fit': np.zeros(np.count_nonzero(idx_xwave)),
                    'success': False, 'message': f"Chunk fit failed: {e}",
                })
            for j, (_, contam_line) in enumerate(filtered_contaminatinglines.iterrows()):
                per_contam_fits.append({
                    'wavelength': contam_line['line'], 'amplitude': 0.0, 'center': contam_line['line'],
                    'sigma': 0.0, 'area': 0.0, 'amplitude_error': None, 'sigma_error': None,
                    'catalog_sigma': line_sigmas[n_co + j], 'noise_std': None,
                    'x_data': xwave[idx_xwave], 'y_data': yflux[idx_xwave],
                    'y_fit': np.zeros(np.count_nonzero(idx_xwave)),
                    'success': False, 'message': f"Chunk fit failed: {e}",
                })

        sum_CO_area = sum_CO_area + current_sum_CO
        sum_contaminating_area = sum_contaminating_area + current_sum_contaminating
        mean_cont = mean_cont + a
        print(f'  [FLUX DEBUG] Cumulative CO flux: {sum_CO_area:.6e} {flux_unit_label}·{wave_unit_label}, '
              f'Cumulative contam: {sum_contaminating_area:.6e} {flux_unit_label}·{wave_unit_label}')

        if progress_callback is not None:
            progress_callback(ii + 1, len(list_wboundaries) - 1)

    end_time = time.time()
    print(f"Fitting took {np.around(end_time - start_time, decimals=1)} seconds")
    print('===== End fitting')
    print(f'[FINAL FLUX] Total CO flux: {sum_CO_area:.6e} {flux_unit_label}·{wave_unit_label}, '
          f'Total contam flux: {sum_contaminating_area:.6e} {flux_unit_label}·{wave_unit_label}, '
          f'Mean continuum: {mean_cont/(len(list_wboundaries) - 1):.6e} {flux_unit_label} '
          f'(all native/raw -- not yet CGS-converted; conversion happens in the caller)')

    mean_cont = mean_cont / (len(list_wboundaries) - 1)

    return sum_CO_area, sum_contaminating_area, mean_cont, df_residual, df_continuum, per_line_fits, per_contam_fits


def _line_sigma(wl, header):
    """
    Instrumental Gaussian sigma at wavelength `wl`, from the instrument's real spectral
    resolution (R = lambda/delta_lambda) in `header`. No fallback: MIRI/NIRSpec headers
    always support this, so a header that can't is missing something and should fail loudly.

    Args:
        wl (float): Wavelength, in the same units as the header's spectral resolution formula
        header (astropy.io.fits.Header): FITS header identifying the instrument (MIRI/NIRSpec)

    Returns:
        sigma (float): Instrumental Gaussian sigma, same units as `wl`

    Raises:
        ValueError: if header is None, the instrument isn't MIRI/NIRSpec, or (NIRSpec) the
            grating is missing/unrecognized
    """
    if header is None:
        raise ValueError("Cannot determine spectral resolution: no FITS header given.")
    instrument = header.get('INSTRUME', '').strip().upper()
    if 'MIRI' in instrument:
        R = get_spectral_resolution(wl, 'MIRI', header)
    elif 'NIRSPEC' in instrument:
        R = get_spectral_resolution(wl, 'NIRSpec', header)
    else:
        raise ValueError(f"No spectral resolution formula for instrument {instrument!r} "
                          f"(only MIRI and NIRSpec are supported).")
    return (wl / R) / 2.355



def _merged_wave_boundaries(centers, sigmas, half_width_sigma=DEFAULT_SIGMA_MULTIPLIER):
    """
    Build fit_glines' list_wboundaries from each line's own instrumental width, not a fixed
    grid: give each line a [center - N*sigma, center + N*sigma] window, then transitively
    merge overlapping windows into shared chunks (capped by max_chunk_width below).

    Args:
        centers (array): Line center wavelengths
        sigmas (array): Each line's own instrumental sigma, same units as centers
        half_width_sigma (float): Each line's window half-width, in multiples of its own sigma

    Returns:
        ownership_boundaries (array): Flat, sorted chunk boundaries (consecutive pairs = one
            chunk) controlling which lines fit_glines assigns to which chunk -- capped, so a
            dense line list still partitions into separate simultaneous fits, not one huge one.
        extent_boundaries (array): Same shape/pairing as ownership_boundaries, but each pair
            is widened (never narrowed) to fully cover every line it owns' own +/-half_width_sigma
            window. Capping a chunk's *membership* must never cost a member its own *data* --
            fit_glines should fit (and plot) from this, not ownership_boundaries.
    """
    # Caps a chunk's width (multiples of its starting line's sigma) so transitive merging can't
    # cascade a dense line list into one huge shared continuum -- verified against real cases.
    max_chunk_width = 3 * (2 * half_width_sigma)  # sigma units
    intervals = sorted(
        (c - half_width_sigma * s, c + half_width_sigma * s, s)
        for c, s in zip(centers, sigmas)
    )
    merged = [list(intervals[0][:2])]
    anchor_start, anchor_sigma = intervals[0][0], intervals[0][2]
    for lo, hi, s in intervals[1:]:
        fits_cap = (hi - anchor_start) <= max_chunk_width * anchor_sigma
        if lo <= merged[-1][1] and fits_cap:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            lo = max(lo, merged[-1][1])  # keep chunks non-overlapping even when capped
            merged.append([lo, hi])
            anchor_start, anchor_sigma = lo, s
    ownership_boundaries = np.array([b for lo, hi in merged for b in (lo, hi)])

    centers_arr = np.asarray(centers, dtype=float)
    windows_lo = centers_arr - half_width_sigma * np.asarray(sigmas, dtype=float)
    windows_hi = centers_arr + half_width_sigma * np.asarray(sigmas, dtype=float)
    extent_pairs = []
    for lo, hi in merged:
        owned = (centers_arr >= lo) & (centers_arr <= hi)
        if np.any(owned):
            lo = min(lo, np.min(windows_lo[owned]))
            hi = max(hi, np.max(windows_hi[owned]))
        extent_pairs.append((lo, hi))
    extent_boundaries = np.array([b for lo, hi in extent_pairs for b in (lo, hi)])

    return ownership_boundaries, extent_boundaries


def fit_hitran_lines_collective(spectrum_x, spectrum_y, hitran_lines, continuum,
                                 header=None, wave_unit=None, contam_lines=None,
                                 sigma_multiplier=DEFAULT_SIGMA_MULTIPLIER, progress_callback=None):
    """
    Fit a hand-picked list of HITRAN lines (e.g. from the species dialog) simultaneously, in
    per-line instrumental-width chunks, via fit_glines() -- same engine CO Flux Map uses, but
    with no SNR-based detection: the caller has already chosen exactly which lines to fit.

    Args:
        spectrum_x (array): Full spectrum wavelength array (numpy or Quantity)
        spectrum_y (array): Full spectrum flux array (numpy or Quantity)
        hitran_lines (list): Dicts with 'lambda_micron' or 'wavelength', plus any metadata
            (species, transition, ...) the caller wants carried through to the result
        continuum (array): Continuum matching spectrum_x/spectrum_y, subtracted before fitting
        header (astropy.io.fits.Header): For the erg/s/cm**2 area conversion (same rule as fit_line)
        wave_unit (str): Wavelength unit, same role as header
        contam_lines (pd.DataFrame): Optional 'line_wave_um' (+ 'species') column; fit alongside
            the requested lines so their flux doesn't leak into a nearby requested line's amplitude
        sigma_multiplier (float): Each line's fit window half-width, in multiples of its own
            _line_sigma. See DEFAULT_SIGMA_MULTIPLIER's own comment for where the default comes from.

    Returns:
        result (tuple): (results, contam_results, None) -- the third element is an unused
            placeholder (no velocity-shift correction; line centers fit freely within
            fit_glines' own bounds instead). results/contam_results are lists of
            (line_dict, GaussianFitResult) pairs, in hitran_lines' order for results.
    """
    spectrum_x = extract_value_from_quantity(spectrum_x)
    spectrum_y = extract_value_from_quantity(spectrum_y)
    continuum = extract_value_from_quantity(continuum)

    valid = []
    for idx, line in enumerate(hitran_lines):
        wl = line.get('lambda_micron')
        if wl is None:
            wl = line.get('wavelength')
        if wl is not None:
            valid.append((idx, float(wl), line))

    if not valid or len(spectrum_x) < 4:
        return [], [], None

    y_corrected = spectrum_y - continuum

    def _local_max_amp(wl):
        """
        Estimate max_amplitude for a line the same simple way find_lines() does: local
        peak-to-trough over a small window around the catalog wavelength.

        Args:
            wl (float): Catalog (rest) wavelength to center the window on

        Returns:
            amplitude (float): Local peak-to-trough amplitude, or 0.0 if the window is empty
        """
        idx_c = int(np.argmin(np.abs(spectrum_x - wl)))
        w0, w1 = max(0, idx_c - 2), min(len(spectrum_x), idx_c + 3)
        local = y_corrected[w0:w1]
        return float(np.nanmax(local) - np.nanmin(local)) if len(local) else 0.0

    lines_data = [(orig_idx, wl, line, max(_local_max_amp(wl), 1e-6)) for orig_idx, wl, line in valid]

    # Per-line sigma from the instrument's own resolution (_line_sigma), not one shared guess
    # -- a single estimate over the whole spectrum can't know each line's real width (varies ~2x).
    detco_lines = pd.DataFrame({
        'line': [d[1] for d in lines_data],
        'disposition': [1] * len(lines_data),
        'max_amplitude': [d[3] for d in lines_data],
        'sigma': [_line_sigma(d[1], header) for d in lines_data],
    })

    if contam_lines is not None and len(contam_lines) > 0:
        contam_wls = contam_lines['line_wave_um'].astype(float).values
        detcontaminating_lines = pd.DataFrame({
            'line': contam_wls,
            'disposition': [1] * len(contam_wls),
            'max_amplitude': [max(_local_max_amp(wl), 1e-6) for wl in contam_wls],
            'sigma': [_line_sigma(wl, header) for wl in contam_wls],
        })
    else:
        detcontaminating_lines = pd.DataFrame(columns=['line', 'disposition', 'max_amplitude', 'sigma'])

    all_centers = list(detco_lines['line']) + list(detcontaminating_lines['line'])
    all_sigmas = list(detco_lines['sigma']) + list(detcontaminating_lines['sigma'])
    wave_boundaries, wave_boundaries_extent = _merged_wave_boundaries(
        all_centers, all_sigmas, half_width_sigma=sigma_multiplier)

    # Only used as fit_glines' scalar fallback (RuntimeError path) -- the per-line 'sigma'
    # column above is what actually drives the real fit now.
    sigma = float(np.median(detco_lines['sigma']))

    # No fallback string on purpose -- MIRI/NIRSpec/MUSE headers always carry BUNIT, so None
    # here means the header is genuinely missing something; the debug prints should show that.
    _, _, _, _, _, per_line_fits, per_contam_fits = fit_glines(
        spectrum_x, y_corrected, wave_boundaries, detco_lines, detcontaminating_lines, sigma,
        progress_callback=progress_callback,
        flux_unit_label=header.get('BUNIT') if header is not None else None,
        wave_unit_label=get_wave_unit(header, wave_unit),
        list_wboundaries_extent=wave_boundaries_extent, half_width_sigma=sigma_multiplier,
    )

    # fit_glines groups/iterates by chunk, not input order -- match back by wavelength so
    # species/transition metadata is preserved and results return in the caller's order.
    by_wavelength = {round(f['wavelength'], 9): f for f in per_line_fits}
    unit = get_wave_unit(header, wave_unit)

    def _to_result(fit):
        """
        Convert one fit_glines per-line fit dict into a GaussianFitResult, with the CGS area
        conversion and a local (this-line-only) R-squared.

        Args:
            fit (dict): One per_line_fits/per_contam_fits entry from fit_glines

        Returns:
            result (GaussianFitResult): Fit result for this one line
        """
        residuals = fit['y_data'] - fit['y_fit']
        baseline = continuum[np.isin(spectrum_x, fit['x_data'])]
        if not fit['success']:
            return GaussianFitResult(
                amplitude=0, center=fit['center'], sigma=0, fwhm=0, area=0,
                baseline_slope=0.0, baseline_intercept=float(baseline[0]) if len(baseline) else 0.0,
                x_data=fit['x_data'], y_data=fit['y_data'],
                baseline=baseline, y_corrected=fit['y_data'],
                y_fit=fit['y_fit'], residuals=residuals,
                r_squared=0, success=False, message=fit['message'],
                catalog_sigma=fit.get('catalog_sigma'),
            )
        fwhm = 2.355 * fit['sigma']
        area_native = fit['area']
        area_cgs = (convert_area_to_cgs(fit['amplitude'], fit['sigma'], fit['center'], header, unit)
                    if header is not None else None)
        if area_cgs is not None:
            area, area_unit = area_cgs, "erg/s/cm²"
        else:
            area = area_native
            # No fallback string on purpose -- a missing BUNIT here means the header is
            # genuinely incomplete; "UNKNOWN" flags that visibly instead of guessing a unit.
            flux_unit = (header.get('BUNIT') if header is not None else None) or 'UNKNOWN'
            area_unit = f"{flux_unit}·{unit.lower() if unit else 'UNKNOWN'}"
        # R2 from this line's own local peak (DISPLAY_CROP_SIGMA_MULTIPLIER window), not the
        # whole shared-chunk span -- else every line sharing a chunk reports the same number.
        local_mask = np.abs(fit['x_data'] - fit['center']) <= DISPLAY_CROP_SIGMA_MULTIPLIER * abs(fit['sigma'])
        local_res, local_y = residuals[local_mask], fit['y_data'][local_mask]
        ss_res = np.sum(local_res ** 2)
        ss_tot = np.sum((local_y - np.mean(local_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        message = (f"Warning: Poor fit quality (R² = {r_squared:.3f})" if r_squared < 0.5
                   else f"Success: Good fit (R² = {r_squared:.3f})")
        return GaussianFitResult(
            amplitude=fit['amplitude'], center=fit['center'], sigma=fit['sigma'],
            fwhm=fwhm, area=area, area_unit=area_unit, area_native=area_native,
            baseline_slope=0.0, baseline_intercept=float(baseline[0]) if len(baseline) else 0.0,
            x_data=fit['x_data'], y_data=fit['y_data'],
            baseline=baseline, y_corrected=fit['y_data'],
            y_fit=fit['y_fit'], residuals=residuals,
            r_squared=r_squared, success=True, message=message,
            amplitude_error=fit.get('amplitude_error'), sigma_error=fit.get('sigma_error'),
            catalog_sigma=fit.get('catalog_sigma'), noise_std=fit.get('noise_std'),
            y_fit_own=fit.get('y_fit_own'),
        )

    results = [None] * len(hitran_lines)
    for orig_idx, wl, line, _max_amp in lines_data:
        fit = by_wavelength.get(round(wl, 9))
        if fit is not None:
            results[orig_idx] = (line, _to_result(fit))

    contam_results = []
    if contam_lines is not None and len(contam_lines) > 0:
        contam_species = dict(zip(contam_lines['line_wave_um'].astype(float), contam_lines.get('species', [])))
        for fit in per_contam_fits:
            if not fit['success'] or fit['amplitude'] <= 0:
                continue
            species = contam_species.get(round(fit['wavelength'], 9), 'Contaminant')
            contam_line = {'species': species, 'wavelength': fit['wavelength']}
            contam_results.append((contam_line, _to_result(fit)))

    return [r for r in results if r is not None], contam_results, None


# ================== High-Level Extraction ==================

def extract_co_flux_from_spectrum(spectrum_wave, spectrum_flux, CO_catalog, contam_catalog,
                                   wave_boundaries=None, wave_min=None, wave_max=None, do_duplicate_removal=True,
                                   excluded_wavelengths=None,
                                   continuum_params=None, snr_threshold=3.0, header=None, wave_unit=None):
    """
    Extract total CO flux from a single spectrum by detecting and fitting all CO lines.

    Args:
        spectrum_wave (array or Quantity): Wavelength array for the spectrum
        spectrum_flux (array or Quantity): Flux array for the spectrum
        CO_catalog (pd.DataFrame): CO line catalog (columns: line_wave_um, species, ...)
        contam_catalog (pd.DataFrame): Contaminating lines catalog (column: line_wave_um)
        wave_boundaries (array): Fit-chunk boundaries. Auto-calculated from wave_min/wave_max if None
        wave_min (float): Minimum wavelength to fit. Defaults to min of CO_catalog line_wave_um
        wave_max (float): Maximum wavelength to fit. Defaults to max of CO_catalog line_wave_um
        do_duplicate_removal (bool): Whether to remove duplicate line detections
        excluded_wavelengths (set or list): Wavelengths to exclude from CO flux calculation
        continuum_params (dict): Overrides for fit_auto_continuum()'s defaults, which work
            well for typical spectra
        snr_threshold (float): Minimum signal-to-noise, in sigma, for a line to count as detected
        header (astropy.io.fits.Header): Required as soon as any line is detected -- each
            line's fit width comes from the instrument's real spectral resolution (_line_sigma,
            MIRI/NIRSpec only), so a missing header raises ValueError rather than guessing a width
        wave_unit (str): Wavelength unit (e.g. 'um'), used alongside header for the CGS conversion

    Returns:
        result (dict): 'co_flux', 'contam_flux', 'mean_continuum', 'detected_lines', 'residuals',
            'continuum_curve', 'flux_unit' ('erg/s/cm2' if the CGS conversion succeeded,
            otherwise the cube's native flux unit)

    Raises:
        ValueError: if a line is detected but header is None
    """
    spectrum_wave, spectrum_wave_unit = extract_units(spectrum_wave)
    spectrum_flux, spectrum_flux_unit = extract_units(spectrum_flux)

    # Subtract continuum before line detection.
    continuum_model = None
    try:
        continuum_model = fit_auto_continuum(
            spectrum_wave, spectrum_flux, **(continuum_params or {})
        )
        if continuum_model is not None and not np.all(np.isnan(continuum_model)):
            spectrum_flux = spectrum_flux - continuum_model
        else:
            print(f"  Warning: Continuum fitting returned NaN, using original spectrum")
    except Exception as e:
        print(f"  Warning: Continuum subtraction failed ({e}), using original spectrum")
        continuum_model = None

    resolved_wave_unit = get_wave_unit(header, spectrum_wave_unit or wave_unit)
    resolved_flux_unit = _resolve_flux_unit(spectrum_flux_unit, header)

    def _to_cgs(value, line_center):
        """
        Convert a native-unit flux value to erg/s/cm**2 at a given line center, or None
        if the conversion isn't reachable (no header, or a unit that doesn't convert).

        Args:
            value (float): Flux value in resolved_flux_unit x resolved_wave_unit
            line_center (float): Line center wavelength, in resolved_wave_unit

        Returns:
            flux (float or None): Flux in erg/s/cm**2, or None if unreachable
        """
        if header is None or value is None or line_center is None or not np.isfinite(line_center):
            return None
        return integrated_flux_to_cgs(
            value, resolved_flux_unit, wavelength=line_center,
            header=header, wave_unit=resolved_wave_unit,
        )

    # Auto-determine wavelength range if not provided
    if wave_min is None:
        wave_min = CO_catalog['line_wave_um'].min()
    if wave_max is None:
        wave_max = CO_catalog['line_wave_um'].max()
    
    print(f"  Extracting CO flux in wavelength range: {wave_min:.6f} - {wave_max:.6f} um")
    print(f"  [DEBUG] Spectrum: {len(spectrum_wave)} wavelength points, flux range {np.nanmin(spectrum_flux):.4e} to {np.nanmax(spectrum_flux):.4e}")
    if spectrum_wave_unit:
        print(f"  Wavelength unit: {spectrum_wave_unit}")
    if spectrum_flux_unit:
        print(f"  Flux unit: {spectrum_flux_unit}")

    # Whether CGS conversion is even reachable here (e.g. ALMA's Jy/beam never is) -- tested
    # once with a dummy value, labels co_flux/contam_flux consistently below. 'UNKNOWN' if not.
    flux_unit_label = spectrum_flux_unit or resolved_flux_unit or 'UNKNOWN'
    if _to_cgs(1.0, (wave_min + wave_max) / 2.0) is not None:
        flux_unit_label = 'erg/s/cm2'
    
    # wave_boundaries (if not given) is computed further down, after this pixel's own
    # lines are detected -- see the comment there for why it can't be built from the full catalog.

    # Narrow to the requested fit range
    fit_range_mask = (spectrum_wave >= wave_min) & (spectrum_wave <= wave_max)
    spectrum_wave = spectrum_wave[fit_range_mask]
    spectrum_flux = spectrum_flux[fit_range_mask]
    if continuum_model is not None:
        continuum_model = continuum_model[fit_range_mask]

    # Filter out excluded wavelengths from CO catalog
    CO_catalog_for_fit = CO_catalog
    if excluded_wavelengths and len(excluded_wavelengths) > 0:
        excluded_set = set(excluded_wavelengths) if not isinstance(excluded_wavelengths, set) else excluded_wavelengths
        # Remove lines within 0.001 micron tolerance of excluded wavelengths
        mask = ~CO_catalog['line_wave_um'].apply(
            lambda w: any(abs(w - ew) < 0.001 for ew in excluded_set)
        )
        CO_catalog_for_fit = CO_catalog[mask].copy()
        # IMPORTANT: Reset index so positions match indices (0, 1, 2, ... not original indices like 0, 5, 10, 220, ...)
        CO_catalog_for_fit = CO_catalog_for_fit.reset_index(drop=True)
        n_excluded = len(CO_catalog) - len(CO_catalog_for_fit)
        if n_excluded > 0:
            print(f"  Excluding {n_excluded} CO lines from extraction")
    
    print(f"  [DEBUG] CO catalog size: {len(CO_catalog_for_fit)} lines")

    # Compute noise
    try:
        shared_noise_std = compute_noise_std(spectrum_wave, spectrum_flux, pdeg=1)
        if shared_noise_std is None or not np.isfinite(shared_noise_std) or shared_noise_std <= 0:
            shared_noise_std = np.nanmedian(np.abs(spectrum_flux - np.nanmedian(spectrum_flux))) * 1.4826
    except Exception:
        shared_noise_std = np.nanmedian(np.abs(spectrum_flux - np.nanmedian(spectrum_flux))) * 1.4826

    # Find detected lines (contam_catalog can be None if user didn't upload contaminating lines)
    detected_lines = find_lines(
        spectrum_wave, spectrum_flux, CO_catalog_for_fit,
        contaminating_lines=contam_catalog,
        remove_duplicate=do_duplicate_removal,
        snr_threshold=snr_threshold,
        noise_std=shared_noise_std
    )
    
    if detected_lines is None or len(detected_lines) == 0:
        # No lines detected at all -- still show the upfront (pre-line-fit) continuum if one
        # was fit, so the plot has something to compare the raw spectrum against
        no_fit_continuum = np.full_like(spectrum_flux, np.nan)
        if continuum_model is not None:
            no_fit_continuum = continuum_model
        return {
            'co_flux': 0,
            'contam_flux': 0,
            'mean_continuum': np.nanmedian(spectrum_flux),
            'detected_lines': pd.DataFrame(),
            'detected_contam': pd.DataFrame(),
            'residuals': pd.DataFrame({'wave': spectrum_wave, 'residual': np.zeros_like(spectrum_flux)}),
            'continuum_curve': pd.DataFrame({'wave': spectrum_wave, 'continuum': no_fit_continuum}),
            'flux_unit': flux_unit_label,
            'mean_detected_wavelength': np.nan
        }
    
    # Identify detected CO and contaminating lines in this spectrum
    # Accept both 'CO' (HITRAN) and '12C16O' (ECSV catalog) species naming conventions
    try:
        detected_co = detected_lines[
            (detected_lines['disposition'] == 1) &
            (detected_lines['species'].isin(['CO', '12C16O']))
        ].copy()
    except KeyError as e:
        print(f"  Error: detected_lines missing column - columns are: {detected_lines.columns.tolist()}")
        print(f"  detected_lines head:\n{detected_lines.head()}")
        raise

    # Per-line sigma from the instrument's own resolution (_line_sigma), same as
    # fit_hitran_lines_collective -- one shared sigma under-resolves narrow lines.
    detected_co['sigma'] = [_line_sigma(wl, header) for wl in detected_co['line']]

    # DEBUG: Show how many CO lines were detected
    n_co_detected = len(detected_co)
    if n_co_detected > 0:
        print(f"  [DEBUG] CO line wavelengths: {detected_co['line'].values[:5].tolist()}...")
        print(f"  [DEBUG] CO line max amplitudes: {detected_co['max_amplitude'].values[:5].tolist()}...")

    # Peak-detects against the contaminant catalog itself, with no CO-catalog avoidance -- a
    # contaminant next to a CO line needs its own Gaussian in fit_glines, not to be dropped.
    if contam_catalog is not None and len(contam_catalog) > 0:
        detected_contam_lines = find_lines(
            spectrum_wave, spectrum_flux, contam_catalog,
            remove_duplicate=do_duplicate_removal,
            snr_threshold=snr_threshold,
            noise_std=shared_noise_std
        )
        detected_contam = detected_contam_lines[detected_contam_lines['disposition'] == 1].copy()
        detected_contam['sigma'] = [_line_sigma(wl, header) for wl in detected_contam['line']]
    else:
        # Empty DataFrame - still needs the right columns for fit_glines
        detected_contam = pd.DataFrame(columns=['line', 'disposition', 'max_amplitude', 'species', 'sigma'])

    # DEBUG: Show detected lines after proper initialization
    n_contam_detected = len(detected_contam)
    print(f"  [DEBUG] Line detection: {n_co_detected} CO lines detected, {n_contam_detected} contaminating lines")

    # Fit if any lines detected
    if len(detected_co) > 0 or len(detected_contam) > 0:
        # Only fit_glines' required positional arg -- every detected line already carries its
        # own real 'sigma', so this scalar is never actually read (see fit_glines' docstring).
        all_sigmas = pd.concat([detected_co['sigma'], detected_contam['sigma']])
        sigma = float(np.median(all_sigmas))

        # Same _merged_wave_boundaries() approach as fit_hitran_lines_collective, built from
        # this pixel's own detected lines -- each pixel detects a different subset, so needs its own.
        wave_boundaries_extent = None
        if wave_boundaries is None:
            all_centers = list(detected_co['line']) + list(detected_contam['line'])
            wave_boundaries, wave_boundaries_extent = _merged_wave_boundaries(all_centers, list(all_sigmas))

        print(f"  [DEBUG] Before fit_glines: median sigma={sigma:.6f}, n_waves={len(wave_boundaries)} boundaries")
        print(f"  [DEBUG] Fitting {len(detected_co)} CO + {len(detected_contam)} contaminating lines")
        # No fallback string on purpose -- see fit_glines' docstring: None here means the
        # header is genuinely missing something a MIRI/NIRSpec/MUSE header shouldn't be missing.
        co_flux, contam_flux, mean_cont, residuals, continuum_curve, _, _ = fit_glines(
            spectrum_wave, spectrum_flux, wave_boundaries,
            detected_co, detected_contam, sigma,
            flux_unit_label=resolved_flux_unit, wave_unit_label=resolved_wave_unit,
            list_wboundaries_extent=wave_boundaries_extent,
        )
        _debug_area_unit = f"{resolved_flux_unit}·{resolved_wave_unit}"
        print(f"  [DEBUG] fit_glines returned (native, pre-CGS-conversion): "
              f"co_flux={co_flux:.6e} {_debug_area_unit}, contam_flux={contam_flux:.6e} {_debug_area_unit}, "
              f"mean_cont={mean_cont:.6e} {resolved_flux_unit}")
        print(f"  [DEBUG] co_flux type={type(co_flux)}, is_finite={np.isfinite(co_flux)}")
    else:
        co_flux, contam_flux, mean_cont = 0, 0, np.nanmedian(spectrum_flux)
        residuals = pd.DataFrame({'wave': spectrum_wave, 'residual': np.zeros_like(spectrum_flux)})
        continuum_curve = pd.DataFrame({'wave': spectrum_wave, 'continuum': np.full_like(spectrum_flux, np.nan)})
        detected_co = pd.DataFrame()
        # print(f"  No lines to fit, returning 0")
    
    # Calculate mean wavelength of detected CO lines for flux conversion
    mean_detected_wavelength = np.nan
    if len(detected_co) > 0 and 'line' in detected_co.columns:
        detected_co_only = detected_co[detected_co['disposition'] == 1]
        if len(detected_co_only) > 0:
            mean_detected_wavelength = np.mean(detected_co_only['line'].values)

    # conversion
    co_flux_cgs = _to_cgs(co_flux, mean_detected_wavelength)
    if co_flux_cgs is not None:
        contam_flux_cgs = _to_cgs(contam_flux, mean_detected_wavelength)
        co_flux, contam_flux = co_flux_cgs, contam_flux_cgs if contam_flux_cgs is not None else contam_flux

    # continuum_curve so far only holds fit_glines' own per-chunk baseline (a + b*wave), fit
    # against the ALREADY continuum-subtracted spectrum.
    if continuum_model is not None:
        continuum_curve = continuum_curve.copy()
        continuum_curve['continuum'] = continuum_model + continuum_curve['continuum'].fillna(0.0).values

    return {
        'co_flux': co_flux,
        'contam_flux': contam_flux,
        'mean_continuum': mean_cont,
        'detected_lines': detected_lines,
        'detected_contam': detected_contam,
        'residuals': residuals,
        'continuum_curve': continuum_curve,
        'flux_unit': flux_unit_label,
        'mean_detected_wavelength': mean_detected_wavelength
    }


# ================== Full Cube Processing ==================

def create_co_flux_map(cube_data, CO_catalog, contam_catalog, n_jobs=8, progress_callback=None, cancel_flag=None,
                       wave_min=None, wave_max=None, excluded_wavelengths=None,
                       continuum_params=None):
    """
    Create a 2D map of total CO rovibrational flux by extracting each pixel's spectrum and
    calling extract_co_flux_from_spectrum() on it, in parallel across rows.

    Args:
        cube_data (CubeData): Loaded cube; needs cube, wavelength, get_pixel_spectrum(x, y)
        CO_catalog (pd.DataFrame): CO line catalog (columns: line_wave_um, species, ...)
        contam_catalog (pd.DataFrame): Contaminating lines catalog (column: line_wave_um)
        n_jobs (int): Number of parallel worker threads for processing rows
        progress_callback (callable): Optional progress_callback(current, total)
        cancel_flag (object): Optional object with a `.cancelled` attribute, checked per pixel
        wave_min (float): Minimum wavelength to fit. Defaults to min of CO_catalog
        wave_max (float): Maximum wavelength to fit. Defaults to max of CO_catalog
        excluded_wavelengths (set or list): CO wavelengths to exclude from flux calculation
        continuum_params (dict): Overrides for fit_auto_continuum()'s defaults, applied
            identically to every pixel

    Returns:
        result (dict): 'flux_map', 'contam_flux_map' (2D arrays, erg/s/cm2 if cube_data.header
            supports the conversion, else native units), 'detection_map'/'contam_detection_map'
            (2D line-count arrays), 'wavelength', 'header', 'wave_range' (actually used),
            'flux_unit' ('erg/s/cm2' if the conversion succeeded)
    """
    
    if cube_data.cube is None:
        print("[ERROR] No cube data loaded")
        return None
    
    # Get cube dimensions
    n_wave, n_y, n_x = cube_data.cube.shape
    wavelength = cube_data.wavelength
    
    # Strip units from wavelength if it's an astropy Quantity (convert to plain numpy array)
    wavelength, wave_unit = extract_units(wavelength)
    if wave_unit:
        print(f"    Wavelength unit: {wave_unit}")
    
    # Calculate wavelength range from CO catalog (not full spectrum) for efficiency
    if wave_min is None:
        wave_min = CO_catalog['line_wave_um'].min()
    if wave_max is None:
        wave_max = CO_catalog['line_wave_um'].max()
    
    # DIAGNOSTIC: Check line catalogs
    print(f"    CO catalog: {len(CO_catalog)} lines")
    if len(CO_catalog) > 0:
        print(f"      Wavelength range: {CO_catalog['line_wave_um'].min():.4f} - {CO_catalog['line_wave_um'].max():.4f}")
    if contam_catalog is not None:
        print(f"    Contaminating catalog: {len(contam_catalog)} lines")
        if len(contam_catalog) > 0 and 'line_wave_um' in contam_catalog.columns:
            print(f"      Wavelength range: {contam_catalog['line_wave_um'].min():.4f} - {contam_catalog['line_wave_um'].max():.4f}")
    else:
        print(f"    Contaminating catalog: None (user did not upload contaminating lines)")
    
    # DIAGNOSTIC: Check if line wavelengths overlap cube data wavelengths
    co_overlap = (CO_catalog['line_wave_um'] >= wave_min) & (CO_catalog['line_wave_um'] <= wave_max)
    print(f"    CO lines in data wavelength range: {np.sum(co_overlap)}/{len(CO_catalog)}")
    
    sys.stdout.flush()
    
    # Create output maps
    flux_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    contam_flux_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    detection_map = np.full((n_y, n_x), 0, dtype=np.int32)
    contam_detection_map = np.full((n_y, n_x), 0, dtype=np.int32)
    wavelength_map = np.full((n_y, n_x), np.nan, dtype=np.float32)  # Track mean detected wavelength per pixel
    
    print(f">>> Creating CO flux map: {n_y}x{n_x} pixels, {n_wave} wavelength elements")
    print(f"    Wavelength range: {wave_min:.6f} - {wave_max:.6f} um")
    print(f"    Continuum subtraction: ENABLED")
    
    # DIAGNOSTIC: Print cube metadata
    print(f"    === CUBE DIAGNOSTICS ===")
    print(f"    Cube shape: {cube_data.cube.shape}")
    print(f"    Cube dtype: {cube_data.cube.dtype}")
    print(f"    Cube min/max: {np.nanmin(cube_data.cube):.4e} / {np.nanmax(cube_data.cube):.4e}")
    print(f"    Cube % NaN: {100 * np.sum(np.isnan(cube_data.cube)) / cube_data.cube.size:.2f}%")
    print(f"    Wavelength shape: {wavelength.shape}")
    print(f"    Wavelength min/max: {wavelength[0]:.6f} / {wavelength[-1]:.6f}")
    print(f"    Spectral axis: {cube_data.spec_axis}")
    sys.stdout.flush()
    
    start_time = time.time()
    
    # Thread-safe pixel counter for per-pixel progress updates
    pixel_counter = {'count': 0, 'lock': Lock()}
    total_pixels = n_y * n_x
    
    # No wave_boundaries computed here -- each pixel detects a different set of lines, so
    # extract_co_flux_from_spectrum builds its own, per call, from what it actually detected.

    # DEBUG: Test extraction on first few pixels
    print(f"    Testing pixel extraction...")
    try:
        # Try multiple pixels to find one with data
        found_valid = False
        for test_y in range(min(3, n_y)):
            for test_x in range(min(3, n_x)):
                test_spectrum = cube_data.get_pixel_spectrum(test_x, test_y)
                if test_spectrum is not None:
                    n_valid = np.sum(~np.isnan(test_spectrum))
                    print(f"      Pixel ({test_x}, {test_y}): {n_valid}/{len(test_spectrum)} valid values")
                    if n_valid > 0:
                        print(f"        Min/max: {np.nanmin(test_spectrum):.4e} / {np.nanmax(test_spectrum):.4e}")
                        found_valid = True
                else:
                    print(f"      Pixel ({test_x}, {test_y}): returned None")
        
        if not found_valid:
            print(f"    [WARNING] No valid pixels found in sample!")
            # Try sampling middle of cube instead of edges
            print(f"    Trying middle pixels...")
            mid_y, mid_x = n_y // 2, n_x // 2
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    try_y, try_x = mid_y + dy, mid_x + dx
                    test_spectrum = cube_data.get_pixel_spectrum(try_x, try_y)
                    if test_spectrum is not None:
                        n_valid = np.sum(~np.isnan(test_spectrum))
                        print(f"      Pixel ({try_x}, {try_y}): {n_valid}/{len(test_spectrum)} valid values")
                        if n_valid > 0:
                            print(f"        Min/max: {np.nanmin(test_spectrum):.4e} / {np.nanmax(test_spectrum):.4e}")
                            found_valid = True
        
        if found_valid:
            print(f"    [OK] Found valid pixels in center region")
        
        sys.stdout.flush()
        
        # Now extract from first pixel
        test_spectrum = cube_data.get_pixel_spectrum(0, 0)
        if test_spectrum is not None and not np.all(np.isnan(test_spectrum)):
            print(f"      Spectrum: shape={test_spectrum.shape}, min={np.nanmin(test_spectrum):.4e}, max={np.nanmax(test_spectrum):.4e}")
            sys.stdout.flush()  # Force flush
            test_result = extract_co_flux_from_spectrum(
                wavelength, test_spectrum, CO_catalog, contam_catalog,
                wave_min=wave_min, wave_max=wave_max,
                do_duplicate_removal=True,
                continuum_params=continuum_params,
                header=cube_data.header, wave_unit=cube_data.wave_unit
            )
            if test_result:
                print(f"      Result: co_flux={test_result['co_flux']:.4e}, detected_lines shape={len(test_result['detected_lines'])}")
                if len(test_result['detected_lines']) > 0:
                    n_co_detected = len(test_result['detected_lines'][test_result['detected_lines']['disposition'] == 1])
                    print(f"      CO lines with disposition=1: {n_co_detected}")
                sys.stdout.flush()
            else:
                print(f"      Result was None!")
                sys.stdout.flush()
        else:
            print(f"      Test spectrum is None or all NaN")
    except Exception as e:
        print(f"      Error: {e}")
        traceback.print_exc()
        sys.stdout.flush()

    
    def process_row(y_idx):
        """
        Extract and fit every pixel's spectrum in one cube row (thread-pool worker unit).

        Args:
            y_idx (int): Row index in the cube's spatial y-axis

        Returns:
            row (tuple): (y_idx, row_fluxes, row_contam_fluxes, row_detections,
                row_contam_detections, row_wavelengths), each a length-n_x array except y_idx
        """
        row_fluxes = np.full(n_x, np.nan, dtype=np.float32)
        row_contam_fluxes = np.full(n_x, np.nan, dtype=np.float32)
        row_detections = np.full(n_x, 0, dtype=np.int32)
        row_contam_detections = np.full(n_x, 0, dtype=np.int32)
        row_wavelengths = np.full(n_x, np.nan, dtype=np.float32)  # Track mean detected wavelength per pixel

        # DIAGNOSTIC: Count successes and failures
        num_success = 0
        num_failures = 0
        num_skipped = 0
        
        for x_idx in range(n_x):
            # Check cancellation flag at each pixel
            if cancel_flag and cancel_flag.cancelled:
                print(f"    Row {y_idx}: Cancellation requested, stopping pixel processing")
                break
            
            # Update pixel counter for progress reporting
            with pixel_counter['lock']:
                pixel_counter['count'] += 1
            try:
                # Step 1: Extract spectrum from cube for this pixel [REUSED: data_loader.get_pixel_spectrum]
                spectrum_flux = cube_data.get_pixel_spectrum(x_idx, y_idx)
                
                if spectrum_flux is None or np.all(np.isnan(spectrum_flux)):
                    num_skipped += 1
                    continue  # Skip pixels with no data
                
                # Step 2: Extract CO flux [FUNCTION: extract_co_flux_from_spectrum]
                result = extract_co_flux_from_spectrum(
                    wavelength, spectrum_flux, CO_catalog, contam_catalog,
                    wave_min=wave_min, wave_max=wave_max,
                    do_duplicate_removal=True,
                    excluded_wavelengths=excluded_wavelengths,
                    continuum_params=continuum_params,
                    header=cube_data.header, wave_unit=cube_data.wave_unit
                )
                
                # Filter detected lines by wavelength range
                if result is not None and len(result['detected_lines']) > 0:
                    result['detected_lines'] = result['detected_lines'][
                        (result['detected_lines']['line'] >= wave_min) &
                        (result['detected_lines']['line'] <= wave_max)
                    ]
                if result is not None and len(result['detected_contam']) > 0:
                    result['detected_contam'] = result['detected_contam'][
                        (result['detected_contam']['line'] >= wave_min) &
                        (result['detected_contam']['line'] <= wave_max)
                    ]

                if result is not None:
                    # Check if flux value is actually valid
                    co_flux_val = result['co_flux']

                    # Step 3: Store results in maps
                    row_fluxes[x_idx] = co_flux_val
                    row_contam_fluxes[x_idx] = result['contam_flux']
                    row_wavelengths[x_idx] = result.get('mean_detected_wavelength', np.nan)  # Store wavelength for conversion
                    
                    # Count detected CO lines
                    n_detected = len(result['detected_lines'][result['detected_lines']['disposition'] == 1])
                    row_detections[x_idx] = n_detected

                    # Count detected contaminating lines
                    n_contam_detected = len(result['detected_contam'][result['detected_contam']['disposition'] == 1]) \
                        if len(result['detected_contam']) > 0 else 0
                    row_contam_detections[x_idx] = n_contam_detected

                    num_success += 1
                else:
                    num_failures += 1
            
            except Exception as e:
                # Log error for debugging - CRITICAL: need to see what's failing
                print(f"  [ERROR] Extraction failed at pixel ({x_idx}, {y_idx}): {str(e)}")
                traceback.print_exc()
                num_failures += 1
                continue
            
            # Print pixel-level progress inside row (every 10 pixels)
            if (x_idx + 1) % max(1, n_x // 5) == 0:
                print(f"    Row {y_idx}: {x_idx + 1}/{n_x} pixels...")
        
        return y_idx, row_fluxes, row_contam_fluxes, row_detections, row_contam_detections, row_wavelengths
    
    # Process rows in parallel
    try:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_row, y) for y in range(n_y)]
            
            for i, future in enumerate(as_completed(futures)):
                # Check if cancellation was requested
                if cancel_flag and cancel_flag.cancelled:
                    print("[WARNING] Extraction cancelled by user - stopping all remaining tasks")
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    # Force shutdown immediately
                    executor.shutdown(wait=False)
                    break
                
                try:
                    y_idx, row_fluxes, row_contam_fluxes, row_detections, row_contam_detections, row_wavelengths = future.result()
                    flux_map[y_idx, :] = row_fluxes
                    contam_flux_map[y_idx, :] = row_contam_fluxes
                    detection_map[y_idx, :] = row_detections
                    contam_detection_map[y_idx, :] = row_contam_detections
                    wavelength_map[y_idx, :] = row_wavelengths
                except Exception:
                    # Skip this row if cancelled or failed
                    pass
                
                # Report actual pixel progress after each row
                if progress_callback:
                    with pixel_counter['lock']:
                        pixels_done = pixel_counter['count']
                    progress_callback(pixels_done, total_pixels)
                    
                # Print progress EVERY row (not just 10%)
                with pixel_counter['lock']:
                    pixels_done = pixel_counter['count']
                percent = 100 * pixels_done / total_pixels
                elapsed = time.time() - start_time
                rate = pixels_done / elapsed if elapsed > 0 else 0
                eta_sec = (total_pixels - pixels_done) / rate if rate > 0 else 0
                print(f"    Row {i+1}/{n_y}: {percent:.1f}% ({pixels_done}/{total_pixels}) in {elapsed:.1f}s, ETA {eta_sec:.1f}s")
                sys.stdout.flush()
    except Exception as e:
        print(f"[ERROR] Error during CO flux extraction: {e}")
        traceback.print_exc()
    
    end_time = time.time()
    print(f"\n>>> CO flux map creation complete ({end_time - start_time:.1f}s)")
    print(f"    Valid pixels: {np.sum(~np.isnan(flux_map))}/{n_y * n_x}")
    print(f"    Mean CO flux: {np.nanmean(flux_map):.4e}")
    print(f"    Max CO flux: {np.nanmax(flux_map):.4e}")
    print(f"    Mean contam flux: {np.nanmean(contam_flux_map):.4e}")
    print(f"    Max contam flux: {np.nanmax(contam_flux_map):.4e}")
    print(f"    Total contam flux (sum over valid pixels): {np.nansum(contam_flux_map):.4e}")
    print(f"    Pixels with a contam detection: {np.sum(contam_detection_map > 0)}/{n_y * n_x}")
    
    # get units
    header = cube_data.header
    resolved_wave_unit = get_wave_unit(header, cube_data.wave_unit)
    resolved_flux_unit = _resolve_flux_unit(None, header)
    test_cgs = None
    if header is not None:
        test_cgs = integrated_flux_to_cgs(1.0, resolved_flux_unit, wavelength=(wave_min + wave_max) / 2.0,
                                           header=header, wave_unit=resolved_wave_unit)
    flux_unit_output = 'erg/s/cm2' if test_cgs is not None else (resolved_flux_unit or 'UNKNOWN')
    print(f"    Flux map units: {flux_unit_output}")

    return {
        'flux_map': flux_map,
        'contam_flux_map': contam_flux_map,
        'detection_map': detection_map,
        'contam_detection_map': contam_detection_map,
        'wavelength': wavelength,
        'header': cube_data.header,
        'wave_range': (wave_min, wave_max),
        'flux_unit': flux_unit_output
    }


def save_flux_map_fits(flux_map_data, output_path, map_type='CO_flux'):
    """
    Save a CO flux map (from create_co_flux_map) as a 2-HDU FITS file: PRIMARY (flux map)
    and DETECTION (line counts per pixel).

    Args:
        flux_map_data (dict): Output of create_co_flux_map() -- needs 'flux_map',
            'detection_map', 'wavelength', 'header', 'wave_range'
        output_path (str): Output file path (e.g. 'CO_flux_map.fits')
        map_type (str): Map type label, e.g. 'CO_flux' or 'contamination_flux'

    Returns:
        success (bool): True if the file was written, False if it failed
    """
    
    try:
        flux_map = flux_map_data['flux_map']
        detection_map = flux_map_data['detection_map']
        wavelength = flux_map_data['wavelength']
        header = flux_map_data.get('header', fits.Header())
        wave_range = flux_map_data.get('wave_range', (np.nanmin(wavelength), np.nanmax(wavelength)))
        flux_unit_output = flux_map_data.get('flux_unit', None)  # Get output flux unit if available
        
        # Prepare header [REUSED: Original header as base]
        if header is not None:
            out_header = header.copy()
        else:
            out_header = fits.Header()
        
        # Extract BUNIT - use output unit if conversion was done, otherwise use header
        try:
            if flux_unit_output and flux_unit_output != 'MJy/sr um':
                # Conversion was applied
                if flux_unit_output == 'erg/s/cm2':
                    flux_unit_str = "erg/s/cm2"
                else:
                    flux_unit_str = flux_unit_output
            else:
                # Use original header -- 'UNKNOWN', not a plausible-looking guess, if it's
                # genuinely not there
                bunit_str = out_header.get('BUNIT', 'UNKNOWN')
                # Try to parse as astropy unit
                try:
                    base_unit = u.Unit(bunit_str)
                    flux_unit_str = f"{base_unit} um"
                except (ValueError, TypeError):
                    # Fallback to string concatenation if not a valid unit
                    flux_unit_str = f"{bunit_str} um"
        except Exception:
            flux_unit_str = flux_unit_output if flux_unit_output else "UNKNOWN"
        
        # Update header with new information
        out_header['NAXIS'] = 2
        out_header['NAXIS1'] = flux_map.shape[1]
        out_header['NAXIS2'] = flux_map.shape[0]
        out_header['MAPTYPE'] = (map_type, 'Type of flux map')
        out_header['WAVEMIN'] = (wave_range[0], 'Minimum wavelength used (um)')
        out_header['WAVEMAX'] = (wave_range[1], 'Maximum wavelength used (um)')
        out_header['BUNIT'] = flux_unit_str
        out_header['COMMENT'] = f'CO Rovibrational {map_type} Map'
        out_header['COMMENT'] = 'Flux extracted by simultaneous multi-Gaussian fitting'
        out_header['COMMENT'] = 'Values are amplitude * sigma * sqrt(2*pi) integrated from all detected CO lines'
        
        # Create primary HDU with flux map
        primary_hdu = fits.PrimaryHDU(flux_map.astype(np.float32), header=out_header)
        
        # Create detection map extension
        detection_hdu = fits.ImageHDU(detection_map.astype(np.int32), name='DETECTION')
        detection_hdu.header['BUNIT'] = 'count'
        detection_hdu.header['COMMENT'] = 'Number of CO lines detected and fit in each pixel'
        
        # Create HDU list and write
        hdul = fits.HDUList([primary_hdu, detection_hdu])
        hdul.writeto(output_path, overwrite=True)
        
        print(f"[OK] Flux map saved to {output_path}")
        print(f"  - PRIMARY HDU: {map_type} ({flux_map.shape[0]}x{flux_map.shape[1]})")
        print(f"  - ERROR HDU: Uncertainty map")
        print(f"  - DETECTION HDU: Line counts per pixel")
        
        return True
    
    except Exception as e:
        print(f"[ERROR] Error saving FITS file: {e}")
        return False
