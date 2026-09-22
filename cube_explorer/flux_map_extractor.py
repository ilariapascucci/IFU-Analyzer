"""
Flux Extraction Module

Fits multiple target lines and contaminating lines simultaneously from IFU spectra,
extracting and summing the total flux to create integrated flux maps.
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
    compute_noise_std, compute_mad_noise_std,
    GaussianFitResult, gaussian,
)
from .utils import get_spectral_resolution
from .unit_converter import (
    integrated_flux_to_cgs, get_wave_unit, _resolve_flux_unit, convert_area_to_cgs,
    extract_units, extract_value_from_quantity,
)

# fit_glines' actual curve_fit bounds on fitted sigma, x instrumental sigma.
SIGMA_BOUND_LOWER_MULTIPLIER = 0.5
SIGMA_BOUND_UPPER_MULTIPLIER = 2.0

# Systematic flux-calibration floor (Pascucci, told directly for the rot-diagram case;
# applied uniformly here so every consumer of area_error gets the same number).
SYSTEMATIC_FLUX_ERROR_FRACTION = 0.05

# Display/diagnostic-only reduced constant (does not affect curve_fit)
DISPLAY_CROP_SIGMA_MULTIPLIER = 3.0

# A line whose area_error exceeds this fraction of its own area is flagged unreliable
# (near-singular covariance, usually two lines blended under 1 instrumental sigma apart).
AREA_ERROR_SANITY_RATIO = 1.0

# Derived from the bounds above (center +/-1sigma, fitted sigma <= SIGMA_BOUND_UPPER_MULTIPLIER):
# 3*(2x instrumental sigma) + 1sigma center shift = 7sigma is as far as real flux can land.
DEFAULT_SIGMA_MULTIPLIER = 7.0

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
    Sum of multiple Gaussians plus a linear continuum. This function uses a broadcast, not
    a loop over each point and line, because curve_fit's Jacobian calls this model many
    times per fit. The result is the same; only the method is different.

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
    # A loop over each point and line here used 81% of one chunk's fit time in tests.
    # The broadcast below computes the same (n_points, n_lines) grid without a loop.
    gaussians = amps[np.newaxis, :] * np.exp(
        -0.5 * ((xl[:, np.newaxis] - cens[np.newaxis, :]) / sigs[np.newaxis, :]) ** 2
    )
    return a + b * (xl - min_freq) + gaussians.sum(axis=1)


# ================== Line Detection ==================

def find_lines(xwave, yflux, lines_to_search, contaminating_lines=None, remove_duplicate=False, snr_threshold=3.0, noise_std=None):
    """
    Identify which spectral lines are detected in a spectrum, and set a starting
    max_amplitude for each one. Finds peaks near each line and compares them to the
    local noise, then removes duplicate lines and avoids contaminating lines.

    Args:
        xwave (array): Wavelength array
        yflux (array): Flux array
        lines_to_search (pd.DataFrame): Columns 'line_wave_um', 'species', others optional
        contaminating_lines (pd.DataFrame): Optional column 'line_wave_um' for lines to avoid
        remove_duplicate (bool): True to remove duplicate detections (keep the strongest per group)
        snr_threshold (float): Signal-to-noise threshold for line detection, in units of sigma
        noise_std (float): Precomputed noise estimate. Pass this to avoid a repeat noise
            calculation for the same (xwave, yflux) pair, for example in
            extract_flux_from_spectrum's two calls. If None, this function computes it.

    Returns:
        lines (pd.DataFrame): Columns line, species, disposition (0/1), max_amplitude, line_flux
    """
    # Suppress repetitive output (called for every pixel)
    # print('>>> Finding lines and removing duplicates...')
    xres = np.nanmean(np.diff(xwave))

    # Estimate noise level for significance testing
    if noise_std is None:
        noise_std = compute_mad_noise_std(yflux)

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
                # No species bias here (see the matching comment in extract_flux_from_spectrum).
                # The catalog can hold any molecule, so this code keeps only the strongest line.
                strongest_line_idx = max(group, key=lambda idx: lines_dispositions.loc[idx, 'line_flux'])
                
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

    # Pipeline: polynomial features + RANSAC. random_state pinned (like
    # gaussian_fitter.py) to keep the continuum and its noise estimate reproducible.
    ransac_pipeline = make_pipeline(
        PolynomialFeatures(degree=pdeg),
        RANSACRegressor(random_state=42)
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
        yflux (array): Flux array. The continuum is still in it; this function fits the
            continuum and the lines together
        list_wboundaries (array): Flat, sorted chunk boundaries: elements [2*i, 2*i+1] make
            chunk i's own (lo, hi). This array sets which lines belong to which chunk; each
            line belongs to exactly one chunk
        detco_lines (pd.DataFrame): Detected CO lines with 'line'/'disposition'/'max_amplitude',
            plus an optional per-line 'sigma' (falls back to the shared `sigma` if absent)
        detcontaminating_lines (pd.DataFrame): Same columns, for contaminating lines
        sigma (float): Shared fallback Gaussian sigma, used only where a line has no own 'sigma'
        progress_callback (callable): Optional progress_callback(current_chunk, total_chunks),
            called after each chunk finishes (converged or not)
        flux_unit_label (str): Cosmetic unit label, for the debug prints only. Every returned
            value stays in xwave/yflux's own native units. This function does not convert to CGS
        wave_unit_label (str): Cosmetic wavelength unit label, same debug-print-only role
        header (astropy.io.fits.Header): Optional. Used only for an approximate erg/s/cm**2
            preview next to the native value in the debug prints. This is not a source of truth
        list_wboundaries_extent (array): Same shape as list_wboundaries, but each chunk can be
            wider (see _merged_wave_boundaries). curve_fit and the plots use this array, so a
            capped chunk never cuts off a member line's own window. If None, uses list_wboundaries
        half_width_sigma (float): Must match the value used to build list_wboundaries. This
            function uses it only to find "borrowed" lines (see below), not to rebuild boundaries

    Returns:
        result (tuple): (sum_CO_area, sum_contaminating_area, mean_cont, df_residual,
            df_continuum, per_line_fits, per_contam_fits), all in native units. per_line_fits
            and per_contam_fits are lists of per-line dicts (amplitude, center, sigma, area,
            x/y/fit data). A caller that needs per-line detail, for example
            fit_hitran_lines_collective, uses them. A caller that needs only the total, for
            example extract_flux_from_spectrum, can ignore them.
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

    # Every known line. Used below to find lines a chunk must borrow: not owned, but close
    # enough to bias the continuum if left unfit. Built once, then filtered per chunk.
    all_lines_pool = pd.concat([detco_lines, detcontaminating_lines], ignore_index=True)
    pool_wave = all_lines_pool['line'].astype(float).values
    pool_disp = all_lines_pool['disposition'].values
    pool_sigma = (all_lines_pool['sigma'].astype(float).values if 'sigma' in all_lines_pool.columns
                  else np.full(len(all_lines_pool), sigma))
    pool_nat_lo = pool_wave - half_width_sigma * pool_sigma
    pool_nat_hi = pool_wave + half_width_sigma * pool_sigma

    def _fit_record(cen_i, amp_i, sig_i, cat_wave, amp_err_i, sig_err_i, catalog_sigma_i,
                     a, b, par, min_freq_chunk, x_local, y_local, noise_std_chunk):
        """
        Build one line's per_line_fits/per_contam_fits dict from a joint fit's own params.

        Args:
            cen_i (float): This line's fitted center
            amp_i (float): This line's fitted amplitude
            sig_i (float): This line's fitted sigma
            cat_wave (float): Catalog (rest) wavelength for this line
            amp_err_i (float): This line's amplitude uncertainty
            sig_err_i (float): This line's sigma uncertainty
            catalog_sigma_i (float): The fixed instrumental sigma this line was given
            a (float): Joint fit's continuum offset
            b (float): Joint fit's continuum slope
            par (array): Joint fit's Gaussian parameters, 3 per line, amp/center/sigma order
            min_freq_chunk (float): This extent's own minimum, continuum's x origin
            x_local (array): This extent's own wavelength samples
            y_local (array): This extent's own flux samples
            noise_std_chunk (float): This extent's own residual-based noise estimate

        Returns:
            record (dict): Per-line fit dict, see fit_glines' own Returns for its keys
        """
        # Full extent model: the local continuum plus every Gaussian in it, not just this
        # line's own. This shows the real data, including overlap with a neighbor.
        y_fit_local = a + b * (x_local - min_freq_chunk)
        for jj in range(0, len(par), 3):
            y_fit_local = y_fit_local + gaussian(x_local, par[jj], par[jj + 1], par[jj + 2])
        # Same local continuum, but only this line's own fitted Gaussian, not cropped.
        # Any gap from y_fit_local near a neighbor line stays visible here.
        y_fit_own = a + b * (x_local - min_freq_chunk) + gaussian(x_local, amp_i, cen_i, sig_i)
        return {
            'wavelength': cat_wave, 'amplitude': amp_i, 'center': cen_i, 'sigma': sig_i,
            'area': amp_i * sig_i * np.sqrt(2 * np.pi),
            # amp_err_i/sig_err_i: from curve_fit's joint covariance matrix, not a separate
            # local fit. A line that overlaps a close neighbor gets a larger error here.
            'amplitude_error': amp_err_i,
            'sigma_error': sig_err_i,
            # catalog_sigma_i: the fixed instrumental sigma given as input, not the fitted
            # `sig_i`. Used for the upper-limit Gaussian when the fit is not real.
            'catalog_sigma': catalog_sigma_i,
            'noise_std': noise_std_chunk,
            'x_data': x_local, 'y_data': y_local, 'y_fit': y_fit_local,
            'y_fit_own': y_fit_own,
            'success': True, 'message': 'Success',
        }

    print('===== Start fitting...')
    start_time = time.time()

    n_chunks = len(list_wboundaries) // 2
    # Keyed by each extent's own (lo, hi). Distinct owners can widen to the exact same
    # extent in a dense line list -- reuse that joint fit instead of repeating it per owner.
    extent_fit_cache = {}
    # Step 2, not 1: each chunk is its own (lo, hi) pair at [ii, ii+1]. Extent pairs can
    # genuinely overlap between chunks by design, so a sliding-window read is wrong here.
    for ii in range(0, len(list_wboundaries), 2):
        xwave_min, xwave_max = list_wboundaries[ii], list_wboundaries[ii + 1]
        # xwave_min/xwave_max set which lines belong to this chunk. ext_min/ext_max (can be
        # wider) set the data the chunk fits and plots, so a small chunk keeps each line's window.
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

        cache_key = (round(float(ext_min), 9), round(float(ext_max), 9))
        cached = extent_fit_cache.get(cache_key)
        if cached is not None and cached['success']:
            print(f'  [CACHE] Reusing joint fit for extent {cache_key}: '
                  f'{n_co} owned CO + {ncont_lines} owned contaminant line(s)')
            a, b, par, cov = cached['a'], cached['b'], cached['par'], cached['cov']
            pool_wave_cached = cached['pool_wave']
            min_freq_chunk, noise_std_chunk = cached['min_freq_chunk'], cached['noise_std_chunk']
            x_local, y_local = xwave[idx_xwave], yflux[idx_xwave]

            # Nearest-match, not exact float equality -- robust to any float-path drift
            # between how this owner's line and the cache-populating owner's line arrived.
            def _cached_slice(line_wave):
                """Look up the cached fit slice nearest to this line's wavelength.

                Args:
                    line_wave (float): Wavelength of the line to look up.

                Returns:
                    slice (tuple): (amplitude, center, sigma, amp_err, sig_err) for the nearest cached line.
                """
                k = int(np.argmin(np.abs(pool_wave_cached - line_wave)))
                amp_i, cen_i, sig_i = par[3 * k], par[3 * k + 1], par[3 * k + 2]
                amp_err_i = float(np.sqrt(np.abs(cov[2 + 3 * k, 2 + 3 * k])))
                sig_err_i = float(np.sqrt(np.abs(cov[2 + 3 * k + 2, 2 + 3 * k + 2])))
                return amp_i, cen_i, sig_i, amp_err_i, sig_err_i

            current_sum_CO = 0.0
            for _, coline in filtered_colines.iterrows():
                amp_i, cen_i, sig_i, amp_err_i, sig_err_i = _cached_slice(float(coline['line']))
                current_sum_CO += amp_i * sig_i * np.sqrt(2 * np.pi)
                catalog_sigma_i = coline['sigma'] if 'sigma' in coline else sigma
                per_line_fits.append(_fit_record(
                    cen_i, amp_i, sig_i, coline['line'], amp_err_i, sig_err_i, catalog_sigma_i,
                    a, b, par, min_freq_chunk, x_local, y_local, noise_std_chunk))
            current_sum_contaminating = 0.0
            for _, contam_line in filtered_contaminatinglines.iterrows():
                amp_j, cen_j, sig_j, amp_err_j, sig_err_j = _cached_slice(float(contam_line['line']))
                current_sum_contaminating += amp_j * sig_j * np.sqrt(2 * np.pi)
                catalog_sigma_j = contam_line['sigma'] if 'sigma' in contam_line else sigma
                per_contam_fits.append(_fit_record(
                    cen_j, amp_j, sig_j, contam_line['line'], amp_err_j, sig_err_j, catalog_sigma_j,
                    a, b, par, min_freq_chunk, x_local, y_local, noise_std_chunk))

            sum_CO_area = sum_CO_area + current_sum_CO
            sum_contaminating_area = sum_contaminating_area + current_sum_contaminating
            mean_cont = mean_cont + a
            if progress_callback is not None:
                progress_callback(ii // 2 + 1, n_chunks)
            continue

        # A borrowed line is not owned by this chunk, but its window overlaps this chunk's
        # data. This code fits it here too, but does not sum its flux; its own chunk does that.
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

        # curve_fit needs at least as many finite points as free params (2 + 3 per line). If
        # there are fewer, this code marks the chunk failed instead of giving a random answer.
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

        # Uses each line's own sigma if the caller gives one. If not, it uses the shared
        # `sigma` for every line, but one shared width can under-resolve a different line.
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
        # NaN-safe and scoped to this chunk, to match p0slope below. The whole-spectrum
        # np.nanmin(yflux) can sit far from this chunk's real level and give bad bounds.
        p0base, p0slope = np.nanmin(yflux[idx_xwave]), (yflux[idx_xwave][-1] - yflux[idx_xwave][0]) / (xwave[idx_xwave][-1] - xwave[idx_xwave][0])
        p0 = np.insert(p0, [0, 0], [p0base, p0slope])

        # Create NumPy arrays with lower and upper bounds
        lower_bounds_2d = np.column_stack((
            [0.0] * len(filtered_lines),
            filtered_lines['line'] - line_sigmas,
            SIGMA_BOUND_LOWER_MULTIPLIER * line_sigmas
        ))
        lower_bounds = np.ravel(lower_bounds_2d)
        lower_bounds = np.insert(lower_bounds, [0, 0], [p0base - 2 * np.abs(p0slope), -2 * np.abs(p0slope)])

        upper_bounds_2d = np.column_stack((
            filtered_lines['max_amplitude'].astype(float) * 1.1,
            filtered_lines['line'] + line_sigmas,
            SIGMA_BOUND_UPPER_MULTIPLIER * line_sigmas
        ))
        upper_bounds = np.ravel(upper_bounds_2d)
        upper_bounds = np.insert(upper_bounds, [0, 0], [p0base + 2 * np.abs(p0slope), 2 * np.abs(p0slope)])

        # Perform the fit (Pascucci fit_Glines). Inside the try on purpose now -- a real
        # non-convergence must hit the RANSAC fallback below, not crash the whole run.
        try:
            if cached is not None and not cached['success']:
                # A prior owner already hit this exact extent and failed; curve_fit is
                # deterministic on the same data, so skip straight to the fallback.
                raise RuntimeError('curve_fit already failed for this extent (cached)')
            par, cov = curve_fit(many_gaussians_list, xwave[idx_xwave], yflux[idx_xwave], p0=p0,
                                 bounds=(lower_bounds, upper_bounds), method='trf', ftol=1e-5, maxfev=5000)
            print(f'  [FIT DEBUG] curve_fit succeeded, returned {len(par)} parameters')

            a, b = par[0], par[1]  # Extracts which parameters relate to the continuum
            par = par[2:]  # Extracts which parameters relate to the Gaussians
            print(f'  [FIT DEBUG] continuum: a={a:.6e} {flux_unit_label}, b={b:.6e} {flux_unit_label}/{wave_unit_label}; '
                  f'{len(par)//3} Gaussian components')
            if len(par) >= 3:
                print(f'  [FIT DEBUG] First line: amp={par[0]:.6e} {flux_unit_label}, center={par[1]:.6f} {wave_unit_label}, '
                      f'sigma={par[2]:.6e} {wave_unit_label}')

            bfit = many_gaussians_list(xwave[idx_xwave], a, b, *par)
            df_residual.loc[idx_xwave, 'residual'] = yflux[idx_xwave] - bfit
            # many_gaussians_list's continuum term is a + b*(x - min_freq), not a + b*x. This
            # line must match that exactly, or the plotted curve is off by b*min_freq.
            df_continuum.loc[idx_xwave, 'continuum'] = a + b * (xwave[idx_xwave] - xwave[idx_xwave].min())

            # Noise from this chunk's own residuals, with the real dof of the full model. This
            # is the same value curve_fit uses to scale `cov`, so it matches the amp errors below.
            n_data_chunk = np.count_nonzero(idx_xwave)
            dof_chunk = n_data_chunk - (len(par) + 2)
            if dof_chunk > 0:
                noise_std_chunk = float(np.sqrt(np.sum((yflux[idx_xwave] - bfit) ** 2) / dof_chunk))
            else:
                print(f'  [NOISE WARN] Chunk {ii}: dof={dof_chunk} <= 0, cannot estimate noise from residuals')
                noise_std_chunk = None

            # Owned params only. This code still fits borrowed lines and subtracts them from
            # the residual, but never sums them here, so n_owned_params excludes them.
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

            # filtered_lines holds CO lines, then contaminating lines. `par` follows the same
            # order in 3-param groups: filtered_colines.iloc[i] maps to par[3*i:3*i+3].
            min_freq_chunk = np.nanmin(xwave[idx_xwave])
            x_local, y_local = xwave[idx_xwave], yflux[idx_xwave]
            extent_fit_cache[cache_key] = {
                'success': True, 'a': a, 'b': b, 'par': par, 'cov': cov,
                'pool_wave': filtered_lines['line'].astype(float).values,
                'min_freq_chunk': min_freq_chunk, 'noise_std_chunk': noise_std_chunk,
            }

            for i, (_, coline) in enumerate(filtered_colines.iterrows()):
                amp_err_i = float(np.sqrt(np.abs(cov[2 + 3 * i, 2 + 3 * i])))
                sig_err_i = float(np.sqrt(np.abs(cov[2 + 3 * i + 2, 2 + 3 * i + 2])))
                per_line_fits.append(_fit_record(
                    par[3 * i + 1], par[3 * i], par[3 * i + 2], coline['line'], amp_err_i, sig_err_i,
                    line_sigmas[i], a, b, par, min_freq_chunk, x_local, y_local, noise_std_chunk))
            for j, (_, contam_line) in enumerate(filtered_contaminatinglines.iterrows()):
                jj = 3 * (n_co + j)
                amp_err_j = float(np.sqrt(np.abs(cov[2 + jj, 2 + jj])))
                sig_err_j = float(np.sqrt(np.abs(cov[2 + jj + 2, 2 + jj + 2])))
                per_contam_fits.append(_fit_record(
                    par[jj + 1], par[jj], par[jj + 2], contam_line['line'], amp_err_j, sig_err_j,
                    line_sigmas[n_co + j], a, b, par, min_freq_chunk, x_local, y_local, noise_std_chunk))

        except RuntimeError as e:
            print("curve_fit did not converge! Fit a straight line and assign 3sigma upper limit", e)
            # Cache the failure too -- a later owner sharing this extent gets the same
            # RuntimeError from curve_fit, skipping straight to its own fallback.
            extent_fit_cache[cache_key] = {'success': False}
            continuum, std_residuals = cont_fit_ransac(xwave[idx_xwave], yflux[idx_xwave], pdeg=1)
            # Uses median(line_sigmas), not the scalar sigma. This gives the same value when
            # broadcast (Flux Map), but the real per-line width when it varies (fit_hitran_lines_collective).
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
            progress_callback(ii // 2 + 1, n_chunks)

    end_time = time.time()
    print(f"Fitting took {np.around(end_time - start_time, decimals=1)} seconds")
    print('===== End fitting')
    print(f'[FINAL FLUX] Total CO flux: {sum_CO_area:.6e} {flux_unit_label}·{wave_unit_label}, '
          f'Total contam flux: {sum_contaminating_area:.6e} {flux_unit_label}·{wave_unit_label}, '
          f'Mean continuum: {mean_cont/n_chunks:.6e} {flux_unit_label} '
          f'(all native/raw -- not yet CGS-converted; conversion happens in the caller)')

    mean_cont = mean_cont / n_chunks

    return sum_CO_area, sum_contaminating_area, mean_cont, df_residual, df_continuum, per_line_fits, per_contam_fits


def _line_sigma(wl, header):
    """
    Instrumental Gaussian sigma at wavelength `wl`, from the real spectral resolution
    (R = lambda/delta_lambda) in `header`. This function has no fallback value: MIRI and
    NIRSpec headers must have this data, so a bad header must raise an error here.

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



def _trough_is_elevated(center_a, center_b, xwave, y_corrected, noise_std, snr_threshold):
    """
    Check the real spectrum between two line centers for a genuine blend.

    Args:
        center_a (float): First line's center wavelength
        center_b (float): Second line's center wavelength
        xwave (array): Real wavelength array
        y_corrected (array): Real continuum-subtracted flux array, same length as xwave
        noise_std (float): Real noise estimate, same units as y_corrected
        snr_threshold (float): Trough must exceed this many noise_std to count as elevated

    Returns:
        elevated (bool): True if the two lines must be fit together (real flux stays above
            noise between them, or there is no real data there to judge by). False if the
            spectrum drops back to baseline, so the two lines can be fit separately.
    """
    mid = (center_a + center_b) / 2.0
    nearest = np.argsort(np.abs(xwave - mid))[:6]
    local = y_corrected[nearest]
    if len(local) == 0 or not np.any(np.isfinite(local)):
        return True
    return float(np.nanmin(local)) > snr_threshold * noise_std


def _merged_wave_boundaries(centers, sigmas, xwave, y_corrected, noise_std,
                             half_width_sigma=DEFAULT_SIGMA_MULTIPLIER, snr_threshold=3.0):
    """
    Build fit_glines' list_wboundaries from each line's own instrumental width and
    the real spectrum, not a fixed grid or a size cap. Each line gets a
    [center-N*sigma, center+N*sigma] window; two overlapping windows merge only when the real data between them stays above the noise floor (a genuine blend).

    Args:
        centers (array): Line center wavelengths
        sigmas (array): Each line's own instrumental sigma, same units as centers
        xwave (array): Real wavelength array, for the blend check between candidate lines
        y_corrected (array): Real continuum-subtracted flux array, same length as xwave
        noise_std (float): Real noise estimate, same units as y_corrected
        half_width_sigma (float): Each line's window half-width, in multiples of its own sigma
        snr_threshold (float): Trough significance bar for the blend check, in units of
            noise_std. Matches find_lines' own default detection threshold.

    Returns:
        ownership_boundaries (array): Flat, sorted chunk boundaries. Each consecutive pair
            makes one chunk. This array sets which lines fit_glines assigns to which chunk.
            Size is whatever the real blending in the data supports, not a fixed limit.
        extent_boundaries (array): Same shape as ownership_boundaries, and paired the same
            way. Each pair is widened, never narrowed, to cover every owned or borrowed
            line's own +/-half_width_sigma window, so none of them is ever data-starved.
            fit_glines must use this array to fit and plot, not ownership_boundaries.
    """
    intervals = sorted(
        (c - half_width_sigma * s, c + half_width_sigma * s, s, c)
        for c, s in zip(centers, sigmas)
    )
    merged = [list(intervals[0][:2])]
    prev_center = intervals[0][3]
    for lo, hi, s, c in intervals[1:]:
        windows_overlap = lo <= merged[-1][1]
        blended = windows_overlap and _trough_is_elevated(
            prev_center, c, xwave, y_corrected, noise_std, snr_threshold)
        if blended:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            # Cut at the real midpoint, not the previous chunk's raw edge -- that edge
            # can reach past this line's window and leave hi < lo otherwise.
            cut = (prev_center + c) / 2.0
            merged[-1][1] = min(merged[-1][1], cut)
            merged.append([max(lo, cut), hi])
        prev_center = c
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
        # Also widen for borrowed neighbors (not just owned members), repeating until
        # stable -- their real data must never be starved either.
        while True:
            reaching_in = (windows_lo < hi) & (windows_hi > lo)
            new_lo = min(lo, np.min(windows_lo[reaching_in])) if np.any(reaching_in) else lo
            new_hi = max(hi, np.max(windows_hi[reaching_in])) if np.any(reaching_in) else hi
            if new_lo >= lo and new_hi <= hi:
                break
            lo, hi = new_lo, new_hi
        extent_pairs.append((lo, hi))
    extent_boundaries = np.array([b for lo, hi in extent_pairs for b in (lo, hi)])

    return ownership_boundaries, extent_boundaries


def fit_hitran_lines_collective(spectrum_x, spectrum_y, hitran_lines, continuum,
                                 header=None, wave_unit=None, contam_lines=None,
                                 sigma_multiplier=DEFAULT_SIGMA_MULTIPLIER, progress_callback=None):
    """
    Fit a hand-picked list of HITRAN lines (for example, from the species dialog) at once,
    in per-line instrumental-width chunks, through fit_glines(). This uses the same engine
    as Flux Map, but with no SNR detection; the caller has already picked the lines.

    Args:
        spectrum_x (array): Full spectrum wavelength array (numpy or Quantity)
        spectrum_y (array): Full spectrum flux array (numpy or Quantity)
        hitran_lines (list): Dicts with 'lambda_micron' or 'wavelength', plus any metadata
            (species, transition, ...) the caller wants carried through to the result
        continuum (array): Continuum matching spectrum_x/spectrum_y, subtracted before fitting
        header (astropy.io.fits.Header): For the erg/s/cm**2 area conversion (same rule as fit_line)
        wave_unit (str): Wavelength unit, same role as header
        contam_lines (pd.DataFrame): Optional. Has a 'line_wave_um' column and, if present, a
            'species' column. This function fits these lines too, so their flux does not
            leak into a nearby requested line's amplitude
        sigma_multiplier (float): Each line's fit window half-width, in multiples of its own
            _line_sigma. See DEFAULT_SIGMA_MULTIPLIER's own comment for where the default comes from.

    Returns:
        result (tuple): (results, contam_results, None). The third element is an unused
            placeholder; this function does not correct for velocity shift, since line
            centers fit freely within fit_glines' own bounds. results and contam_results
            are lists of (line_dict, GaussianFitResult) pairs. results keeps hitran_lines' order.

    Raises:
        ValueError: if the noise of the spectrum cannot be estimated for the blend check
            (the estimate is missing, zero, negative, or not finite)
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

    def _local_max_amp(wl, half_window=2):
        """
        Estimate max_amplitude for a line the same simple way find_lines() does: local
        peak-to-trough over a small window around the catalog wavelength.

        Args:
            wl (float): Catalog (rest) wavelength to center the window on
            half_window (int): Samples on each side of the nearest pixel to include

        Returns:
            amplitude (float): Local peak-to-trough amplitude, or NaN if the window is empty
                or every sample in it is NaN (a real gap, not a genuinely zero-flux line)
        """
        idx_c = int(np.argmin(np.abs(spectrum_x - wl)))
        w0, w1 = max(0, idx_c - half_window), min(len(spectrum_x), idx_c + half_window + 1)
        local = y_corrected[w0:w1]
        if len(local) == 0 or not np.any(np.isfinite(local)):
            return np.nan
        return float(np.nanmax(local) - np.nanmin(local))

    def _no_signal_result(wl, reason):
        """
        A failed GaussianFitResult for a line that fit_glines never tried to fit. This
        function reports it the same honest way as a fit that ran and failed. It does not
        drop the line or give it a made-up amplitude.

        Args:
            wl (float): Catalog wavelength this line was requested at
            reason (str): Human-readable reason no fit was attempted

        Returns:
            result (GaussianFitResult): success=False, message=reason, all-zero fit fields
        """
        empty = np.array([])
        return GaussianFitResult(
            amplitude=0, center=wl, sigma=0, fwhm=0, area=0,
            baseline_slope=0.0, baseline_intercept=0.0,
            x_data=empty, y_data=empty, baseline=empty, y_corrected=empty,
            y_fit=empty, residuals=empty,
            r_squared=0, success=False, message=reason,
        )

    # No amplitude floor. A line with no real local signal gets one retry with a wider
    # window. If that fails too, this code excludes it and calls _no_signal_result.
    lines_data = []
    skipped = []
    for orig_idx, wl, line in valid:
        amp = _local_max_amp(wl)
        if not np.isfinite(amp) or amp <= 0:
            amp = _local_max_amp(wl, half_window=5)
        if not np.isfinite(amp) or amp <= 0:
            skipped.append((orig_idx, line, _no_signal_result(
                wl, f"No usable signal near {wl:.6f} µm (flat or missing data even in a "
                    f"widened window) -- line excluded rather than fit with a guessed amplitude.")))
            continue
        lines_data.append((orig_idx, wl, line, amp))

    if not lines_data:
        results = [None] * len(hitran_lines)
        for orig_idx, line, result in skipped:
            results[orig_idx] = (line, result)
        return [r for r in results if r is not None], [], None

    # Per-line sigma from the instrument's own resolution (_line_sigma), not one shared
    # guess. A single estimate over the whole spectrum cannot know each line's real width.
    detco_lines = pd.DataFrame({
        'line': [d[1] for d in lines_data],
        'disposition': [1] * len(lines_data),
        'max_amplitude': [d[3] for d in lines_data],
        'sigma': [_line_sigma(d[1], header) for d in lines_data],
    })

    if contam_lines is not None and len(contam_lines) > 0:
        # Uses the same rule as the requested lines above: exclude a line with no real
        # signal instead of faking it. A phantom contaminant would add a Gaussian with nothing to explain.
        contam_pairs = []
        for wl in contam_lines['line_wave_um'].astype(float).values:
            amp = _local_max_amp(wl)
            if not np.isfinite(amp) or amp <= 0:
                amp = _local_max_amp(wl, half_window=5)
            if np.isfinite(amp) and amp > 0:
                contam_pairs.append((wl, amp))
        detcontaminating_lines = pd.DataFrame({
            'line': [p[0] for p in contam_pairs],
            'disposition': [1] * len(contam_pairs),
            'max_amplitude': [p[1] for p in contam_pairs],
            'sigma': [_line_sigma(p[0], header) for p in contam_pairs],
        })
    else:
        detcontaminating_lines = pd.DataFrame(columns=['line', 'disposition', 'max_amplitude', 'sigma'])

    # Noise for the merge blend-check below. An unusable value stops the run
    # instead of being replaced by a different estimator.
    noise_std = compute_noise_std(spectrum_x, y_corrected, degree=1)
    if noise_std is None or not np.isfinite(noise_std) or noise_std <= 0:
        raise ValueError(
            "Cannot estimate this spectrum's noise for the blend check: "
            "the estimate is missing, zero, negative, or not finite."
        )

    all_centers = list(detco_lines['line']) + list(detcontaminating_lines['line'])
    all_sigmas = list(detco_lines['sigma']) + list(detcontaminating_lines['sigma'])
    wave_boundaries, wave_boundaries_extent = _merged_wave_boundaries(
        all_centers, all_sigmas, spectrum_x, y_corrected, noise_std, half_width_sigma=sigma_multiplier)

    # This is only fit_glines' scalar fallback, for the RuntimeError path. The per-line
    # 'sigma' column above drives the real fit.
    sigma = float(np.median(detco_lines['sigma']))

    # This code has no fallback string. MIRI, NIRSpec, and MUSE headers always carry BUNIT.
    # None here means the header is missing something real, and the debug prints must show it.
    _, _, _, _, _, per_line_fits, per_contam_fits = fit_glines(
        spectrum_x, y_corrected, wave_boundaries, detco_lines, detcontaminating_lines, sigma,
        progress_callback=progress_callback,
        flux_unit_label=header.get('BUNIT') if header is not None else None,
        wave_unit_label=get_wave_unit(header, wave_unit),
        list_wboundaries_extent=wave_boundaries_extent, half_width_sigma=sigma_multiplier,
    )

    # fit_glines groups and loops by chunk, not by input order. This code matches results
    # back by wavelength, so it keeps species/transition data and the caller's own order.
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
            # This code has no fallback string. A missing BUNIT here means the header is
            # incomplete, so "UNKNOWN" flags that instead of guessing a unit.
            flux_unit = (header.get('BUNIT') if header is not None else None) or 'UNKNOWN'
            area_unit = f"{flux_unit}·{unit.lower() if unit else 'UNKNOWN'}"

        # This is Pascucci's statistical area-error formula, plus her flux-calibration floor
        # in quadrature. It gives one number, used the same way for every consumer.
        amp_err, sig_err = fit.get('amplitude_error'), fit.get('sigma_error')
        if amp_err is not None and sig_err is not None and area_native:
            area_error_native = np.sqrt(2 * np.pi * (
                fit['sigma'] ** 2 * amp_err ** 2 + fit['amplitude'] ** 2 * sig_err ** 2
            ))
            area_error_statistical = area_error_native * (area / area_native)
            area_error = float(np.sqrt(
                area_error_statistical ** 2 + (SYSTEMATIC_FLUX_ERROR_FRACTION * area) ** 2
            ))
        else:
            area_error = None

        # R2 comes from this line's own local peak (the DISPLAY_CROP_SIGMA_MULTIPLIER window),
        # not the whole shared chunk. Otherwise every line in a chunk reports the same number.
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
            area_error=area_error,
            catalog_sigma=fit.get('catalog_sigma'), noise_std=fit.get('noise_std'),
            y_fit_own=fit.get('y_fit_own'),
        )

    results = [None] * len(hitran_lines)
    for orig_idx, wl, line, _max_amp in lines_data:
        fit = by_wavelength.get(round(wl, 9))
        if fit is not None:
            results[orig_idx] = (line, _to_result(fit))
    # These lines were excluded before fit_glines ran (see _no_signal_result above). This
    # code carries them through, so the caller sees every requested line, success or not.
    for orig_idx, line, result in skipped:
        results[orig_idx] = (line, result)

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

def extract_flux_from_spectrum(spectrum_wave, spectrum_flux, target_catalog, contam_catalog,
                                   wave_boundaries=None, wave_min=None, wave_max=None, do_duplicate_removal=True,
                                   excluded_wavelengths=None,
                                   continuum_params=None, snr_threshold=3.0, header=None, wave_unit=None):
    """
    Extract total flux from a single spectrum by detecting and fitting all target lines.

    Args:
        spectrum_wave (array or Quantity): Wavelength array for the spectrum
        spectrum_flux (array or Quantity): Flux array for the spectrum
        target_catalog (pd.DataFrame): Target line catalog, any species (columns: line_wave_um, species, ...)
        contam_catalog (pd.DataFrame): Contaminating lines catalog (column: line_wave_um)
        wave_boundaries (array): Fit-chunk boundaries. Auto-calculated from wave_min/wave_max if None
        wave_min (float): Minimum wavelength to fit. Defaults to min of target_catalog line_wave_um
        wave_max (float): Maximum wavelength to fit. Defaults to max of target_catalog line_wave_um
        do_duplicate_removal (bool): True to remove duplicate line detections
        excluded_wavelengths (set or list): Wavelengths to exclude from flux calculation
        continuum_params (dict): Overrides for fit_auto_continuum()'s defaults, which work
            well for typical spectra
        snr_threshold (float): Minimum signal-to-noise, in sigma, for a line to count as detected
        header (astropy.io.fits.Header): Required as soon as this function detects a line.
            Each line's fit width comes from the instrument's real spectral resolution
            (_line_sigma, MIRI/NIRSpec only). A missing header raises ValueError instead of
            guessing a width
        wave_unit (str): Wavelength unit, for example 'um', used with header for the CGS conversion

    Returns:
        result (dict): 'flux', 'contam_flux', 'mean_continuum', 'detected_lines', 'residuals',
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
        wave_min = target_catalog['line_wave_um'].min()
    if wave_max is None:
        wave_max = target_catalog['line_wave_um'].max()
    
    print(f"  Extracting flux in wavelength range: {wave_min:.6f} - {wave_max:.6f} um")
    print(f"  [DEBUG] Spectrum: {len(spectrum_wave)} wavelength points, flux range {np.nanmin(spectrum_flux):.4e} to {np.nanmax(spectrum_flux):.4e}")
    if spectrum_wave_unit:
        print(f"  Wavelength unit: {spectrum_wave_unit}")
    if spectrum_flux_unit:
        print(f"  Flux unit: {spectrum_flux_unit}")

    # Tests once, with a dummy value, whether CGS conversion is reachable here (for
    # example, ALMA's Jy/beam never is). This sets one label for flux/contam_flux below.
    flux_unit_label = spectrum_flux_unit or resolved_flux_unit or 'UNKNOWN'
    if _to_cgs(1.0, (wave_min + wave_max) / 2.0) is not None:
        flux_unit_label = 'erg/s/cm2'
    
    # If wave_boundaries is not given, this code builds it further down, after it
    # detects this pixel's own lines. See the comment there for why.

    # Narrow to the requested fit range
    fit_range_mask = (spectrum_wave >= wave_min) & (spectrum_wave <= wave_max)
    spectrum_wave = spectrum_wave[fit_range_mask]
    spectrum_flux = spectrum_flux[fit_range_mask]
    if continuum_model is not None:
        continuum_model = continuum_model[fit_range_mask]

    # Filter out excluded wavelengths from target catalog
    target_catalog_for_fit = target_catalog
    if excluded_wavelengths and len(excluded_wavelengths) > 0:
        excluded_set = set(excluded_wavelengths) if not isinstance(excluded_wavelengths, set) else excluded_wavelengths
        # Remove lines within 0.001 micron tolerance of excluded wavelengths
        mask = ~target_catalog['line_wave_um'].apply(
            lambda w: any(abs(w - ew) < 0.001 for ew in excluded_set)
        )
        target_catalog_for_fit = target_catalog[mask].copy()
        # IMPORTANT: Reset index so positions match indices (0, 1, 2, ... not original indices like 0, 5, 10, 220, ...)
        target_catalog_for_fit = target_catalog_for_fit.reset_index(drop=True)
        n_excluded = len(target_catalog) - len(target_catalog_for_fit)
        if n_excluded > 0:
            print(f"  Excluding {n_excluded} lines from extraction")

    print(f"  [DEBUG] Target catalog size: {len(target_catalog_for_fit)} lines")

    # Whole spectrum with lines in it: MAD, since the lines would inflate a plain std.
    shared_noise_std = compute_mad_noise_std(spectrum_flux)

    # Find detected lines (contam_catalog can be None if user didn't upload contaminating lines)
    detected_lines = find_lines(
        spectrum_wave, spectrum_flux, target_catalog_for_fit,
        contaminating_lines=contam_catalog,
        remove_duplicate=do_duplicate_removal,
        snr_threshold=snr_threshold,
        noise_std=shared_noise_std
    )
    
    if detected_lines is None or len(detected_lines) == 0:
        # No lines detected at all. This code still shows the early (pre-line-fit) continuum
        # if it fit one, so the plot has something to compare against the raw spectrum.
        no_fit_continuum = np.full_like(spectrum_flux, np.nan)
        if continuum_model is not None:
            no_fit_continuum = continuum_model
        return {
            'flux': 0,
            'contam_flux': 0,
            'mean_continuum': np.nanmedian(spectrum_flux),
            'detected_lines': pd.DataFrame(),
            'detected_contam': pd.DataFrame(),
            'residuals': pd.DataFrame({'wave': spectrum_wave, 'residual': np.zeros_like(spectrum_flux)}),
            'continuum_curve': pd.DataFrame({'wave': spectrum_wave, 'continuum': no_fit_continuum}),
            'flux_unit': flux_unit_label,
            'mean_detected_wavelength': np.nan
        }
    
    # No species filter here: target_catalog can hold any molecule, and every detected line
    # in it counts toward 'flux'. Only contam_catalog's lines count toward 'contam_flux'.
    try:
        detected_target = detected_lines[
            (detected_lines['disposition'] == 1)
        ].copy()
    except KeyError as e:
        print(f"  Error: detected_lines missing column - columns are: {detected_lines.columns.tolist()}")
        print(f"  detected_lines head:\n{detected_lines.head()}")
        raise

    # Per-line sigma from the instrument's own resolution (_line_sigma), the same as
    # fit_hitran_lines_collective. One shared sigma under-resolves a narrow line.
    detected_target['sigma'] = [_line_sigma(wl, header) for wl in detected_target['line']]

    # DEBUG: Show how many target lines were detected
    n_target_detected = len(detected_target)
    if n_target_detected > 0:
        print(f"  [DEBUG] Target line wavelengths: {detected_target['line'].values[:5].tolist()}...")
        print(f"  [DEBUG] Target line max amplitudes: {detected_target['max_amplitude'].values[:5].tolist()}...")

    # This runs peak detection against the contaminant catalog itself, with no avoidance
    # of target lines. A contaminant near a target line needs its own Gaussian, not to be dropped.
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
    print(f"  [DEBUG] Line detection: {n_target_detected} target lines detected, {n_contam_detected} contaminating lines")

    # Fit if any lines detected
    if len(detected_target) > 0 or len(detected_contam) > 0:
        # This is only fit_glines' required positional arg. Every detected line already
        # carries its own real 'sigma', so fit_glines never reads this scalar.
        all_sigmas = pd.concat([detected_target['sigma'], detected_contam['sigma']])
        sigma = float(np.median(all_sigmas))

        # Uses the same _merged_wave_boundaries() approach as fit_hitran_lines_collective,
        # built from this pixel's own detected lines. Each pixel needs its own set.
        wave_boundaries_extent = None
        if wave_boundaries is None:
            all_centers = list(detected_target['line']) + list(detected_contam['line'])
            wave_boundaries, wave_boundaries_extent = _merged_wave_boundaries(
                all_centers, list(all_sigmas), spectrum_wave, spectrum_flux, shared_noise_std)

        print(f"  [DEBUG] Before fit_glines: median sigma={sigma:.6f}, n_waves={len(wave_boundaries)} boundaries")
        print(f"  [DEBUG] Fitting {len(detected_target)} target + {len(detected_contam)} contaminating lines")
        # This code has no fallback string (see fit_glines' docstring). None here means the
        # header is missing something normally present in a MIRI, NIRSpec, or MUSE header.
        flux, contam_flux, mean_cont, residuals, continuum_curve, _, _ = fit_glines(
            spectrum_wave, spectrum_flux, wave_boundaries,
            detected_target, detected_contam, sigma,
            flux_unit_label=resolved_flux_unit, wave_unit_label=resolved_wave_unit,
            list_wboundaries_extent=wave_boundaries_extent,
        )
        _debug_area_unit = f"{resolved_flux_unit}·{resolved_wave_unit}"
        print(f"  [DEBUG] fit_glines returned (native, pre-CGS-conversion): "
              f"flux={flux:.6e} {_debug_area_unit}, contam_flux={contam_flux:.6e} {_debug_area_unit}, "
              f"mean_cont={mean_cont:.6e} {resolved_flux_unit}")
        print(f"  [DEBUG] flux type={type(flux)}, is_finite={np.isfinite(flux)}")
    else:
        flux, contam_flux, mean_cont = 0, 0, np.nanmedian(spectrum_flux)
        residuals = pd.DataFrame({'wave': spectrum_wave, 'residual': np.zeros_like(spectrum_flux)})
        continuum_curve = pd.DataFrame({'wave': spectrum_wave, 'continuum': np.full_like(spectrum_flux, np.nan)})
        detected_target = pd.DataFrame()
        # print(f"  No lines to fit, returning 0")
    
    # Calculate mean wavelength of detected target lines for flux conversion
    mean_detected_wavelength = np.nan
    if len(detected_target) > 0 and 'line' in detected_target.columns:
        detected_co_only = detected_target[detected_target['disposition'] == 1]
        if len(detected_co_only) > 0:
            mean_detected_wavelength = np.mean(detected_co_only['line'].values)

    # contam_flux needs its own reference wavelength (a pixel can have a contaminant
    # but no target lines). detected_contam is already disposition==1 only -- no extra filter needed.
    mean_detected_contam_wavelength = np.nan
    if len(detected_contam) > 0 and 'line' in detected_contam.columns:
        mean_detected_contam_wavelength = np.mean(detected_contam['line'].values)

    # conversion -- flux and contam_flux each convert off their own detections, not
    # gated on the other one having succeeded.
    flux_cgs = _to_cgs(flux, mean_detected_wavelength)
    if flux_cgs is not None:
        flux = flux_cgs
    contam_flux_cgs = _to_cgs(contam_flux, mean_detected_contam_wavelength)
    if contam_flux_cgs is not None:
        contam_flux = contam_flux_cgs

    # continuum_curve so far only holds fit_glines' own per-chunk baseline (a + b*wave), fit
    # against the ALREADY continuum-subtracted spectrum.
    if continuum_model is not None:
        continuum_curve = continuum_curve.copy()
        continuum_curve['continuum'] = continuum_model + continuum_curve['continuum'].fillna(0.0).values

    return {
        'flux': flux,
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

def create_flux_map(cube_data, target_catalog, contam_catalog, n_jobs=8, progress_callback=None, cancel_flag=None,
                       wave_min=None, wave_max=None, excluded_wavelengths=None,
                       continuum_params=None):
    """
    Create a 2D map of total flux by extracting each pixel's spectrum and
    calling extract_flux_from_spectrum() on it, in parallel across rows.

    Args:
        cube_data (CubeData): Loaded cube; needs cube, wavelength, get_pixel_spectrum(x, y)
        target_catalog (pd.DataFrame): Target line catalog, any species (columns: line_wave_um, species, ...)
        contam_catalog (pd.DataFrame): Contaminating lines catalog (column: line_wave_um)
        n_jobs (int): Number of parallel worker threads for processing rows
        progress_callback (callable): Optional progress_callback(current, total)
        cancel_flag (object): Optional object with a `.cancelled` attribute, checked per pixel
        wave_min (float): Minimum wavelength to fit. Defaults to min of target_catalog
        wave_max (float): Maximum wavelength to fit. Defaults to max of target_catalog
        excluded_wavelengths (set or list): Wavelengths to exclude from flux calculation
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
    
    # Calculate wavelength range from target catalog (not full spectrum) for efficiency
    if wave_min is None:
        wave_min = target_catalog['line_wave_um'].min()
    if wave_max is None:
        wave_max = target_catalog['line_wave_um'].max()

    # DIAGNOSTIC: Check line catalogs
    print(f"    Target catalog: {len(target_catalog)} lines")
    if len(target_catalog) > 0:
        print(f"      Wavelength range: {target_catalog['line_wave_um'].min():.4f} - {target_catalog['line_wave_um'].max():.4f}")
    if contam_catalog is not None:
        print(f"    Contaminating catalog: {len(contam_catalog)} lines")
        if len(contam_catalog) > 0 and 'line_wave_um' in contam_catalog.columns:
            print(f"      Wavelength range: {contam_catalog['line_wave_um'].min():.4f} - {contam_catalog['line_wave_um'].max():.4f}")
    else:
        print(f"    Contaminating catalog: None (user did not upload contaminating lines)")

    # DIAGNOSTIC: Check if line wavelengths overlap cube data wavelengths
    target_overlap = (target_catalog['line_wave_um'] >= wave_min) & (target_catalog['line_wave_um'] <= wave_max)
    print(f"    Target lines in data wavelength range: {np.sum(target_overlap)}/{len(target_catalog)}")
    
    sys.stdout.flush()
    
    # Create output maps
    flux_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    contam_flux_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    detection_map = np.full((n_y, n_x), 0, dtype=np.int32)
    contam_detection_map = np.full((n_y, n_x), 0, dtype=np.int32)
    wavelength_map = np.full((n_y, n_x), np.nan, dtype=np.float32)  # Track mean detected wavelength per pixel
    
    print(f">>> Creating flux map: {n_y}x{n_x} pixels, {n_wave} wavelength elements")
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
    
    # This code does not compute wave_boundaries here. Each pixel detects a different set
    # of lines, so extract_flux_from_spectrum builds its own boundaries per call.

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
            test_result = extract_flux_from_spectrum(
                wavelength, test_spectrum, target_catalog, contam_catalog,
                wave_min=wave_min, wave_max=wave_max,
                do_duplicate_removal=True,
                continuum_params=continuum_params,
                header=cube_data.header, wave_unit=cube_data.wave_unit
            )
            if test_result:
                print(f"      Result: flux={test_result['flux']:.4e}, detected_lines shape={len(test_result['detected_lines'])}")
                if len(test_result['detected_lines']) > 0:
                    n_target_detected = len(test_result['detected_lines'][test_result['detected_lines']['disposition'] == 1])
                    print(f"      Target lines with disposition=1: {n_target_detected}")
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
                
                # Step 2: Extract flux [FUNCTION: extract_flux_from_spectrum]
                result = extract_flux_from_spectrum(
                    wavelength, spectrum_flux, target_catalog, contam_catalog,
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
                    flux_val = result['flux']

                    # Step 3: Store results in maps
                    row_fluxes[x_idx] = flux_val
                    row_contam_fluxes[x_idx] = result['contam_flux']
                    row_wavelengths[x_idx] = result.get('mean_detected_wavelength', np.nan)  # Store wavelength for conversion
                    
                    # Count detected target lines
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
        print(f"[ERROR] Error during flux extraction: {e}")
        traceback.print_exc()
    
    end_time = time.time()
    print(f"\n>>> Flux map creation complete ({end_time - start_time:.1f}s)")
    print(f"    Valid pixels: {np.sum(~np.isnan(flux_map))}/{n_y * n_x}")
    print(f"    Mean flux: {np.nanmean(flux_map):.4e}")
    print(f"    Max flux: {np.nanmax(flux_map):.4e}")
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


def save_flux_map_fits(flux_map_data, output_path, map_type='flux'):
    """
    Save a flux map (from create_flux_map) as a 2-HDU FITS file: PRIMARY (flux map)
    and DETECTION (line counts per pixel).

    Args:
        flux_map_data (dict): Output of create_flux_map(). Must have keys 'flux_map',
            'detection_map', 'wavelength', 'header', 'wave_range'
        output_path (str): Output file path, for example 'flux_map.fits'
        map_type (str): Map type label, for example 'flux' or 'contamination_flux'

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
                # Use the original header. Use 'UNKNOWN' if BUNIT is not there, not a guess.
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
        out_header['COMMENT'] = f'Rovibrational {map_type} Map'
        out_header['COMMENT'] = 'Flux extracted by simultaneous multi-Gaussian fitting'
        out_header['COMMENT'] = 'Values are amplitude * sigma * sqrt(2*pi) integrated from all detected target lines'
        
        # Create primary HDU with flux map
        primary_hdu = fits.PrimaryHDU(flux_map.astype(np.float32), header=out_header)
        
        # Create detection map extension
        detection_hdu = fits.ImageHDU(detection_map.astype(np.int32), name='DETECTION')
        detection_hdu.header['BUNIT'] = 'count'
        detection_hdu.header['COMMENT'] = 'Number of target lines detected and fit in each pixel'
        
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
