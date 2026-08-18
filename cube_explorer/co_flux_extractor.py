"""
CO Rovibrational Flux Extraction Module

Fits multiple CO lines and contaminating lines simultaneously from IFU spectra,
extracting and summing the total CO flux to create integrated flux maps.

Uses scipy.optimize.curve_fit for simultaneous multi-gaussian + continuum fitting.

Functions Reused from gaussian_fitter.py:
- compute_noise_std()           : Robust noise/std estimation using RANSAC polynomial fit

This architecture ensures consistency across the codebase and avoids code duplication.
"""

import numpy as np
import pandas as pd
import time
import warnings
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
    GaussianFitResult, _extract_value_from_quantity,
)
from .utils import get_spectral_resolution
from .unit_converter import (
    integrated_flux_to_cgs, get_wave_unit, _resolve_flux_unit, convert_area_to_cgs,
    extract_units,
)

# ================== Unit Utilities ==================

def fit_auto_continuum(wavelength, flux, window_size=51, sigma_threshold=2.0,
                                peak_width=1, peak_distance=3, peak_exclusion=3,
                                iterations=2, smoothing_factor=5):
    """
    Fit continuum using peak detection and median filter method.
    
    This is the same method used by the GUI's Automatic Continuum button.
    It finds emission peaks, excludes them, and fits a smooth continuum through
    the remaining points using median filtering and spline smoothing.
    
    Parameters
    ----------
    wavelength : array-like
        Wavelength array
    flux : array-like
        Flux array
    window_size : int
        Size of median filter window (must be odd), default 51
    sigma_threshold : float
        Prominence threshold for peak detection in units of sigma, default 2.0
    peak_width : int
        Minimum width of peaks in pixels, default 1
    peak_distance : int  
        Minimum distance between peaks in pixels, default 3
    peak_exclusion : int
        Number of pixels to exclude on each side of peak, default 3
    iterations : int
        Number of iterative sigma clipping iterations, default 2
    smoothing_factor : float
        Spline smoothing parameter divisor, default 5
        
    Returns
    -------
    continuum : array-like
        Fitted continuum values matching input wavelength array
        
    Notes
    -----
    Returns array of NaN if fitting fails or insufficient data
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

def gaussian(x, amp, cen, sig, cont_base=None, cont_slope=None):
    """
    Single Gaussian component with optional linear continuum.
    Define a Gaussian function
    
    Parameters
    ----------
    x : array-like
        Wavelength values
    amp : float
        Gaussian amplitude
    cen : float
        Gaussian center (line center wavelength)
    sig : float
        Gaussian standard deviation (sigma)
    cont_base : float, optional
        Continuum baseline level
    cont_slope : float, optional
        Continuum slope
    
    Returns
    -------
    array-like
        Gaussian model values
    """
    if (cont_base is None) & (cont_slope is None):
        gauss_return = amp * np.exp(-0.5 * ((x - cen) / sig) ** 2)
    elif (cont_base is not None) & (cont_slope is None):
        gauss_return = amp * np.exp(-0.5 * ((x - cen) / sig) ** 2) + cont_base
    elif (cont_base is not None) & (cont_slope is not None):
        gauss_return = amp * np.exp(-0.5 * ((x - cen) / sig) ** 2) + cont_base + cont_slope * x
    return gauss_return


def many_gaussians_list(xl, a, b, *params):
    """
    Sum of multiple Gaussians with linear continuum.
    Calculates the y-values for a list of x-values for a sum of Gaussian curves with a continuum slope.

    Vectorized over both points and lines at once (xl[:, None] against each Gaussian's own
    params, broadcasting to a (n_points, n_lines) array before summing over lines) -- not a
    per-point, per-line Python loop. That loop was the actual bottleneck of the whole fit:
    curve_fit's numerical Jacobian re-evaluates this model ~2x per free parameter on every
    iteration, so a Python-level loop here dominated total runtime (profiled at 81% of one
    chunk's fit time) and got dramatically worse as more lines shared a chunk. Same math,
    same result for every input -- this is purely how it's computed, not what it computes.

    Parameters
    ----------
    xl : array-like
        Wavelength values to evaluate at
    a : float
        Continuum baseline (a in a + b*x)
    b : float
        Continuum slope (b in a + b*x)
    *params : float
        Variable number of Gaussian parameters organized as:
        [amp0, cen0, sig0, amp1, cen1, sig1, ..., ampN, cenN, sigN]
        Each Gaussian uses 3 parameters: amplitude, center, sigma

    Returns
    -------
    np.ndarray
        Sum of all Gaussians plus continuum evaluated at xl
    """
    N = 3  # Parameters per Gaussian
    xl = np.asarray(xl)
    params = np.asarray(params)
    min_freq = np.min(xl)
    amps, cens, sigs = params[0::N], params[1::N], params[2::N]
    gaussians = amps[np.newaxis, :] * np.exp(
        -0.5 * ((xl[:, np.newaxis] - cens[np.newaxis, :]) / sigs[np.newaxis, :]) ** 2
    )
    return a + b * (xl - min_freq) + gaussians.sum(axis=1)


# ================== Line Detection ==================

def find_lines(xwave, yflux, lines_to_search, contaminating_lines=None, remove_duplicate='False', snr_threshold=3.0, noise_std=None):
    """
    Identify which spectral lines are detected in a spectrum and assign a starting max_amplitude.

    Performs peak detection on a fine wavelength grid around each line and compares
    to local noise. Handles duplicate/close lines and contaminating line avoidance.

    Parameters
    ----------
    xwave : array-like
        Wavelength array
    yflux : array-like
        Flux array
    lines_to_search : pd.DataFrame
        DataFrame with columns: 'line_wave_um', 'species', others optional
    contaminating_lines : pd.DataFrame, optional
        DataFrame with column 'line_wave_um' for lines to avoid
    remove_duplicate : str
        'True' to remove duplicate detections (keep strongest line in close groups)
    snr_threshold : float
        Signal-to-noise threshold for line detection (default: 3.0 sigma)
    noise_std : float, optional
        Precomputed noise estimate for (xwave, yflux), to skip recomputing it below. Only
        worth passing when a caller is about to call find_lines() more than once on the
        *same* (xwave, yflux) -- e.g. extract_co_flux_from_spectrum's separate CO/contaminant
        detection passes, which would otherwise redundantly recompute an identical value
        (now that compute_noise_std() is seeded, the two calls' own computations would agree
        anyway -- this just skips doing the same RANSAC fit twice). If None, computed fresh.

    Returns
    -------
    pd.DataFrame
        Lines with columns: line, species, disposition (0/1), max_amplitude, line_flux
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
    if remove_duplicate == 'True':
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

def cont_fit_ransac(x, y, pdeg, show='False', return_params=False, exclude_wavelength_ranges=None):
    """
    Fit continuum using RANSAC polynomial regression (from Pascucci).
    
    Robust to outliers, useful when fitting continuum through noisy data.
    
    Parameters
    ----------
    x : array-like
        Wavelength values
    y : array-like
        Flux values
    pdeg : int
        Polynomial degree (1=linear, 2=quadratic, etc.)
    show : str
        'True' to plot result, 'False' otherwise
    return_params : bool
        If True, return polynomial coefficients; if False, return continuum array
    exclude_wavelength_ranges : list of tuples, optional
        Wavelength ranges to exclude from fitting
    
    Returns
    -------
    continuum : array-like
        Fitted continuum values (or coefficients if return_params=True)
    std_residuals : float
        Standard deviation of residuals
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
    
    if show == 'True':
        import matplotlib.pyplot as plt
        plt.plot(x, y, color='steelblue', drawstyle='steps')
        plt.plot(x, yfit_all, color='orange')
        plt.ylim(np.nanmin(y), np.nanmax(y))
        plt.xlim(np.nanmin(x), np.nanmax(x))
        plt.xlabel('Wavelength [micron]')
        plt.show()
    
    if return_params:
        best_fit_params = ransac_pipeline.named_steps['ransacregressor'].estimator_.coef_
        return best_fit_params, std_residuals
    else:
        return yfit_all, std_residuals


# ================== Multi-Gaussian Fitting ====================

def fit_glines(xwave, yflux, list_wboundaries, detco_lines, detcontaminating_lines, sigma,
                progress_callback=None, flux_unit_label=None, wave_unit_label=None, header=None):
    """
    Fit simultaneously many Gaussians and the underlying continuum in given
    chunks of wavelengths (given in list_wboundaries).

    progress_callback, if given, is called as progress_callback(current_chunk, total_chunks)
    after each chunk finishes (converged or not) -- e.g. to drive a GUI progress bar.

    flux_unit_label, wave_unit_label: purely cosmetic, for the debug prints below -- every
    flux/amplitude number here is in `xwave`/`yflux`'s own native units (this function never
    does a CGS conversion for its *returned* values; that stays the caller's job), so without
    a label every print is ambiguous about what unit it's actually in. No placeholder default
    on purpose: this app only supports MIRI/NIRSpec/MUSE, whose headers always carry BUNIT and
    a wavelength unit, so if a caller has None here that means the header is actually missing
    something it shouldn't be -- the prints below should show that plainly (literally "None"),
    not paper over it with a plausible-looking fake unit.

    header: optional, purely for an approximate erg/s/cm² preview alongside the native value in
    the debug prints below (via integrated_flux_to_cgs, evaluated at each chunk's own center
    wavelength) -- a sanity-check convenience, not a second source of truth; the actual returned
    sum_CO_area/sum_contaminating_area stay in native units regardless.

    Returns:
        1. the sum of the areas of the CO Gaussians (sum_CO_area)
        2. the sum of the areas of the contaminating lines Gaussians (sum_contaminating_area)
        3. the mean of the continuum (mean_cont)
        4. a pandas dataframe with the residual which is good to check the results (df_residual)
        5. a pandas dataframe with the fitted continuum (a + b*wave) per wavelength, NaN outside
           any fitted chunk (df_continuum) -- useful for plotting the actual continuum trend
           rather than just its mean
        6. a list of per-CO-line fit dicts (per_line_fits) -- the individual (amplitude, center,
           sigma, area) each CO line's own Gaussian component converged to within its chunk, plus
           a local x_data/y_data/y_fit slice for plotting. Callers that only need the aggregate
           flux (e.g. extract_co_flux_from_spectrum) can ignore this; it exists for callers that
           need per-line detail (e.g. fit_hitran_lines_collective) without re-deriving it from `par`.
        7. the same, for contaminating lines (per_contam_fits)

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

    print('===== Start fitting...')
    start_time = time.time()

    for ii in range(len(list_wboundaries) - 1):
        xwave_min, xwave_max = list_wboundaries[ii], list_wboundaries[ii + 1]
        # Excludes NaN/inf points up front (real IFU cubes commonly have partially-covered
        # spaxels -- dither edges, bad-pixel masking -- inside an otherwise valid chunk) so a
        # handful of bad samples don't throw out this chunk's otherwise-good data. Every use of
        # idx_xwave below (p0/bounds, curve_fit's own inputs, df_residual/df_continuum) then
        # only ever sees finite data -- not a fallback, since nothing is guessed, just data
        # that was never NaN in the first place.
        idx_xwave = (xwave >= xwave_min) & (xwave <= xwave_max) & np.isfinite(xwave) & np.isfinite(yflux)
        print('> Fitting wave range ', np.around(xwave_min, decimals=4), np.around(xwave_max, decimals=4))

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

        # Append contaminating lines at the end of the CO lines
        filtered_lines = pd.concat([filtered_colines, filtered_contaminatinglines], ignore_index=True)

        if len(filtered_lines) == 0:
            print('  No lines to fit in this range')
            continue

        # curve_fit needs at least as many finite data points as free parameters (2 for the
        # continuum + 3 per Gaussian) to be well-determined at all -- fewer than that and it's
        # either an outright error or, worse, an "answer" chosen arbitrarily among infinitely
        # many that fit the few points equally well. A pixel that fails this is honestly
        # unfittable in this chunk, not silently guessable, so it's marked failed like any
        # other unfittable chunk rather than handed to curve_fit.
        n_finite = np.count_nonzero(idx_xwave)
        n_params_needed = 2 + 3 * len(filtered_lines)
        if n_finite <= n_params_needed:
            print(f'  [DATA WARN] Only {n_finite} finite points for {n_params_needed} parameters '
                  f'in this chunk -- skipping (not enough data to fit)')
            for _, coline in filtered_colines.iterrows():
                per_line_fits.append({
                    'wavelength': coline['line'], 'amplitude': 0.0, 'center': coline['line'],
                    'sigma': 0.0, 'area': 0.0, 'amplitude_error': None,
                    'catalog_sigma': coline['sigma'] if 'sigma' in coline else None, 'noise_std': None,
                    'x_data': xwave[idx_xwave], 'y_data': yflux[idx_xwave],
                    'y_fit': np.zeros(n_finite),
                    'success': False,
                    'message': f"Not enough finite data points ({n_finite}) for {n_params_needed} parameters",
                })
            for _, contam_line in filtered_contaminatinglines.iterrows():
                per_contam_fits.append({
                    'wavelength': contam_line['line'], 'amplitude': 0.0, 'center': contam_line['line'],
                    'sigma': 0.0, 'area': 0.0, 'amplitude_error': None,
                    'catalog_sigma': contam_line['sigma'] if 'sigma' in contam_line else None, 'noise_std': None,
                    'x_data': xwave[idx_xwave], 'y_data': yflux[idx_xwave],
                    'y_fit': np.zeros(n_finite),
                    'success': False,
                    'message': f"Not enough finite data points ({n_finite}) for {n_params_needed} parameters",
                })
            continue

        # Per-line sigma when the caller supplied one (e.g. fit_hitran_lines_collective,
        # using _line_sigma's real spectral-resolution width per line) -- otherwise broadcast
        # the single shared `sigma` to every line, exactly as before (CO Flux Map's own
        # detected_co/detected_contam never carry a 'sigma' column, so this is a no-op change
        # for it). A single shared sigma across a whole multi-line chunk is only right when
        # every line in it genuinely has the same width; forcing that assumption is what
        # pinned every line's fitted sigma to the same lower bound and produced the smoothed,
        # under-resolved fits seen without a per-line width.
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
        # NaN-safe and scoped to this fit window (idx_xwave), matching p0slope just below --
        # np.nanmin(yflux) alone (the whole spectrum) can sit far from this chunk's actual
        # local flux level, producing continuum bounds that don't bracket the chunk's real
        # data even though the Gaussian bounds are correctly chunk-scoped -- that mismatch is
        # exactly the kind of inconsistent bounds box that makes curve_fit's initial guess
        # infeasible.
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
            # many_gaussians_list's continuum term is a + b*(x - min_freq), offset from the
            # window's own minimum wavelength -- not a + b*x -- so this has to match exactly or
            # the recorded curve is off by b*min_freq (huge, since b is O(1e3) and min_freq is
            # O(4.7)). Doesn't affect the flux number itself (computed from `par` directly),
            # only what gets plotted for inspection.
            df_continuum.loc[idx_xwave, 'continuum'] = a + b * (xwave[idx_xwave] - xwave[idx_xwave].min())

            # Noise level for this chunk's own fit, from its own residuals with the real dof
            # of the actual simultaneous model (every Gaussian + the 2 continuum params) --
            # the same quantity curve_fit itself uses internally to scale `cov` when no sigma=
            # is given, so this stays consistent with the per-line amplitude errors below
            # rather than being a second, differently-derived noise estimate.
            n_data_chunk = np.count_nonzero(idx_xwave)
            dof_chunk = n_data_chunk - (len(par) + 2)
            if dof_chunk > 0:
                noise_std_chunk = float(np.sqrt(np.sum((yflux[idx_xwave] - bfit) ** 2) / dof_chunk))
            else:
                print(f'  [NOISE WARN] Chunk {ii}: dof={dof_chunk} <= 0, cannot estimate noise from residuals')
                noise_std_chunk = None

            # Different approach in case there are contaminating lines
            if ncont_lines > 0:
                current_sum_CO = sum(par[jj] * par[jj + 2] * np.sqrt(2 * np.pi)
                                    for jj in range(0, len(par) - 3 * ncont_lines, 3))
                current_sum_contaminating = sum(par[jj] * par[jj + 2] * np.sqrt(2 * np.pi)
                                               for jj in range(len(par) - 3 * ncont_lines, len(par), 3))
            else:
                current_sum_contaminating = 0  # there are no contaminating lines in this wave range!
                current_sum_CO = sum(par[jj] * par[jj + 2] * np.sqrt(2 * np.pi)
                                    for jj in range(0, len(par), 3))

            area_unit_label = f'{flux_unit_label}·{wave_unit_label}'
            print(f'  [FLUX DEBUG] Range {ii}: {len(par)//3} fitted lines, '
                  f'CO flux segment={current_sum_CO:.6e} {area_unit_label}, contam={current_sum_contaminating:.6e} {area_unit_label}')
            if current_sum_CO < 0:
                print(f'  [FLUX WARN] Negative CO flux detected! intensities ({flux_unit_label}): '
                      f'{[par[jj] for jj in range(0, len(par) - 3 * ncont_lines, 3)]}')

            # Per-line records for callers that need individual (amplitude, center, sigma)
            # rather than just the chunk's summed flux (e.g. fit_hitran_lines_collective).
            # filtered_lines was built as CO lines first, then contaminating lines, and `par`
            # follows that same order in 3-parameter groups -- filtered_colines.iloc[i] lines
            # up with par[3*i:3*i+3], and filtered_contaminatinglines.iloc[j] with
            # par[3*n_co+3*j : 3*n_co+3*j+3].
            min_freq_chunk = np.nanmin(xwave[idx_xwave])
            n_co = len(filtered_colines)

            def _local_fit_record(cen_i, amp_i, sig_i, cat_wave, amp_err_i, catalog_sigma_i):
                local_mask = (
                    (xwave >= max(xwave_min, cen_i - 3 * abs(sig_i))) &
                    (xwave <= min(xwave_max, cen_i + 3 * abs(sig_i)))
                )
                x_local = xwave[local_mask]
                y_local = yflux[local_mask]
                # Full chunk model (local continuum correction + every Gaussian in the chunk,
                # not just this line's own) -- honestly reflects what explains the data here,
                # including any overlap with a neighboring line sharing this chunk.
                y_fit_local = a + b * (x_local - min_freq_chunk)
                for jj in range(0, len(par), 3):
                    y_fit_local = y_fit_local + gaussian(x_local, par[jj], par[jj + 1], par[jj + 2])
                return {
                    'wavelength': cat_wave, 'amplitude': amp_i, 'center': cen_i, 'sigma': sig_i,
                    'area': amp_i * sig_i * np.sqrt(2 * np.pi),
                    # amp_err_i: this line's own amplitude uncertainty straight from curve_fit's
                    # joint covariance matrix (cov), not a separately re-fit local model -- so a
                    # line degenerate with a close neighbor in this same chunk correctly gets a
                    # larger error instead of one derived as if it were fit alone.
                    'amplitude_error': amp_err_i,
                    # catalog_sigma_i: the fixed instrumental sigma (_line_sigma) this line was
                    # given as input, not the fitted `sig_i` -- used to build the artificial
                    # upper-limit Gaussian (3-sigma-noise amplitude, catalog center, this sigma)
                    # when the caller decides this line's real fit isn't significant.
                    'catalog_sigma': catalog_sigma_i,
                    'noise_std': noise_std_chunk,
                    'x_data': x_local, 'y_data': y_local, 'y_fit': y_fit_local,
                    'success': True, 'message': 'Success',
                }

            for i, (_, coline) in enumerate(filtered_colines.iterrows()):
                amp_err_i = float(np.sqrt(np.abs(cov[2 + 3 * i, 2 + 3 * i])))
                per_line_fits.append(_local_fit_record(
                    par[3 * i + 1], par[3 * i], par[3 * i + 2], coline['line'], amp_err_i, line_sigmas[i]
                ))
            for j, (_, contam_line) in enumerate(filtered_contaminatinglines.iterrows()):
                jj = 3 * (n_co + j)
                amp_err_j = float(np.sqrt(np.abs(cov[2 + jj, 2 + jj])))
                per_contam_fits.append(_local_fit_record(
                    par[jj + 1], par[jj], par[jj + 2], contam_line['line'], amp_err_j, line_sigmas[n_co + j]
                ))

        except RuntimeError as e:
            print("curve_fit did not converge! Fit a straight line and assign 3sigma upper limit", e)
            continuum, std_residuals = cont_fit_ransac(xwave[idx_xwave], yflux[idx_xwave], pdeg=1, show='False')
            # median(line_sigmas), not the scalar sigma -- identical to it when line_sigmas is
            # just a broadcast of the scalar (CO Flux Map's case), but reflects the real
            # per-line widths when they vary (fit_hitran_lines_collective's case).
            fallback_sigma = np.median(line_sigmas)
            current_sum_contaminating = 3 * np.median(std_residuals) * fallback_sigma * np.sqrt(2 * np.pi)
            current_sum_CO = 3 * np.median(std_residuals) * fallback_sigma * np.sqrt(2 * np.pi)
            a = np.nanmedian(continuum)
            df_continuum.loc[idx_xwave, 'continuum'] = continuum
            for i, (_, coline) in enumerate(filtered_colines.iterrows()):
                per_line_fits.append({
                    'wavelength': coline['line'], 'amplitude': 0.0, 'center': coline['line'],
                    'sigma': 0.0, 'area': 0.0, 'amplitude_error': None,
                    'catalog_sigma': line_sigmas[i], 'noise_std': None,
                    'x_data': xwave[idx_xwave], 'y_data': yflux[idx_xwave],
                    'y_fit': np.zeros(np.count_nonzero(idx_xwave)),
                    'success': False, 'message': f"Chunk fit failed: {e}",
                })
            for j, (_, contam_line) in enumerate(filtered_contaminatinglines.iterrows()):
                per_contam_fits.append({
                    'wavelength': contam_line['line'], 'amplitude': 0.0, 'center': contam_line['line'],
                    'sigma': 0.0, 'area': 0.0, 'amplitude_error': None,
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
    Instrumental Gaussian sigma at wavelength `wl`, from the real spectral resolution
    (R = lambda/delta_lambda) of the instrument in `header`.

    No fallback on purpose: this app only fits CO/H2 lines for MIRI and NIRSpec (MUSE is
    optical, far from these wavelengths, and never reaches this function with a real line to
    size), so a header that can't produce a real R here is missing something a MIRI/NIRSpec
    header shouldn't be missing -- that should fail loudly right here, not get silently
    papered over with a generic pixel-count guess that then quietly corrupts the fit itself.

    Raises:
        ValueError: if `header` is None, the instrument isn't MIRI/NIRSpec, or (NIRSpec)
            the grating is missing/unrecognized.
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


def _estimate_velocity_shift(x_data, y_data, candidate_wavelengths, search_samples=3, min_lines=3):
    """Return median of velocity between HITRAN center and wavelength peak (shifted)"""
    c = 299792.458  # km/s
    if len(candidate_wavelengths) < min_lines:
        return None

    velocities = []
    for wl in candidate_wavelengths:
        idx = np.argmin(np.abs(x_data - wl))
        lo, hi = max(0, idx - search_samples), min(len(x_data), idx + search_samples + 1)
        local_x, local_y = x_data[lo:hi], y_data[lo:hi]
        if len(local_y) == 0:
            continue
        peak_wl = local_x[np.argmax(local_y)]
        velocities.append(c * (peak_wl - wl) / wl)

    if len(velocities) < min_lines:
        return None
    return float(np.median(velocities))


def _merged_wave_boundaries(centers, sigmas, half_width_sigma=7.0):
    """
    Build fit_glines' list_wboundaries from each line's own instrumental width, not a
    fixed chunk grid: give every line its own [center - N*sigma, center + N*sigma]
    window, then merge any windows that overlap into one shared chunk.

    A line far enough from anything else that its window never overlaps another's gets
    its own small, tightly-scoped chunk -- no unrelated spectral structure (a broad
    nearby feature, a different line entirely) gets pulled into its continuum estimate
    just because it happened to fall inside an arbitrary fixed-width window. Lines close
    enough to genuinely blend (including a contaminant near a requested line) still end
    up sharing one chunk, same as before -- just because their windows actually overlap,
    not because they landed in the same slice of a one-size-fits-all grid.

    Returns a flat, sorted array of boundary points (consecutive pairs = one chunk, the
    same format fit_glines already expects). A gap between two merged, non-overlapping
    intervals becomes its own chunk with no lines in it -- a harmless no-op in fit_glines
    ("No lines to fit in this range"), not an error.
    """
    intervals = sorted(
        (c - half_width_sigma * s, c + half_width_sigma * s)
        for c, s in zip(centers, sigmas)
    )
    merged = [list(intervals[0])]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return np.array([b for lo, hi in merged for b in (lo, hi)])


def fit_hitran_lines_collective(spectrum_x, spectrum_y, hitran_lines, continuum,
                                 header=None, wave_unit=None, contam_lines=None,
                                 sigma_multiplier=7.0, progress_callback=None):
    """
    Fit a hand-picked list of HITRAN lines (e.g. from the species dialog) the same way
    CO Flux Map fits its detected lines: simultaneously, in wavelength chunks sized to
    each line's own instrumental width, via fit_glines(). Lines that share a chunk are
    fit together with one local continuum, so close/blended lines no longer get
    independent, uninformed fits that double-claim overlapping flux, and any contaminant
    sharing a chunk gets its own Gaussian too, so it stops leaking into a nearby
    requested line's amplitude.

    Unlike extract_co_flux_from_spectrum, this does no SNR-based line detection -- the
    caller has already chosen exactly which lines (and, if given, contaminants) to fit.

    Args:
        spectrum_x, spectrum_y: full spectrum arrays (numpy or Quantity)
        hitran_lines: list of dicts with 'lambda_micron' or 'wavelength', plus whatever
            metadata (species, transition, ...) the caller wants carried through
        continuum: continuum array matching spectrum_x/spectrum_y, subtracted before fitting
        header, wave_unit: for the erg/s/cm^2 area conversion (same rule as fit_line)
        contam_lines: optional pd.DataFrame with a 'line_wave_um' column (and optionally
            'species'); non-requested lines fit alongside the requested ones so their flux is
            absorbed by their own amplitude instead of a nearby requested line's.
        sigma_multiplier: each line's own fit window is +/- this many multiples of its own
            _line_sigma (default 7, derived directly from fit_glines' own curve_fit bounds --
            unchanged from Pascucci's original fit_Glines -- not a separate guess: the center
            is bounded to +/-1 sigma of the catalog wavelength, and the fitted sigma itself is
            bounded to [0.5, 2]x the instrumental value, so the widest Gaussian the fit could
            possibly converge to has ~99.7% of its area within 3 * 2sigma = 6sigma of its own
            center; adding the 1sigma max center shift gives 7sigma from the catalog
            wavelength as the point beyond which the bounds can never place real flux).
            See _merged_wave_boundaries().

    Returns:
        (results, contam_results, None) tuple -- the third element is a placeholder (no
        velocity-shift correction in this version; centers are free within fit_glines' own
        bounds, so a modest systemic shift is absorbed by each line's own fit already):
            results: list of (line_dict, GaussianFitResult) pairs, one per requested line
                with a wavelength, in the same order as `hitran_lines`.
            contam_results: list of (contam_line_dict, GaussianFitResult) pairs, one per
                contaminating line that fell in range and converged.
    """
    spectrum_x = _extract_value_from_quantity(spectrum_x)
    spectrum_y = _extract_value_from_quantity(spectrum_y)
    continuum = _extract_value_from_quantity(continuum)

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

    # max_amplitude per line, estimated the same simple way find_lines() does: local
    # peak-to-trough over a small window around the catalog wavelength.
    def _local_max_amp(wl):
        idx_c = int(np.argmin(np.abs(spectrum_x - wl)))
        w0, w1 = max(0, idx_c - 2), min(len(spectrum_x), idx_c + 3)
        local = y_corrected[w0:w1]
        return float(np.nanmax(local) - np.nanmin(local)) if len(local) else 0.0

    lines_data = [(orig_idx, wl, line, max(_local_max_amp(wl), 1e-6)) for orig_idx, wl, line in valid]

    # Per-line sigma from the instrument's own spectral resolution (_line_sigma), not one
    # shared guess for the whole multi-line fit -- a single estimate_gaussian_params() call
    # over the entire spectrum has no way to know each line's real width, and on a wide
    # range (e.g. the full CO band) the true width can vary 2x line to line, so forcing one
    # shared value pins every line to the same bound and under-resolves the narrow ones.
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
    wave_boundaries = _merged_wave_boundaries(all_centers, all_sigmas, half_width_sigma=sigma_multiplier)

    # Only used as fit_glines' required scalar fallback (RuntimeError path when a whole
    # chunk fails to converge) -- the per-line 'sigma' column above is what actually drives
    # the real fit now.
    sigma = float(np.median(detco_lines['sigma']))

    # No fallback string on purpose -- MIRI/NIRSpec/MUSE headers always carry BUNIT, so if
    # this is None the header is genuinely missing something, and the debug prints below
    # should show that plainly instead of hiding it behind a placeholder.
    _, _, _, _, _, per_line_fits, per_contam_fits = fit_glines(
        spectrum_x, y_corrected, wave_boundaries, detco_lines, detcontaminating_lines, sigma,
        progress_callback=progress_callback,
        flux_unit_label=header.get('BUNIT') if header is not None else None,
        wave_unit_label=get_wave_unit(header, wave_unit),
    )

    # fit_glines groups/iterates by chunk, not input order -- match records back to their
    # originating line dict by wavelength so species/transition metadata is preserved and
    # results come back in the caller's original order.
    by_wavelength = {round(f['wavelength'], 9): f for f in per_line_fits}
    unit = get_wave_unit(header, wave_unit)

    def _to_result(fit):
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
            # No fallback string on purpose -- MIRI/NIRSpec/MUSE headers always carry BUNIT,
            # so a missing one here means the header is genuinely incomplete for a supported
            # instrument. "UNKNOWN" is a visible flag of that, not a plausible-looking guess.
            flux_unit = (header.get('BUNIT') if header is not None else None) or 'UNKNOWN'
            area_unit = f"{flux_unit}·{unit.lower() if unit else 'UNKNOWN'}"
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((fit['y_data'] - np.mean(fit['y_data'])) ** 2)
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
            amplitude_error=fit.get('amplitude_error'), catalog_sigma=fit.get('catalog_sigma'),
            noise_std=fit.get('noise_std'),
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
                                   continuum_poly_deg=3, excluded_wavelengths=None,
                                   continuum_params=None, snr_threshold=3.0, header=None, wave_unit=None):
    """
    Extract total CO flux from a single spectrum by fitting all CO lines.

    Parameters
    ----------
    spectrum_wave : array-like or Quantity
        Wavelength array for the spectrum (can have astropy units)
    spectrum_flux : array-like or Quantity
        Flux array for the spectrum (can have astropy units)
    CO_catalog : pd.DataFrame
        CO line catalog (columns: line_wave_um, species, ...)
    contam_catalog : pd.DataFrame
        Contaminating lines catalog (columns: line_wave_um, ...)
    wave_boundaries : array-like, optional
        Wavelength boundaries for fitting chunks. If None, auto-calculated from wave_min/wave_max.
    wave_min : float, optional
        Minimum wavelength to fit. If None, uses min of CO_catalog line_wave_um.
    wave_max : float, optional
        Maximum wavelength to fit. If None, uses max of CO_catalog line_wave_um.
    do_duplicate_removal : bool
        Whether to remove duplicate line detections
    continuum_poly_deg : int
        Unused parameter (kept for backward compatibility)
    excluded_wavelengths : set or list, optional
        Set of wavelengths to exclude from CO flux calculation
    continuum_params : dict, optional
        Override the default fit_auto_continuum() parameters (window_size,
        sigma_threshold, peak_width, peak_distance, peak_exclusion, iterations,
        smoothing_factor). If None, uses the defaults, which work well for typical spectra.
    snr_threshold : float
        Minimum signal-to-noise for a line to count as detected (default: 3.0)
    header : astropy.io.fits.Header
        FITS header. No longer just for the erg/s/cm2 conversion -- every detected line's
        fit width now comes from the instrument's real spectral resolution at that
        wavelength (_line_sigma, MIRI/NIRSpec only), so this is required whenever any line
        is actually detected: a spectrum with no header raises ValueError as soon as a line
        is found, rather than silently fitting with a guessed width.
    wave_unit : str, optional
        Wavelength unit (e.g. 'um'). Used alongside header for the CGS conversion.

    Returns
    -------
    dict
        Keys: 'co_flux', 'contam_flux', 'mean_continuum', 'detected_lines', 'residuals',
        'continuum_curve', 'flux_unit' (unit of 'co_flux'/'contam_flux' -- 'erg/s/cm2' if the
        conversion succeeded, otherwise the cube's native flux unit)
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

    # Whether a CGS conversion is even reachable for this header/flux unit combination
    # (e.g. ALMA's Jy/beam never is) -- tested once with a dummy value at the band center,
    # used to label 'co_flux'/'contam_flux' consistently across every return path below.
    # 'UNKNOWN', not a plausible-looking guess, if the unit couldn't be found at all.
    flux_unit_label = spectrum_flux_unit or resolved_flux_unit or 'UNKNOWN'
    if _to_cgs(1.0, (wave_min + wave_max) / 2.0) is not None:
        flux_unit_label = 'erg/s/cm2'
    
    # wave_boundaries (if not explicitly given) gets computed further down, after this
    # pixel's own lines are detected -- see the comment there for why it can't be built
    # from the full catalog up front.

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
        remove_duplicate='True' if do_duplicate_removal else 'False',
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

    # Per-line sigma from the instrument's own spectral resolution (_line_sigma), same as
    # fit_hitran_lines_collective -- a single shared sigma across every detected line pins
    # them all to the same width and under-resolves the narrow ones (the "horrible fits" bug
    # the Multi-Gaussian path had before this). No fallback if a line's sigma can't be
    # determined: that means the header is missing something MIRI/NIRSpec shouldn't be
    # missing, and this should fail loudly rather than fit with a guessed width.
    detected_co['sigma'] = [_line_sigma(wl, header) for wl in detected_co['line']]

    # DEBUG: Show how many CO lines were detected
    n_co_detected = len(detected_co)
    if n_co_detected > 0:
        print(f"  [DEBUG] CO line wavelengths: {detected_co['line'].values[:5].tolist()}...")
        print(f"  [DEBUG] CO line max amplitudes: {detected_co['max_amplitude'].values[:5].tolist()}...")

    # Run peak detection against the contaminating-line catalog itself. Deliberately no
    # contaminating_lines= avoidance against CO_catalog_for_fit here: that would reject a
    # real H2 detection whenever its peak sits within one pixel of a CO catalog line -- but
    # a contaminant sitting right next to a CO line is exactly the case that most needs its
    # own Gaussian in fit_glines' simultaneous fit (see _line_sigma / the H2-vs-CO flux
    # misattribution this whole contaminant-fitting path exists to fix). A generous/blended
    # max_amplitude guess here is harmless -- curve_fit's own per-line bounds resolve the
    # real split -- but silently dropping the detection isn't.
    if contam_catalog is not None and len(contam_catalog) > 0:
        detected_contam_lines = find_lines(
            spectrum_wave, spectrum_flux, contam_catalog,
            remove_duplicate='True' if do_duplicate_removal else 'False',
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
        # Only used as fit_glines' required positional argument -- every detected line above
        # already carries its own real 'sigma', so fit_glines' per-line-sigma branch is what
        # actually drives the fit; this scalar is never read (see fit_glines' own docstring).
        all_sigmas = pd.concat([detected_co['sigma'], detected_contam['sigma']])
        sigma = float(np.median(all_sigmas))

        # Same _merged_wave_boundaries() approach as fit_hitran_lines_collective, built from
        # this pixel's own detected lines -- not the full catalog, which would chain-merge
        # densely-packed catalog lines into one oversized chunk regardless of what this
        # specific pixel actually detected. Each pixel gets its own boundaries because each
        # pixel detects a different subset.
        if wave_boundaries is None:
            all_centers = list(detected_co['line']) + list(detected_contam['line'])
            wave_boundaries = _merged_wave_boundaries(all_centers, list(all_sigmas))

        print(f"  [DEBUG] Before fit_glines: median sigma={sigma:.6f}, n_waves={len(wave_boundaries)} boundaries")
        print(f"  [DEBUG] Fitting {len(detected_co)} CO + {len(detected_contam)} contaminating lines")
        # No fallback string on purpose -- see fit_glines' own docstring on
        # flux_unit_label/wave_unit_label: None here means the header is genuinely missing
        # something a MIRI/NIRSpec/MUSE header shouldn't be missing.
        co_flux, contam_flux, mean_cont, residuals, continuum_curve, _, _ = fit_glines(
            spectrum_wave, spectrum_flux, wave_boundaries,
            detected_co, detected_contam, sigma,
            flux_unit_label=resolved_flux_unit, wave_unit_label=resolved_wave_unit,
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
                       continuum_poly_deg=3, wave_min=None, wave_max=None, excluded_wavelengths=None,
                       continuum_params=None):
    """
    Create 2D map of total CO rovibrational flux by processing all spatial pixels.
    
    For each pixel (x, y) in the cube:
    1. Extract 1D spectrum using cube_data.get_pixel_spectrum(x, y)
    2. Call extract_co_flux_from_spectrum() to fit and measure CO flux
    3. Store flux values in 2D map
    
    Parameters
    ----------
    cube_data : CubeData
        Data loader object containing the cube and wavelength information
        Must have: cube, wavelength, get_pixel_spectrum(x, y) method
    CO_catalog : pd.DataFrame
        CO line catalog (columns: line_wave_um, species, ...)
    contam_catalog : pd.DataFrame
        Contaminating lines catalog (columns: line_wave_um, ...)
    n_jobs : int
        Number of parallel jobs for processing rows
    progress_callback : callable, optional
        Callback function to report progress: progress_callback(current, total)
    continuum_poly_deg : int
        Unused parameter (kept for backward compatibility)
    wave_min : float, optional
        Minimum wavelength to fit (μm). If None, uses min of CO_catalog.
    wave_max : float, optional
        Maximum wavelength to fit (μm). If None, uses max of CO_catalog.
    excluded_wavelengths : set or list, optional
        Set of CO wavelengths to exclude from flux calculation
    continuum_params : dict, optional
        Override the default fit_auto_continuum() parameters, applied identically
        to every pixel. If None, uses the defaults, which work well for typical spectra.


    Returns
    -------
    dict
        Keys:
        - 'flux_map': 2D array of CO flux for each pixel, in erg/s/cm2 if cube_data.header
          supports the conversion (see 'flux_unit'), otherwise the cube's native flux unit
        - 'contam_flux_map': 2D array of contaminating-line flux per pixel, same units as
          'flux_map'. Already computed as part of separating contaminants from CO in each
          pixel's fit; this just keeps that number instead of discarding it.
        - 'detection_map': 2D array with number of detected CO lines per pixel
        - 'contam_detection_map': 2D array with number of detected contaminating lines per pixel
        - 'wavelength': 1D wavelength array used
        - 'header': FITS header from original cube
        - 'wave_range': (wave_min, wave_max) actually used for the fit
        - 'flux_unit': unit of 'flux_map'/'contam_flux_map' -- 'erg/s/cm2' if the conversion succeeded
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
    
    import sys
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
    import sys
    sys.stdout.flush()
    
    start_time = time.time()
    
    # Thread-safe pixel counter for per-pixel progress updates
    from threading import Lock
    pixel_counter = {'count': 0, 'lock': Lock()}
    total_pixels = n_y * n_x
    
    # No wave_boundaries computed here -- each pixel detects a different set of lines, so
    # each pixel needs its own boundaries built from what it actually detected, not one
    # structure shared across the cube (see extract_co_flux_from_spectrum, which now
    # computes this itself, per call, from its own detected_co/detected_contam).

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
            import sys
            sys.stdout.flush()  # Force flush
            test_result = extract_co_flux_from_spectrum(
                wavelength, test_spectrum, CO_catalog, contam_catalog,
                wave_min=wave_min, wave_max=wave_max,
                do_duplicate_removal=True,
                continuum_poly_deg=continuum_poly_deg,
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
        import traceback, sys
        print(f"      Error: {e}")
        traceback.print_exc()
        sys.stdout.flush()

    
    # Worker function for processing a single row
    def process_row(y_idx):
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
                    continuum_poly_deg=continuum_poly_deg,
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
                import traceback
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
                import sys
                sys.stdout.flush()
    except Exception as e:
        print(f"[ERROR] Error during CO flux extraction: {e}")
        import traceback
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
    Save CO flux map as FITS file with proper metadata.
    
    Parameters
    ----------
    flux_map_data : dict
        Output dictionary from create_co_flux_map() containing:
        - 'flux_map': 2D array of flux values
        - 'detection_map': 2D array of line counts
        - 'wavelength': 1D wavelength array (for reference)
        - 'header': Original FITS header
        - 'wave_range': (wave_min, wave_max) used
    output_path : str
        Output file path (e.g., 'CO_flux_map.fits')
    map_type : str
        Type of map: 'CO_flux', 'contamination_flux', or custom name
    
    Returns
    -------
    bool
        True if successful, False otherwise
    
    Notes
    -----
    Creates a FITS file with 2 HDUs:
    1. PRIMARY: CO flux map (main data)
    2. DETECTION: Number of detected lines per pixel
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
        out_header['COMMENT'] = 'Values are amplitude × σ × √(2π) integrated from all detected CO lines'
        
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
