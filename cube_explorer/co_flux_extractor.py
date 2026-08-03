"""
CO Rovibrational Flux Extraction Module

Fits multiple CO lines and contaminating lines simultaneously from IFU spectra,
extracting and summing the total CO flux to create integrated flux maps.

Uses scipy.optimize.curve_fit for simultaneous multi-gaussian + continuum fitting.

Functions Reused from gaussian_fitter.py:
- estimate_gaussian_params()    : Better initial guesses for Gaussian parameters from data
- calculate_baseline()          : Linear baseline calculation between two points
- compute_noise_std()           : Robust noise/std estimation using RANSAC polynomial fit

This architecture ensures consistency across the codebase and avoids code duplication.
"""

import numpy as np
import pandas as pd
import time
import warnings
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
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

# REUSE: Import helper functions from gaussian_fitter
from .gaussian_fitter import (
    estimate_gaussian_params, calculate_baseline, compute_noise_std,
    get_wave_unit, _resolve_flux_unit, GaussianFitResult, convert_area_to_cgs,
    _extract_value_from_quantity,
)
from .rotational_diagram_fitter import get_spectral_resolution
from .unit_converter import integrated_flux_to_cgs


# ================== Unit Utilities ==================

def extract_units(data):
    """
    Extract value and unit from astropy Quantity or numpy array.
    
    Parameters
    ----------
    data : Quantity or array-like
        Data that may have units
    
    Returns
    -------
    tuple
        (values, unit_str) where values is numpy array and unit_str is string representation
    """
    if hasattr(data, 'unit'):
        # It's an astropy Quantity
        return data.value, str(data.unit)
    else:
        # Plain numpy array or list
        return np.asarray(data), None


def fit_continuum_median_peaks(wavelength, flux, window_size=51, sigma_threshold=2.0,
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
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks
    from scipy.interpolate import UnivariateSpline
    
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
    list
        Sum of all Gaussians plus continuum evaluated at xl
    """
    N = 3  # Parameters per Gaussian
    params = np.asarray(params)
    min_freq = min(xl)
    retval = [sum(gaussian(x, params[i], params[i + 1], params[i + 2]) 
                   for i in np.arange(len(params) // N) * N) + a + b * (x - min_freq) 
              for x in xl]
    return retval


# ================== Line Detection ==================

def find_lines(xwave, yflux, lines_to_search, contaminating_lines=None, remove_duplicate='False', snr_threshold=3.0):
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
    
    Returns
    -------
    pd.DataFrame
        Lines with columns: line, species, disposition (0/1), max_amplitude, line_flux
    """
    # Suppress repetitive output (called for every pixel)
    # print('>>> Finding lines and removing duplicates...')
    xres = np.nanmean(np.diff(xwave))
    
    # Estimate noise level for significance testing
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
                print(f"    [LINE {ii}] {cline:.6f} μm: DETECTED! amplitude={peak_amplitude:.6e} (threshold={snr_threshold * noise_std:.6e})")
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
                    reason = f"Peak at {cwave[cflux_idx_max]:.6f} μm, outside range {cline - xres:.6f}-{cline + xres:.6f}"
                elif peak_amplitude <= snr_threshold * noise_std:
                    reason = f"amplitude={peak_amplitude:.6e} < threshold={snr_threshold * noise_std:.6e}"
                print(f"    [LINE {ii}] {cline:.6f} μm: REJECTED - {reason}")
    
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

def fit_glines(xwave, yflux, list_wboundaries, detco_lines, detcontaminating_lines, sigma):
    """
    Fit simultaneously many Gaussians and the underlying continuum in given
    chunks of wavelengths (given in list_wboundaries).

    Returns:
        1. the sum of the areas of the CO Gaussians (sum_CO_area)
        2. the sum of the areas of the contaminating lines Gaussians (sum_contaminating_area)
        3. the mean of the continuum (mean_cont)
        4. a pandas dataframe with the residual which is good to check the results (df_residual)
        5. a pandas dataframe with the fitted continuum (a + b*wave) per wavelength, NaN outside
           any fitted chunk (df_continuum) -- useful for plotting the actual continuum trend
           rather than just its mean

    """
    sum_CO_area, sum_contaminating_area, mean_cont = 0, 0, 0
    df_residual = pd.DataFrame({
        'wave': xwave,
        'residual': [0] * len(xwave)
    })
    df_continuum = pd.DataFrame({
        'wave': xwave,
        'continuum': np.full(len(xwave), np.nan)
    })

    print('===== Start fitting...')
    start_time = time.time()

    for ii in range(len(list_wboundaries) - 1):
        xwave_min, xwave_max = list_wboundaries[ii], list_wboundaries[ii + 1]
        idx_xwave = (xwave >= xwave_min) & (xwave <= xwave_max)
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

        # Create the NumPy array with input params
        p0_2d = np.column_stack((
            filtered_lines['max_amplitude'].astype(float),
            filtered_lines['line'],
            [sigma] * len(filtered_lines)
        ))
        p0 = np.ravel(p0_2d)  # 1d array
        p0base, p0slope = np.min(yflux), (yflux[idx_xwave][-1] - yflux[idx_xwave][0]) / (xwave[idx_xwave][-1] - xwave[idx_xwave][0])
        p0 = np.insert(p0, [0, 0], [p0base, p0slope])

        # Create NumPy arrays with lower and upper bounds
        lower_bounds_2d = np.column_stack((
            [0.0] * len(filtered_lines),
            filtered_lines['line'] - sigma,
            [0.5 * sigma] * len(filtered_lines)
        ))
        lower_bounds = np.ravel(lower_bounds_2d)
        lower_bounds = np.insert(lower_bounds, [0, 0], [p0base - 2 * np.abs(p0slope), -2 * np.abs(p0slope)])

        upper_bounds_2d = np.column_stack((
            filtered_lines['max_amplitude'].astype(float) * 1.1,
            filtered_lines['line'] + sigma,
            [2 * sigma] * len(filtered_lines)
        ))
        upper_bounds = np.ravel(upper_bounds_2d)
        upper_bounds = np.insert(upper_bounds, [0, 0], [p0base + 2 * np.abs(p0slope), 2 * np.abs(p0slope)])

        # Perform the fit
        try:
            par, cov = curve_fit(many_gaussians_list, xwave[idx_xwave], yflux[idx_xwave], p0=p0,
                                 bounds=(lower_bounds, upper_bounds), method='trf', ftol=1e-5, maxfev=10000)
            print(f'  [FIT DEBUG] curve_fit succeeded, returned {len(par)} parameters')
        except (ValueError, RuntimeError) as e:
            # Bounds are invalid or fit failed - use 3-sigma fallback
            print(f"  curve_fit failed ({str(e)}), using 3-sigma upper limit")
            continuum, std_residuals = cont_fit_ransac(xwave[idx_xwave], yflux[idx_xwave], pdeg=1, show='False')
            current_sum_contaminating = 3 * np.median(std_residuals) * sigma * np.sqrt(2 * np.pi)
            current_sum_CO = 3 * np.median(std_residuals) * sigma * np.sqrt(2 * np.pi)
            a = np.nanmedian(continuum)
            df_continuum.loc[idx_xwave, 'continuum'] = continuum
            sum_CO_area = sum_CO_area + current_sum_CO
            sum_contaminating_area = sum_contaminating_area + current_sum_contaminating
            mean_cont = mean_cont + a
            continue

        a, b = par[0], par[1]  # Extracts which parameters relate to the continuum
        par = par[2:]  # Extracts which parameters relate to the Gaussians
        print(f'  [FIT DEBUG] continuum: a={a:.6e}, b={b:.6e}; {len(par)//3} Gaussian components')
        if len(par) >= 3:
            print(f'  [FIT DEBUG] First line: amp={par[0]:.6e}, center={par[1]:.6f}, sigma={par[2]:.6e}')

        try:
            bfit = many_gaussians_list(xwave[idx_xwave], a, b, *par)
            df_residual.loc[idx_xwave, 'residual'] = yflux[idx_xwave] - bfit
            df_continuum.loc[idx_xwave, 'continuum'] = a + b * xwave[idx_xwave]

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

            print(f'  [FLUX DEBUG] Range {ii}: {len(par)//3} fitted lines, CO flux segment={current_sum_CO:.6e}, contam={current_sum_contaminating:.6e}')
            if current_sum_CO < 0:
                print(f'  [FLUX WARN] Negative CO flux detected! intensities: {[par[jj] for jj in range(0, len(par) - 3 * ncont_lines, 3)]}')

        except RuntimeError as e:
            print("curve_fit did not converge! Fit a straight line and assign 3sigma upper limit", e)
            continuum, std_residuals = cont_fit_ransac(xwave[idx_xwave], yflux[idx_xwave], pdeg=1, show='False')
            current_sum_contaminating = 3 * np.median(std_residuals) * sigma * np.sqrt(2 * np.pi)
            current_sum_CO = 3 * np.median(std_residuals) * sigma * np.sqrt(2 * np.pi)
            a = np.nanmedian(continuum)
            df_continuum.loc[idx_xwave, 'continuum'] = continuum

        sum_CO_area = sum_CO_area + current_sum_CO
        sum_contaminating_area = sum_contaminating_area + current_sum_contaminating
        mean_cont = mean_cont + a
        print(f'  [FLUX DEBUG] Cumulative CO flux: {sum_CO_area:.6e}, Cumulative contam: {sum_contaminating_area:.6e}')

    end_time = time.time()
    print(f"Fitting took {np.around(end_time - start_time, decimals=1)} seconds")
    print('===== End fitting')
    print(f'[FINAL FLUX] Total CO flux: {sum_CO_area:.6e}, Total contam flux: {sum_contaminating_area:.6e}, Mean continuum: {mean_cont/(len(list_wboundaries) - 1):.6e}')

    mean_cont = mean_cont / (len(list_wboundaries) - 1)

    return sum_CO_area, sum_contaminating_area, mean_cont, df_residual, df_continuum


def _line_sigma(wl, header, dx, fallback_sigma_pixels):
    """
    Instrumental Gaussian sigma for a line at wavelength `wl`, from the same spectral
    resolution formulas already used for rotational-diagram upper limits
    (get_spectral_resolution(), rotational_diagram_fitter.py) -- R(lambda) = lambda/delta_lambda,
    MIRI/NIRSpec-specific -- converted FWHM (delta_lambda) -> sigma the same way the rest of
    this codebase does (fwhm = 2.355 * sigma). Falls back to a small multiple of the
    wavelength array's own pixel spacing when the instrument/grating can't be determined
    (e.g. no header, or an instrument get_spectral_resolution doesn't cover).
    """
    try:
        instrument = (header.get('INSTRUME', '') if header is not None else '').strip().upper()
        if 'MIRI' in instrument:
            R = get_spectral_resolution(wl, 'MIRI', header)
        elif 'NIRSPEC' in instrument:
            R = get_spectral_resolution(wl, 'NIRSpec', header)
        else:
            raise ValueError(f"No spectral resolution formula for instrument {instrument!r}")
        return (wl / R) / 2.355
    except Exception:
        return fallback_sigma_pixels * dx


def fit_hitran_lines_collective(spectrum_x, spectrum_y, hitran_lines, continuum,
                                 header=None, wave_unit=None, fallback_sigma_pixels=2.5,
                                 display_window_pixels=20):
    """
    Fit a hand-picked list of HITRAN lines (e.g. from the species dialog) all at once via
    linear least squares (np.linalg.lstsq), instead of the nonlinear curve_fit machinery
    fit_glines() uses. Every line's center is fixed at its catalog wavelength and its sigma
    is fixed from the instrument's own spectral resolution (get_spectral_resolution(),
    reused from rotational_diagram_fitter.py) -- these lines are unresolved, so width is a
    property of the instrument, not something to re-estimate per line or per fit. Fixing
    both turns the problem into a single design-matrix solve: one Gaussian template column
    per line (evaluated with gaussian(), reused as-is) plus a constant term for any small
    residual left after the global continuum subtraction, solved once with lstsq.

    Why this instead of curve_fit (either independent per-line, or grouped/simultaneous):
    - Speed: a linear solve doesn't need iterative refinement or a numerical Jacobian, so it
      stays fast regardless of how many lines are selected -- no need to decide which lines
      are "close enough" to fit together, which is exactly the judgment call that failed
      twice in earlier attempts (a data-driven sigma estimate breaking down in a crowded
      window, and curve_fit's cost blowing up for large simultaneous fits).
    - Blending: ordinary least squares naturally apportions shared flux between overlapping
      templates on its own -- no clustering step needed for correctness either.
    - Fixes the original small/weak-line bug directly: a line's template position can't drift
      away from its catalog wavelength the way auto_detect_boundaries's noisy search could,
      so a weak line's amplitude is always solved for at the right place.

    Trade-off, so it's explicit: centers are fixed, not fitted, so a small systemic
    wavelength/velocity offset (a few lines in earlier test output showed ~0.0001-0.0005um
    shifts from their catalog wavelength) isn't captured -- every line is evaluated exactly
    at its catalog position. lstsq doesn't constrain amplitudes >= 0 the way curve_fit's
    bounds do, so a non-detection can come back with a small negative amplitude from noise;
    that's treated as success=False rather than clipped and reported as a real detection.

    Unlike extract_co_flux_from_spectrum, this does no SNR-based line detection -- the
    caller has already chosen exactly which lines to fit.

    Args:
        spectrum_x, spectrum_y: full spectrum arrays (numpy or Quantity)
        hitran_lines: list of dicts with 'lambda_micron' or 'wavelength', plus whatever
            metadata (species, transition, ...) the caller wants carried through
        continuum: continuum array matching spectrum_x/spectrum_y, subtracted once, globally,
            before fitting -- e.g. from fit_continuum()/fit_continuum_median_peaks().
        header, wave_unit: FITS header (for INSTRUME/GRATING -> spectral resolution, and the
            erg/s/cm^2 area conversion) and wavelength unit string.
        fallback_sigma_pixels: shared Gaussian sigma, as a multiple of the wavelength array's
            own pixel spacing, used only when the instrument's spectral resolution can't be
            determined (no header, or an unsupported instrument).
        display_window_pixels: half-width, in pixels, of the local x_data/y_data/y_fit slice
            kept per line for plotting and R² -- purely cosmetic, does not affect the fit
            itself, which always solves for every line in the selection together.

    Returns:
        list of (line_dict, GaussianFitResult) pairs, one per line with a wavelength, in
        the same order as `hitran_lines`.
    """
    spectrum_x = _extract_value_from_quantity(spectrum_x)
    spectrum_y = _extract_value_from_quantity(spectrum_y)
    continuum = _extract_value_from_quantity(continuum)

    results = [None] * len(hitran_lines)
    valid = []
    for idx, line in enumerate(hitran_lines):
        wl = line.get('lambda_micron')
        if wl is None:
            wl = line.get('wavelength')
        if wl is None:
            results[idx] = (line, GaussianFitResult(
                amplitude=0, center=0, sigma=0, fwhm=0, area=0,
                baseline_slope=0, baseline_intercept=0,
                x_data=np.array([]), y_data=np.array([]),
                baseline=np.array([]), y_corrected=np.array([]),
                y_fit=np.array([]), residuals=np.array([]),
                r_squared=0, success=False, message='Missing wavelength',
            ))
            continue
        valid.append((idx, float(wl), line))

    if not valid or len(spectrum_x) < 4:
        return [r for r in results if r is not None]

    y_corrected = spectrum_y - continuum

    # Pixel spacing -- only used as the fallback line-width source, and to pad/size the
    # cosmetic display window; the fit itself is driven by the physical sigmas below.
    dx = float(np.median(np.diff(spectrum_x)))
    line_sigmas = [_line_sigma(wl, header, dx, fallback_sigma_pixels) for _, wl, _ in valid]
    display_half_width = display_window_pixels * dx

    wavelengths = [wl for _, wl, _ in valid]
    pad = 4 * max(line_sigmas)
    win_mask = (spectrum_x >= min(wavelengths) - pad) & (spectrum_x <= max(wavelengths) + pad)
    x_data = spectrum_x[win_mask]
    y_data = y_corrected[win_mask]
    baseline_window = continuum[win_mask]

    if len(x_data) < len(valid) + 1:
        # Not enough points to solve for every line plus the baseline term -- fail all lines
        # rather than silently return an under-determined/meaningless solution.
        for orig_idx, wl, line in valid:
            results[orig_idx] = (line, GaussianFitResult(
                amplitude=0, center=wl, sigma=0, fwhm=0, area=0,
                baseline_slope=0, baseline_intercept=0,
                x_data=x_data, y_data=y_data,
                baseline=baseline_window, y_corrected=y_data,
                y_fit=np.zeros_like(x_data), residuals=y_data,
                r_squared=0, success=False,
                message="Not enough spectral points in this wavelength range to fit",
            ))
        return [r for r in results if r is not None]

    # Design matrix: one fixed-position, fixed-width Gaussian template per line (amplitude 1,
    # solved for by lstsq), plus a constant term to absorb any small residual left after the
    # global continuum subtraction.
    template_cols = [gaussian(x_data, 1.0, wl, sig) for (_, wl, _), sig in zip(valid, line_sigmas)]
    template_cols.append(np.ones_like(x_data))
    A = np.column_stack(template_cols)

    coeffs, _, _, _ = np.linalg.lstsq(A, y_data, rcond=None)
    amplitudes = coeffs[:-1]
    fitted_curve = A @ coeffs

    unit = get_wave_unit(header, wave_unit)

    for i, (orig_idx, wl, line) in enumerate(valid):
        amp = float(amplitudes[i])
        sigma_i = line_sigmas[i]
        fwhm = 2.355 * sigma_i
        half_width = max(display_half_width, 3 * sigma_i)
        local_mask = (x_data >= wl - half_width) & (x_data <= wl + half_width)
        x_local = x_data[local_mask]
        y_local = y_data[local_mask]
        y_fit_local = fitted_curve[local_mask]
        baseline_local = baseline_window[local_mask]
        residuals_local = y_local - y_fit_local

        if amp <= 0:
            result = GaussianFitResult(
                amplitude=0, center=wl, sigma=sigma_i, fwhm=fwhm, area=0,
                baseline_slope=0.0, baseline_intercept=float(baseline_local[0]) if len(baseline_local) else 0.0,
                x_data=x_local, y_data=y_local,
                baseline=baseline_local, y_corrected=y_local,
                y_fit=y_fit_local, residuals=residuals_local,
                r_squared=0, success=False,
                message=f"No significant signal (fitted amplitude {amp:.3g} <= 0)",
            )
        else:
            area_native = amp * sigma_i * np.sqrt(2 * np.pi)
            area_cgs = convert_area_to_cgs(amp, sigma_i, wl, header, unit) if header is not None else None
            if area_cgs is not None:
                area, area_unit = area_cgs, "erg/s/cm²"
            else:
                area = area_native
                flux_unit = header.get('BUNIT', 'flux') if header is not None else 'flux'
                area_unit = f"{flux_unit}·{unit.lower() if unit else 'wave'}"
            ss_res = np.sum(residuals_local ** 2)
            ss_tot = np.sum((y_local - np.mean(y_local)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            message = (f"Warning: Poor fit quality (R² = {r_squared:.3f})" if r_squared < 0.5
                       else f"Success: Good fit (R² = {r_squared:.3f})")
            result = GaussianFitResult(
                amplitude=amp, center=wl, sigma=sigma_i, fwhm=fwhm,
                area=area, area_unit=area_unit, area_native=area_native,
                baseline_slope=0.0, baseline_intercept=float(baseline_local[0]) if len(baseline_local) else 0.0,
                x_data=x_local, y_data=y_local,
                baseline=baseline_local, y_corrected=y_local,
                y_fit=y_fit_local, residuals=residuals_local,
                r_squared=r_squared, success=True, message=message,
            )
        results[orig_idx] = (line, result)

    return [r for r in results if r is not None]


# ================== High-Level Extraction ==================

def extract_co_flux_from_spectrum(spectrum_wave, spectrum_flux, CO_catalog, contam_catalog,
                                   wave_boundaries=None, wave_min=None, wave_max=None, do_duplicate_removal=True,
                                   subtract_continuum=False, continuum_poly_deg=3, excluded_wavelengths=None,
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
    subtract_continuum : bool
        Whether to subtract continuum before fitting lines using median filter + peak
        detection method (same as GUI's Automatic Continuum) (default: False)
    continuum_poly_deg : int
        Unused parameter (kept for backward compatibility)
    excluded_wavelengths : set or list, optional
        Set of wavelengths to exclude from CO flux calculation
    continuum_params : dict, optional
        Override the default fit_continuum_median_peaks() parameters (window_size,
        sigma_threshold, peak_width, peak_distance, peak_exclusion, iterations,
        smoothing_factor). If None, uses the defaults, which work well for typical spectra.
    snr_threshold : float
        Minimum signal-to-noise for a line to count as detected (default: 3.0)
    header : astropy.io.fits.Header, optional
        FITS header, used to convert 'co_flux'/'contam_flux' to erg/s/cm2 (via
        unit_converter.integrated_flux_to_cgs -- the same conversion used for Gaussian
        line areas). Without it, they stay in the cube's native flux units.
    wave_unit : str, optional
        Wavelength unit (e.g. 'um'). Required alongside header for the CGS conversion.

    Returns
    -------
    dict
        Keys: 'co_flux', 'contam_flux', 'mean_continuum', 'detected_lines', 'residuals',
        'continuum_curve', 'flux_unit' (unit of 'co_flux'/'contam_flux' -- 'erg/s/cm2' if the
        conversion succeeded, otherwise the cube's native flux unit)
    """
    # Extract values and units from Quantities if needed. Named separately from the
    # header/wave_unit CGS-conversion parameters above, which take priority when both
    # are given (a Quantity's own unit is just a fallback).
    spectrum_wave, spectrum_wave_unit = extract_units(spectrum_wave)
    spectrum_flux, spectrum_flux_unit = extract_units(spectrum_flux)

    # Apply continuum subtraction if requested (BEFORE line detection)
    # Uses the same median filter + peak detection method as the GUI's Automatic Continuum
    continuum_model = None
    if subtract_continuum:
        try:
            # Use the same continuum fitting method as the GUI, with caller-supplied
            # parameters if given, otherwise the defaults baked into fit_continuum_median_peaks
            continuum_model = fit_continuum_median_peaks(
                spectrum_wave, spectrum_flux, **(continuum_params or {})
            )
            # Subtract continuum from flux
            if continuum_model is not None and not np.all(np.isnan(continuum_model)):
                spectrum_flux = spectrum_flux - continuum_model
            else:
                print(f"  Warning: Continuum fitting returned NaN, using original spectrum")
        except Exception as e:
            print(f"  Warning: Continuum subtraction failed ({e}), using original spectrum")
            continuum_model = None

    # Resolve the wave/flux units once, for the CGS conversion below and for labelling
    # 'co_flux'/'contam_flux' in every return path (even the zero-flux ones)
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
    
    print(f"  Extracting CO flux in wavelength range: {wave_min:.4f} - {wave_max:.4f} μm")
    print(f"  [DEBUG] Spectrum: {len(spectrum_wave)} wavelength points, flux range {np.nanmin(spectrum_flux):.4e} to {np.nanmax(spectrum_flux):.4e}")
    if spectrum_wave_unit:
        print(f"  Wavelength unit: {spectrum_wave_unit}")
    if spectrum_flux_unit:
        print(f"  Flux unit: {spectrum_flux_unit}")

    # Whether a CGS conversion is even reachable for this header/flux unit combination
    # (e.g. ALMA's Jy/beam never is) -- tested once with a dummy value at the band center,
    # used to label 'co_flux'/'contam_flux' consistently across every return path below
    flux_unit_label = spectrum_flux_unit or resolved_flux_unit or 'native'
    if _to_cgs(1.0, (wave_min + wave_max) / 2.0) is not None:
        flux_unit_label = 'erg/s/cm2'
    
    # Auto-determine boundaries if not provided
    if wave_boundaries is None:
        # Use constant 0.1 micron chunks for consistent fitting resolution
        chunk_size = 0.1  # microns
        n_chunks = max(2, int(np.ceil((wave_max - wave_min) / chunk_size)) + 1)
        wave_boundaries = np.linspace(wave_min, wave_max, n_chunks)
    
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
    
    # Find detected lines (contam_catalog can be None if user didn't upload contaminating lines)
    detected_lines = find_lines(
        spectrum_wave, spectrum_flux, CO_catalog_for_fit,
        contaminating_lines=contam_catalog,
        remove_duplicate='True' if do_duplicate_removal else 'False',
        snr_threshold=snr_threshold
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
        ]
    except KeyError as e:
        print(f"  Error: detected_lines missing column - columns are: {detected_lines.columns.tolist()}")
        print(f"  detected_lines head:\n{detected_lines.head()}")
        raise
    
    # DEBUG: Show how many CO lines were detected
    n_co_detected = len(detected_co)
    if n_co_detected > 0:
        print(f"  [DEBUG] CO line wavelengths: {detected_co['line'].values[:5].tolist()}...")
        print(f"  [DEBUG] CO line max amplitudes: {detected_co['max_amplitude'].values[:5].tolist()}...")
    
    # Filter contam_catalog to only include detected lines
    if len(detected_lines) > 0 and 'line' in detected_lines.columns:
        detected_wavelengths = detected_lines.loc[detected_lines['disposition'] == 1, 'line'].values
        if len(detected_wavelengths) > 0 and contam_catalog is not None and 'line_wave_um' in contam_catalog.columns:
            detected_contam = contam_catalog[
                contam_catalog['line_wave_um'].isin(detected_wavelengths)
            ].copy()
        else:
            detected_contam = pd.DataFrame()
    else:
        detected_contam = pd.DataFrame()
    
    # CRITICAL: Normalize detected_contam to match detected_co structure
    if len(detected_contam) > 0:
        # Rename 'line_wave_um' to 'line' if needed
        if 'line_wave_um' in detected_contam.columns and 'line' not in detected_contam.columns:
            detected_contam = detected_contam.rename(columns={'line_wave_um': 'line'})
        # Add 'disposition' column if missing
        if 'disposition' not in detected_contam.columns:
            detected_contam['disposition'] = 1
        # Add 'max_amplitude' column if missing (estimate from local spectrum)
        if 'max_amplitude' not in detected_contam.columns:
            detected_contam['max_amplitude'] = np.nanmax(spectrum_flux) - np.nanmin(spectrum_flux)
    else:
        # Empty DataFrame - still needs the right columns for fit_glines
        detected_contam = pd.DataFrame(columns=['line', 'disposition', 'max_amplitude', 'species'])
    
    # DEBUG: Show detected lines after proper initialization
    n_contam_detected = len(detected_contam)
    print(f"  [DEBUG] Line detection: {n_co_detected} CO lines detected, {n_contam_detected} contaminating lines")
    
    # Fit if any lines detected
    if len(detected_co) > 0 or len(detected_contam) > 0:
        # This is the line width in wavelength units, typically 1-2 pixels worth
        try:
            y_for_sigma = spectrum_flux - np.nanmin(spectrum_flux)  # Remove baseline
            _, _, sigma = estimate_gaussian_params(spectrum_wave, y_for_sigma)
            # Ensure sigma is reasonable (between 0.001 and 0.1 microns for typical NIRSpec)
            if sigma is None or not np.isfinite(sigma) or sigma < 0.0001 or sigma > 0.1:
                sigma = 0.005  # Reasonable default for NIRSpec at 5 microns
        except Exception as e:
            # print(f"  Warning: sigma estimation failed ({e}), using default 0.005")
            sigma = 0.005
        
        print(f"  [DEBUG] Before fit_glines: sigma={sigma:.6f}, n_waves={len(wave_boundaries)} boundaries")
        print(f"  [DEBUG] Fitting {len(detected_co)} CO + {len(detected_contam)} contaminating lines")
        co_flux, contam_flux, mean_cont, residuals, continuum_curve, _per_line_fits = fit_glines(
            spectrum_wave, spectrum_flux, wave_boundaries,
            detected_co, detected_contam, sigma
        )
        print(f"  [DEBUG] fit_glines returned: co_flux={co_flux:.6e}, contam_flux={contam_flux:.6e}, mean_cont={mean_cont:.6e}")
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

    # Convert co_flux/contam_flux to erg/s/cm2 -- same conversion used for Gaussian line
    # areas (unit_converter.integrated_flux_to_cgs), keyed on the mean detected line's
    # wavelength since that's what the flux was actually integrated around
    co_flux_cgs = _to_cgs(co_flux, mean_detected_wavelength)
    if co_flux_cgs is not None:
        contam_flux_cgs = _to_cgs(contam_flux, mean_detected_wavelength)
        co_flux, contam_flux = co_flux_cgs, contam_flux_cgs if contam_flux_cgs is not None else contam_flux

    # continuum_curve so far only holds fit_glines' own per-chunk baseline (a + b*wave), fit
    # against the ALREADY continuum-subtracted spectrum -- so it sits near zero. Add back the
    # upfront continuum_model so the returned curve tracks the original (un-subtracted)
    # spectrum's baseline, which is what actually gets plotted for inspection.
    if continuum_model is not None:
        continuum_curve = continuum_curve.copy()
        continuum_curve['continuum'] = continuum_model + continuum_curve['continuum'].fillna(0.0).values

    return {
        'co_flux': co_flux,
        'contam_flux': contam_flux,
        'mean_continuum': mean_cont,
        'detected_lines': detected_lines,
        'residuals': residuals,
        'continuum_curve': continuum_curve,
        'flux_unit': flux_unit_label,
        'mean_detected_wavelength': mean_detected_wavelength
    }


# ================== Full Cube Processing ==================

def create_co_flux_map(cube_data, CO_catalog, contam_catalog, n_jobs=8, progress_callback=None, cancel_flag=None,
                       subtract_continuum=False, continuum_poly_deg=3, wave_min=None, wave_max=None, excluded_wavelengths=None,
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
    subtract_continuum : bool
        Whether to subtract continuum from each pixel using median filter + peak
        detection method (same as GUI's Automatic Continuum) before fitting lines
    continuum_poly_deg : int
        Unused parameter (kept for backward compatibility)
    wave_min : float, optional
        Minimum wavelength to fit (μm). If None, uses min of CO_catalog.
    wave_max : float, optional
        Maximum wavelength to fit (μm). If None, uses max of CO_catalog.
    excluded_wavelengths : set or list, optional
        Set of CO wavelengths to exclude from flux calculation
    continuum_params : dict, optional
        Override the default fit_continuum_median_peaks() parameters, applied identically
        to every pixel. If None, uses the defaults, which work well for typical spectra.


    Returns
    -------
    dict
        Keys:
        - 'flux_map': 2D array of CO flux for each pixel, in erg/s/cm2 if cube_data.header
          supports the conversion (see 'flux_unit'), otherwise the cube's native flux unit
        - 'detection_map': 2D array with number of detected CO lines per pixel
        - 'wavelength': 1D wavelength array used
        - 'header': FITS header from original cube
        - 'wave_range': (wave_min, wave_max) actually used for the fit
        - 'flux_unit': unit of 'flux_map' -- 'erg/s/cm2' if the conversion succeeded

    Notes
    -----
    - Uses ThreadPoolExecutor for parallel processing (row-wise)
    - Skips pixels with all NaN spectra
    - Returns NaN for failed fits or edge cases
    - Gaussian sigma is estimated from each spectrum's data
    - If subtract_continuum=True, applies median filter + peak detection continuum fit to each pixel
    """
    
    if cube_data.cube is None:
        print("❌ Error: No cube data loaded")
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
    detection_map = np.full((n_y, n_x), 0, dtype=np.int32)
    wavelength_map = np.full((n_y, n_x), np.nan, dtype=np.float32)  # Track mean detected wavelength per pixel
    
    print(f">>> Creating CO flux map: {n_y}×{n_x} pixels, {n_wave} wavelength elements")
    print(f"    Wavelength range: {wave_min:.4f} - {wave_max:.4f} μm")
    print(f"    Continuum subtraction: {'ENABLED' if subtract_continuum else 'DISABLED'}")
    
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
    
    # Define auto wavelength boundaries for fitting
    # Use constant 0.1 micron chunks for consistent, predictable fitting resolution
    chunk_size = 0.1  # microns
    n_chunks = max(2, int(np.ceil((wave_max - wave_min) / chunk_size)) + 1)
    wave_boundaries = np.linspace(wave_min, wave_max, n_chunks)
    
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
            print(f"    ⚠️  WARNING: No valid pixels found in sample!")
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
            print(f"    ✓ Found valid pixels in center region")
        
        sys.stdout.flush()
        
        # Now extract from first pixel
        test_spectrum = cube_data.get_pixel_spectrum(0, 0)
        if test_spectrum is not None and not np.all(np.isnan(test_spectrum)):
            print(f"      Spectrum: shape={test_spectrum.shape}, min={np.nanmin(test_spectrum):.4e}, max={np.nanmax(test_spectrum):.4e}")
            import sys
            sys.stdout.flush()  # Force flush
            test_result = extract_co_flux_from_spectrum(
                wavelength, test_spectrum, CO_catalog, contam_catalog,
                wave_boundaries=wave_boundaries, wave_min=wave_min, wave_max=wave_max,
                do_duplicate_removal=True,
                subtract_continuum=subtract_continuum,
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
        row_detections = np.full(n_x, 0, dtype=np.int32)
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
                    wave_boundaries=wave_boundaries, wave_min=wave_min, wave_max=wave_max,
                    do_duplicate_removal=True,
                    subtract_continuum=subtract_continuum,
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
                
                if result is not None:
                    # Check if flux value is actually valid
                    co_flux_val = result['co_flux']
                    
                    # Step 3: Store results in maps
                    row_fluxes[x_idx] = co_flux_val
                    row_wavelengths[x_idx] = result.get('mean_detected_wavelength', np.nan)  # Store wavelength for conversion
                    
                    # Count detected CO lines
                    n_detected = len(result['detected_lines'][result['detected_lines']['disposition'] == 1])
                    row_detections[x_idx] = n_detected
                    
                    num_success += 1
                else:
                    num_failures += 1
            
            except Exception as e:
                # Log error for debugging - CRITICAL: need to see what's failing
                import traceback
                print(f"  ❌ Extraction failed at pixel ({x_idx}, {y_idx}): {str(e)}")
                traceback.print_exc()
                num_failures += 1
                continue
            
            # Print pixel-level progress inside row (every 10 pixels)
            if (x_idx + 1) % max(1, n_x // 5) == 0:
                print(f"    Row {y_idx}: {x_idx + 1}/{n_x} pixels...")
        
        return y_idx, row_fluxes, row_detections, row_wavelengths
    
    # Process rows in parallel
    try:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_row, y) for y in range(n_y)]
            
            for i, future in enumerate(as_completed(futures)):
                # Check if cancellation was requested
                if cancel_flag and cancel_flag.cancelled:
                    print("⚠️  Extraction cancelled by user - stopping all remaining tasks")
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    # Force shutdown immediately
                    executor.shutdown(wait=False)
                    break
                
                try:
                    y_idx, row_fluxes, row_detections, row_wavelengths = future.result()
                    flux_map[y_idx, :] = row_fluxes
                    detection_map[y_idx, :] = row_detections
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
        print(f"❌ Error during CO flux extraction: {e}")
        import traceback
        traceback.print_exc()
    
    end_time = time.time()
    print(f"\n>>> CO flux map creation complete ({end_time - start_time:.1f}s)")
    print(f"    Valid pixels: {np.sum(~np.isnan(flux_map))}/{n_y * n_x}")
    print(f"    Mean CO flux: {np.nanmean(flux_map):.4e}")
    print(f"    Max CO flux: {np.nanmax(flux_map):.4e}")
    
    # flux_map is already in erg/s/cm2 (or native units) per-pixel -- each extract_co_flux_from_spectrum
    # call above converted it individually via unit_converter.integrated_flux_to_cgs, using that
    # pixel's own detected-line wavelength. Determine the map-wide label the same way (dimensional
    # reachability doesn't vary pixel to pixel; only whether a line was found does).
    header = cube_data.header
    resolved_wave_unit = get_wave_unit(header, cube_data.wave_unit)
    resolved_flux_unit = _resolve_flux_unit(None, header)
    test_cgs = None
    if header is not None:
        test_cgs = integrated_flux_to_cgs(1.0, resolved_flux_unit, wavelength=(wave_min + wave_max) / 2.0,
                                           header=header, wave_unit=resolved_wave_unit)
    flux_unit_output = 'erg/s/cm2' if test_cgs is not None else (resolved_flux_unit or 'native')
    print(f"    Flux map units: {flux_unit_output}")

    return {
        'flux_map': flux_map,
        'detection_map': detection_map,
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
                # Use original header or fallback
                bunit_str = out_header.get('BUNIT', 'MJy/sr')
                # Try to parse as astropy unit
                try:
                    base_unit = u.Unit(bunit_str)
                    flux_unit_str = f"{base_unit} um"
                except (ValueError, TypeError):
                    # Fallback to string concatenation if not a valid unit
                    flux_unit_str = f"{bunit_str} um"
        except Exception:
            flux_unit_str = flux_unit_output if flux_unit_output else "MJy/sr um"
        
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
        
        print(f"✓ Flux map saved to {output_path}")
        print(f"  - PRIMARY HDU: {map_type} ({flux_map.shape[0]}×{flux_map.shape[1]})")
        print(f"  - ERROR HDU: Uncertainty map")
        print(f"  - DETECTION HDU: Line counts per pixel")
        
        return True
    
    except Exception as e:
        print(f"❌ Error saving FITS file: {e}")
        return False
