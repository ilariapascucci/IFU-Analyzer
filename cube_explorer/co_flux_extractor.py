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
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed

# RANSAC for continuum fitting (from Pascucci)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor

# REUSE: Import helper functions from gaussian_fitter
from .gaussian_fitter import estimate_gaussian_params, calculate_baseline, compute_noise_std


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

def find_lines(xwave, yflux, lines_to_search, contaminating_lines=None, remove_duplicate='False'):
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
    
    Returns
    -------
    pd.DataFrame
        Lines with columns: line, species, disposition (0/1), max_amplitude, line_flux
    """
    # Suppress repetitive output (called for every pixel)
    # print('>>> Finding lines and removing duplicates...')
    xres = np.nanmean(np.diff(xwave))
    
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
        
        if (cflux_max > cflux_min) and (cwave[cflux_idx_max] <= cline + xres) and (cwave[cflux_idx_max] >= cline - xres):
            if isinstance(contaminating_lines, pd.DataFrame):
                # Check if peak is away from contaminating lines
                if all(np.abs(cwave[cflux_idx_max] - iline) > xres for iline in contaminating_lines['line_wave_um']):
                    lines_dispositions.loc[ii, 'disposition'] = 1
                    lines_dispositions.loc[ii, 'max_amplitude'] = cflux_max - cflux_min
                    lines_dispositions.loc[ii, 'line_flux'] = yflux[idx_cline]
                    potential_lines.append((ii, yflux[idx_cline]))
            else:
                # No need to check for contaminating lines
                lines_dispositions.loc[ii, 'disposition'] = 1
                lines_dispositions.loc[ii, 'max_amplitude'] = cflux_max - cflux_min
                lines_dispositions.loc[ii, 'line_flux'] = yflux[idx_cline]
                potential_lines.append((ii, yflux[idx_cline]))
    
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
                # Prefer main isotopologue (12C16O) if present; get the indices of lines that are not rare isotopologues
                valid_indices = [idx for idx in group if lines_dispositions.loc[idx, 'species'] == '12C16O']
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
    
    """
    sum_CO_area, sum_contaminating_area, mean_cont = 0, 0, 0
    df_residual = pd.DataFrame({
        'wave': xwave,
        'residual': [0] * len(xwave)
    })
    
    print('===== Start fitting...')
    start_time = time.time()
    
    for ii in range(len(list_wboundaries) - 1):
        xwave_min, xwave_max = list_wboundaries[ii], list_wboundaries[ii + 1]
        idx_xwave = (xwave >= xwave_min) & (xwave <= xwave_max)
        print('> Fitting wave range ', np.around(xwave_min, decimals=1), np.around(xwave_max, decimals=1))
        
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
        
        # === TESTING === NOT FROM ORIGINAL FUNCTION
        # Validate bounds before fitting
        try:
            # Check for NaN or Inf in bounds
            if np.any(~np.isfinite(lower_bounds)) or np.any(~np.isfinite(upper_bounds)):
                raise ValueError("Bounds contain NaN or Inf")
            # Check that lower <= upper
            if np.any(lower_bounds > upper_bounds):
                raise ValueError("Lower bounds exceed upper bounds")
            # Check that p0 is within bounds
            if np.any(p0 < lower_bounds) or np.any(p0 > upper_bounds):
                raise ValueError("Initial guess p0 is outside bounds")
            
            # Perform the fit
            par, cov = curve_fit(many_gaussians_list, xwave[idx_xwave], yflux[idx_xwave], p0=p0, 
                                 bounds=(lower_bounds, upper_bounds), method='trf', ftol=1e-5, maxfev=10000)
        except (ValueError, RuntimeError) as e:
            # Bounds are invalid or fit failed - use 3-sigma fallback
            print(f"  curve_fit failed ({str(e)}), using 3-sigma upper limit")
            continuum, std_residuals = cont_fit_ransac(xwave[idx_xwave], yflux[idx_xwave], pdeg=1, show='False')
            current_sum_contaminating = 3 * np.median(std_residuals) * sigma * np.sqrt(2 * np.pi)
            current_sum_CO = 3 * np.median(std_residuals) * sigma * np.sqrt(2 * np.pi)
            a = np.nanmedian(continuum)
            sum_CO_area = sum_CO_area + current_sum_CO
            sum_contaminating_area = sum_contaminating_area + current_sum_contaminating
            mean_cont = mean_cont + a
            continue
        # === TESTING === ENDS

        a, b = par[0], par[1]  # Extracts which parameters relate to the continuum
        par = par[2:]  # Extracts which parameters relate to the Gaussians
        
        try:
            bfit = many_gaussians_list(xwave[idx_xwave], a, b, *par)
            df_residual.loc[idx_xwave, 'residual'] = yflux[idx_xwave] - bfit
            
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
        
        except RuntimeError as e:
            print("curve_fit did not converge! Fit a straight line and assign 3sigma upper limit", e)
            continuum, std_residuals = cont_fit_ransac(xwave[idx_xwave], yflux[idx_xwave], pdeg=1, show='False')
            current_sum_contaminating = 3 * np.median(std_residuals) * sigma * np.sqrt(2 * np.pi)
            current_sum_CO = 3 * np.median(std_residuals) * sigma * np.sqrt(2 * np.pi)
            a = np.nanmedian(continuum)
        
        sum_CO_area = sum_CO_area + current_sum_CO
        sum_contaminating_area = sum_contaminating_area + current_sum_contaminating
        mean_cont = mean_cont + a
    
    end_time = time.time()
    print(f"Fitting took {np.around(end_time - start_time, decimals=1)} seconds")
    print('===== End fitting')
    
    mean_cont = mean_cont / (len(list_wboundaries) - 1)
    
    return sum_CO_area, sum_contaminating_area, mean_cont, df_residual


# ================== High-Level Extraction ==================

def extract_co_flux_from_spectrum(spectrum_wave, spectrum_flux, CO_catalog, contam_catalog, 
                                   wave_boundaries=None, wave_min=None, wave_max=None, do_duplicate_removal=True):
    """
    Extract total CO flux from a single spectrum by fitting all CO lines.
    
    Parameters
    ----------
    spectrum_wave : array-like
        Wavelength array for the spectrum
    spectrum_flux : array-like
        Flux array for the spectrum
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
    
    Returns
    -------
    dict
        Keys: 'co_flux', 'contam_flux', 'mean_continuum', 'detected_lines', 'residuals'
    """
    # Auto-determine wavelength range if not provided
    if wave_min is None:
        wave_min = CO_catalog['line_wave_um'].min()
    if wave_max is None:
        wave_max = CO_catalog['line_wave_um'].max()
    
    print(f"  Extracting CO flux in wavelength range: {wave_min:.4f} - {wave_max:.4f} μm")
    
    # Auto-determine boundaries if not provided
    if wave_boundaries is None:
        n_chunks = max(4, int((wave_max - wave_min) * 200))  # ~1 chunk per 0.005 microns of range
        n_chunks = max(2, min(n_chunks, 10))  # Clamp between 2-10 chunks
        wave_boundaries = np.linspace(wave_min, wave_max, n_chunks)
    
    # Find detected lines
    detected_lines = find_lines(
        spectrum_wave, spectrum_flux, CO_catalog,
        contaminating_lines=contam_catalog,
        remove_duplicate='True' if do_duplicate_removal else 'False'
    )
    
    if detected_lines is None or len(detected_lines) == 0:
        # No lines detected at all
        return {
            'co_flux': 0,
            'contam_flux': 0,
            'mean_continuum': np.nanmedian(spectrum_flux),
            'detected_lines': pd.DataFrame(),
            'residuals': pd.DataFrame({'wave': spectrum_wave, 'residual': np.zeros_like(spectrum_flux)})
        }
    
    # Identify detected CO and contaminating lines in this spectrum
    try:
        detected_co = detected_lines[
            (detected_lines['disposition'] == 1) & 
            (detected_lines['species'] == '12C16O')
        ]
    except KeyError as e:
        print(f"  Error: detected_lines missing column - columns are: {detected_lines.columns.tolist()}")
        print(f"  detected_lines head:\n{detected_lines.head()}")
        raise
    
    # DEBUG: Show how many CO lines were detected
    n_co_detected = len(detected_co)
    # print(f"  extract_co_flux: Found {n_co_detected} CO lines in this spectrum")
    
    # Filter contam_catalog to only include detected lines
    if len(detected_lines) > 0 and 'line' in detected_lines.columns:
        detected_wavelengths = detected_lines.loc[detected_lines['disposition'] == 1, 'line'].values
        if len(detected_wavelengths) > 0 and 'line_wave_um' in contam_catalog.columns:
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
        
        co_flux, contam_flux, mean_cont, residuals = fit_glines(
            spectrum_wave, spectrum_flux, wave_boundaries,
            detected_co, detected_contam, sigma
        )
        # print(f"  fit_glines returned: co_flux={co_flux}, type={type(co_flux)}, is_finite={np.isfinite(co_flux)}")
    else:
        co_flux, contam_flux, mean_cont = 0, 0, np.nanmedian(spectrum_flux)
        residuals = pd.DataFrame({'wave': spectrum_wave, 'residual': np.zeros_like(spectrum_flux)})
        # print(f"  No lines to fit, returning 0")
    
    return {
        'co_flux': co_flux,
        'contam_flux': contam_flux,
        'mean_continuum': mean_cont,
        'detected_lines': detected_lines,
        'residuals': residuals
    }


# ================== Full Cube Processing ==================

def create_co_flux_map(cube_data, CO_catalog, contam_catalog, n_jobs=8, progress_callback=None, cancel_flag=None):
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
    
    Returns
    -------
    dict
        Keys:
        - 'flux_map': 2D array of CO flux for each pixel [counts or flux units × wavelength units]
        - 'error_map': 2D array of uncertainties (std of residuals)
        - 'detection_map': 2D array with number of detected CO lines per pixel
        - 'wavelength': 1D wavelength array used
        - 'header': FITS header from original cube
    
    Notes
    -----
    - Uses ThreadPoolExecutor for parallel processing (row-wise)
    - Skips pixels with all NaN spectra
    - Returns NaN for failed fits or edge cases
    - Gaussian sigma is estimated from each spectrum's data
    """
    
    if cube_data.cube is None:
        print("❌ Error: No cube data loaded")
        return None
    
    # Get cube dimensions
    n_wave, n_y, n_x = cube_data.cube.shape
    wavelength = cube_data.wavelength
    
    # Strip units from wavelength if it's an astropy Quantity (convert to plain numpy array)
    if hasattr(wavelength, 'unit'):
        wavelength = wavelength.value
    
    # Calculate wavelength range from CO catalog (not full spectrum) for efficiency
    wave_min = CO_catalog['line_wave_um'].min()
    wave_max = CO_catalog['line_wave_um'].max()
    
    # DIAGNOSTIC: Check line catalogs
    print(f"    CO catalog: {len(CO_catalog)} lines")
    if len(CO_catalog) > 0:
        print(f"      Wavelength range: {CO_catalog['line_wave_um'].min():.4f} - {CO_catalog['line_wave_um'].max():.4f}")
    print(f"    Contaminating catalog: {len(contam_catalog)} lines")
    if len(contam_catalog) > 0:
        print(f"      Wavelength range: {contam_catalog['line_wave_um'].min():.4f} - {contam_catalog['line_wave_um'].max():.4f}")
    
    # DIAGNOSTIC: Check if line wavelengths overlap cube data wavelengths
    co_overlap = (CO_catalog['line_wave_um'] >= wave_min) & (CO_catalog['line_wave_um'] <= wave_max)
    print(f"    CO lines in data wavelength range: {np.sum(co_overlap)}/{len(CO_catalog)}")
    
    import sys
    sys.stdout.flush()
    
    # Create output maps
    flux_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    error_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    detection_map = np.full((n_y, n_x), 0, dtype=np.int32)
    
    print(f">>> Creating CO flux map: {n_y}×{n_x} pixels, {n_wave} wavelength elements")
    print(f"    Wavelength range: {wave_min:.4f} - {wave_max:.4f} μm")
    
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
    
    # Define auto wavelength boundaries for fitting (fewer chunks for speed)
    # CO lines are sparse enough that 4-6 chunks covers most cases
    # Reduce from n_wave//50 to n_wave//500 for 5x speedup
    n_chunks = max(4, int((wave_max - wave_min) * 200))  # ~1 chunk per 0.005 microns of range
    n_chunks = max(2, min(n_chunks, 10))  # Clamp between 2-10 chunks
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
                do_duplicate_removal=True
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
        row_errors = np.full(n_x, np.nan, dtype=np.float32)
        row_detections = np.full(n_x, 0, dtype=np.int32)
        
        # DIAGNOSTIC: Count successes and failures
        num_success = 0
        num_failures = 0
        num_skipped = 0
        
        for x_idx in range(n_x):
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
                    do_duplicate_removal=True
                )
                
                if result is not None:
                    # Check if flux value is actually valid
                    co_flux_val = result['co_flux']
                    
                    # Step 3: Store results in maps
                    row_fluxes[x_idx] = co_flux_val
                    
                    # Calculate error as std of residuals [IMPROVED: Using gaussian_fitter.compute_noise_std]
                    residuals = result['residuals']['residual'].values
                    # Use robust noise estimation
                    noise_std = compute_noise_std(
                        wavelength, spectrum_flux, pdeg=1
                    )
                    row_errors[x_idx] = noise_std if noise_std is not None else np.nanstd(residuals)
                    
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
        
        return y_idx, row_fluxes, row_errors, row_detections
    
    # Process rows in parallel
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(process_row, y) for y in range(n_y)]
        
        for i, future in enumerate(as_completed(futures)):
            # Check if cancellation was requested
            if cancel_flag and cancel_flag.cancelled:
                print("⚠️  Extraction cancelled by user")
                executor.shutdown(wait=False)
                break
            
            y_idx, row_fluxes, row_errors, row_detections = future.result()
            flux_map[y_idx, :] = row_fluxes
            error_map[y_idx, :] = row_errors
            detection_map[y_idx, :] = row_detections
            
            # Report actual pixel progress
            if progress_callback:
                with pixel_counter['lock']:
                    pixels_done = pixel_counter['count']
                progress_callback(pixels_done, total_pixels)
            
            # Print progress every 10% of rows, but show total pixels processed
            if (i + 1) % max(1, n_y // 10) == 0:
                elapsed = time.time() - start_time
                total_pixels_done = (i + 1) * n_x
                total_pixels = n_y * n_x
                percent = 100 * total_pixels_done / total_pixels
                print(f"  Progress: {total_pixels_done}/{total_pixels} pixels ({percent:.1f}%) in {elapsed:.1f}s")
    
    end_time = time.time()
    print(f"\n>>> CO flux map creation complete ({end_time - start_time:.1f}s)")
    print(f"    Valid pixels: {np.sum(~np.isnan(flux_map))}/{n_y * n_x}")
    print(f"    Mean CO flux: {np.nanmean(flux_map):.4e}")
    print(f"    Max CO flux: {np.nanmax(flux_map):.4e}")
    
    return {
        'flux_map': flux_map,
        'error_map': error_map,
        'detection_map': detection_map,
        'wavelength': wavelength,
        'header': cube_data.header,
        'wave_range': (wave_min, wave_max)
    }


def save_flux_map_fits(flux_map_data, output_path, map_type='CO_flux'):
    """
    Save CO flux map as FITS file with proper metadata.
    
    Parameters
    ----------
    flux_map_data : dict
        Output dictionary from create_co_flux_map() containing:
        - 'flux_map': 2D array of flux values
        - 'error_map': 2D array of uncertainties
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
    Creates a FITS file with 3 HDUs:
    1. PRIMARY: CO flux map (main data)
    2. ERROR: Error/uncertainty map
    3. DETECTION: Number of detected lines per pixel
    """
    
    try:
        flux_map = flux_map_data['flux_map']
        error_map = flux_map_data['error_map']
        detection_map = flux_map_data['detection_map']
        wavelength = flux_map_data['wavelength']
        header = flux_map_data.get('header', fits.Header())
        wave_range = flux_map_data.get('wave_range', (np.nanmin(wavelength), np.nanmax(wavelength)))
        
        # Prepare header [REUSED: Original header as base]
        if header is not None:
            out_header = header.copy()
        else:
            out_header = fits.Header()
        
        # Update header with new information
        out_header['NAXIS'] = 2
        out_header['NAXIS1'] = flux_map.shape[1]
        out_header['NAXIS2'] = flux_map.shape[0]
        out_header['MAPTYPE'] = (map_type, 'Type of flux map')
        out_header['WAVEMIN'] = (wave_range[0], 'Minimum wavelength used [μm]')
        out_header['WAVEMAX'] = (wave_range[1], 'Maximum wavelength used [μm]')
        out_header['BUNIT'] = 'Integrated Flux'
        out_header['COMMENT'] = f'CO Rovibrational {map_type} Map'
        out_header['COMMENT'] = 'Flux extracted by simultaneous multi-Gaussian fitting'
        out_header['COMMENT'] = 'Values are amplitude × σ × √(2π) integrated from all detected CO lines'
        
        # Create primary HDU with flux map
        primary_hdu = fits.PrimaryHDU(flux_map.astype(np.float32), header=out_header)
        
        # Create error map extension
        error_hdu = fits.ImageHDU(error_map.astype(np.float32), name='ERROR')
        error_hdu.header['BUNIT'] = 'Standard Deviation'
        error_hdu.header['COMMENT'] = 'Noise/uncertainty per pixel from fit residuals'
        
        # Create detection map extension
        detection_hdu = fits.ImageHDU(detection_map.astype(np.int32), name='DETECTION')
        detection_hdu.header['BUNIT'] = 'Lines Detected'
        detection_hdu.header['COMMENT'] = 'Number of CO lines detected and fit in each pixel'
        
        # Create HDU list and write
        hdul = fits.HDUList([primary_hdu, error_hdu, detection_hdu])
        hdul.writeto(output_path, overwrite=True)
        
        print(f"✓ Flux map saved to {output_path}")
        print(f"  - PRIMARY HDU: {map_type} ({flux_map.shape[0]}×{flux_map.shape[1]})")
        print(f"  - ERROR HDU: Uncertainty map")
        print(f"  - DETECTION HDU: Line counts per pixel")
        
        return True
    
    except Exception as e:
        print(f"❌ Error saving FITS file: {e}")
        return False
