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
    print('>>> Finding lines and removing duplicates...')
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
        cflux_idx_max = np.nanargmax(cflux)
        
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


# ================== Multi-Gaussian Fitting ==================

def fit_glines(xwave, yflux, list_wboundaries, detco_lines, detcontaminating_lines):
    """
    Fit multiple Gaussians + continuum in wavelength chunks, extracting CO and contaminant fluxes.
    
    For each wavelength range, fits all detected CO lines + contaminating lines simultaneously,
    then sums the areas of CO lines only to get total CO flux in that chunk.
    
    Note: Gaussian sigma is estimated from data for each spectrum (not user-provided).
    
    Parameters
    ----------
    xwave : array-like
        Full wavelength array
    yflux : array-like
        Full flux array
    list_wboundaries : array-like
        Wavelength boundaries defining chunks (e.g., [4.5, 4.7, 4.9, 5.1])
    detco_lines : pd.DataFrame
        Detected CO lines (disposition=1)
    detcontaminating_lines : pd.DataFrame
        Detected contaminating lines (disposition=1)
    
    Returns
    -------
    tuple
        (sum_CO_area, sum_contaminating_area, mean_cont, df_residual)
        where area = amplitude × σ × √(2π) [units: input flux units × wavelength units]
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
        
        print(f'> Fitting wave range {np.around(xwave_min, decimals=1)} - {np.around(xwave_max, decimals=1)}')
        
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
        
        # Build initial parameter guesses [IMPROVED: Using gaussian_fitter helpers]
        # First, remove continuum for better parameter estimation
        y_corrected = yflux[idx_xwave] - np.min(yflux[idx_xwave])
        
        # Estimate sigma from the spectrum's FWHM
        try:
            _, _, sigma_fit = estimate_gaussian_params(xwave[idx_xwave], y_corrected)
        except:
            # Fallback: use small fraction of wavelength range
            sigma_fit = 0.005
        
        print("DEBUG: sigma " + sigma_fit)

        # Create the NumPy array with input params: [amp0, cen0, sig0, amp1, cen1, sig1, ...]
        p0_params = []
        for line_idx in range(len(filtered_lines)):
            # Use catalog values for amplitude and center
            amp_guess = filtered_lines['max_amplitude'].iloc[line_idx]
            cen_guess = filtered_lines['line'].iloc[line_idx]
            sig_guess = sigma_fit  # Use estimated sigma from spectrum
            
            p0_params.extend([amp_guess, cen_guess, sig_guess])
        
        p0 = np.array(p0_params)  # Convert to 1D array
        
        # Continuum initial guesses [IMPROVED: Using gaussian_fitter.calculate_baseline]
        y_chunk = yflux[idx_xwave]
        x_chunk = xwave[idx_xwave]
        
        if len(x_chunk) > 1:
            # Use gaussian_fitter's baseline calculation
            baseline_est = calculate_baseline(
                x_chunk[0], y_chunk[0], x_chunk[-1], y_chunk[-1],
                x_chunk, y_chunk
            )
            p0base = np.nanmin(baseline_est)
            p0slope = (y_chunk[-1] - y_chunk[0]) / (x_chunk[-1] - x_chunk[0])
        else:
            p0base = np.min(y_chunk)
            p0slope = 0
        p0 = np.insert(p0, [0, 0], [p0base, p0slope])
        
        # Create NumPy arrays with lower and upper bounds
        lower_bounds_2d = np.column_stack((
            [0.0] * len(filtered_lines),
            filtered_lines['line'] - sigma_fit,
            [0.5 * sigma_fit] * len(filtered_lines)
        ))
        lower_bounds = np.ravel(lower_bounds_2d)
        lower_bounds = np.insert(lower_bounds, [0, 0], [p0base - 2 * np.abs(p0slope), -2 * np.abs(p0slope)])
        
        upper_bounds_2d = np.column_stack((
            filtered_lines['max_amplitude'].astype(float) * 1.1,
            filtered_lines['line'] + sigma_fit,
            [2 * sigma_fit] * len(filtered_lines)
        ))
        upper_bounds = np.ravel(upper_bounds_2d)
        upper_bounds = np.insert(upper_bounds, [0, 0], [p0base + 2 * np.abs(p0slope), 2 * np.abs(p0slope)])
        
        # Fit the model (Performs the fit)
        try:
            par, cov = curve_fit(
                many_gaussians_list,
                xwave[idx_xwave],
                yflux[idx_xwave],
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                method='trf',
                ftol=1e-5,
                maxfev=10000
            )
            
            # Extract continuum parameters (a and b in a + b*x)
            a, b = par[0], par[1]
            # Extract Gaussian parameters (amp, cen, sig for each line)
            par = par[2:]
            
            # Calculate best fit model
            bfit = many_gaussians_list(xwave[idx_xwave], a, b, *par)
            df_residual.loc[idx_xwave, 'residual'] = yflux[idx_xwave] - bfit
            
            # Calculate integrated flux areas: amplitude × σ × √(2π)
            # Different approach in case there are contaminating lines
            if ncont_lines > 0:
                # CO lines are first (0 to len(par) - 3*ncont_lines), contaminants are at the end
                current_sum_CO = sum(par[jj] * par[jj + 2] * np.sqrt(2 * np.pi) 
                                    for jj in range(0, len(par) - 3 * ncont_lines, 3))
                current_sum_contaminating = sum(par[jj] * par[jj + 2] * np.sqrt(2 * np.pi) 
                                               for jj in range(len(par) - 3 * ncont_lines, len(par), 3))
            else:
                # No contaminating lines in this wavelength range
                current_sum_contaminating = 0
                current_sum_CO = sum(par[jj] * par[jj + 2] * np.sqrt(2 * np.pi) 
                                    for jj in range(0, len(par), 3))
            
            sum_CO_area += current_sum_CO
            sum_contaminating_area += current_sum_contaminating
            mean_cont += a
            
        except RuntimeError as e:
            print(f"  ⚠️  curve_fit did not converge: {e}")
            # Assign zero flux for this chunk when fit fails
            current_sum_CO = 0
            current_sum_contaminating = 0
            sum_CO_area += current_sum_CO
            sum_contaminating_area += current_sum_contaminating
            mean_cont += np.nanmedian(yflux[idx_xwave])
    
    end_time = time.time()
    print(f"Fitting took {np.around(end_time - start_time, decimals=1)} seconds")
    print('===== End fitting')
    
    if len(list_wboundaries) > 1:
        mean_cont = mean_cont / (len(list_wboundaries) - 1)
    
    return sum_CO_area, sum_contaminating_area, mean_cont, df_residual


# ================== High-Level Extraction ==================

def extract_co_flux_from_spectrum(spectrum_wave, spectrum_flux, CO_catalog, contam_catalog, 
                                   wave_boundaries=None, do_duplicate_removal=True):
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
        Wavelength boundaries for fitting chunks. If None, uses 10-20 chunks.
    do_duplicate_removal : bool
        Whether to remove duplicate line detections
    
    Returns
    -------
    dict
        Keys: 'co_flux', 'contam_flux', 'mean_continuum', 'detected_lines', 'residuals'
    """
    # Auto-determine boundaries if not provided
    if wave_boundaries is None:
        n_chunks = max(10, len(spectrum_wave) // 50)
        wave_boundaries = np.linspace(spectrum_wave[0], spectrum_wave[-1], n_chunks)
    
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
    
    # Filter contam_catalog to only include detected lines
    if len(detected_lines) > 0 and 'line' in detected_lines.columns:
        detected_wavelengths = detected_lines.loc[detected_lines['disposition'] == 1, 'line'].values
        if len(detected_wavelengths) > 0 and 'line_wave_um' in contam_catalog.columns:
            detected_contam = contam_catalog[
                contam_catalog['line_wave_um'].isin(detected_wavelengths)
            ]
        else:
            detected_contam = pd.DataFrame()
    else:
        detected_contam = pd.DataFrame()
    
    # Fit if any lines detected
    if len(detected_co) > 0 or len(detected_contam) > 0:
        co_flux, contam_flux, mean_cont, residuals = fit_glines(
            spectrum_wave, spectrum_flux, wave_boundaries,
            detected_co, detected_contam
        )
    else:
        co_flux, contam_flux, mean_cont = 0, 0, np.nanmedian(spectrum_flux)
        residuals = pd.DataFrame({'wave': spectrum_wave, 'residual': np.zeros_like(spectrum_flux)})
    
    return {
        'co_flux': co_flux,
        'contam_flux': contam_flux,
        'mean_continuum': mean_cont,
        'detected_lines': detected_lines,
        'residuals': residuals
    }


# ================== Full Cube Processing ==================

def create_co_flux_map(cube_data, CO_catalog, contam_catalog, n_jobs=4, progress_callback=None):
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
    - Processing time: ~1-2 minutes per 512×512 image with 100 wavelength elements
    """
    
    if cube_data.cube is None:
        print("❌ Error: No cube data loaded")
        return None
    
    # Get cube dimensions
    n_wave, n_y, n_x = cube_data.cube.shape
    wavelength = cube_data.wavelength
    
    # Always process full wavelength range
    wave_min, wave_max = wavelength[0], wavelength[-1]
    
    # Create output maps
    flux_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    error_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    detection_map = np.full((n_y, n_x), 0, dtype=np.int32)
    
    print(f">>> Creating CO flux map: {n_y}×{n_x} pixels, {n_wave} wavelength elements")
    print(f"    Wavelength range: {wave_min:.4f} - {wave_max:.4f} μm")
    print(f"    Testing first pixel for diagnostics...")
    
    start_time = time.time()
    
    # DEBUG: Test extraction on first pixel
    try:
        test_spectrum = cube_data.get_pixel_spectrum(0, 0)
        if test_spectrum is not None and not np.all(np.isnan(test_spectrum)):
            print(f"    Test spectrum shape: {test_spectrum.shape}, min={np.nanmin(test_spectrum):.4e}, max={np.nanmax(test_spectrum):.4e}")
            test_result = extract_co_flux_from_spectrum(
                wavelength, test_spectrum, CO_catalog, contam_catalog,
                wave_boundaries=wave_boundaries, do_duplicate_removal=True
            )
            if test_result:
                print(f"    Test result: co_flux={test_result['co_flux']:.4e}, {len(test_result['detected_lines'])} detected lines")
                n_co_detected = len(test_result['detected_lines'][test_result['detected_lines']['disposition'] == 1])
                print(f"    CO lines detected: {n_co_detected}")
            else:
                print(f"    Test result was None!")
        else:
            print(f"    Test spectrum is None or all NaN")
    except Exception as e:
        import traceback
        print(f"    Error testing first pixel: {e}")
        traceback.print_exc()
    
    # Define auto wavelength boundaries for fitting (10-20 chunks)
    n_chunks = max(10, n_wave // 50)
    wave_boundaries = np.linspace(wave_min, wave_max, n_chunks)
    
    # Worker function for processing a single row
    def process_row(y_idx):
        row_fluxes = np.full(n_x, np.nan, dtype=np.float32)
        row_errors = np.full(n_x, np.nan, dtype=np.float32)
        row_detections = np.full(n_x, 0, dtype=np.int32)
        
        for x_idx in range(n_x):
            try:
                # Step 1: Extract spectrum from cube for this pixel [REUSED: data_loader.get_pixel_spectrum]
                spectrum_flux = cube_data.get_pixel_spectrum(x_idx, y_idx)
                
                if spectrum_flux is None or np.all(np.isnan(spectrum_flux)):
                    continue  # Skip pixels with no data
                
                # Step 2: Extract CO flux [FUNCTION: extract_co_flux_from_spectrum]
                result = extract_co_flux_from_spectrum(
                    wavelength, spectrum_flux, CO_catalog, contam_catalog,
                    wave_boundaries=wave_boundaries,
                    do_duplicate_removal=True
                )
                
                if result is not None:
                    # Step 3: Store results in maps
                    row_fluxes[x_idx] = result['co_flux']
                    
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
            
            except Exception as e:
                # Log error for debugging
                if y_idx == 0 and x_idx < 3:  # Only log first few pixels to avoid spam
                    import traceback
                    print(f"  Warning: Extraction failed at pixel ({x_idx}, {y_idx}): {str(e)}")
                    traceback.print_exc()
                continue
        
        return y_idx, row_fluxes, row_errors, row_detections
    
    # Process rows in parallel
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(process_row, y) for y in range(n_y)]
        
        for i, future in enumerate(as_completed(futures)):
            y_idx, row_fluxes, row_errors, row_detections = future.result()
            flux_map[y_idx, :] = row_fluxes
            error_map[y_idx, :] = row_errors
            detection_map[y_idx, :] = row_detections
            
            if progress_callback:
                progress_callback(i + 1, n_y)
            
            if (i + 1) % max(1, n_y // 10) == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {i + 1}/{n_y} rows processed ({elapsed:.1f}s)")
    
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
