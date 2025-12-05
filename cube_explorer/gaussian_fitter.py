from astropy.io import fits
def export_residual_cube_fits(cube, wavelength, lam_min=None, lam_max=None, n_jobs=None, out_path="residual_cube.fits", return_cube=False, header=None, precomputed_cube=None, instrument_hint=None):
    """
    Fit Gaussians per pixel and export a cube of normalized residuals (residual/std) as a FITS file.
    Each voxel: (y_corrected - y_fit) / std for that pixel.
    
    Args:
        return_cube: If True, return the cube instead of writing to file
        header: Optional FITS header from original cube to copy metadata
    """
    # Use same wavelength mask as fit_gaussians_per_pixel
    n_wave, n_y, n_x = cube.shape
    if lam_min is None:
        lam_min = wavelength[0]
    if lam_max is None:
        lam_max = wavelength[-1]
    wave_mask = (wavelength >= lam_min) & (wavelength <= lam_max)
    x_data = wavelength[wave_mask]
    n_wave_sel = len(x_data)

    # Prepare output cube
    # If provided, use precomputed residual cube to avoid recomputation
    if precomputed_cube is not None:
        residual_cube = precomputed_cube.astype(np.float32, copy=False)
    else:
        residual_cube = np.full((n_wave_sel, n_y, n_x), np.nan, dtype=np.float32)

    # Fit per pixel (parallel)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def _fit_residual_row(row_idx):
        spectra_row = cube[wave_mask, row_idx, :]  # shape: (n_wave_sel, n_x)
        row_residuals = np.full((n_wave_sel, n_x), np.nan, dtype=np.float32)
        for ix in range(n_x):
            y_data = spectra_row[:, ix]
            # Skip pixels with all NaNs
            if np.all(np.isnan(y_data)):
                continue
            # Skip if all NaN
            if np.all(np.isnan(y_data)):
                continue
            y1 = y_data[0]
            y2 = y_data[-1]
            baseline = calculate_baseline(x_data[0], y1, x_data[-1], y2, x_data)
            y_corrected = y_data - baseline
            mean_val = np.nanmean(y_corrected)
            std_val = np.nanstd(y_corrected)
            # Skip if mean or std are NaN
            if np.isnan(mean_val) or np.isnan(std_val):
                continue
            if std_val < 0.5 * abs(mean_val):
                mad = np.nanmedian(np.abs(y_corrected - np.nanmedian(y_corrected)))
                robust_std = 1.4826 * mad
                sigma_for_fit = robust_std if robust_std > 0 else std_val
                try:
                    amp_guess, center_guess, _ = estimate_gaussian_params(x_data, y_corrected)
                    sigma_guess = sigma_for_fit
                except Exception:
                    continue
            else:
                try:
                    amp_guess, center_guess, sigma_guess = estimate_gaussian_params(x_data, y_corrected)
                except Exception:
                    continue
            if np.max(y_corrected) <= 0:
                continue
            bounds = ([0, x_data[0], 0.001 * (x_data[-1] - x_data[0])],
                      [10 * np.max(y_corrected), x_data[-1], 2 * (x_data[-1] - x_data[0])])
            # Clamp initial guesses to be within bounds
            amp_guess = np.clip(amp_guess, bounds[0][0], bounds[1][0])
            center_guess = np.clip(center_guess, bounds[0][1], bounds[1][1])
            sigma_guess = np.clip(sigma_guess, bounds[0][2], bounds[1][2])
            try:
                popt, _ = curve_fit(
                    gaussian, x_data, y_corrected,
                    p0=[amp_guess, center_guess, sigma_guess],
                    bounds=bounds,
                    maxfev=10000
                )
                amplitude, center, sigma = popt
                y_fit = gaussian(x_data, amplitude, center, sigma)
                # Normalize residuals by std
                std_for_norm = std_val if std_val > 0 else 1.0
                norm_residuals = (y_corrected - y_fit) / std_for_norm
                row_residuals[:, ix] = norm_residuals.astype(np.float32)
            except Exception:
                continue
        return row_idx, row_residuals

    # Determine workers
    import os
    if n_jobs is None:
        try:
            cpu = os.cpu_count() or 1
        except Exception:
            cpu = 1
        n_jobs = max(1, min(cpu, 8))

    if precomputed_cube is None:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_fit_residual_row, iy) for iy in range(n_y)]
            for fut in as_completed(futures):
                row_idx, row_residuals = fut.result()
                residual_cube[:, row_idx, :] = row_residuals

    # Return cube or write FITS file
    if return_cube:
        return residual_cube
    else:
        # Create HDU with header if provided
        if header is not None:
            out_header = header.copy()
        else:
            out_header = fits.Header()

        # Use instrument from header if present, fallback to hint
        instr = out_header.get('INSTRUME', '').strip().upper() or instrument_hint or 'UNKNOWN'
        teles = out_header.get('TELESCOP', '').strip().upper() or 'UNKNOWN'
        # Update header with detected values
        if instr != 'UNKNOWN':
            out_header['INSTRUME'] = instr
        if teles != 'UNKNOWN':
            out_header['TELESCOP'] = teles
        out_header['BUNIT'] = 'Normalized Residuals'
        out_header['OBJECT'] = (out_header.get('OBJECT', 'Unknown') + ' (Residuals)')
        out_header['COMMENT'] = 'Gaussian fit residuals: (data - fit) / noise; wavelength-limited'

        # Update NAXIS and axis sizes
        out_header['NAXIS'] = 3
        out_header['NAXIS1'] = int(n_x)
        out_header['NAXIS2'] = int(n_y)
        out_header['NAXIS3'] = int(n_wave_sel)

        hdu = fits.PrimaryHDU(residual_cube, header=out_header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(out_path, overwrite=True)
        print(f"Residual cube FITS written to {out_path}")

"""
Gaussian Fitting Module for Spectral Line Analysis

This module provides tools for fitting Gaussian profiles to spectral lines
after baseline subtraction. Used for measuring line properties like FWHM,
equivalent width, and integrated flux.
"""

import numpy as np
from scipy.optimize import curve_fit
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class GaussianFitResult:
    """Container for Gaussian fit results and diagnostics"""
    # Fitted parameters
    amplitude: float  # Peak height of Gaussian (A)
    center: float  # Center wavelength (μ)
    sigma: float  # Gaussian width (σ)
    
    # Derived quantities
    fwhm: float  # Full Width at Half Maximum
    area: float  # Integrated area under Gaussian
    
    # Baseline parameters
    baseline_slope: float  # Linear baseline slope
    baseline_intercept: float  # Linear baseline at x1
    
    # Data arrays
    x_data: np.ndarray  # Wavelength values
    y_data: np.ndarray  # Original flux values
    baseline: np.ndarray  # Baseline model
    y_corrected: np.ndarray  # Baseline-subtracted flux
    y_fit: np.ndarray  # Fitted Gaussian model
    residuals: np.ndarray  # Fit residuals
    
    # Quality metrics
    chi_squared: float  # Chi-squared goodness of fit
    r_squared: float  # R-squared coefficient
    success: bool  # Whether fit converged
    message: str  # Status or error message
    
    def __str__(self):
        """Human-readable summary of fit results"""
        return (
            f"Gaussian Fit Results:\n"
            f"  Center (μ):     {self.center:.4f}\n"
            f"  Amplitude (A):  {self.amplitude:.4e}\n"
            f"  Sigma (σ):      {self.sigma:.4f}\n"
            f"  FWHM:           {self.fwhm:.4f}\n"
            f"  Area:           {self.area:.4e}\n"
            f"  R²:             {self.r_squared:.4f}\n"
            f"  Status:         {self.message}"
        )


def calculate_baseline(x1: float, y1: float, x2: float, y2: float, 
                       x_data: np.ndarray) -> np.ndarray:
    """
    Calculate linear baseline between two points.
    
    The baseline is a straight line connecting (x1, y1) and (x2, y2):
    y_baseline(x) = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    Args:
        x1, y1: Left boundary point
        x2, y2: Right boundary point
        x_data: Wavelength values where baseline is evaluated
    
    Returns:
        baseline: Array of baseline values at each x_data point
    """
    if x2 == x1:
        # Avoid division by zero - return constant baseline
        return np.full_like(x_data, y1)
    
    slope = (y2 - y1) / (x2 - x1)
    baseline = y1 + slope * (x_data - x1)
    return baseline


def gaussian(x: np.ndarray, amplitude: float, center: float, 
             sigma: float) -> np.ndarray:
    """
    Gaussian function.
    
    G(x) = A * exp(-(x - μ)² / (2σ²))
    
    Args:
        x: Independent variable (wavelength)
        amplitude: Peak height (A)
        center: Center position (μ)
        sigma: Width parameter (σ)
    
    Returns:
        Gaussian values at each x
    """
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))


def estimate_gaussian_params(x_data: np.ndarray, 
                             y_data: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate initial Gaussian parameters from data.
    
    Strategy:
    - Amplitude: maximum y value
    - Center: x value at maximum y
    - Sigma: estimated from width at half maximum
    
    Args:
        x_data: Wavelength values
        y_data: Baseline-corrected flux values
    
    Returns:
        (amplitude_guess, center_guess, sigma_guess)
    """
    # Find peak
    max_idx = np.argmax(y_data)
    amplitude_guess = y_data[max_idx]
    center_guess = x_data[max_idx]
    
    # Estimate sigma from half-width
    # Find points near half maximum
    half_max = amplitude_guess / 2.0
    above_half = y_data > half_max
    
    if np.sum(above_half) >= 2:
        # Find approximate FWHM
        indices = np.where(above_half)[0]
        left_idx = indices[0]
        right_idx = indices[-1]
        fwhm_guess = x_data[right_idx] - x_data[left_idx]
        # Convert FWHM to sigma: FWHM = 2.355 * sigma
        sigma_guess = fwhm_guess / 2.355
    else:
        # Default: use 1/4 of the range
        sigma_guess = (x_data[-1] - x_data[0]) / 4.0
    
    # Ensure positive and reasonable values
    amplitude_guess = max(amplitude_guess, 1e-10)
    sigma_guess = max(sigma_guess, (x_data[1] - x_data[0]))  # At least one pixel
    
    return amplitude_guess, center_guess, sigma_guess


def fit_gaussian_to_line(x1: float, x2: float, 
                         spectrum_x: np.ndarray, 
                         spectrum_y: np.ndarray) -> GaussianFitResult:
    """
    Fit a Gaussian to a spectral line between two selected points.
    
    Process:
    1. Extract data between x1 and x2
    2. Calculate linear baseline connecting endpoints
    3. Subtract baseline to isolate the line
    4. Fit Gaussian to baseline-subtracted data
    5. Calculate derived quantities (FWHM, area, etc.)
    
    Args:
        x1: Left boundary wavelength
        x2: Right boundary wavelength
        spectrum_x: Full spectrum wavelength array
        spectrum_y: Full spectrum flux array
    
    Returns:
        GaussianFitResult object containing all fit information
    """
    # Ensure x1 < x2
    if x1 > x2:
        x1, x2 = x2, x1
    
    # Extract data within selected region
    mask = (spectrum_x >= x1) & (spectrum_x <= x2)
    x_data = spectrum_x[mask]
    y_data = spectrum_y[mask]
    
    # Validation checks
    if len(x_data) < 4:
        return GaussianFitResult(
            amplitude=0, center=0, sigma=0, fwhm=0, area=0,
            baseline_slope=0, baseline_intercept=0,
            x_data=x_data, y_data=y_data,
            baseline=np.array([]), y_corrected=np.array([]),
            y_fit=np.array([]), residuals=np.array([]),
            chi_squared=np.inf, r_squared=0,
            success=False,
            message="Error: Not enough points in selected region (need at least 4)"
        )
    
    # Get boundary flux values for baseline
    y1 = y_data[0]
    y2 = y_data[-1]
    
    # Calculate baseline
    baseline = calculate_baseline(x1, y1, x2, y2, x_data)
    baseline_slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
    
    # Subtract baseline
    y_corrected = y_data - baseline

    # Noise/peak logic
    mean_val = np.nanmean(y_corrected)
    std_val = np.nanstd(y_corrected)
    if std_val < 3 * abs(mean_val):
        # Treat as noise, recalculate std using robust estimator (median absolute deviation)
        mad = np.nanmedian(np.abs(y_corrected - np.nanmedian(y_corrected)))
        robust_std = 1.4826 * mad  # Approximate std from MAD
        sigma_for_fit = robust_std if robust_std > 0 else std_val
        # Use robust std for initial guess
        amp_guess, center_guess, _ = estimate_gaussian_params(x_data, y_corrected)
        sigma_guess = sigma_for_fit
    else:
        # Normal peak, use default estimation
        amp_guess, center_guess, sigma_guess = estimate_gaussian_params(x_data, y_corrected)

    # Check if there's a meaningful peak
    if np.max(y_corrected) <= 0:
        return GaussianFitResult(
            amplitude=0, center=np.mean(x_data), sigma=0, fwhm=0, area=0,
            baseline_slope=baseline_slope, baseline_intercept=y1,
            x_data=x_data, y_data=y_data,
            baseline=baseline, y_corrected=y_corrected,
            y_fit=np.zeros_like(x_data), residuals=y_corrected,
            chi_squared=np.inf, r_squared=0,
            success=False,
            message="Error: No positive peak found after baseline subtraction"
        )
    
    # Set up bounds to constrain fit
    # Amplitude: must be positive, reasonable upper limit
    # Center: must be within selected range
    # Sigma: must be positive, not larger than range
    x_range = x2 - x1
    bounds = (
        [0, x1, 0.001 * x_range],  # Lower bounds
        [10 * np.max(y_corrected), x2, 2 * x_range]  # Upper bounds
    )
    
    # Clamp initial guesses to be within bounds (prevents "x0 is infeasible" error)
    amp_guess = np.clip(amp_guess, bounds[0][0], bounds[1][0])
    center_guess = np.clip(center_guess, bounds[0][1], bounds[1][1])
    sigma_guess = np.clip(sigma_guess, bounds[0][2], bounds[1][2])
    
    try:
        # Perform least-squares fit
        popt, pcov = curve_fit(
            gaussian, x_data, y_corrected,
            p0=[amp_guess, center_guess, sigma_guess],
            bounds=bounds,
            maxfev=5000  # Reduced from 10000 for faster convergence
        )
        
        amplitude, center, sigma = popt
        
        # Generate fitted curve
        y_fit = gaussian(x_data, amplitude, center, sigma)
        residuals = y_corrected - y_fit
        
        # Calculate derived quantities
        fwhm = 2.355 * sigma  # FWHM = 2*sqrt(2*ln(2)) * sigma
        area = amplitude * sigma * np.sqrt(2 * np.pi)
        
        # Calculate goodness of fit metrics
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_corrected - np.mean(y_corrected))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        chi_squared = ss_res / len(x_data)
        
        # Check fit quality
        if r_squared < 0.5:
            message = f"Warning: Poor fit quality (R² = {r_squared:.3f})"
            success = True  # Still succeeded, just warn
        else:
            message = f"Success: Good fit (R² = {r_squared:.3f})"
            success = True
        
        return GaussianFitResult(
            amplitude=amplitude, center=center, sigma=sigma,
            fwhm=fwhm, area=area,
            baseline_slope=baseline_slope, baseline_intercept=y1,
            x_data=x_data, y_data=y_data,
            baseline=baseline, y_corrected=y_corrected,
            y_fit=y_fit, residuals=residuals,
            chi_squared=chi_squared, r_squared=r_squared,
            success=success, message=message
        )
        
    except Exception as e:
        # Fit failed
        return GaussianFitResult(
            amplitude=0, center=center_guess, sigma=sigma_guess,
            fwhm=0, area=0,
            baseline_slope=baseline_slope, baseline_intercept=y1,
            x_data=x_data, y_data=y_data,
            baseline=baseline, y_corrected=y_corrected,
            y_fit=np.zeros_like(x_data), residuals=y_corrected,
            chi_squared=np.inf, r_squared=0,
            success=False,
            message=f"Error: Fit failed - {str(e)}"
        )


def calculate_equivalent_width(fit_result: GaussianFitResult, 
                               continuum_level: Optional[float] = None) -> float:
    """
    Calculate equivalent width of the fitted line.
    
    EW = ∫ (1 - F_line/F_continuum) dλ ≈ Area / F_continuum
    
    Args:
        fit_result: GaussianFitResult object
        continuum_level: Mean continuum flux level. If None, uses mean baseline.
    
    Returns:
        Equivalent width in wavelength units
    """
    if not fit_result.success:
        return 0.0
    
    if continuum_level is None:
        continuum_level = np.mean(fit_result.baseline)
    
    if continuum_level <= 0:
        return 0.0
    
    # For emission lines: EW = Area / Continuum
    equivalent_width = fit_result.area / continuum_level
    return equivalent_width


def fit_gaussians_per_pixel(cube: np.ndarray, wavelength: np.ndarray,
                            lam_min: Optional[float] = None,
                            lam_max: Optional[float] = None,
                            n_jobs: Optional[int] = None) -> dict:
    """
    Fit a Gaussian to each pixel's spectrum in the cube within a wavelength range.
    
    This function performs the same Gaussian fitting process used in fit_gaussian_to_line(),
    but applies it to every spatial pixel in the cube. For each pixel:
    1. Extract spectrum within [lam_min, lam_max]
    2. Calculate linear baseline from endpoints
    3. Subtract baseline
    4. Fit Gaussian using same initial guesses and bounds
    5. Calculate area using analytic formula: A * σ * sqrt(2π)
    
    Args:
        cube: 3D data cube with shape (n_wavelength, n_y, n_x)
        wavelength: 1D wavelength array with length n_wavelength
        lam_min: Minimum wavelength for fitting region (default: use full range)
        lam_max: Maximum wavelength for fitting region (default: use full range)
    
    Returns:
        Dictionary containing 2D parameter maps:
            'amplitude': Peak amplitude map
            'center': Center wavelength map
            'sigma': Width parameter map
            'fwhm': FWHM map
            'area': Integrated area under Gaussian (AUC) map
            'r_squared': Reduced chi-squared (χ²_red) fit quality map
            'success': Boolean mask of successful fits
    
    Notes:
        - Failed fits store NaN in parameter maps
        - Uses same estimation and bounds as fit_gaussian_to_line()
        - Area calculated analytically: A * σ * sqrt(2π)
        - Fit quality: χ²_red ≈ 1 is good, >>1 is poor, <<1 is overfit
    """
    # Get cube dimensions
    n_wave, n_y, n_x = cube.shape
    
    # Set wavelength range
    if lam_min is None:
        lam_min = wavelength[0]
    if lam_max is None:
        lam_max = wavelength[-1]
    
    # Ensure lam_min < lam_max
    if lam_min > lam_max:
        lam_min, lam_max = lam_max, lam_min
    
    # Create wavelength mask
    wave_mask = (wavelength >= lam_min) & (wavelength <= lam_max)
    x_data = wavelength[wave_mask]
    
    if len(x_data) < 4:
        raise ValueError(f"Not enough wavelength points in range [{lam_min}, {lam_max}] (need at least 4)")
    
    # Initialize output maps
    amplitude_map = np.full((n_y, n_x), np.nan)
    center_map = np.full((n_y, n_x), np.nan)
    sigma_map = np.full((n_y, n_x), np.nan)
    fwhm_map = np.full((n_y, n_x), np.nan)
    area_map = np.full((n_y, n_x), np.nan)
    r_squared_map = np.full((n_y, n_x), np.nan)
    success_map = np.zeros((n_y, n_x), dtype=bool)
    
    # Initialize residuals cube (for caching during fitting)
    n_wave_sel = np.sum(wave_mask)
    residuals_cube = np.full((n_wave_sel, n_y, n_x), np.nan, dtype=np.float32)
    
    # Get boundary wavelengths
    x1 = x_data[0]
    x2 = x_data[-1]
    x_range = x2 - x1
    
    # Progress tracking
    total_pixels = n_y * n_x
    completed = 0
    
    print(f"Fitting Gaussians to {total_pixels} pixels in wavelength range [{lam_min:.4f}, {lam_max:.4f}]")
    
    def _fit_row(row_idx: int):
        spectra_row = cube[wave_mask, row_idx, :]  # shape: (n_wave_sel, n_x)
        amp_row = np.full(n_x, np.nan)
        ctr_row = np.full(n_x, np.nan)
        sig_row = np.full(n_x, np.nan)
        fwhm_row = np.full(n_x, np.nan)
        area_row = np.full(n_x, np.nan)
        qual_row = np.full(n_x, np.nan)
        succ_row = np.zeros(n_x, dtype=bool)
        residuals_row = np.full((n_wave_sel, n_x), np.nan, dtype=np.float32)

        for ix in range(n_x):
            y_data = spectra_row[:, ix]

            # Baseline from endpoints
            y1 = y_data[0]
            y2 = y_data[-1]
            baseline = calculate_baseline(x1, y1, x2, y2, x_data)
            y_corrected = y_data - baseline
            
            # Require at least 2 finite points to compute stats
            finite_mask = np.isfinite(y_corrected)
            if np.count_nonzero(finite_mask) < 2:
                continue
            
            # Noise/peak logic
            mean_val = np.nanmean(y_corrected)
            std_val = np.nanstd(y_corrected)
            if std_val < 0.5 * abs(mean_val):
                mad = np.nanmedian(np.abs(y_corrected - np.nanmedian(y_corrected)))
                robust_std = 1.4826 * mad
                sigma_for_fit = robust_std if robust_std > 0 else std_val
                try:
                    amp_guess, center_guess, _ = estimate_gaussian_params(x_data, y_corrected)
                    sigma_guess = sigma_for_fit
                except Exception:
                    continue
            else:
                try:
                    amp_guess, center_guess, sigma_guess = estimate_gaussian_params(x_data, y_corrected)
                except Exception:
                    continue

            # Require positive peak
            if np.max(y_corrected) <= 0:
                continue

            bounds = (
                [0, x1, 0.001 * x_range],
                [10 * np.max(y_corrected), x2, 2 * x_range]
            )

            # Clamp initial guesses to be within bounds
            amp_guess = np.clip(amp_guess, bounds[0][0], bounds[1][0])
            center_guess = np.clip(center_guess, bounds[0][1], bounds[1][1])
            sigma_guess = np.clip(sigma_guess, bounds[0][2], bounds[1][2])

            try:
                popt, _ = curve_fit(
                    gaussian, x_data, y_corrected,
                    p0=[amp_guess, center_guess, sigma_guess],
                    bounds=bounds,
                    maxfev=5000,  # Reduced from 10000 for faster convergence
                    ftol=1e-4,  # Function tolerance for faster convergence
                    xtol=1e-4   # Parameter tolerance for faster convergence
                )
                amplitude, center, sigma = popt
                y_fit = gaussian(x_data, amplitude, center, sigma)
                residuals = y_corrected - y_fit

                fwhm = 2.355 * sigma
                area = amplitude * sigma * np.sqrt(2 * np.pi)

                n_points = len(y_corrected)
                dof = max(1, n_points - 3)
                noise_estimate = np.std(residuals, ddof=0)
                if noise_estimate > 0:
                    chi_squared_red = np.sum(residuals**2) / (dof * noise_estimate**2)
                else:
                    chi_squared_red = np.inf

                amp_row[ix] = amplitude
                ctr_row[ix] = center
                sig_row[ix] = sigma
                fwhm_row[ix] = fwhm
                area_row[ix] = area
                qual_row[ix] = chi_squared_red
                succ_row[ix] = True
                
                # Cache normalized residuals
                std_val_for_norm = std_val if std_val > 0 else 1.0
                normalized_residuals = residuals / std_val_for_norm
                residuals_row[:, ix] = normalized_residuals.astype(np.float32)
            except Exception:
                # leave NaNs/False
                pass

        return row_idx, amp_row, ctr_row, sig_row, fwhm_row, area_row, qual_row, succ_row, residuals_row

    # Determine workers
    if n_jobs is None:
        try:
            cpu = os.cpu_count() or 1
        except Exception:
            cpu = 1
        n_jobs = max(1, min(cpu, 8))  # cap to avoid oversubscription

    if n_jobs <= 1:
        # Serial path
        for iy in range(n_y):
            _, amp_row, ctr_row, sig_row, f_row, a_row, q_row, s_row, res_row = _fit_row(iy)
            amplitude_map[iy, :] = amp_row
            center_map[iy, :] = ctr_row
            sigma_map[iy, :] = sig_row
            fwhm_map[iy, :] = f_row
            area_map[iy, :] = a_row
            r_squared_map[iy, :] = q_row
            success_map[iy, :] = s_row
            residuals_cube[:, iy, :] = res_row

            # Progress per row
            completed += n_x
            if completed % 1000 == 0:
                print(f"  Progress: {completed}/{total_pixels} pixels ({100*completed/total_pixels:.1f}%)")
    else:
        # Parallel path - adaptive strategy for better performance
        # ThreadPool: less overhead, good for small-medium cubes
        # ProcessPool: better CPU scaling for large cubes (bypasses GIL)
        use_processes = total_pixels > 50000  # Threshold: 50k pixels
        
        if use_processes:
            print(f"Using {n_jobs} worker processes (large cube: {total_pixels:,} pixels)")
            from concurrent.futures import ProcessPoolExecutor
            executor_class = ProcessPoolExecutor
        else:
            print(f"Using {n_jobs} worker threads ({total_pixels:,} pixels)")
            executor_class = ThreadPoolExecutor
        
        with executor_class(max_workers=n_jobs) as ex:
            futures = [ex.submit(_fit_row, iy) for iy in range(n_y)]
            for fut in as_completed(futures):
                row_idx, amp_row, ctr_row, sig_row, f_row, a_row, q_row, s_row, res_row = fut.result()
                amplitude_map[row_idx, :] = amp_row
                center_map[row_idx, :] = ctr_row
                sigma_map[row_idx, :] = sig_row
                fwhm_map[row_idx, :] = f_row
                area_map[row_idx, :] = a_row
                r_squared_map[row_idx, :] = q_row
                success_map[row_idx, :] = s_row
                residuals_cube[:, row_idx, :] = res_row

                completed += n_x
                if completed % (n_x * 5) == 0:
                    print(f"  Progress: {completed}/{total_pixels} pixels ({100*completed/total_pixels:.1f}%)")
    
    # Final summary
    n_success = np.sum(success_map)
    print(f"Fitting complete: {n_success}/{total_pixels} pixels successful ({100*n_success/total_pixels:.1f}%)")
    
    return {
        'amplitude': amplitude_map,
        'center': center_map,
        'sigma': sigma_map,
        'fwhm': fwhm_map,
        'area': area_map,
        'r_squared': r_squared_map,
        'success': success_map,
        'residuals_cube': residuals_cube  # Cached residuals computed during fitting
    }


'''
Area (AUC) Map - ⭐ Most Important
This shows the integrated flux (total energy) from the spectral line at each position:

- Bright regions = strong emission from that line
- Dark regions = weak or no emission
- Physical meaning: Maps where your target (star, galaxy, nebula) is emitting that specific spectral line
- Use case: See the spatial distribution of H-alpha emission, [O III], CO lines, etc.

Amplitude Map
Shows the peak height of the line at each pixel:

- Tells you line intensity, but doesn't account for width
- Bright = tall narrow lines OR tall broad lines
- Less informative than Area for total flux

Center Map
Shows the wavelength/velocity at the line center for each pixel:

- Physical meaning: Detects kinematic motion (redshift/blueshift)
- Blue regions = material moving toward you (blueshifted)
- Red regions = material moving away (redshifted)
- Use case: See rotation in galaxies, outflows in jets, velocity structure in disks

Sigma / FWHM Maps
Shows the line width at each pixel:

- Physical meaning: Line broadening from turbulence, temperature, or rotation
- Broad lines (large sigma) = hot gas, turbulent regions, or velocity dispersion
- Narrow lines (small sigma) = cold, ordered gas
- Use case: Find shocked regions, temperature maps, turbulent mixing zones

R² Map (Fit Quality)
Shows how well the Gaussian fit worked:

- Values near 1.0 (green) = good fit
- Values near 0.5 (yellow) = poor fit, maybe not Gaussian-shaped
- Use case: Quality control - identify pixels where the line is complex (double-peaked, absorption, etc.)

Success Mask
Binary map showing which pixels had successful fits:

- White = fit succeeded
- Black = fit failed (no line detected, too noisy, etc.)
'''