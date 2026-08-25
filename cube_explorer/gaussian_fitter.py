"""
Gaussian Fitting Module for Spectral Line Analysis

This module provides tools for fitting Gaussian profiles to spectral lines
after baseline subtraction. Used for measuring line properties like FWHM,
equivalent width, and integrated flux.
"""

import os
from astropy.io import fits
from dataclasses import dataclass
from astropy.units import Quantity
from typing import Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor
from scipy.optimize import curve_fit

from .unit_converter import get_wave_unit, convert_area_to_cgs

def _extract_value_from_quantity(value):
    """
    Extract numeric value from a Quantity object
    
    Args:
        value (Quantity): Scalar or array
    
    Returns:
        value (number): Numeric value with units stripped
    """

    if isinstance(value, Quantity):
        return value.value
    elif isinstance(value, np.ndarray):
        # Check if array elements are Quantities
        if len(value) > 0 and isinstance(value.flat[0], Quantity):
            return np.array([v.value for v in value.flat]).reshape(value.shape)
        return value
    else:
        return value


def export_residual_cube_fits(cube, wavelength, lam_min=None, lam_max=None, n_jobs=None, out_path="residual_cube.fits", return_cube=False, header=None, precomputed_cube=None, instrument_hint=None):
    """
    Fit a Gaussian to every pixel and write a cube of normalized residuals, (y_corrected - y_fit) / std, to FITS.

    Args:
        cube (array): 3D data cube with shape (n_wavelength, n_y, n_x)
        wavelength (array): 1D wavelength array with length n_wavelength
        lam_min (float): Minimum wavelength for the fit window. Defaults to wavelength[0]
        lam_max (float): Maximum wavelength for the fit window. Defaults to wavelength[-1]
        n_jobs (int): Number of worker threads. Defaults to auto-detect CPUs
        out_path (str): Output FITS filename
        return_cube (bool): If True, return the cube instead of writing to file
        header (astropy.io.fits.Header): Optional FITS header from the original cube to copy metadata
        precomputed_cube (array): Already-fitted residual cube to reuse instead of refitting
        instrument_hint (str): Instrument name used if the header lacks INSTRUME/TELESCOP

    Returns:
        result (array or None): Residual cube if return_cube is True, else None (writes FITS to out_path)
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
            #NBedit: Added y_data to the calculate_baseline function
            baseline = calculate_baseline(x_data[0], y1, x_data[-1], y2, x_data, y_data)
            y_corrected = y_data - baseline
            mean_val = np.nanmean(y_corrected)
            std_val = np.nanstd(y_corrected)
            # Skip if mean or std are NaN
            if np.isnan(mean_val) or np.isnan(std_val):
                continue
            guess = estimate_initial_gaussian_guess(x_data, y_corrected, std_val, mean_val)
            if guess is None:
                continue
            amp_guess, center_guess, sigma_guess = guess
            if np.nanmax(y_corrected) <= 0:
                continue
            bounds = ([0, x_data[0], 0.001 * (x_data[-1] - x_data[0])],
                      [10 * np.nanmax(y_corrected), x_data[-1], 2 * (x_data[-1] - x_data[0])])
            # Clamp initial guesses to be within bounds
            amp_guess = np.clip(amp_guess, bounds[0][0], bounds[1][0])
            center_guess = np.clip(center_guess, bounds[0][1], bounds[1][1])
            sigma_guess = np.clip(sigma_guess, bounds[0][2], bounds[1][2])
            try:
                popt, _ = curve_fit(
                    gaussian, x_data, y_corrected,
                    p0=[amp_guess, center_guess, sigma_guess],
                    bounds=bounds,
                    maxfev=1000
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
    r_squared: float  # R-squared coefficient
    success: bool  # Whether fit converged
    message: str  # Status or error message

    # Unit bookkeeping for area
    area_unit: str = ""  # Units of area, e.g. "erg/s/cm²"
    area_native: float = 0.0  # Area before any CGS conversion

    # Only populated by fit_hitran_lines_collective's simultaneous curve_fit
    amplitude_error: Optional[float] = None
    sigma_error: Optional[float] = None
    catalog_sigma: Optional[float] = None
    noise_std: Optional[float] = None

    def __str__(self):
        """Human-readable summary of fit results"""
        return (
            f"Gaussian Fit Results:\n"
            f"  Center (μ):     {self.center:.4f}\n"
            f"  Amplitude (A):  {self.amplitude:.4e}\n"
            f"  Sigma (σ):      {self.sigma:.6g}\n"
            f"  FWHM:           {self.fwhm:.6g}\n"
            f"  Area:           {self.area:.4e} {self.area_unit}\n"
            f"  R²:             {self.r_squared:.4f}\n"
            f"  Status:         {self.message}"
        )

#NBedit: Added baseline calculation for cases where y1 and y2 are NaN (happens at the edge of IFU)
def calculate_baseline(x1: float, y1: float, x2: float, y2: float, 
                       x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    """
    Calculate linear baseline between two points.
    The baseline is a straight line connecting (x1, y1) and (x2, y2):
    
    Args:
        x1, y1 (float): Left boundary point
        x2, y2 (float): Right boundary point
        x_data (array): Wavelength values where baseline is evaluated
        y_data (array): Flux values where baseline is evaluated
    
    Returns:
        baseline (array): Array of baseline values at each x_data point
    """
        
    finite_indices = np.flatnonzero(np.isfinite(y_data))

    if finite_indices.size == 0:
        return np.zeros_like(x_data)
    
    # if the new y1 and y2 go completely crazy, we could just return a flat baseline at the median of the finite y values
    # If y1 is NaN → replace with lowest x having finite y
    if np.isnan(y1):
        first_idx = finite_indices[0]
        x1 = x_data[first_idx]
        y1 = y_data[first_idx]

    # If y2 is NaN → replace with largest x having finite y
    if np.isnan(y2):
        last_idx = finite_indices[-1]
        x2 = x_data[last_idx]
        y2 = y_data[last_idx]
    
    if x2 == x1:
        # Avoid division by zero - return constant baseline
        return np.full_like(x_data, y1)

    slope = (y2 - y1) / (x2 - x1)
    baseline = y1 + slope * (x_data - x1)
    return baseline


def compute_noise_std(x_data: np.ndarray, y_data: np.ndarray, degree: int = 1) -> Optional[float]:
    """
    Compute a robust noise standard deviation from a line-free region using RANSAC polynomial fitting.

    Args:
        x_data (array): Wavelength for noise region
        y_data (array): Flux for noise region
        degree (int): Polynomial degree (1=linear, 2=quadratic, 3=cubic)

    Returns:
        std (float): Robust noise std (std of residuals from RANSAC fit), or None if insufficient data
    """

    if x_data is None or y_data is None:
        return None

    # Remove NaN values for fitting
    finite_mask = np.isfinite(y_data) & np.isfinite(x_data)
    if np.count_nonzero(finite_mask) < max(4, degree + 2):  # Need enough points for polynomial fit
        return None

    x_noise = x_data[finite_mask]
    y_noise = y_data[finite_mask]

    if len(x_noise) < max(4, degree + 2):
        return None

    try:
        ransac_pipeline = make_pipeline(
            PolynomialFeatures(degree=degree),
            RANSACRegressor(random_state=42)
        )
        
        # Fit the pipeline and calculate std of residuals
        ransac_pipeline.fit(x_noise[:, np.newaxis], y_noise)
        y_fit = ransac_pipeline.predict(x_noise[:, np.newaxis])
        residuals = y_noise - y_fit
        # Inliers only
        inlier_mask = ransac_pipeline.named_steps['ransacregressor'].inlier_mask_
        std_residuals = np.std(residuals[inlier_mask]) if np.any(inlier_mask) else np.std(residuals)

        if np.isfinite(std_residuals) and std_residuals > 0:
            return float(std_residuals)
    except Exception:
        # RANSAC failed, fall back to simple linear baseline + MAD
        # NBedit: Added y_noise to the calculate_baseline function
        baseline = calculate_baseline(x_noise[0], y_noise[0], x_noise[-1], y_noise[-1], x_noise, y_noise)
        y_corrected = y_noise - baseline
        mad = np.nanmedian(np.abs(y_corrected - np.nanmedian(y_corrected)))
        robust_std = 1.4826 * mad
        if np.isfinite(robust_std) and robust_std > 0:
            return float(robust_std)
    
    return None


def safe_nanstd(arr: np.ndarray, fallback: float = 0.0) -> float:
    """
    Safely compute nanstd without triggering "Degrees of freedom <= 0" warnings.
    
    Uses ddof=0 (population std) to avoid Bessel's correction issues with small arrays.
    Returns fallback value if array has fewer than 2 finite values.
    
    Args:
        arr (array): Input array (may contain NaNs)
        fallback (float): Value to return if insufficient finite values
    
    Returns:
        std (float): Standard deviation, or fallback if too few finite values
    """

    finite_mask = np.isfinite(arr)
    n_finite = np.count_nonzero(finite_mask)
    
    if n_finite < 2:
        return fallback
    
    # Use ddof=0 (population std) to avoid Bessel's correction warnings
    return float(np.nanstd(arr, ddof=0))


def gaussian(x: np.ndarray, amplitude: float, center: float, 
             sigma: float) -> np.ndarray:
    """
    Gaussian function.
    
    G(x) = A * exp(-(x - μ)² / (2σ²))
    
    Args:
        x (array): Independent variable (wavelength)
        amplitude (float): Peak height (A)
        center (float): Center position (μ)
        sigma (float): Width parameter (σ)
    
    Returns:
        gaussian (array): Gaussian values at each x
    """

    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))


def fit_gaussian_with_ransac(x_data: np.ndarray, y_data: np.ndarray, 
                              p0: np.ndarray, bounds: Tuple, 
                              ransac_thresh: float = 2.5,
                              ransac_iters: int = 2) -> Tuple[np.ndarray, bool, np.ndarray]:
    """
    Fit Gaussian with RANSAC outlier rejection.
    
    Uses an iterative approach: fit Gaussian, identify outliers using residuals,
    refit using only inliers. This provides robust fitting resistant to bad pixels.
    
    Args:
        x_data (array): Wavelength array
        y_data (array): Flux array
        p0 (array): Initial parameter guess [amplitude, center, sigma]
        bounds (tuple): Parameter bounds for curve_fit
        ransac_thresh (float): Threshold for outlier detection (in sigma of residuals)
        ransac_iters (int): Maximum iterations for RANSAC refinement
    
    Returns:
        fit (tuple): Tuple of (popt, success, inlier_mask)
            popt (array): Fitted parameters [amplitude, center, sigma]
            success (bool): Whether fit converged
            inlier_mask (array): Boolean mask of inlier points
    """

    # Start with all points as inliers
    inlier_mask = np.ones(len(x_data), dtype=bool)
    best_popt = p0
    best_inlier_count = 0
    
    try:
        # Initial fit with all points (reduced maxfev for speed)
        popt, pcov = curve_fit(gaussian, x_data, y_data, p0=p0, bounds=bounds, maxfev=1000)
        y_fit = gaussian(x_data, *popt)
        residuals = y_data - y_fit
        residuals_std = np.std(residuals)
        
        # Iterative outlier rejection (reduced iterations)
        for iteration in range(ransac_iters):
            # Identify outliers: points > thresh * sigma from fit
            if residuals_std > 0:
                outlier_mask = np.abs(residuals) > ransac_thresh * residuals_std
            else:
                outlier_mask = np.zeros_like(residuals, dtype=bool)
            
            inlier_mask = ~outlier_mask
            n_inliers = np.sum(inlier_mask)
            
            # Track best fit (most inliers)
            if n_inliers > best_inlier_count:
                best_inlier_count = n_inliers
                best_popt = popt.copy()
            
            # If all points are inliers, convergence reached
            if n_inliers == len(x_data):
                break
            
            # Require at least 4 inliers to refit
            if n_inliers < 4:
                break
            
            # Refit using only inliers (reduced maxfev for speed)
            try:
                popt, pcov = curve_fit(gaussian, x_data[inlier_mask], y_data[inlier_mask], 
                                      p0=popt, bounds=bounds, maxfev=1000)
                y_fit = gaussian(x_data, *popt)
                residuals = y_data - y_fit
                residuals_std = np.std(residuals[inlier_mask])
            except RuntimeError:
                # If refit fails, use best found so far
                popt = best_popt
                break
        
        # Final inlier identification with best fit
        y_fit_final = gaussian(x_data, *best_popt)
        residuals_final = y_data - y_fit_final
        residuals_std_final = np.std(residuals_final)
        if residuals_std_final > 0:
            inlier_mask_final = np.abs(residuals_final) <= ransac_thresh * residuals_std_final
        else:
            inlier_mask_final = np.ones_like(x_data, dtype=bool)
        
        return best_popt, True, inlier_mask_final
        
    except Exception as e:
        # If RANSAC fails, return initial guess with all points as inliers
        return p0, False, inlier_mask


def estimate_gaussian_params(x_data: np.ndarray, 
                             y_data: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate initial Gaussian parameters (amplitude, center, sigma) from the peak and half-width of the data.

    Args:
        x_data (array): Wavelength values
        y_data (array): Baseline-corrected flux values
    
    Returns:
        guesses (tuple): (amplitude_guess, center_guess, sigma_guess)
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


def estimate_initial_gaussian_guess(x_data: np.ndarray, y_corrected: np.ndarray,
                                    std_val: float, mean_val: float) -> Optional[Tuple[float, float, float]]:
    """
    Estimate an initial (amplitude, center, sigma) guess for a single pixel's spectrum.

    Uses a MAD-based sigma (outlier-resistant) when the signal dominates the noise, and a
    plain shape-based estimate from estimate_gaussian_params otherwise. Shared by the per-pixel
    detection pipeline (fit_gaussians_per_pixel) and the residual-cube export, which both need
    the same noise-dependent guess before running their own fit (RANSAC vs plain curve_fit).

    Args:
        x_data (array): Wavelength values
        y_corrected (array): Baseline-corrected flux values
        std_val (float): Noise standard deviation for this pixel
        mean_val (float): Mean of y_corrected, used to classify the low/high noise regime

    Returns:
        guess (tuple or None): (amp_guess, center_guess, sigma_guess), or None if estimation failed
    """
    if std_val < 0.5 * abs(mean_val):
        # Low noise case: signal dominates noise. Use robust MAD for sigma (outlier-resistant).
        mad = np.nanmedian(np.abs(y_corrected - np.nanmedian(y_corrected)))
        robust_std = 1.4826 * mad  # Convert MAD to sigma-equivalent
        sigma_for_fit = robust_std if robust_std > 0 else std_val
        try:
            amp_guess, center_guess, _ = estimate_gaussian_params(x_data, y_corrected)
            return amp_guess, center_guess, sigma_for_fit
        except Exception:
            return None
    else:
        # High noise case: noise comparable to signal. Shape-based estimate only (simple, stable).
        try:
            return estimate_gaussian_params(x_data, y_corrected)
        except Exception:
            return None


def _invalid_selection(x_data: np.ndarray, y_data: np.ndarray,
                       message: str) -> GaussianFitResult:
    """
    Failed result for a selection that cannot be fitted, with the reason.
    
    Args:
        x_data (array): Wavelength values
        y_data (array): Baseline-corrected flux values
        message (str): Result message
    
    Returns:
        fail (GaussianFitResult): Failed Gaussian
    """

    empty = np.array([])
    return GaussianFitResult(
        amplitude=0, center=0, sigma=0, fwhm=0, area=0,
        baseline_slope=0, baseline_intercept=0,
        x_data=x_data, y_data=y_data,
        baseline=empty, y_corrected=empty,
        y_fit=empty, residuals=empty,
        r_squared=0,
        success=False,
        message=message
    )


def fit_gaussian_to_line(x1: float, x2: float,
                         spectrum_x: np.ndarray,
                         spectrum_y: np.ndarray,
                         peak_x: float,
                         noise_min: Optional[float] = None,
                         noise_max: Optional[float] = None,
                         header: Optional[object] = None,
                         wave_unit: Optional[str] = None) -> GaussianFitResult:
    """
    Fit a Gaussian to a spectral line between two selected points.

    Args:
        x1 (float): Left boundary wavelength (scalar or Quantity)
        x2 (float): Right boundary wavelength (scalar or Quantity)
        spectrum_x (array): Full spectrum wavelength array (numpy array or Quantity array)
        spectrum_y (array): Full spectrum flux array
        peak_x (float): Explicit peak wavelength the user selected (scalar or Quantity)
        noise_min (float): Optional left boundary of a line-free noise region
        noise_max (float): Optional right boundary of a line-free noise region
        header (astropy.io.fits.Header): Optional FITS header for unit conversion (reads CDELT1, CDELT2)
        wave_unit (str): Wavelength unit string, used for the area's CGS conversion

    Returns:
        fit (GaussianFitResult): Fit result; on failure, success is False and message explains why
    """

    # Extract values from Quantities if needed
    x1 = _extract_value_from_quantity(x1)
    x2 = _extract_value_from_quantity(x2)
    peak_x = _extract_value_from_quantity(peak_x)
    spectrum_x = _extract_value_from_quantity(spectrum_x)
    spectrum_y = _extract_value_from_quantity(spectrum_y)
    noise_min = _extract_value_from_quantity(noise_min)
    noise_max = _extract_value_from_quantity(noise_max)
    
    # Boundaries must be given in order.
    if x1 >= x2:
        return _invalid_selection(
            np.array([]), np.array([]),
            f"Error: Left boundary ({x1:.6f}) must be less than the right "
            f"boundary ({x2:.6f}). Re-select the boundaries in order."
        )

    # ========== STEP 1: EXTRACT & BASELINE SUBTRACTION ==========
    # Extract data within selected region [x1, x2]
    mask = (spectrum_x >= x1) & (spectrum_x <= x2)
    x_data = spectrum_x[mask]
    y_data = spectrum_y[mask]
    
    # Validation checks
    if len(x_data) < 4:
        return _invalid_selection(
            x_data, y_data,
            "Error: Not enough points in selected region (need at least 4)"
        )

    # Reject peak outside boundaries
    if not (x1 <= peak_x <= x2):
        return _invalid_selection(
            x_data, y_data,
            f"Error: Peak ({peak_x:.6f}) lies outside the fit window "
            f"[{x1:.6f}, {x2:.6f}]. Move the Peak marker inside the "
            f"Left/Right boundaries."
        )

    # Get boundary flux values for baseline
    y1 = y_data[0]
    y2 = y_data[-1]
    
    # STEP 1a: Calculate linear baseline connecting endpoints
    # Baseline = y1 + slope*(x - x1), isolates emission line by removing continuum
    #NBedit: Added y_data to the calculate_baseline function
    baseline = calculate_baseline(x1, y1, x2, y2, x_data, y_data)
    baseline_slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
    
    # STEP 1b: Subtract baseline to isolate the spectral line
    # y_corrected = y_data - baseline, ready for Gaussian fitting
    y_corrected = y_data - baseline

    # Optional noise region for robust std estimation
    noise_std = None
    if noise_min is not None and noise_max is not None:
        nmin, nmax = (noise_min, noise_max) if noise_min <= noise_max else (noise_max, noise_min)
        noise_mask = (spectrum_x >= nmin) & (spectrum_x <= nmax)
        if np.count_nonzero(noise_mask) >= 2:
            noise_std = compute_noise_std(spectrum_x[noise_mask], spectrum_y[noise_mask])

    std_val_fit = safe_nanstd(y_corrected, fallback=1.0)
    std_val = noise_std if (noise_std is not None and np.isfinite(noise_std)) else std_val_fit
    
    # ========== STEP 2: ESTIMATE INITIAL PARAMETERS ==========
    # For single-pixel mode: use METHOD B (per-pixel from spectrum shape)
    # Use the user-selected peak with robust amplitude estimation
    center_guess = peak_x
    # Find nearest index to peak
    peak_idx = np.argmin(np.abs(x_data - peak_x))
    # Use window around peak (+/- 4 points) to estimate amplitude robustly
    window_start = max(0, peak_idx - 4)
    window_end = min(len(y_corrected), peak_idx + 5)
    amp_guess = np.nanmax(y_corrected[window_start:window_end])
    # For selected peak: use NARROW sigma based on local window spacing
    # This keeps Gaussian focused on just the selected peak, not spreading into empty space
    if window_end - window_start > 1:
        local_width = x_data[min(window_end-1, len(x_data)-1)] - x_data[window_start]
        sigma_guess = local_width / 2.0  # Half of the local window width
    else:
        sigma_guess = (x2 - x1) / 20.0  # Fallback: very narrow
    print(f"[DEBUG] Peak at lambda={peak_x:.4f}: amp={amp_guess:.4e}, sigma={sigma_guess:.4e}")

    # STEP 2b: Validate that a positive peak exists for fitting
    # Check if there's a meaningful peak
    if np.nanmax(y_corrected) <= 0:
        return GaussianFitResult(
            amplitude=0, center=np.mean(x_data), sigma=0, fwhm=0, area=0,
            baseline_slope=baseline_slope, baseline_intercept=y1,
            x_data=x_data, y_data=y_data,
            baseline=baseline, y_corrected=y_corrected,
            y_fit=np.zeros_like(x_data), residuals=y_corrected,
            r_squared=0,
            success=False,
            message="Error: No positive peak found after baseline subtraction"
        )
    
    # ========== STEP 3: DEFINE BOUNDS (CONSTRAINTS) ==========
    # Set up bounds to constrain fit and prevent unphysical solutions
    # Amplitude: [0, 10×max(y_corrected)] - must be positive, reasonable upper limit
    # Center: [λ_min, λ_max] - must be within selected range
    # Sigma: [reasonable_min, reasonable_max] - must be positive, not too large
    x_range = x2 - x1
    
    # If peak is explicitly selected, keep Gaussian narrow and focused
    if peak_x is not None:
        sigma_lower = 0.0001 * x_range  # Very tight lower bound
        sigma_upper = 0.15 * x_range  # Max 15% of range - much tighter focus on the peak
    else:
        sigma_lower = 0.001 * x_range
        sigma_upper = 2 * x_range  # Default: 2x range for auto-detection
    
    bounds = (
        [0, x1, sigma_lower],  # Lower bounds
        [10 * np.nanmax(y_corrected), x2, sigma_upper]  # Upper bounds
    )
    
    # STEP 3b: Clamp initial guesses to bounds (prevents "x0 is infeasible" error from curve_fit)
    amp_guess = np.clip(amp_guess, bounds[0][0], bounds[1][0])
    center_guess = np.clip(center_guess, bounds[0][1], bounds[1][1])
    sigma_guess = np.clip(sigma_guess, bounds[0][2], bounds[1][2])
    
    # ========== STEP 4: FIT GAUSSIAN WITH RANSAC OUTLIER REJECTION ==========
    try:
        # STEP 4a: Use RANSAC to iteratively fit Gaussian and reject outliers (cosmic rays, bad pixels)
        # Returns best-fit parameters [amplitude, center, sigma] and inlier mask
        popt, ransac_success, inlier_mask = fit_gaussian_with_ransac(
            x_data, y_corrected,
            p0=[amp_guess, center_guess, sigma_guess],
            bounds=bounds,
            ransac_thresh=2.0,
            ransac_iters=3
        )
        
        amplitude, center, sigma = popt
        
        # ========== STEP 5: CALCULATE FITTED PARAMETERS & DERIVED QUANTITIES ==========
        # STEP 5a: Generate fitted Gaussian curve at all wavelengths
        # y_fit = A × exp(-(λ - μ)² / (2σ²))
        y_fit = gaussian(x_data, amplitude, center, sigma)
        residuals = y_corrected - y_fit
        
        # STEP 5b: Calculate derived quantities from fitted parameters
        # FWHM = 2.355 × σ (Full Width at Half Maximum)
        fwhm = 2.355 * sigma  # FWHM = 2*sqrt(2*ln(2)) * sigma
        
        # STEP 5c: Calculate integrated area (AUC) - total flux under the Gaussian
        # Area = A × σ × √(2π) [integrated line flux formula]
        area_native = amplitude * sigma * np.sqrt(2 * np.pi)
        
        # STEP 5d: Convert area to CGS units (erg/s/cm²) if FITS header provided with pixel scales
        unit = get_wave_unit(header, wave_unit)
        area_cgs = convert_area_to_cgs(amplitude, sigma, center, header, unit) if header is not None else None
        if area_cgs is not None:
            area = area_cgs
            area_unit = "erg/s/cm²"
        else:
            # Conversion unavailable (no header, or flux is not MJy/sr). Report the
            # native area labelled with the units it is actually in, rather than
            # assuming MJy/sr.
            area = area_native
            flux_unit = (header.get('BUNIT', 'flux') if header is not None else 'flux')
            area_unit = f"{flux_unit}·{unit.lower() if unit else 'wave'}"
        
        # STEP 5e: Calculate goodness-of-fit metrics (quality indicators)
        # R² = 1 - (SS_residuals / SS_total) - measures variance explained by fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_corrected - np.mean(y_corrected))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Check fit quality
        if r_squared < 0.5:
            message = f"Warning: Poor fit quality (R² = {r_squared:.3f})"
            success = True  # Still succeeded, just warn
        else:
            message = f"Success: Good fit (R² = {r_squared:.3f})"
            success = True
        
        return GaussianFitResult(
            amplitude=amplitude, center=center, sigma=sigma,
            fwhm=fwhm, area=area, area_unit=area_unit, area_native=area_native,
            baseline_slope=baseline_slope, baseline_intercept=y1,
            x_data=x_data, y_data=y_data,
            baseline=baseline, y_corrected=y_corrected,
            y_fit=y_fit, residuals=residuals,
            r_squared=r_squared,
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
            r_squared=0,
            success=False,
            message=f"Error: Fit failed - {str(e)}"
        )


def calculate_equivalent_width(fit_result: GaussianFitResult,
                               continuum_level: Optional[float] = None) -> float:
    """
    Calculate equivalent width of the fitted line, EW = Area / F_continuum.

    Args:
        fit_result (GaussianFitResult): Result from fit_gaussian_to_line
        continuum_level (float): Mean continuum flux level. Defaults to the mean baseline

    Returns:
        ew (float): Equivalent width in wavelength units, or 0.0 if the fit failed or continuum <= 0
    """
    if not fit_result.success:
        return 0.0
    
    if continuum_level is None:
        continuum_level = np.mean(fit_result.baseline)
    
    if continuum_level <= 0:
        return 0.0
    
    # For emission lines: EW = Area / Continuum. Use the native area so numerator and
    # denominator share the same flux units - `area` may have been converted to CGS,
    # which would not divide out against a continuum in the cube's native units.
    equivalent_width = fit_result.area_native / continuum_level
    return equivalent_width


# Per-pixel outcome codes. Three states, and they partition the map: every pixel is
# a detection, an upper limit, or had no data at all. A failed fit is not a fourth
# state -- it is an upper limit whose limit came from a fit that never ran (see
# 'fit_failed', which records the reason, not a different outcome).
# Ordered worst -> best so a sequential colormap reads correctly.
STATUS_NO_DATA = 0
STATUS_UPPER_LIMIT = 1
STATUS_DETECTION = 2

STATUS_LABELS = {
    STATUS_NO_DATA: "No data",
    STATUS_UPPER_LIMIT: "Upper limit",
    STATUS_DETECTION: "Detection",
}


def pixel_fit_status(results: dict, row: int, col: int) -> int:
    """
    Return the STATUS_* code for one pixel of a fit_gaussians_per_pixel result.

    Args:
        results (dict): Output of fit_gaussians_per_pixel
        row (int): Pixel row index
        col (int): Pixel column index

    Returns:
        status (int): One of STATUS_NO_DATA, STATUS_UPPER_LIMIT, STATUS_DETECTION
    """
    if bool(results['is_detection'][row, col]):
        return STATUS_DETECTION
    if bool(results.get('is_upper_limit', np.zeros_like(results['is_detection']))[row, col]):
        return STATUS_UPPER_LIMIT
    return STATUS_NO_DATA


def fit_status_map(results: dict) -> np.ndarray:
    """
    Collapse the per-pixel boolean masks into one categorical STATUS_* map.

    Returned as float (not int) so it survives the GUI's region masking, which NaNs out
    non-boolean maps outside the selected region.

    Args:
        results (dict): Output of fit_gaussians_per_pixel

    Returns:
        status_map (array): Float array of STATUS_* codes, same shape as the input masks
    """
    det = np.asarray(results['is_detection'], dtype=bool)
    upper = np.asarray(results.get('is_upper_limit', np.zeros_like(det)), dtype=bool)

    status = np.full(det.shape, float(STATUS_NO_DATA))
    status[upper] = STATUS_UPPER_LIMIT
    status[det] = STATUS_DETECTION  # last: a detection is never also a limit
    return status


def fit_gaussians_per_pixel(cube: np.ndarray, wavelength: np.ndarray,
                            lam_min: Optional[float] = None,
                            lam_max: Optional[float] = None,
                            n_jobs: Optional[int] = None,
                            fac_above_stdres: float = 3.0,
                            noise_min: Optional[float] = None,
                            noise_max: Optional[float] = None,
                            header: Optional[object] = None,
                            wave_unit: Optional[str] = None,
                            flux_unit: Optional[str] = None,
                            lambda_0: Optional[float] = None,
                            progress_callback: Optional[callable] = None,
                            cancel_event: Optional[threading.Event] = None) -> Optional[dict]:
    """
    Fit a Gaussian to each pixel's spectrum in the cube, using the same per-pixel fit as
    fit_gaussian_to_line() but seeded by a global fit to the stacked spectrum first.

    Args:
        cube (array): 3D data cube with shape (n_wavelength, n_y, n_x)
        wavelength (array): 1D wavelength array with length n_wavelength
        lam_min (float): Minimum wavelength for the fit window. Defaults to the full range
        lam_max (float): Maximum wavelength for the fit window. Defaults to the full range
        n_jobs (int): Number of worker threads. Defaults to auto-detect CPUs
        fac_above_stdres (float): Detection threshold in units of noise std (3.0 = 3-sigma)
        noise_min (float): Optional left boundary of a line-free noise region
        noise_max (float): Optional right boundary of a line-free noise region
        header (astropy.io.fits.Header): Optional FITS header for unit conversion (reads CDELT1, CDELT2)
        wave_unit (str): Wavelength unit string
        flux_unit (str): Input flux unit (e.g. 'MJy/sr', 'Jy/beam'), used for CGS conversion validation
        lambda_0 (float): Optional line-list reference wavelength (um) to seed the global 1D fit
        progress_callback (callable): Optional progress_callback(current, total) called after each row
        cancel_event (threading.Event): Checked once per cube row; if set, fitting stops after
            in-flight rows finish and the function returns None instead of a results dict

    Returns:
        results (dict or None): None if cancelled via cancel_event, otherwise 2D parameter maps:
            'amplitude': Peak amplitude map, native flux unit (not CGS-converted)
            'center': Center wavelength map
            'sigma': Width parameter map
            'fwhm': FWHM map (2.355 * sigma)
            'area': Integrated area map, in erg/s/cm**2 if header is given, else native units
            'r_squared': Reduced chi-squared (chi2_red) fit quality map
            'success': Boolean mask of successful fits
            'is_detection': Boolean mask of true detections (amplitude > fac_above_stdres * noise)
            'is_upper_limit': Boolean mask of upper limits (failed the detection threshold)
            'fit_failed': Subset of 'is_upper_limit' whose limit came from a fit that never ran
                (too few finite samples, or parameter estimation/fit raised), not one that
                converged on something too weak
            'residuals_cube': Normalized (residual / std) residuals, cached during fitting
            'fac_above_stdres': The detection threshold that was used
            'wavelength_unit': Unit string, from the Quantity if wavelength was one, else wave_unit
            'global_fit': dict with the global 1D fit used to seed initial guesses, or None
    """
    # ========== PER-PIXEL GAUSSIAN FITTING SETUP ==========
    # Get cube dimensions
    n_wave, n_y, n_x = cube.shape
    
    # Get wavelength unit
    unit = get_wave_unit(header, wave_unit)
    
    # Extract values from Quantity if needed (handle both raw arrays and Quantities)
    wavelength_values = _extract_value_from_quantity(wavelength)
    lam_min_orig = _extract_value_from_quantity(lam_min)
    lam_max_orig = _extract_value_from_quantity(lam_max)
    noise_min_orig = _extract_value_from_quantity(noise_min)
    noise_max_orig = _extract_value_from_quantity(noise_max)
    lambda_0_orig = _extract_value_from_quantity(lambda_0)
    
    # Set wavelength range
    if lam_min_orig is None:
        lam_min_orig = wavelength_values[0]
    if lam_max_orig is None:
        lam_max_orig = wavelength_values[-1]
    
    # Ensure lam_min < lam_max
    if lam_min_orig > lam_max_orig:
        lam_min_orig, lam_max_orig = lam_max_orig, lam_min_orig
    
    # Create wavelength mask
    wave_mask = (wavelength_values >= lam_min_orig) & (wavelength_values <= lam_max_orig)
    x_data = wavelength_values[wave_mask]
    
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
    is_detection_map = np.zeros((n_y, n_x), dtype=bool)  # True if SNR threshold passed
    is_upper_limit_map = np.zeros((n_y, n_x), dtype=bool)  # True if upper limit computed
    fit_failed_map = np.zeros((n_y, n_x), dtype=bool)  # True if the fit could not be attempted/converged
    
    # Optional noise region mask for robust std estimation
    noise_wave_mask = None
    noise_x = None
    if noise_min_orig is not None and noise_max_orig is not None:
        nmin, nmax = (noise_min_orig, noise_max_orig) if noise_min_orig <= noise_max_orig else (noise_max_orig, noise_min_orig)
        noise_wave_mask = (wavelength_values >= nmin) & (wavelength_values <= nmax)
        if np.count_nonzero(noise_wave_mask) < 2:
            noise_wave_mask = None
        else:
            noise_x = wavelength_values[noise_wave_mask]

    # Initialize residuals cube (for caching during fitting)
    n_wave_sel = np.sum(wave_mask)
    residuals_cube = np.full((n_wave_sel, n_y, n_x), np.nan, dtype=np.float32)
    
    # Get boundary wavelengths
    x1 = x_data[0]
    x2 = x_data[-1]
    x_range = x2 - x1

    # ========== STEP 2 GLOBAL: METHOD A - Global 1D Fit (seed all pixels) ==========
    # Fit a Gaussian to the stacked/averaged spectrum to get initial guesses for all pixels
    # This provides consistent, fast starting parameters for per-pixel fitting
    global_guess = None
    global_fit_info = None
    try:
        # STEP 2 GLOBAL-a: Stack all spatial pixels into one average spectrum
        #NBedit: Using sum instead of mean to preserve total flux for better peak detection
        total_spectrum = np.nansum(cube[wave_mask, :, :], axis=(1, 2))
        finite_mask_total = np.isfinite(total_spectrum) & np.isfinite(x_data)
        if np.count_nonzero(finite_mask_total) >= 4:
            x_total = x_data[finite_mask_total]
            y_total = total_spectrum[finite_mask_total]
            # STEP 1 GLOBAL: Extract baseline from total spectrum endpoints
            #NBedit: Added y_total to the calculate_baseline function
            baseline_total = calculate_baseline(x_total[0], y_total[0], x_total[-1], y_total[-1], x_total, y_total)
            # STEP 1 GLOBAL: Subtract baseline from total spectrum
            y_total_corrected = y_total - baseline_total

            # STEP 2 GLOBAL-b: Estimate initial Gaussian parameters from total spectrum (METHOD B)
            # Find peak, center at peak wavelength, estimate sigma from half-width
            amp_guess_tot, center_guess_tot, sigma_guess_tot = estimate_gaussian_params(x_total, y_total_corrected)
            center_source = "total-spectrum"
            # STEP 2 GLOBAL-c: Optionally override center with line-list wavelength (METHOD A variant)
            if lambda_0_orig is not None and x_total[0] <= lambda_0_orig <= x_total[-1]:
                center_guess_tot = lambda_0_orig
                center_source = "line-list"

            # STEP 3 GLOBAL: Define bounds for global fit
            bounds_tot = (
                [0, x1, 0.001 * x_range],
                [10 * np.nanmax(y_total_corrected), x2, 2 * x_range]
            )

            # STEP 3 GLOBAL-b: Clamp guesses to bounds
            amp_guess_tot = np.clip(amp_guess_tot, bounds_tot[0][0], bounds_tot[1][0])
            center_guess_tot = np.clip(center_guess_tot, bounds_tot[0][1], bounds_tot[1][1])
            sigma_guess_tot = np.clip(sigma_guess_tot, bounds_tot[0][2], bounds_tot[1][2])

            # STEP 4 GLOBAL: Fit Gaussian to total spectrum with RANSAC outlier rejection
            popt_tot, fit_success_tot, _ = fit_gaussian_with_ransac(
                x_total, y_total_corrected,
                p0=[amp_guess_tot, center_guess_tot, sigma_guess_tot],
                bounds=bounds_tot,
                ransac_thresh=2.0,
                ransac_iters=2
            )
            # STEP 5 GLOBAL: Extract final global guess for seeding all pixels
            if fit_success_tot:
                global_guess = tuple(popt_tot)
            else:
                global_guess = (amp_guess_tot, center_guess_tot, sigma_guess_tot)

            global_fit_info = {
                "amplitude": float(global_guess[0]),  # native flux unit, e.g. MJy/sr -- not CGS-converted
                "center": float(global_guess[1]),
                "sigma": float(global_guess[2]),
                "center_source": center_source,
                "success": bool(fit_success_tot)
            }
    except Exception:
        # Global fit failed - will fall back to per-pixel METHOD B or C
        global_guess = None
        global_fit_info = None
    
    # Progress tracking
    total_pixels = n_y * n_x
    completed = 0
    
    print(f"Fitting Gaussians to {total_pixels} pixels in wavelength range [{lam_min:.4f}, {lam_max:.4f}]")
    
    # ========== MAIN PER-PIXEL FITTING LOOP ==========
    def _fit_row(row_idx: int):
        """Fit Gaussian to all pixels in one row (y=const) of the cube."""
        spectra_row = cube[wave_mask, row_idx, :]  # shape: (n_wave_sel, n_x)
        spectra_noise_row = cube[noise_wave_mask, row_idx, :] if noise_wave_mask is not None else None
        amp_row = np.full(n_x, np.nan)
        ctr_row = np.full(n_x, np.nan)
        sig_row = np.full(n_x, np.nan)
        fwhm_row = np.full(n_x, np.nan)
        area_row = np.full(n_x, np.nan)
        qual_row = np.full(n_x, np.nan)
        succ_row = np.zeros(n_x, dtype=bool)
        det_row = np.zeros(n_x, dtype=bool)  # Detection flag
        uplim_row = np.zeros(n_x, dtype=bool)  # Upper limit flag
        fail_row = np.zeros(n_x, dtype=bool)  # Fit-could-not-be-made flag (subset of uplim_row)
        residuals_row = np.full((n_wave_sel, n_x), np.nan, dtype=np.float32)

        def _mark_upper_limit(ix, std, sigma_for_area, center_for_area, fit_failed=False):
            """Record a fac_above_stdres-sigma upper limit for pixel ix, NaN-ing the fitted-parameter maps.

            Shared by every non-detection path below; only `std`/`sigma_for_area`/`center_for_area`
            (and whether the fit machinery itself broke, vs. just found nothing) differ per call site.
            """
            upper_limit_amplitude = fac_above_stdres * std
            area_native = upper_limit_amplitude * sigma_for_area * np.sqrt(2 * np.pi)
            if header is not None:
                area_cgs = convert_area_to_cgs(upper_limit_amplitude, sigma_for_area, center_for_area, header, unit, flux_unit)
                area_row[ix] = area_cgs if area_cgs is not None else area_native
            else:
                area_row[ix] = area_native
            amp_row[ix] = np.nan
            ctr_row[ix] = np.nan
            sig_row[ix] = np.nan
            fwhm_row[ix] = np.nan
            qual_row[ix] = np.nan
            succ_row[ix] = False
            uplim_row[ix] = True
            if fit_failed:
                fail_row[ix] = True

        for ix in range(n_x):
            # ===== PIXEL (row_idx, ix) FITTING START =====
            y_data = spectra_row[:, ix]
            noise_std = None
            if spectra_noise_row is not None and noise_x is not None:
                # Compute robust noise estimate from line-free region (if provided)
                noise_std = compute_noise_std(noise_x, spectra_noise_row[:, ix])

            # ===== STEP 1: EXTRACT & BASELINE SUBTRACTION =====
            # Get boundary flux values for this pixel's spectrum
            y1 = y_data[0]
            y2 = y_data[-1]
            # STEP 1a: Calculate linear baseline from endpoints
            #NBedit: Added y_data to the calculate_baseline function
            baseline = calculate_baseline(x1, y1, x2, y2, x_data, y_data)
            # STEP 1b: Subtract baseline to isolate the spectral line
            y_corrected = y_data - baseline

            # ===== DATA VALIDATION: Require minimum 5 finite points =====
            # Skip pixels with too many NaNs or masked values (prevents R\u00b2 calculation warnings)
            finite_mask = np.isfinite(y_corrected)
            n_finite = np.count_nonzero(finite_mask)

            if n_finite == 0:
                # Completely outside the FOV (all-NaN spectrum) \u2014 leave every output as NaN.
                # Do NOT assign an upper-limit area here; the noise estimate would rely on the
                # safe_nanstd fallback (1.0), which is an entirely artificial number.
                continue

            if n_finite < 5:
                # Some data but insufficient for a full fit \u2014 assign an upper limit.
                # (prevents meaningless R\u00b2 computations on near-empty spectra)
                # Use fallback=None so safe_nanstd signals "no estimate" when y_data has
                # fewer than 2 finite values (n_finite == 1).  In that case there is no
                # real noise measurement available and we must not invent one.
                std_temp = noise_std if (noise_std is not None and np.isfinite(noise_std)) else safe_nanstd(y_data, fallback=None)
                if std_temp is None or not np.isfinite(std_temp):
                    # Cannot compute a real noise estimate \u2014 pixel has no usable data,
                    # leave every output NaN (treat as outside FOV).
                    continue
                # Too few samples to fit at all, not a weak line
                _mark_upper_limit(ix, std_temp, x_range / 4.0, (x1 + x2) / 2, fit_failed=True)
                continue

            # ===== STEP 2: ESTIMATE INITIAL PARAMETERS =====
            # Compute noise level and peak statistics for parameter estimation strategy
            mean_val = np.nanmean(y_corrected)
            std_val_fit = safe_nanstd(y_corrected, fallback=1.0)
            std_val = noise_std if (noise_std is not None and np.isfinite(noise_std)) else std_val_fit

            # STEP 2 Decision: Use METHOD A (global), B (shape), or C (robust MAD)?
            if global_guess is not None:
                # Use METHOD A: Global fit parameters for all pixels (fastest)
                amp_guess, center_guess, sigma_guess = global_guess
            else:
                # Use METHOD B/C: shape-based or MAD-based guess depending on noise regime
                guess = estimate_initial_gaussian_guess(x_data, y_corrected, std_val, mean_val)
                if guess is None:
                    # Could not even form an initial guess. Only the area (an upper limit) is
                    # meaningful here -- storing the limit amplitude and a window-midpoint
                    # center would put invented numbers into a pixel that was never fitted.
                    _mark_upper_limit(ix, std_val, x_range / 4.0, (x1 + x2) / 2, fit_failed=True)
                    continue
                amp_guess, center_guess, sigma_guess = guess

            # STEP 2c: Constrain sigma estimate to be conservative
            # Default to 1-sigma (smaller/conservative) for upper limits in weak detections
            # This ensures conservative fitting and smaller intensity estimates
            sigma_guess = min(sigma_guess, 0.01 * x_range)

            # STEP 2d: Validate that a positive peak exists
            # Require positive peak
            if np.nanmax(y_corrected) <= 0:
                # Genuine non-detection, not an estimation failure
                _mark_upper_limit(ix, std_val, x_range / 4.0, (x1 + x2) / 2)
                continue

            # ===== STEP 3: DEFINE BOUNDS (CONSTRAINTS) =====
            # Set reasonable bounds to prevent unphysical fit solutions
            #NBedit: updated sigma_lower as it should not depend on x_range and center bounds they should use the global fit info
            sigma_lower = 0
            sigma_upper = 2 * x_range
            center_lower = x1
            center_upper = x2
            x_space = x_range / len(x_data)  # Average wavelength spacing
            if global_guess is not None:
                # Refine bounds based on global fit sigma estimate
                _, center_global, sigma_global = global_guess
                sigma_lower = max(sigma_lower, 0.25 * sigma_global)
                sigma_upper = min(sigma_upper, 4.0 * sigma_global)
                center_lower = max(x1, center_global - 3 * x_space)
                center_upper = min(x2, center_global + 3 * x_space)

            bounds = (
                [0.5 * np.nanmax(y_corrected[(x_data >= center_lower) & (x_data <= center_upper)]), center_lower, sigma_lower],
                [2 * np.nanmax(y_corrected[(x_data >= center_lower) & (x_data <= center_upper)]), center_upper, sigma_upper]
            )

            # STEP 3b: Clamp all initial guesses to be within bounds
            # Prevents "x0 is infeasible" error from scipy.optimize.curve_fit
            #NBedit: Updated center and sigma guesses to be informed by global fit if available
            amp_guess = np.clip(amp_guess, bounds[0][0], bounds[1][0])
            center_guess = np.clip(center_guess, bounds[0][1], bounds[1][1])
            sigma_guess = np.clip(sigma_guess, bounds[0][2], bounds[1][2])
            if global_guess is not None:
                # Further refine center guess to be near global fit center
                amp_guess = np.nanmax(y_corrected[(x_data >= center_lower) & (x_data <= center_upper)])  # Peak in local window
                center_guess = center_global
                sigma_guess = sigma_global

            # ===== STEP 4: FIT GAUSSIAN WITH RANSAC OUTLIER REJECTION =====
            try:
                # STEP 4a: Use RANSAC to fit Gaussian and reject bad pixels (2-iteration, fast)
                # Always use RANSAC for robust outlier rejection
                popt, fit_success, inlier_mask = fit_gaussian_with_ransac(
                    x_data, y_corrected,
                    p0=[amp_guess, center_guess, sigma_guess],
                    bounds=bounds,
                    ransac_thresh=2.0,  # 2-sigma threshold for outliers
                    ransac_iters=2  # Very fast: 2 iterations is enough, minimal outliers in spectral data
                )

                amplitude, center, sigma = popt

                # ===== STEP 5: CALCULATE FITTED PARAMETERS & DERIVED QUANTITIES =====
                # STEP 5a: Generate fitted Gaussian curve
                # y_fit = A \u00d7 exp(-(\u03bb - \u03bc)\u00b2 / (2\u03c3\u00b2))
                y_fit = gaussian(x_data, amplitude, center, sigma)
                residuals = y_corrected - y_fit

                # STEP 5b: Calculate FWHM (Full Width at Half Maximum)
                # FWHM = 2.355 \u00d7 \u03c3 (standard Gaussian relationship)
                fwhm = 2.355 * sigma

                # STEP 5c: Calculate integrated area (AUC) - total flux under Gaussian
                # Area = A \u00d7 \u03c3 \u00d7 \u221a(2\u03c0) [analytical integral of Gaussian]
                area_native = amplitude * sigma * np.sqrt(2 * np.pi)
                # STEP 5d: Convert area to CGS units (erg/s/cm\u00b2) if FITS header provided
                if header is not None:
                    area_cgs = convert_area_to_cgs(amplitude, sigma, center, header, unit, flux_unit)
                    area = area_cgs if area_cgs is not None else area_native
                else:
                    area = area_native

                # STEP 5e: Calculate goodness-of-fit metrics
                # \u03c7\u00b2_red = \u2211(residuals\u00b2) / (dof \u00d7 \u03c3_noise\u00b2), using the independently
                # measured noise (the mandatory noise region, or the per-pixel fallback
                # in std_val) -- NOT the residuals' own scatter. Normalizing by the
                # residuals' own std reduces algebraically to (n/dof)*(1 + mean(r)^2 /
                # std(r)^2): floored at n/dof, incapable of reflecting fit quality
                # relative to the real noise level. See the "chi2 caveat" this replaced
                # in docs/user_guide/gaussian_fitting_cube.rst.
                n_points = len(y_corrected)
                dof = max(1, n_points - 3)
                if std_val is not None and np.isfinite(std_val) and std_val > 0:
                    chi_squared_red = np.sum(residuals**2) / (dof * std_val**2)
                else:
                    chi_squared_red = np.inf

                # ===== STEP 6: APPLY DETECTION THRESHOLD =====
                # Determine if this is a true detection or upper limit
                # Check: (1) fit converged, (2) amplitude > SNR threshold, (3) fit improves residuals
                std_no_fit = std_val if (std_val is not None and np.isfinite(std_val)) else safe_nanstd(y_corrected, fallback=1.0)
                # Residual scatter after fitting, used only to ask "did the fit reduce
                # the scatter versus the raw data" -- a different question from chi2
                # above, which asks "how big are the residuals relative to real noise".
                residual_scatter = np.std(residuals, ddof=0)
                std_with_fit = residual_scatter
                fit_improves = fit_success and (std_with_fit < std_no_fit)

                # DETECTION CRITERION: two amplitude-based pathways, chi2 NOT involved.
                #
                # (A) High SNR: amplitude >= fac_above_stdres*std AND fit improves [strict]
                # (B) Weak-but-real: fit converged AND amplitude > 0.5*std [lenient]
                #
                # chi2 used to gate pathway B (require 0.5 <= chi2_red <= 2.5). It looked
                # like a quality filter but was not one: the old chi2 formula was floored
                # at n/dof (~1.1) and never observed above ~2.2, so that gate was inside
                # [0.5, 2.5] for every real pixel and never once excluded anything. The
                # detection counts this cube's science is validated against were produced
                # by pathway B being, in effect, just "converged and amplitude > 0.5*std".
                #
                # Now that chi2 above is a real, unfloored reduced chi-squared (see the
                # comment on its computation), reintroducing it here would silently make
                # it start gating detections for the first time -- a policy change, not a
                # bug fix. Keep it out: chi2 is reported as a fit-quality diagnostic, not
                # used to decide what counts as a detection.
                snr_passes_threshold = amplitude >= fac_above_stdres * std_val
                amplitude_weak = amplitude > 0.5 * std_val
                is_detection = (snr_passes_threshold and fit_improves) or (fit_success and amplitude_weak)

                if is_detection:
                    # ===== DETECTION: Store full fitted parameters =====
                    amp_row[ix] = amplitude
                    ctr_row[ix] = center
                    sig_row[ix] = sigma
                    fwhm_row[ix] = fwhm
                    area_row[ix] = area
                    qual_row[ix] = chi_squared_red
                    succ_row[ix] = True
                    det_row[ix] = True

                    # Cache normalized residuals (residual/std) for later analysis
                    std_val_for_norm = std_val if (std_val is not None and std_val > 0) else 1.0
                    normalized_residuals = residuals / std_val_for_norm
                    residuals_row[:, ix] = normalized_residuals.astype(np.float32)
                else:
                    # ===== UPPER LIMIT: Amplitude below detection threshold =====
                    # Upper limit = fac_above_stdres \u03c3 * sigma_guess * sqrt(2\u03c0)
                    _mark_upper_limit(ix, std_val, sigma_guess, center_guess if 'center_guess' in locals() else (x1 + x2) / 2)

            except Exception:
                # ===== EXCEPTION: Fitting algorithm failed =====
                fail_row[ix] = True
                if 'sigma_guess' in locals():
                    _mark_upper_limit(ix, std_val, sigma_guess, center_guess if 'center_guess' in locals() else (x1 + x2) / 2)
                else:
                    amp_row[ix] = np.nan
                    area_row[ix] = np.nan
                    ctr_row[ix] = np.nan
                    sig_row[ix] = np.nan
                    fwhm_row[ix] = np.nan
                    uplim_row[ix] = True

        return (row_idx, amp_row, ctr_row, sig_row, fwhm_row, area_row, qual_row,
                succ_row, det_row, uplim_row, fail_row, residuals_row)

    # Determine workers
    if n_jobs is None:
        try:
            cpu = os.cpu_count() or 1
        except Exception:
            cpu = 1
        n_jobs = max(1, min(cpu, 8))  # cap to avoid oversubscription

    cancelled = False

    if n_jobs <= 1:
        # Serial path
        for iy in range(n_y):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            (_, amp_row, ctr_row, sig_row, f_row, a_row, q_row,
             s_row, d_row, u_row, fl_row, res_row) = _fit_row(iy)
            amplitude_map[iy, :] = amp_row
            center_map[iy, :] = ctr_row
            sigma_map[iy, :] = sig_row
            fwhm_map[iy, :] = f_row
            area_map[iy, :] = a_row
            r_squared_map[iy, :] = q_row
            success_map[iy, :] = s_row
            is_detection_map[iy, :] = d_row
            is_upper_limit_map[iy, :] = u_row
            fit_failed_map[iy, :] = fl_row
            residuals_cube[:, iy, :] = res_row

            # Progress per row. The callback fires every row so a GUI progress bar
            # actually moves; the console print stays throttled to every 5th row so a
            # large cube does not flood the terminal.
            completed += n_x
            if progress_callback:
                progress_callback(completed, total_pixels)
            if completed % (n_x * 5) == 0:
                print(f"  Progress: {completed}/{total_pixels} pixels ({100*completed/total_pixels:.1f}%)")
    else:
        # Parallel path using ThreadPoolExecutor
        # ThreadPoolExecutor works with nested functions (unlike ProcessPoolExecutor)
        # and provides good performance for Gaussian fitting
        print(f"Using {n_jobs} worker threads ({total_pixels:,} pixels)")

        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_fit_row, iy) for iy in range(n_y)]
            for fut in as_completed(futures):
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    # Drop rows that have not started yet; the executor's own __exit__
                    # still blocks (wait=True) until whatever is already in flight
                    # finishes, so a handful of rows complete after this point.
                    for f in futures:
                        f.cancel()
                    break
                (row_idx, amp_row, ctr_row, sig_row, f_row, a_row, q_row,
                 s_row, d_row, u_row, fl_row, res_row) = fut.result()
                amplitude_map[row_idx, :] = amp_row
                center_map[row_idx, :] = ctr_row
                sigma_map[row_idx, :] = sig_row
                fwhm_map[row_idx, :] = f_row
                area_map[row_idx, :] = a_row
                r_squared_map[row_idx, :] = q_row
                success_map[row_idx, :] = s_row
                is_detection_map[row_idx, :] = d_row
                is_upper_limit_map[row_idx, :] = u_row
                fit_failed_map[row_idx, :] = fl_row
                residuals_cube[:, row_idx, :] = res_row

                # See the serial path: callback every row, console every 5th row.
                completed += n_x
                if progress_callback:
                    progress_callback(completed, total_pixels)
                if completed % (n_x * 5) == 0:
                    print(f"  Progress: {completed}/{total_pixels} pixels ({100*completed/total_pixels:.1f}%)")

    if cancelled:
        print(f"Fitting cancelled: {completed}/{total_pixels} pixels completed before stopping")
        return None

    # Final summary
    n_success = np.sum(success_map)
    n_detections = np.sum(is_detection_map)
    n_upper_limits = np.sum(is_upper_limit_map)
    n_failed = np.sum(fit_failed_map)
    print(f"Fitting complete: {n_success}/{total_pixels} successful fits ({100*n_success/total_pixels:.1f}%)")
    print(f"  Detections: {n_detections} ({100*n_detections/total_pixels:.1f}%)")
    print(f"  Upper limits: {n_upper_limits} ({100*n_upper_limits/total_pixels:.1f}%)")
    print(f"    of which the fit itself failed: {n_failed} ({100*n_failed/total_pixels:.1f}%)")

    return {
        'amplitude': amplitude_map,  # Native flux unit (e.g. MJy/sr) -- not CGS-converted
        'center': center_map,
        'sigma': sigma_map,
        'fwhm': fwhm_map,
        'area': area_map,
        'r_squared': r_squared_map,
        'success': success_map,
        'is_detection': is_detection_map,  # True detections (passed SNR threshold)
        'is_upper_limit': is_upper_limit_map,  # Upper limits (failed detection)
        'fit_failed': fit_failed_map,  # Subset of is_upper_limit: the fit itself never ran/converged
        'residuals_cube': residuals_cube,  # Cached residuals computed during fitting
        'fac_above_stdres': fac_above_stdres,  # Detection threshold used
        # Prefer the Quantity's own unit, then the wave_unit argument the caller passed.
        # Reporting 'unknown' while wave_unit was supplied is just dropping information.
        'wavelength_unit': (wavelength.unit.to_string() if isinstance(wavelength, Quantity)
                            else (wave_unit or 'unknown')),
        'global_fit': global_fit_info  # Global 1D fit used for initial guesses
    }