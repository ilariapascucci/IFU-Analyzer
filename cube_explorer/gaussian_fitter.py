from astropy.io import fits
from astropy.units import Quantity
from typing import Optional
import threading
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor

from .unit_converter import (
    integrated_flux_to_cgs, flux_density_to_cgs,
    get_wave_unit, convert_area_to_cgs, _resolve_flux_unit, _normalize_wave_unit,
)

def _extract_value_from_quantity(value, unit=None):
    """
    Extract numeric value from a Quantity object, or return raw value if not a Quantity.
    
    This helper ensures all functions can handle both:
    - Quantity objects with units (e.g., from astropy)
    - Raw numeric values (for backward compatibility)
    
    Args:
        value: Quantity, scalar, or array
        unit: Optional - expected unit (for validation/info)
    
    Returns:
        Numeric value with units stripped, or original value if not a Quantity
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


def convert_amplitude_to_cgs(amplitude_mjy_sr, center_val, header, wave_unit: Optional[str], flux_unit_input: Optional[str] = None):
    """
    Convert a Gaussian amplitude to erg/s/cm² per wavelength unit.

    An amplitude is a flux *density*, so unlike the area it keeps its spectral
    dependence: the result is erg/s/cm²/μm for a μm axis, erg/s/cm²/Å for an Å axis.
    Supports the same instruments as :func:`convert_area_to_cgs`.

    Args:
        amplitude_mjy_sr: Peak amplitude in the cube's native flux units
        center_val: Line center wavelength, in `wave_unit`
        header: FITS header supplying CDELT1/CDELT2
        wave_unit: Wavelength unit (Å, µm, km/s)
        flux_unit_input: Flux unit from the header. Falls back to BUNIT if None.

    Returns:
        float: Amplitude in erg/s/cm² per wavelength unit, or None if unsupported
    """
    if (header is None or amplitude_mjy_sr is None or
        center_val is None or not np.isfinite(center_val)):
        return None

    try:
        unit = _normalize_wave_unit(wave_unit)
        if unit == "KM/S":
            return None  # velocity axis needs rest wavelength

        return flux_density_to_cgs(
            amplitude_mjy_sr,
            _resolve_flux_unit(flux_unit_input, header),
            wavelength=center_val,
            header=header,
            wave_unit=unit,
        )
    except Exception:
        return None

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
            #NBedit: Added y_data to the calculate_baseline function
            baseline = calculate_baseline(x_data[0], y1, x_data[-1], y2, x_data, y_data)
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

from scipy.optimize import curve_fit
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Tuple, Optional
from sklearn.linear_model import RANSACRegressor


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

    # Unit bookkeeping for `area`. The CGS conversion only runs for MJy/sr input
    # with a header, so `area` is in erg/s/cm² for MIRI/NIRSpec and in native
    # flux×wavelength units otherwise. `area_unit` says which, so callers never
    # have to guess; `area_native` is always the unconverted value, which is what
    # ratios against the baseline (e.g. equivalent width) need.
    area_unit: str = ""  # Units of `area`, e.g. "erg/s/cm²"
    area_native: float = 0.0  # Area before any CGS conversion, in flux×wavelength units

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
    y_baseline(x) = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    Args:
        x1, y1: Left boundary point
        x2, y2: Right boundary point
        x_data: Wavelength values where baseline is evaluated
    
    Returns:
        baseline: Array of baseline values at each x_data point
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


def calculate_local_baseline(x_data: np.ndarray, y_data: np.ndarray, 
                             degree: int = 3) -> np.ndarray:
    """
    Calculate polynomial baseline using only the outer edges of the spectrum.
    This avoids emission lines in the middle affecting the continuum fit.
    
    Args:
        x_data: Wavelength array
        y_data: Flux array
        degree: Polynomial degree (1=linear, 2=quadratic, 3=cubic)
    
    Returns:
        baseline: Polynomial baseline evaluated at each x_data point
    """
    n = len(x_data)
    if n < 6:
        # Too few points for edge fitting, fall back to linear
        return y_data[0] + (y_data[-1] - y_data[0]) * (x_data - x_data[0]) / (x_data[-1] - x_data[0])
    
    # Use outer 20% on each side for baseline estimation (more local)
    n_edge = max(3, n // 5)
    left_indices = np.arange(n_edge)
    right_indices = np.arange(n - n_edge, n)
    edge_indices = np.concatenate([left_indices, right_indices])
    
    x_edges = x_data[edge_indices]
    y_edges = y_data[edge_indices]
    
    # Fit polynomial to edges
    coeffs = np.polyfit(x_edges, y_edges, degree)
    baseline = np.polyval(coeffs, x_data)
    
    return baseline


def compute_noise_std(x_data: np.ndarray, y_data: np.ndarray, pdeg: int = 1) -> Optional[float]:
    """
    Compute a robust noise standard deviation from a line-free region using RANSAC polynomial fitting.

    Uses RANSAC regression with polynomial features to fit the continuum, then computes
    the standard deviation of residuals. This approach is robust to outliers and handles
    curved continuum emission common in IR spectroscopy.

    Args:
        x_data: Wavelength array for the noise region
        y_data: Flux array for the noise region
        pdeg: Polynomial degree for continuum fit (1=linear, 2=quadratic, 3=cubic)

    Returns:
        Robust noise std (std of residuals from RANSAC fit), or None if insufficient data
    """
    if x_data is None or y_data is None:
        return None

    # Remove NaN values for fitting
    finite_mask = np.isfinite(y_data) & np.isfinite(x_data)
    if np.count_nonzero(finite_mask) < max(4, pdeg + 2):  # Need enough points for polynomial fit
        return None

    x_noise = x_data[finite_mask]
    y_noise = y_data[finite_mask]

    if len(x_noise) < max(4, pdeg + 2):
        return None

    try:
        # random_state fixed: RANSACRegressor's default draws from numpy's global random
        # state, so the exact same (x_data, y_data) can return a different noise_std on every
        # call -- confirmed empirically (10 calls on identical real data spanned ~148.8-150.3).
        # Nothing here benefits from that variability; fixing the seed makes this reproducible.
        ransac_pipeline = make_pipeline(
            PolynomialFeatures(degree=pdeg),
            RANSACRegressor(random_state=42)
        )
        
        # Fit the pipeline and calculate std of residuals
        ransac_pipeline.fit(x_noise[:, np.newaxis], y_noise)
        y_fit = ransac_pipeline.predict(x_noise[:, np.newaxis])
        residuals = y_noise - y_fit
        std_residuals = np.std(residuals)
        
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
        arr: Input array (may contain NaNs)
        fallback: Value to return if insufficient finite values
    
    Returns:
        Standard deviation, or fallback if too few finite values
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
        x: Independent variable (wavelength)
        amplitude: Peak height (A)
        center: Center position (μ)
        sigma: Width parameter (σ)
    
    Returns:
        Gaussian values at each x
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
        x_data: Wavelength array
        y_data: Flux array
        p0: Initial parameter guess [amplitude, center, sigma]
        bounds: Parameter bounds for curve_fit
        ransac_thresh: Threshold for outlier detection (in sigma of residuals)
        ransac_iters: Maximum iterations for RANSAC refinement
    
    Returns:
        Tuple of (popt, success, inlier_mask)
            popt: Fitted parameters [amplitude, center, sigma]
            success: Whether fit converged
            inlier_mask: Boolean mask of inlier points
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
                                      p0=popt, bounds=bounds, maxfev=800)
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


def _invalid_selection(x_data: np.ndarray, y_data: np.ndarray,
                       message: str) -> GaussianFitResult:
    """Failed result for a selection that cannot be fitted, with the reason."""
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
    
    Process:
    1. Extract data between x1 and x2
    2. Calculate linear baseline connecting endpoints
    3. Subtract baseline to isolate the line
    4. Fit Gaussian to baseline-subtracted data
    5. Calculate derived quantities (FWHM, area, etc.)
    6. Convert area to erg/s/cm² if FITS header with pixel scale is provided
    
    Supports both raw numeric arrays and astropy Quantity objects for wavelength.
    Quantities are automatically converted to raw values for internal calculations.
    
    Args:
        x1: Left boundary wavelength (scalar or Quantity)
        x2: Right boundary wavelength (scalar or Quantity)
        spectrum_x: Full spectrum wavelength array (numpy array or Quantity array)
        spectrum_y: Full spectrum flux array (numpy array)
        peak_x: Explicit peak wavelength (scalar or Quantity - user must select the peak)
        noise_min: Optional left boundary of a line-free noise region
        noise_max: Optional right boundary of a line-free noise region
        header: Optional FITS header for unit conversion (reads CDELT1, CDELT2)
    
    Returns:
        GaussianFitResult object containing all fit information
    """
    # Extract values from Quantities if needed
    x1 = _extract_value_from_quantity(x1)
    x2 = _extract_value_from_quantity(x2)
    peak_x = _extract_value_from_quantity(peak_x)
    spectrum_x = _extract_value_from_quantity(spectrum_x)
    spectrum_y = _extract_value_from_quantity(spectrum_y)
    noise_min = _extract_value_from_quantity(noise_min)
    noise_max = _extract_value_from_quantity(noise_max)
    
    # Boundaries must be given in order. This used to swap them silently, which
    # meant a mis-ordered selection produced a fit the user never actually asked
    # for; make the caller fix the input instead.
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

    # A peak guess outside the window used to be clamped to the nearest boundary,
    # which returned a confident-looking fit with the centre pinned to the edge and
    # a negative R^2. Reject it: there is no meaningful fit to give.
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
    if np.max(y_corrected) <= 0:
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
        [10 * np.max(y_corrected), x2, sigma_upper]  # Upper bounds
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



def auto_detect_boundaries(peak_x: float, spectrum_x: np.ndarray, spectrum_y: np.ndarray, 
                           continuum_y: Optional[np.ndarray] = None, threshold_percent: float = 30.0) -> tuple:
    """
    Auto-detect LEFT and RIGHT boundaries for a line around a peak.
    
    Simple algorithm: Find the peak, then go wavelength-by-wavelength left and right
    until the slope changes direction (indicating a valley between peaks).
    
    Args:
        peak_x: Line center wavelength (peak location)
        spectrum_x: Full wavelength array
        spectrum_y: Full flux array
        continuum_y: Optional continuum array (will subtract before analysis)
        threshold_percent: Unused (kept for compatibility)
    
    Returns:
        (left_boundary, right_boundary) tuple
    """
    # Use continuum-subtracted if provided
    if continuum_y is not None:
        spectrum_to_analyze = spectrum_y - continuum_y
    else:
        spectrum_to_analyze = spectrum_y
    
    # Find index nearest to peak
    peak_idx = np.argmin(np.abs(spectrum_x - peak_x))
    
    # Get peak height
    peak_height = spectrum_to_analyze[peak_idx]
    
    if peak_height <= 0:
        # Peak is below baseline, use fixed width
        window = 0.03
        return peak_x - window, peak_x + window
    
    # Search LEFT: go backwards from peak, stop when slope changes direction (starts going up)
    left_idx = peak_idx
    for i in range(peak_idx - 1, max(0, peak_idx - 80), -1):
        # Check slope: is it going up (positive slope)?
        slope = spectrum_to_analyze[i] - spectrum_to_analyze[i + 1]
        if slope > 0:  # Going UP = we found the valley
            left_idx = i
            break
    else:
        # No slope change found, use bounded fallback
        left_idx = max(0, peak_idx - 80)
    
    # Search RIGHT: go forwards from peak, stop when slope changes direction (starts going up)
    right_idx = peak_idx
    for i in range(peak_idx + 1, min(len(spectrum_x), peak_idx + 80)):
        # Check slope: is it going up (positive slope)?
        slope = spectrum_to_analyze[i] - spectrum_to_analyze[i - 1]
        if slope > 0:  # Going UP = we found the valley
            right_idx = i
            break
    else:
        # No slope change found, use bounded fallback
        right_idx = min(len(spectrum_x) - 1, peak_idx + 80)
    
    # Get wavelength values
    left_boundary = spectrum_x[left_idx]
    right_boundary = spectrum_x[right_idx]
    
    # Ensure we have at least some width
    min_width = 0.01
    if right_boundary - left_boundary < min_width:
        center = peak_x
        left_boundary = center - min_width / 2
        right_boundary = center + min_width / 2
    
    return left_boundary, right_boundary


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
    
    # For emission lines: EW = Area / Continuum. Use the native area so numerator and
    # denominator share the same flux units - `area` may have been converted to CGS,
    # which would not divide out against a continuum in the cube's native units.
    equivalent_width = fit_result.area_native / continuum_level
    return equivalent_width


def fit_gaussians_per_pixel(cube: np.ndarray, wavelength: np.ndarray,
                            lam_min: Optional[float] = None,
                            lam_max: Optional[float] = None,
                            n_jobs: Optional[int] = None,
                            fac_above_stdres: float = 2.0,
                            noise_min: Optional[float] = None,
                            noise_max: Optional[float] = None,
                            header: Optional[object] = None,
                            wave_unit: Optional[str] = None,
                            flux_unit: Optional[str] = None,
                            lambda_0: Optional[float] = None,
                            store_wavelength_unit: bool = True,
                            progress_callback: Optional[callable] = None,
                            cancel_event: Optional[threading.Event] = None) -> Optional[dict]:
    """
    Fit a Gaussian to each pixel's spectrum in the cube within a wavelength range.
    
    This function performs the same Gaussian fitting process used in fit_gaussian_to_line(),
    but applies it to every spatial pixel in the cube. For each pixel:
    1. Extract spectrum within [lam_min, lam_max]
    2. Calculate linear baseline from endpoints
    3. Subtract baseline
    4. Fit Gaussian using same initial guesses and bounds
    5. Apply SNR-based detection threshold
    6. Calculate area using analytic formula: A * σ * sqrt(2π)
    7. Compute 1σ upper limits for non-detections
    
    Args:
        cube: 3D data cube with shape (n_wavelength, n_y, n_x)
        wavelength: 1D wavelength array with length n_wavelength
        lam_min: Minimum wavelength for fitting region (default: use full range)
        lam_max: Maximum wavelength for fitting region (default: use full range)
        n_jobs: Number of worker threads (default: auto-detect CPUs)
        fac_above_stdres: Detection threshold in units of noise std (default: 2.0 for 2σ)
        noise_min: Optional left boundary of a line-free noise region
        noise_max: Optional right boundary of a line-free noise region
        header: Optional FITS header for unit conversion (reads CDELT1, CDELT2)
        wave_unit: Wavelength unit string
        flux_unit: Input flux unit (e.g., 'MJy/sr', 'Jy/beam') - used for CGS conversion validation
        lambda_0: Optional line-list reference wavelength (µm) to seed global 1D fit
        store_wavelength_unit: Whether to store wavelength unit in results (default: True)
        progress_callback: Optional callback function with signature progress_callback(current, total) for progress reporting
        cancel_event: Optional threading.Event. Checked once per cube row; if set, fitting
            stops after the in-flight rows finish and the function returns None instead of
            a results dict. Lets a caller (e.g. a closing GUI window) abort a run that would
            otherwise keep computing and printing progress after nothing is listening.

    Note:
        RANSAC outlier rejection is always enabled with 3 iterations for robust, fast fitting.
    
    Returns:
        Dictionary containing 2D parameter maps:
            'amplitude': Peak amplitude map
            'center': Center wavelength map
            'sigma': Width parameter map
            'fwhm': FWHM map (2.355 * sigma)
            'area': Integrated area under Gaussian (AUC) map in erg/s/cm² (if header provided) or MJy/sr·μm
            'r_squared': Reduced chi-squared (χ²_red) fit quality map
            'success': Boolean mask of successful fits
            'is_detection': Boolean mask indicating true detections (amplitude > fac_above_stdres * noise)
            'is_upper_limit': Boolean mask indicating upper limits (failed detection threshold)
    
    Notes:
        - Detections: amplitude >= fac_above_stdres * std_residuals AND fit improves residuals
        - Upper limits: For non-detections, area = std_residuals * sigma_guess * sqrt(2π) (1σ limit)
        - Failed fits store NaN in parameter maps
        - Area calculated analytically: A * σ * sqrt(2π), then converted to erg/s/cm² if header provided
        - Fit quality: χ²_red ≈ 1 is good, >>1 is poor, <<1 is overfit
        - Supports astropy Quantity objects for wavelength (automatically extracts values)
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
                "amplitude": float(global_guess[0]),
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
        residuals_row = np.full((n_wave_sel, n_x), np.nan, dtype=np.float32)

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
            # Skip pixels with too many NaNs or masked values (prevents R² calculation warnings)
            finite_mask = np.isfinite(y_corrected)
            n_finite = np.count_nonzero(finite_mask)

            if n_finite == 0:
                # Completely outside the FOV (all-NaN spectrum) — leave every output as NaN.
                # Do NOT assign an upper-limit area here; the noise estimate would rely on the
                # safe_nanstd fallback (1.0), which is an entirely artificial number.
                continue

            if n_finite < 5:
                # Some data but insufficient for a full fit — assign an upper limit.
                # (prevents meaningless R² computations on near-empty spectra)
                # Use fallback=None so safe_nanstd signals "no estimate" when y_data has
                # fewer than 2 finite values (n_finite == 1).  In that case there is no
                # real noise measurement available and we must not invent one.
                std_temp = noise_std if (noise_std is not None and np.isfinite(noise_std)) else safe_nanstd(y_data, fallback=None)
                if std_temp is None or not np.isfinite(std_temp):
                    # Cannot compute a real noise estimate — pixel has no usable data,
                    # leave every output NaN (treat as outside FOV).
                    continue
                upper_limit_amplitude = fac_above_stdres * std_temp
                upper_limit_area_native = upper_limit_amplitude * (x_range / 4.0) * np.sqrt(2 * np.pi)
                if header is not None:
                    area_cgs = convert_area_to_cgs(
                        upper_limit_amplitude,
                        x_range / 4.0,
                        (x1 + x2) / 2,
                        header,
                        unit,
                        flux_unit
                    )
                    upper_limit_area = area_cgs if area_cgs is not None else upper_limit_area_native
                else:
                    upper_limit_area = upper_limit_area_native
                amp_row[ix] = np.nan
                area_row[ix] = upper_limit_area
                ctr_row[ix] = np.nan
                sig_row[ix] = np.nan
                fwhm_row[ix] = np.nan
                uplim_row[ix] = True
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
                # Low noise case: signal dominates noise
                if std_val < 0.5 * abs(mean_val):
                    # Use METHOD C: Robust MAD for sigma (outlier-resistant)
                    # MAD = Median Absolute Deviation, robust to cosmic rays
                    mad = np.nanmedian(np.abs(y_corrected - np.nanmedian(y_corrected)))
                    robust_std = 1.4826 * mad  # Convert MAD to sigma-equivalent
                    sigma_for_fit = robust_std if robust_std > 0 else std_val
                    try:
                        # Use METHOD B: Estimate amplitude and center from peak/shape
                        amp_guess, center_guess, _ = estimate_gaussian_params(x_data, y_corrected)
                        # Override sigma with robust MAD estimate
                        sigma_guess = sigma_for_fit
                    except Exception:
                        # Assign fac_above_stdres σ upper limit for failed parameter estimation
                        upper_limit_amplitude = fac_above_stdres * std_val
                        upper_limit_area_native = upper_limit_amplitude * (x_range / 4.0) * np.sqrt(2 * np.pi)
                        # Convert to erg/s/cm² if header available
                        if header is not None:
                            area_cgs = convert_area_to_cgs(
                                upper_limit_amplitude,
                                x_range / 4.0,
                                (x1 + x2) / 2,
                                header,
                                unit,
                                flux_unit
                            )
                            upper_limit_area = area_cgs if area_cgs is not None else upper_limit_area_native
                        else:
                            upper_limit_area = upper_limit_area_native
                        # For upper limit: only store area, set other parameters to NaN
                        amp_row[ix] = np.nan
                        area_row[ix] = upper_limit_area
                        ctr_row[ix] = np.nan
                        sig_row[ix] = np.nan
                        fwhm_row[ix] = np.nan
                        uplim_row[ix] = True
                        continue
                else:
                    # High noise case: noise comparable to signal
                    # Use METHOD B only: shape-based parameter estimation (simple, stable)
                    try:
                        amp_guess, center_guess, sigma_guess = estimate_gaussian_params(x_data, y_corrected)
                    except Exception:
                        # Assign fac_above_stdres σ upper limit for failed parameter estimation
                        upper_limit_amplitude = fac_above_stdres * std_val
                        upper_limit_area_native = upper_limit_amplitude * (x_range / 4.0) * np.sqrt(2 * np.pi)
                        # Convert to erg/s/cm² if header available
                        if header is not None:
                            area_cgs = convert_area_to_cgs(
                                upper_limit_amplitude,
                                x_range / 4.0,
                                (x1 + x2) / 2,
                                header,
                                unit,
                                flux_unit
                            )
                            upper_limit_area = area_cgs if area_cgs is not None else upper_limit_area_native
                        else:
                            upper_limit_area = upper_limit_area_native
                        amp_row[ix] = upper_limit_amplitude
                        area_row[ix] = upper_limit_area
                        ctr_row[ix] = (x1 + x2) / 2
                        sig_row[ix] = x_range / 4.0
                        fwhm_row[ix] = 2.355 * (x_range / 4.0)
                        uplim_row[ix] = True
                        continue
            
            # STEP 2c: Constrain sigma estimate to be conservative
            # Default to 1-sigma (smaller/conservative) for upper limits in weak detections
            # This ensures conservative fitting and smaller intensity estimates
            sigma_guess = min(sigma_guess, 0.01 * x_range)

            # STEP 2d: Validate that a positive peak exists
            # Require positive peak
            if np.max(y_corrected) <= 0:
                # Assign fac_above_stdres σ upper limit for pixels with no positive peak
                upper_limit_amplitude = fac_above_stdres * std_val
                upper_limit_area_native = upper_limit_amplitude * (x_range / 4.0) * np.sqrt(2 * np.pi)
                # Convert to erg/s/cm² if header available
                if header is not None:
                    area_cgs = convert_area_to_cgs(
                        upper_limit_amplitude,
                        x_range / 4.0,
                        (x1 + x2) / 2,
                        header,
                        unit,
                        flux_unit
                    )
                    upper_limit_area = area_cgs if area_cgs is not None else upper_limit_area_native
                else:
                    upper_limit_area = upper_limit_area_native
                # For upper limit: only store area, set other parameters to NaN
                amp_row[ix] = np.nan
                area_row[ix] = upper_limit_area
                ctr_row[ix] = np.nan
                sig_row[ix] = np.nan
                fwhm_row[ix] = np.nan
                uplim_row[ix] = True
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
                # y_fit = A × exp(-(λ - μ)² / (2σ²))
                y_fit = gaussian(x_data, amplitude, center, sigma)
                residuals = y_corrected - y_fit

                # STEP 5b: Calculate FWHM (Full Width at Half Maximum)
                # FWHM = 2.355 × σ (standard Gaussian relationship)
                fwhm = 2.355 * sigma
                
                # STEP 5c: Calculate integrated area (AUC) - total flux under Gaussian
                # Area = A × σ × √(2π) [analytical integral of Gaussian]
                area_native = amplitude * sigma * np.sqrt(2 * np.pi)
                # STEP 5d: Convert area to CGS units (erg/s/cm²) if FITS header provided
                if header is not None:
                    area_cgs = convert_area_to_cgs(amplitude, sigma, center, header, unit, flux_unit)
                    area = area_cgs if area_cgs is not None else area_native
                else:
                    area = area_native

                # STEP 5e: Calculate goodness-of-fit metrics
                # χ²_red = ∑(residuals²) / (dof × σ_noise²), using the independently
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
                    # Compute fac_above_stdres σ upper limit when detection fails
                    # Upper limit = fac_above_stdres σ * sigma_guess * sqrt(2π)
                    upper_limit_amplitude = fac_above_stdres * std_val
                    upper_limit_area_native = upper_limit_amplitude * sigma_guess * np.sqrt(2 * np.pi)
                    # Convert to erg/s/cm² if header available
                    if header is not None:
                        area_cgs = convert_area_to_cgs(
                            upper_limit_amplitude,
                            sigma_guess,
                            center_guess if 'center_guess' in locals() else (x1 + x2) / 2,
                            header,
                            unit,
                            flux_unit
                        )
                        upper_limit_area = area_cgs if area_cgs is not None else upper_limit_area_native
                    else:
                        upper_limit_area = upper_limit_area_native
                    
                    # For upper limit: only store area, set other parameters to NaN
                    amp_row[ix] = np.nan
                    ctr_row[ix] = np.nan
                    sig_row[ix] = np.nan
                    fwhm_row[ix] = np.nan
                    area_row[ix] = upper_limit_area  # fac_above_stdres σ upper limit
                    qual_row[ix] = np.nan
                    succ_row[ix] = False
                    uplim_row[ix] = True  # Mark as upper limit
                    
            except Exception:
                # ===== EXCEPTION: Fitting algorithm failed =====
                # Compute fac_above_stdres σ upper limit using initial guesses
                # Case: Fit not converged - assign conservative upper limit
                if 'sigma_guess' in locals():
                    upper_limit_amplitude = fac_above_stdres * std_val
                    upper_limit_area_native = upper_limit_amplitude * sigma_guess * np.sqrt(2 * np.pi)
                    # Convert to erg/s/cm² if header available
                    if header is not None:
                        area_cgs = convert_area_to_cgs(
                            upper_limit_amplitude,
                            sigma_guess,
                            center_guess if 'center_guess' in locals() else (x1 + x2) / 2,
                            header,
                            unit,
                            flux_unit
                        )
                        upper_limit_area = area_cgs if area_cgs is not None else upper_limit_area_native
                    else:
                        upper_limit_area = upper_limit_area_native
                else:
                    upper_limit_amplitude = np.nan
                    upper_limit_area = np.nan
                
                amp_row[ix] = np.nan
                area_row[ix] = upper_limit_area
                ctr_row[ix] = np.nan
                sig_row[ix] = np.nan
                fwhm_row[ix] = np.nan
                uplim_row[ix] = True

        return row_idx, amp_row, ctr_row, sig_row, fwhm_row, area_row, qual_row, succ_row, det_row, uplim_row, residuals_row

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
            _, amp_row, ctr_row, sig_row, f_row, a_row, q_row, s_row, d_row, u_row, res_row = _fit_row(iy)
            amplitude_map[iy, :] = amp_row
            center_map[iy, :] = ctr_row
            sigma_map[iy, :] = sig_row
            fwhm_map[iy, :] = f_row
            area_map[iy, :] = a_row
            r_squared_map[iy, :] = q_row
            success_map[iy, :] = s_row
            is_detection_map[iy, :] = d_row
            is_upper_limit_map[iy, :] = u_row
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
                row_idx, amp_row, ctr_row, sig_row, f_row, a_row, q_row, s_row, d_row, u_row, res_row = fut.result()
                amplitude_map[row_idx, :] = amp_row
                center_map[row_idx, :] = ctr_row
                sigma_map[row_idx, :] = sig_row
                fwhm_map[row_idx, :] = f_row
                area_map[row_idx, :] = a_row
                r_squared_map[row_idx, :] = q_row
                success_map[row_idx, :] = s_row
                is_detection_map[row_idx, :] = d_row
                is_upper_limit_map[row_idx, :] = u_row
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
    print(f"Fitting complete: {n_success}/{total_pixels} successful fits ({100*n_success/total_pixels:.1f}%)")
    print(f"  Detections: {n_detections} ({100*n_detections/total_pixels:.1f}%)")
    print(f"  Upper limits: {n_upper_limits} ({100*n_upper_limits/total_pixels:.1f}%)")

    # Convert amplitude to CGS if header available (MJy/sr -> erg/s/cm²/sr)
    amplitude_map_cgs = amplitude_map.copy()
    if header is not None:
        for iy in range(n_y):
            for ix in range(n_x):
                amp_val = amplitude_map[iy, ix]
                center_val = center_map[iy, ix]
                if np.isfinite(amp_val) and np.isfinite(center_val):
                    amp_cgs = convert_amplitude_to_cgs(amp_val, center_val, header, unit, flux_unit)
                    if amp_cgs is not None:
                        amplitude_map_cgs[iy, ix] = amp_cgs

    return {
        'amplitude': amplitude_map_cgs,  # CGS units (erg/s/cm²/sr) if available, else native (MJy/sr)
        'amp_originalunits': amplitude_map,  # Original units (MJy/sr)
        'center': center_map,
        'sigma': sigma_map,
        'fwhm': fwhm_map,
        'area': area_map,
        'r_squared': r_squared_map,
        'success': success_map,
        'is_detection': is_detection_map,  # True detections (passed SNR threshold)
        'is_upper_limit': is_upper_limit_map,  # Upper limits (failed detection)
        'residuals_cube': residuals_cube,  # Cached residuals computed during fitting
        'fac_above_stdres': fac_above_stdres,  # Detection threshold used
        # Prefer the Quantity's own unit, then the wave_unit argument the caller passed.
        # Reporting 'unknown' while wave_unit was supplied is just dropping information.
        'wavelength_unit': (wavelength.unit.to_string() if isinstance(wavelength, Quantity)
                            else (wave_unit or 'unknown')),
        'global_fit': global_fit_info  # Global 1D fit used for initial guesses
    }

'''
The Fitting Process for Each Pixel
For each spatial pixel (y, x) in your image, here's what happens:

Step 1: Extract & Baseline Subtraction
Extract the spectrum at that pixel within your wavelength range [λ_min, λ_max]
Calculate a linear baseline connecting the two endpoints (edges of your fitting region)
Subtract baseline: y_corrected = y_data - baseline
This isolates the emission/absorption line from the continuum.

Step 2: Estimate Initial Parameters
Three strategies create starting guesses:

┌─ Check if Global 1D Fit succeeded?
│  ├─ YES → Use global guess for ALL pixels ✓
│  │        (Methods B & C skipped)
│  │
│  └─ NO → Check pixel noise level
│     ├─ LOW noise (std < 0.5*|mean|)
│     │  → Use Method C (MAD robust) + Part of Method B
│     │
│     └─ HIGH noise
│        → Use Method B (per-pixel shape only)

Method A - Global 1D fit (seed all pixels):
Stack all spectra in the cube into one average spectrum
Fit a Gaussian to that
Use as initial guess for every pixel (faster convergence)

Method B - Per-pixel from spectrum shape:
Find the peak: amplitude_guess = max(y_corrected)
Peak location: center_guess = wavelength[argmax(y_corrected)]
Width estimate: Find points above half-max, measure their spread, convert: sigma_guess = FWHM / 2.355

Method C - Robust when noisy:
If data is noisy, use median absolute deviation (MAD): robust_std = 1.4826 * MAD(y_corrected)
Use that for sigma instead of RMS

Scenario	                    What's Used	                            Why
Global fit worked	            Method A only	                        Fast, consistent, one fit guides all pixels
Global fit failed + low noise	Method C (σ) + Method B (A, center)     Robust sigma from noise, amplitudes from peaks
Global fit failed + high noise	Method B only	                        Noise too high to estimate robust std

Step 3: Define Bounds (Constraints)
Prevents unphysical fits:
Amplitude:  [0, 10*max(y_corrected)]
Center:     [λ_min, λ_max]           ← must be in fitting region
Sigma:      [0.001*(λ_max-λ_min), 2*(λ_max-λ_min)]  ← reasonable width

STEP 3b: Conservative sigma constraint
After estimating initial sigma, apply: sigma_guess = min(sigma_guess, 0.01*(λ_max-λ_min))
This keeps weak detections conservative, avoiding overestimated line widths

Step 4: Fit with RANSAC Outlier Rejection
Uses iterative refinement:

Initial fit: Use scipy's curve_fit() with all points
Find outliers: Points where |residual| > 2σ_residual are bad pixels
Refit: Use only inlier points
Repeat 2-3 times until convergence
This handles cosmic rays, bad pixels, and noise robustly.

Step 5: Calculate Fitted Parameters
The Gaussian function:
G(λ)=A⋅exp(-(λ−μ)^2/2σ^2)

From curve_fit() you get: A (amplitude), μ (center), σ (sigma)

Then derive:
FWHM=2.355*σ
Area/AUC (integrated flux under the Gaussian):
Area =∫−∞;∞ G(λ)dλ=A*σ*sqrt(2π)

This is the total flux contained in the line.

Step 6: Apply Detection Threshold (HYBRID CRITERION)
⚠️ UPDATED: Uses TWO independent pathways (either pathway marks as detection):

PATHWAY A - High SNR detection (strict):
✓ amplitude >= fac_above_stdres * std_noise (default 2.0σ)  AND
✓ fit_improves = std_residuals < std_before_fit
Both conditions must be TRUE

PATHWAY B - Quality-based weak signal detection (allows lower SNR):
✓ χ²_red ∈ [0.5, 2.5] (fit quality near 1.0)  AND
✓ amplitude > 0.5 * std_noise (weak but real signals)
Catches lines that are statistically weak but clearly detected in excellent fits

DETECTION = PATHWAY A OR PATHWAY B

Example:
- Clean data, strong line (SNR=5σ) → both pathways triggered → DETECTED
- Noisy data, barely visible line (SNR=10mσ, χ²_red=0.8) → pathway B triggers → DETECTED
- Flat noise, random spike (SNR=1.5σ, χ²_red=10) → no pathway triggers → UPPER LIMIT

Why two pathways? Because good fits (χ²_red ~ 1) with moderate amplitude are physically real even if SNR < 2σ.
This captures weak emission lines in noisy data while avoiding random noise artifacts.

The Five Parameter Maps
1. Amplitude Map
What it is: Peak height of the Gaussian at each pixel
Units: MJy/sr (or erg/s/cm²/sr if converted)
Calculation: Direct output from curve_fit, called A
Meaning: Says line intensity but ignores width—a narrow tall line and broad short line can have same area but different amplitudes

2. Center Map
What it is: The wavelength μ where the Gaussian is centered
Units: μm or Å
Calculation: Direct output from curve_fit, called μ
Physical meaning:
Detects motion/velocity shifts (Doppler shift)
Bluer values = object moving toward you
Redder values = object moving away
Use case: See rotation in galaxies, measure wind outflows, find shock waves

3. Sigma Map
What it is: The width parameter σ of the Gaussian
Units: Same as wavelength (μm or Å)
Calculation: Direct output from curve_fit, called σ
Physical meaning:
Temperature: Hotter gas = more thermal broadening = larger σ
Turbulence: Chaotic motion broadens lines
Rotation: If there's velocity structure, lines broaden
Use case: Find temperature maps, locate shock regions, detect turbulence

4. FWHM (Full Width at Half Maximum) Map
What it is: Line width at 50% of peak height
Units: Same as wavelength (μm or Å)
Formula: FWHM=2.355*σ
Relationship to sigma: Always exactly 2.355 times wider
Use case: Equivalent to sigma but more intuitive (FWHM is what you measure from line profile plots)

5. Area Map (Most Important)
What it is: Total flux integrated over the entire Gaussian profile
Formula: Area=A×σ×sqrt(2π) 
Units:
Native: MJy/sr·μm
Converted: erg/s/cm² (if FITS header provided)
Calculation steps:
Compute A*σ*sqrt(2π) in native units
If FITS header has CDELT1, CDELT2 (pixel scales):
Convert pixel solid angle: conv = CDELT1*CDELT2*1e6/(180/π)²
Convert frequency: c / λ² wavelength-to-frequency conversion
Result: area = area_native*conv*(c/λ²)*1e-23 ← erg/s/cm²

Bright vs Dark in Area Map:
🟦 White/Bright: Strong emission from that line → lots of total energy
⬛ Black/Dark: Weak/no emission → little energy
This shows the spatial distribution of your emission

Handling Detections vs Upper Limits
Not every pixel has a detected line. Your code applies two flags:

✅ DETECTION (amplitude > threshold OR fit quality good):
If EITHER pathway A OR pathway B triggers:
Stores full parameters: amplitude, center, sigma, fwhm, area, χ²_red
Marked as: is_detection = True

⚠️ UPPER LIMIT (amplitude below threshold AND fit quality poor):
If NEITHER pathway A NOR pathway B triggers:
Computes 2σ upper limit: area_upper_limit = (fac_above_stdres*std) * sigma_guess * sqrt(2π)
(default: 2σ upper limit using initial sigma estimate)
Other parameters stored as NaN (no true fit achieved)
Marked as: is_upper_limit = True

📊 DATA VALIDATION:
Pixels with <5 finite data points (too many NaNs/masked regions):
Automatically assigned upper limits WITHOUT attempting fit
Prevents meaningless χ²/R² calculations on near-empty spectra
Still provides magnitude limits for incomplete data

Use case: In weak sources, you can still constrain maximum possible flux
The Conversion to CGS (erg/s/cm²)
This is handled by gaussian_fitter.py:66:

For MIRI/NIRSpec data in MJy/sr:

Native area: A*σ*sqrt(2π) → MJy/sr·μm

Pixel solid angle:
Convert angular pixels to steradians: conv = CDELT1*CDELT2*1e6/(180/π)²
Result: Jy·μm

Flux calibration:
Wavelength-to-frequency conversion: F_ν = F_λ*(c / λ²)
Jansky to CGS: 1 Jy = 1e-23 erg/s/cm²/Hz
Final: erg/s/cm²

For other instruments (ALMA Jy/beam, MUSE erg/cm²/s/Å):
Returns None—uses native units instead since conversion is ambiguous

Robust noise estimation:
1. If line-free noise region provided: Uses compute_noise_std() with RANSAC polynomial fitting
2. If not provided: Falls back to safe_nanstd(y_corrected) with conservative estimates
3. Handles cases where std < 0.5*|mean| using Median Absolute Deviation (MAD):
   robust_std = 1.4826 * MAD(y_corrected)
   More resistant to outliers/cosmic rays than simple RMS

Quality Metrics (R² Map)

Chi-squared (reduced):
χ_red^2=∑(residuals^2)/(degrees of freedom)*σ_noise^2

Interpretation:
~1.0 (green): Perfect fit
0.5-0.8 (yellow): Acceptable fit
>2 (red): Poor fit—line may be non-Gaussian (double-peaked, absorption artifact, etc.)

Summary: One Map, What It Shows
Map	        Gives You	            How It's Calculated
Amplitude	Line strength	        Peak of fitted Gaussian (A)
Center	    Doppler shift/velocity	Center of fitted Gaussian (μ)
Sigma	    Line broadening	        Width parameter of fitted Gaussian (σ)
FWHM	    Intuitive line width	2.355*σ
Area        Total flux spatially    A*σ*sqrt(2π) → CGS via wavelength-frequency conversion
'''