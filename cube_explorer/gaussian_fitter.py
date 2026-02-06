from astropy.io import fits
import astropy.units as u
from typing import Optional
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor

def get_pixel_solid_angle(header):
    """
    Calculate the solid angle per spaxel (pixel) from FITS header.
    
    Reads CDELT1 and CDELT2 (pixel scale in degrees) from the header and converts
    to solid angle in steradians.
    
    Args:
        header: astropy.io.fits.Header object
        
    Returns:
        float: Solid angle per pixel in steradians, or None if unable to calculate
    """
    if header is None:
        return None
    
    try:
        # Read pixel scales in degrees per pixel
        cdelt1 = header.get('CDELT1', None)
        cdelt2 = header.get('CDELT2', None)
        
        if cdelt1 is None or cdelt2 is None:
            return None
        
        # Convert degrees to arcseconds
        cdelt1_arcsec = abs(float(cdelt1)) * 3600.0  # deg to arcsec
        cdelt2_arcsec = abs(float(cdelt2)) * 3600.0  # deg to arcsec
        
        # Convert arcseconds to radians and calculate solid angle
        # 1 arcsec = (π/180/3600) radians
        arcsec_to_rad = np.pi / 180.0 / 3600.0
        omega_sr = (cdelt1_arcsec * arcsec_to_rad) * (cdelt2_arcsec * arcsec_to_rad)
        
        return omega_sr
        
    except (KeyError, ValueError, TypeError):
        return None

def _normalize_wave_unit(wave_unit: Optional[str]) -> Optional[str]:
    if not wave_unit:
        return None
    unit = wave_unit.strip().upper()
    if unit in ["UM", "MICRON", "MICRONS", "µM", "ΜM"]:
        return "UM"
    if unit in ["ANGSTROM", "A", "Å"]:
        return "ANGSTROM"
    if unit in ["KM/S", "KMS", "KM S-1", "KM S^-1"]:
        return "KM/S"
    return unit

def get_wave_unit(header, fallback: Optional[str] = None) -> Optional[str]:
    if header is not None:
        header_unit = header.get("CUNIT3", None)
        norm = _normalize_wave_unit(header_unit)
        if norm is not None:
            return norm
    return _normalize_wave_unit(fallback)

def convert_area_to_cgs(amplitude_mjy_sr, sigma_val, center_val, solid_angle_sr, wave_unit: Optional[str]):
    """
    Convert Gaussian area to erg/s/cm² using astropy.units.

    This treats the amplitude as a flux density per frequency (MJy/sr) and
    converts it to per-wavelength using the spectral_density equivalency at
    the line center, then integrates over wavelength:

        area = (F_lambda [erg/s/cm²/μm/sr]) × (sigma [μm]) × sqrt(2π)
        area_total = area × solid_angle [sr]

    Args:
        amplitude_mjy_sr: Peak amplitude in MJy/sr
        sigma_um: Gaussian sigma in μm
        center_um: Line center wavelength in μm
        solid_angle_sr: Solid angle per pixel in steradians

    Returns:
        float: Area in erg/s/cm², or None if conversion fails
    """
    if (solid_angle_sr is None or amplitude_mjy_sr is None or
        sigma_val is None or center_val is None or not np.isfinite(center_val)):
        return None

    try:
        unit = _normalize_wave_unit(wave_unit)
        if unit == "KM/S":
            return None  # velocity axis needs rest wavelength to convert

        amp = amplitude_mjy_sr * (u.MJy / u.sr)

        if unit == "ANGSTROM":
            center = (center_val * u.AA).to(u.um)
            sigma = (sigma_val * u.AA).to(u.um)
        else:
            center = center_val * u.um
            sigma = sigma_val * u.um

        amp_flam = amp.to(u.erg / u.s / u.cm**2 / u.um / u.sr,
                          equivalencies=u.spectral_density(center))

        area = amp_flam * sigma * np.sqrt(2 * np.pi)
        area_total = area * (solid_angle_sr * u.sr)

        return area_total.to_value(u.erg / u.s / u.cm**2)
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
        # Create RANSAC pipeline with polynomial features
        ransac_pipeline = make_pipeline(
            PolynomialFeatures(degree=pdeg),
            RANSACRegressor()
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
        baseline = calculate_baseline(x_noise[0], y_noise[0], x_noise[-1], y_noise[-1], x_noise)
        y_corrected = y_noise - baseline
        mad = np.nanmedian(np.abs(y_corrected - np.nanmedian(y_corrected)))
        robust_std = 1.4826 * mad
        if np.isfinite(robust_std) and robust_std > 0:
            return float(robust_std)
    
    return None


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
    
    Args:
        x1: Left boundary wavelength
        x2: Right boundary wavelength
        spectrum_x: Full spectrum wavelength array
        spectrum_y: Full spectrum flux array
        peak_x: Explicit peak wavelength (REQUIRED - user must select the peak)
        noise_min: Optional left boundary of a line-free noise region
        noise_max: Optional right boundary of a line-free noise region
        header: Optional FITS header for unit conversion (reads CDELT1, CDELT2)
    
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
    
    # Calculate linear baseline
    baseline = calculate_baseline(x1, y1, x2, y2, x_data)
    baseline_slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
    
    # Subtract baseline
    y_corrected = y_data - baseline

    # Optional noise region for robust std estimation
    noise_std = None
    if noise_min is not None and noise_max is not None:
        nmin, nmax = (noise_min, noise_max) if noise_min <= noise_max else (noise_max, noise_min)
        noise_mask = (spectrum_x >= nmin) & (spectrum_x <= nmax)
        if np.count_nonzero(noise_mask) >= 2:
            noise_std = compute_noise_std(spectrum_x[noise_mask], spectrum_y[noise_mask])

    std_val_fit = np.nanstd(y_corrected)
    std_val = noise_std if (noise_std is not None and np.isfinite(noise_std)) else std_val_fit
    
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
    print(f"[DEBUG] Peak at λ={peak_x:.4f}: amp={amp_guess:.4e}, σ={sigma_guess:.4e}")

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
    
    # Clamp initial guesses to be within bounds (prevents "x0 is infeasible" error)
    amp_guess = np.clip(amp_guess, bounds[0][0], bounds[1][0])
    center_guess = np.clip(center_guess, bounds[0][1], bounds[1][1])
    sigma_guess = np.clip(sigma_guess, bounds[0][2], bounds[1][2])
    
    try:
        # Perform RANSAC fit (default for robust outlier rejection)
        popt, ransac_success, inlier_mask = fit_gaussian_with_ransac(
            x_data, y_corrected,
            p0=[amp_guess, center_guess, sigma_guess],
            bounds=bounds,
            ransac_thresh=2.0,
            ransac_iters=3
        )
        
        amplitude, center, sigma = popt
        
        # Generate fitted curve
        y_fit = gaussian(x_data, amplitude, center, sigma)
        residuals = y_corrected - y_fit
        
        # Calculate derived quantities
        fwhm = 2.355 * sigma  # FWHM = 2*sqrt(2*ln(2)) * sigma
        
        # Calculate area in native units (MJy/sr · μm)
        area_native = amplitude * sigma * np.sqrt(2 * np.pi)
        
        # Convert to erg/s/cm² if header is provided
        solid_angle = get_pixel_solid_angle(header)
        unit = get_wave_unit(header, wave_unit)
        if solid_angle is not None:
            area_cgs = convert_area_to_cgs(amplitude, sigma, center, solid_angle, unit)
            area = area_cgs if area_cgs is not None else area_native
            area_unit = "erg/s/cm²" if area_cgs is not None else "MJy/sr·μm"
        else:
            area = area_native
            area_unit = "MJy/sr·μm"
        
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
                            n_jobs: Optional[int] = None,
                            fac_above_stdres: float = 3.0,
                            noise_min: Optional[float] = None,
                            noise_max: Optional[float] = None,
                            header: Optional[object] = None,
                            wave_unit: Optional[str] = None) -> dict:
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
        fac_above_stdres: Detection threshold in units of noise std (default: 3.0 for 3σ)
        noise_min: Optional left boundary of a line-free noise region
        noise_max: Optional right boundary of a line-free noise region
        header: Optional FITS header for unit conversion (reads CDELT1, CDELT2)
    
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
    """
    # Get cube dimensions
    n_wave, n_y, n_x = cube.shape
    
    # Calculate solid angle for unit conversion
    solid_angle = get_pixel_solid_angle(header)
    unit = get_wave_unit(header, wave_unit)
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
    is_detection_map = np.zeros((n_y, n_x), dtype=bool)  # True if SNR threshold passed
    is_upper_limit_map = np.zeros((n_y, n_x), dtype=bool)  # True if upper limit computed
    
    # Optional noise region mask for robust std estimation
    noise_wave_mask = None
    noise_x = None
    if noise_min is not None and noise_max is not None:
        nmin, nmax = (noise_min, noise_max) if noise_min <= noise_max else (noise_max, noise_min)
        noise_wave_mask = (wavelength >= nmin) & (wavelength <= nmax)
        if np.count_nonzero(noise_wave_mask) < 2:
            noise_wave_mask = None
        else:
            noise_x = wavelength[noise_wave_mask]

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
            y_data = spectra_row[:, ix]
            noise_std = None
            if spectra_noise_row is not None and noise_x is not None:
                noise_std = compute_noise_std(noise_x, spectra_noise_row[:, ix])

            # Baseline from endpoints
            y1 = y_data[0]
            y2 = y_data[-1]
            baseline = calculate_baseline(x1, y1, x2, y2, x_data)
            y_corrected = y_data - baseline
            
            # Require at least 2 finite points to compute stats
            finite_mask = np.isfinite(y_corrected)
            n_finite = np.count_nonzero(finite_mask)
            
            # Case 3: All NaN pixels (outside ROI) - keep as NaN
            if n_finite == 0:
                # Leave as NaN (already initialized) - no data in this pixel
                continue
            
            # Case 2/4: Insufficient data but not all NaN - assign upper limit
            if n_finite < 2:
                # Assign fac_above_stdres σ upper limit for pixels with insufficient data
                std_temp = noise_std if (noise_std is not None and np.isfinite(noise_std)) else np.nanstd(y_data)
                upper_limit_amplitude = fac_above_stdres * std_temp
                upper_limit_area_native = upper_limit_amplitude * (x_range / 4.0) * np.sqrt(2 * np.pi)
                # Convert to erg/s/cm² if solid angle available
                if solid_angle is not None:
                    area_cgs = convert_area_to_cgs(
                        upper_limit_amplitude,
                        x_range / 4.0,
                        (x1 + x2) / 2,
                        solid_angle,
                        unit
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
            
            # Noise/peak logic
            mean_val = np.nanmean(y_corrected)
            std_val_fit = np.nanstd(y_corrected)
            std_val = noise_std if (noise_std is not None and np.isfinite(noise_std)) else std_val_fit
            if std_val < 0.5 * abs(mean_val):
                mad = np.nanmedian(np.abs(y_corrected - np.nanmedian(y_corrected)))
                robust_std = 1.4826 * mad
                sigma_for_fit = robust_std if robust_std > 0 else std_val
                try:
                    amp_guess, center_guess, _ = estimate_gaussian_params(x_data, y_corrected)
                    sigma_guess = sigma_for_fit
                except Exception:
                    # Assign fac_above_stdres σ upper limit for failed parameter estimation
                    upper_limit_amplitude = fac_above_stdres * std_val
                    upper_limit_area_native = upper_limit_amplitude * (x_range / 4.0) * np.sqrt(2 * np.pi)
                    # Convert to erg/s/cm² if solid angle available
                    if solid_angle is not None:
                        area_cgs = convert_area_to_cgs(
                            upper_limit_amplitude,
                            x_range / 4.0,
                            (x1 + x2) / 2,
                            solid_angle,
                            unit
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
            else:
                try:
                    amp_guess, center_guess, sigma_guess = estimate_gaussian_params(x_data, y_corrected)
                except Exception:
                    # Assign fac_above_stdres σ upper limit for failed parameter estimation
                    upper_limit_amplitude = fac_above_stdres * std_val
                    upper_limit_area_native = upper_limit_amplitude * (x_range / 4.0) * np.sqrt(2 * np.pi)
                    # Convert to erg/s/cm² if solid angle available
                    if solid_angle is not None:
                        area_cgs = convert_area_to_cgs(
                            upper_limit_amplitude,
                            x_range / 4.0,
                            (x1 + x2) / 2,
                            solid_angle,
                            unit
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
            
            # Default to 1-sigma (smaller/conservative) for upper limits
            # This ensures conservative fitting and smaller intensity estimates
            sigma_guess = min(sigma_guess, 0.01 * x_range)

            # Require positive peak
            if np.max(y_corrected) <= 0:
                # Assign fac_above_stdres σ upper limit for pixels with no positive peak
                upper_limit_amplitude = fac_above_stdres * std_val
                upper_limit_area = upper_limit_amplitude * (x_range / 4.0) * np.sqrt(2 * np.pi)
                amp_row[ix] = upper_limit_amplitude
                area_row[ix] = upper_limit_area
                ctr_row[ix] = (x1 + x2) / 2
                sig_row[ix] = x_range / 4.0
                fwhm_row[ix] = 2.355 * (x_range / 4.0)
                uplim_row[ix] = True
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
                # Always use RANSAC for robust outlier rejection
                popt, fit_success, inlier_mask = fit_gaussian_with_ransac(
                    x_data, y_corrected,
                    p0=[amp_guess, center_guess, sigma_guess],
                    bounds=bounds,
                    ransac_thresh=2.0,  # 2-sigma threshold for outliers
                    ransac_iters=2  # Very fast: 2 iterations is enough, minimal outliers in spectral data
                )
                
                amplitude, center, sigma = popt
                y_fit = gaussian(x_data, amplitude, center, sigma)
                residuals = y_corrected - y_fit

                fwhm = 2.355 * sigma
                
                # Calculate area in native units (MJy/sr · μm)
                area_native = amplitude * sigma * np.sqrt(2 * np.pi)
                # Convert to erg/s/cm² if solid angle available
                if solid_angle is not None:
                    area_cgs = convert_area_to_cgs(amplitude, sigma, center, solid_angle, unit)
                    area = area_cgs if area_cgs is not None else area_native
                else:
                    area = area_native

                # Calculate noise from residuals (std of continuum fit)
                n_points = len(y_corrected)
                dof = max(1, n_points - 3)
                noise_estimate = np.std(residuals, ddof=0)
                if noise_estimate > 0:
                    chi_squared_red = np.sum(residuals**2) / (dof * noise_estimate**2)
                else:
                    chi_squared_red = np.inf

                # SNR-based detection threshold
                # Check: (1) fit converged, (2) amplitude > threshold, (3) fit improves residuals
                std_no_fit = std_val if (std_val is not None and np.isfinite(std_val)) else np.nanstd(y_corrected)
                std_with_fit = noise_estimate  # std of residuals after fit
                fit_improves = fit_success and (std_with_fit < std_no_fit)
                
                is_detection = (amplitude >= fac_above_stdres * std_val) and fit_improves
                
                if is_detection:
                    # True detection: store fitted parameters
                    amp_row[ix] = amplitude
                    ctr_row[ix] = center
                    sig_row[ix] = sigma
                    fwhm_row[ix] = fwhm
                    area_row[ix] = area
                    qual_row[ix] = chi_squared_red
                    succ_row[ix] = True
                    det_row[ix] = True
                    
                    # Cache normalized residuals
                    std_val_for_norm = std_val if (std_val is not None and std_val > 0) else 1.0
                    normalized_residuals = residuals / std_val_for_norm
                    residuals_row[:, ix] = normalized_residuals.astype(np.float32)
                else:
                    # Failed detection threshold: compute fac_above_stdres σ upper limit
                    # Upper limit = fac_above_stdres σ * sigma_guess * sqrt(2π)
                    upper_limit_amplitude = fac_above_stdres * std_val
                    upper_limit_area_native = upper_limit_amplitude * sigma_guess * np.sqrt(2 * np.pi)
                    # Convert to erg/s/cm² if solid angle available
                    if solid_angle is not None:
                        area_cgs = convert_area_to_cgs(
                            upper_limit_amplitude,
                            sigma_guess,
                            center_guess if 'center_guess' in locals() else (x1 + x2) / 2,
                            solid_angle,
                            unit
                        )
                        upper_limit_area = area_cgs if area_cgs is not None else upper_limit_area_native
                    else:
                        upper_limit_area = upper_limit_area_native
                    
                    # If upper limit is too small, use edge/baseline values instead
                    edge_value = np.mean([np.abs(y1), np.abs(y2)])
                    if upper_limit_area < 0.1 * edge_value or np.isnan(upper_limit_area):
                        upper_limit_area = edge_value
                    
                    amp_row[ix] = upper_limit_amplitude
                    ctr_row[ix] = center_guess  # Use guessed center
                    sig_row[ix] = sigma_guess  # Use guessed width
                    fwhm_row[ix] = 2.355 * sigma_guess
                    area_row[ix] = upper_limit_area  # 1σ upper limit
                    qual_row[ix] = np.nan
                    succ_row[ix] = False
                    uplim_row[ix] = True  # Mark as upper limit
                    
            except Exception:
                # Fit failed completely: compute fac_above_stdres σ upper limit using initial guesses
                # case (3) - fit not achieved, assign fac_above_stdres σ upper limit
                if 'sigma_guess' in locals():
                    upper_limit_amplitude = fac_above_stdres * std_val
                    upper_limit_area_native = upper_limit_amplitude * sigma_guess * np.sqrt(2 * np.pi)
                    # Convert to erg/s/cm² if solid angle available
                    if solid_angle is not None:
                        area_cgs = convert_area_to_cgs(
                            upper_limit_amplitude,
                            sigma_guess,
                            center_guess if 'center_guess' in locals() else (x1 + x2) / 2,
                            solid_angle,
                            unit
                        )
                        upper_limit_area = area_cgs if area_cgs is not None else upper_limit_area_native
                    else:
                        upper_limit_area = upper_limit_area_native
                else:
                    upper_limit_amplitude = np.nan
                    upper_limit_area = np.nan
                
                amp_row[ix] = upper_limit_amplitude
                area_row[ix] = upper_limit_area
                ctr_row[ix] = center_guess if 'center_guess' in locals() else np.nan
                sig_row[ix] = sigma_guess if 'sigma_guess' in locals() else np.nan
                fwhm_row[ix] = 2.355 * sigma_guess if 'sigma_guess' in locals() else np.nan
                uplim_row[ix] = True

        return row_idx, amp_row, ctr_row, sig_row, fwhm_row, area_row, qual_row, succ_row, det_row, uplim_row, residuals_row

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

            # Progress per row
            completed += n_x
            if completed % 1000 == 0:
                print(f"  Progress: {completed}/{total_pixels} pixels ({100*completed/total_pixels:.1f}%)")
    else:
        # Parallel path using ThreadPoolExecutor
        # ThreadPoolExecutor works with nested functions (unlike ProcessPoolExecutor)
        # and provides good performance for Gaussian fitting
        print(f"Using {n_jobs} worker threads ({total_pixels:,} pixels)")
        
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_fit_row, iy) for iy in range(n_y)]
            for fut in as_completed(futures):
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

                completed += n_x
                if completed % (n_x * 5) == 0:
                    print(f"  Progress: {completed}/{total_pixels} pixels ({100*completed/total_pixels:.1f}%)")
    
    # Final summary
    n_success = np.sum(success_map)
    n_detections = np.sum(is_detection_map)
    n_upper_limits = np.sum(is_upper_limit_map)
    print(f"Fitting complete: {n_success}/{total_pixels} successful fits ({100*n_success/total_pixels:.1f}%)")
    print(f"  Detections (>{fac_above_stdres}σ): {n_detections} ({100*n_detections/total_pixels:.1f}%)")
    print(f"  Upper limits: {n_upper_limits} ({100*n_upper_limits/total_pixels:.1f}%)")
    
    return {
        'amplitude': amplitude_map,
        'center': center_map,
        'sigma': sigma_map,
        'fwhm': fwhm_map,
        'area': area_map,
        'r_squared': r_squared_map,
        'success': success_map,
        'is_detection': is_detection_map,  # True detections (passed SNR threshold)
        'is_upper_limit': is_upper_limit_map,  # Upper limits (failed detection)
        'residuals_cube': residuals_cube,  # Cached residuals computed during fitting
        'fac_above_stdres': fac_above_stdres  # Detection threshold used
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