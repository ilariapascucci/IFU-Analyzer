"""
Gaussian Fitting Module for Spectral Line Analysis

This module provides tools for fitting Gaussian profiles to spectral lines
after baseline subtraction. Used for measuring line properties like FWHM,
equivalent width, and integrated flux.
"""

import numpy as np
from scipy.optimize import curve_fit
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
    
    # Estimate initial parameters
    amp_guess, center_guess, sigma_guess = estimate_gaussian_params(x_data, y_corrected)
    
    # Set up bounds to constrain fit
    # Amplitude: must be positive, reasonable upper limit
    # Center: must be within selected range
    # Sigma: must be positive, not larger than range
    x_range = x2 - x1
    bounds = (
        [0, x1, 0.001 * x_range],  # Lower bounds
        [10 * np.max(y_corrected), x2, 2 * x_range]  # Upper bounds
    )
    
    try:
        # Perform least-squares fit
        popt, pcov = curve_fit(
            gaussian, x_data, y_corrected,
            p0=[amp_guess, center_guess, sigma_guess],
            bounds=bounds,
            maxfev=10000
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
                            lam_max: Optional[float] = None) -> dict:
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
            'r_squared': R² goodness of fit map
            'success': Boolean mask of successful fits
    
    Notes:
        - Failed fits store NaN in parameter maps
        - Uses same estimation and bounds as fit_gaussian_to_line()
        - Area calculated analytically: A * σ * sqrt(2π)
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
    
    # Get boundary wavelengths
    x1 = x_data[0]
    x2 = x_data[-1]
    x_range = x2 - x1
    
    # Progress tracking
    total_pixels = n_y * n_x
    completed = 0
    
    print(f"Fitting Gaussians to {total_pixels} pixels in wavelength range [{lam_min:.4f}, {lam_max:.4f}]")
    
    # Loop over all spatial pixels
    for iy in range(n_y):
        for ix in range(n_x):
            # Extract spectrum for this pixel
            spectrum = cube[wave_mask, iy, ix]
            y_data = spectrum
            
            # Get boundary flux values for baseline
            y1 = y_data[0]
            y2 = y_data[-1]
            
            # Calculate baseline (same as fit_gaussian_to_line)
            baseline = calculate_baseline(x1, y1, x2, y2, x_data)
            
            # Subtract baseline
            y_corrected = y_data - baseline
            
            # Check if there's a meaningful peak
            if np.max(y_corrected) <= 0:
                # Skip this pixel - no positive peak
                continue
            
            # Estimate initial parameters (same as fit_gaussian_to_line)
            try:
                amp_guess, center_guess, sigma_guess = estimate_gaussian_params(x_data, y_corrected)
            except Exception:
                # Skip if parameter estimation fails
                continue
            
            # Set up bounds (same as fit_gaussian_to_line)
            bounds = (
                [0, x1, 0.001 * x_range],  # Lower bounds
                [10 * np.max(y_corrected), x2, 2 * x_range]  # Upper bounds
            )
            
            # Attempt fit
            try:
                popt, pcov = curve_fit(
                    gaussian, x_data, y_corrected,
                    p0=[amp_guess, center_guess, sigma_guess],
                    bounds=bounds,
                    maxfev=10000
                )
                
                amplitude, center, sigma = popt
                
                # Generate fitted curve
                y_fit = gaussian(x_data, amplitude, center, sigma)
                residuals = y_corrected - y_fit
                
                # Calculate derived quantities
                fwhm = 2.355 * sigma  # FWHM = 2*sqrt(2*ln(2)) * sigma
                
                # Calculate area using analytic formula (same as fit_gaussian_to_line)
                area = amplitude * sigma * np.sqrt(2 * np.pi)
                
                # Calculate R² goodness of fit
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_corrected - np.mean(y_corrected))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Store results
                amplitude_map[iy, ix] = amplitude
                center_map[iy, ix] = center
                sigma_map[iy, ix] = sigma
                fwhm_map[iy, ix] = fwhm
                area_map[iy, ix] = area
                r_squared_map[iy, ix] = r_squared
                success_map[iy, ix] = True
                
            except Exception as e:
                # Fit failed for this pixel - leave as NaN
                pass
            
            # Progress update every 1000 pixels
            completed += 1
            if completed % 1000 == 0:
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
        'success': success_map
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