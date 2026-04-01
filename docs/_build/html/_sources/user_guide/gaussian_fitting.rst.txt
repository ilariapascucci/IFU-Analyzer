================
Gaussian Fitting
================

Fit Gaussian profiles to spectral lines to extract physical parameters. This page covers both
1D single-pixel spectrum fitting and full-cube automated fitting.

Why Fit Gaussians?
==================

Gaussian fitting is essential for quantitative spectral analysis. It provides:

- **Line Centers**: Exact wavelength of emission or absorption features
- **Peak Intensities**: Maximum flux value at the line center
- **Line Widths**: FWHM (Full Width Half Maximum) and line width (σ)
- **Integrated Flux**: Total energy/intensity in the spectral line
- **Physical Parameters**: From line properties, derive physical quantities like temperature, 
  density, or mass
- **Quality Metrics**: χ², R² values, and residuals to assess fit reliability and distinguish 
  real features from noise

1D Spectrum Fitting
===================

**Quick Fit (Interactive)**

This feature allows you to fit a Gaussian to a single pixel's 1D spectrum. This is typically 
done to understand your data before running full-cube fitting or to target specific regions 
of interest.

**Steps:**

1. Click on **Gaussian Fit** on the middle toolbar
2. Click on the spectrum to mark three wavelengths - this defines your fit region:
   
   - **Left** (λ_left): The shorter wavelength boundary marking the left edge of the feature
   - **Peak** (λ_peak): The approximate center wavelength of the emission/absorption line
   - **Right** (λ_right): The longer wavelength boundary marking the right edge of the feature
   
   These three points define a wavelength window encompassing your spectral line.

3. Click **Fit Gaussian** button to perform the fit
4. A Gaussian curve overlay will appear on your spectrum
5. Fit parameters are displayed:
   
   - **Center** (μ): Line center wavelength (the peak of the fitted Gaussian)
   - **Amplitude** (A): Peak intensity  
   - **Sigma** (σ): Standard deviation of the Gaussian profile
   - **FWHM**: Full Width Half Maximum (related to σ by FWHM = 2.355 × σ)
   - **Gaussian Area**: Integrated flux under the Gaussian curve
   - **Equivalent Width**: Width of a rectangular profile with same flux as the Gaussian
   - **Slope**: Baseline slope (continuum gradient in the fit region)
   - **Left Endpoint**: Wavelength value at the left edge of the fit region
   - **R²**: Coefficient of determination (ranges 0-1, higher is better)
   - **χ²**: Reduced chi-squared fit quality metric (lower is better)
   - **Data Points**: Number of wavelength points included in the fit

6. Close informational window to return to the main interface

Pixel-by-Pixel Cube Fitting
===========================

**Full-Cube Gaussian Fitting**

Fit Gaussians to every pixel in your spectral cube automatically. This operation extracts 
parameter maps showing how line properties vary across your observed region.

**Prerequisite**: You should first fit the spectrum at your region of interest using 
**1D Spectrum Fitting** (described above). This helps you identify the appropriate wavelength 
range and understand the spectral features in your data.

**Starting Cube Fit**

1. Select **Noise L** and **Noise R** to gather noise information near the fit
2. Click **Fit Gaussian Map** button
3. Click **Start** to begin

**Progress Monitoring**

- Progress bar shows % complete

**Output Maps**

After fitting, the following Integrated Intensity Maps and Pixel-by-Pixel Gaussian Fitting 
maps are created:

- **Area Map**: Integrated line flux (total intensity) across the cube
- **Amplitude Map**: Peak intensity at each pixel
- **Center Map**: Line center wavelength at each spatial location
- **Sigma Map**: Line width (σ values) at each pixel
- **FWHM Map**: Full width at half maximum across the field
- **Chi-Squared Map**: Fit quality at each pixel (lower values = better fits)
- **Success Mask**: Binary map showing which fits succeeded vs. failed
- **Residuals Map**: Difference between data and fitted model at each pixel

Viewing Fit Results
===================

**Display Integrated Intensity Maps and Pixel-by-Pixel Fitting Results**

After pixel-by-pixel cube fitting is complete, parameter maps appear automatically. 
Closing this window keeps the maps in memory unless new fitting values are input.

**Constraints**

1. Left → Peak → Right should be in a logical wavelength order (Left < Peak < Right)
2. For more accurate results, the noise estimate wavelengths (Noise L and Noise R) 
   should not be too far from the actual fit region.

Exporting Fit Parameters
========================

**Save Integrated Intensity Maps and Pixel-by-Pixel Fitting Results as FITS**

To export your fitting results:

1. For each parameter map you want to save, click on **Save as FITS file**
2. This saves the parameter map as a 1-slice 3D FITS file
3. You can then open these maps in a new window or use them in external analysis tools


Using the Gaussian Fitting API
===============================

You can also perform Gaussian fitting programmatically in Python scripts or Jupyter notebooks 
using the IFU Analyzer API.

**1D Spectrum Fitting**

Fit a Gaussian profile to a single spectrum:

.. code-block:: python

    from cube_explorer.api import load_cube, fit_line
    
    # Load your data
    cube = load_cube("data.fits")
    
    # Fit a single line by specifying left and right wavelengths
    result = fit_line(
        wavelength=cube.wavelength,
        spectrum=cube.spectrum,
        x1=4.701,      # Left boundary (µm)
        x2=4.723,      # Right boundary (µm)
        line_name="CO v=1-0"
    )
    
    if result.success:
        print(f"Line Center: {result.center:.6f} µm")
        print(f"Amplitude: {result.amplitude:.3f}")
        print(f"FWHM: {result.fwhm:.6f} µm")
        print(f"Integrated Flux: {result.area:.3f}")
        print(f"Fit Quality (R²): {result.r_squared:.4f}")
        print(f"Fit Quality (χ²): {result.chi_squared:.6f}")

**Fitting Multiple Lines**

Fit multiple spectral features in sequence:

.. code-block:: python

    from cube_explorer.api import load_cube, fit_multiple_lines
    
    cube = load_cube("data.fits")
    
    # Define multiple lines by their boundaries
    lines = [
        (4.701, 4.723),    # CO v=1-0
        (5.905, 5.925),    # CO v=0-0
        (2.292, 2.300)     # Another line
    ]
    
    results = fit_multiple_lines(cube.wavelength, cube.spectrum, lines)
    
    for i, result in enumerate(results):
        if result.success:
            print(f"Line {i}: center={result.center:.6f} µm, FWHM={result.fwhm:.6f} µm")
        else:
            print(f"Line {i}: Fit failed - {result.message}")

**HITRAN Database Lines**

Automatically fit all lines from a molecular species in your spectrum:

.. code-block:: python

    from cube_explorer.api import load_cube, fit_hitran_lines
    
    cube = load_cube("data.fits")
    
    # Fit all CO v=1->0 rotational lines in the spectrum
    results = fit_hitran_lines(
        wavelength=cube.wavelength,
        spectrum=cube.spectrum,
        molecule='CO',
        v_upper=1,          # Upper vibrational level
        v_lower=0           # Lower vibrational level
    )
    
    successful = [r for r in results if r['success']]
    print(f"Successfully detected {len(successful)} CO v=1->0 lines")
    
    for line in successful:
        print(f"  {line['transition']:12} at {line['wavelength']:.6f} µm - "
              f"FWHM: {line['fwhm']:.6f} µm, Area: {line['area']:.3f}")

**Access Individual Fit Parameters**

.. code-block:: python

    result = fit_line(wavelength, spectrum, 4.701, 4.723)
    
    # Commonly used parameters
    fitted_center = result.center           # Line center wavelength
    fitted_amplitude = result.amplitude     # Peak height
    fitted_sigma = result.sigma             # Gaussian sigma
    fitted_fwhm = result.fwhm               # Full width half maximum
    fitted_area = result.area               # Integrated area (flux)
    fit_quality_r2 = result.r_squared       # R² value (0-1, higher is better)
    fit_quality_chi2 = result.chi_squared   # χ² value (lower is better)
    
    # Continuum information
    continuum_flux = result.continuum       # Fitted continuum at line location
    
    # Status information
    was_successful = result.success         # Boolean: fit succeeded?
    status_message = result.message         # String: status information
