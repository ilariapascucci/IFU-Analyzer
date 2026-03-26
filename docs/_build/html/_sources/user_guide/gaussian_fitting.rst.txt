================
Gaussian Fitting
================

Fit Gaussian profiles to spectral lines to extract physical parameters.

Why Fit Gaussians?
==================

Gaussian fitting provides:

- **Line Centers**: Exact wavelength/velocity of emission
- **Peak Intensities**: Maximum flux value
- **Line Widths**: FWHM and velocity dispersion
- **Integrated Flux**: Total energy in the line
- **Quality Metrics**: χ², R², residuals for assessment

Interactive Single-Pixel Fitting
===============================

**Quick Fit (Interactive)**

1. Click on **Gaussian Fit** on the middle toolbar
2. Select **Left**, **Peak**, and **Right** wavelengths on corresponding fit
2. Click **Fit Gaussian** button
3. Gaussian overlay appears on spectrum
4. Fit parameters shown:
   
   - **Center** (μ): Line center wavelength
   - **Amplitude** (A): Peak intensity
   - **Sigma** (σ): Standard deviation
   - **FWHM**: Full Width Half Maximum
   - **Gaussian Area**: Integrated flux
   - **Equivalent Width**: 
   - **Slope**:
   - **Left Endpoint**:
   - **R²**: Coefficient of determination (higher is better)
   - **χ²**: Fit quality (lower is better)
   - **Data Points**: Number of wavelength points in the fit

5. Close informational window

Full-Cube Gaussian Fitting
==========================

Fit all pixels in the cube simultaneously:

**Starting Cube Fit**

1. Select **Noise L** and **Noise R** to gather noise information near the fit
2. Click **Fit Gaussian Map** button
3. Click **Start** to begin

**Progress Monitoring**

- Progress bar shows % complete

**Output Maps**

After fitting, created maps include:

- **Area Map**: Integrated line flux
- **Amplitude Map**: Peak intensity at each pixel
- **Center Map**: Line center wavelength
- **Sigma Map**: Line width (σ values)
- **FWHM Map**: Full width at half maximum
- **Chi-Squared Map**: Fit quality at each pixel
- **Success Mask**: Good vs. failed fits
- **Residuals**:

Viewing Fit Results
===================

**Display Parameter Map**

Parameter maps appear after gaussian mapping is complete. Closing this window keeps the
maps in memory unless new values are input.

**Constraints**

1. Left→Peak→Right should be in a logical order
2. For more accurate results, noise should not be too far from the original fit.

Exporting Fit Parameters
========================

**Save Parameter Maps as FITS**

On each map, click on **Save as FITS file**. This saves the parameter map as
a 1-slice 3D FITS file available to open in a new window.
