===================
Spectral Analysis
===================

Extract and analyze spectral information from your IFU data cube.

Understanding IFU Spectra
=========================

In IFU data, you can extract spectra from:

1. **Single Spaxels**: Spectrum of one spatial pixel
2. **Selected Regions**: Average spectrum of multiple pixels
3. **Entire Cube**: Integrated spectrum of all pixels

The spectrum shows flux (or counts) as a function of wavelength or velocity.

Viewing Spectra
===============

**Integrated Spectrum Display**

The right panel shows the spectrum of your current selection:

- **X-axis**: Wavelength (or velocity/frequency)
- **Y-axis**: Flux/intensity
- **Red curve**: Data spectrum
- **Blue points** (optional): Uncertainty/errors
- **Green overlay** (optional): Fitted profile or residuals

**Spectrum Update**

Spectrum updates when:

- You change wavelength slider
- You select a new region
- You move cursor over the map

To update manually: Press **Enter** or click **Update Spectrum** button

Extracting Spectra
==================

**Extract Current Spectrum**

1. Select region of interest (or click single pixel)
2. Right-click spectrum panel
3. **Export Spectrum**
4. Choose format (ASCII or FITS)
5. Specify filename

**Export Full Cube Spectrum**

1. Select no region (click empty space to deselect)
2. File → Export Spectrum
3. Entire cube spectrum saved

**Batch Export**

Export spectra from multiple regions:

1. Define and save each region (Tools → Save Region)
2. File → Batch Export Spectra
3. Select regions to export
4. Choose output directory
5. Click Start

Spectrum Operations
===================

**Normalize Spectrum**

Remove continuum level:

1. Right-click spectrum
2. **Normalize**
3. Options:
   - By median: Divide by median value
   - By maximum: Divide by maximum value
   - By selected range: Divide by range

**Rebin/Smooth Spectrum**

Reduce noise by combining channels:

1. Right-click spectrum
2. **Rebin**
3. Specify bin size (e.g., 2, 3, 5)
4. Smoothed spectrum displays

**Median Filter**

Apply statistical smoothing:

1. Right-click spectrum
2. **Median Filter**
3. Specify window width
4. Removes spikes while preserving features

**Interpolate**

Fill gaps or interpolate to new wavelength grid:

1. Right-click spectrum
2. **Interpolate**
3. Choose method (linear, cubic, spline)
4. New grid created

Spectrum Analysis
=================

**Feature Detection**

Automatically find emission/absorption lines:

1. Tools → Spectral Analysis → Detect Features
2. Specify detection threshold (in sigma)
3. Features highlighted with markers

**Line Measurements**

Measure properties of identified lines:

1. Select line region (manually or auto-detected)
2. Right-click → **Line Properties**
3. Shows:
   - Wavelength/velocity
   - Flux/area
   - Width (FWHM)
   - Peak intensity

**Equivalent Width**

Measure line strength as equivalent width:

1. Select line + continuum region
2. Tools → Measure → Equivalent Width
3. Result displayed in Å or velocity units

**Profile Analysis**

Analyze line profile shape:

1. Select spectral line region
2. Right-click → **Profile Shape**
3. Shows:
   - Gaussian fit parameters
   - Lorentzian comparison
   - Skewness/kurtosis

**Flux Measurement**

Integrate flux in wavelength range:

1. Select wavelength range on spectrum
2. Right-click → **Measure Flux**
3. Shows total flux (with uncertainty)

Line Identification
===================

**Automatic Line ID**

Identify emission/absorption features:

1. Tools → Line Identification
2. Select source catalog:
   - HITRAN (molecules)
   - Rest line list (emission lines)
   - Custom list
3. Wavelength range automatically scanned
4. Matching lines highlighted

**Manual Line ID**

Specify known lines:

1. Right-click feature on spectrum
2. **Identify Line**
3. Enter rest wavelength or line name
4. Offset calculated and displayed

**Line Catalog Viewer**

Browse available spectral lines:

1. Tools → Line Catalog
2. Filter by:
   - Species/element
   - Wavelength range
   - Intensity
3. Double-click to highlight in spectrum

Spectral Diagnostics
====================

**Ratio Diagnostics**

Use line ratios for physical diagnostics:

1. Measure flux in two spectral regions
2. Right-click → **Calculate Ratio**
3. Diagnostic plot displayed
4. Compare to diagnostic grids (if available)

Useful for:
- Temperature estimation
- Density measurement
- Extinction determination
- Abundance calculation

**Velocity Analysis**

Measure velocity shifts and dispersions:

1. Fit spectral lines (see :doc:`gaussian_fitting`)
2. Center positions map velocity shifts
3. Width map shows velocity dispersion

Sigma (μ):
  Velocity dispersion in km/s

FWHM:
  Full width at half maximum = 2.355 × σ

**Redshift Determination**

Measure redshift from wavelength shift:

1. Tools → Measure → Redshift
2. Specify rest wavelength
3. Observed wavelength from fit or input
4. Redshift calculated: z = (λ_obs - λ_rest) / λ_rest

Comparing Multiple Spectra
===========================

**Side-by-Side Comparison**

Load and compare multiple spectra:

1. File → Open Multiple
2. Select multiple FITS files
3. Spectra displayed in separate panels
4. Scroll/zoom synchronized

**Spectrum Subtraction**

Subtract one spectrum from another:

1. Load reference spectrum
2. Right-click → **Subtract Spectrum**
3. Choose second spectrum
4. Residual spectrum displayed

**Spectrum Overlay**

Overlay multiple spectra:

1. Display first spectrum
2. Tools → Overlay Spectrum
3. Select second spectrum
4. Both visible, different colors

**Scaling and Shifting**

Adjust spectrum scale or position:

1. Right-click spectrum
2. **Scale**: Multiply by constant
3. **Shift**: Add constant offset
4. Match to reference spectrum

Spectrum Export Formats
=======================

**ASCII Format**

Tab-delimited text file:

.. code-block:: text

   Wavelength(Angstrom)    Flux(mJy)      Error
   4005.123               45.32          2.1
   4005.456               45.81          2.0

**FITS Format**

Binary FITS table:

- Primary HDU: Single 1D spectrum
- Extension HDU: Wavelength, flux, error arrays

**Custom Format**

Specify column order and units:

1. File → Export Spectrum → Advanced
2. Choose columns, units, delimiter
3. Configure header lines
4. Export

Spectral Uncertainty
====================

**Error Estimation**

Estimate uncertainties from data:

1. Tools → Estimate Uncertainty
2. Methods:
   - From exposure time
   - From noise estimation
   - From replication (if available)

**Using Error Bars**

Spectrum display can show uncertainties:

1. Display → Show Error Bars
2. If errors in file, displayed as bands
3. Used in fitting (weighted fit)

Performance Notes
=================

For large cubes extracting many spectra:

- Use downsampled cube for initial testing
- Specify wavelength ranges to speed extraction
- Use batch export for efficiency
- Consider external pipeline for millions of spectra

Troubleshooting
===============

**Spectrum looks noisy**

- Apply smoothing/rebinning
- Check if noise is real (compare with other regions)
- Try median filter

**Lines not identified**

- Verify wavelength calibration
- Check against catalog in external tool
- Confirm line rest wavelength is correct

**Flux measurement seems wrong**

- Verify wavelength range
- Check continuum level
- Compare with other analysis methods

Next Steps
==========

After analyzing spectra:

1. :doc:`gaussian_fitting` for line profile analysis
2. :doc:`continuum_subtraction` if needed
3. :doc:`exporting_data` to save results
