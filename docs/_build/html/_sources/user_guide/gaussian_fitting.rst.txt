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

Used for:
- Kinematic analysis (velocity maps)
- Temperature estimation (width measures)
- Flux measurements (integrated area)
- Quality assessment (residuals)

Interactive Single-Pixel Fitting
===============================

**Quick Fit (Interactive)**

1. Click on spectrum display to select wavelength region
2. Click **Fit Single Pixel** button (or press F)
3. Gaussian overlay appears on spectrum
4. Fit parameters shown:
   
   - **Amplitude** (A): Peak intensity
   - **Center** (μ): Line center wavelength
   - **Sigma** (σ): Standard deviation
   - **FWHM**: Full Width Half Maximum
   - **Area**: Integrated flux
   - **χ²**: Fit quality (lower is better)
   - **R²**: Coefficient of determination (higher is better)

5. **Accept** to keep fit, **Reject** to discard

**Manual Adjustment**

Refine the fit by dragging:

- **Peak marker**: Adjust amplitude
- **Center line**: Move center wavelength
- **Width handles**: Adjust line width
- Press **Enter** to finalize

**Fit Quality Indicators**

- **χ² < 0.1**: Excellent fit
- **χ² 0.1-1.0**: Good fit
- **χ² 1.0-10**: Marginal fit
- **χ² > 10**: Poor fit, try different method
- **R² > 0.95**: Excellent fit
- **R² > 0.80**: Acceptable fit

Full-Cube Gaussian Fitting
==========================

Fit all pixels in the cube simultaneously:

**Starting Cube Fit**

1. Click **Fit Full Cube** button (or press C)
2. Configuration dialog:
   
   - **Wavelength range**: Lines to analyze
   - **Continuum subtraction**: Yes/No
   - **Number of processors**: Use all cores for speed
   - **Preview**: Show first fit before processing
   
3. Click **Start** to begin

**Progress Monitoring**

- Progress bar shows % complete
- Current pixel displayed
- Estimated time remaining
- Cancel button available

**Output Maps**

After fitting, created maps include:

- **Amplitude Map**: Peak intensity at each pixel
- **Center Map**: Line center wavelength (shows kinematics)
- **Sigma Map**: Line width (σ values)
- **FWHM Map**: Full width at half maximum
- **Area Map**: Integrated line flux
- **Chi-Squared Map**: Fit quality at each pixel
- **Success Mask**: Good vs. failed fits

Viewing Fit Results
===================

**Display Parameter Map**

1. Analysis → View Fit Results
2. Select which map to display:
   
   - Amplitude → Star (★) symbol in icon bar
   - Center → Diamond symbol
   - Width → ∑ symbol
   
3. Map appears in main display

**Map Overlays**

Display multiple maps simultaneously:

1. Analysis → Overlay Maps
2. Select primary and secondary maps
3. Primary: Color, Secondary: Transparency overlay

**Quality Map**

Visualize which fits succeeded:

1. Analysis → Success Mask
2. Green: Good fits (χ² < threshold)
3. Red: Failed fits
4. Adjust threshold in dialog

Fitting Parameters
==================

**Wavelength Range**

Specify which spectral region to fit:

1. Select on spectrum (click + drag)
2. Or enter manually: **From**, **To** wavelength
3. Gaussian fit limited to this range

**Initial Guess**

Improve fitting by providing initial guess:

1. Fit → Options → Initial Guess
2. Estimate:
   - **Center**: Approximate line wavelength
   - **Sigma**: Approximate width
   - **Amplitude**: Approximate peak height
3. Better initial guess = faster convergence

**Constraints**

Limit parameter ranges:

1. Fit → Options → Constraints
2. Set min/max for each parameter:
   - Amplitude: 0 to max possible flux
   - Center: Valid wavelength range
   - Sigma: Minimum line width
3. Prevents non-physical solutions

**Weight by Uncertainty**

Use data uncertainties in fitting:

1. Analysis → Weighted Fitting
2. If error column exists in data, used automatically
3. Pixels with high uncertainty: lower weight
4. Results in more robust fits

Multi-Component Fitting
========================

Fit multiple Gaussians to deblended lines:

**Two-Component Fit**

1. Spectrum shows blended features
2. Analysis → Multi-Component Fit → Two Gaussians
3. Software identifies two components
4. Each component fitted separately
5. Results for each component displayed

**Three+ Components**

For more complex profiles:

1. Analysis → Multi-Component Fit → Custom Number
2. Specify number of components
3. Manual adjustments may be needed
4. Each component's parameters extracted

Fitting Multiple Lines
======================

**Simultaneous Multi-Line Fit**

Fit several lines at once:

1. Tools → Spectral Analysis → Multi-Line Fit
2. Select lines to fit
3. All fitted with shared continuum
4. Results for each line displayed
5. More robust when lines close together

Baseline/Continuum Fitting
==========================

**Fitting Polynomial Continuum**

Remove underlying continuum before line fitting:

1. Analysis → Fit Continuum
2. Select continuum fitting method:
   - **Linear**: Straight line
   - **Quadratic**: Parabola
   - **Cubic**: Higher-order polynomial
   - **Spline**: Flexible curve
3. Specify baseline wavelength regions
4. Continuum subtracted from spectrum

**Automatic Baseline**

Let software identify continuum:

1. Analysis → Auto Baseline
2. Software identifies line-free regions
3. Baseline computed from those regions
4. Typically works well for isolated lines

Exporting Fit Parameters
========================

**Save Parameter Maps as FITS**

1. File → Export Results
2. Select which maps to save:
   - Check Amplitude, Center, Sigma, FWHM, Area
   - Check Chi-Squared, Success Mask
3. Specify output filename
4. FITS file created with extensions for each map

**Save Parameter Table**

Export individual pixel parameters as ASCII:

1. Analysis → Export Parameters
2. Choose format:
   - Tab-delimited (.txt)
   - CSV (.csv)
   - FITS table (.fits)
3. Columns: X, Y, Amplitude, Center, Sigma, Area, Chi²

**Save Residuals**

Export residual cube (data minus fit):

1. File → Export Residuals
2. Cube dimensions match input
3. Each pixel: residuals = data - Gaussian

Quality Assessment
==================

**Examining Residuals**

Check goodness of fit via residuals:

1. File → Export Residuals
2. Visualize residuals cube
3. Random residuals: Good fit ✓
4. Systematic patterns: Bad fit, try different method ✗

**Chi-Squared Analysis**

Assess fit quality:

1. Analysis → View Results → Chi-Squared Map
2. Color scale: Brighter = worse fit
3. Investigate red pixels for issues
4. Manual refit of problem regions if needed

**Visual Inspection**

Always examine fitted spectra:

1. For each region, view:
   - Fitted curve overlaid on data
   - Residual difference
   - Confidence bands
2. Look for:
   - Systematic deviations
   - Poor fit in wings
   - Underlying noise patterns

Advanced Fitting Options
=========================

**Robust Fitting**

Reduce impact of outliers:

1. Analysis → Fit Options → Robust Mode
2. Uses outlier-resistant algorithm
3. Downweights extreme values
4. Better for noisy data

**Regularization**

Prevent overfitting:

1. Analysis → Fit Options → Regularization
2. Choose strength level
3. Favors smoother solutions
4. Good for noisy data

**Parallelization**

Speed up full-cube fitting:

1. Analysis → Fit Options → Processes
2. Specify number of processors to use
3. Typically: Use all available cores
4. Dramatic speed improvement for large cubes

Troubleshooting Fitting
=======================

**Fits look bad (high χ², wrong parameters)**

- Check wavelength calibration
- Try different initial guess
- Examine residuals for patterns
- Consider multi-component fit if blended
- Use robust fitting for noisy data

**Fits fail for some pixels (red in mask)**

- Those pixels may have no signal
- Baseline may be too high
- Line may be noise
- Check individual spectrum quality

**Fitting is slow**

- Use fewer processors for initial testing
- Fit smaller wavelength range
- Consider downsampling cube
- Upgrade computer hardware if repeatedly slow

**Parameters unrealistic**

- Check constraints are appropriate
- Verify wavelength calibration
- Compare with other methods
- Manually check a few fits

Best Practices
==============

1. **Always inspect fits visually**
2. **Check χ² values for all pixels**
3. **Examine residuals for systematic errors**
4. **Verify constraints are physically sensible**
5. **Compare with other analysis methods when possible**
6. **Document your fitting procedure**
7. **Save both data and results for reproducibility**

Next Steps
==========

After fitting Gaussians:

1. Export results using :doc:`exporting_data`
2. Use parameter maps for kinematic analysis
3. Calculate physical quantities from line properties
4. Compare with other datasets
