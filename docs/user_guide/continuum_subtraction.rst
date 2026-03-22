=======================
Continuum Subtraction
=======================

Remove or model the underlying continuum from spectral data before analyzing emission or absorption lines.

When to Subtract Continuum
==========================

Remove continuum when:

- **Measuring weak lines**: Enhances signal-to-noise
- **Analyzing line profiles**: Isolates pure line signal
- **Comparing different regions**: Removes illumination differences
- **Measuring equivalent widths**: Required for accurate measurement

Keep continuum when:

- **Measuring total flux**: Includes all emission
- **Studying continuum properties**: Necessary for analysis
- **Creating broad-band images**: Need full spectrum

Understanding the Continuum
===========================

The continuum consists of:

- **Stellar/nebular emission**: Underlying hot gas
- **Dust emission**: Thermal dust radiation
- **Scattered light**: Reflected starlight
- **Instrumental effects**: Baseline issues

Continuum Models
================

IFU Analyzer supports several continuum fitting models:

**Linear Model**

Simplest: Straight line through spectrum

.. math::

   f(\lambda) = a + b\lambda

- Best for: Small wavelength ranges, modest continuum
- Quick to compute
- May miss curvature

**Quadratic Model**

Parabolic curve

.. math::

   f(\lambda) = a + b\lambda + c\lambda^2

- Best for: Moderate wavelength ranges
- Captures simple slope changes
- Standard choice

**Cubic Model**

Higher-order polynomial

.. math::

   f(\lambda) = a + b\lambda + c\lambda^2 + d\lambda^3

- Best for: Complex continuum shapes
- Risk of overfitting
- Use mainly when needed

**Spline Model**

Piecewise polynomial curves

- Best for: Very complex continuum
- More flexible than polynomials
- Requires careful knot placement

**Blackbody Model**

Thermal emission from hot object

.. math::

   f(\lambda) \propto \frac{1}{\lambda^5(e^{hc/\lambda k T} - 1)}

- Best for: Stellar/thermal sources
- Requires temperature estimate
- Physically motivated

**Power-Law Model**

.. math::

   f(\lambda) = A \lambda^{-\alpha}

- Best for: AGN, synchrotron sources
- Defines slope
- Simple, often physical

How Continuum Subtraction Works
===============================

**Step 1: Define Continuum Regions**

Select wavelength ranges that represent pure continuum:

- Avoid strong spectral lines
- Choose regions on both sides of feature to analyze
- Ensure enough pixels for stable fit

**Step 2: Fit Continuum**

Software fits selected model through continuum regions:

- Ignores line pixels
- Computes best-fit parameters
- Shows fit overlay on spectrum

**Step 3: Subtract**

Continuum-subtracted spectrum:

.. math::

   f_{cont-sub}(\lambda) = f_{obs}(\lambda) - f_{cont}(\lambda)

**Step 4: Analyze**

Now analyze continuum-subtracted spectrum:

- Line profiles stand out
- Easier to measure equivalent widths
- More accurate Gaussian fits

Interactive Continuum Subtraction
=================================

**Simple Method**

1. Click **Continuum Subtraction** button in toolbar
2. Dialog appears
3. Select model from dropdown (Linear, Quadratic, etc.)
4. Click **Select Continuum Regions** on spectrum
5. Drag to select continuum-only areas
6. Preview shows fitted continuum
7. Click **Apply**

**Using Mask**

Specify exact pixels for continuum fitting:

1. Tools → Continuum → Advanced
2. **Use Mask**: Check this option
3. Specify pixels to use/exclude:
   - By wavelength range
   - By intensity range (avoid high emission)
   - By custom mask
4. Click **Apply**

**Automatic Continuum Detection**

Let software find continuum:

1. Tools → Continuum → Automatic
2. Algorithm identifies line-free regions
3. Usually works well for typical spectra
4. Manual adjustment available if needed

Full-Cube Continuum Subtraction
===============================

Apply continuum subtraction to entire cube:

**Batch Processing**

1. Tools → Continuum → Fit All Pixels
2. Configuration dialog:
   - **Model selection**: Choose continuum model
   - **Method**: Automatic or manual regions
   - **Parallelization**: Use multiple cores
3. Click **Start**
4. Processed cube saved as new file

**Using Same Model Everywhere**

Single continuum fit applied to all pixels:

1. Define continuum on integrated spectrum
2. Tools → Continuum → Apply to All Pixels
3. Same model applied everywhere
4. Fast but assumes continuum constant spatially

**Per-Pixel Determination**

Find continuum separately for each pixel:

1. Tools → Continuum → Pixel-by-Pixel
2. Each pixel's spectrum analyzed independently
3. Accounts for spatial variations
4. Much slower but more accurate

Visualizing Subtraction
=======================

**Show Before/After**

Display spectra before and after subtraction:

1. Spectrum panel → **Toggle View**
2. Switch between:
   - Original spectrum
   - Continuum model (red curve)
   - Continuum-subtracted (original minus model)
3. Use arrow keys to compare

**Residuals Display**

Show what was subtracted:

1. Spectrum panel → **Show Residuals**
2. Green shaded area: Subtracted continuum
3. Clearly see what you removed

Continuum Fitting Parameters
============================

**Optimization Parameters**

Control fitting behavior:

- **Convergence threshold**: When algorithm stops (low = very precise)
- **Maximum iterations**: Prevent infinite loops
- **Residual tolerance**: Acceptable fitting error

**Model-Specific Parameters**

Blackbody model example:

- **Temperature**: Estimate starting value
- **Scaling**: Flux normalization

Power-law model example:

- **Slope (α)**: Steepness of power-law
- **Normalization**: Absolute flux scaling

**Weighting**

Use uncertainties in fitting:

1. Analysis → Fit Options → Weighted Fit
2. Pixels with larger errors: Lower weight
3. More robust against noisy pixels

Continuum Subtraction Issues
============================

**Issue: Over-subtraction (negative fluxes)**

Fitting curve too high:

- Check continuum region selection
- Try different model (lower order polynomial)
- Manually adjust fit

**Issue: Under-subtraction (residual bump)**

Fitting curve too low:

- Add more continuum regions
- Include broader wavelength range
- Use higher-order polynomial

**Issue: Ringing/Oscillation**

Overfitting artifacts appear:

- Reduce polynomial order
- Use spline with fewer knots
- Switch to simpler model

**Issue: Spatially-varying continuum**

Continuum different at each pixel:

- Use pixel-by-pixel subtraction
- Or define location-dependent regions
- Will be slower

Saving Subtracted Data
======================

**Export Subtracted Cube**

Save continuum-subtracted cube as FITS:

1. File → Export Results → Continuum-Subtracted Cube
2. Specify filename
3. FITS file created with subtracted data
4. Original FITS header preserved

**Save Continuum Model**

Export the fitted continuum:

1. Analysis → Export → Continuum Model
2. Format: 
   - ASCII: Wavelength vs. continuum
   - FITS: Extension with continuum array
3. Useful for documentation and reproducibility

**Save Parameters**

Export continuum model parameters:

1. Analysis → Export Continuum Parameters
2. Saves fitting results:
   - Polynomial coefficients (if used)
   - Temperature (if blackbody)
   - Other model parameters
3. Format: ASCII or FITS

Quality Assessment
==================

**Visual Inspection**

Always check your continuum subtraction:

1. Display original spectrum
2. Display continuum model overlay
3. Display subtracted result
4. Look for:
   - Lines removed properly
   - No over/under-subtraction
   - Flat baseline after subtraction

**Statistical Tests**

Check goodness of subtraction:

1. Analysis → Continuum Quality
2. Shows:
   - Reduced χ²: Goodness of fit
   - Baseline level: Should be ~ 0 after subtraction
   - RMS residuals: Should be noise level

**Comparing Methods**

Test different continuum models:

1. Apply Model 1 (e.g., Linear)
2. Export results
3. Undo, apply Model 2 (e.g., Quadratic)
4. Compare outputs
5. Use model giving best results

Advanced Topics
===============

**Continuum in Multi-Component Spectra**

When multiple features present:

- Be careful not to fit over lines
- Use robust statistics to downweight line pixels
- Consider multiple-component continuum model
- May require manual intervention

**Dust Extinction Correction**

Combine continuum fitting with extinction correction:

1. Model continuum with stellar template
2. Compare to observed continuum
3. Difference attributed to extinction
4. Extinction E(B-V) determined
5. Use to correct extinction

**Narrow vs. Broad Components**

When emission has multiple width components:

- Narrow component: From narrow line region
- Broad component: From broad line region
- May require separate continuum modeling
- Advanced technique

Best Practices
==============

1. **Always visually inspect** the continuum fit
2. **verify continuum regions** don't include lines
3. **Document your method** in analysis notes
4. **Save original data** (never overwrite)
5. **Test sensitivity** to model choice
6. **Be conservative** - under-subtract rather than over
7. **Compare methods** when analyzing critical data

Common Errors to Avoid
======================

- ❌ Fitting over strong emission lines
- ❌ Using too high polynomial order
- ❌ Forgetting to document your procedure
- ❌ Discarding original data before fully confident
- ❌ Using continuum subtraction when not needed
- ✓ Always keep original data alongside processed

Next Steps
==========

After continuum subtraction:

1. :doc:`gaussian_fitting` for line profile analysis
2. :doc:`exporting_data` to save results
3. Perform kinematic or flux analysis on clean spectrum
