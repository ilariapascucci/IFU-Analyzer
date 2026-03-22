===============
Troubleshooting
===============

Solutions for common problems in IFU Analyzer.

Installation Issues
===================

**"Python not found" / "Command not recognized"**

Problem: Cannot run python or pip commands.

Solutions:

1. Check Python installation:

   .. code-block:: bash

      python --version
      # or
      python3 --version

2. If not found, install Python from python.org

3. Add Python to PATH:

   - **Windows**: During Python installation, check "Add Python to PATH"
   - **macOS**: May need to use ``python3`` instead of ``python``
   - **Linux**: Python usually pre-installed, use ``python3``

**"No module named 'PyQt5'" (or other package)**

Problem: Import error when starting application.

Solutions:

.. code-block:: bash

   # Install missing package
   pip install PyQt5  # or numpy, astropy, matplotlib, scipy

   # Or reinstall all dependencies
   pip install -r requirements.txt

**"Permission denied" on macOS/Linux**

Problem: Installation fails with permission error.

Solutions:

.. code-block:: bash

   # Option 1: Use --user flag
   pip install --user PyQt5

   # Option 2: Use sudo (less recommended)
   sudo pip install PyQt5

   # Option 3: Use virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or on Windows:
   venv\Scripts\activate
   pip install PyQt5

File Opening Issues
===================

**"File will not open" / "Invalid FITS file"**

Problem: FITS file won't load in IFU Analyzer.

Check:

1. **File format**: Verify it's actually a FITS file

   .. code-block:: bash

      file mycube.fits

2. **File integrity**: Try opening with external tool (DS9, astropy)

   .. code-block:: python

      from astropy.io import fits
      hdul = fits.open('mycube.fits')
      hdul.info()

3. **File corruption**: Re-download or copy file

Solutions:

- Use "Generic" instrument option if instrument isn't detecting automatically
- Try opening file with DS9 or FITS exploration software first
- If file is valid but won't open, report as bug with file details

**"Wrong instrument type detected"**

Problem: Auto-detection chooses wrong instrument.

Solutions:

1. Manually select correct instrument in dialog
2. Check FITS header keywords match instrument
3. Use "Generic" option to avoid auto-detection
4. Report issue if auto-detection consistently wrong

**"File dimensions seem wrong"**

Problem: Displayed cube size doesn't match expectations.

Check:

1. **FITS header**: View FITS keywords

   .. code-block:: bash

      python -c "from astropy.io import fits; fits.open('file.fits')[0].header"

2. **Multiple HDUs**: File might have multiple data extensions

   - Try opening different extension
   - Menu: File → Advanced → Choose HDU

3. **Transposed axes**: Some tools create transposed data

   Solutions:
   - Note axis order (usually RA × Dec × λ)
   - Use external tool to verify
   - Report if consistently wrong

Display Issues
==============

**"Data looks too dark/bright"**

Problem: Intensity map not showing expected features.

Solutions:

1. **Adjust intensity range**:

   - Click "Auto Scale" button
   - Manually set vmin/vmax sliders
   - Try different color scales (Linear, Log, etc.)

2. **Change colormap**:

   - Click "Colormap" button
   - Select different colormap
   - Try Hot, Cool, or Viridis

3. **Verify wavelength**:

   - Check current wavelength slider
   - Jump to expected wavelength
   - Verify you're looking at emission region

**"Text/buttons too small" (or too large)**

Problem: Interface doesn't scale properly.

Solutions:

- Tools → Preferences → Font Size: Increase/Decrease
- Application must restart for changes
- Check system display scaling settings

**"Colors don't look right" (washed out, etc.)**

Problem: Colors appear wrong or washed out.

Solutions:

1. Check monitor gamma/calibration
2. Try different colormaps
3. Adjust vmin/vmax for better contrast
4. Check color scale (Linear vs. Log)

**"Very slow performance" / "Freezing"**

Problem: Application runs slowly or becomes unresponsive.

Solutions:

1. **Check data cube size**:

   - Status bar shows dimensions
   - Very large cubes (>1 GB) cause slowness
   - Use downsampled version for initial exploration

2. **Reduce processing load**:

   - Enable Display Decimation (Tools → Performance)
   - Reduce display resolution (Tools → Resolution)
   - Close other applications

3. **Limit analysis scope**:

   - Work with sub-regions rather than full cube
   - Fit smaller wavelength ranges

4. **Upgrade hardware**:

   - Close unnecessary applications
   - Increase RAM if fitting large cubes
   - Use SSD for data storage

Spectral Analysis Issues
========================

**"Spectrum looks noisy"**

Problem: Spectrum has excessive noise.

Solutions:

1. **Smooth/rebin**: 

   - Right-click spectrum → Rebin
   - Specify bin size (e.g., 3-5 pixels)
   - Reduces noise but loses resolution

2. **Apply median filter**:

   - Right-click spectrum → Median Filter
   - Smooths while preserving features

3. **Select larger region**

   - Larger spatial regions → better S/N
   - Use rectangle tool instead of point

4. **Verify data quality**:

   - Check if noise is real (compare other positions)
   - Verify wavelength calibration

**"Lines not visible / too faint"**

Problem: Expected spectral features don't appear.

Check:

1. **Wavelength accuracy**:

   - Verify wavelength calibration
   - Compare known emission line positions
   - Check against line catalog

2. **Brightness/scaling**:

   - Adjust vmin/vmax
   - Try logarithmic color scale
   - Select actual bright region (check intensity map)

3. **Data quality**:

   - Is region actually bright in intensity map?
   - Check continuum level
   - Verify total flux in region

4. **Hidden by continuum**:

   - Perform continuum subtraction first
   - Then line becomes more visible

Gaussian Fitting Issues
=======================

**"Fits look bad / χ² very high"**

Problem: Gaussian fits have poor quality.

Solutions:

1. **Improve initial guess**:

   - Fit → Options → Initial Guess
   - Provide better starting wavelength/width/amplitude
   - Better initial guess → better convergence

2. **Adjust wavelength range**:

   - Fit → Wavelength Range
   - Make sure fitting only includes line
   - Too broad range includes noise

3. **Subtract continuum first**:

   - Tools → Continuum Subtraction
   - Remove baseline before fitting
   - Much better fits without background

4. **Check data quality**:

   - Look at raw spectrum
   - Is profile reasonable (not noisy)?
   - Multiple peaks? (Use multi-component fitting)

5. **Try different method**:

   - Robust fitting for noisy data
   - Multi-component fitting for blended lines
   - Manual adjustment of initial guess

**"Fits fail for many pixels" (red in success mask)**

Problem: Full-cube fitting has low success rate.

Check:

1. **S/N threshold**:

   - Low S/N pixels may not fit well
   - Threshold may be too strict
   - Adjust χ² threshold in quality map

2. **Line parameters**:

   - Wavelength range too narrow
   - Expected line absent in many pixels
   - Continuum level too high

3. **Data issues**:

   - Certain regions may be bad/masked
   - Detector artifacts possible
   - Verify region actually contains line

Solutions:

- Lower quality threshold to accept more fits
- Manually check worst-fitting regions
- Consider using mask/exclude bad pixels

**"Parameters unrealistic"**

Problem: Fit parameters (amplitude, width, etc.) are unphysical.

Solutions:

1. **Set constraints**:

   - Analysis → Fit Options → Constraints
   - Set min/max for each parameter
   - Prevents non-physical solutions

2. **Improve wavelength calibration**:

   - Verify wavelength solution
   - Compare against known lines
   - Adjust if necessary

3. **Manual inspection**:

   - Look at several fits visually
   - Compare to fitted curve
   - Determine if physically unreasonable

Continuum Subtraction Issues
=============================

**"Over-subtraction" (negative fluxes)**

Problem: Continuum subtraction removes too much.

Solutions:

1. **Adjust fitting regions**:

   - Select continuum regions more carefully
   - Avoid selecting near absorption features
   - Ensure selected regions are truly continuum

2. **Try simpler model**:

   - Use Linear instead of Quadratic
   - Lower polynomial order reduces artifacts

3. **Manual adjustment**:

   - Visually inspect fit
   - See if curve too high
   - Manually lower slightly before applying

**"Under-subtraction" (residual bump)**

Problem: Continuum not fully removed.

Solutions:

1. **Use higher-order polynomial**:

   - Try Quadratic or Cubic
   - Captures continuum curvature better

2. **Expand fitting regions**:

   - Include wider wavelength range
   - More data gives better fit

3. **Different model**:

   - Try Spline instead of polynomial
   - May fit complex curves better

**"Ringing/Oscillation" (artifacts)**

Problem: Subraction creates oscillating pattern.

Solutions:

1. **Reduce polynomial order**:

   - Overfitting causes artifacts
   - Use simpler model

2. **Switch to Spline**:

   - Smoother response than high-order polynomial
   - Fewer artifacts

3. **Improve fitting regions**:

   - Better region selection
   - More stable fit

Export Issues
=============

**"File too large"**

Problem: Exported file much larger than expected.

Solutions:

- **Compress FITS**: Use gzip compression (reduces ~50%)
- **Export subset**: Rather than full cube, specify range
- **Delete temporary files**: May have backups accumulating

**"File won't open in other software"**

Problem: Exported file can't be read by external tools.

Check:

1. **File format compatibility**:

   - ASCII files: Open in any text editor
   - FITS: Use DS9 or astropy.io.fits
   - Try with different reader

2. **File corruption**:

   - Check file size > 0
   - Try re-exporting
   - Check disk space during export

3. **Format issues**:

   - Verify correct format selected
   - Try alternative format
   - Test with simple file first

Solutions:

- Export to standard FITS format (most compatible)
- Test with free viewer (DS9) first
- Check file metadata for clues

Performance Issues
==================

**"Analysis runs very slowly"**

Problem: Fitting or processing takes excessive time.

Solutions:

1. **Use fewer processors** (if applicable):

   - Analysis → Options → Processes
   - Start with 1-2, increase if faster

2. **Reduce data scope**:

   - Fit smaller wavelength range
   - Work with sub-region
   - Downsample cube resolution

3. **Upgrade hardware**:

   - Add more RAM if running out
   - Faster CPU helps processing
   - SSD much faster than HDD

**"Memory running out"**

Problem: "Out of memory" or "Cannot allocate" errors.

Solutions:

1. **Close other applications**: Frees RAM

2. **Work with smaller cubes**:

   - Downsample data
   - Analyze sub-regions
   - Split into wavelength ranges

3. **Add physical RAM**:

   - Permanent solution for large analyses

4. **Use downsampled cube**:

   - Tools → Resolution
   - Temporarily reduce resolution for speed

General Debugging
=================

**When something goes wrong:**

1. **Reproduce the issue**:

   - Note exact steps taken
   - Try with different data
   - Isolated or widespread?

2. **Check related documentation**:

   - :doc:`faq` for common issues
   - User guide for feature info
   - Keyboard shortcuts reference

3. **Inspect error messages**:

   - Error dialog usually explains problem
   - Error log may have details
   - Copy exact error message

4. **Search for solutions**:

   - Search this troubleshooting guide
   - Check project FAQ
   - Search online for error message

5. **Report as bug** (if not documented):

   - Provide exact error message
   - Include data file example (if shareable)
   - List steps to reproduce
   - Specify OS and Python version

Performance Profiling
=====================

If customizing or developing:

.. code-block:: python

   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   
   # ... code to profile ...
   
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)  # Top 20 functions

Getting Help
============

1. Check this guide's troubleshooting section
2. Review :doc:`faq` for common Q&A  
3. See online documentation
4. Consult project issue tracker
5. Contact development team

Next Steps
==========

If this doesn't solve your problem:

- See :doc:`faq` for frequently asked questions
- Check documentation for related topics
- File a bug report with details
