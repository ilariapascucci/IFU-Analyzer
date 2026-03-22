===
FAQ
===

Frequently Asked Questions about IFU Analyzer.

Getting Started
===============

**Q: I just installed IFU Analyzer. Where do I start?**

A: Follow these steps:

1. Read :doc:`quickstart` for a 5-minute overview
2. Install sample data (or find your own FITS cube)
3. Launch the application: ``python main.py``
4. File → Open FITS to load your first data cube
5. Explore the interface following Quick Start guide

**Q: Do I need to know Python?**

A: No! The GUI handles everything. Advanced users can write Python scripts for batch processing, but normal use requires no coding.

**Q: What FITS data can I use?**

A: Any standard FITS cube with:

- 2D or 3D data (images or cubes)
- Wavelength calibration (FITS header keywords)
- Valid coordinate system (optional but useful)

Supported instruments: MUSE, ALMA, MIRI, NIRSpec, or any generic FITS.

Data & Files
============

**Q: Where do I find example data?**

A: IFU Analyzer doesn't include sample data. You'll need to:

- Use your own observational data
- Download public data (e.g., from ESO, STScI, ALMA archives)
- Ensure it's in FITS format

**Q: Can I use .fits.gz (compressed) files?**

A: Yes! IFU Analyzer handles compressed FITS files automatically.

**Q: How large can cubes be?**

A: Practically limited by your computer's RAM:

- Small cubes (< 100×100×100): No issues
- Medium cubes (up to 1 GB): Moderate slowness
- Large cubes (> 1 GB): Expect delays, consider using downsampled regions

**Q: Can I use non-IFU spectroscopy data?**

A: Yes. Load as "Generic" instrument. As long as it's valid FITS, it will open.

**Q: How do I load multiple files?**

A: Use File → Open FITS for each file. Results from previous analysis remain in memory but aren't displayed. Use File → Recent Files to switch.

Visualization
==============

**Q: What's the best colormap for publications?**

A: Use perceptually uniform colormaps:

- **Viridis** (most common)
- **Plasma**
- **Inferno**

Avoid Jet and rainbow colormaps (introduce artifacts).

**Q: Why can't I see my features?**

A: Try:

1. Adjust vmin/vmax (intensity range) sliders
2. Change color scale (Linear → Logarithmic)
3. Try different colormap
4. Verify you're looking at correct wavelength
5. Use intensity map to find bright region

**Q: How do I zoom in/out?**

A: Multiple methods:

- Mouse scroll wheel
- Ctrl+Plus/Ctrl+Minus keys
- View → Zoom menu
- Click zoom buttons in toolbar

**Q: How do I save/print an image?**

A: File → Export Visualization

- Choose format (PNG, PDF, JPEG)
- Select resolution (1×, 2×, 4×)
- Click Export

**Q: Can I create 3-color RGB composites?**

A: Yes, but requires manual setup:

1. Select three wavelengths
2. Create three intensity maps
3. Export each as separate image
4. Combine in image editor (Photoshop, GIMP, Python)

Analysis
========

**Q: When should I subtract the continuum?**

A: Before Gaussian fitting to:

- Isolate line features
- Improve fit quality
- Get more accurate line measurements

Use when analyzing narrow emission or absorption lines.

**Q: Which Gaussian fitting method should I use?**

A:

- **Interactive (single pixel)**: For exploring, testing
- **Full-cube**: For mapping, creating parameter maps
- **Multi-component**: For blended or complex profiles

**Q: How do I know if my fit is good?**

A: Check:

- **χ²**: Lower is better (ideally < 1)
- **R²**: Closer to 1 is better (ideally > 0.95)
- **Visual inspection**: Does curve match data?
- **Residuals**: Should be random, no patterns

**Q: What does "χ² = 0.5" mean?**

A: Chi-squared measures fit quality:

- χ² < 0.1: Excellent fit
- χ² 0.1-1: Good fit  
- χ² 1-10: Marginal fit
- χ² > 10: Poor fit

Your χ² of 0.5 is a good fit!

**Q: Can I fit multiple lines simultaneously?**

A: Yes:

- Tools → Spectral Analysis → Multi-Line Fit
- Select lines to fit together
- All fitted with shared continuum

**Q: What's "optical depth" and do I need it?**

A: Optical depth (τ) measures how opaque gas is:

- τ << 1: Optically thin (see through)
- τ >> 1: Optically thick (absorbed)

Important for interpreting source properties. Derive from line ratios if available.

Export & Results
================

**Q: What's the best format to export?**

A: Depends on use:

- **FITS**: For analysis in Python/DS9/astronomy software
- **ASCII**: For statistics packages or spreadsheets
- **PNG/PDF**: For presentations and papers

**Q: How do I export multiple spectra?**

A:

1. Select region
2. File → Export Spectrum (exports that region)
3. Select different region
4. Repeat

Or use: File → Batch Export Spectra

**Q: Can I get results as a table (spreadsheet)?**

A: Yes:

- Analysis → Export as Table (CSV, .xlsx)
- Opens in Excel or spreadsheet software
- Includes all fit parameters

**Q: Do exported FITS files preserve my WCS/coordinates?**

A: Yes, full WCS is preserved by default. Verify it's readable with:

.. code-block:: python

   from astropy.io import fits
   from astropy.wcs import WCS
   wcs = WCS(fits.getheader('file.fits'))

**Q: Can I export publication-ready figures?**

A: Yes:

- File → Export Visualization → PDF
- Select 2-4× resolution for high quality
- Include colorbars and labels as needed

Performance & Troubleshooting
==============================

**Q: Application is slow. What can I do?**

A: Try:

1. Enable Display Decimation (Tools → Performance)
2. Reduce display resolution
3. Close other applications
4. Use downsampled cube for exploration
5. Upgrade computer RAM/CPU

**Q: I get an error opening a file. What's wrong?**

A: Try:

1. Verify file is valid FITS (open with DS9 first)
2. Try "Generic" as instrument
3. Check FITS header is valid
4. Try opening different HDU (File → Advanced)

See :doc:`troubleshooting` for detailed solutions.

**Q: Why is cube fitting so slow?**

A: Large cubes (> 500×500 pixels) slow because:

- Many pixels to process
- Complex fitting at each pixel
- Solution: Use fewer cores, smaller wavelength range, or subset region

**Q: How do I interpret fit quality maps?**

A: 

- **Green region in success mask**: Good fits
- **Red region**: Poor fits or no signal
- **High χ² map**: Shows where fits worst
- Investigate red/high-χ² regions manually

Features & Methods
==================

**Q: What's different about CO flux extraction?**

A: Specialized tool for CO emission:

- Measures multiple CO transitions
- Calculates excitation temperature
- Determines optical depth
- Useful for molecular mass estimates

**Q: Does IFU Analyzer support polarimetry?**

A: Not directly. Import polarimetric data as generic FITS and analyze manually.

**Q: Can I do 3D analysis (not just 2D+λ)?**

A: Not directly in GUI. Use Python modules for advanced 3D analysis:

.. code-block:: python

   from cube_explorer import data_loader
   cube = data_loader.load_fits('data.fits')
   # Custom 3D analysis...

**Q: Does IFU Analyzer support redshift-corrected analysis?**

A: Not automatic. You can:

- Manually shift wavelength scale
- Do analysis in velocity (km/s)
- Export and analyze in external tools

**Q: Can I mask bad pixels?**

A: Limited support. Workarounds:

- Select good region only
- Set very strict quality thresholds in fitting
- Use external tools to create pixel masks

Instruments & Data
===================

**Q: Which instrument should I use my data with?**

A: If uncertain, use "Generic". Or specify based on:

- **MUSE**: VLT optical spectroscopy (4750-9350 Å)
- **NIRSpec**: JWST near-infrared (0.6-5.3 μm)
- **MIRI**: JWST mid-infrared (5-27 μm)
- **ALMA**: Millimeter/submillimeter

**Q: How do I combine data from multiple instruments?**

A: 

1. Load and analyze each separately
2. Export results from each
3. Combine/compare results externally
4. IFU Analyzer designed for single instruments

**Q: Can I use ALMA imaging (not spectra)?**

A: Yes, as long as it's in FITS format. Load as generic 2D image.

**Q: Does IFU Analyzer support time-variable data?**

A: Not specifically. For time-series:

1. Analyze separate epochs separately
2. Export results from each
3. Compare changes between epochs
4. Use external tools for analysis

Contributing & Development
===========================

**Q: How do I contribute to IFU Analyzer?**

A: 

- Report bugs with detailed info
- Suggest features (provide use cases)
- Contribute code improvements (fork and pull request)
- Share data/examples for testing

**Q: I want to extend IFU Analyzer with custom functionality. How?**

A: 

1. Learn Python/PyQt5
2. Study existing code structure
3. Fork repository
4. Create custom modules
5. Submit pull request if useful for others

**Q: Is there an API for batch processing?**

A: Yes! Use Python directly:

.. code-block:: python

   from cube_explorer import data_loader, gaussian_fitter
   
   cube = data_loader.load_fits('data.fits')
   results = gaussian_fitter.fit_cube(cube)

See API documentation for details.

License & Use
=============

**Q: What's the license for IFU Analyzer?**

A: Check LICENSE file in repository (typically open source).

**Q: Can I use IFU Analyzer for commercial purposes?**

A: Check license terms. Most open source licenses allow use but may require attribution.

**Q: How do I cite IFU Analyzer in a paper?**

A: Check repository for citation guidance. Generally:

> Pascucci, I. (2024). IFU Analyzer: An interactive tool for analyzing integral field unit spectroscopic data. Zenodo.

**Q: Can I redistribute IFU Analyzer?**

A: Check license. Typically allowed with attribution.

Citation & Acknowledgments
==========================

**Q: Do I need to acknowledge IFU Analyzer use?**

A: Depends on license. Open source typically requests acknowledgment. Check repository.

**Q: What about instrument acknowledgments?**

A: If using data from specific telescope:

- Include facility acknowledgment (JWST, VLT, ALMA, etc.)
- Credit funding agencies
- Acknowledge archival sources

Still Cannot Find Answer?
===========================

1. Search :doc:`troubleshooting` guide
2. Check related user guide sections
3. Try online documentation
4. File an issue on project repository
5. Contact development team

Most Common Issues Quick Links
==============================

- Slow performance? → :doc:`troubleshooting` Performance section
- File won't open? → :doc:`troubleshooting` File Opening section
- Fits look bad? → :doc:`troubleshooting` Gaussian Fitting section
- Wrong colors? → :doc:`troubleshooting` Display Issues section

Happy analyzing!
