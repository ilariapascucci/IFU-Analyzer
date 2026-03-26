==============
Exporting Data
==============

Save your analysis results in various formats for further processing and publication.

Export Options Overview
=======================

IFU Analyzer provides several export capabilities:

========================== ==================== ================================================
What to Export             Format                Purpose
========================== ==================== ================================================
**Spectrum**               ASCII, FITS           Publication, further analysis
**Parameter Maps**         FITS                  Imaging analysis tools (DS9, CASA, Python)
**Residuals**              FITS                  Quality assessment, error analysis
**Visualization**          PNG, JPEG, PDF        Presentations, publications
**Analysis Results**       ASCII/CSV table       Statistics, databases
**Animation**              GIF, MP4, WebM        Web, presentations
========================== ==================== ================================================

Exporting Spectra
=================

**ASCII Format**

Export spectrum as tab-delimited text:

1. Right-click spectrum panel → **Export Spectrum**
2. Format: ASCII
3. Specify filename with .txt extension
4. File created with columns:

.. code-block:: text

   Wavelength(um)    Flux(mJy)    Error(mJy)
   4.5000            12.34        0.56
   4.5012            12.41        0.55
   ...

**FITS Format**

Export as FITS binary table:

1. Spectrum panel → **Export Spectrum**
2. Format: FITS
3. Creates 1D FITS file with:
   - Primary header (instrument, date, etc.)
   - Binary table extension (wavelength, flux, error columns)

**Normalized vs. Original**

Choose which version to export:

- **Original**: Raw spectrum values
- **Normalized**: Divided by median/maximum
- **Continuum-subtracted**: Baseline removed

Select in export dialog before clicking **Export**.

Exporting Parameter Maps
========================

**Export Gaussian Fitting Results**

Save maps created from full-cube fitting:

1. File → **Export Results**
2. Select maps to include:
   ☐ Amplitude Map
   ☐ Center Map (kinematics)
   ☐ Sigma Map (line width)
   ☐ FWHM Map
   ☐ Area Map (flux)
   ☐ Chi-Squared Map (fit quality)
   ☐ Success Mask (good/bad fits)
3. Output format: FITS (multi-extension)
4. Specify filename

**FITS Output Structure**

Multi-extension FITS file created:

.. code-block:: text

   Primary HDU
   ├─ Header: Instrument, wavelength, etc.
   ├─ Data: None (metadata only)
   
   Extension 1: Amplitude Map
   ├─ Header: Units, wavelength, etc.
   ├─ Data: 2D array (RA x Dec)
   
   Extension 2: Center Map
   ├─ Header: "Kinematics"
   ├─ Data: 2D array
   
   ... (additional extensions for each map)

**Accessing Maps in Other Software**

Load FITS file in analysis tools:

**Python/Astropy:**

.. code-block:: python

   from astropy.io import fits
   
   hdul = fits.open('results.fits')
   amplitude = hdul[1].data  # First extension
   velocity = hdul[2].data   # Second extension
   hdul.close()

**DS9:**

.. code-block:: bash

   ds9 results.fits

**CASA:**

.. code-block:: python

   import casatasks
   # Maps can be converted to CASA images

**Image Export Formats**

Export intensity map as image:

1. Display desired map/wavelength
2. File → **Export Visualization**
3. Choose format:
   - **PNG**: Lossless, good for web (default)
   - **JPEG**: Lossy, smaller file size
   - **PDF**: Vector format, publication quality
   - **SVG**: Scalable vector

4. Resolution options:
   - 1×: Current display size
   - 2×: High resolution
   - 4×: Very high resolution
   - Custom: User-specified DPI

Exporting Residuals
===================

**Export Residuals Cube**

Save residuals from Gaussian fitting:

1. File → **Export Residuals**
2. Residuals = Data - Gaussian Fit
3. Same dimensions as input cube
4. Format: FITS (3D)
5. Useful for:
   - Assessing fit quality
   - Detecting systematic patterns
   - Finding outliers/bad pixels

**Batch Export Residuals**

Export for multiple wavelength ranges:

1. Analysis → **Batch Export Residuals**
2. Specify wavelength ranges
3. One residuals file per range
4. Each file contains 3D residuals

**Using Residuals Outside IFU Analyzer**

Analyze residuals with Python:

.. code-block:: python

   from astropy.io import fits
   import numpy as np
   
   # Load residuals
   residuals = fits.getdata('residuals.fits')
   
   # Calculate RMS
   rms = np.sqrt(np.mean(residuals**2))
   
   # Find outliers
   bad_pixels = np.abs(residuals) > 3*rms

Batch Export
============

**Export Multiple Wavelengths**

Create a sequence of exports:

1. File → **Batch Export**
2. Configure:
   - **Starting wavelength**
   - **Ending wavelength**
   - **Step size** (e.g., Every 10 Å)
   - **What to export**: Spectrum, map, image, etc.
   - **Output directory**: Where to save

3. Click **Start** to begin

Example: Export spectra every 10 Å from 4000-5000 Å:

.. code-block:: text

   spectrum_4000.fits
   spectrum_4010.fits
   spectrum_4020.fits
   ...
   spectrum_4990.fits
   spectrum_5000.fits

Exporting Analysis for Publication
===================================

**Complete Results Package**

Create publication-ready output:

1. File → **Export for Publication**
2. Automatically includes:
   - Parameter maps (FITS)
   - Extracted spectra (ASCII/FITS)
   - Metadata file (parameter summary)
   - Visualizations (PDF quality)
311. Specifies archive format

3. Output: Compressed archive (.tar.gz or .zip)

**Included in Archive**

.. code-block:: text

   publication/
   ├── data/
   │   ├── amplitude_map.fits
   │   ├── center_map.fits
   │   ├── width_map.fits
   │   └── spectra/
   │       ├── spectrum_region1.txt
   │       ├── spectrum_region2.txt
   │       └── ...
   ├── figures/
   │   ├── map_amplitude.pdf
   │   ├── map_velocity.pdf
   │   ├── spectrum_overview.pdf
   │   └── ...
   ├── metadata.txt
   └── README.txt

**Metadata File Contents**

Automatic documentation:

.. code-block:: text

   Analysis Date: 2024-01-15
   Data File: cosmic_object.fits
   Instrument: JWST/NIRSpec
   
   Analysis Steps:
   1. Gaussian fitting: 4.5-5.0 μm
   2. Continuum subtraction: Linear model
   3. Parameters extracted: Amplitude, Center, FWHM, Area
   
   Results:
   - 100 × 100 pixels fitted
   - 98% success rate
   - Mean χ² = 0.45

Exporting for External Software
===============================

**For Image Analysis (DS9, Aladin)**

1. Export maps as FITS
2. Both tools directly open .fits files
3. Use standard visualization features

**For Statistical Analysis (Python, R)**

Export as ASCII table:

1. Analysis → **Export as Table**
2. Format: CSV or Tab-delimited
3. Columns: X, Y, Amplitude, Center, FWHM, Area, Chi2

```csv
X,Y,Amplitude,Center,FWHM,Area,Chi2
0,0,45.2,4742.1,2.3,234.5,0.32
0,1,43.1,4741.9,2.4,221.3,0.41
...
```

**For Kinematic Analysis (GIPSY, Miriad)**

Export velocity maps:

1. Analysis → **Export Kinematics**
2. Creates velocity cube format
3. Can interface with radio synthesis software

**For Publishing (LaTeX, Word)**

Export as table:

1. Analysis → **Export as LaTeX Table**
2. Creates publication-ready LaTeX code:

.. code-block:: latex

   \begin{table}
   \centering
   \begin{tabular}{cccccc}
   Region & F(CO) & V & W & T & Qual \\
   ...
   \end{tabular}
   \end{table}

Advanced Export Options
=======================

**Custom File Format**

Specify exact export format:

1. File → **Export Advanced**
2. Choose columns to include
3. Select delimiters (tab, comma, space)
4. Set number precision (4 decimals, scientific notation, etc.)
5. Add/remove header lines
6. Specify units

**Compression**

Reduce file size:

- **FITS**: Automatic compression available
- **ASCII**: Use gzip when specified
- **Archive**: Always offers compression (.tar.gz vs. .tar)

**Metadata Inclusion**

Control what information is saved:

1. **Full Header**: All FITS keywords, telescope info
2. **Minimal Header**: Just essential info
3. **No Header**: Data only (dangerous for reproducibility!)
4. Recommendation: Always use **Full Header**

**Versioning**

IFU Analyzer can append version info:

- Automatic filename: spectrum_v1.fits, spectrum_v2.fits
- Timestamp: spectrum_2024-01-15.fits
- Manual suffix: spectrum_final.fits

Troubleshooting Export Issues
=============================

**Problem: File won't open in other software**

- Check file format compatibility
- Try different format (FITS vs. ASCII)
- Verify file isn't corrupted: File size > 0?
- Test with free viewers (DS9 for FITS, notepad for ASCII)

**Problem: Very large file sizes**

- Use FITS compression: reduces 50%
- ASCII files larger than FITS: Try FITS format
- Export subset instead of entire cube
- Delete temporary backup files

**Problem: Missing data in export**

- Ensure regions are properly selected
- Check that analysis completed successfully
- Verify not exporting empty wavelength range
- Review success mask for fitting

**Problem: Coordinates don't match**

- Verify RA/DEC vs. pixel coordinate mode
- Check WCS information in header
- Compare with original FITS file
- Use external tool to verify: FITS header viewer

**Problem: "Permission Denied" error**

- Check folder permissions
- Try exporting to Desktop or Documents
- On Linux: May need write access to directory

Best Practices
==============

1. **Always save raw data alongside processed results**
2. **Include metadata** in export (don't lose information)
3. **Document your analysis** in a separate README
4. **Use version control** for important results
5. **Validate exported data** in external tool before relying on it
6. **Keep original files** unchanged
7. **Document export settings** for reproducibility

Quick Reference
===============

============ ============ =========================================
What         Format       How to Access
============ ============ =========================================
Spectrum     ASCII        Right-click on spectrum → Export
Spectrum     FITS         Right-click on spectrum → Export
Map          FITS         Use data export dialog
Residuals    FITS         Use residuals export dialog
Image        PNG          Use display export options
Parameters   CSV          After fitting, use results export
Archive      ZIP          Bundle results using archive function
============ ============ =========================================

Next Steps
==========

After exporting:

1. Validate results in external tools
2. Create publication figures
3. Archive backup copy
4. Document analysis parameters
5. Share with collaborators

For publication, read our :doc:`/reference/publishing` guide.
