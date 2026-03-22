============
File Formats
============

Supported file formats and their specifications.

FITS Files
==========

**Primary Format**: FITS (Flexible Image Data System)

Standard format for astronomical data:

- **Multi-dimensional**: Supports 1D, 2D, 3D, nD data
- **Headers**: FITS keywords for metadata
- **HDUs**: Multiple extensions in single file
- **Compression**: Gzip compression supported
- **Extensions**: .fits, .fit, .fits.gz

**FITS Structure**

Primary HDU:
- Contains main data (usually image/cube)
- Header with keywords and metadata

Extension HDUs:
- Additional data (variance, mask, processing)
- Each has separate header

**Typical IFU Cube Structure**

.. code-block:: text

   Primary HDU (data)
   ├─ Header
   │  ├─ SIMPLE = T
   │  ├─ BITPIX = -32
   │  ├─ NAXIS = 3
   │  ├─ CRPIX1, CRPIX2, CRPIX3 (reference pixels)
   │  ├─ CRVAL1, CRVAL2, CRVAL3 (reference values)
   │  ├─ CDELT1, CDELT2, CDELT3 (pixel scales)
   │  ├─ CTYPE1, CTYPE2, CTYPE3 (axis types)
   │  ├─ CUNIT1, CUNIT2, CUNIT3 (units)
   │  └─ ... other keywords ...
   └─ Data (2D or 3D array)

ASCII Formats
=============

**Text Spectra Export**

Tab or comma-delimited text:

.. code-block:: text

   Wavelength(um)    Flux(mJy)      Error(mJy)    Flag
   4.500000         12.34          0.56          0
   4.500123         12.41          0.55          0
   4.500246         12.38          0.57          0
   ...

Formats supported:
- Tab-delimited (.txt)
- Comma-separated (.csv)
- Space-delimited (.dat)
- Custom delimiters

**ASCII Table Exports**

Parameter tables:

.. code-block:: text

   RA        Dec       Amplitude  Center    FWHM     Area     Chi2
   153.2341  -27.1234  45.2       4742.1    2.3      234.5    0.32
   153.2356  -27.1245  43.1       4741.9    2.4      221.3    0.41
   ...

FITS Binary Tables
==================

For tabular data in FITS format:

- Column-based structure
- Efficient storage
- Supports multiple data types
- Can include metadata

Example: Fit parameters FITS table

.. code-block:: text

   EXTNAME = 'PARAMETERS'
   COLUMNS:
   - RA (float32): Right Ascension
   - Dec (float32): Declination
   - Amplitude (float32): Peak intensity
   - Center (float32): Line center
   - FWHM (float32): Line width
   - Area (float32): Integrated flux
   - Chi2 (float32): Fit quality

Parameter Map FITS Files
========================

Multi-extension FITS files containing analysis results:

**Extension 0**: Amplitude map
**Extension 1**: Center map (velocity)
**Extension 2**: Sigma/FWHM map
**Extension 3**: Area map (flux)
**Extension 4**: Chi-squared map
**Extension 5**: Success mask

Each extension:
- 2D array (RA × Dec)
- Header with wavelength/units
- WCS information for all

**Opening in Python**

.. code-block:: python

   from astropy.io import fits
   
   # Open file
   hdul = fits.open('parameters.fits')
   
   # Access amplitude map
   amplitude = hdul[0].data
   
   # Access velocity map
   velocity = hdul[1].data
   
   # Close
   hdul.close()

**Opening in DS9**

.. code-block:: bash

   ds9 parameters.fits
   
   # Syntax to load specific extension
   ds9 parameters.fits[1]  # Load first extension

Residuals Cube Format
=====================

3D FITS cube of residuals:

- Dimensions: RA × Dec × Wavelength
- Values: Data - Fitted model
- Same shape as input cube
- Can be visualized as video or analyzed statistically

**Using Residuals**

.. code-block:: python

   residuals = fits.getdata('residuals.fits')
   
   # RMS of residuals
   rms = np.sqrt(np.mean(residuals**2))
   
   # Find outliers
   bad = np.abs(residuals) > 3*rms

WCS (World Coordinate System)
=======================================

Astronomical coordinate system stored in FITS:

**Key Keywords**

.. code-block:: text

   CRPIX1   = 1024.0          / Reference pixel RA
   CRVAL1   = 153.123         / Reference RA (degrees)
   CDELT1   = 0.0001          / Pixel scale (degrees)
   CTYPE1   = 'RA---TAN'      / RA with TAN projection
   
   CRPIX2   = 1024.0          / Reference pixel Dec
   CRVAL2   = -27.456         / Reference Dec (degrees)
   CDELT2   = 0.0001          / Pixel scale (degrees)
   CTYPE2   = 'DEC--TAN'      / Dec with TAN projection
   
   CRPIX3   = 1.0             / Reference pixel wavelength
   CRVAL3   = 4750.0          / Reference wavelength (Angstrom)
   CDELT3   = 0.8             / Wavelength scale (Angstrom)
   CTYPE3   = 'WAVE'          / Wavelength axis

**Using WCS in Python**

.. code-block:: python

   from astropy.wcs import WCS
   
   # Read WCS from FITS
   wcs = WCS(fits.getheader('data.fits'))
   
   # Convert pixel to world coordinates
   sky = wcs.pixel_to_world(100, 200, 500)
   print(f"RA: {sky.ra}, Dec: {sky.dec}")
   
   # Convert world to pixel
   pixel = wcs.world_to_pixel(sky, 4750)

Compressed FITS
===============

**FITS File Compression**

Saves ~50% disk space while maintaining full data integrity:

.. code-block:: text

   Compression Methods:
   - Rice compression (lossless, good for floats)
   - GZIP compression (lossless, good for integers)
   - PLIO compression (legacy)

**Creating Compressed FITS**

.. code-block:: python

   from astropy.io import fits
   
   # Compress existing FITS
   hdul = fits.open('data.fits')
   hdul.writeto('data.fits.gz', compression='rice')

**Automatic Handling**

IFU Analyzer automatically handles compressed FITS:

- Reads .fits.gz seamlessly
- Can export as compressed
- No extra steps needed

Legacy Formats
==============

**Older Format Support**

While not primary focus:

- IRAF .imh format (limited)
- Simple binary arrays with headers
- Recommend converting to standard FITS

**Converting Other Formats**

Use external tools:

.. code-block:: bash

   # IRAF to FITS
   imcopy input.imh output.fits
   
   # HDF5 to FITS
   python convert_hdf5_to_fits.py input.h5 output.fits

File Format Recommendations
===========================

**Best Practices**

- ✓ Use standard FITS for preservation
- ✓ Save with full WCS information
- ✓ Include all relevant keywords
- ✓ Compress large files
- ✓ Document format in README

**For Different Purposes**

============= ========= =====================
Purpose       Format    Compression
============= ========= =====================
Archival      FITS      gzip
Publication   FITS      None
Web/Sharing   FITS.gz   gzip
Analysis      FITS      None
Table Data    FITS/CSV  gzip
Spectra       ASCII     None
============= ========= =====================

Header Keywords
===============

Essential FITS keywords for IFU data:

**Observational**

- INSTRUME: Instrument name
- TELESCOP: Telescope name
- DATE-OBS: Observation date
- EXPTIME: Exposure time (seconds)

**Wavelength Calibration**

- CRPIX3, CRVAL3, CDELT3: Wavelength axis
- CTYPE3: Axis type (WAVE, FREQ, etc.)
- CUNIT3: Wavelength unit

**Spatial Coordinates**

- CRPIX1/2: Reference pixels (RA/Dec)
- CRVAL1/2: Reference values (degrees)
- CDELT1/2: Pixel scales (degrees)
- CTYPE1/2: Axis types (RA---TAN, DEC--TAN, etc.)

**Units**

- BUNIT: Brightness unit (Jy/pixel, erg/s/cm²/Å, K, etc.)

**Science Image Data**

- SIMPLE: FITS standard compliance (T)
- BITPIX: Bits per pixel (-32 for float32)
- NAXIS: Number of axes (usually 3)
- NAXIS1/2/3: Size of each axis

Reference Pixels
================

Convention for FITS coordinate pixels:

- CRPIX values: 1-indexed (first pixel = 1)
- Pixel coordinates: 0-indexed internally
- Conversion: pixel_index = pixel_value - 0.5 in theory

Best to keep CRPIX at 1 and let WCS handle conversion.

Common Issues
=============

**Issue: Wrong wavelength calibration**

- Verify CRVAL3, CDELT3, CRPIX3
- Check CTYPE3 and CUNIT3
- Compare with known spectral lines

**Issue: WCS not recognized**

- Ensure CTYPE keywords present
- Check RA/Dec format correct
- May need supplementary WCS keywords

**Issue: File won't open**

- Verify FITS standard compliance (SIMPLE, BITPIX)
- Check with external reader (DS9, FITS utilities)
- Try converting with convertformat tools

**Issue: File too large**

- Compress with gzip
- Consider downsampling cubes
- Split into wavelength ranges

Next Steps
==========

For more on files:

- :doc:`/user_guide/loading_data` for file loading
- :doc:`/user_guide/exporting_data` for file saving
