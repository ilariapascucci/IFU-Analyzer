============
Loading Data
============

IFU Analyzer works with FITS (Flexible Image Data System) files containing spectral data cubes or 2D images. This section explains how to load and prepare your data for analysis.

Supported Instruments
=====================

IFU Analyzer automatically detects and handles data from:

- **MUSE** (Multi Unit Spectroscopic Explorer): VLT optical spectroscopy
- **MIRI** (Mid-Infrared Instrument): JWST mid-infrared
- **NIRSpec** (Near-Infrared Spectrograph): JWST near-infrared

Supported File Formats
======================

The application accepts:

- **3D FITS Cubes**: Standard (RA, Dec, λ) format
- **2D FITS Images**: Automatically treated as single-wavelength slices
- **Multiple HDUs**: Extensions within FITS files

Opening a FITS File
===================

**Via the GUI:**

1. Click **Open FITS**
2. Navigate to your file location
3. Select your FITS file
4. The file loads and displays:
   
   - Intensity map at the first wavelength
   - 1D spectrum
   - Header information

File Information & Headers
==========================

After loading, inspect your file information:

- **Header** to see:
  
  - Data dimensions (RA pixels, Dec pixels, wavelength points)
  - Instrument name
  - Observation date
  - Other FITS header information

Multiple File Management
========================

**Loading Multiple Files:**

1. Click **Open FITS** button (first file)
2. Click **🗗** (second file)

Next Steps
==========

After loading your data:

1. Explore :doc:`visualization` options
2. Learn :doc:`selecting_regions` techniques
3. Perform :doc:`gaussian_fitting`


Using the Data Loading API
===========================

Load and access FITS data programmatically in Python scripts or Jupyter notebooks.

**Basic Data Loading**

.. code-block:: python

    from cube_explorer.api import load_cube
    
    # Load a FITS file (instrument auto-detected)
    cube = load_cube("Level3_ch1-long_s3d.fits")
    
    print(f"Instrument: {cube.instrument}")
    print(f"Wavelength range: {cube.wavelength.min():.4f} - {cube.wavelength.max():.4f}")
    print(f"Cube shape (wavelength, x, y): {cube.cube.shape}")
    print(f"Spectrum shape: {cube.spectrum.shape}")

**Access Cube Components**

.. code-block:: python

    cube = load_cube("data.fits")
    
    # Wavelength array
    wavelength = cube.wavelength        # Array of wavelength values
    print(f"Wavelengths: {len(wavelength)} points")
    print(f"Range: {wavelength.min():.4f} - {wavelength.max():.4f}")
    
    # 3D spectral cube
    full_cube = cube.cube              # Shape: (wavelength, x_pixels, y_pixels)
    print(f"Cube dimensions: {full_cube.shape}")
    
    # 1D spectrum (summed over all spatial pixels)
    spectrum_1d = cube.spectrum         # Array of flux values
    
    # Metadata
    header_info = cube.header           # FITS header dictionary
    wcs_info = cube.wcs                 # World Coordinate System
    instrument = cube.instrument        # Instrument name (MUSE, MIRI, etc.)

**Extract Spectra from Specific Regions**

.. code-block:: python

    cube = load_cube("data.fits")
    
    # Extract spectrum from a single pixel
    x, y = 20, 25
    pixel_spectrum = cube.cube[:, x, y]

    # Extract spectrum from a spatial region (average)
    x_start, x_end = 10, 30
    y_start, y_end = 15, 35
    region = cube.cube[:, x_start:x_end, y_start:y_end]
    averaged_spectrum = region.mean(axis=(1, 2))
    
    # Plot the spectrum
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(cube.wavelength, averaged_spectrum, 'b-', linewidth=1)
    plt.xlabel('Wavelength (µm)')
    plt.ylabel('Flux')
    plt.title(f'Spectrum: pixels [{x_start}:{x_end}, {y_start}:{y_end}]')
    plt.grid(True, alpha=0.3)
    plt.show()

**Available Cube Properties**

.. code-block:: python

    cube = load_cube("data.fits")

    # Data access
    print(f"Wavelength points: {len(cube.wavelength)}")
    print(f"Spectrum points:   {len(cube.spectrum)}")
    print(f"Cube shape:        {cube.cube.shape}")

    # Metadata
    print(f"Instrument:        {cube.instrument}")
    print(f"Header cards:      {len(cube.header)}")
    print(f"WCS:               {type(cube.wcs).__name__}")

    # File information
    print(f"Filename:          {cube.filename}")

    # Remaining metadata comes from the FITS header. Keyword names are
    # instrument-specific, so use .get() rather than indexing.
    print(f"Target:            {cube.header.get('TARGNAME')}")
    print(f"Observation date:  {cube.header.get('DATE-OBS')}")
