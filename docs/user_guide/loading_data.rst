============
Loading Data
============

IFU Analyzer works with FITS (Flexible Image Data System) files containing spectral data cubes or 2D images. This section explains how to load and prepare your data for analysis.

Supported Instruments
=====================

IFU Analyzer automatically detects and handles data from:

- **MUSE** (Multi Unit Spectroscopic Explorer): VLT optical spectroscopy
- **ALMA** (Atacama Large Millimeter Array): Radio interferometry
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
