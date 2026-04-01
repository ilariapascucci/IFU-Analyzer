============
Introduction
============

What is IFU Analyzer?
=====================

**IFU Analyzer** is a comprehensive, interactive Python application designed for visualizing and analyzing three-dimensional spectroscopic data cubes from Integral Field Unit (IFU) instruments. It provides an intuitive graphical interface combined with powerful analysis tools, making it accessible to both casual users and advanced researchers.

The application supports data from multiple state-of-the-art instruments:

- **MUSE** (Multi Unit Spectroscopic Explorer) - VLT optical spectroscopy
- **MIRI** (Mid-Infrared Instrument) - JWST mid-infrared spectroscopy
- **NIRSpec** (Near-Infrared Spectrograph) - JWST near-infrared spectroscopy

Key Capabilities
================

Interactive Visualization
-------------------------

- **3D Spectral Data Cube Exploration**: Load and navigate through 3D FITS data cubes with seamless wavelength slicing
- **Multiple Wavelength Views**: Explore data across different wavelength ranges interactively
- **Real-time Updates**: Spectrum updates instantly as you select regions or pixels
- **Advanced Color Controls**: Apply linear, logarithmic, power-law, and square-root scaling with customizable colormaps
- **2D Imaging**: Support for 2D images

Spectral Analysis
-----------------

- **1D Gaussian Line Fitting**:

  - Interactive single-pixel spectrum fitting with visual overlay and quality metrics
  - Pixel-by-pixel full-cube fitting with comprehensive parameter extraction
  - Baseline subtraction and continuum removal
  - Quality assessment through χ², R², FWHM, and area measurements

- **Integrated Intensity Maps & Pixel-by-Pixel Gaussian Fitting**:

  - Integrated area (intensity maps) from line fitting
  - Peak amplitude distributions across the field
  - Line center measurements at each spatial position
  - Line width analysis (σ, FWHM) variations
  - Fit quality metrics and diagnostics
  - Success/failure masks showing fit reliability

- **Spectral Processing**:

  - Region-based spectral extraction
  - Automated continuum baseline subtraction
  - Spectral line identification using HITRAN database

Data Management
---------------

- **Multi-Format Support**: Works with FITS files from various instruments and custom formats
- **Memory-Efficient Processing**: Handle large data cubes efficiently
- **Multiple Selection Tools**: Point, rectangular, circular, elliptical, and free-form selection
- **Coordinate Systems**: Seamless conversion between pixel and RA/DEC coordinates

Export & Output
---------------

- **ASCII Spectrum Export**: Save displayed spectra as tab-delimited text files
- **Automatic Metadata**: Instrument detection and header preservation during export
- **Custom Formatting**: Configure output format to match your analysis pipeline

Project Status
==============

IFU Analyzer is actively developed. The current version (1.0) includes all core functionality for:

- Interactive data visualization
- Gaussian line fitting
- Continuum subtraction
- CO flux extraction
- Spectral analysis tools

Future versions will expand functionality based on user feedback and research requirements.
