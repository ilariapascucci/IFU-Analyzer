============
Introduction
============

What is IFU Analyzer?
=====================

**IFU Analyzer** is a comprehensive, interactive Python application designed for visualizing and analyzing three-dimensional spectroscopic data cubes from Integral Field Unit (IFU) instruments. It provides an intuitive graphical interface combined with powerful analysis tools.

The application supports data from multiple instruments:

- **MUSE** (Multi Unit Spectroscopic Explorer) - VLT optical spectroscopy
- **MIRI** (Mid-Infrared Instrument) - JWST mid-infrared spectroscopy
- **NIRSpec** (Near-Infrared Spectrograph) - JWST near-infrared spectroscopy

Key Capabilities
================

Interactive Visualization
-------------------------

- **3D Spectral Data Cube Exploration**: Load and navigate through 3D FITS data cubes with wavelength slicing
- **Multiple Wavelength Views**: Explore data across different wavelength ranges interactively
- **Real-time Updates**: Spectra update instantly as you select regions or pixels
- **Advanced Color Controls**: Apply linear, logarithmic, and power-law scaling (with an adjustable gamma), an absolute or percentile color range, and customizable colormaps
- **2D Imaging**: Support for 2D images
- **Multiple Selection Tools**: Point, rectangular, circular, elliptical, and free-form selection

Spectral Analysis
-----------------

- **1D Gaussian Line Fitting**:

  - Interactive single-pixel spectrum fitting with visual overlay and quality metrics
  - Pixel-by-pixel full-cube fitting with comprehensive parameter extraction
  - Baseline subtraction and continuum removal
  - Quality assessment through R², FWHM, and area measurements

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

- **Rovibrational Flux Mapping** (any target species):

  - Simultaneous multi-line Gaussian fitting per pixel, with shared local continuum
  - Contaminating-line exclusion and SNR-based detection thresholding
  - Full-cube flux and detection maps in CGS units

- **H2 Rotational Diagram Fitting**:

  - 1- and 2-temperature excitation fits from HITRAN or user-supplied line fluxes
  - Ortho-to-para ratio (OPR) fitting alongside temperature
  - Bayesian censored regression for datasets containing upper limits

Data Management
---------------

- **Instrument-Aware FITS Support**: Automatic instrument detection and header handling for MUSE, MIRI, and NIRSpec cubes
- **Memory-Efficient Processing**: Handle large data cubes efficiently
- **Coordinate and Unit Conversion**: Conversion between pixel and RA/DEC coordinates, and from native cube flux units to CGS

Export & Output
---------------

- **ASCII Spectra Export**: Save displayed spectra as tab-delimited text files
- **Automatic Metadata**: Instrument detection and header preservation during export
- **Multiple Output Formats**: Spectra as ASCII text, fit maps as FITS files, and visualizations as PNG, JPEG, or PDF

Project Status
==============

IFU Analyzer is actively developed. The current version (1.0) includes all core functionality for:

- Interactive data visualization
- Gaussian line fitting
- Continuum subtraction
- Flux extraction (any target species)
- Rotational diagram temperature fitting
- 2D FITS dual image viewer
- Spectral analysis tools

Future versions will expand functionality based on user feedback and research requirements.
