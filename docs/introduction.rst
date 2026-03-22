============
Introduction
============

What is IFU Analyzer?
====================

**IFU Analyzer** is a comprehensive, interactive Python application designed for visualizing and analyzing three-dimensional spectroscopic data cubes from Integral Field Unit (IFU) instruments. It provides an intuitive graphical interface combined with powerful analysis tools, making it accessible to both casual users and advanced researchers.

The application supports data from multiple state-of-the-art instruments:

- **MUSE** (Multi Unit Spectroscopic Explorer) - VLT optical spectroscopy
- **ALMA** (Atacama Large Millimeter/submillimeter Array) - Radio interferometry
- **MIRI** (Mid-Infrared Instrument) - JWST mid-infrared spectroscopy
- **NIRSpec** (Near-Infrared Spectrograph) - JWST near-infrared spectroscopy

Key Capabilities
================

Interactive Visualization
-------------------------

- **3D Spectral Data Cube Exploration**: Load and navigate through 3D FITS data cubes with seamless wavelength/velocity slicing
- **Dual-View Interface**: Compare wavelength and velocity representations side-by-side
- **Real-time Updates**: Spectrum updates instantly as you select regions or pixels
- **Advanced Color Controls**: Apply linear, logarithmic, power-law, and square-root scaling with customizable colormaps
- **2D Imaging**: Support for 2D images including intensity maps and derived Gaussian parameter maps

Spectral Analysis
-----------------

- **Gaussian Line Fitting**: 
  - Interactive single-pixel fitting with visual overlay and quality metrics
  - Automated full-cube parallel fitting with comprehensive parameter extraction
  - Baseline subtraction and continuum removal
  - Quality assessment through χ², R², FWHM, and area measurements

- **Parameter Maps**:
  - Integrated area (intensity maps)
  - Peak amplitude distributions
  - Line center measurements (kinematics)
  - Line width analysis (σ, FWHM)
  - Fit quality metrics
  - Success/failure masks

- **Spectral Processing**:
  - Region-based spectral extraction
  - Statistical analysis of selected regions
  - Spectrum normalization
  - Automated continuum baseline subtraction
  - Spectral line identification using HITRAN database

Data Management
---------------

- **Multi-Format Support**: Works with FITS files from various instruments and custom formats
- **Memory-Efficient Processing**: Handle large data cubes efficiently
- **Multiple Selection Tools**: Point, rectangular, circular, elliptical, and custom polygon selection
- **Coordinate Systems**: Seamless conversion between pixel and RA/DEC coordinates

Export & Output
---------------

- **ASCII Spectrum Export**: Save displayed spectra (normalized or original) as tab-delimited text files
- **Residuals Analysis**: Export 3D normalized residuals cubes with full FITS header metadata
- **Automatic Metadata**: Instrument detection and header preservation during export
- **Custom Formatting**: Configure output format to match your analysis pipeline

Design Philosophy
==================

IFU Analyzer was designed with the following principles in mind:

1. **User-Friendly**: An intuitive interface that doesn't require deep Python knowledge
2. **Flexible**: Support for multiple instruments and analysis techniques
3. **Powerful**: Advanced analysis tools for research-grade work
4. **Documented**: Comprehensive documentation and examples
5. **Reproducible**: Full tracking of analysis parameters and fit metrics

Target Users
============

IFU Analyzer is suitable for:

- **Astronomers & Astrophysicists**: Analyzing IFU spectroscopy data for scientific research
- **Post-Doctoral Researchers**: Quick exploration and preliminary analysis of large datasets
- **Graduate Students**: Learning spectroscopic analysis techniques
- **Data Processing Pipelines**: Integration into automated analysis workflows
- **Educational Use**: Teaching spectroscopic analysis methods

Quick Example
=============

Here's what a typical workflow looks like:

1. **Load Data**: Open a FITS cube from your telescope (MUSE, ALMA, MIRI, or NIRSpec)
2. **Explore**: Navigate through wavelength slices, zoom, and select regions of interest
3. **Analyze**: Fit Gaussian profiles to spectral features or extract continuum maps
4. **Export**: Save results as FITS files or ASCII spectra
5. **Iterate**: Refine your analysis with different settings and parameters

The entire process is interactive—changes appear in real-time as you adjust parameters.

Project Status
==============

IFU Analyzer is actively developed. The current version (1.0) includes all core functionality for:

- Interactive data visualization
- Gaussian line fitting
- Continuum subtraction
- CO flux extraction
- Spectral analysis tools

Future versions will expand functionality based on user feedback and research requirements.

Citation
========

If you use IFU Analyzer in your research, please cite:

    Pascucci, I. (2024). IFU Analyzer: An interactive tool for analyzing integral field unit spectroscopic data. Zenodo.

Contact & Support
=================

For questions, bug reports, or feature requests, please visit the project repository or contact the development team.
