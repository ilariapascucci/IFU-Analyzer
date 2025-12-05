# IFU Analyzer

An interactive Python tool for visualizing and analyzing Integral Field Unit (IFU) spectroscopic data from various astronomical instruments. This application provides a user-friendly interface for exploring 3D spectral data cubes, analyzing spectra, performing Gaussian line fitting, and exporting results.

## Features

### Instrument Support
- **MUSE** (VLT) - Optical spectroscopy
- **ALMA** - Radio interferometry
- **MIRI** (JWST) - Mid-infrared spectroscopy
- **NIRSpec** (JWST) - Near-infrared spectroscopy

### Interactive Visualization
- Full spectrum display with wavelength/velocity axes
- Dual slice view for wavelength/velocity comparison
- Real-time spectrum updates on selection
- Multiple region selection tools (point, rectangle, circle, ellipse, custom polygon)
- Adjustable color scales (Linear, Log, Power, Square Root)
- Customizable colormaps with interactive dialog
- Support for 2D images (intensity maps, Gaussian-derived maps)

### Spectral Analysis
- **Gaussian Line Fitting:**
  - Single-pixel interactive fitting with visual overlay
  - Full-cube parallel fitting with parameter maps
  - Automated baseline subtraction
  - Quality metrics (R², χ², FWHM, area under curve)
  - Background processing (non-blocking GUI)
  
- **Parameter Maps:**
  - Integrated area (intensity map)
  - Peak amplitude
  - Line center (kinematics)
  - Line width (σ, FWHM)
  - Fit quality (χ²)
  - Success mask
  
- **Residuals Analysis:**
  - Normalized residuals cube export
  - Wavelength-dependent residuals visualization
  - Quality assessment of fits

### Data Processing
- Coordinate conversion (pixel ↔ RA/DEC)
- Region-based spectral extraction
- Statistical analysis of selected regions
- Spectrum normalization
- Continuum baseline subtraction
- Spectral line identification (HITRAN database)
- Memory-efficient handling of large data cubes

### Export Capabilities
- **ASCII Spectrum Export:** Save displayed spectrum (normalized or original) as tab-delimited text
- **Residuals FITS Export:** Export 3D normalized residuals cube with full header metadata
- Automatic instrument detection and header preservation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Dependencies

```bash
# Install required Python packages
pip install numpy matplotlib astropy PyQt5 scipy
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/astro-sophie/IFU-Analyzer.git
cd IFU-Analyzer
```

2. (Optional) Create and activate a virtual environment:
```bash
# Using venv
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

### Data Files

The program works with FITS format data cubes from various instruments. You can open FITS files from any location on your computer using the GUI's "Open FITS" button. Supported data types include:
- MUSE: 3D spectral cubes (.fits)
- ALMA: Radio interferometry cubes (.fits)
- MIRI: Mid-infrared spectral cubes (.fits)
- NIRSpec: Near-infrared spectral cubes (.fits)
- 2D FITS images (automatically converted to single-slice cubes)

Note: FITS files are not included in the repository. You'll need to provide your own data files from your observations or archives.

## Usage

### Basic Operation

1. Start the application:
```bash
python main.py
```

2. Load and visualize data:
   - Click "Open FITS" to select a data cube
   - Click "Load Cube" to load the selected file
   - Use the spectrum plot (bottom) to navigate wavelength/velocity
   - Select slices using "Set Left Slice" / "Set Right Slice"
   - Adjust color scale and colormap as needed

### Region Analysis

3. Select and analyze regions:
   - Click shape buttons (⭐, ▭, ◯, ⬭, ✏) to activate region tools
   - Draw regions on either slice view
   - Statistics update automatically
   - Extracted spectra appear in bottom panel
   - Use "Copy Left→Right" to transfer regions

### Gaussian Fitting

4. Perform Gaussian line fitting:
   
   **Single-Pixel Fitting:**
   - Click "Left" to mark left boundary on spectrum
   - Click "Right" to mark right boundary on spectrum
   - Click "Fit Gaussian" to perform fit
   - View fit results, parameters, and residuals
   
   **Full-Cube Mapping:**
   - Mark boundaries as above
   - Click "Fit Gaussian Map" to fit all pixels
   - Wait for background processing (~10-30 seconds typical)
   - Explore parameter maps (Area, Amplitude, Center, FWHM, etc.)
   - Export residuals cube as FITS if needed

### Data Export

5. Export results:
   - **Spectrum:** Click "Save Spectrum" below spectrum plot
   - **Residuals:** Select "Residuals" in Gaussian map results → "Export Residuals Cube"
   - All exports preserve metadata and instrument information

### Additional Features

6. Advanced tools:
   - Normalize spectrum using baseline points
   - Load spectral line catalogs (LST files)
   - Query HITRAN database for molecular lines
   - Adjust wavelength range dynamically
   - View FITS header information

## Performance

- **Optimized Gaussian Fitting:** Adaptive parallelization with 8 workers (threads/processes)
- **Reduced convergence iterations:** maxfev=5000 for ~2-5x speedup
- **Background processing:** GUI remains responsive during computations
- **Memory efficiency:** Memory-mapped access for large cubes
- **Typical performance:** ~10-30 seconds for full-cube Gaussian mapping

## File Structure

```
IFU-Analyzer/
├── main.py                    # Application entry point
├── analyzer.py                # Core analysis wrapper
├── cube_explorer/
│   ├── __init__.py
│   ├── gui.py                 # Main GUI (2500+ lines)
│   ├── data_loader.py         # FITS loading & instrument detection
│   ├── gaussian_fitter.py     # Gaussian fitting algorithms
│   ├── visualization.py       # Plotting functions
│   ├── region_selector.py     # Region selection tools
│   ├── colormap_dialog.py     # Colormap customization
│   ├── species_dialog.py      # Spectral line selection
│   ├── line_catalog.py        # Line identification
│   ├── unit_converter.py      # Wavelength/frequency conversion
│   └── utils.py               # Utility functions
└── data/                      # (Optional) Sample FITS files
```

## Troubleshooting

Common issues and solutions:

1. **Memory Errors:**
   - Try closing other applications
   - Use smaller data cubes
   - Application automatically uses memory mapping for large files

2. **Display Issues:**
   - Ensure PyQt5 is properly installed
   - Update graphics drivers if experiencing rendering problems
   - Try different color scales (Log, Power) for better contrast

3. **File Loading Errors:**
   - Verify file format matches supported instruments
   - Check FITS header for required keywords (INSTRUME, TELESCOP)
   - Ensure file is not corrupted
   - 2D FITS images are supported (auto-converted to 3D)

4. **Gaussian Fitting Issues:**
   - Ensure adequate wavelength range selected (>4 points)
   - Check for positive peaks after baseline subtraction
   - "x0 is infeasible" errors are automatically handled with bounds clamping
   - Low success rate may indicate noise-dominated regions

5. **Performance Issues:**
   - Gaussian mapping optimized to ~10-30 seconds for typical cubes
   - Large cubes (>100k pixels) may take 1-2 minutes
   - Background processing prevents GUI freezing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to your branch (`git push origin feature/NewFeature`)
5. Create a Pull Request

## Version History

- **Week 1:** Initial GUI implementation, basic FITS loading
- **Week 2:** Added dual slice view, coordinate display
- **Week 3:** Implemented region selection, improved layout
- **Week 4:** Enhanced performance, live updates, layout improvements
- **Week 5:** LST files integration for spectral lines
- **Week 6:** HITRAN database integration
- **Week 7:** Added scaling options (Log, Power), colormap customization
- **Week 8:** Gaussian fitting implementation (single-pixel and full-cube)
- **Week 9:** 
  - Gaussian parameter maps (Area, Amplitude, Center, FWHM, χ², Success)
  - Normalized residuals cube export with metadata preservation
  - Background processing for Gaussian mapping (QThread)
  - Performance optimization (~5x speedup: maxfev reduction, adaptive parallelization)
  - ASCII spectrum export functionality
  - 2D FITS image support (intensity maps)
  - Instrument auto-detection without hardcoding
  - Bounds clamping to prevent fitting errors

## Known Limitations

- Gaussian fitting assumes single-component lines (no multi-Gaussian support yet)
- Residuals are normalized by noise std (may need raw residuals for some analyses)
- 2D FITS images display as single-slice cubes (spectrum panel not meaningful)
- ProcessPoolExecutor may have startup overhead on very small cubes

## Future Enhancements

- Multi-Gaussian fitting for complex line profiles
- Full-cube continuum subtraction and export
- Additional fitting models (Lorentzian, Voigt)
- Interactive residuals visualization
- Batch processing mode
- Custom parameter map exports