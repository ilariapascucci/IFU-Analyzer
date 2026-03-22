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
- Interactive region handles for dragging and resizing
- Adjustable color scales (Linear, Log, Power, Square Root)
- Customizable colormaps with interactive dialog (including colorblind-friendly options)
- Support for 2D images (intensity maps, Gaussian-derived maps)
- Copy colormaps and regions between dual displays

### CO Rovibrational Flux Mapping
- **Full-Cube CO Flux Extraction:**
  - Simultaneous multi-line fitting of CO rovibrational transitions
  - Contaminating line exclusion and masking
  - Parallel processing across all spatial pixels
  - Optional continuum subtraction before line detection
  - Automatic flux integration and SNR thresholding
  - Background processing with progress tracking
  - Complete flux maps and statistics extraction

### 2D FITS Image Viewer
- Standalone dual-image viewer for intensity maps and derived products
- Image rotation (arbitrary angles) with smooth interpolation
- Contour overlays with customizable levels and visualization
- Contour mask export and import functionality
- WCS-aware coordinate display
- Automatic 2D FITS file loading and display

### Spectral Analysis
- **Gaussian Line Fitting:**
  - Single-pixel interactive fitting with visual overlay
  - Full-cube parallel fitting with parameter maps
  - Multi-Gaussian component fitting for complex line profiles
  - Automated baseline subtraction
  - Quality metrics (R², χ², FWHM, area under curve)
  - Background processing (non-blocking GUI)
  - Pixel-level diagnostics with fit visualization
  
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

- **Advanced Continuum Fitting:**
  - Custom interactive continuum subtraction with point snapping
  - Automatic continuum fitting with tunable parameters
  - RANSAC-based robust polynomial fitting
  - Median filtering and sigma clipping
  - Spline smoothing for smooth baselines

### Data Processing
- Coordinate conversion (pixel ↔ RA/DEC)
- Region-based spectral extraction
- Statistical analysis of selected regions
- Spectrum normalization with baseline points
- Continuum baseline subtraction (interactive and automatic)
- Spectral line identification (HITRAN database and custom catalogs)
- Multi-line automatic detection with SNR thresholding
- Doppler shift correction (multiplicative wavelength shift)
- Wavelength-to-frequency conversion with CGS unit support
- Full astropy Quantity object support for unit handling
- Memory-efficient handling of large data cubes
- Dynamic wavelength/velocity range windowing

### Export Capabilities
- **ASCII Spectrum Export:** Save displayed spectrum (normalized or original) as tab-delimited text
- **Residuals FITS Export:** Export 3D normalized residuals cube with full header metadata
- **CO Flux Maps:** Export full-cube fitting results with integrated flux estimates
- **Contour Masks:** Export 2D region masks from image viewer
- Automatic instrument detection and header preservation
- Unit-aware exports with CGS calibration support

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
git clone https://github.com/ilariapascucci/IFU-Analyzer.git
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
   - Click "Pixel Diagnostics" to view fit quality and residual details
   
   **Full-Cube Mapping:**
   - Mark boundaries as above
   - Click "Fit Gaussian Map" to fit all pixels
   - Wait for background processing (~10-30 seconds typical)
   - Explore parameter maps (Area, Amplitude, Center, FWHM, etc.)
   - Use Pixel Diagnostics to identify problematic fit regions
   - Export residuals cube as FITS if needed

### Advanced Continuum Fitting

5. Use custom continuum subtraction:
   
   **Interactive Method:**
   - Click "Custom Continuum" to activate point selection mode
   - Click on the spectrum to mark baseline points
   - Points automatically snap to spectrum data
   - Cubic spline interpolates smooth baseline
   - Subtract from spectrum with one click
   
   **Automatic Method:**
   - Click "Automatic Continuum" to open tuning dialog
   - Adjust parameters: window size, sigma threshold, peak width
   - Algorithm: median filter → peak detection → sigma clipping → spline fit
   - Apply to current spectrum or full cube

### CO Rovibrational Flux Mapping

6. Extract CO flux from full data cube:
   - Click "CO Flux Map" in the Gaussian Fitting menu
   - Configure CO line list and wavelength ranges
   - Select contaminating lines to exclude
   - Optional: enable continuum subtraction before detection
   - Click "Fit CO Lines" to process (background computation)
   - View flux maps, SNR statistics, and detection quality
   - Export results with integrated flux estimates

### 2D Image Viewer

7. Analyze 2D images and intensity maps:
   - Open a 2D FITS image or generate from cube analysis
   - Right-click to open with "2D Viewer" or use menu option
   - **Rotate image:** Use slider or input arbitrary rotation angles
   - **Add contours:** Click "Add Contours" to overlay contour levels
   - **Export contour mask:** Select regions and export as FITS
   - **View coordinates:** WCS-aware tools show RA/DEC positions

### Data Export

8. Export results:
   - **Spectrum:** Click "Save Spectrum" below spectrum plot
   - **Residuals:** Select "Residuals" in Gaussian map results → "Export Residuals Cube"
   - **CO Flux Maps:** Export flux estimates with units conversion (CGS)
   - **Contour Masks:** Export selected regions from 2D viewer
   - All exports preserve metadata and instrument information

### Additional Features

9. Advanced tools:
   - **Header Viewer:** Inspect FITS header information for selected HDU
   - **Line Catalog:** Load custom spectral line catalogs (LST format)
   - **HITRAN Database:** Query molecular lines with species filtering dialog
   - **Normalize Spectrum:** Use baseline points for normalized display
   - **Doppler Shift:** Apply multiplicative wavelength correction (km/s)
   - **Wavelength Range:** Dynamic min/max windowing with linked view updates
   - **Launch New Instance:** Open independent application windows for multi-file comparison

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
│   ├── ifu_analyzer_2d.py     # 2D FITS image viewer with rotation & contours
│   ├── data_loader.py         # FITS loading & instrument detection
│   ├── gaussian_fitter.py     # Gaussian fitting algorithms
│   ├── co_flux_extractor.py   # CO rovibrational flux mapping
│   ├── visualization.py       # Plotting and rendering functions
│   ├── region_selector.py     # Interactive region selection tools
│   ├── pixel_diagnostics.py   # Pixel-level fit inspection dialog
│   ├── colormap_dialog.py     # Colormap customization
│   ├── continuum_dialog.py    # Automatic continuum fitting with tuning
│   ├── custom_continuum.py    # Interactive custom continuum subtraction
│   ├── co_flux_dialogs.py     # CO flux mapping configuration
│   ├── gaussian_dialogs.py    # Gaussian fitting dialogs
│   ├── species_dialog.py      # Spectral line species selection
│   ├── line_catalog.py        # Line identification and HITRAN integration
│   ├── unit_converter.py      # Wavelength/frequency and unit conversion
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

## Known Limitations

- Residuals are normalized by noise std (may need raw residuals for some analyses)
- 2D FITS images display as single-slice cubes in main viewer (use dedicated 2D viewer for full functionality)
- CO flux mapping requires well-defined line catalog with accurate wavelengths
- ProcessPoolExecutor may have startup overhead on very small cubes

## Future Enhancements

- Full-cube continuum subtraction and export
- Additional fitting models (Lorentzian, Voigt profiles)
- Interactive residuals visualization in spectral domain
- Batch processing mode for automated analysis of multiple cubes
- Custom parameter map exports with calibration metadata
- Machine learning-based line detection and classification