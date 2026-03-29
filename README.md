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

### 2D FITS Dual Image Viewer

A standalone side-by-side viewer for 2D intensity maps, moment maps, and derived FITS products. Fully WCS-aware with interactive coordinate display and multi-image analysis tools.

#### Image Display

- Dual-panel layout — load independent 2D FITS files in Left and Right panels
- Full WCS grid overlay with RA/Dec tick labels at correct positions for any rotation angle
- Tick label format auto-scales to field of view (degrees → arcmin → arcsec → sub-arcsec)
- Live cursor readout: pixel coordinates, RA (hh:mm:ss), Dec (dd:mm:ss), flux value
- Image statistics panel: min, max, mean with flux unit from FITS header
- Adjustable color scales: Linear, Log, Sqrt, Power (γ=2)
- Customizable colormaps via interactive dialog

#### Image Rotation

- Rotate images by arbitrary angles (clockwise or counter-clockwise)
- WCS header is updated correctly on every rotation — RA/Dec labels remain accurate
- Cumulative rotation tracking (rotations compose correctly)
- Save rotated image as a new FITS file with updated WCS header

#### Contour Plotting

- **Background estimation:** Astropy sigma-clipped stats (automatic) or user-supplied value
- **Sigma levels:** Comma-separated array of multipliers (e.g. `3, 5, 10`) — each value N draws a contour line at N × background_std above the background median
- **WCS-matched cross-image contours:** Generate contours from one panel and overlay them on the other panel using full WCS coordinate transformation — works correctly even when the two images have different pixel scales, array sizes, or orientations
- Configurable contour color
- Multiple contour sets can be stacked
- Clear all contours with one click

#### Contour Export / Import

- **Save as .txt:** Exports all contour path vertices as pixel coordinates + RA/Dec (degrees) with sigma level labels — human-readable, compatible with any analysis tool
- **Load saved contour:** Load a previously saved binary FITS mask and overlay it on either panel via WCS mapping
- **Save contour mask as FITS:** Export binary 0/1 mask with WCS header for use in other tools

#### WCS Superimpose

- Overlay two images with different WCS grids (different pixel scale, array size, orientation, pointing) onto a single panel
- Reprojects the overlay image onto the background image's pixel grid using WCS transforms
- Reprojection is fully self-implemented using `numpy`/`scipy` and `astropy.wcs` — no external packages required
- Configurable overlay colormap and alpha transparency (0.1–1.0)
- Built-in peak separation diagnostic prints angular offset between the two image peaks to terminal — confirms alignment accuracy
- Uses bicubic interpolation (`order=3`) with NaN-aware masking for accurate spatial alignment

#### Region Selection

- Point, rectangle, circle, ellipse, and custom polygon tools
- Selected region statistics (min, max, mean, sum, std) appended to image stats panel

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
  - **Automatic mode:** Median filter → peak detection → iterative sigma clipping → cubic spline fit
    - Tunable parameters: window size, sigma threshold, peak exclusion width, iterations, smoothing factor
    - Fitted continuum displayed in orange over original spectrum before any subtraction
    - One-click global subtraction
  - **Custom interactive mode:**
    - Click to place continuum anchor points directly on the spectrum
    - Points snap to the nearest data value
    - Cubic spline interpolates a smooth baseline through all points
    - Delete individual points with Delete Point mode
    - **Preview before subtract:** Clicking OK shows the fitted continuum in orange over the original spectrum — a confirmation dialog asks before applying the subtraction
    - Full spectrum displayed after subtraction with correct y-axis scaling (no clipping)
    - Save fitted continuum as ASCII or save continuum-subtracted spectrum

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
- **Contour Coordinates (.txt):** Export contour path vertices with pixel + RA/Dec positions and sigma level labels
- **Contour Mask (FITS):** Export binary region mask with WCS header
- **Rotated Image (FITS):** Save rotated image with updated WCS header
- Automatic instrument detection and header preservation
- Unit-aware exports with CGS calibration support

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Dependencies

```bash
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
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

### Data Files

The program works with FITS format data cubes and 2D images. Supported data types:

- MUSE: 3D spectral cubes (.fits)
- ALMA: Radio interferometry cubes (.fits)
- MIRI: Mid-infrared spectral cubes and 2D slice maps (.fits)
- NIRSpec: Near-infrared spectral cubes (.fits)
- 2D FITS images: intensity maps, moment maps, Gaussian parameter maps

Note: FITS files are not included in the repository. Provide your own data from observations or archives.

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

5. Use continuum subtraction:

   **Automatic Method:**
   - Toggle "Automatic Continuum" button
   - Click "Create Continuum Line" to open parameter dialog
   - Adjust: window size, sigma threshold, peak exclusion width, iterations, smoothing factor
   - Fitted continuum appears in orange over original spectrum
   - Click "Subtract Continuum" to apply — full-range subtracted spectrum is displayed

   **Custom Interactive Method:**
   - Toggle "Custom Continuum" button
   - Click "Custom Subtract" to enter point-placement mode
   - Click directly on the spectrum to place anchor points (snap to data)
   - Use "Delete Point" mode to remove misplaced points (minimum 3 points required)
   - Click "OK" — the fitted cubic spline continuum is shown in orange for preview
   - Confirm subtraction in the dialog — "No" returns to editing mode
   - Full-range subtracted spectrum is displayed after confirmation
   - Save the result with "Save Spectrum" or "Save Continuum"

### 2D Image Viewer

6. Analyze 2D FITS images:
   - Launch the 2D Viewer from the main window or run standalone
   - Load images into Left and/or Right panels using "Open 2D FITS"
   - WCS grid (RA/Dec) is displayed automatically if FITS header contains WCS

   **Rotation:**
   - Select direction (Clockwise / Counter-Clockwise) and angle
   - Click "Rotate" and choose which panel to rotate
   - WCS labels update correctly for any angle

   **Contours:**
   - Click "Plot Contour" and configure:
     - Which panel to draw on (Left/Right)
     - Source image (same panel or opposite panel for WCS-matched cross-image contours)
     - Background std (leave blank for Astropy auto, or enter a value)
     - Sigma levels as comma-separated list: e.g. `3, 5, 10`
     - Contour color
   - Contour lines are drawn at each N × sigma above background median
   - "Clear Contours" removes all contour overlays from a panel

   **Save Contour:**
   - Click "Save Contour Mask" → choose panel → saves .txt file with columns:
     `level, x_pixel, y_pixel, RA_deg, Dec_deg`

   **Load Saved Contour:**
   - Click "Load Saved Contour" → select a binary FITS mask file
   - Mask is overlaid on the chosen panel via WCS coordinate mapping

   **Superimpose:**
   - Load different images in Left and Right panels
   - Click "Superimpose WCS"
   - Choose which panel is the background (reference grid)
   - Select overlay colormap and alpha (0.5 recommended as starting point)
   - The overlay image is reprojected onto the background WCS grid using built-in WCS coordinate transforms and blended
   - Terminal prints peak separation in arcsec to confirm alignment accuracy

### CO Rovibrational Flux Mapping

7. Extract CO flux from full data cube:
   - Click "CO Flux Map" in the Gaussian Fitting menu
   - Configure CO line list and wavelength ranges
   - Select contaminating lines to exclude
   - Optional: enable continuum subtraction before detection
   - Click "Fit CO Lines" to process (background computation)
   - View flux maps, SNR statistics, and detection quality
   - Export results with integrated flux estimates

### Data Export

8. Export results:
   - **Spectrum:** Click "Save Spectrum" below spectrum plot
   - **Continuum:** Click "Save Continuum" to export the fitted baseline
   - **Residuals:** Select "Residuals" in Gaussian map results → "Export Residuals Cube"
   - **CO Flux Maps:** Export flux estimates with units conversion (CGS)
   - **Contour .txt:** Pixel + RA/Dec coordinates of all contour vertices
   - **Contour FITS mask:** Binary 0/1 mask with WCS header
   - **Rotated FITS:** Rotated image with corrected WCS header

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
├── main.py                     # Application entry point
├── analyzer.py                 # Core analysis wrapper
├── cube_explorer/
│ ├── __init__.py
│ ├── gui.py                    # Main GUI (2500+ lines)
│ ├── gui_continuum.py          # Continuum subtraction mixin (auto + custom modes)
│ ├── gui_gaussian.py           # Gaussian fitting mixin
│ ├── gui_coflux.py             # CO flux mapping mixin
│ ├── ifu_analyzer_2d.py        # 2D FITS dual viewer (WCS, rotation, contours, superimpose)
│ ├── data_loader.py            # FITS loading & instrument detection
│ ├── gaussian_fitter.py        # Gaussian fitting algorithms
│ ├── co_flux_extractor.py      # CO rovibrational flux mapping
│ ├── visualization.py          # Plotting and rendering functions
│ ├── region_selector.py        # Interactive region selection tools
│ ├── pixel_diagnostics.py      # Pixel-level fit inspection dialog
│ ├── colormap_dialog.py        # Colormap customization
│ ├── continuum_dialog.py       # Automatic continuum fitting parameter dialog
│ ├── custom_continuum.py       # Interactive custom continuum point selector
│ ├── co_flux_dialogs.py        # CO flux mapping configuration dialogs
│ ├── gaussian_dialogs.py       # Gaussian fitting dialogs
│ ├── species_dialog.py         # Spectral line species selection
│ ├── line_catalog.py           # Line identification and HITRAN integration
│ ├── unit_converter.py         # Wavelength/frequency and unit conversion
│ └── utils.py                  # Utility functions
└── data/                       # (Optional) Sample FITS files
```

## Troubleshooting

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
   - 2D FITS images are supported natively in the 2D Viewer

4. **WCS Grid Not Visible:**
   - Confirm the FITS header contains CTYPE1/CTYPE2 with RA/Dec projection
   - Check terminal output for `DEBUG rebuild_axes` messages confirming WCS parse status
   - Non-celestial WCS falls back to plain pixel axes automatically

5. **Superimpose Misalignment:**
   - Check terminal for `DEBUG superimpose: Peak separation` — should be < 0.5 px
   - Large separation indicates a WCS header discrepancy between the two files

6. **Gaussian Fitting Issues:**
   - Ensure adequate wavelength range selected (>4 points)
   - Check for positive peaks after baseline subtraction
   - "x0 is infeasible" errors are automatically handled with bounds clamping
   - Low success rate may indicate noise-dominated regions

7. **Performance Issues:**
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
- WCS superimpose uses bicubic interpolation with NaN masking; accuracy is sub-pixel for typical FITS WCS grids

## Future Enhancements

- Full-cube continuum subtraction and export
- Additional fitting models (Lorentzian, Voigt profiles)
- Interactive residuals visualization in spectral domain
- Batch processing mode for automated analysis of multiple cubes
- Custom parameter map exports with calibration metadata
- Machine learning-based line detection and classification
