# IFU Analyzer

IFU Analyzer is a Python application for the analysis of Integral Field Unit (IFU) spectroscopic data. It reads 3D spectral data cubes from MUSE, MIRI, and NIRSpec. The interface shows the data cube, extracts spectra from selected regions, fits Gaussian line profiles, subtracts the continuum, and exports the results.

## Features

### Instrument Support

- **MUSE** (VLT) — optical spectroscopy
- **MIRI** (JWST) — mid-infrared spectroscopy
- **NIRSpec** (JWST) — near-infrared spectroscopy

### Interactive Visualization

- Full-spectrum display with wavelength and velocity axes
- Dual slice view for a wavelength or velocity comparison
- The spectrum updates when you change the selection
- Region tools: point, rectangle, circle, ellipse, and freehand polygon
- Draggable region handles for a move or a resize
- Colour scales: Linear, Log, Power, and Square Root
- Colormap selection through a dialog, with colourblind-friendly options
- Support for 2D images (intensity maps and Gaussian parameter maps)
- Copy of colormaps and regions between the two displays

### Rovibrational Flux Mapping

- **Full-cube flux extraction for any target species:**
  - Simultaneous multi-line fitting of rovibrational transitions
  - Exclusion and masking of contaminating lines
  - Parallel processing across all spatial pixels
  - Automatic continuum subtraction before line detection
  - Automatic flux integration and SNR thresholding
  - Background processing with a progress display
  - Output of complete flux maps and statistics

### 2D FITS Dual Viewer

The 2D viewer is a side-by-side viewer for 2D intensity maps, moment maps, and other 2D FITS products. It is WCS-aware. It shows the cursor coordinates and gives multi-image analysis tools.

#### Image Display

- Dual-panel layout. Load an independent 2D FITS file into the left panel or the right panel.
- Full WCS grid overlay. The RA and Dec tick labels stay in the correct position at any rotation angle.
- The tick-label format adjusts to the field of view (degrees, then arcmin, then arcsec, then sub-arcsec).
- Live cursor readout: pixel coordinates, RA (hh:mm:ss), Dec (dd:mm:ss), and flux value
- Intensity statistics panel: minimum, maximum, and mean, with the flux unit from the FITS header
- Colour scales: Linear, Log, Sqrt, and Power (γ = 2)
- Colormap selection through a dialog

#### Image Rotation

- Rotate an image by any angle. Select "Counter-Clockwise" or "Clockwise", then click "Rotate Left" or "Rotate Right".
- The WCS header is updated on each rotation, so the RA and Dec labels stay correct.
- The rotation is cumulative. Successive rotations compose correctly.
- Save the rotated image as a new FITS file with the updated WCS header.

#### Contour Plotting

- **Background estimation:** Astropy sigma-clipped statistics (automatic), or a value that you supply
- **Sigma levels:** a comma-separated list of multipliers (for example `3, 5, 10`). Each value N draws a contour line at N × background_std above the background median.
- **WCS-matched cross-image contours:** Generate the contours from one panel and draw them on the other panel through a full WCS coordinate transformation. This works when the two images have a different pixel scale, array size, or orientation.
- The contour colour is configurable.
- You can stack more than one contour set.
- "Clear" removes all contours from a panel.

#### Contour Export and Import

- **Save (.txt):** writes all contour path vertices as pixel coordinates and RA/Dec (degrees) with the sigma-level labels. The file is human-readable.
- **Load:** reads a saved binary FITS mask and draws it on either panel through the WCS mapping.
- **Save contour mask as FITS:** writes a binary 0/1 mask with a WCS header for use in other tools.

#### WCS Superimpose

- Overlay two images that have a different WCS grid (a different pixel scale, array size, orientation, or pointing) onto one panel.
- The overlay image is reprojected onto the pixel grid of the background image with WCS transforms.
- The reprojection is self-implemented with `numpy`, `scipy`, and `astropy.wcs`. No external package is necessary.
- The overlay colormap and the alpha transparency (0.1 to 1.0) are configurable.
- A peak-separation diagnostic prints the angular offset between the two image peaks to the terminal. Use it to confirm the alignment.
- The reprojection uses bicubic interpolation (`order=3`) with NaN-aware masking.

#### Resolution Matching — Pair (two images)

Match a pair of images to a common spatial resolution. This is the standard step before a multi-instrument flux comparison.

- Load two FITS images with a different pixel scale into the left panel and the right panel.
- Click **Pair** to open the confirmation dialog.
- The tool finds which image is finer and which is coarser. It then convolves the **finer** image up to the **coarser** pixel scale with a Gaussian kernel.
- Kernel width: from the quadrature subtraction `σ = √(σ_target² − σ_input²) / 2.3548`, in fine-image pixels.
- The convolution uses `astropy.convolution.convolve_fft` with these settings:
  - `nan_treatment='interpolate'` — NaN pixels (for example the IFU footprint border) are handled correctly and do not propagate.
  - `normalize_kernel=True` — the kernel sum is unity, so the surface brightness per pixel is kept.
  - NaN padding of width `⌈4σ⌉` before the FFT, so the kernel tails stay inside the domain.
  - A final flux renormalisation: `Σ(output_finite) == Σ(input_finite)` to machine precision (about 1e-7 relative error).
- After the smoothing, the tool reprojects the convolved image onto a new pixel grid at the coarse pixel scale. It uses a flux-conserving spherical-polygon-intersection algorithm (equivalent to `reproject_exact`, self-implemented, with no external `reproject` dependency).
- The output shape is large enough to contain the full original field of view of the fine image. No sky coverage is lost.
- **Same-resolution guard:** if both panels have the same pixel scale (within 0.0001 arcsec/px), the tool shows a "nothing to do" message and does not run. This prevents an unnecessary convolution.
- The output is a 2-HDU FITS file: the image, and a `COVERAGE` extension (the per-pixel overlap fraction, 0 to 1).
- The save dialog shows a flux audit. `Ratio out/in : 1.000000` confirms exact flux conservation.

#### Resolution Matching — Batch (multiple images)

Bring any number of multi-instrument images to one common resolution and field of view in a single step.

- Click **Batch** to open the batch dialog.
- Click **+ Add FITS Files** to add two or more 2D FITS files (any mix of instruments, pixel scales, and orientations).
- The tool does the following:
  1. It finds the **coarsest pixel scale** among all inputs. This becomes the target scale.
  2. It finds the **largest-area image** (the most sky coverage). This sets the output orientation and centre.
  3. It builds one common output WCS at the coarsest scale, large enough to cover the **union of all input footprints**. No image loses sky coverage.
  4. For each input, it applies the correct Gaussian kernel (σ from quadrature) and reprojects onto the common grid.
  5. An image that is already at the coarsest scale gets σ = 0 (no smoothing, and a reproject only if it is necessary).
- Click **Process & Download ZIP**. All outputs are packed into a `.zip` file: one FITS file per input, all on the same WCS grid, ready for a pixel-by-pixel comparison.
- Each output FITS file includes the `COVERAGE` extension.
- The completion dialog shows a flux-audit table (`ratio = 1.000000` per file).

#### Region Selection

- Point, rectangle, circle, ellipse, and freehand polygon tools
- The statistics of the selected region (minimum, maximum, mean, sum, and standard deviation) are added to the intensity statistics panel.

### Spectral Analysis

- **Gaussian line fitting:**
  - Single-pixel interactive fitting with a visual overlay
  - Full-cube parallel fitting with parameter maps
  - Multi-Gaussian fitting for a complex line profile
  - Automatic baseline subtraction
  - Quality metrics (R², χ², FWHM, and area under the curve)
  - Background processing, so the GUI does not block
  - Pixel-level diagnostics with a fit plot

- **Parameter maps:**
  - Integrated area (intensity map)
  - Peak amplitude
  - Line centre (kinematics)
  - Line width (σ and FWHM)
  - Fit quality (χ²)
  - Success mask

- **Residuals analysis:**
  - Export of a normalised residuals cube
  - Plot of the wavelength-dependent residuals
  - Quality assessment of the fits

- **Continuum fitting:**
  - **Automatic mode:** median filter, then peak detection, then iterative sigma clipping, then a cubic-spline fit
    - Adjustable parameters: window size, sigma threshold, peak-exclusion width, iterations, and smoothing factor
    - The fitted continuum is shown in orange over the original spectrum before any subtraction.
    - One click subtracts the continuum from the whole spectrum.
  - **Custom interactive mode:**
    - Click on the spectrum to place a continuum anchor point.
    - Each point snaps to the nearest data value.
    - A cubic spline interpolates a smooth baseline through all points.
    - Remove a single point with "Delete Point" mode.
    - **Preview before the subtraction:** when you click "OK", the fitted continuum is shown in orange over the original spectrum, and a confirmation dialog asks before it applies the subtraction.
    - The full spectrum is shown after the subtraction, with a correct y-axis scale (no clipping).
    - Save the fitted continuum as ASCII, or save the continuum-subtracted spectrum.

- **Rotational-diagram analysis:**
  - A separate analyzer window for rotational-temperature diagrams (1T and 2T fits)
  - Fetch line data from HITRAN, or import a custom CSV of transitions.
  - Select or clear individual lines. Enter the flux and the flux error for each line.
  - Mark a non-detection as an upper limit. A dialog computes the limit directly from a FITS cube.
  - Automatic method selection: ordinary least squares (`scipy.curve_fit`) when all points are detections; PyMC Bayesian censored regression when one or more upper limits are present.
  - Emitting-radius and distance inputs (au and pc) for the column-density normalisation
  - The results dialog reports the fitted temperature or temperatures and the column densities.

### Data Processing

- Coordinate conversion (pixel to RA and Dec)
- Region-based spectral extraction
- Statistical analysis of a selected region
- Spectrum normalisation with baseline points
- Continuum baseline subtraction (interactive and automatic)
- Spectral-line identification (the HITRAN database and custom catalogs)
- Multi-line automatic detection with SNR thresholding
- Doppler-shift correction (a multiplicative wavelength shift)
- Wavelength-to-frequency conversion with CGS unit support
- Full `astropy` Quantity support for unit handling
- Memory-efficient handling of a large data cube
- Dynamic wavelength and velocity range windowing

### Export Capabilities

- **Continuum results (ASCII):** the "Save Spectrum" button writes a four-column file — wavelength, original spectrum, fitted continuum, and continuum-subtracted spectrum — in tab-delimited format. A missing column (for example, no continuum is fitted yet) is written as `NaN`. The `# Type: full_results` header line identifies the file format.
- **Residuals cube (FITS):** export of the 3D normalised residuals cube with full header metadata
- **Flux maps:** export of the full-cube fitting results with the integrated flux estimates
- **Contour coordinates (.txt):** export of the contour path vertices with the pixel and RA/Dec positions and the sigma-level labels
- **Slice or region (FITS):** a single 2D slice, or the full cube masked to a selected region, from the main window's "Export" tool group
- **Region as sky coordinates (.txt):** a selected region's shape saved as RA/Dec, importable onto a different cube at the same sky position
- **Contour mask (FITS):** export of the binary region mask with a WCS header
- **Rotated image (FITS):** save of the rotated image with the updated WCS header
- **Convolved image (FITS):** save of the resolution-matched image (the "Pair" output) with the 2-HDU layout (image and `COVERAGE`)
- **Batch ZIP:** a zip archive of all resolution-matched FITS files on a common WCS grid
- The instrument is detected automatically, and the header is preserved.
- The exports are unit-aware and support CGS calibration.

---

## Command-Line Interface (clip.py)

`clip.py` gives the core analysis pipeline as a command-line tool. Use it for scripts, batch jobs, or a headless server where the GUI is not available.

### Setup

```bash
# Run from the project root (one level above cube_explorer/)
python clip.py <command> [options]
```

> **Note:** Keep `clip.py` **outside** the `cube_explorer/` folder, so that the relative `cube_explorer.data_loader` import resolves correctly.

---

### Commands

#### `auto-continuum` — automatic continuum detection and export

Fits a continuum baseline to the integrated spectrum of the cube with median filtering, peak detection, and iterative sigma clipping. It then saves the fitted curve as a text file.

```bash
python clip.py auto-continuum -i <cube.fits> [-p window sigma width distance exclusion iters smoothing]
```

| Flag              | Description                                      | Default            |
| ----------------- | ------------------------------------------------ | ------------------ |
| `-i` / `--input`  | Path to the input FITS cube                      | _(required)_       |
| `-p` / `--params` | Space-separated algorithm parameters (see below) | `51 2.0 1 3 3 2 5` |

**Parameters (in order):**

| Position | Name        | Meaning                                          | Default |
| -------- | ----------- | ------------------------------------------------ | ------- |
| 1        | `window`    | Median-filter window size (channels)            | `51`    |
| 2        | `sigma`     | Peak-detection threshold (× RMS)                | `2.0`   |
| 3        | `width`     | Minimum peak width (channels)                   | `1`     |
| 4        | `distance`  | Minimum distance between peaks (channels)       | `3`     |
| 5        | `exclusion` | Channels to mask on each side of a peak         | `3`     |
| 6        | `iters`     | Sigma-clipping iterations                       | `2`     |
| 7        | `smoothing` | Spline smoothing factor                         | `5`     |

**Algorithm:**

1. Apply a median filter of width `window` as a first-pass continuum estimate.
2. Detect emission peaks above `sigma × RMS` with `scipy.signal.find_peaks`.
3. Mask `exclusion` channels on each side of every detected peak.
4. Interpolate over the masked regions, then apply the median filter again.
5. Sigma-clip the outliers for `iters` passes.
6. Fit a smoothing `UnivariateSpline` (k=3) to the cleaned continuum.
7. Ask for an output filename, then save `<name>.txt` with the columns `Wavelength  Continuum_Fit`.

**Examples:**

```bash
# Default parameters
python clip.py auto-continuum -i data/miri_cube.fits

# Custom parameters: larger window, tighter sigma, more exclusion
python clip.py auto-continuum -i data/miri_cube.fits -p 75 1.5 2 5 5 3 8
```

---

#### `custom-continuum` — continuum from user-supplied points

Fits a cubic spline through a set of user-defined wavelength and flux anchor points, then subtracts the result from the spectrum.

```bash
python clip.py custom-continuum -i <cube.fits> -p <points.csv>
```

| Flag              | Description                                   |
| ----------------- | -------------------------------------------- |
| `-i` / `--input`  | Path to the input FITS cube                  |
| `-p` / `--points` | Path to the CSV file with the anchor points |

**Points file format** — comma-separated, one point per line, no header:

```
3.5e-6,0.012
4.0e-6,0.015
4.8e-6,0.011
```

The spline is fitted with `bc_type='natural'` and evaluated at every wavelength channel. The output is the continuum-subtracted spectrum, saved as `<name>.txt` with the columns `Wavelength  Original_Flux  Continuum_Fit  Subtracted_Result`.

**Example:**

```bash
python clip.py custom-continuum -i data/muse_cube.fits -p my_points.csv
```

---

#### `subtract-continuum` — subtract a continuum from a spectrum

Subtracts a constant offset, a previously exported fitted-continuum file, or a raw-points file from the spectrum of the cube.

```bash
python clip.py subtract-continuum -i <cube.fits> -c <value_or_file>
```

| Flag                 | Description                                                                                |
| -------------------- | ---------------------------------------------------------------------------------------- |
| `-i` / `--input`     | Path to the input FITS cube                                                              |
| `-c` / `--continuum` | A numeric constant, a path to a fitted-continuum `.txt`, or a path to a raw-points `.txt` or `.csv` |

**Three modes, detected automatically from the `-c` argument:**

| Mode                      | How it is detected                                   | Behaviour                                                                       |
| ------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Constant**              | `-c` is a plain number, for example `0.05`         | Subtracts the constant from every channel                                       |
| **Fitted-continuum file** | The file has a `# Type: fitted_continuum` header line | Interpolates the saved curve onto the current wavelength grid, then subtracts   |
| **Raw-points file**       | The file has no recognised header                  | Reads comma-separated `(wavelength, flux)` pairs, fits a cubic spline, subtracts |

Output file columns: `Wavelength  Original_Flux  Continuum_Fit  Subtracted_Result`

**Examples:**

```bash
# Constant baseline
python clip.py subtract-continuum -i data/muse_cube.fits -c 0.03

# Re-use a previously fitted continuum
python clip.py subtract-continuum -i data/muse_cube.fits -c continuum_export.txt

# Subtract from raw anchor points
python clip.py subtract-continuum -i data/nirspec_cube.fits -c my_baseline_points.csv
```

---

#### `2d-rotate` — rotate a 2D FITS image

Rotates a 2D FITS image by any angle and updates the WCS CD matrix, so the RA and Dec coordinates stay correct after the rotation.

```bash
python clip.py 2d-rotate -i <image.fits> -a <angle_degrees>
```

| Flag             | Description                                             |
| ---------------- | ----------------------------------------------------- |
| `-i` / `--input` | Path to the input 2D FITS file                         |
| `-a` / `--angle` | Rotation angle in degrees (a positive value is counter-clockwise) |

**What happens internally:**

- `scipy.ndimage.rotate` with `reshape=True` and `cval=NaN` (an empty border pixel is NaN, not zero)
- The CD matrix is updated: `CD_new = CD_orig × R⁻¹`, where `R` is the 2D rotation matrix for `-angle`.
- `CRPIX` does not change. The reference pixel keeps the same sky coordinate.

The tool asks for an output filename and saves `<name>.fits`.

**Example:**

```bash
python clip.py 2d-rotate -i data/miri_intensity.fits -a 45
# Prompt: Enter filename to save ROTATED FITS: rotated_45deg
# Result: rotated_45deg.fits
```

---

#### `2d-contour` — extract contour coordinates from a 2D FITS image

Generates contour lines at user-specified sigma levels, computes the RA and Dec for every contour vertex with the image WCS, and saves the result as a text file.

```bash
python clip.py 2d-contour -i <image.fits> -l <level1> [level2 ...]
```

| Flag              | Description                                     |
| ----------------- | --------------------------------------------- |
| `-i` / `--input`  | Path to the input 2D FITS file                 |
| `-l` / `--levels` | One or more sigma multipliers (space-separated) |

**Level definition:**

- The background median and RMS are estimated with the Astropy `sigma_clipped_stats` function (3σ clipping).
- Each level `N` draws a contour at `median + N × RMS`.

**Output file format** — comma-separated:

```
# IFU Analyzer 2D - Contour Export
# level, x_pixel, y_pixel, RA_deg, Dec_deg
3.000000e+00, 42.1234, 18.7654, 67.90591, 18.20682
...
```

The tool asks for an output filename and saves `<name>.txt`.

**Examples:**

```bash
# Single level
python clip.py 2d-contour -i data/moment0.fits -l 3

# Multiple levels
python clip.py 2d-contour -i data/moment0.fits -l 3 5 10 20

# High-significance contours only
python clip.py 2d-contour -i data/miri_intensity.fits -l 5 10 30 50
```

---

### Output File Formats

Every CLI command that produces a spectrum saves a tab-delimited or comma-delimited `.txt` file with a commented header. A file that the CLI exports is compatible with the `subtract-continuum` import: a continuum file from `auto-continuum` can go straight to `subtract-continuum -c`.

| Command              | Output columns                                               |
| -------------------- | ----------------------------------------------------------- |
| `auto-continuum`     | `Wavelength  Continuum_Fit`                                  |
| `custom-continuum`   | `Wavelength  Original_Flux  Continuum_Fit  Subtracted_Result` |
| `subtract-continuum` | `Wavelength  Original_Flux  Continuum_Fit  Subtracted_Result` |
| `2d-rotate`          | FITS file                                                    |
| `2d-contour`         | `level, x_pixel, y_pixel, RA_deg, Dec_deg`                   |

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip (the Python package installer)

### Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies: `PyQt5`, `matplotlib`, `numpy`, `scipy`, `pandas`, `scikit-learn`, and `astropy`. `pymc` and `nutpie` are necessary for the Bayesian censored-regression fits (the rotational-diagram upper limits). `hitran-api` is necessary for live HITRAN database queries. Refer to `requirements.txt` for the full list with versions.

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
# On Unix or macOS
source venv/bin/activate
```

### Data Files

The application reads FITS data cubes and 2D FITS images. Supported data:

- MUSE: 3D spectral cubes (`.fits`)
- MIRI: mid-infrared spectral cubes and 2D slice maps (`.fits`)
- NIRSpec: near-infrared spectral cubes (`.fits`)
- 2D FITS images: intensity maps, moment maps, and Gaussian parameter maps

The repository does not include FITS files. Provide your own data from observations or an archive.

## Usage

### Basic Operation

1. Start the application:

```bash
python main.py
```

2. Load and view the data:
   - Click "Open FITS" and select a data cube. The file loads directly.
   - Use the spectrum plot (bottom) to move along the wavelength or velocity axis.
   - Click "Set Left Slice" and "Set Right Slice" to set the two slice views.
   - Adjust the colour scale and the colormap.

### Region Analysis

3. Select and examine a region:
   - Click a shape button (point, rectangle, circle, ellipse, or freehand) to start a region tool.
   - Draw the region on either slice view.
   - The statistics update automatically.
   - The extracted spectrum shows in the bottom panel.
   - Click the copy button to copy a region to the other panel.

### Gaussian Fitting

4. Fit a Gaussian line profile. Open the "Gaussian Fit" group.

   **Single-pixel fitting:**
   - Click "Left" to mark the left boundary on the spectrum.
   - Click "Right" to mark the right boundary on the spectrum.
   - Click "Fit Gaussian" to run the fit.
   - Read the fit results, the parameters, and the residuals.
   - Open "Pixel Diagnostics" for the fit quality and the residual detail.

   **Full-cube mapping:**
   - Mark the boundaries as above.
   - Click "Fit Gaussian Map" to fit every pixel.
   - Wait for the background processing (10 to 30 seconds is typical).
   - View the parameter maps (Area, Amplitude, Centre, FWHM, and so on).
   - Use "Pixel Diagnostics" to find a problem region.
   - Export the residuals cube as FITS if it is necessary.

### Continuum Fitting

5. Subtract the continuum. Open the "Continuum Tools" group.

   **Automatic method:**
   - Click "Automatic Continuum".
   - Click "Fit Continuum" to open the parameter dialog.
   - Set the window size, the sigma threshold, the peak-exclusion width, the iterations, and the smoothing factor.
   - The fitted continuum shows in orange over the original spectrum.
   - Click "Subtract Continuum". The full subtracted spectrum is shown.

   **Custom interactive method:**
   - Click "Custom Continuum".
   - Click "Custom Subtract" to enter point-placement mode.
   - Click on the spectrum to place an anchor point. The point snaps to the data.
   - Use "Delete Point" mode to remove a point. A minimum of 3 points is necessary.
   - Click "OK". The fitted cubic-spline continuum is shown in orange as a preview.
   - Confirm the subtraction in the dialog. "No" returns to editing mode.
   - The full subtracted spectrum is shown after the confirmation.
   - Save the result with "Save Points" or "Save Spectrum".

### 2D Image Viewer

6. Examine a 2D FITS image.

   > Click the `2D` button in the main window to open the viewer.

   - Load an image with "Open 2D FITS (Left)" or "Open 2D FITS (Right)".
   - The WCS grid (RA and Dec) shows automatically if the FITS header has WCS.

   **Rotation:**
   - Select "Counter-Clockwise" or "Clockwise", and set the angle.
   - Click "Rotate Left" or "Rotate Right".
   - The WCS labels stay correct at any angle.

   **Contours.** Open the "Contour" group.
   - Click "Plot" and set:
     - The panel to draw on ("Plot contour ON")
     - The source image ("Generate contour FROM" — the same panel, or the other panel for a WCS-matched cross-image contour)
     - The background std (blank for the Astropy automatic value, or enter a value)
     - The sigma levels as a comma-separated list, for example `3, 5, 10`
     - The contour colour
   - A contour line is drawn at each N × sigma above the background median.
   - "Clear" removes all contour overlays from a panel.

   **Save contour:**
   - Click "Save", then choose a panel. The tool saves a `.txt` file with the columns `level, x_pixel, y_pixel, RA_deg, Dec_deg`.

   **Load a saved contour:**
   - Click "Load" and select a binary FITS mask file.
   - The mask is drawn on the chosen panel through the WCS mapping.

   **Superimpose:**
   - Load a different image into the left panel and the right panel.
   - Click "Superimpose".
   - Choose the background panel (the reference grid).
   - Select the overlay colormap and the alpha (0.5 is a good starting value).
   - The overlay image is reprojected onto the background WCS grid and blended.
   - The terminal prints the peak separation in arcsec, so you can confirm the alignment.

   **Pair (resolution matching, two images).** Open the "Convolution" group.
   - Load two images with a different pixel scale into the left panel and the right panel.
   - Click "Pair".
   - The confirmation dialog shows which image is convolved, the Gaussian kernel σ, and the output grid size.
   - Click "Yes" to run. The output FITS file has a 2-HDU layout (image and `COVERAGE`).
   - The success dialog shows the flux audit: `Ratio out/in : 1.000000`.
   - If both panels have the same pixel scale, a message explains that no convolution is necessary.

   **Batch (resolution matching, several files):**
   - Click "Batch".
   - Click "+ Add FITS Files" and select two or more 2D FITS files.
   - Click "Process & Download ZIP".
   - Choose a save path for the output `.zip` file.
   - All outputs share the same WCS grid (the coarsest pixel scale, the union field of view), ready for a comparison.

### Rovibrational Flux Mapping

7. Extract the flux from the full data cube for any target species. Open the "Lines & Fit" group.
   - Load lines with "Fetch HITRAN" or "Import Lines", then choose the ones you want with "Show Lines".
   - Optionally, click "Contam. Lines" to exclude contaminating lines from the fit.
   - Click "Flux Map". The continuum subtraction always runs automatically before the detection, with the default or the custom parameters.
   - View the flux maps, the SNR statistics, and the detection quality.
   - Export the results with the integrated flux estimates.

### Rotational-Diagram Analysis

8. Fit the rotational temperatures.
   - Click "Rotational Diagram" to open the analyzer window.
   - Fetch line data with "Fetch from HITRAN", or import a custom CSV with "Import from CSV".
   - Select the lines to include. Enter the measured flux and the flux error for each line.
   - For a non-detection, mark the line as an upper limit. Use "Load FITS Files" and "Calculate Fluxes" to derive the limit from the cube.
   - Enter the emitting radius ("R (au)") and the distance ("Distance (pc)"), then choose a 1T or a 2T fit in "Fit".
   - Click "Run Fit". If any line is an upper limit, the fit switches to PyMC Bayesian censored regression instead of ordinary least squares.
   - Read the fitted temperature or temperatures and the column densities in the results dialog.

### Data Export

9. Export the results:
   - **Spectrum:** click "Save Spectrum" below the spectrum plot.
   - **Continuum:** click "Save Points" to export the anchor points, or "Save Spectrum" for the full four-column file.

### Additional Features

10. Other tools:
   - **Header viewer:** open "Header" to inspect the FITS header of a selected HDU.
   - **Line catalog:** "Import Lines" loads a custom spectral-line catalog (LST format).
   - **HITRAN database:** "Fetch HITRAN" queries molecular lines. "Show Lines" lists the loaded lines and lets you choose which ones to plot.
   - **Normalize spectrum:** the "Normalize" checkbox uses baseline points for a normalised display.
   - **Doppler shift:** "Shift" applies a multiplicative wavelength correction (km/s). Use "Apply Shift", "Cancel Preview", and "Revert Shift" to control it.
   - **Wavelength range:** set the minimum and the maximum for a linked view update.
   - **New window:** the new-window button opens an independent application window for a multi-file comparison.

## File Structure

```
IFU-Analyzer/
├── main.py                              # Application entry point (GUI)
├── clip.py                              # Command-line interface
├── example_basic_api.py                 # Scripted, non-GUI analysis example
├── cube_explorer/
│   ├── __init__.py
│   ├── gui.py                           # Main window: spectrum and slice plots, layout (~3000 lines)
│   ├── gui_continuum.py                 # Continuum-subtraction mixin (automatic and custom modes)
│   ├── gui_gaussian.py                  # Gaussian-fitting mixin (single-pixel, full-cube, multi-Gaussian)
│   ├── gui_flux_map.py                  # Flux-map mixin
│   ├── gui_spectrum.py                  # Spectrum-plot mixin (tabs, normalisation, overplot)
│   ├── gui_region_selection.py          # Shared region-selection controller (main window and 2D viewer)
│   ├── gui_rotational_diagram.py        # Rotational-diagram analyzer window
│   ├── gui_2d.py                        # 2D FITS dual viewer (WCS, rotation, contours, superimpose, resolution matching)
│   ├── analyzer_2d.py                   # GUI-free Gaussian convolution and flux-conserving reprojection for gui_2d.py
│   ├── data_loader.py                   # CubeData: FITS loading and instrument detection
│   ├── gaussian_fitter.py              # Single and multi-Gaussian line fitting; full-cube parallel fits
│   ├── flux_map_extractor.py           # Full-cube rovibrational flux mapping for any species
│   ├── rotational_diagram_h2_fitter.py       # H2 rotational-diagram temperature fits (1T and 2T)
│   ├── rotational_diagram_general_fitter.py  # Rotational-diagram fits for non-H2 species
│   ├── visualization.py               # SpectrumPlotter and SlicePlotter rendering classes
│   ├── region_selector.py             # Matplotlib region-selector widgets
│   ├── pixel_diagnostics.py           # Pixel-level fit-inspection dialog
│   ├── colormap_dialog.py             # Colormap-customisation dialog
│   ├── continuum_dialog.py            # Automatic-continuum parameter dialog
│   ├── custom_continuum.py            # Interactive continuum anchor-point selector
│   ├── flux_map_dialogs.py            # Flux-map configuration dialog, worker, and results dialog
│   ├── gaussian_dialogs.py            # Gaussian-map worker and results dialog
│   ├── species_dialog.py             # HITRAN species-selection dialog
│   ├── line_catalog.py               # Line identification and HITRAN integration (cached in hitran_local/)
│   ├── unit_converter.py             # Wavelength, frequency, velocity, and CGS unit conversion
│   └── utils.py                      # Instrument detection and helper functions
└── data/                            # Local sample FITS files (not tracked)
```

The `CubeExplorer` class in `gui.py` is composed from the `gui_*` mixins listed above.

---

## Troubleshooting

1. **Memory errors:**
   - Close other applications.
   - Use a smaller data cube.
   - The application uses memory mapping for a large file automatically.

2. **Display problems:**
   - Confirm that PyQt5 is installed correctly.
   - Update the graphics drivers if the rendering is wrong.
   - Try a different colour scale (Log or Power) for a better contrast.

3. **File-loading errors:**
   - Confirm that the file format matches a supported instrument.
   - Check the FITS header for the required keywords (`INSTRUME`, `TELESCOP`).
   - Confirm that the file is not corrupt.
   - The 2D viewer reads 2D FITS images directly.

4. **The WCS grid is not visible:**
   - Confirm that the FITS header has `CTYPE1` and `CTYPE2` with an RA/Dec projection.
   - Check the terminal for the `DEBUG rebuild_axes` messages, which report the WCS parse status.
   - A non-celestial WCS falls back to plain pixel axes automatically.

5. **Superimpose misalignment:**
   - Check the terminal for `DEBUG superimpose: Peak separation`. It should be less than 0.5 px.
   - A large separation indicates a WCS header difference between the two files.

6. **"Pair" or "Batch" problems:**
   - If the output sum differs from the input sum, confirm that both files have a valid `CDELT` or CD matrix in the header. The pixel scale is necessary to compute the kernel size.
   - The flux-audit dialog (`Ratio out/in`) should always read `1.000000`. A large deviation indicates a corrupt WCS header.
   - A "same resolution" message is expected when both panels have the same pixel scale. No action is necessary.
   - A "No Sky Overlap" warning means that the two images cover disjoint sky regions. Load images of the same field. In "Batch" mode, the warning lists the file pairs that do not share sky, so you can remove the off-field file and try again.

7. **Continuum-subtraction problems:**
   - Every fit, in either mode, is applied to the original spectrum. You can refit and re-subtract repeatedly without a compounding error.
   - A "No Continuum" message means that you clicked "Subtract Continuum" before the fit. Click "Fit Continuum" first, or place at least 3 custom points and click "OK".
   - A "Too Few Points" message in custom mode means that you tried to fit with fewer than 3 anchor points. A cubic spline needs at least 3.
   - For a very noisy spectrum, increase the median-filter window size to 101 and the sigma threshold to 3.0.

8. **Gaussian-fitting problems:**
   - Select an adequate wavelength range (more than 4 points).
   - Check for a positive peak after the baseline subtraction.
   - An "x0 is infeasible" error is handled automatically with bounds clamping.
   - A low success rate can indicate a noise-dominated region.

9. **CLI (clip.py) problems:**
   - Run `clip.py` from the project root, not from inside `cube_explorer/`.
   - If you see `ImportError: Could not import CubeData`, move `clip.py` up one directory level.
   - The `--params` flag for `auto-continuum` needs all seven values in order: `window sigma width distance exclusion iters smoothing`.
