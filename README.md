# IFU Analyzer

IFU Analyzer is a Python application for the analysis of Integral Field Unit (IFU) spectroscopic data. It reads 3D spectral data cubes from MUSE, MIRI, and NIRSpec. The interface shows the data cube, extracts spectra from selected regions, fits Gaussian line profiles, subtracts the continuum, and exports the results.

> [!WARNING]
> **This project is still under development.** Some features can change, and some parts can have bugs.
>
> If you find an issue, please tell us. [Open an issue on GitHub](https://github.com/ilariapascucci/IFU-Analyzer/issues) and include the steps to repeat it, the instrument (MIRI, NIRSpec, or MUSE), and any error message.

## Features

### Instrument Support

- **MUSE** (VLT): optical spectroscopy
- **MIRI** (JWST): mid-infrared spectroscopy
- **NIRSpec** (JWST): near-infrared spectroscopy

Some tools have a narrower scope:

- **Fetch HITRAN, Flux Map, and the rotational diagram** read wavelengths in µm, so use them on MIRI and NIRSpec cubes. The offline HITRAN data in `hitran_local/` covers 1 to 30 µm. For another range, the app fetches the lines from HITRAN online, so it needs an internet connection.
- **Pair and Batch** (resolution matching) work on MIRI and NIRSpec images only.

### Interactive Visualization

- Full-spectrum display with wavelength and velocity axes
- Dual slice view for a wavelength or velocity comparison
- The spectrum updates when you change the selection
- Region tools: point, rectangle, circle, ellipse, and freehand polygon
- Draggable region handles for a move or a resize
- Colour scales: Linear, Log, and Power (with an adjustable gamma), with an absolute or a percentile colour range
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
- Intensity statistics panel: minimum, maximum, mean, sum, and standard deviation, plus the colour range in use, with the flux unit from the FITS header
- Colour scales: Linear, Log, and Power (with an adjustable gamma), with an absolute or a percentile colour range
- Colormap selection through a dialog

#### Image Rotation

- Rotate an image by any angle. Select "Counter-Clockwise" or "Clockwise", then click "Rotate (Left)" or "Rotate (Right)".
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
- **Load:** reads a saved contour (`.txt` or `.csv`) and draws it on either panel through the WCS mapping. The contour lines up on an image with a different pixel scale or orientation.

#### WCS Superimpose

- Overlay two images that have a different WCS grid (a different pixel scale, array size, orientation, or pointing) onto one panel.
- The overlay image is reprojected onto the pixel grid of the background image with `reproject_exact` from the `reproject` package. This method conserves flux.
- The overlay colormap and the alpha transparency (0.1 to 1.0) are configurable.

#### Resolution Matching: Pair (two images)

Match two images of one field to the same spatial resolution. This is the standard step before a flux comparison between two lines or two instruments.

- **Input:** two MIRI or NIRSpec images in a flux-per-pixel unit, one in each panel. The images must overlap on the sky. A surface-brightness unit such as MJy/sr is rejected.
- Each image needs its line wavelength. The tool reads it from the `LINEWAVE` header key (written by the rotational-diagram "Save Maps"), or from the middle of `LAMMIN` and `LAMMAX` (written by the Gaussian-map export). If the header has neither, the tool asks for the wavelength.
- Click **Pair** to open the confirmation dialog. It shows the two PSF widths, the kernel width, and the output grid.
- The image with the **longer wavelength** is the reference, because its PSF is wider. The tool convolves the other image with a Gaussian kernel up to the reference PSF.
- Kernel width: `σ = √(σ_ref² − σ_image²)`. Each σ comes from the PSF FWHM of the instrument at the line wavelength. MIRI: `FWHM(") = 0.033 × λ(µm) + 0.106`. NIRSpec: an approximate quadratic fit.
- The tool then reprojects the convolved image onto the **native pixel grid of the reference image**, and it conserves the flux.
- The output is a 2-HDU FITS file: the image, and a `COVERAGE` extension (the per-pixel overlap fraction, 0 to 1).
- If both panels have the same wavelength, the tool shows a "nothing to convolve" message and does not run.
- The save dialog shows a flux audit: the input total, the output total, and `Ratio out/in`. A ratio close to 1 (within about 1%) means that the flux is conserved. A clearly lower ratio means that part of the image falls outside the grid of the reference image. A small deviation can also come from NaN pixels at the edge of the image footprint.

#### Resolution Matching: Batch (multiple images)

Bring several images to one resolution in a single step.

- Click **Batch** to open the batch dialog.
- Click **+ Add FITS Files** to add two or more 2D FITS files. The requirements are the same as for Pair: MIRI or NIRSpec, a flux-per-pixel unit, and an overlap on the sky.
- The tool reads or asks for the line wavelength of each file, as in Pair.
- The file with the **longest wavelength** is the reference. The tool convolves every other file to the PSF of the reference and reprojects it onto the native grid of the reference. The reference file is not smoothed.
- Click **Process & Download ZIP** and choose a save path. The `.zip` file has one `matched_<name>.fits` file for each input. All files are on the same grid, ready for a pixel-by-pixel comparison.
- Each output FITS file includes the `COVERAGE` extension.
- The completion dialog shows the flux audit (the output total divided by the input total) for each file.

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
  - Integrated area (intensity map) and its error
  - Peak amplitude
  - Line centre and velocity (kinematics)
  - Line width (σ and FWHM)
  - Fit quality (χ²)
  - Fit status and success mask

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
  - A separate analyzer window for rotational-temperature diagrams (1T and 2T fits), for H2 and CO
  - Fetch line data from HITRAN, or import a custom CSV of transitions.
  - Select or clear individual lines. Enter the flux and the flux error for each line, or calculate them from loaded cubes.
  - Type the flux error or the error percentage. The other value updates automatically.
  - A flux that "Calculate Fluxes" made is read-only. If you edit its error, the cell turns yellow.
  - Mark a non-detection as an upper limit. "Calculate Fluxes" computes the limit directly from a FITS cube.
  - Automatic method selection: ordinary least squares (`scipy.curve_fit`) when all points are detections; PyMC Bayesian censored regression when one or more upper limits are present.
  - Ortho-to-para ratio (OPR) fit for H2
  - A distance input (pc), which is required, and an emitting-radius input (au), which is optional. With a radius, N is a column density. Without a radius, N is a total molecule count.
  - The results dialog reports the emitting radius, the distance, the OPR, the fitted temperature or temperatures, and the column densities.
  - "Save Fit" in the results dialog, and in the dialog of a pixel in the temperature map, saves the results, the plotted points, and the fit line to a CSV file.
  - Restrict "Calculate Fluxes" and "Run Fit" to a region. Draw the region on a flux map, or import a saved region. Save the region to reuse it.
  - Read the FITS header of each loaded cube with "Header". Open a second analyzer window with the new-window button, for example to compare two sets of cubes.
  - **Pixel-by-pixel temperature maps:** "Calculate Maps" makes a flux map for each line. "Convolve Maps" puts the maps on one grid at the coarsest resolution. "Run Map" fits the rotational diagram at every pixel. "Load Maps" adds flux maps that you made in another tool.

### Data Processing

- Coordinate conversion (pixel to RA and Dec)
- Region-based spectral extraction
- Statistical analysis of a selected region
- Spectrum normalisation to the peak of the full spectrum, or to the peak of a region's own spectrum
- Continuum baseline subtraction (interactive and automatic)
- Spectral-line identification (the HITRAN database and custom catalogs)
- Multi-line automatic detection with SNR thresholding
- Doppler-shift correction (a multiplicative wavelength shift)
- Wavelength-to-frequency conversion with CGS unit support
- Full `astropy` Quantity support for unit handling
- Memory-efficient handling of a large data cube
- Dynamic wavelength and velocity range windowing

### Export Capabilities

- **Continuum results (ASCII):** the "Save Spectrum" button writes a four-column file (wavelength, original spectrum, fitted continuum, and continuum-subtracted spectrum) in tab-delimited format. A missing column (for example, no continuum is fitted yet) is written as `NaN`. The `# Type: full_results` header line identifies the file format.
- **Residuals cube (FITS):** export of the 3D normalised residuals cube with full header metadata
- **Flux maps:** export of the full-cube fitting results with the integrated flux estimates
- **Contour coordinates (.txt):** export of the contour path vertices with the pixel and RA/Dec positions and the sigma-level labels
- **Slice or region (FITS):** a single 2D slice, or the full cube masked to a selected region, from the main window's "Export" tool group
- **Region as sky coordinates (.txt):** a selected region's shape saved as RA/Dec, importable onto a different cube at the same sky position
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

#### `auto-continuum`: automatic continuum detection and export

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

#### `custom-continuum`: continuum from user-supplied points

Fits a cubic spline through a set of user-defined wavelength and flux anchor points, then subtracts the result from the spectrum.

```bash
python clip.py custom-continuum -i <cube.fits> -p <points.csv>
```

| Flag              | Description                                   |
| ----------------- | -------------------------------------------- |
| `-i` / `--input`  | Path to the input FITS cube                  |
| `-p` / `--points` | Path to the CSV file with the anchor points |

**Points file format** (comma-separated, one point per line, no header):

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

#### `subtract-continuum`: subtract a continuum from a spectrum

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

#### `2d-rotate`: rotate a 2D FITS image

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

#### `2d-contour`: extract contour coordinates from a 2D FITS image

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

**Output file format** (comma-separated):

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

## Python API (cube_explorer.api)

`cube_explorer/api.py` gives the analysis functions of the GUI without the GUI. Use it in scripts and in Jupyter notebooks. The functions return plain Python objects, NumPy arrays, and dictionaries.

```python
from cube_explorer.api import load_cube, fit_line

cube = load_cube("my_data.fits")
result = fit_line(wavelength=cube.wavelength, spectrum=cube.spectrum, x1=7.43, x2=7.49)
print(f"Line centre: {result.center:.4f} um, FWHM: {result.fwhm:.4f} um")
```

| Task                | Functions                                                                                                   |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| Load data           | `load_cube`, `extract_pixel_spectrum`, `extract_region_spectrum`                                            |
| Continuum           | `fit_continuum`, `save_continuum_cubes`                                                                     |
| Gaussian fitting    | `fit_line`, `fit_hitran_lines`, `measure_equivalent_width`, `create_gaussian_map`, `export_residual_cube`   |
| Flux                | `extract_flux`, `create_flux_map`                                                                           |
| Rotational diagrams | `rotational_diagram`, `fit_region`, `fit_pixel`, `create_rotational_diagram_map`, `save_rotational_diagram_map_fits`, and the `_multi` versions for cubes at different pixel scales |

- Each function has a docstring with its arguments and an example. Use `help(function_name)` to read it.
- The map functions use several processes. On Windows and macOS, put the code of your script under `if __name__ == "__main__":`.

---

## Installation

### Prerequisites

- Python 3.13. The pinned versions in `requirements.txt` are tested on 3.13 only. Python 3.12 also works, but it is not tested. Python 3.11 fails at the install step. If you do not have 3.13, use conda (see Setup). For an exact copy of the tested environment, install `requirements-lock.txt` instead of `requirements.txt`.
- pip (the Python package installer)
- A 64-bit computer with Windows, Linux (Intel or AMD processor), or macOS (Intel or Apple Silicon). Linux on an ARM processor is not supported, because PyQt5 has no files for it.

### Setup

1. Get the code. Using Git is recommended, because it lets you update later. If Git is not on your computer yet: on macOS run `brew install git`, on Windows use the installer from https://git-scm.com, and on Linux use your package manager (for example `sudo apt install git`). Then clone the repository:

```bash
git clone https://github.com/ilariapascucci/IFU-Analyzer.git
cd IFU-Analyzer
```

   After the clone, you are on the `main` branch. To switch to another branch, run `git checkout <branch-name>`.

   Without Git: open https://github.com/ilariapascucci/IFU-Analyzer, click **Code**, then **Download ZIP**. Extract the ZIP file and open a terminal in the extracted folder.

2. Create an environment with Python 3.13. With conda, which installs Python for you:

```bash
conda create -n ifu-analyzer python=3.13
conda activate ifu-analyzer
```

   Without conda, use `venv`. Python 3.13 must already be installed:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or macOS
source venv/bin/activate
```

3. Install the dependencies. This can take several minutes:

```bash
pip install -r requirements.txt
```

Core dependencies: `PyQt5`, `matplotlib`, `numpy`, `scipy`, `pandas`, `scikit-learn`, and `astropy`. `pymc` and `nutpie` are necessary for the Bayesian censored-regression fits (the rotational-diagram upper limits). `hitran-api` is necessary for live HITRAN database queries. Refer to `requirements.txt` for the full list with versions.

### Updating

If you cloned the repository with Git, run `git pull` in the `IFU-Analyzer` folder each time before you run the program. If `requirements.txt` changed, run `pip install -r requirements.txt` again. A ZIP download cannot update itself: download the ZIP file again.

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
   - The extracted spectrum shows in the bottom panel, on the "Left Region" or "Right Region" tab.
   - Use the copy button to copy a selection, a colormap, or a colour scale to the other panel.

### Gaussian Fitting

4. Fit a Gaussian line profile. Open the "Gaussian Fit" group.

   **Single-line fitting (the spectrum on the active tab):**
   - Click "Fit Left", then click the spectrum to mark the left edge of the fit window.
   - Click "Peak", then click the spectrum to mark the line peak.
   - Click "Fit Right", then click the spectrum to mark the right edge. "Fit Left" must be to the left of "Fit Right".
   - Optionally, mark a line-free noise region with "Noise Left" and "Noise Right".
   - Click "Fit Gaussian" to run the fit.
   - Read the fit results: the centre, the amplitude, the width, the area, the baseline, and R².

   **Full-cube mapping:**
   - Mark "Fit Left", "Peak", and "Fit Right" as above. The map fit also needs the noise region ("Noise Left" and "Noise Right").
   - Click "Fit Gaussian Map" to fit every pixel.
   - Wait for the background processing (10 to 30 seconds is typical).
   - View the parameter maps (Area, Amplitude, Centre, Velocity, FWHM, and so on).
   - The Velocity map needs a rest line. When you select "Velocity" in the parameter list, choose the molecule or atom of the line in the fit window. A table lists the lines of that species from HITRAN and the LST catalog. Select the line that the peak shows. Its vacuum wavelength is zero velocity. If you cancel, the Velocity map is not made, and the results window gives the reason. To use another line, click "Change rest line..." next to the parameter list. The map redraws with the new line.
   - To add lines (for example from HITEMP), click "Load line list file..." in the table. Use a CSV with the columns `wavelength` and `species`, a HITRAN-format `.par` file, or an LST catalog `.lst` file. Give vacuum wavelengths in a CSV. Unzip a HITEMP file before you load it. Lines from a file show in amber.
   - The LST catalog starts at 0.70 µm, and HITRAN has no atomic lines. For a MUSE line such as H-alpha, you must load a file.
   - MUSE cubes use air wavelengths (axis type `AWAV`). The app converts the rest line to air before it makes the map. A cube header that gives no `AWAV` or `WAVE` axis type has no Velocity map.
   - With "Pixel Select: ON" in the results window, click a pixel on the map to see its fit.
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
   - Click "Rotate (Left)" or "Rotate (Right)".
   - The WCS labels stay correct at any angle.

   **Contours.** Open the "Contour" group.
   - Click "Plot" and set:
     - The panel to draw on ("Plot contour ON")
     - The source image ("Generate contour FROM": the same panel, or the other panel for a WCS-matched cross-image contour)
     - The background std (blank for the Astropy automatic value, or enter a value)
     - The sigma levels as a comma-separated list, for example `3, 5, 10`
     - The contour colour
   - A contour line is drawn at each N × sigma above the background median.
   - "Clear" removes all contour overlays from a panel.

   **Save contour:**
   - Click "Save". If both panels have contours, choose one. The tool saves a `.txt` file with the columns `level, x_pixel, y_pixel, RA_deg, Dec_deg`.

   **Load a saved contour:**
   - Click "Load", choose a panel, and select a saved contour `.txt` or `.csv` file. Then choose the line colour.
   - The contour is drawn on the chosen panel through the WCS mapping.

   **Superimpose:**
   - Load a different image into the left panel and the right panel.
   - Click "Superimpose".
   - Choose the background panel (the reference grid).
   - Select the overlay colormap and the alpha (0.5 is a good starting value).
   - The overlay image is reprojected onto the background WCS grid and blended.

   **Pair (resolution matching, two images).** Open the "Convolution" group.
   - Load two MIRI or NIRSpec flux-per-pixel images of the same field into the left panel and the right panel.
   - Click "Pair". If a header has no `LINEWAVE` key, enter the line wavelength (in µm) when the tool asks.
   - The confirmation dialog shows which image is convolved, the two PSF widths, the kernel σ, and the output grid size. The image with the longer wavelength is the reference.
   - Click "Yes" to run, then choose a save path. The output FITS file has a 2-HDU layout (image and `COVERAGE`).
   - The success dialog shows the flux audit: the input total, the output total, and `Ratio out/in`.
   - If both panels have the same wavelength, a message explains that there is nothing to convolve.

   **Batch (resolution matching, several files):**
   - Click "Batch".
   - Click "+ Add FITS Files" and select two or more 2D FITS files (MIRI or NIRSpec, flux-per-pixel units, the same field).
   - Click "Process & Download ZIP" and choose a save path for the output `.zip` file.
   - The file with the longest wavelength is the reference. All outputs are on its pixel grid, ready for a comparison.

### Rovibrational Flux Mapping

7. Extract the flux from the full data cube for any target species. Open the "Lines & Fit" group.
   - Load lines with "Fetch HITRAN" or "Import Lines", then choose the ones you want with "Show Lines".
   - Optionally, click "Contam. Lines" to exclude contaminating lines from the fit.
   - Click "Flux Map". The continuum subtraction always runs automatically before the detection, with the default or the custom parameters.
   - View the flux maps, the SNR statistics, and the detection quality.
   - A pixel with no detected line is NaN (empty) in the flux map, not 0. The statistics count the detected pixels only.
   - Export the results with the integrated flux estimates.

### Rotational-Diagram Analysis

8. Fit the rotational temperatures.
   - Click "Rotational Diagram" to open the analyzer window. The molecule shows in the top row. It shows "--" until you fetch or import lines.
   - Fetch line data with "Fetch from HITRAN", or import a custom CSV with "Import from CSV".
   - Select the lines to include. Click a check box or drag down the column. "Select All" and "Deselect All" change every line.
   - Enter the measured flux for each line. Enter the flux error or the error percentage. The other value updates when you change one.
   - For a non-detection, mark the line as an upper limit. Use "Load Cubes" and "Calculate Fluxes" to derive the flux or the limit from the cube.
   - The flux from "Calculate Fluxes" is read-only. You can still edit its error. The error cell then turns yellow, and its tooltip shows the value from the fit.
   - Click "Header" to read the FITS header of a loaded cube.
   - Enter the distance ("Distance (pc)"), which is required. Optionally enter the emitting radius ("Emitting Radius (au)"). Both fields start empty. Without a radius, N is a total molecule count instead of a column density. Then choose a 1T or a 2T fit in "Fit".
   - Click "Run Fit". If any line is an upper limit, the fit switches to PyMC Bayesian censored regression instead of ordinary least squares.
   - Read the fitted temperature or temperatures and the column densities in the results dialog. The dialog also shows the radius and the distance that the fit used.
   - Click "Save Fit" to save the results to a CSV file. The results are the lines that start with "#". Below them is one row for each plotted point: the wavelength, E_u, log10(N_u/g_u), its error, and the upper-limit flag. After a blank line comes the fit line, as plotted: 200 values of E_u and the model log10(N_u/g_u). A 2T fit has a column for each component and one for the combined model. The file does not contain the flux values. "Save Fluxes" saves those.
   - Click "Save Fluxes" to save the fluxes as a CSV file. "Import from CSV" reads the file back.

   **Start again:**
   - Click "Clear Cubes" to remove the loaded cubes. This also removes the region, the calculated fluxes, and the calculated maps. The window asks first. Fluxes that you typed stay.
   - When you fetch or import a new line list and cubes are loaded, the window asks if it must clear the cubes.
   - To keep the first analysis, click the new-window button. The new window starts empty.

   **Temperature map (every pixel):**
   - Load the cubes with "Load Cubes". To add a flux map that you made in another tool, use "Load Maps".
   - Click "Calculate Maps". The tool asks for the detection threshold (3σ is the default) and makes one flux map for each checked line. If you check another line later, it makes only the map of the new line. It keeps the other maps, unless you change the threshold, the contaminating lines, or the cubes. Double-click "View" in the "Map" column to see a map.
   - Click "Convolve Maps" to put all maps on one grid at the coarsest resolution. It needs two or more lines. It shows its plan first. Double-click "View" in the "Convolved Map" column to see a map.
   - Click "Run Map". It needs at least 3 lines (5 lines for a 2T fit). The results window shows the temperature map, and a click on a pixel refits the diagram of that pixel.
   - Uncheck and check lines without a new calculation. The flux maps stay, and the "Map" and "Convolved Map" links stay. "Run Map" works at once if all checked lines are in the last convolution and the coarsest checked line is the reference line of that convolution. If you uncheck the coarsest line, or check a line that is not in the last convolution, "Run Map" is off until you click "Convolve Maps" again. "Convolve Maps" uses the flux maps that you already have. It does not fit them again.
   - Click "Save Maps" to save the maps as FITS and PNG files in one zip file.

   **Region (optional):**
   - Click "Draw Region" to draw a region on the flux map of a line. It needs "Calculate Maps" first. Or click "Import Region" to load a region that you saved with "Save Region" in the main window or in the 2D viewer.
   - Click "Save Region" to save the region as a text file. The main window and the 2D viewer can read it.
   - "Calculate Fluxes" and "Run Fit" then use only that region. "Delete Region" goes back to the whole cube.
   - A region change clears the calculated fluxes, because they belong to the old region. Fluxes that you typed stay.
   - "Calculate Maps", "Convolve Maps", and "Run Map" always use the whole cube.

### Data Export

9. Export the results:
   - **Spectrum:** click "Save Spectrum" in the "Export" group.
   - **Continuum:** click "Save Points" to export the anchor points, or "Save Spectrum" for the full four-column file.

### Additional Features

10. Other tools:
   - **Header viewer:** open "Header" to inspect the FITS header of a selected HDU.
   - **Line catalog:** "Import Lines" loads a custom spectral-line catalog (LST format).
   - **HITRAN database:** "Fetch HITRAN" queries molecular lines. "Show Lines" lists the loaded lines and lets you choose which ones to plot.
   - **Normalize spectrum:** the "Normalize" checkbox divides every spectrum by the peak of the full spectrum. "Show Left" and "Show Right" overlay a region's spectrum on the "Full" tab. "Self-Normalize Left" and "Self-Normalize Right" divide a region's spectrum by its own peak.
   - **Doppler shift:** "Shift" applies a multiplicative wavelength correction (km/s). Use "Apply Shift", "Cancel Preview", and "Revert Shift" to control it.
   - **Wavelength range:** the "λ min" and "λ max" boxes (in µm) in the "Lines & Fit" group set the range for "Fetch HITRAN", "Import Lines", "Contam. Lines", and "Flux Map". They fill from the cube when you load it.
   - **New window:** the new-window button opens an independent application window for a multi-file comparison.

## File Structure

```
IFU-Analyzer/
├── main.py                                   # Application entry point (GUI)
├── clip.py                                   # Command-line interface
├── example_basic_api.py                      # Scripted, non-GUI analysis example
├── warm_hitran_cache.py                      # Builds the offline HITRAN starter pack in hitran_local/
├── cube_explorer/
│   ├── __init__.py
│   ├── api.py                                # Programmatic (non-GUI) interface for scripts and notebooks
│   ├── gui.py                                # Main window: spectrum and slice plots, layout (~3700 lines)
│   ├── gui_continuum.py                      # Continuum-subtraction mixin (automatic and custom modes)
│   ├── gui_gaussian.py                       # Gaussian-fitting mixin (single-line, full-cube, multi-Gaussian)
│   ├── gui_flux_map.py                       # Flux-map mixin
│   ├── gui_spectrum.py                       # Spectrum-plot mixin (tabs, normalisation, overplot)
│   ├── gui_region_selection.py               # Shared region-selection controller (main window and 2D viewer)
│   ├── gui_mixins.py                         # Shared helpers for the two-panel windows and the map dialogs
│   ├── gui_rotational_diagram.py             # Rotational-diagram analyzer window
│   ├── gui_2d.py                             # 2D FITS dual viewer (WCS, rotation, contours, superimpose, resolution matching)
│   ├── data_loader.py                        # CubeData: FITS loading and instrument detection
│   ├── gaussian_fitter.py                    # Single and multi-Gaussian line fitting; full-cube parallel fits
│   ├── flux_map_extractor.py                 # Full-cube rovibrational flux mapping for any species
│   ├── continuum_cube.py                     # Per-pixel continuum fits that write the continuum and continuum-subtracted cubes
│   ├── rotational_diagram_h2_fitter.py       # H2 rotational-diagram temperature fits (1T and 2T)
│   ├── rotational_diagram_general_fitter.py  # Rotational-diagram fits for non-H2 species
│   ├── rotational_diagram_map.py             # Pixel-by-pixel rotational-diagram maps
│   ├── rotational_diagram_map_dialogs.py     # Workers and dialogs for the rotational-diagram maps
│   ├── convolve_reproject.py                 # PSF-matching convolution and reprojection (2D viewer and rotational-diagram maps)
│   ├── import_map_dialog.py                  # Header checks for the Load Maps button
│   ├── fits_export.py                        # Shared FITS headers and writers for the map exports
│   ├── region_io.py                          # Save and load a region as sky coordinates
│   ├── visualization.py                      # SpectrumPlotter and SlicePlotter rendering classes
│   ├── region_selector.py                    # Matplotlib region-selector widgets
│   ├── scale_dialog.py                       # Colour-scale and colour-range dialog (Linear, Log, Power)
│   ├── colormap_dialog.py                    # Colormap-selection dialog
│   ├── continuum_dialog.py                   # Automatic-continuum parameter dialog
│   ├── custom_continuum.py                   # Interactive continuum anchor-point selector
│   ├── pixel_diagnostics.py                  # Pixel-level fit-inspection dialog
│   ├── flux_map_dialogs.py                   # Flux-map configuration dialog, worker, and results dialog
│   ├── gaussian_dialogs.py                   # Gaussian-map worker and results dialog
│   ├── species_dialog.py                     # Line-selection dialog ("Show Lines")
│   ├── table_drag_select.py                  # Sortable tables with drag-to-check boxes
│   ├── line_catalog.py                       # Line identification and HITRAN integration (cached in hitran_local/)
│   ├── transition_patterns.py                # Per-molecule transition patterns for HITRAN lines
│   ├── molecule_transition_filters.py        # Transition filter prompts for HITRAN lines
│   ├── unit_converter.py                     # Wavelength, frequency, velocity, and CGS unit conversion
│   ├── utils.py                              # Instrument detection and helper functions
│   ├── icon_loader.py                        # Loads the SVG button icons
│   └── icons/                                # SVG button icons (Calcite Sharp Line Icons, MIT License)
├── hitran_local/                             # Offline HITRAN starter pack
├── lst_local/                                # LST line catalog (.lst) and its ReadMe, for the Velocity map rest-line table
├── nirspec_disp/                             # JWST NIRSpec resolving-power tables
├── docs/                                     # Sphinx documentation sources
└── data/                                     # Local sample FITS files (not tracked)
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
   - A visible offset between the two images indicates a WCS header difference between the two files.
   - Open "Header" on each panel and compare the WCS keywords.

6. **"Pair" or "Batch" problems:**
   - An "Unsupported Instrument" message means that a file is not from MIRI or NIRSpec. The PSF formulas exist for those two instruments only.
   - An "Unsupported BUNIT" message means that a file is not in a flux-per-pixel unit. A surface-brightness unit such as MJy/sr is rejected.
   - The tool asks for a line wavelength when the header has no `LINEWAVE` key. Enter the wavelength in µm.
   - A "same wavelength" message is expected when both panels have the same wavelength. No action is necessary.
   - A "No Sky Overlap" warning means that the two images cover disjoint sky regions. Load images of the same field. In "Batch" mode, the warning lists the file pairs that do not share sky, so you can remove the off-field file and try again.
   - A "Missing Pixel Scale" message means that a header has no valid `CDELT` or CD matrix. The pixel scale is necessary to compute the kernel size.
   - A flux-audit ratio clearly below 1 means that part of the image falls outside the grid of the reference image. A ratio near 0 means that the two images barely overlap. A deviation of about 1% or less can come from NaN pixels at the edge of the image footprint.

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

10. **Installation problems:**
   - A "Failed building wheel for llvmlite" or "numba" message means that pip tried to build a package from source code. Run `python -m pip install --upgrade pip`. Then confirm that `python --version` shows 3.13, and run `pip install -r requirements.txt` again.
   - On an Intel Mac, pip installs older versions of `llvmlite` and `numba`, because the newest versions have no files for Intel Macs. This is expected.
   - A "No matching distribution found for PyQt5" message on Linux means that the computer has an ARM processor. PyQt5 has no files for it, so Linux on ARM is not supported.
   - On an Apple Silicon Mac with conda, run `python -c "import platform; print(platform.machine())"`. If it prints `x86_64`, conda runs under Rosetta. The program works, but a conda for Apple Silicon (for example Miniforge) is faster.

## Credits

The button icons in `cube_explorer/icons/` are from the Calcite Sharp Line Icons collection by Esri, used under the MIT License. See `cube_explorer/icons/LICENSE`.

IFU Analyzer is released under the MIT License. See `LICENSE`.
