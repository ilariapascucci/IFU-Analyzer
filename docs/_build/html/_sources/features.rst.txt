========
Features
========

A complete reference for everything IFU Analyzer can do. For task-oriented
walkthroughs, see the :doc:`user_guide/index`.

.. contents::
   :local:
   :depth: 2


Instrument Support
==================

IFU Analyzer reads 3D spectral cubes and 2D maps from a range of optical,
infrared, and radio facilities:

- **MUSE (VLT)** — optical spectroscopy
- **ALMA** — radio interferometry
- **MIRI (JWST)** — mid-infrared spectroscopy
- **NIRSpec (JWST)** — near-infrared spectroscopy

The application auto-detects the instrument from FITS headers (``INSTRUME``,
``TELESCOP``) and applies appropriate wavelength axis handling. Generic 2D
FITS images — intensity maps, moment maps, Gaussian-derived parameter maps —
are also fully supported through the dedicated 2D Dual Image Viewer.


Interactive Visualization
=========================

Spectral cube display
---------------------

- Full spectrum display with wavelength or velocity axes
- Dual slice view for side-by-side wavelength/velocity comparison
- Real-time spectrum updates when regions are selected
- Adjustable color scales: Linear, Log, Power, Square Root
- Customizable colormaps via an interactive dialog, including
  colorblind-friendly options
- Support for 2D images (intensity maps, Gaussian-derived maps)
- Copy colormaps and regions between dual displays

Region selection tools
----------------------

- Point, rectangle, circle, ellipse, and custom polygon tools
- Interactive region handles for dragging and resizing
- Live statistics updates (min, max, mean, sum, std) on the stats panel
- Copy a region from the Left display to the Right (and vice-versa)


2D FITS Dual Image Viewer
=========================

A standalone side-by-side viewer for 2D intensity maps, moment maps, and
derived FITS products. Fully WCS-aware with interactive coordinate display
and multi-image analysis tools. The complete walkthrough is in
:doc:`user_guide/2d_viewer`; below is a feature summary.

Image display
-------------

- Dual-panel layout — load independent 2D FITS files in Left and Right panels
- Full WCS grid overlay with RA/Dec tick labels at correct positions for any
  rotation angle
- Tick label format auto-scales to field of view
  (degrees → arcmin → arcsec → sub-arcsec)
- Live cursor readout: pixel coordinates, RA (hh:mm:ss), Dec (dd:mm:ss),
  flux value
- Image statistics panel: min, max, mean with flux unit from FITS header
- Adjustable color scales (Linear, Log, Sqrt, Power γ=2) per panel
- Customizable colormaps via interactive dialog (🌈 button per panel)
- FITS header viewer — click **Header** to inspect all keywords

Image rotation
--------------

- Rotate by arbitrary angles, clockwise or counter-clockwise
- WCS header is updated on every rotation — RA/Dec labels stay accurate
- Cumulative rotation tracking (rotations compose correctly)
- Save rotated image as a new FITS with updated WCS header

Contour plotting
----------------

- Background estimation via Astropy sigma-clipped stats, or user-supplied value
- Sigma levels as a comma-separated array (e.g. ``3, 5, 10``); each value N
  draws a contour at ``N × background_std`` above background median
- WCS-matched cross-image contours: generate contours on one panel and overlay
  them on the other using full WCS coordinate transformation. Works even when
  the two images have different pixel scales, array sizes, or orientations.
- Configurable contour color
- Multiple contour sets can be stacked
- Clear all contours with one click

Contour export / import
-----------------------

- **Save as .txt** — exports all contour path vertices as pixel coordinates
  + RA/Dec (degrees) with sigma level labels (human-readable)
- **Load saved contour** — load a previously saved binary FITS mask and
  overlay on either panel via WCS mapping
- **Save contour mask as FITS** — binary 0/1 mask with WCS header

WCS superimpose
---------------

- Overlay two images with different WCS grids (different pixel scale, array
  size, orientation, pointing) onto a single panel
- Reprojection uses ``numpy``/``scipy`` and ``astropy.wcs`` — no external
  reprojection package required
- Configurable overlay colormap and alpha transparency (0.1–1.0)
- Built-in peak-separation diagnostic prints the angular offset between the
  two image peaks to the terminal, confirming alignment accuracy
- Bicubic interpolation (order=3) with NaN-aware masking

Slides Convolute (two-image resolution matching)
------------------------------------------------

Match a pair of images to a common spatial resolution — the standard step
before any multi-instrument flux comparison.

- Load two FITS images of different pixel scales into Left and Right panels
- Click **Slides Convolute** to open the confirmation dialog
- The tool automatically identifies the finer image and convolves it up to
  the coarser pixel scale using a Gaussian kernel
- Kernel width: ``σ = √(σ_target² − σ_input²) / 2.3548`` (in fine-image pixels)
- Convolution uses ``astropy.convolution.convolve_fft`` with:

  - ``nan_treatment='interpolate'`` so NaN pixels (e.g. IFU footprint
    borders) are handled without propagating gaps
  - ``normalize_kernel=True`` so surface brightness is preserved per pixel
  - NaN-padding of width ``⌈4σ⌉`` before the FFT so kernel tails stay inside
    the domain
  - Final flux renormalisation: ``Σ(output_finite) == Σ(input_finite)`` to
    machine precision (~1e-7 relative error)

- After smoothing, the convolved image is reprojected onto a new pixel grid
  at the coarse pixel scale using a flux-conserving spherical-polygon
  intersection algorithm (equivalent to ``reproject_exact`` —
  self-implemented, no external ``reproject`` dependency)
- Output shape is sized to fully contain the fine image's original FOV
- **Same-resolution guard:** if both panels have the same pixel scale
  (within 0.0001 arcsec/px), Slides Convolute shows a "nothing to do"
  message instead of running
- Output FITS has two HDUs: image + ``COVERAGE`` extension (per-pixel
  overlap fraction 0–1)
- The save dialog shows a flux audit: ``Ratio out/in : 1.000000`` confirms
  exact flux conservation

Batch Convolution (multi-image resolution matching)
---------------------------------------------------

Bring any number of multi-instrument images to a single common resolution
and FOV in one step.

- Click **Batch Convolution** to open the batch dialog
- Use **+ Add FITS Files** to add two or more 2D FITS files (any mix of
  instruments, pixel scales, orientations)
- The tool:

  1. Identifies the **worst (coarsest) pixel scale** — this becomes the
     target scale
  2. Identifies the **largest-area image** to define the output orientation
     and centre
  3. Builds a single common output WCS at the worst scale, sized to cover
     the **union of all input footprints** — no image loses sky coverage
  4. For each input, applies the appropriate Gaussian kernel (σ from
     quadrature) and reprojects onto the common grid
  5. Images already at the worst scale receive ``σ = 0`` (no smoothing;
     only a reproject if needed)

- All outputs are packed into a ``.zip`` file — one FITS per input, all
  on the same WCS grid, ready for pixel-by-pixel comparison
- Each output FITS includes the ``COVERAGE`` extension
- The completion dialog shows a flux audit table (``ratio = 1.000000``
  per file)


Spectral Analysis
=================

Gaussian line fitting
---------------------

- Single-pixel interactive fitting with visual overlay
- Full-cube parallel fitting with parameter maps
- Multi-Gaussian component fitting for complex line profiles
- Automated baseline subtraction
- Quality metrics: R², χ², FWHM, area under curve
- Background processing keeps the GUI responsive
- Pixel-level diagnostics with fit visualization

Parameter maps
--------------

A complete set of derived 2D maps is produced from a full-cube fit:

- Integrated area (intensity map)
- Peak amplitude
- Line center (kinematics)
- Line width (σ and FWHM)
- Fit quality (χ²)
- Success mask

Residuals analysis
------------------

- Normalized residuals cube export (3D FITS)
- Wavelength-dependent residuals visualization
- Quality assessment of individual fits

CO rovibrational flux mapping
-----------------------------

Full-cube CO flux extraction:

- Simultaneous multi-line fitting of CO rovibrational transitions
- Contaminating line exclusion and masking
- Parallel processing across all spatial pixels
- Optional continuum subtraction before line detection
- Automatic flux integration and SNR thresholding
- Background processing with progress tracking
- Complete flux maps and statistics extraction

Continuum fitting
-----------------

**Automatic mode:** Median filter → peak detection → iterative sigma
clipping → cubic spline fit.

- Tunable parameters: window size, sigma threshold, peak exclusion width,
  iterations, smoothing factor
- Fitted continuum displayed in orange over the original spectrum before
  any subtraction
- One-click global subtraction

**Custom interactive mode:**

- Click to place continuum anchor points directly on the spectrum
- Points snap to the nearest data value
- Cubic spline interpolates a smooth baseline through all points
- Delete individual points with Delete Point mode
- **Preview before subtract:** OK shows the fitted continuum in orange
  over the original spectrum; a confirmation dialog asks before applying
- Full spectrum displayed after subtraction with correct y-axis scaling
- One-click **Save Results** export: wavelength, original spectrum,
  fitted continuum, and continuum-subtracted spectrum all written to a
  single ASCII file


Data Processing
===============

- Coordinate conversion (pixel ↔ RA/DEC)
- Region-based spectral extraction
- Statistical analysis of selected regions
- Spectrum normalization with baseline points
- Continuum baseline subtraction (interactive and automatic)
- Spectral line identification (HITRAN database and custom catalogs)
- Multi-line automatic detection with SNR thresholding
- Doppler shift correction (multiplicative wavelength shift)
- Wavelength-to-frequency conversion with CGS unit support
- Full astropy ``Quantity`` object support for unit handling
- Memory-efficient handling of large data cubes
- Dynamic wavelength/velocity range windowing


Export Capabilities
===================

- **Continuum Results Export (ASCII)** — single ``Save Results`` button
  writes wavelength, original spectrum, fitted continuum, and
  continuum-subtracted spectrum in one tab-delimited file
- **Residuals FITS Export** — export 3D normalized residuals cube with
  full header metadata
- **CO Flux Maps** — export full-cube fitting results with integrated
  flux estimates
- **Contour Coordinates (.txt)** — export contour path vertices with
  pixel + RA/Dec positions and sigma level labels
- **Contour Mask (FITS)** — export binary region mask with WCS header
- **Rotated Image (FITS)** — save rotated image with updated WCS header
- **Convolved Image (FITS)** — Slides Convolute output, 2-HDU layout
  (image + COVERAGE)
- **Batch Convolution ZIP** — archive of all resolution-matched FITS
  files on a common WCS grid
- Automatic instrument detection and header preservation
- Unit-aware exports with CGS calibration support


Performance
===========

- **Optimized Gaussian fitting:** adaptive parallelization with 8 workers
  (threads or processes)
- **Reduced convergence iterations:** ``maxfev=5000`` for ~2–5× speedup
- **Background processing:** the GUI stays responsive during computations
- **Memory efficiency:** memory-mapped access for large cubes
- **Typical performance:** ~10–30 seconds for full-cube Gaussian mapping


Known Limitations
=================

- Residuals are normalized by noise std; some analyses may require raw
  residuals
- 2D FITS images display as single-slice cubes in the main viewer — use
  the dedicated 2D Viewer for full functionality
- CO flux mapping requires a well-defined line catalog with accurate
  wavelengths
- ``ProcessPoolExecutor`` may have startup overhead on very small cubes
- WCS superimpose uses bicubic interpolation with NaN masking; accuracy
  is sub-pixel for typical FITS WCS grids
- Batch Convolution polygon-overlap reprojection is pure Python;
  performance scales with the number of output pixels — very large images
  (>500×500) may be slow
