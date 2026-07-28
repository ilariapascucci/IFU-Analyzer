===========================
Pixel-by-Pixel Cube Fitting
===========================

Fit a Gaussian to the spectrum of **every spatial pixel** in the cube, producing 2D parameter
maps that show how the line properties vary across the field.

This is the full-cube counterpart to :doc:`gaussian_fitting_1d`. The per-pixel fit is the same
algorithm, but the workflow, the required inputs and the outputs are different: a noise region
becomes mandatory, non-detections are turned into upper limits rather than discarded, and the
result is a set of maps rather than a single number per parameter.

.. note::

   **Fit the 1D spectrum first.** Run :doc:`gaussian_fitting_1d` on a representative pixel or
   region before starting a cube fit. That is how you find a sensible wavelength window and
   confirm the line is actually there. A full-cube run over a badly chosen window takes
   minutes and produces maps of noise.

Setting Up the Fit
==================

The cube fit reuses the same toggle buttons as the 1D fit, but requires a different subset
of them.

Wavelength window (required)
----------------------------

Set **Left** and **Right** to bracket the line. These become the fit window for every pixel.

**Peak is not required.** Unlike the 1D fit, the cube fit does not use your λ_peak click.
Instead it fits the summed spectrum of the whole cube once to derive initial guesses, and
if a HITRAN line list is loaded it will use the nearest catalog line in the window as the
center seed. This "global 1D fit" is reported back to you in the results window.

Noise region (required)
-----------------------

Set **Noise L** and **Noise R** over a line-free stretch of spectrum. **The cube fit refuses
to start without them** — you will get a "Noise Region Required" warning.

Unlike the 1D fit — where a noise region is only demanded for very narrow windows, and
does not affect the result even then (see :ref:`noise-region-1d`) — here it is always
required *and* it directly shapes the output.

The noise region is not optional decoration here. The standard deviation measured in it sets
the detection threshold: a pixel counts as a detection when its fitted amplitude exceeds
**3σ** of the noise *and* the fit improves on the residuals. Everything else is recorded as
an upper limit. Choose a region close in wavelength to the line, and genuinely free of
features — a noise region that clips a line wing inflates σ and silently demotes real
detections to upper limits.

Running the Fit
===============

1. Click **Fit Gaussian Map**
2. Confirm at the "Continue with map fitting?" prompt
3. The fit runs on 8 worker threads in the background

**Watching progress.** The dialog that appears is an *indeterminate* busy indicator — it
does not show a percentage, and the "Hide" button dismisses it without cancelling the run.
Actual progress is printed to the **console** you launched the application from:

.. code-block:: text

    Fitting Gaussians to 122500 pixels in wavelength range [6.9000, 6.9500]
    Using 8 worker threads (122,500 pixels)
      Progress: 12250/122500 pixels (10.0%)
      Progress: 24500/122500 pixels (20.0%)
      ...
    Fitting complete: 98211/122500 successful fits (80.2%)
      Detections (>3.0 sigma): 41022 (33.5%)
      Upper limits: 57189 (46.7%)

If the window appears frozen, check the console — that is where liveness shows up.

**Results are cached.** Re-running with the same Left/Right *and* the same Noise L/Noise R
reuses the previous fit instead of recomputing it, and tells you so. Change any of the four
boundaries to force a fresh run.

Reading the Results
===================

The results window opens automatically when the fit completes, headed by a summary:

- **Successfully fit N / M pixels** — how many converged at all
- **Detections | Upper limits** — the 3σ split described above
- **Global 1D fit** — the summed-spectrum fit used to seed the per-pixel initial guesses,
  and where its center came from

Below that, a dropdown selects which map to display, with colormap and scale controls
(Linear, Log, or Percentile 2–98%).

Output Maps
===========

Available in the dropdown:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Map
     - Contents
   * - **Area (AUC)**
     - Integrated line flux per pixel, ``A × σ × √(2π)``. In **erg/s/cm²** for MIRI,
       NIRSpec and MUSE (see :ref:`the unit table on the 1D page <area-units-1d>`);
       otherwise in the cube's native flux×wavelength units.
   * - **Amplitude**
     - Peak height above the baseline, converted to CGS where the header allows it
   * - **Center**
     - Fitted line center wavelength — this is your velocity map, once converted
   * - **Sigma**
     - Gaussian σ, in the spectral axis unit
   * - **FWHM**
     - ``2.355 × σ``
   * - **chi2 (fit quality)**
     - Reduced chi-squared per pixel. ``χ²_red ≈ 1`` is a good fit, ``>> 1`` is poor,
       ``<< 1`` means overfitting. This is a real χ² — it exists here, and not in the 1D
       fit, precisely because the mandatory noise region supplies the σ it needs.
   * - **Success Mask**
     - Binary map of which pixels converged
   * - **Residuals**
     - Data minus fitted model. Display only — see the export note below.

Upper limits
------------

For pixels that fail the detection threshold the area is **not** set to NaN. It is replaced
with a 1σ upper limit, ``std_residuals × σ_guess × √(2π)``, and the pixel is flagged in the
``is_upper_limit`` mask.

This matters downstream: :doc:`rotational_diagram` detects censored points and switches from
ordinary least squares to Bayesian censored regression rather than discarding them. Failed
fits — as opposed to non-detections — do store NaN.

Exporting Maps as FITS
======================

With a map selected in the dropdown, click **Save as FITS File**. Each map is written as a
single-slice FITS file you can reopen in the 2D viewer (:doc:`2d_viewer`) or in external
tools.

.. warning::

   **Residuals cannot be exported from this dialog.** It appears in the display dropdown but
   not in the export list, so choosing it and clicking save produces a "Cannot export"
   warning. To get residuals to disk, use ``export_residual_cube`` from the API (below),
   which writes the full 3D residual cube.

Using the API
=============

**Fitting the cube**

.. code-block:: python

    from cube_explorer.api import load_cube, create_gaussian_map

    cube = load_cube("data.fits")

    maps = create_gaussian_map(
        cube,
        lam_min=6.90,
        lam_max=6.95,
        noise_min=7.00,        # Line-free region, as in the GUI
        noise_max=7.05,
        fac_above_stdres=3.0,  # Detection threshold in sigma
        n_jobs=8,
    )

    print(f"Converged:    {maps['success'].sum()}")
    print(f"Detections:   {maps['is_detection'].sum()}")
    print(f"Upper limits: {maps['is_upper_limit'].sum()}")

    area_map = maps['area']          # 2D array, shape (n_y, n_x)

.. important::

   ``fac_above_stdres`` defaults to **2.0** in the API but the GUI passes **3.0**. If you are
   trying to reproduce a GUI result in a script, set it to 3.0 explicitly — otherwise you
   will get more detections than the GUI reported for the same cube.

   ``n_jobs`` defaults to auto-detected CPU count in the API; the GUI hardcodes 8.

**Returned keys**

.. code-block:: python

    maps['amplitude']         # Peak amplitude, CGS where available
    maps['amp_originalunits'] # Peak amplitude in the cube's own units
    maps['center']            # Line center wavelength
    maps['sigma']             # Gaussian sigma
    maps['fwhm']              # 2.355 * sigma
    maps['area']              # Integrated flux (or 1-sigma upper limit)
    maps['r_squared']         # SEE WARNING BELOW - this is reduced chi-squared
    maps['success']           # Boolean: did the fit converge?
    maps['is_detection']      # Boolean: passed the sigma threshold
    maps['is_upper_limit']    # Boolean: area is a 1-sigma upper limit
    maps['residuals_cube']    # 3D residuals cached during fitting
    maps['fac_above_stdres']  # The threshold that was used
    maps['wavelength_unit']   # Spectral axis unit string
    maps['global_fit']        # The summed-spectrum fit used for initial guesses

.. danger::

   **The key** ``'r_squared'`` **does not contain R².** It holds the reduced chi-squared map
   — the same values the GUI correctly labels "chi2 (fit quality)". The key name is a
   historical misnomer.

   Interpret it as χ²_red: **near 1 is good, large is bad.** Code that treats it like R²
   and keeps pixels above some threshold will select the *worst* fits in the cube.

**Exporting a residual cube**

Fits and subtracts a Gaussian at every pixel, writing the normalized residuals to a 3D FITS
file:

.. code-block:: python

    from cube_explorer.api import load_cube, export_residual_cube

    cube = load_cube("data.fits")

    export_residual_cube(
        cube,
        out_path="residual.fits",
        lam_min=6.90,
        lam_max=6.95,
    )

    # Or keep the array in memory as well
    path, residuals = export_residual_cube(
        cube, out_path="residual.fits",
        lam_min=6.90, lam_max=6.95, return_cube=True,
    )

Constraints and Tips
====================

1. **Left < Right**, and the noise region must be set — the run will not start otherwise
2. Keep the noise region **close in wavelength** to the fit window. Detector noise varies
   with wavelength, so a distant noise estimate misstates the detection threshold across
   the whole map
3. Keep the fit window **tight**. Every extra wavelength sample is fitted for every pixel,
   so a window twice as wide costs roughly twice the runtime and degrades χ²_red
4. Always check the **Success Mask** and the detection/upper-limit counts before
   interpreting a parameter map. A beautiful-looking area map built from 15% converged
   pixels is mostly interpolated nothing
5. Use the **chi2** map to find where the model breaks down — blended lines and
   P-Cygni-like profiles show up as coherent high-χ² structure, not as scattered noise

Next Steps
==========

- :doc:`rotational_diagram` — turn per-pixel line fluxes into temperatures, handling the
  upper limits correctly
- :doc:`co_flux_extraction` — simultaneous multi-line fitting for CO rovibrational bands
- :doc:`exporting_data` — other export paths
- :doc:`2d_viewer` — open the exported parameter maps with WCS support
