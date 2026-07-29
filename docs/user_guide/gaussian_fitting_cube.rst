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

   **Fit the 1D spectrum first — the application will not make you.** Nothing stops you
   clicking **Fit Gaussian Map** without ever having run **Fit Gaussian**, and no warning
   appears if you do. Run :doc:`gaussian_fitting_1d` on the summed spectrum anyway: it is how
   you find a sensible wavelength window and confirm the line is really there. A full-cube run
   over a badly chosen window costs minutes and returns maps of noise.

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

The **Peak** marker stays on the plot from the 1D fit, so the panel shows five markers while
a cube fit reads only four — **Left**, **Right**, **Noise L** and **Noise R**. Moving **Peak**
changes nothing about a map.

Noise region (required)
-----------------------

Set **Noise L** and **Noise R** over a line-free stretch of spectrum. **The cube fit refuses
to start without them** — you will get a "Noise Region Required" warning.

Unlike the 1D fit — where a noise region is only demanded for very narrow windows, and
does not affect the result even then (see :ref:`noise-region-1d`) — here it is always
required *and* it directly shapes the output.

The standard deviation measured in it feeds the detection test and the size of the upper
limits. Choose a region close in wavelength to the line, and genuinely free of features — a
noise region that clips a line wing inflates σ and moves the whole detection boundary.

.. _detection-caveat:

.. danger::

   **"Detections (>3.0 sigma)" does not mean 3σ.** The detection test has two pathways and a
   pixel passes if *either* succeeds:

   * **A** — amplitude ≥ ``fac_above_stdres × σ`` (3σ) **and** the fit reduces the scatter
   * **B** — ``0.5 ≤ χ²_red ≤ 2.5`` **and** amplitude > **0.5σ**

   Because χ²_red is bounded to a narrow range by construction
   (:ref:`chi2-caveat`), the first half of **B** is satisfied by essentially every converged
   fit, so **B collapses to a 0.5σ threshold**.

   On the worked example above, of 960 "detections":

   * 156 (16%) cleared 3σ
   * 804 (84%) did not — median SNR **1.5**, and 274 of them below **1σ**
   * 0 passed via **A** alone, so pathway **A** never decides anything on its own

   The upper limits are consequently *brighter* than most detections — median upper limit
   is 1.7× the median detected area, and 530 of 553 upper limits exceed it. A pixel can be
   called a detection at a flux below the limit you would have quoted had it failed.

   Treat the mask as "a positive bump was fitted here", not as a significance cut, and
   apply your own SNR threshold before using an area map for science.

Running the Fit
===============

1. Click **Fit Gaussian Map**
2. Confirm at the "Continue with map fitting?" prompt
3. The fit runs on 8 worker threads in the background

**Watching progress.** A progress dialog reports the percentage of pixels completed, updated
once per cube row.

.. important::

   The **Hide** button dismisses the dialog *without cancelling the run*. The fit continues in
   the background and the results window still opens when it finishes. Closing the whole
   application *does* cancel it — the in-flight rows finish (typically a few seconds) and
   then the process exits, rather than continuing to compute and print progress to a console
   nothing is reading anymore.

The same progress is echoed to the console you launched the application from, together with
the final tallies:

.. code-block:: text

    Fitting Gaussians to 2009 pixels in wavelength range [7.4923, 7.5174]
    Using 8 worker threads (2,009 pixels)
      Progress: 245/2009 pixels (12.2%)
      Progress: 490/2009 pixels (24.4%)
      ...
    Fitting complete: 957/2009 successful fits (47.6%)
      Detections (>3.0 sigma): 957 (47.6%)
      Upper limits: 556 (27.7%)

**Runtime.** That example — a 49 × 41 spaxel MIRI cube, 2,009 pixels, a 0.025 µm window —
takes about **100 seconds** single-threaded, and **not less** on the GUI's 8 worker threads
(120s measured). Cost scales roughly with pixels × samples in the window, so a wider window
or a larger detector is proportionally slower.

.. warning::

   **The 8 worker threads do not currently speed this up, and measured slightly slower than
   one thread on this machine.** Per-pixel noise estimation (see the chi2 caveat below)
   runs scikit-learn's ``RANSACRegressor``, which is Python-and-NumPy-heavy rather than a
   single release-the-GIL C call, and NumPy's own BLAS backend may already be
   multithreaded — running 8 Python threads on top of that can oversubscribe the machine's
   cores rather than using them more efficiently. This has not been root-caused; it is
   reported here because the docs should not claim a speedup that a real run does not show.

**Results are cached.** Re-running with the same Left/Right *and* the same Noise L/Noise R
reuses the previous fit instead of recomputing it, and tells you so. Change any of the four
boundaries to force a fresh run.

Reading the Results
===================

The results window opens automatically when the fit completes, headed by a summary:

- **Successfully fit N / M pixels** — how many converged
- **Detections | Upper limits** — the 3σ split described above. The detection count always
  equals the "successfully fit" count; see the note under `Output Maps`_
- **Global 1D fit** — the summed-spectrum fit used to seed the per-pixel initial guesses,
  and where its center came from, e.g.

  .. code-block:: text

      Global 1D fit: A=1.8570e+05, mu=7.5039, sigma=0.0042 (center from total-spectrum)

  ``center from total-spectrum`` means the seed came from fitting the summed cube. With a
  HITRAN list loaded it can instead come from the nearest catalog line in the window.

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
     - Peak height above the baseline. A flux *density*, so where the conversion applies it
       is **erg/s/cm² per wavelength unit** (erg/s/cm²/µm on a µm axis) — not per steradian
   * - **Center**
     - Fitted line center wavelength — this is your velocity map, once converted
   * - **Sigma**
     - Gaussian σ, in the spectral axis unit
   * - **FWHM**
     - ``2.355 × σ``
   * - **chi2 (fit quality)**
     - Labelled reduced χ², but read :ref:`chi2-caveat` before using it — it does not use
       the noise region and cannot fall below ``n/(n-3)``.
   * - **Success Mask**
     - Binary map of converged pixels
   * - **Residuals**
     - Data minus fitted model. Display only — see the export note below.

.. note::

   ``success`` and ``is_detection`` are the **same array**. The fitter sets both on the same
   line and nowhere else, so a pixel that converged is by definition a detection and the two
   counts in the summary are always identical. Do not read the pair as "converged, of which
   some were detected" — there is no such distinction in the output.

.. _chi2-caveat:

What the chi2 map actually measures
-----------------------------------

.. danger::

   **It is not a reduced χ² and it does not use the noise region.** The residuals are
   normalized by *their own* standard deviation rather than by the measured noise:

   .. code-block:: text

       chi2_red = sum(r^2) / (dof * std(r)^2)

   Since ``sum(r²) = n·std(r)² + n·mean(r)²``, that algebra collapses to

   .. code-block:: text

       chi2_red = (n / (n-3)) * (1 + mean(r)^2 / std(r)^2)

   which depends only on the **mean offset of the residuals relative to their own
   scatter** — never on the actual noise level. Consequences:

   * The map has a hard floor at ``n/(n-3)``. For the 31-sample window above that is
     **1.10714**, which is exactly the observed minimum, and 153 pixels sit on it.
   * Values **below 1 are impossible**, so "≈ 1 is a good fit" is unreachable by
     construction — 1.107 is the floor, not the target.
   * A badly wrong fit whose residuals happen to be centred scores the same as a perfect
     one. Width errors, which produce symmetric structured residuals, are invisible to it.
   * Doubling the noise changes nothing.

   Use it only as a weak flag for *asymmetric* residuals. For real fit quality, inspect the
   **Residuals** map, which *is* normalized by the noise region and should be ~N(0,1).

Bounds and railed pixels
------------------------

Per-pixel bounds are derived from the global 1D fit, not from the window you selected:

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Parameter
     - Bound
     - Example values
   * - Center
     - ``μ_global ± 3 × (window / n_samples)``
     - [7.501607, 7.506253]
   * - Sigma
     - ``[0.25 × σ_global, 4 × σ_global]``
     - [0.001046, 0.016731]
   * - Amplitude
     - ``[0.5 × local peak, 2 × local peak]``
     - cannot be small, even for pure noise

.. warning::

   Noise-dominated pixels rail against these bounds, and a railed value looks like a
   measurement. In the worked example, of 960 detections:

   * **60% of centers** sit exactly on a center bound (24% low, 36% high)
   * **13% of sigmas** sit exactly on a sigma bound

   The map minima and maxima you see in the statistics box are therefore often the *bounds
   themselves*, not the data — compare the Center map's range to the table above.

   This is why the Center map shows a large flat region away from the source: those pixels
   are pinned at a bound, not measured. **The Center map is a velocity map only where the
   line is genuinely detected.** Mask it by SNR before interpreting it.

   Note also that the amplitude lower bound is half the local peak, so a pure-noise pixel is
   *required* to return an amplitude of roughly its own largest positive noise excursion.
   The fitter has no way to express "there is no line here" other than the upper-limit path.

Where every pixel goes
----------------------

Each pixel lands in exactly one of three states, and they account for the whole map. From
the MIRI run above:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - State
     - Example
     - What is stored
   * - **Detection**
     - 957 (47.6%)
     - Converged and passed the detection test — which is **not** a 3σ cut, see
       :ref:`the detection caveat <detection-caveat>`. Every parameter populated.
   * - **Upper limit**
     - 556 (27.7%)
     - ``area`` holds an upper limit; amplitude, center, sigma, FWHM and χ²_red are **NaN**.
   * - **No data**
     - 496 (24.7%)
     - The spectrum was entirely NaN — outside the field of view. Everything is NaN and
       *no* flag is set, so these pixels appear in neither mask.

957 + 556 + 496 = 2009. If your three numbers do not add up to the pixel count, something
is wrong with the window or the cube.

.. note::

   **The counts are not reproducible to the pixel.** Four runs of this identical fit gave
   957, 958, 958 and 960 detections. Every map's minimum and maximum stayed identical, so
   the maps themselves are stable — it is only pixels sitting on the decision boundary that
   move, and given how many sit just above threshold
   (:ref:`the detection caveat <detection-caveat>`) a handful flipping is expected. The per-pixel code contains no random number generator; the most
   likely source is last-bit variation from threaded linear algebra inside ``curve_fit``.

   Never quote a detection count to better than "about 960".

Upper limits
------------

For pixels that fail the detection threshold the area is **not** set to NaN. It is replaced
with an upper limit at the detection threshold — ``fac_above_stdres × σ_noise × σ_guess ×
√(2π)``, so **3σ** with the GUI's settings, not 1σ — and the pixel is flagged in the
``is_upper_limit`` mask.

This matters downstream: :doc:`rotational_diagram` detects censored points and switches from
ordinary least squares to Bayesian censored regression rather than discarding them.

.. warning::

   ``is_upper_limit`` is set by **several** paths, not only "amplitude below threshold": a
   window with fewer than 5 finite samples, a failed noise estimate, and an exception inside
   the fit all land here too. In the last case, if the failure happened before an initial σ
   guess existed, ``area`` is **NaN while** ``is_upper_limit`` **is True**.

   Any code consuming upper limits — censored regression especially — must therefore check
   ``np.isfinite(area)`` as well as the flag. A NaN silently entering a censored fit
   poisons the whole posterior.

Exporting Maps as FITS
======================

With a map selected in the dropdown, click **Save as FITS File**. Each map is written as a
single-slice FITS file you can reopen in the 2D viewer (:doc:`2d_viewer`) or in external
tools.

The header of each exported map carries the fit's provenance: ``LAMMIN``/``LAMMAX`` for the
window, ``NSUCC``/``NTOTAL`` for the detection count, ``PARAM`` for which map it is, and
``BUNIT`` set to the same unit shown on the colorbar.

.. note::

   **Residuals export as a 3D cube, not a 2D map.** Selecting **Residuals** and clicking
   **Save as FITS File** writes the full normalized residual cube — shape
   ``(n_wave, n_y, n_x)`` over the fit window, with ``BUNIT = 'Normalized residuals'`` —
   even though the dialog only ever displays the middle wavelength slice. The API
   equivalent is ``export_residual_cube`` (below).

Using the API
=============

**Fitting the cube**

.. code-block:: python

    from cube_explorer.api import load_cube, create_gaussian_map

    cube = load_cube("data.fits")

    maps = create_gaussian_map(
        cube,
        lam_min=7.492291,        # Left boundary
        lam_max=7.517421,        # Right boundary
        noise_min=7.529647,      # Line-free region, as in the GUI
        noise_max=7.564965,
        fac_above_stdres=3.0,    # Detection threshold in sigma
        n_jobs=8,
    )

    print(f"Detections:   {maps['is_detection'].sum()}")
    print(f"Upper limits: {maps['is_upper_limit'].sum()}")
    print(f"No data:      {maps['area'].size - maps['is_detection'].sum() - maps['is_upper_limit'].sum()}")

    area_map = maps['area']          # 2D array, shape (n_y, n_x)

.. important::

   ``fac_above_stdres`` defaults to **2.0** in the API but the GUI passes **3.0**. If you are
   trying to reproduce a GUI result in a script, set it to 3.0 explicitly — otherwise you
   will get more detections than the GUI reported for the same cube.

   ``n_jobs`` defaults to auto-detected CPU count in the API; the GUI hardcodes 8.

**Returned keys**

.. code-block:: python

    maps['amplitude']         # Peak amplitude, erg/s/cm2 PER WAVELENGTH where convertible
    maps['amp_originalunits'] # Peak amplitude in the cube's own units
    maps['center']            # Line center wavelength
    maps['sigma']             # Gaussian sigma
    maps['fwhm']              # 2.355 * sigma
    maps['area']              # Integrated flux, or an upper limit - check is_upper_limit
    maps['r_squared']         # SEE WARNING BELOW - this is reduced chi-squared
    maps['success']           # Boolean: converged. Identical to is_detection.
    maps['is_detection']      # Boolean: same array as success
    maps['is_upper_limit']    # Boolean: area is an upper limit (may still be NaN)
    maps['residuals_cube']    # 3D residuals cached during fitting
    maps['fac_above_stdres']  # The threshold that was used
    maps['wavelength_unit']   # Spectral axis unit string
    maps['global_fit']        # The summed-spectrum fit used for initial guesses

.. note::

   ``amplitude`` is a flux *density* and keeps a per-wavelength — erg/s/cm²/µm for a µm
   axis, erg/s/cm²/Å for an Å axis. ``area`` has integrated the spectral axis away and is a
   plain erg/s/cm². The relation is ``area = amplitude × sigma × √(2π)``, which only
   balances dimensionally with amplitude read this way. Neither carries a per-steradian: the
   pixel solid angle from ``CDELT1``/``CDELT2`` is divided out during conversion.

   Where the cube's units cannot reach CGS at all — ALMA's ``Jy/beam``, or any cube whose
   header lacks the pixel scale — both maps stay in the cube's native units instead.

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
        lam_min=7.492291,
        lam_max=7.517421,
    )

    # Or keep the array in memory as well
    path, residuals = export_residual_cube(
        cube, out_path="residual.fits",
        lam_min=7.492291, lam_max=7.517421, return_cube=True,
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
