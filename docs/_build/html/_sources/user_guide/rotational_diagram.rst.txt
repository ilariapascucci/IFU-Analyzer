==================
Rotational Diagram
==================

Fit rotational excitation temperatures from a set of line fluxes (H2 or CO), using a
single-temperature (1T) or two-temperature (2T) model, with an optional grid search over
the ortho-to-para ratio (OPR) for H2.

Using the Rotational Diagram Tool
==================================

**Accessing the Tool**

1. Click the **Rot Diag** button in the main toolbar (**Rotational Diagram** group)

**Step 1: Get Line Data**

Two ways to populate the line table:

- **Fetch from HITRAN** — set the wavelength range (μm), molecule (**H2** or **CO**) and
  isotopologue, and either check **All transitions** or specify ``v_upper``/``v_lower``,
  then click **Fetch from HITRAN**. Populates the table with every matching line's
  wavelength, A coefficient, E\ :sub:`upper` and statistical weight — the flux columns are
  left blank for you to fill in.
- **Import from CSV** — load a file with ``species``, ``wavelength`` and ``flux`` columns
  (a column with "error" in its name is optional), e.g. ``data/H2_lines.csv``. Each row is
  matched to the nearest HITRAN line by wavelength to fill in A/E\ :sub:`upper`/g\ :sub:`upper`
  automatically, and its flux (and error, if present) populate the table directly.

A **Calculate Fluxes** button also appears once HITRAN lines are fetched, for measuring
fluxes directly from a loaded cube instead of entering them by hand. Every line covered by
the same FITS file is fit simultaneously via ``curve_fit`` (the same chunked engine as
:doc:`co_flux_extraction`'s **Multi-Gaussian Fit**, not a separate implementation), and the
results, including flux error, upper-limit status, and fit quality, are written straight
into the table below.

**Step 2: Select Lines and Enter Flux Data**

The table shows every fetched/imported line, in three groups of columns:

- **☑**, **λ (μm)**, **Transition**, **A_ul (s⁻¹)**, **E_u (K)**, **g_u** — from HITRAN, not
  editable. ☑ controls whether the line is included in the fit.
- **Flux**, **Flux Error**, **Error %**, **Upper Limit?** — enter by hand, or let
  **Calculate Fluxes** fill them in. A flux and either an absolute error or a percentage
  auto-calculate from each other as you type (toggle which one drives the calculation with
  the **Error input** dropdown above the table). Check **Upper Limit?** instead of entering
  an error for a non-detection.
- **UL Flux (3σ)**, **Calculated?**, **Source**, **Fit R²**, **Fit**, **H2O Nearby** — only
  filled in by **Calculate Fluxes**:

  - **UL Flux (3σ)**: the 3σ upper-limit flux for this line, computed the same way whether
    the line is a detection or not:

    .. math::

       F_{UL} = 3 \sigma_{noise} \cdot \sigma \cdot \sqrt{2\pi}

    the same area formula every fit in this app uses (:math:`A \sigma \sqrt{2\pi}`), but
    with :math:`3 \sigma_{noise}` (three times the noise level) standing in for the
    fitted amplitude :math:`A`, and :math:`\sigma` the line's real instrumental width
  - **Calculated?**: checked if this row's flux came from **Calculate Fluxes** rather than
    manual entry
  - **Source**: which instrument and FITS file the flux was measured from
  - **Fit R²**: quality of the underlying Gaussian fit, flagged red below 0.5 (the same
    threshold used elsewhere in the app)
  - **Fit**: click to open a plot of this line's own fit (spectrum, baseline, fitted curve,
    flux, amplitude, R², FWHM), the same style as **Multi-Gaussian Fit**'s display
  - **H2O Nearby**: informational only, warns if real HITRAN H2O lines fall close to this
    line, never affects the fit itself

.. important::

   Detections need a flux error. A detection with no error entered is silently dropped
   from the fit rather than assigned a made-up default (you'll get a popup listing which
   lines were excluded, if any). Upper limits are the exception: they don't need an error,
   just the checkbox.

   If *any* line is marked as an upper limit, the fit automatically switches from
   ``curve_fit`` to a Bayesian censored regression (PyMC) that treats upper limits as
   one-sided constraints rather than dropping them.

**How Calculate Fluxes measures a line**

Flux error combines two terms in quadrature:

- **Statistical** — from the error-propagation formula for
  :math:`\text{area} = A \sigma \sqrt{2\pi}`, assuming amplitude and sigma are
  uncorrelated:

  .. math::

     \sigma_{area} = \sqrt{2\pi \left(\sigma^2 \sigma_A^2 + A^2 \sigma_\sigma^2\right)}

  using ``curve_fit``'s own covariance for :math:`\sigma_A` and :math:`\sigma_\sigma`.
- **Systematic** — a fixed 5% of the flux, representing flux-calibration and
  continuum-estimation uncertainty that doesn't shrink just because a particular fit was
  well-constrained.

A line is classified as an upper limit if its fitted flux doesn't clear the 3σ threshold
above. **Calculate Fluxes** auto-checks **Upper Limit?** for those lines, though you can
still override the checkbox by hand afterward.

**Step 3: Set Physical Parameters and Fit Options**

- **R (AU)** — emitting region radius. Check **Unknown R** to skip it entirely; N\ :sub:`u`
  is then reported as a total molecule count rather than a column density.
- **Distance (pc)**
- **Fit** — **1T** or **2T**
- **Fit OPR** — grid-search the ortho-to-para ratio to minimize χ² instead of assuming the
  fixed equilibrium value **OPR = 3**. Only affects H2 S-branch lines. Click **▶ Advanced**
  to override the search grid (default: OPR from 0.1 to 3.0 in 400 steps).

**Step 4: Run Fit → Results**

Click **Run Fit**. The results window shows both text and a plot:

- **Detections** — black points with error bars
- **Upper limits** — red downward-pointing arrows (no error bar, since the constraint is
  one-sided)
- **1T mode**: a single dashed line through the points
- **2T mode**: dotted **Warm**/**Hot** lines for each component, plus a solid red
  **2-component model** line for their sum
- Y-axis is log\ :sub:`10`\ (N\ :sub:`u` / g\ :sub:`u`), not natural log, despite the fit
  internally working in ln space

If **Fit OPR** was checked, the text results also report the best-fit OPR and its 1σ range.

When any line is an upper limit, the censored (PyMC) fit also reports its own convergence
diagnostics: ``divergence(s)`` (ideally 0) and ``max r_hat`` (should be at most 1.01), so a
fit that struggled to sample doesn't look identical to one that converged cleanly.

Worked Examples
================

The dataset here is the standard H2 v=0→0 S(1)–S(7) pure-rotational line set (5.5–17 µm),
imported from ``data/H2_lines.csv``.

.. figure:: ../_static/1TFit.png
   :alt: 1T rotational diagram fit on seven H2 lines, showing poor agreement at high E_u
   :width: 100%

   1T fit on all 7 lines: T = 536.6 ± 4.8 K. The single line can't track the curvature,
   since the highest-E\ :sub:`u` points sit well below it, which is exactly the signature
   that motivates using 2T here instead.

.. figure:: ../_static/2TFit.png
   :alt: 2T rotational diagram fit with OPR fixed at 3, seven H2 lines
   :width: 100%

   2T fit (OPR fixed at 3) on the *same* 7 lines: Warm (T = 273.3 ± 9.6 K) and Hot
   (T = 863.2 ± 34.1 K) components, tracking the curvature the 1T fit missed.

.. figure:: ../_static/2TOPRFit.png
   :alt: 2T rotational diagram fit with OPR fitted, five highest lines, larger error bars
   :width: 100%

   2T fit with **Fit OPR** enabled, on the 5 highest-E\ :sub:`u` lines only and larger
   flux errors — a messier, more realistic case. The OPR search converged to 1.859 ± 1.072;
   with only 5 points feeding a 4-parameter 2T model plus OPR, T1's own uncertainty balloons
   (202.9 ± 937.3 K) even though the fit still runs.

.. figure:: ../_static/1TULFit.png
   :alt: 1T rotational diagram fit with two upper limits, shown as red downward arrows
   :width: 100%

   1T fit with two of the seven lines marked as upper limits (red arrows): T = 410.3 ± 8.6 K.
   The results text reports ``[PyMC: 0 divergence(s), max r_hat = 1.000]`` since marking any
   line as an upper limit switches this to the censored PyMC fit automatically.

.. figure:: ../_static/2TULFit.png
   :alt: 2T rotational diagram fit with upper limits, shown as red downward arrows
   :width: 100%

   2T fit on the same upper-limit dataset: Warm (T = 350.7 ± 11.2 K) and Hot
   (T = 2333.7 ± 470.1 K) components, again via the censored PyMC fit
   (``max r_hat = 1.010``).

Using the API
==============

``rotational_diagram()`` is not a separate wrapper — it's the exact function the GUI calls,
imported directly. There's no GUI/API divergence to account for here, unlike CO flux above.

.. code-block:: python

    from cube_explorer.api import rotational_diagram

    # H2 v=0->0 S(1)-S(7), matching data/H2_lines.csv, matched against HITRAN
    molecular_data = {
        'wavelengths': {1: 17.034846, 2: 12.278612, 3: 9.664911, 4: 8.025041,
                        5: 6.909509, 6: 6.108564, 7: 5.511183},
        'Aul': {1: 4.758e-10, 2: 2.753e-9, 3: 9.83e-9, 4: 2.641e-8,
                5: 5.876e-8, 6: 1.141e-7, 7: 2.0e-7},
        'Eu': {1: 1015.947, 2: 1683.068, 3: 2505.873, 4: 3477.452,
               5: 4589.958, 6: 5834.798, 7: 7202.827},
        'gu': {1: 7, 2: 9, 3: 11, 4: 13, 5: 15, 6: 17, 7: 19},  # bare 2*J_upper+1;
            # apply_OPR() applies the ortho/para nuclear-spin weighting on top of this
    }

    result = rotational_diagram(
        fluxes={1: 178e-16, 2: 55.9e-16, 3: 122e-16, 4: 28.4e-16,
                5: 72.3e-16, 6: 14.4e-16, 7: 22.2e-16},
        flux_errors={1: 9e-16, 2: 3.2e-16, 3: 6.3e-16, 4: 2.4e-16,
                     5: 4e-16, 6: 3.2e-16, 7: 3.8e-16},
        molecular_data=molecular_data,
        R=8.5,      # AU
        dist=71.5,  # pc
        fit2T=True,
    )
    print(f"1T Temperature: {result['1T']['T']:.1f} K")
    print(f"2T Temperatures: {result['2T']['T1']:.1f} K, {result['2T']['T2']:.1f} K")

This reproduces the first worked example above exactly (273.4 K / 863.4 K).

.. important::

   A 2T fit has 4 free parameters, so you need meaningfully more than 4 lines for the fit
   (and its uncertainties) to mean anything. Passing ``fit2T=True`` with too few points
   doesn't raise an error; it just returns a bad fit.

Next Steps
==========

- :doc:`co_flux_extraction` — a different molecule's line fluxes, fit the same way
- :doc:`gaussian_fitting_cube` — the per-pixel Gaussian fitting **Calculate Fluxes** shares
  its algorithm with
- :doc:`exporting_data` — other export paths
