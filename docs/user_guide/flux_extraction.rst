================
Flux Extraction
================

Extract and measure spectral line fluxes for a chosen target species (CO, H2, or any
molecule/atom HITRAN or a custom catalog supports) from spectral data cubes. This tool
analyzes emission or absorption across multiple rotational transitions of that species.

Starting Flux Extraction
=========================

**Fetch HITRAN**

1. Enter **λ min** and **λ max** (in µm) in the **Lines & Fit** group, then click **Fetch HITRAN**.
   The boxes fill from the cube when you load it
2. Choose a molecule (required)
3. Choose isotopologue number (optional)
4. Choose transition level (optional, different molecules have different transition levels)
5. Refresh HITRAN cache (optional)

Seven common molecules are preloaded for 1–30 µm and ready offline, no fetch needed: **H2O,
CO2, CO, OH, HCN, C2H2, H2**. Any other molecule, or any other wavelength range, is fetched
live from HITRAN the first time you request it (via HAPI, version 1.3.0.0), so it needs an
internet connection. The lines are then cached locally for next time.

52 molecules support transition-level filtering (step 4 above). These are the molecules that
have HITRAN lines between 1 and 30 µm:

CO, H2, OH, H2O, HCN, C2H2, CO2, CS, OCS, SO2, C2H6, C4H2, HC3N, CH4, NH3, CH3, H3⁺, HF,
HCl, HBr, HI, N2, NO⁺, NO, ClO, O2, SO, O3, N2O, NO2, HNO3, H2CO, HOCl, H2O2, H2S, HO2,
COF2, COCl2, CS2, PH3, SO3, GeH4, HCOOH, C2H4, CH3OH, CH3Cl, CH3Br, CH3CN, CH3F, CH3I,
COFCl, ClNO2.

**Import Lines**

When HITRAN isn't reachable, click **Import Lines** to load your own line list from a CSV
instead. Both ``wavelength`` and ``species`` columns are required.

Once lines are loaded (either way), the spectrum shows them. Hovering over a line shows its
species, wavelength, transition, and Einstein coefficients. Click **Show Lines** to open a table
of every currently loaded line: detections and contaminants together, with "Detection
Only"/"Contaminating Only" filters. Check/uncheck rows to change what's plotted. Click any
column header to sort by that column.

Load Contaminating Lines
=========================

Additionally (and optionally), you can mark contaminating (non-target) lines so both
**Flux Map** and **Multi-Gauss** (below) can account for them instead of letting
their flux distort a nearby target line's measurement. Two ways, and they merge together:

**From a CSV**

1. Click **Contam. Lines** and load a CSV file with unwanted lines

Both ``wavelength`` and ``species`` columns are required.

**By marking one on the spectrum**

1. Click **Mark Contaminant**, then click the feature on the spectrum
2. No HITRAN match is needed -- it's listed with species "N/A"
3. The button stays toggled on so you can mark several in a row; click it again to stop
4. To remove one, uncheck it in **Show Lines** or click its marker on the spectrum

Run **Flux Map** / **Multi-Gauss** again after loading or marking one for the fit to
pick it up.

Multi-Gaussian Fit
=====================

The **Multi-Gauss** button fits every currently-plotted line at once via ``curve_fit``, using each line's own
instrumental width. Lines that land in the same wavelength chunk (close or blended) are fit
together against one shared continuum, so they no longer double-claim overlapping flux the
way independent, uninformed per-line fits would.

Each line's starting width comes from the instrument's own spectral resolution, but center
and sigma are **fitted, not fixed** because each is constrained to a bounded range around
that starting point (center within ±1 instrumental σ of the catalog wavelength, sigma within
0.5–2× the instrumental value), so a real line can settle slightly off its catalog center or
resolve a little narrower/wider than nominal.

If you've loaded or marked a contaminating line (above) sharing a wavelength chunk with a
requested line, it gets its own Gaussian component too, so its flux is absorbed there
instead of leaking into that line's amplitude. It's drawn in **grey** on the plot and
interleaved with the target lines in the results, in wavelength order. It's never counted
toward the target line total or "Total Integrated Flux", but summed on its own as "Total
Contaminant Flux" (see *Reading the results* below).

1. (Optional) Fit a continuum first via **Continuum Tools → Automatic Continuum → Fit
   Continuum**. If you skip this, **Multi-Gauss** computes one automatically the first
   time you click it.
2. Click **Multi-Gauss**

A Gaussian curve is overlaid on the spectrum for each plotted line, and a results window
opens with one block per line, an overall summary, and a separate section listing any
contaminating lines that got their own component. Re-running with the same lines,
contaminants, and active tab offers the cached results instead of re-fitting.

.. figure:: ../_static/multigaussian_contam_lines.png
   :alt: 56 fitted CO v=1->0 Gaussians overlaid on a NIRSpec spectrum, with six H2 contaminants shown separately in grey
   :width: 100%

   56 CO v=1→0 lines (4.40–4.80 µm) fitted simultaneously via **Multi-Gaussian Fit** on a
   NIRSpec G395H spectrum -- CO used here as the worked example; any target species works
   the same way. The grey curves are H2 lines from the loaded contaminating-lines catalog,
   each fit as its own component. The tallest, near 4.69 µm, would otherwise have distorted
   its neighboring CO lines' amplitudes.

Reading the results
--------------------

Each line's block reports two groups of numbers:

- **Catalog identification**: ``Line λ``, ``Transition``, ``A coeff`` and ``E_upper``,
  copied straight from the HITRAN entry, not measured. Useful for cross-checking that the
  fit actually locked onto the line you think it did.
- **Fitted parameters**: ``Center``, ``Amplitude``, ``Sigma``, ``FWHM``, ``Integrated
  Area`` and ``R²``. ``Center`` and ``Sigma`` are fitted, not fixed, but each is bounded
  around the catalog wavelength and the instrument's spectral resolution respectively (see
  above). Units follow the same rules as the 1D page (:doc:`gaussian_fitting_1d`):
  ``Integrated Area`` is erg/s/cm² wherever the header supports the conversion.

An **OVERALL** summary follows every line block, e.g. from a real 56-line CO run:

.. code-block:: text

    OVERALL:
      Total lines fitted:    56
      Total Integrated Flux: 2.7085e-14 erg/s/cm²
      Mean R²:               0.9768
      Lines with R² >= 0.5:  56/56
      Total Contaminant Flux: 2.3774e-15 erg/s/cm²

``Total Contaminant Flux`` only appears when a contaminating-lines catalog is loaded, and is
summed separately from ``Total Integrated Flux`` (contaminants are never considered for the
target total).

Contaminant lines are not grouped into a trailing section. Each one appears as its own
``Contam N`` block, interleaved with the target ``Line N`` blocks by fitted wavelength --
matching where it actually sits in the spectrum, not sorted apart from the detections:

.. code-block:: text

    FITTED LINES (target and contaminant, by wavelength):
    ======================================================================
    Line 1: CO
      ...
      ────────────────────────────────────────────────────────────
    Contam 1: H2 (not one of the 56 target lines; shown in grey on the plot)
      Center (λ):            4.410011 µm
      Amplitude:             6.3049e+03 MJy/sr
      Integrated Area:       2.9563e-16 erg/s/cm²
      ────────────────────────────────────────────────────────────
    Line 2: CO
      ...

``Contam N`` numbers its own kind (1, 2, 3, ...), separate from the ``Line N`` count above
it -- neither numbering is global across the whole listing.

How Chunking Works
--------------------

**Multi-Gauss** does not fit each line alone. It groups nearby lines into chunks
first.

A chunk is a group of lines close enough in wavelength to overlap. Lines in one chunk
share one continuum. The fit solves for every line in the chunk at the same time, in one
``curve_fit`` call.

This matters because two close lines share flux where they overlap. A fit that sees only
one line assigns it all of that shared flux. An independent fit of the next line then
claims some of the same flux again. Together, the two fits overcount the true flux.

A joint fit avoids this. Each line in the chunk still gets its own Gaussian, but every
line in the chunk fits against the same data and the same baseline at once. The fit splits
overlapping flux between the lines that actually produced it, instead of double-counting it.

Lines just outside a chunk can still affect it. The fit borrows these nearby lines into the
joint model too, so their tails do not leak into the chunk's own lines. A borrowed line's
flux does not count toward the chunk it was borrowed into -- it is only there to keep that
chunk's own fit honest.

The result: every plotted line, in every chunk, is fit at the same time as its neighbors.
No line's flux comes from a fit that ignored what sits next to it.

Using the API
--------------

.. code-block:: python

    from cube_explorer.api import load_cube, fit_hitran_lines
    from cube_explorer.line_catalog import load_contaminating_lines_catalog

    cube = load_cube("data.fits")
    contam = load_contaminating_lines_catalog("path/to/ContaminatingLines.csv")  # optional

    results = fit_hitran_lines(
        wavelength=cube.wavelength,
        spectrum=cube.spectrum,
        molecule='CO',
        wavelength_min=4.40,
        wavelength_max=4.80,
        v_upper=1,              # v=1->0, same filter as the HITRAN dialog
        v_lower=0,
        header=cube.header,     # Required for the erg/s/cm2 conversion
        wave_unit=cube.wave_unit,
        contam_catalog=contam,  # omit (or None) to skip contaminant handling entirely
    )

    detected = [r for r in results if r['success']]
    print(f"Fitted {len(detected)}/{len(results)} lines")
    for r in sorted(detected, key=lambda r: r['wavelength']):
        print(f"  {r['wavelength']:.6f} um  area={r['area']:.4e} {r['area_unit']}  R2={r['r_squared']:.4f}")

.. code-block:: text

    Fitted 56/56 lines
      4.402940 um  area=1.7523e-16 erg/s/cm²  R2=0.9522
      4.407340 um  area=2.0514e-16 erg/s/cm²  R2=0.9760
      4.411409 um  area=3.2444e-16 erg/s/cm²  R2=0.9919
      ...
      4.773897 um  area=5.4440e-16 erg/s/cm²  R2=0.9945
      4.783681 um  area=5.0950e-16 erg/s/cm²  R2=0.9889
      4.793562 um  area=5.5141e-16 erg/s/cm²  R2=0.9156

**Run the Full-Cube Flux Map**

Extract a flux map across every spatial pixel, for whichever lines are currently checked
in the **Show Lines** dialog (any species -- the map isn't restricted to CO):

1. Set **λ min** and **λ max** (in µm) in the **Lines & Fit** group. The map uses only the
   checked lines inside this range.
2. Click on **Flux Map**. The application checks the range and the lines first. If it finds
   a problem, a message tells you what to change, for example "No Lines Selected": click
   **Show Lines** and check some lines in the range.
3. A popup explains that continuum subtraction runs automatically per pixel, using default
   parameters that work well for typical spectra. Click **No** to use those defaults, or
   **Yes** to open the same parameter dialog as the manual Automatic Continuum tool and set
   your own values instead. **Cancel** stops here.
4. A preview shows the brightest pixel within your wavelength range (a sanity check before
   committing to a full-cube run), and exactly the checked lines that will be fit. Click
   **YES - Extract Map** to proceed.

If you've already run an extraction with the same wavelength range, line selection, and
continuum choice, you'll be offered the cached results instead of re-running.

**Progress Monitoring**

The progress dialog updates per pixel as the cube is processed.

**Output Products**

The **Display** dropdown in the results window switches between four maps, all from the
same run. The flux/detection labels reflect whichever species was targeted (e.g. "CO Flux"
for a CO run, "H2 Flux" for H2) -- the example below used CO:

.. figure:: ../_static/flux_map.png
   :alt: CO rovibrational flux map, showing a bright compact source
   :width: 100%

   **CO Flux**: integrated flux per pixel, for a run targeting CO.

.. figure:: ../_static/detection_map.png
   :alt: Map of the number of CO lines detected per pixel, diamond-shaped field of view
   :width: 100%

   **Detection Count**: number of target lines detected per pixel, for the same cube.

.. figure:: ../_static/contam_map.png
   :alt: Contaminating-line flux map, peaking at the same location as the CO flux map but roughly an order of magnitude fainter
   :width: 100%

   **Contam Flux**: integrated contaminating-line flux per pixel, computed as part of the
   same per-pixel fit and summed separately from the target flux (never mixed into the
   target total).

.. figure:: ../_static/contam_detection_map.png
   :alt: Map of the number of contaminating lines detected per pixel
   :width: 100%

   **Contam Detection Count**: number of contaminating lines detected per pixel.

**Export as FITS** / **Show Statistics** buttons in the results window apply to whichever
map is currently selected.

**Inspecting Individual Pixels**

With **Pixel Select: ON** (the default), clicking a pixel on any of the four maps opens that pixel's fit:

.. figure:: ../_static/pixel_spectrum.png
   :alt: Pixel spectrum zoomed to the fitted wavelength range, with the fitted continuum and every CO line marked in red
   :width: 100%

   Spectrum for an individual pixel, zoomed to the run's fitted wavelength range (here the
   full 4.40–4.80 µm, 56-line CO band), with the real per-chunk fitted continuum trend and
   every detected target line marked in red.

Turn **Pixel Select** to OFF to zoom/pan the map freely instead of reopening this dialog on every
click.


Using the Flux API
====================

Extract fluxes programmatically for single spectra or full data cubes, using
``extract_flux()`` and ``create_flux_map()``, the same algorithm as the GUI's
**Flux Map** button.

.. note::

   This is a **different algorithm** from **Multi-Gauss**, and the two don't produce
   identical totals for the same spectrum. Both now fit lines simultaneously via the same
   function, so the real remaining difference isn't the fitting method itself, it's *which*
   lines get fit: ``extract_flux``/``create_flux_map`` auto-detect lines per pixel and
   only count one that clears a signal-to-noise threshold first (``snr_threshold``, default
   3σ). ``fit_hitran_lines`` has no SNR gate at all: it fits whatever you selected, weak
   lines included. Summed over many pixels this adds up to Flux Map running
   systematically lower. Lowering ``snr_threshold`` would let more noise into the flux
   instead (a trade-off, not a fix).

**Extract Flux from a Single Spectrum**

.. code-block:: python

    from cube_explorer.api import load_cube, extract_flux

    cube = load_cube("data.fits")

    flux_result = extract_flux(
        wavelength=cube.wavelength,
        spectrum=cube.spectrum,
        target_catalog_path="path/to/CO_catalog.ecsv",
        header=cube.header,      # Required for the erg/s/cm2 conversion
        wave_unit=cube.wave_unit,
    )

    print(f"Total flux: {flux_result['flux']:.4e} {flux_result['flux_unit']}")
    print(f"Number of lines detected: {len(flux_result['detected_lines'])}")
    print(f"Mean continuum level: {flux_result['mean_continuum']:.3f}")

**Create Flux Maps**

Generate flux maps for the entire data cube:

.. code-block:: python

    from cube_explorer.api import load_cube, create_flux_map

    cube = load_cube("data.fits")

    # Uses HITRAN (CO by default) automatically, or pass target_catalog_path for a custom catalog
    flux_map_result = create_flux_map(
        cube,
        wavelength_min=4.70, wavelength_max=4.80,
        n_jobs=4,
    )

    print(f"Flux map shape: {flux_map_result['flux_map'].shape}, unit: {flux_map_result['flux_unit']}")
    print(f"Mean flux: {flux_map_result['flux_map'][flux_map_result['flux_map'] > 0].mean():.4e}")
    print(f"Pixels with a detection: {(flux_map_result['detection_map'] > 0).sum()}")
    print(f"Pixels with a contam detection: {(flux_map_result['contam_detection_map'] > 0).sum()}")
    print(f"Mean contam flux: {flux_map_result['contam_flux_map'][flux_map_result['contam_flux_map'] > 0].mean():.4e}")

Continuum subtraction works the same way as ``extract_flux`` above (automatic per
pixel, with the default parameters unless you pass your own via ``continuum_params``).

**Expected Catalog Format**

The target catalog file should be in ECSV (Enhanced Character Separated Values) format with columns:

- ``line_wave_um``: Wavelength in micrometers
- ``species``: Molecule name (e.g., "CO", "H2")
- ``transition``: Transition description (e.g., "v=1->0, J=4->5")
- ``a``: Einstein A coefficient

Columns are loaded as-is, with no validation: a file missing ``line_wave_um`` or
``species`` won't fail at load time, only later inside the fit, as a ``KeyError``.

**Tips for Flux Extraction**

- Ensure your wavelength calibration is accurate (errors affect line identification)
- The algorithm automatically measures line widths and integrates flux
- For more control over individual line fits, use ``fit_hitran_lines()`` instead

Next Steps
==========

- :doc:`gaussian_fitting_cube`: the general per-pixel Gaussian fitting this shares its
  algorithm with
- :doc:`exporting_data`: other export paths
- :doc:`2d_viewer`: open the exported target/contam flux maps with WCS support
