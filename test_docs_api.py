#!/usr/bin/env python
"""
Documentation API Test Suite
=============================

Runs the code examples published in the IFU Analyzer API documentation exactly
as written. If a section fails here, the documentation is wrong.

Each test body is a copy of a documented snippet. Keep them literal -- the value
of this file is that you can diff it against the docs. Resist adding checks the
documentation does not actually claim.

Documentation sources:
  - cube_explorer/api.py module docstring ("Example Usage", incl. rotational_diagram)
  - docs/user_guide/api_examples.rst
  - docs/user_guide/loading_data.rst ("Using the API", incl. MUSE instrument detection)
  - docs/user_guide/selecting_regions.rst ("Using the API")
  - docs/user_guide/gaussian_fitting_1d.rst ("Using the API")
  - docs/user_guide/gaussian_fitting_cube.rst ("Using the API")
  - docs/user_guide/co_flux_extraction.rst ("Using the API", under Multi-Gaussian Fit)
  - docs/user_guide/co_flux_extraction.rst ("Using the CO Flux API")

data/ is gitignored, so a fresh clone has no cubes. Sections needing a FITS file
report [SKIP] rather than failing.

Run:
    python test_docs_api.py
"""

import os
import sys

# Documented snippets end with plt.show(), which blocks on an interactive
# backend. Force a headless backend so the suite runs unattended. This is a
# harness concern -- the snippets themselves stay verbatim.
os.environ.setdefault("MPLBACKEND", "Agg")

CUBE = os.path.join("data", "MIRI_CITau", "Level3_ch1-long_s3d.fits")
NIRSPEC_CUBE = os.path.join("data", "NIRSpec", "jw01621-o001_t001_nirspec_g395h-f290lp_s3d.fits")
MUSE_CUBE = os.path.join("data", "MUSE", "ADP.2019-03-04T22_21_26.534.fits")

results = []

# A full-cube Gaussian fit takes ~3 minutes, so the two cube sections share one run.
_CUBE_MAPS = {}


def run(name, func):
    """Run one documented example; record pass/fail/skip."""
    print("\n" + "=" * 70)
    print(f" {name}")
    print("=" * 70)

    if getattr(func, "needs_cube", False) and not os.path.exists(CUBE):
        print(f"[SKIP] sample cube not found: {CUBE}")
        results.append((name, "SKIP"))
        return

    if getattr(func, "needs_nirspec_cube", False) and not os.path.exists(NIRSPEC_CUBE):
        print(f"[SKIP] sample cube not found: {NIRSPEC_CUBE}")
        results.append((name, "SKIP"))
        return

    if getattr(func, "needs_muse_cube", False) and not os.path.exists(MUSE_CUBE):
        print(f"[SKIP] sample cube not found: {MUSE_CUBE}")
        results.append((name, "SKIP"))
        return

    try:
        func()
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        results.append((name, "FAIL"))
    else:
        print("[OK] example ran as documented")
        results.append((name, "PASS"))


def needs_cube(func):
    func.needs_cube = True
    return func


def needs_nirspec_cube(func):
    func.needs_nirspec_cube = True
    return func


def needs_muse_cube(func):
    func.needs_muse_cube = True
    return func


# ---------------------------------------------------------------------------
# BASIC DATA LOADING
# ---------------------------------------------------------------------------

@needs_cube
def basic_data_loading():
    from cube_explorer.api import load_cube

    # Load a FITS file (instrument auto-detected)
    cube = load_cube(CUBE)

    print(f"Instrument: {cube.instrument}")
    print(f"Wavelength range: {cube.wavelength.min():.4f} - {cube.wavelength.max():.4f}")
    print(f"Cube shape (wavelength, x, y): {cube.cube.shape}")
    print(f"Spectrum shape: {cube.spectrum.shape}")


# ---------------------------------------------------------------------------
# ACCESS CUBE COMPONENTS
# ---------------------------------------------------------------------------

@needs_cube
def access_cube_components():
    from cube_explorer.api import load_cube

    cube = load_cube(CUBE)

    # Wavelength array
    wavelength = cube.wavelength        # Array of wavelength values
    print(f"Wavelengths: {len(wavelength)} points")
    print(f"Range: {wavelength.min():.4f} - {wavelength.max():.4f}")

    # 3D spectral cube
    full_cube = cube.cube              # Shape: (wavelength, x_pixels, y_pixels)
    print(f"Cube dimensions: {full_cube.shape}")

    # 1D spectrum (summed over all spatial pixels)
    spectrum_1d = cube.spectrum         # Array of flux values

    # Metadata
    header_info = cube.header           # FITS header dictionary
    wcs_info = cube.wcs                 # World Coordinate System
    instrument = cube.instrument        # Instrument name (MUSE, MIRI, etc.)

    print(f"Spectrum points: {len(spectrum_1d)}")
    print(f"Header cards: {len(header_info)}")
    print(f"WCS: {type(wcs_info).__name__}")
    print(f"Instrument: {instrument}")


# ---------------------------------------------------------------------------
# EXTRACT SPECTRA FROM SPECIFIC REGIONS (docs/user_guide/selecting_regions.rst,
# "Using the API"). extract_pixel_spectrum/extract_region_spectrum wrap the same
# CubeData.get_pixel_spectrum/compute_region_spectrum methods the GUI itself uses
# (gui_spectrum.py) -- this checks the docs' two concrete claims: (1) the pixel
# call uses [y, x] axis order, not [x, y], and (2) the region call excludes NaN
# pixels from the mean instead of propagating NaN like a naive .mean() would.
# ---------------------------------------------------------------------------

@needs_cube
def extract_spectra_from_regions():
    import numpy as np
    from cube_explorer.api import load_cube, extract_pixel_spectrum, extract_region_spectrum

    cube = load_cube(CUBE)

    # Extract spectrum from a single pixel (equivalent to the Point tool)
    x, y = 20, 25
    pixel_spectrum = extract_pixel_spectrum(cube, x, y)

    # "uses [y, x] axis order" -- matches raw cube.cube[:, y, x], not [:, x, y]
    assert np.allclose(pixel_spectrum, cube.cube[:, y, x], equal_nan=True), \
        "docs claim extract_pixel_spectrum(cube, x, y) indexes as cube.cube[:, y, x]"

    # Extract spectrum from a spatial region (equivalent to the Rectangle tool)
    x_start, x_end = 10, 30
    y_start, y_end = 15, 35
    mask = np.zeros(cube.cube.shape[1:], dtype=bool)
    mask[y_start:y_end, x_start:x_end] = True
    averaged_spectrum = extract_region_spectrum(cube, mask)

    # "excludes NaN pixels rather than propagating NaN" -- pick a region that
    # actually touches the cube's ragged edge, where a naive .mean() goes NaN.
    nan_ys, nan_xs = np.where(np.isnan(cube.cube[0]))
    assert len(nan_ys) > 0, "expected this fixture to have some NaN spatial pixels"
    ny, nx = nan_ys[0], nan_xs[0]
    edge_mask = np.zeros(cube.cube.shape[1:], dtype=bool)
    y0, y1 = max(0, ny - 2), ny + 3
    x0, x1 = max(0, nx - 2), nx + 3
    edge_mask[y0:y1, x0:x1] = True

    naive_mean = cube.cube[:, y0:y1, x0:x1].mean(axis=(1, 2))
    edge_spectrum = extract_region_spectrum(cube, edge_mask)
    assert np.isnan(naive_mean).any(), "expected the naive mean to hit NaN at this edge region"
    assert not np.isnan(edge_spectrum).any(), \
        "docs claim extract_region_spectrum excludes NaN pixels instead of propagating NaN"

    # Plot the spectrum
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(cube.wavelength, averaged_spectrum, 'b-', linewidth=1)
    plt.xlabel('Wavelength (µm)')
    plt.ylabel('Flux')
    plt.title(f'Spectrum: pixels [{x_start}:{x_end}, {y_start}:{y_end}]')
    plt.grid(True, alpha=0.3)
    plt.show()


# ---------------------------------------------------------------------------
# AVAILABLE CUBE PROPERTIES
# ---------------------------------------------------------------------------

@needs_cube
def available_cube_properties():
    from cube_explorer.api import load_cube

    cube = load_cube(CUBE)

    # Data access
    print(f"Wavelength points: {len(cube.wavelength)}")
    print(f"Spectrum points:   {len(cube.spectrum)}")
    print(f"Cube shape:        {cube.cube.shape}")

    # Metadata
    print(f"Instrument:        {cube.instrument}")
    print(f"Header cards:      {len(cube.header)}")
    print(f"WCS:               {type(cube.wcs).__name__}")

    # File information
    print(f"Filename:          {cube.filename}")

    # Remaining metadata comes from the FITS header. Keyword names are
    # instrument-specific, so use .get() rather than indexing.
    print(f"Target:            {cube.header.get('TARGNAME')}")
    print(f"Observation date:  {cube.header.get('DATE-OBS')}")


# ---------------------------------------------------------------------------
# MUSE INSTRUMENT DETECTION (docs/user_guide/loading_data.rst, "Supported
# Instruments"). basic_data_loading/access_cube_components/available_cube_properties
# above cover the API shape using the MIRI fixture; this checks the docs' separate
# claim that MUSE is auto-detected too, which nothing else in this suite exercises.
# ---------------------------------------------------------------------------

@needs_muse_cube
def muse_instrument_detection():
    from cube_explorer.api import load_cube

    cube = load_cube(MUSE_CUBE)

    print(f"Instrument: {cube.instrument}")
    print(f"Wavelength range: {cube.wavelength.min():.4f} - {cube.wavelength.max():.4f}")
    print(f"Cube shape (wavelength, x, y): {cube.cube.shape}")

    assert cube.instrument == 'MUSE', \
        f"docs claim MUSE is auto-detected; got instrument={cube.instrument!r}"


# ---------------------------------------------------------------------------
# 1D SPECTRUM FITTING -- FITTING ONE LINE
# ---------------------------------------------------------------------------

@needs_cube
def fitting_one_line():
    from cube_explorer.api import load_cube, fit_line

    cube = load_cube(CUBE)

    result = fit_line(
        wavelength=cube.wavelength,
        spectrum=cube.spectrum,
        x1=7.491492,           # Left boundary
        x2=7.516985,           # Right boundary
        peak_x=7.503455,       # Centre guess - the GUI's "Peak" click
        header=cube.header,    # Required for the erg/s/cm2 conversion
        wave_unit=cube.wave_unit,
    )

    if result.success:
        print(f"Center:    {result.center:.6f} {cube.wave_unit}")
        print(f"Amplitude: {result.amplitude:.4e} {cube.flux_unit}")
        print(f"FWHM:      {result.fwhm:.6g} {cube.wave_unit}")
        print(f"Area:      {result.area:.4e} {result.area_unit}")
        print(f"R2:        {result.r_squared:.4f}")
    else:
        print(f"Fit failed: {result.message}")


# ---------------------------------------------------------------------------
# 1D SPECTRUM FITTING -- RESULT FIELDS
# ---------------------------------------------------------------------------

@needs_cube
def result_fields():
    from cube_explorer.api import load_cube, fit_line

    cube = load_cube(CUBE)

    result = fit_line(cube.wavelength, cube.spectrum, 7.491492, 7.516985,
                      peak_x=7.503455,
                      header=cube.header, wave_unit=cube.wave_unit)

    # Fitted parameters
    result.center            # Line center, spectral axis unit
    result.amplitude         # Peak height above baseline, cube flux unit
    result.sigma             # Gaussian sigma, spectral axis unit
    result.fwhm              # 2.355 * sigma

    # Integrated flux
    result.area              # Converted area (erg/s/cm2 when possible)
    result.area_unit         # String naming the unit of `area` - always check this
    result.area_native       # Area before conversion, in cube flux x wavelength units

    # Baseline
    result.baseline           # Fitted linear baseline evaluated at each data point
    result.baseline_slope     # Continuum gradient
    result.baseline_intercept # Baseline flux at x1 (the "Left endpoint" in the GUI)

    # Quality and status
    result.r_squared         # Coefficient of determination
    result.success           # Did the fit converge?
    result.message           # Human-readable status

    print(f"All documented fields readable; area_unit={result.area_unit}")


# ---------------------------------------------------------------------------
# PIXEL-BY-PIXEL CUBE FITTING -- FITTING THE CUBE
# ---------------------------------------------------------------------------
# Slow-ish: a real full-cube run, ~1-2 minutes. It stays because the page quotes
# concrete detection counts, and an example nobody runs is how the counts drift
# away from the code.

@needs_cube
def cube_fitting_the_cube():
    from cube_explorer.api import load_cube, create_gaussian_map

    cube = load_cube(CUBE)

    maps = create_gaussian_map(
        cube,
        lam_min=7.446784,        # Left boundary
        lam_max=7.478707,        # Right boundary
        noise_min=7.516742,      # Line-free region, as in the GUI
        noise_max=7.562928,
        fac_above_stdres=3.0,    # Detection threshold in sigma
        n_jobs=8,
    )

    print(f"Detections:   {maps['is_detection'].sum()}")
    print(f"Upper limits: {maps['is_upper_limit'].sum()}")
    print(f"No data:      {maps['area'].size - maps['is_detection'].sum() - maps['is_upper_limit'].sum()}")

    area_map = maps['area']          # 2D array, shape (n_y, n_x)

    # --- checks on claims the page makes, beyond "it runs" ---
    import numpy as np

    # "the three states account for the whole map"
    n_det = int(maps['is_detection'].sum())
    n_up = int(maps['is_upper_limit'].sum())
    n_none = area_map.size - n_det - n_up
    assert n_det + n_up + n_none == area_map.size
    assert not np.any(maps['is_detection'] & maps['is_upper_limit']), \
        "docs claim the states are mutually exclusive"

    # "amplitude/center/sigma/fwhm/chi2 are NaN wherever area is an upper limit"
    up = maps['is_upper_limit']
    if up.any():
        for key in ('amplitude', 'center', 'sigma', 'fwhm', 'r_squared'):
            assert np.all(np.isnan(maps[key][up])), \
                f"docs claim {key} is NaN at upper limits"

    print(f"Verified: {n_det} detections + {n_up} upper limits + {n_none} no-data "
          f"= {area_map.size} pixels")

    # Hand the result to the returned-keys test rather than refitting the cube.
    _CUBE_MAPS['maps'] = maps


# ---------------------------------------------------------------------------
# PIXEL-BY-PIXEL CUBE FITTING -- RETURNED KEYS
# ---------------------------------------------------------------------------

@needs_cube
def cube_returned_keys():
    # Reuses the fit from the previous test: the documented snippet here is the key
    # list, not the call, and a second full-cube run would cost another three minutes
    # to re-derive maps we already have.
    maps = _CUBE_MAPS.get('maps')
    if maps is None:
        raise RuntimeError("cube fit unavailable - the preceding cube test must run first")

    maps['amplitude']         # Peak amplitude, in the cube's native flux unit (e.g. MJy/sr)
    maps['center']            # Line center wavelength
    maps['sigma']             # Gaussian sigma
    maps['fwhm']              # 2.355 * sigma
    maps['area']              # Integrated flux, or an upper limit - check is_upper_limit
    maps['success']           # Boolean: did the pixel converge and pass the detection test?
    maps['is_upper_limit']    # Boolean: area is an upper limit (may still be NaN)
    maps['residuals_cube']    # 3D residuals cached during fitting
    maps['fac_above_stdres']  # The threshold that was used
    maps['wavelength_unit']   # Spectral axis unit string
    maps['global_fit']        # The summed-spectrum fit used for initial guesses (also
                               # native-unit amplitude)

    assert 'amp_originalunits' not in maps, \
        "docs claim amp_originalunits was removed -- amplitude is the only amplitude key now"

    assert 'amplitude_cgs' not in maps['global_fit'], \
        "docs claim global_fit's amplitude stays native, not CGS-converted"

    print(f"All documented keys present; wavelength_unit={maps['wavelength_unit']!r}, "
          f"fac_above_stdres={maps['fac_above_stdres']}, "
          f"global amplitude={maps['global_fit']['amplitude']:.4e} (native)")


# ---------------------------------------------------------------------------
# CO FLUX EXTRACTION -- MULTI-GAUSSIAN FIT, USING THE API
# ---------------------------------------------------------------------------
# fit_hitran_lines() runs the same algorithm as the GUI's Multi-Gaussian Fit button:
# simultaneous nonlinear curve_fit per wavelength chunk, via fit_glines() (see
# fit_hitran_lines_collective() in co_flux_extractor.py) -- this checks the docs' claim
# that they produce identical results, not just that the call doesn't crash.
# Centers/sigmas are bounded (center within +/-1 instrumental sigma of the catalog
# wavelength, sigma within 0.5-2x the instrumental value) but genuinely fitted, not fixed
# -- so unlike the exact-catalog-match this test used to check, the match here is against
# a fixed reproducibility tolerance, not zero.

@needs_nirspec_cube
def co_flux_multi_gaussian_api():
    from cube_explorer.api import load_cube, fit_hitran_lines
    from cube_explorer.line_catalog import load_contaminating_lines_catalog

    cube = load_cube(NIRSPEC_CUBE)
    contam = load_contaminating_lines_catalog('data/ContaminatingLines.csv')

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
        contam_catalog=contam,
    )

    detected = [r for r in results if r['success']]
    print(f"Fitted {len(detected)}/{len(results)} lines")
    for r in sorted(detected, key=lambda r: r['wavelength']):
        print(f"  {r['wavelength']:.6f} um  area={r['area']:.4e} {r['area_unit']}  R2={r['r_squared']:.4f}")

    # "This reproduces the worked example above exactly" -- all 56 v=1->0 fitted centers
    # (bounded, not catalog-exact) and areas from the GUI screenshot, to the digit. Lines
    # 55/56 here (4.783681, 4.793562) are the exact two lines shown in the docs' "Reading
    # the results" worked example, cross-checked against a real GUI run.
    expected = {
        4.402940: (1.7523e-16, 0.9522), 4.407340: (2.0514e-16, 0.9760),
        4.411409: (3.2444e-16, 0.9919), 4.416001: (1.1136e-16, 0.9964),
        4.421199: (2.3017e-16, 0.9395), 4.425630: (2.0015e-16, 0.9826),
        4.430406: (2.3911e-16, 0.9785), 4.435261: (2.2495e-16, 0.9708),
        4.440246: (3.7372e-16, 0.9716), 4.445402: (2.5733e-16, 0.9856),
        4.450527: (2.5820e-16, 0.9529), 4.456012: (3.0321e-16, 0.9540),
        4.461357: (3.0388e-16, 0.9959), 4.466731: (2.6792e-16, 0.9830),
        4.472096: (3.7597e-16, 0.9799), 4.477679: (4.3253e-16, 0.8657),
        4.483579: (2.9120e-16, 0.9950), 4.489508: (4.0173e-16, 0.9944),
        4.495355: (3.6845e-16, 0.9927), 4.501209: (4.1052e-16, 0.9905),
        4.507249: (3.5471e-16, 0.9688), 4.513572: (3.7954e-16, 0.9976),
        4.519887: (4.1149e-16, 0.9817), 4.526375: (5.0071e-16, 0.9669),
        4.532801: (5.6417e-16, 0.9849), 4.539256: (4.6573e-16, 0.9992),
        4.545856: (5.6962e-16, 0.9865), 4.552541: (5.3910e-16, 0.9397),
        4.559529: (4.5148e-16, 0.9861), 4.566508: (4.6830e-16, 0.9991),
        4.573532: (4.2554e-16, 0.9913), 4.580754: (5.4132e-16, 0.9459),
        4.588250: (7.4931e-16, 0.7944), 4.595350: (7.1870e-16, 0.9976),
        4.602698: (6.6183e-16, 0.9995), 4.610161: (6.3713e-16, 0.9295),
        4.617923: (6.0135e-16, 0.9808), 4.625737: (5.3929e-16, 0.9952),
        4.633578: (4.7096e-16, 0.9941), 4.641540: (4.3847e-16, 0.9907),
        4.649593: (4.6516e-16, 0.9927), 4.657808: (2.6092e-16, 0.9749),
        4.674518: (5.6176e-16, 0.9825), 4.682977: (8.0562e-16, 0.9979),
        4.691537: (7.9505e-16, 0.9995), 4.700179: (9.0172e-16, 0.9958),
        4.709032: (9.3687e-16, 0.9932), 4.718021: (8.8954e-16, 0.9949),
        4.727012: (8.4744e-16, 0.9992), 4.736167: (7.3723e-16, 0.9999),
        4.745498: (7.3614e-16, 0.9987), 4.754820: (6.9598e-16, 0.9954),
        4.764279: (6.0208e-16, 0.9969), 4.773897: (5.4440e-16, 0.9945),
        4.783681: (5.0950e-16, 0.9889), 4.793562: (5.5141e-16, 0.9156),
    }
    assert len(detected) == 56, f"docs' worked example has 56 detected lines, got {len(detected)}"
    for r in detected:
        match = min(expected, key=lambda c: abs(c - r['wavelength']))
        # Centers are bounded (+/-1 instrumental sigma of the catalog wavelength) but
        # genuinely fitted -- close, not catalog-exact. curve_fit itself is deterministic
        # for identical input, so this checks reproducibility against a verified real run,
        # not "hasn't drifted from HITRAN."
        assert abs(match - r['wavelength']) < 1e-3, \
            f"fitted center {r['wavelength']:.6f} doesn't match any documented line closely enough"
        exp_area, exp_r2 = expected[match]
        assert abs(r['area'] - exp_area) / exp_area < 1e-3, \
            f"line at {match}: area {r['area']:.4e} != documented {exp_area:.4e}"
        assert abs(r['r_squared'] - exp_r2) < 1e-3, \
            f"line at {match}: R2 {r['r_squared']:.4f} != documented {exp_r2:.4f}"
        assert r['area_unit'] == 'erg/s/cm²', \
            f"docs claim area_unit is erg/s/cm2 when header/wave_unit are given; got {r['area_unit']!r}"

    print(f"\nVerified: all {len(detected)} lines match the documented worked example "
          f"(centers, areas, R2, units)")


# ---------------------------------------------------------------------------
# CO FLUX -- SINGLE SPECTRUM / FULL MAP API (docs/user_guide/co_flux_extraction.rst,
# "Using the CO Flux API"). Unlike fit_hitran_lines above, extract_co_flux/create_co_flux_map
# need a CO line catalog file on disk -- the docs show a placeholder path
# ("path/to/CO_catalog.ecsv"), so this builds a real one from HITRAN first, in the column
# format load_CO_line_catalog() actually expects (line_wave_um/species/transition/a), not
# the wrong wavelength/einstein_coeff format the docs used to show.

def _build_co_catalog_ecsv():
    from cube_explorer.line_catalog import fetch_hitran_data
    from astropy.table import Table
    hitran_lines, _ = fetch_hitran_data(4.70, 4.80, 'CO', isotopologue=1, v_upper=1, v_lower=0)
    t = Table()
    t['line_wave_um'] = [l['wavelength'] for l in hitran_lines]
    t['species'] = [l['species'] for l in hitran_lines]
    t['transition'] = [l['transition'] for l in hitran_lines]
    t['a'] = [l['a_coeff'] for l in hitran_lines]
    path = os.path.join(os.environ.get("TEMP", "."), "test_docs_co_catalog.ecsv")
    t.write(path, format='ascii.ecsv', overwrite=True)
    return path


@needs_nirspec_cube
def co_flux_single_spectrum_api():
    import numpy as np
    from cube_explorer.api import load_cube, extract_co_flux

    cube = load_cube(NIRSPEC_CUBE)
    catalog_path = _build_co_catalog_ecsv()

    co_result = extract_co_flux(
        wavelength=cube.wavelength,
        spectrum=cube.spectrum,
        co_catalog_path=catalog_path,
        header=cube.header,      # Required for the erg/s/cm2 conversion
        wave_unit=cube.wave_unit,
    )

    print(f"Total CO flux: {co_result['co_flux']:.4e} {co_result['flux_unit']}")
    print(f"Number of lines detected: {len(co_result['detected_lines'])}")
    print(f"Mean continuum level: {co_result['mean_continuum']:.3f}")

    assert co_result['flux_unit'] == 'erg/s/cm2', \
        f"docs claim erg/s/cm2 when header/wave_unit are given; got {co_result['flux_unit']!r}"
    assert np.isfinite(co_result['co_flux']) and co_result['co_flux'] > 0, \
        f"expected a positive finite co_flux, got {co_result['co_flux']!r}"
    assert len(co_result['detected_lines']) > 0, "expected at least some detected lines"


@needs_nirspec_cube
def co_flux_map_api():
    import numpy as np
    from cube_explorer.api import load_cube, create_co_flux_map

    cube = load_cube(NIRSPEC_CUBE)

    # Uses HITRAN automatically by default, or pass co_catalog_path for a custom catalog
    co_flux_map = create_co_flux_map(
        cube,
        wavelength_min=4.70, wavelength_max=4.80,
        n_jobs=4,
    )

    print(f"CO flux map shape: {co_flux_map['flux_map'].shape}, unit: {co_flux_map['flux_unit']}")
    print(f"Mean CO flux: {co_flux_map['flux_map'][co_flux_map['flux_map'] > 0].mean():.4e}")
    print(f"Pixels with CO detection: {(co_flux_map['detection_map'] > 0).sum()}")
    print(f"Pixels with a contam detection: {(co_flux_map['contam_detection_map'] > 0).sum()}")
    print(f"Mean contam flux: {co_flux_map['contam_flux_map'][co_flux_map['contam_flux_map'] > 0].mean():.4e}")

    assert co_flux_map['flux_unit'] == 'erg/s/cm2', \
        f"docs claim erg/s/cm2 for this NIRSpec cube; got {co_flux_map['flux_unit']!r}"
    assert co_flux_map['wave_range'] == (4.70, 4.80), \
        f"wavelength_min/max should be forwarded as the fit's wave_range; got {co_flux_map['wave_range']}"
    n_valid = (~np.isnan(co_flux_map['flux_map'])).sum()
    assert n_valid > 0, "expected at least some valid (non-NaN) pixels"

    # docs claim create_co_flux_map now always returns these two additional maps too
    for key in ('contam_flux_map', 'contam_detection_map'):
        assert key in co_flux_map, f"docs claim create_co_flux_map always returns {key!r}"
        assert co_flux_map[key].shape == co_flux_map['flux_map'].shape, \
            f"docs claim {key!r} is the same shape as flux_map"


# ---------------------------------------------------------------------------
# ROTATIONAL DIAGRAM (cube_explorer/api.py module docstring, "Example Usage").
# molecular_data/fluxes below are the H2 v=0->0 S(1)-S(7) lines from data/H2_lines.csv,
# matched against HITRAN (see api.py's docstring note) -- no FITS fixture needed, so this
# runs unconditionally rather than behind @needs_cube.

def rotational_diagram_docstring():
    from cube_explorer.api import rotational_diagram

    molecular_data = {
        'wavelengths': {1: 17.034846, 2: 12.278612, 3: 9.664911, 4: 8.025041,
                        5: 6.909509, 6: 6.108564, 7: 5.511183},
        'Aul': {1: 4.758e-10, 2: 2.753e-9, 3: 9.83e-9, 4: 2.641e-8,
                5: 5.876e-8, 6: 1.141e-7, 7: 2.0e-7},
        'Eu': {1: 1015.947, 2: 1683.068, 3: 2505.873, 4: 3477.452,
               5: 4589.958, 6: 5834.798, 7: 7202.827},
        'gu': {1: 7, 2: 9, 3: 11, 4: 13, 5: 15, 6: 17, 7: 19},
    }
    result = rotational_diagram(
        fluxes={1: 178e-16, 2: 55.9e-16, 3: 122e-16, 4: 28.4e-16,
                5: 72.3e-16, 6: 14.4e-16, 7: 22.2e-16},
        flux_errors={1: 9e-16, 2: 3.2e-16, 3: 6.3e-16, 4: 2.4e-16,
                     5: 4e-16, 6: 3.2e-16, 7: 3.8e-16},
        molecular_data=molecular_data,
        R=8.5, dist=71.5, fit2T=True
    )
    print(f"1T Temperature: {result['1T']['T']:.1f} K")
    print(f"2T Temperatures: {result['2T']['T1']:.1f} K, {result['2T']['T2']:.1f} K")

    assert abs(result['1T']['T'] - 535.6) < 0.1, f"1T doesn't match docstring: {result['1T']['T']:.1f}"
    assert abs(result['2T']['T1'] - 273.4) < 0.1, f"2T T1 doesn't match docstring: {result['2T']['T1']:.1f}"
    assert abs(result['2T']['T2'] - 863.4) < 0.1, f"2T T2 doesn't match docstring: {result['2T']['T2']:.1f}"


# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print(" IFU ANALYZER -- DOCUMENTATION API TEST SUITE")
    print("=" * 70)

    run("BASIC DATA LOADING", basic_data_loading)
    run("ACCESS CUBE COMPONENTS", access_cube_components)
    run("EXTRACT SPECTRA FROM REGIONS", extract_spectra_from_regions)
    run("AVAILABLE CUBE PROPERTIES", available_cube_properties)
    run("MUSE INSTRUMENT DETECTION", muse_instrument_detection)
    run("1D FITTING -- ONE LINE", fitting_one_line)
    run("1D FITTING -- RESULT FIELDS", result_fields)
    run("CUBE FITTING -- FULL CUBE (slow, ~1-2 min)", cube_fitting_the_cube)
    run("CUBE FITTING -- RETURNED KEYS", cube_returned_keys)
    run("CO FLUX -- MULTI-GAUSSIAN FIT API", co_flux_multi_gaussian_api)
    run("CO FLUX -- SINGLE SPECTRUM API", co_flux_single_spectrum_api)
    run("CO FLUX -- FULL MAP API (slow, ~2-3 min)", co_flux_map_api)
    run("ROTATIONAL DIAGRAM API", rotational_diagram_docstring)

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    for name, status in results:
        print(f"  [{status}] {name}")

    failed = [n for n, s in results if s == "FAIL"]
    if failed:
        print(f"\n[ERROR] {len(failed)} documented example(s) failed.")
        return 1

    print("\n[OK] All documented examples ran.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
