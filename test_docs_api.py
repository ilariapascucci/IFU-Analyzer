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
  - docs/user_guide/api_examples.rst ("Load a FITS Cube", "Fetch HITRAN Molecular
    Lines", "Fit Spectral Lines")
  - docs/user_guide/loading_data.rst ("Using the API", incl. MUSE instrument detection)
  - docs/user_guide/selecting_regions.rst ("Using the API")
  - docs/user_guide/gaussian_fitting_1d.rst ("Using the API")
  - docs/user_guide/gaussian_fitting_cube.rst ("Using the API")
  - docs/user_guide/flux_extraction.rst ("Using the API", under Multi-Gaussian Fit)
  - docs/user_guide/flux_extraction.rst ("Using the Flux API")
  - docs/user_guide/rotational_diagram.rst ("Run Map, from a script", "Run Map, multiple
    cubes at different pixel scales")

data/ is gitignored, so a fresh clone has no cubes. Sections needing a FITS file
report [SKIP] rather than failing.

Run:
    python test_docs_api.py
"""

import os
import sys

# Force a headless backend so documented plt.show() calls don't block; a
# harness concern only -- the snippets themselves stay verbatim.
os.environ.setdefault("MPLBACKEND", "Agg")

CUBE = os.path.join("data", "MIRI_CITau", "Level3_ch1-long_s3d.fits")
NIRSPEC_CUBE = os.path.join("data", "NIRSpec", "jw01621-o001_t001_nirspec_g395h-f290lp_s3d.fits")
MUSE_CUBE = os.path.join("data", "MUSE", "ADP.2019-03-04T22_21_26.534.fits")
MIRI_CH1_LONG = os.path.join("data", "HH30", "jw01282-c1024_t029_miri_ch1-long_s3d.fits")
MIRI_CH2_LONG = os.path.join("data", "HH30", "jw01282-c1024_t029_miri_ch2-long_s3d.fits")

results = []

# A full-cube Gaussian fit takes ~3 minutes, so the two cube sections share one run.
_CUBE_MAPS = {}


def run(name, func):
    """Run one documented example; record pass/fail/skip.

    Args:
        name (str): Label for the example, printed and stored in results.
        func (callable): Example function to run.

    Returns:
        None
    """
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

    if getattr(func, "needs_hh30_miri", False) and not (
            os.path.exists(MIRI_CH1_LONG) and os.path.exists(MIRI_CH2_LONG)):
        print(f"[SKIP] sample cubes not found: {MIRI_CH1_LONG}, {MIRI_CH2_LONG}")
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
    """Mark an example as needing the sample MIRI cube; skip it if absent.

    Args:
        func (callable): Example function to mark.

    Returns:
        func (callable): The same function, with needs_cube set.
    """
    func.needs_cube = True
    return func


def needs_nirspec_cube(func):
    """Mark an example as needing the sample NIRSpec cube; skip it if absent.

    Args:
        func (callable): Example function to mark.

    Returns:
        func (callable): The same function, with needs_nirspec_cube set.
    """
    func.needs_nirspec_cube = True
    return func


def needs_muse_cube(func):
    """Mark an example as needing the sample MUSE cube; skip it if absent.

    Args:
        func (callable): Example function to mark.

    Returns:
        func (callable): The same function, with needs_muse_cube set.
    """
    func.needs_muse_cube = True
    return func


def needs_hh30_miri(func):
    """Mark an example as needing both HH30 MIRI channel cubes; skip if absent.

    Args:
        func (callable): Example function to mark.

    Returns:
        func (callable): The same function, with needs_hh30_miri set.
    """
    func.needs_hh30_miri = True
    return func


# --- Basic data loading ---

@needs_cube
def basic_data_loading():
    """Documented example: load a cube and print its basic properties.

    Args:
        None

    Returns:
        None
    """
    from cube_explorer.api import load_cube

    # Load a FITS file (instrument auto-detected)
    cube = load_cube(CUBE)

    print(f"Instrument: {cube.instrument}")
    print(f"Wavelength range: {cube.wavelength.min():.4f} - {cube.wavelength.max():.4f}")
    print(f"Cube shape (wavelength, x, y): {cube.cube.shape}")
    print(f"Spectrum shape: {cube.spectrum.shape}")


# --- Access cube components ---

@needs_cube
def access_cube_components():
    """Documented example: access every attribute of a loaded cube.

    Args:
        None

    Returns:
        None
    """
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


# --- Available cube properties ---

@needs_cube
def available_cube_properties():
    """Documented example: print every property listed for a loaded cube.

    Args:
        None

    Returns:
        None
    """
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


# --- MUSE instrument detection ---

@needs_muse_cube
def muse_instrument_detection():
    """Documented example: earlier tests cover the API shape using the MIRI
    fixture; this checks the docs' separate claim that MUSE is auto-detected
    too, which nothing else in this suite exercises.

    Args:
        None

    Returns:
        None
    """
    from cube_explorer.api import load_cube

    cube = load_cube(MUSE_CUBE)

    print(f"Instrument: {cube.instrument}")
    print(f"Wavelength range: {cube.wavelength.min():.4f} - {cube.wavelength.max():.4f}")
    print(f"Cube shape (wavelength, x, y): {cube.cube.shape}")

    assert cube.instrument == 'MUSE', \
        f"docs claim MUSE is auto-detected; got instrument={cube.instrument!r}"


# --- api_examples.rst: load + fetch HITRAN lines + fit spectral lines ---

@needs_nirspec_cube
def api_examples_fit_spectral_lines():
    """Documented example: api_examples.rst's "Load a FITS Cube" ->
    "Fetch HITRAN Molecular Lines" -> "Fit Spectral Lines" walkthrough.
    Replaces the old cube_explorer.gaussian_fitter.fit_hitran_lines import
    (that function no longer exists) with cube_explorer.api.fit_hitran_lines,
    which fetches from HITRAN and fits in one call.

    Args:
        None

    Returns:
        None
    """
    from cube_explorer.data_loader import CubeData
    from cube_explorer.line_catalog import fetch_hitran_data
    from cube_explorer.api import fit_hitran_lines

    cube = CubeData()
    cube.load_from_file(NIRSPEC_CUBE)

    wavelength = cube.wavelength
    spectrum = cube.spectrum
    full_cube = cube.cube
    print(f"Wavelength range: {wavelength.min():.4f} - {wavelength.max():.4f}")
    print(f"Cube shape: {full_cube.shape}")

    # Fetch HITRAN Molecular Lines -- line inspection only, not consumed below
    hitran_lines, _ = fetch_hitran_data(
        wavelength_min=4.7,
        wavelength_max=4.8,
        molecule='CO',
        isotopologue=1,
        v_upper=1,
        v_lower=0
    )
    print(f"Found {len(hitran_lines)} lines")

    # Fit Spectral Lines -- fetches its own HITRAN lines internally
    fit_results = fit_hitran_lines(
        wavelength, spectrum,
        molecule='CO',
        wavelength_min=4.7,
        wavelength_max=4.8,
        isotopologue=1,
        v_upper=1,
        v_lower=0,
        header=cube.header,
        wave_unit=cube.wave_unit,
    )
    detected = [r for r in fit_results if r['success']]
    print(f"Successfully fit {len(detected)} lines")

    assert len(hitran_lines) == 10, f"docs' worked example expects 10 CO v=1->0 lines in 4.7-4.8um, got {len(hitran_lines)}"
    assert len(detected) == len(hitran_lines), \
        f"expected every fetched line to fit successfully; got {len(detected)}/{len(hitran_lines)}"


# --- 1D spectrum fitting: fitting one line ---

@needs_cube
def fitting_one_line():
    """Documented example: fit a single Gaussian line with fit_line().

    Args:
        None

    Returns:
        None
    """
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


# --- 1D spectrum fitting: result fields ---

@needs_cube
def result_fields():
    """Documented example: read every documented field of a fit_line() result.

    Args:
        None

    Returns:
        None
    """
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


# --- Pixel-by-pixel cube fitting: fitting the cube ---

@needs_cube
def cube_fitting_the_cube():
    """Documented example: a real full-cube fit (~1-2 minutes), kept because
    the docs page quotes concrete detection counts that would otherwise
    drift unnoticed from the code.

    Args:
        None

    Returns:
        None
    """
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


# --- Pixel-by-pixel cube fitting: returned keys ---

@needs_cube
def cube_returned_keys():
    """Documented example: reuses the previous test's fit (the snippet here
    is the key list, not the call) so a second three-minute full-cube run
    isn't needed to re-derive maps already computed.

    Args:
        None

    Returns:
        None
    """
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


# --- Flux: single spectrum / full map API ---

def _build_co_catalog_ecsv():
    """Build a real HITRAN-derived CO catalog ECSV, since extract_flux/
    create_flux_map need one on disk and the docs only show a placeholder
    path, in the column format load_CO_line_catalog() actually expects.

    Args:
        None

    Returns:
        path (str): Path to the written ECSV catalog file.
    """
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
def flux_single_spectrum_api():
    """Documented example: extract total flux from a summed spectrum with
    extract_flux() using a real HITRAN-derived CO catalog.

    Args:
        None

    Returns:
        None
    """
    import numpy as np
    from cube_explorer.api import load_cube, extract_flux

    cube = load_cube(NIRSPEC_CUBE)
    catalog_path = _build_co_catalog_ecsv()

    flux_result = extract_flux(
        wavelength=cube.wavelength,
        spectrum=cube.spectrum,
        target_catalog_path=catalog_path,
        header=cube.header,      # Required for the erg/s/cm2 conversion
        wave_unit=cube.wave_unit,
    )

    print(f"Total flux: {flux_result['flux']:.4e} {flux_result['flux_unit']}")
    print(f"Number of lines detected: {len(flux_result['detected_lines'])}")
    print(f"Mean continuum level: {flux_result['mean_continuum']:.3f}")

    assert flux_result['flux_unit'] == 'erg/s/cm2', \
        f"docs claim erg/s/cm2 when header/wave_unit are given; got {flux_result['flux_unit']!r}"
    assert np.isfinite(flux_result['flux']) and flux_result['flux'] > 0, \
        f"expected a positive finite flux, got {flux_result['flux']!r}"
    assert len(flux_result['detected_lines']) > 0, "expected at least some detected lines"


@needs_nirspec_cube
def flux_map_api():
    """Documented example: build a per-pixel flux map with create_flux_map(),
    using HITRAN automatically (no target_catalog_path given).

    Args:
        None

    Returns:
        None
    """
    import numpy as np
    from cube_explorer.api import load_cube, create_flux_map

    cube = load_cube(NIRSPEC_CUBE)

    # Uses HITRAN automatically by default, or pass target_catalog_path for a custom catalog
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

    assert flux_map_result['flux_unit'] == 'erg/s/cm2', \
        f"docs claim erg/s/cm2 for this NIRSpec cube; got {flux_map_result['flux_unit']!r}"
    assert flux_map_result['wave_range'] == (4.70, 4.80), \
        f"wavelength_min/max should be forwarded as the fit's wave_range; got {flux_map_result['wave_range']}"
    n_valid = (~np.isnan(flux_map_result['flux_map'])).sum()
    assert n_valid > 0, "expected at least some valid (non-NaN) pixels"

    # docs claim create_flux_map now always returns these two additional maps too
    for key in ('contam_flux_map', 'contam_detection_map'):
        assert key in flux_map_result, f"docs claim create_flux_map always returns {key!r}"
        assert flux_map_result[key].shape == flux_map_result['flux_map'].shape, \
            f"docs claim {key!r} is the same shape as flux_map"


# --- Rotational diagram ---

def rotational_diagram_docstring():
    """Documented example: fit an H2 v=0->0 S(1)-S(7) rotational diagram.
    No FITS fixture needed, so this runs unconditionally.

    Args:
        None

    Returns:
        None
    """
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
    """Run every documented example and print a pass/fail/skip summary.

    Args:
        None

    Returns:
        None
    """
    print("\n" + "=" * 70)
    print(" IFU ANALYZER -- DOCUMENTATION API TEST SUITE")
    print("=" * 70)

    run("BASIC DATA LOADING", basic_data_loading)
    run("ACCESS CUBE COMPONENTS", access_cube_components)
    run("AVAILABLE CUBE PROPERTIES", available_cube_properties)
    run("MUSE INSTRUMENT DETECTION", muse_instrument_detection)
    run("API EXAMPLES -- FIT SPECTRAL LINES", api_examples_fit_spectral_lines)
    run("1D FITTING -- ONE LINE", fitting_one_line)
    run("1D FITTING -- RESULT FIELDS", result_fields)
    run("CUBE FITTING -- FULL CUBE (slow, ~1-2 min)", cube_fitting_the_cube)
    run("CUBE FITTING -- RETURNED KEYS", cube_returned_keys)
    run("FLUX -- SINGLE SPECTRUM API", flux_single_spectrum_api)
    run("FLUX -- FULL MAP API (slow, ~2-3 min)", flux_map_api)
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
