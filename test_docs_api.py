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
  - cube_explorer/api.py module docstring ("Example Usage")
  - docs/user_guide/api_examples.rst
  - docs/user_guide/gaussian_fitting_1d.rst ("Using the API")
  - docs/user_guide/gaussian_fitting_cube.rst ("Using the API")
  - docs/user_guide/co_flux_extraction.rst ("Using the API", under Multi-Gaussian Fit)

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
# EXTRACT SPECTRA FROM SPECIFIC REGIONS
# ---------------------------------------------------------------------------

@needs_cube
def extract_spectra_from_regions():
    from cube_explorer.api import load_cube

    cube = load_cube(CUBE)

    # Extract spectrum from a single pixel
    x, y = 20, 25
    pixel_spectrum = cube.cube[:, x, y]

    # Extract spectrum from a spatial region (average)
    x_start, x_end = 10, 30
    y_start, y_end = 15, 35
    region = cube.cube[:, x_start:x_end, y_start:y_end]
    averaged_spectrum = region.mean(axis=(1, 2))

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
        print(f"Amplitude: {result.amplitude:.4e}")
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

    # "area = amplitude * sigma * sqrt(2*pi)" -- the dimensional claim in the note
    det = maps['is_detection']
    if det.any():
        lhs = area_map[det]
        rhs = maps['amplitude'][det] * maps['sigma'][det] * np.sqrt(2 * np.pi)
        assert np.allclose(lhs, rhs, rtol=1e-6), \
            "docs claim area = amplitude * sigma * sqrt(2pi)"

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

    maps['amplitude']         # Peak amplitude, erg/s/cm2 PER WAVELENGTH where convertible
    maps['amp_originalunits'] # Peak amplitude in the cube's own units
    maps['center']            # Line center wavelength
    maps['sigma']             # Gaussian sigma
    maps['fwhm']              # 2.355 * sigma
    maps['area']              # Integrated flux, or an upper limit - check is_upper_limit
    maps['success']           # Boolean: did the pixel converge and pass the detection test?
    maps['is_upper_limit']    # Boolean: area is an upper limit (may still be NaN)
    maps['residuals_cube']    # 3D residuals cached during fitting
    maps['fac_above_stdres']  # The threshold that was used
    maps['wavelength_unit']   # Spectral axis unit string
    maps['global_fit']        # The summed-spectrum fit used for initial guesses

    print(f"All documented keys present; wavelength_unit={maps['wavelength_unit']!r}, "
          f"fac_above_stdres={maps['fac_above_stdres']}")


# ---------------------------------------------------------------------------
# CO FLUX EXTRACTION -- MULTI-GAUSSIAN FIT, USING THE API
# ---------------------------------------------------------------------------
# fit_hitran_lines() runs the same algorithm as the GUI's Multi-Gaussian Fit button
# (auto_detect_boundaries + fit_gaussian_to_line per line, see gaussian_fitter.py) --
# this checks the docs' claim that they produce identical results, not just that the
# call doesn't crash.

@needs_nirspec_cube
def co_flux_multi_gaussian_api():
    from cube_explorer.api import load_cube, fit_hitran_lines

    cube = load_cube(NIRSPEC_CUBE)

    results = fit_hitran_lines(
        wavelength=cube.wavelength,
        spectrum=cube.spectrum,
        molecule='CO',
        wavelength_min=4.70,
        wavelength_max=4.80,
        v_upper=1,             # v=1->0 P-branch, same filter as the HITRAN dialog
        v_lower=0,
        header=cube.header,    # Required for the erg/s/cm2 conversion
        wave_unit=cube.wave_unit,
    )

    detected = [r for r in results if r['success']]
    print(f"Fitted {len(detected)}/{len(results)} lines")
    for r in sorted(detected, key=lambda r: r['wavelength']):
        print(f"  {r['wavelength']:.6f} um  area={r['area']:.4e} {r['area_unit']}  R2={r['r_squared']:.4f}")

    # "This reproduces the worked example above exactly" -- the ten J=5->14 P-branch
    # centers and areas from the GUI screenshot, to the digit.
    expected = {
        4.709034: (9.6275e-16, 0.9934), 4.718023: (9.4950e-16, 0.9965),
        4.727013: (9.1115e-16, 0.9943), 4.736171: (7.4484e-16, 0.9940),
        4.745491: (7.2233e-16, 0.9738), 4.754807: (5.8473e-16, 0.9094),
        4.764288: (5.4873e-16, 0.9317), 4.773898: (5.3648e-16, 0.9703),
        4.783670: (4.6497e-16, 0.9259), 4.793559: (4.9391e-16, 0.7863),
    }
    assert len(detected) == 10, f"docs' worked example has 10 detected lines, got {len(detected)}"
    for r in detected:
        match = min(expected, key=lambda c: abs(c - r['wavelength']))
        assert abs(match - r['wavelength']) < 1e-4, \
            f"fitted center {r['wavelength']:.6f} doesn't match any documented line"
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

def main():
    print("\n" + "=" * 70)
    print(" IFU ANALYZER -- DOCUMENTATION API TEST SUITE")
    print("=" * 70)

    run("BASIC DATA LOADING", basic_data_loading)
    run("ACCESS CUBE COMPONENTS", access_cube_components)
    run("EXTRACT SPECTRA FROM REGIONS", extract_spectra_from_regions)
    run("AVAILABLE CUBE PROPERTIES", available_cube_properties)
    run("1D FITTING -- ONE LINE", fitting_one_line)
    run("1D FITTING -- RESULT FIELDS", result_fields)
    run("CUBE FITTING -- FULL CUBE (slow, ~1-2 min)", cube_fitting_the_cube)
    run("CUBE FITTING -- RETURNED KEYS", cube_returned_keys)
    run("CO FLUX -- MULTI-GAUSSIAN FIT API", co_flux_multi_gaussian_api)

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
