#!/usr/bin/env python
"""
Consolidated API Test Suite
Tests import functionality, vibrational level filtering, and the full-cube
Gaussian mapping / residual cube export functions.
"""

import os
import sys
import types

import numpy as np

print("\n" + "="*70)
print(" API TEST 1: Import Verification")
print("="*70)

try:
    from cube_explorer.api import (
        load_cube, fit_line, fit_hitran_lines,
        fit_continuum, measure_equivalent_width, extract_flux, create_flux_map,
        create_gaussian_map, export_residual_cube, rotational_diagram
    )
    print("[OK] All API functions imported successfully")
    print("  Available functions:")
    print("    - load_cube()")
    print("    - fit_line()")
    print("    - fit_hitran_lines()  [NEW: supports v_upper, v_lower filtering]")
    print("    - fit_continuum()")
    print("    - measure_equivalent_width()")
    print("    - extract_flux()")
    print("    - create_flux_map()")
    print("    - create_gaussian_map()")
    print("    - export_residual_cube()")
    print("    - rotational_diagram()")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print(" API TEST 2: Vibrational Level Filtering (CO v=1->0)")
print("="*70)

# Load cube and extract spectrum
print("\n[1/3] Loading NIRSpec cube...")
try:
    cube = load_cube('data/NIRSpec/jw01621-o001_t001_nirspec_g395h-f290lp_s3d.fits')
    nx, ny = cube.cube.shape[1:3]
    x_min, x_max = nx // 3, 2 * nx // 3
    y_min, y_max = ny // 3, 2 * ny // 3
    region = cube.cube[:, x_min:x_max, y_min:y_max]
    spectrum = np.nanmean(region, axis=(1, 2))

    if hasattr(spectrum, 'value'):
        spectrum = spectrum.value
    if hasattr(cube.wavelength, 'value'):
        wavelength = cube.wavelength.value
    else:
        wavelength = cube.wavelength
    print(f"[OK] Loaded {cube.instrument} data ({len(wavelength)} wavelength points)")
except Exception as e:
    print(f"[ERROR] Failed to load cube: {e}")
    sys.exit(1)

# Test 1: v=1->0 filtering
print("\n[2/3] Testing v=1->0 filtering...")
try:
    results_filtered = fit_hitran_lines(
        wavelength, spectrum,
        molecule='CO',
        wavelength_min=4.7,
        wavelength_max=4.8,
        v_upper=1,
        v_lower=0
    )
    detected_filtered = [r for r in results_filtered if r['success']]
    print(f"[OK] Found {len(detected_filtered)} v=1->0 CO lines")
    if len(detected_filtered) != 10:
        print(f"  [WARNING] Expected 10 lines, got {len(detected_filtered)}")
except Exception as e:
    print(f"[ERROR] v=1->0 filtering failed: {e}")
    sys.exit(1)

# Test 2: No filtering (all transitions)
print("\n[3/3] Testing unfiltered CO lines (all vibrational levels)...")
try:
    results_all = fit_hitran_lines(
        wavelength, spectrum,
        molecule='CO',
        wavelength_min=4.7,
        wavelength_max=4.8
    )
    detected_all = [r for r in results_all if r['success']]
    print(f"[OK] Found {len(detected_all)} total CO lines (all vibrational levels)")
    if len(detected_all) < len(detected_filtered):
        print(f"  [ERROR] Unfiltered should have >= filtered lines!")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] Unfiltered search failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print(" TEST 2 RESULTS SUMMARY")
print("="*70)
print(f"v=1->0 filtered:      {len(detected_filtered):2d} lines")
print(f"All transitions:      {len(detected_all):2d} lines")
print(f"Other vibrational:    {len(detected_all) - len(detected_filtered):2d} lines (v!=1->0)")
print("[OK] Vibrational level filtering tests passed")
print("="*70 + "\n")

# Show specific lines
print("v=1->0 CO lines (4.7-4.8 um):")
for r in sorted(detected_filtered, key=lambda r: r['wavelength']):
    print(f"  {r['wavelength']:.6f} um  {r['transition']:<30s}  J={r['transition'].split('J=')[1] if 'J=' in r['transition'] else 'N/A'}")


print("\n" + "="*70)
print(" API TEST 3: Full-Cube Gaussian Map / Residual Cube (Synthetic Data)")
print("="*70)
print("Builds a synthetic cube with a known Gaussian line (no dependency on any")
print("local FITS fixture) so results can be checked against ground truth.")

np.random.seed(0)

N_WAVE = 300
NY, NX = 4, 4
CONTINUUM = 1000.0
TRUE_AMP = 80.0
TRUE_CENTER = 4.675
TRUE_SIGMA = 0.008
NOISE_STD = 2.0

synthetic_wavelength = np.linspace(4.60, 4.75, N_WAVE)
line = TRUE_AMP * np.exp(-0.5 * ((synthetic_wavelength - TRUE_CENTER) / TRUE_SIGMA) ** 2)

synthetic_cube = np.empty((N_WAVE, NY, NX), dtype=np.float64)
for iy in range(NY):
    for ix in range(NX):
        noise = np.random.normal(0, NOISE_STD, size=N_WAVE)
        synthetic_cube[:, iy, ix] = CONTINUUM + line + noise

# Minimal stand-in for CubeData -- only the attributes the API functions read
synthetic_cube_data = types.SimpleNamespace(
    cube=synthetic_cube,
    wavelength=synthetic_wavelength,
    header=None,
    wave_unit="um",
)

print(f"[OK] Built synthetic cube: shape={synthetic_cube.shape}, "
      f"true center={TRUE_CENTER}um, true sigma={TRUE_SIGMA}um, true amp={TRUE_AMP}")

print("\n[1/3] Testing create_gaussian_map recovers known parameters...")
try:
    maps = create_gaussian_map(synthetic_cube_data, lam_min=4.60, lam_max=4.75, n_jobs=2)
    required_keys = {'amplitude', 'center', 'sigma', 'fwhm', 'area',
                     'r_squared', 'success', 'is_detection', 'is_upper_limit'}
    missing = required_keys - set(maps.keys())
    if missing:
        print(f"[ERROR] Missing keys in result: {missing}")
        sys.exit(1)
    print(f"[OK] Result contains all expected keys: {sorted(required_keys)}")
except Exception as e:
    print(f"[ERROR] create_gaussian_map raised an exception: {e}")
    sys.exit(1)

n_detections = int(np.sum(maps['is_detection']))
print(f"  Detections: {n_detections}/{NY * NX} pixels")
if n_detections != NY * NX:
    print(f"[ERROR] Expected all {NY * NX} pixels to be detections (amp={TRUE_AMP} >> noise={NOISE_STD})")
    sys.exit(1)
print(f"[OK] All {NY * NX} pixels detected as expected")

center_err = np.nanmean(np.abs(maps['center'] - TRUE_CENTER))
sigma_err = np.nanmean(np.abs(maps['sigma'] - TRUE_SIGMA))
amp_err = np.nanmean(np.abs(maps['amplitude'] - TRUE_AMP)) / TRUE_AMP

print(f"  Mean |center error|:      {center_err:.6f} um")
print(f"  Mean |sigma error|:       {sigma_err:.6f} um")
print(f"  Mean relative amp error:  {amp_err:.4f}")

if center_err > 0.002:
    print(f"[ERROR] Recovered center off by too much: {center_err:.6f} um")
    sys.exit(1)
if sigma_err > 0.003:
    print(f"[ERROR] Recovered sigma off by too much: {sigma_err:.6f} um")
    sys.exit(1)
if amp_err > 0.15:
    print(f"[ERROR] Recovered amplitude off by too much: {amp_err:.4f} relative error")
    sys.exit(1)
print("[OK] Recovered center/sigma/amplitude all within tolerance of injected truth")

print("\n[2/3] Testing export_residual_cube (return_cube=True)...")
try:
    residual = export_residual_cube(synthetic_cube_data, lam_min=4.60, lam_max=4.75,
                                     n_jobs=2, return_cube=True)
    print(f"[OK] Residual cube returned, shape={residual.shape}")
except Exception as e:
    print(f"[ERROR] export_residual_cube(return_cube=True) raised an exception: {e}")
    sys.exit(1)

if residual.shape != synthetic_cube.shape:
    print(f"[ERROR] Residual shape {residual.shape} != input cube shape {synthetic_cube.shape}")
    sys.exit(1)

finite_residuals = residual[np.isfinite(residual)]
mean_abs_resid = np.nanmean(np.abs(finite_residuals))
print(f"  Mean |normalized residual|: {mean_abs_resid:.4f}")
if mean_abs_resid > 1.5:
    print(f"[ERROR] Residuals too large for a good fit: {mean_abs_resid:.4f}")
    sys.exit(1)
print("[OK] Residuals are small, consistent with a good per-pixel fit")

print("\n[3/3] Testing export_residual_cube writes a real FITS file...")
residual_out_path = os.path.join(
    os.environ.get("TEMP", "."), "test_api_residual_cube.fits"
)
try:
    export_residual_cube(synthetic_cube_data, out_path=residual_out_path,
                          lam_min=4.60, lam_max=4.75, n_jobs=2)
    if not os.path.exists(residual_out_path):
        print(f"[ERROR] Expected output file not found: {residual_out_path}")
        sys.exit(1)
    size = os.path.getsize(residual_out_path)
    print(f"[OK] FITS file written: {residual_out_path} ({size} bytes)")
finally:
    if os.path.exists(residual_out_path):
        os.remove(residual_out_path)
        print(f"  Cleaned up {residual_out_path}")

print("\n" + "="*70)
print(" API TEST 4: Public Surface Coverage")
print("="*70)
print("Calls every function in cube_explorer.api.__all__ at least once.")
print("Importing a name proves it exists; only calling it proves it works.")
print("The function list is read from __all__, so adding a new public")
print("function without a check here shows up immediately as UNTESTED.\n")

import cube_explorer.api as _api

_SAMPLE_CUBE = os.path.join("data", "MIRI_CITau", "Level3_ch1-long_s3d.fits")


def _cube_fixture():
    """Load the sample cube once, or None if the gitignored data is absent.

    Args:
        None

    Returns:
        cube (CubeData): The loaded sample cube, or None if unavailable.
    """
    if not os.path.exists(_SAMPLE_CUBE):
        return None
    return _api.load_cube(_SAMPLE_CUBE)


_cube = _cube_fixture()
_no_data = "sample cube not available (data/ is gitignored)"


def _needs_cube(fn):
    """Wrap a check so it reports UNTESTED instead of failing without data.

    Args:
        fn (callable): Check function taking a loaded cube.

    Returns:
        wrapper (callable): No-arg function that runs fn(cube) if the
            sample cube is available, else returns the UNTESTED note.
    """
    def wrapper():
        """Run fn against the sample cube, or report it as untested.

        Args:
            None

        Returns:
            result (tuple): (message, note) -- fn's result, or (None, note)
                when the sample cube is unavailable.
        """
        if _cube is None:
            return None, _no_data
        return fn(_cube)
    return wrapper


@_needs_cube
def _check_load_cube(cube):
    """Check that load_cube() detects the instrument and cube shape.

    Args:
        cube (CubeData): Loaded sample cube.

    Returns:
        result (tuple): (message, note) for the test report.
    """
    return f"{cube.instrument}, {cube.cube.shape}", None


@_needs_cube
def _check_fit_continuum(cube):
    """Check that fit_continuum() returns a continuum array.

    Args:
        cube (CubeData): Loaded sample cube.

    Returns:
        result (tuple): (message, note) for the test report.
    """
    cont = _api.fit_continuum(cube.wavelength, cube.spectrum)
    return f"continuum of {len(cont)} points", None


# A real line in the MIRI ch1-long cube, from an actual GUI session (R^2 = 0.965).
# Also used in test_docs_api.py and docs/user_guide/gaussian_fitting_1d.rst.
LINE_X1, LINE_PEAK, LINE_X2 = 7.491492, 7.503455, 7.516985


@_needs_cube
def _check_fit_line(cube):
    """Check that fit_line() fits the reference MIRI line successfully.

    Args:
        cube (CubeData): Loaded sample cube.

    Returns:
        result (tuple): (message, note) for the test report.
    """
    r = _api.fit_line(cube.wavelength, cube.spectrum, LINE_X1, LINE_X2,
                      peak_x=LINE_PEAK, header=cube.header,
                      wave_unit=cube.wave_unit)
    return (f"success={r.success}, center={r.center:.5f}, "
            f"R2={r.r_squared:.4f}, area={r.area:.4e} {r.area_unit}"), None


@_needs_cube
def _check_measure_equivalent_width(cube):
    """Check that measure_equivalent_width() returns a value for the reference line.

    Args:
        cube (CubeData): Loaded sample cube.

    Returns:
        result (tuple): (message, note) for the test report.
    """
    fit = _api.fit_line(cube.wavelength, cube.spectrum, LINE_X1, LINE_X2,
                        peak_x=LINE_PEAK, header=cube.header,
                        wave_unit=cube.wave_unit)
    cont = _api.fit_continuum(cube.wavelength, cube.spectrum)
    ew = _api.measure_equivalent_width(fit, cube.wavelength, cont)
    return f"EW={ew}", None


def _check_create_gaussian_map():
    """Check that create_gaussian_map() returns maps for a synthetic cube.

    Args:
        None

    Returns:
        result (tuple): (message, note) for the test report.
    """
    maps = create_gaussian_map(synthetic_cube_data, lam_min=4.60, lam_max=4.75,
                               n_jobs=2)
    return f"{len(maps)} maps from synthetic cube", None


def _check_export_residual_cube():
    """Check that export_residual_cube() returns a residual cube array.

    Args:
        None

    Returns:
        result (tuple): (message, note) for the test report.
    """
    resid = export_residual_cube(synthetic_cube_data, lam_min=4.60,
                                 lam_max=4.75, n_jobs=2, return_cube=True)
    return f"residual cube {resid.shape}", None


def _check_fit_hitran_lines():
    """Report fit_hitran_lines() as covered elsewhere rather than re-testing it.

    Args:
        None

    Returns:
        result (tuple): (None, note) marking this check as covered elsewhere.
    """
    # Exercised in full by API TEST 2 above, against the NIRSpec fixture.
    return None, "covered by API TEST 2 (needs NIRSpec fixture + HITRAN cache)"


def _check_extract_flux():
    """Report extract_flux() as untested due to a missing fixture.

    Args:
        None

    Returns:
        result (tuple): (None, note) explaining why this check is skipped.
    """
    return None, "needs a CO line catalog (.ecsv) fixture"


def _check_create_flux_map():
    """Report create_flux_map() as untested due to a missing fixture.

    Args:
        None

    Returns:
        result (tuple): (None, note) explaining why this check is skipped.
    """
    return None, "needs a CO line catalog fixture and a full-cube run"


def _check_rotational_diagram():
    """Report rotational_diagram() as untested due to a missing fixture.

    Args:
        None

    Returns:
        result (tuple): (None, note) explaining why this check is skipped.
    """
    return None, "needs molecular_data + flux dict fixture"


_CHECKS = {
    "load_cube": _check_load_cube,
    "fit_continuum": _check_fit_continuum,
    "fit_line": _check_fit_line,
    "fit_hitran_lines": _check_fit_hitran_lines,
    "measure_equivalent_width": _check_measure_equivalent_width,
    "create_gaussian_map": _check_create_gaussian_map,
    "export_residual_cube": _check_export_residual_cube,
    "extract_flux": _check_extract_flux,
    "create_flux_map": _check_create_flux_map,
    "rotational_diagram": _check_rotational_diagram,
}

_results = {}

for _name in _api.__all__:
    _check = _CHECKS.get(_name)
    if _check is None:
        _results[_name] = ("UNTESTED", "no check defined")
        continue
    try:
        _detail, _skip_reason = _check()
    except Exception as exc:
        _results[_name] = ("FAIL", f"{type(exc).__name__}: {exc}")
    else:
        if _skip_reason:
            _results[_name] = ("UNTESTED", _skip_reason)
        else:
            _results[_name] = ("PASS", _detail)

print(f"  {'function':<26} {'status':<10} detail")
print("  " + "-"*68)
for _name in _api.__all__:
    _status, _detail = _results[_name]
    print(f"  {_name:<26} [{_status:<8}] {_detail}")

_passed = sum(1 for s, _ in _results.values() if s == "PASS")
_failed = [n for n, (s, _) in _results.items() if s == "FAIL"]
_untested = [n for n, (s, _) in _results.items() if s == "UNTESTED"]

print()
print(f"  Covered:  {_passed}/{len(_api.__all__)}")
print(f"  Untested: {len(_untested)}  {_untested if _untested else ''}")

if _failed:
    print(f"\n[ERROR] {len(_failed)} API function(s) failed when called: {_failed}")
    sys.exit(1)

print("\n" + "="*70)
print(" [OK] ALL TESTS PASSED!")
print("="*70 + "\n")
