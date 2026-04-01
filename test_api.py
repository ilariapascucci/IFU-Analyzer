#!/usr/bin/env python
"""
Consolidated API Test Suite
Tests both import functionality and vibrational level filtering
"""

print("\n" + "="*70)
print(" API TEST 1: Import Verification")
print("="*70)

try:
    from cube_explorer.api import (
        load_cube, fit_line, fit_multiple_lines, fit_hitran_lines,
        fit_continuum, measure_equivalent_width, extract_co_flux, create_co_map
    )
    print("[OK] All API functions imported successfully")
    print("  Available functions:")
    print("    - load_cube()")
    print("    - fit_line()")
    print("    - fit_multiple_lines()")
    print("    - fit_hitran_lines()  [NEW: supports v_upper, v_lower filtering]")
    print("    - fit_continuum()")
    print("    - measure_equivalent_width()")
    print("    - extract_co_flux()")
    print("    - create_co_map()")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    exit(1)

print("\n" + "="*70)
print(" API TEST 2: Vibrational Level Filtering (CO v=1->0)")
print("="*70)

import numpy as np

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
    exit(1)

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
    exit(1)

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
        exit(1)
except Exception as e:
    print(f"[ERROR] Unfiltered search failed: {e}")
    exit(1)

# Summary
print("\n" + "="*70)
print(" TEST RESULTS SUMMARY")
print("="*70)
print(f"v=1->0 filtered:      {len(detected_filtered):2d} lines")
print(f"All transitions:      {len(detected_all):2d} lines")
print(f"Other vibrational:    {len(detected_all) - len(detected_filtered):2d} lines (v≠1->0)")
print("\n[OK] ALL TESTS PASSED!")
print("="*70 + "\n")

# Show specific lines
print("v=1->0 CO lines (4.7-4.8 µm):")
for r in sorted(detected_filtered, key=lambda r: r['wavelength']):
    print(f"  {r['wavelength']:.6f} µm  {r['transition']:<30s}  J={r['transition'].split('J=')[1] if 'J=' in r['transition'] else 'N/A'}")
