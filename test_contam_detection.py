#!/usr/bin/env python
"""
Verifies that extract_flux_from_spectrum() actually detects and fits
contaminating lines (H2), instead of always finding zero -- the bug
described around detected_contam construction in flux_map_extractor.py.

Uses real data: the NIRSpec g395h-f290lp cube (which covers the CO
v=1->0 bandhead near 4.7 um), the real HITRAN CO v_upper=1/v_lower=0
line list, and the real data/ContaminatingLines.csv H2 catalog -- the
same combination the GUI's Flux Map tool would use.
"""
import sys
import numpy as np
import pandas as pd

from cube_explorer.api import load_cube
from cube_explorer.line_catalog import fetch_hitran_data, load_contaminating_lines_catalog
from cube_explorer.flux_map_extractor import extract_flux_from_spectrum

WAVE_MIN, WAVE_MAX = 4.69, 4.75

print("\n" + "=" * 70)
print(" Contaminating-line detection test (real NIRSpec data)")
print("=" * 70)

cube = load_cube("data/NIRSpec/jw01621-o001_t001_nirspec_g395h-f290lp_s3d.fits")
wave = cube.wavelength.value if hasattr(cube.wavelength, "value") else np.asarray(cube.wavelength)
mask = (wave >= WAVE_MIN) & (wave <= WAVE_MAX)
print(f"[INFO] {mask.sum()} wavelength points in {WAVE_MIN}-{WAVE_MAX} um")

# Pick the brightest spaxel in this band (most likely to show CO + H2 like the boss's plot)
band_sum = np.nansum(cube.cube[mask, :, :], axis=0)
iy, ix = np.unravel_index(np.nanargmax(band_sum), band_sum.shape)
print(f"[INFO] Using brightest spaxel in band: (y={iy}, x={ix})")

spectrum_wave = wave[mask]
spectrum_flux = cube.cube[mask, iy, ix]

# Real HITRAN CO v=1->0 line list in this range
hitran_rows, _ = fetch_hitran_data(WAVE_MIN, WAVE_MAX, "CO", v_upper=1, v_lower=0)
if not hitran_rows:
    print("[ERROR] No HITRAN CO v=1->0 lines returned -- cannot run test")
    sys.exit(1)
CO_catalog = pd.DataFrame(hitran_rows).rename(columns={"wavelength": "line_wave_um"})
CO_catalog["species"] = "CO"
print(f"[INFO] CO catalog: {len(CO_catalog)} lines in range")

# Real H2 contaminating-line catalog
contam_catalog = load_contaminating_lines_catalog("data/ContaminatingLines.csv")
if contam_catalog is None:
    print("[ERROR] Failed to load data/ContaminatingLines.csv")
    sys.exit(1)
contam_in_range = contam_catalog[
    (contam_catalog["line_wave_um"] >= WAVE_MIN) & (contam_catalog["line_wave_um"] <= WAVE_MAX)
]
print(f"[INFO] Contaminating catalog: {len(contam_in_range)} H2 lines in range "
      f"(of {len(contam_catalog)} total)")

result = extract_flux_from_spectrum(
    spectrum_wave, spectrum_flux, CO_catalog, contam_catalog,
    wave_min=WAVE_MIN, wave_max=WAVE_MAX,
    snr_threshold=3.0,
    header=cube.header, wave_unit=cube.wave_unit,
)

print("\n" + "-" * 70)
print(f"flux        = {result['flux']}")
print(f"contam_flux = {result['contam_flux']}")
print(f"flux_unit   = {result['flux_unit']}")
print("-" * 70)

if result["contam_flux"] not in (0, None) and np.isfinite(result["contam_flux"]):
    print("[OK] Contaminating line (H2) was detected and fitted (contam_flux != 0)")
    sys.exit(0)
else:
    print("[ERROR] Contaminating line was NOT fitted (contam_flux == 0) -- bug still present")
    sys.exit(1)
