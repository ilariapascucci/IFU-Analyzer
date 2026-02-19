"""
Quick test: Extract CO flux from a single spectrum to verify operations work correctly
"""
import sys
import numpy as np
import pandas as pd
from cube_explorer.data_loader import CubeData
from cube_explorer.line_catalog import load_CO_line_catalog, load_contaminating_lines_catalog
from cube_explorer.co_flux_extractor import extract_co_flux_from_spectrum

# Get cube file from command line or use default
if len(sys.argv) > 1:
    cube_file = sys.argv[1]
else:
    cube_file = r"data/MIRI_CITau/Level3_ch1-short_s3d.fits"

# Load cube
print("Loading cube...")
cube_file = r"data/MIRI_CITau/Level3_ch1-short_s3d.fits"
cube_data = CubeData()
cube_data.load_from_file(cube_file)
print(f"✓ Cube loaded: {cube_data.cube.shape}")

# Load catalogs
print("\nLoading catalogs...")
CO_lines = load_CO_line_catalog(r"data/CO/CO_comprehensive_linelist_with_g.ecsv")
contam_lines = load_contaminating_lines_catalog(r"data/CO/ContaminatingLines.csv")
print(f"✓ CO lines: {len(CO_lines)}")
print(f"✓ Contaminant lines: {len(contam_lines)}")

# Get one spectrum
print("\nExtracting spectrum from pixel (20, 20)...")
spectrum_wave = cube_data.wavelength
if hasattr(spectrum_wave, 'unit'):
    spectrum_wave = spectrum_wave.value
spectrum_flux = cube_data.get_pixel_spectrum(20, 20)
print(f"✓ Spectrum shape: {spectrum_flux.shape}")
print(f"  Valid values: {np.sum(~np.isnan(spectrum_flux))}")
print(f"  Min/Max flux: {np.nanmin(spectrum_flux):.4e} / {np.nanmax(spectrum_flux):.4e}")

# Extract CO flux with wavelength constraints
print("\nExtracting CO flux...")
wave_min = CO_lines['line_wave_um'].min()
wave_max = CO_lines['line_wave_um'].max()
print(f"CO lines wavelength range: {wave_min:.4f} - {wave_max:.4f} μm")

result = extract_co_flux_from_spectrum(
    spectrum_wave, spectrum_flux, CO_lines, contam_lines,
    wave_min=wave_min, wave_max=wave_max,
    do_duplicate_removal=True
)

# Get units from FITS header
bunit = cube_data.header.get('BUNIT', 'MJy/sr')  # Standard MIRI unit
flux_unit = f"{bunit} × μm" if bunit else "MJy/sr × μm"

# Print results
print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"CO Flux: {result['co_flux']:.6e} {flux_unit}")
print(f"Contaminent Flux: {result['contam_flux']:.6e} {flux_unit}")
print(f"Mean Continuum: {result['mean_continuum']:.6e} {bunit}")
print(f"Detected lines: {len(result['detected_lines'])}")
n_co = len(result['detected_lines'][result['detected_lines']['disposition'] == 1])
print(f"Detected CO lines (disposition=1): {n_co}")
residuals = result['residuals']['residual'].values
print(f"Residuals std: {np.std(residuals):.4e}")
print(f"Residuals mean: {np.mean(residuals):.4e}")
print(f"Residuals min/max: {np.min(residuals):.4e} / {np.max(residuals):.4e}")
print("="*60)

# Check if flux is valid
if np.isfinite(result['co_flux']) and result['co_flux'] > 0:
    print("✓ SUCCESS: Got valid, positive CO flux!")
elif result['co_flux'] == 0:
    print("⚠ WARNING: CO flux is exactly 0 (no lines fitted?)")
else:
    print(f"❌ ERROR: CO flux is invalid: {result['co_flux']}")
