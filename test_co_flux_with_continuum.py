"""
Test CO Flux Extraction with Automatic Continuum Subtraction

Preset parameters:
- Wavelength range: 4.5 to 4.8 μm
- Molecule: CO (12C16O)
- ISO: 1
- Transition: v_upper=1, v_lower=0

This script demonstrates:
1. Loading a cube and extracting a single pixel spectrum
2. Fetching HITRAN CO lines with specific transition filter
3. Applying automatic continuum subtraction (median filter + peak detection)
4. Extracting CO flux with fitted gaussians
5. Visualizing all steps with plots
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

# Import cube explorer modules
from cube_explorer.data_loader import CubeData
from cube_explorer.co_flux_extractor import fit_continuum_median_peaks, extract_co_flux_from_spectrum

# =============================================================================
# PRESET PARAMETERS
# =============================================================================
WAVE_MIN = 4.5  # μm
WAVE_MAX = 4.8  # μm
MOLECULE = 'CO'
ISO_ID = 1  # 12C16O
V_UPPER = 1  # Transition upper vibrational level
V_LOWER = 0  # Transition lower vibrational level

# Default cube file (can override with command line)
DEFAULT_CUBE = r"data\NIRSpec\jw01621-o001_t001_nirspec_g395h-f290lp_s3d.fits"
PIXEL_X = 20
PIXEL_Y = 20

# =============================================================================
# STEP 1: LOAD CUBE
# =============================================================================
print("="*70)
print("CO FLUX EXTRACTION TEST WITH AUTOMATIC CONTINUUM")
print("="*70)

cube_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CUBE
print(f"\nLoading cube: {cube_file}")

cube_data = CubeData()
cube_data.load_from_file(cube_file)
print(f"✓ Cube loaded: {cube_data.cube.shape} (spectral, y, x)")

# =============================================================================
# STEP 2: FETCH HITRAN CO LINES
# =============================================================================
print(f"\nFetching HITRAN CO lines...")
print(f"  Wavelength range: {WAVE_MIN} - {WAVE_MAX} μm")
print(f"  Molecule: {MOLECULE} (ISO {ISO_ID})")
print(f"  Transition filter: v={V_UPPER}→{V_LOWER}")

try:
    from hapi import fetch, tableList, getColumns
    
    # Determine molecule ID (CO = 5)
    M_ID = 5
    
    # Convert wavelength range from μm to cm⁻¹
    nu_min = 1e4 / WAVE_MAX  # cm⁻¹
    nu_max = 1e4 / WAVE_MIN  # cm⁻¹
    
    # Fetch from HITRAN
    table_name = f'CO_iso{ISO_ID}_v{V_UPPER}to{V_LOWER}'
    
    print(f"  Fetching from HITRAN (ν = {nu_min:.1f} - {nu_max:.1f} cm⁻¹)...")
    fetch(
        table_name,
        M_ID,
        ISO_ID,
        nu_min,
        nu_max
    )
    
    # Load fetched data - getColumns returns a list of arrays when multiple columns requested
    data = getColumns(table_name, ['molec_id', 'local_iso_id', 'nu', 'sw', 'gamma_air', 'elower'])
    molec_id, local_iso_id, nu, sw, gamma_air, elower = data
    
    # Convert to pandas DataFrame
    CO_lines_hitran = pd.DataFrame({
        'molec_id': molec_id,
        'local_iso_id': local_iso_id,
        'nu': nu,  # cm⁻¹
        'line_wave_um': 1e4 / nu,  # Convert to μm
        'sw': sw,
        'gamma_air': gamma_air,
        'elower': elower,
        'species': '12C16O'
    })
    
    print(f"✓ Fetched {len(CO_lines_hitran)} CO lines from HITRAN")
    
    # Load contaminating lines catalog
    contam_file = r"data/CO/ContaminatingLines.csv"
    try:
        contam_lines = pd.read_csv(contam_file)
        print(f"✓ Loaded {len(contam_lines)} contaminating lines")
    except FileNotFoundError:
        print("  Warning: Contaminating lines file not found, using empty catalog")
        contam_lines = pd.DataFrame(columns=['line_wave_um'])
    
except ImportError:
    print("  ERROR: HAPI not installed, using fallback catalog")
    from cube_explorer.line_catalog import load_CO_line_catalog
    CO_lines_hitran = load_CO_line_catalog(r"data/CO/CO_comprehensive_linelist_with_g.ecsv")
    # Filter by wavelength
    CO_lines_hitran = CO_lines_hitran[
        (CO_lines_hitran['line_wave_um'] >= WAVE_MIN) & 
        (CO_lines_hitran['line_wave_um'] <= WAVE_MAX)
    ]
    contam_lines = pd.DataFrame(columns=['line_wave_um'])
    print(f"✓ Loaded {len(CO_lines_hitran)} CO lines from local catalog")

# =============================================================================
# STEP 3: EXTRACT PIXEL SPECTRUM
# =============================================================================
print(f"\nExtracting spectrum from pixel ({PIXEL_X}, {PIXEL_Y})...")

spectrum_wave = cube_data.wavelength
if hasattr(spectrum_wave, 'value'):
    spectrum_wave = spectrum_wave.value

spectrum_flux = cube_data.get_pixel_spectrum(PIXEL_X, PIXEL_Y)

print(f"✓ Spectrum extracted: {spectrum_flux.shape[0]} wavelength points")
print(f"  Valid values: {np.sum(~np.isnan(spectrum_flux))}")
print(f"  Flux range: {np.nanmin(spectrum_flux):.4e} to {np.nanmax(spectrum_flux):.4e}")

# =============================================================================
# STEP 4: APPLY AUTOMATIC CONTINUUM SUBTRACTION
# =============================================================================
print(f"\nApplying automatic continuum subtraction...")
print("  Method: Median filter + peak detection + spline smoothing")

continuum = fit_continuum_median_peaks(
    spectrum_wave, spectrum_flux,
    window_size=51,
    sigma_threshold=2.0,
    peak_width=1,
    peak_distance=3,
    peak_exclusion=3,
    iterations=2,
    smoothing_factor=5
)

if continuum is not None and not np.all(np.isnan(continuum)):
    spectrum_flux_subtracted = spectrum_flux - continuum
    print(f"✓ Continuum fitted and subtracted")
    print(f"  Mean continuum level: {np.nanmean(continuum):.4e}")
    continuum_applied = True
else:
    spectrum_flux_subtracted = spectrum_flux
    continuum = np.zeros_like(spectrum_flux)
    print("  Warning: Continuum fitting failed, using original spectrum")
    continuum_applied = False

# =============================================================================
# STEP 5: EXTRACT CO FLUX
# =============================================================================
print(f"\nExtracting CO flux...")

result = extract_co_flux_from_spectrum(
    spectrum_wave,
    spectrum_flux,  # Use original flux (function will subtract continuum internally)
    CO_lines_hitran,
    contam_lines,
    wave_min=WAVE_MIN,
    wave_max=WAVE_MAX,
    do_duplicate_removal=True,
    subtract_continuum=True,  # Enable continuum subtraction
    continuum_poly_deg=3
)

# Get flux unit from FITS
bunit = cube_data.header.get('BUNIT', 'MJy/sr')
flux_unit = f"{bunit} × μm"

# =============================================================================
# STEP 6: ANALYZE RESULTS
# =============================================================================
detected_lines = result['detected_lines']
n_total_detected = len(detected_lines)
n_co_detected = 0
n_contam_detected = 0

# Calculate noise level for comparison
from cube_explorer.gaussian_fitter import compute_noise_std
if continuum_applied:
    noise_std = compute_noise_std(spectrum_wave, spectrum_flux_subtracted, pdeg=1)
else:
    noise_std = compute_noise_std(spectrum_wave, spectrum_flux, pdeg=1)
if noise_std is None or not np.isfinite(noise_std):
    noise_std = np.nanmedian(np.abs(spectrum_flux - np.nanmedian(spectrum_flux))) * 1.4826

if len(detected_lines) > 0:
    # Count CO lines (disposition=1 and species='12C16O')
    co_mask = (detected_lines['disposition'] == 1) & (detected_lines['species'] == '12C16O')
    n_co_detected = np.sum(co_mask)
    
    # Check for contaminating lines
    if 'species' in detected_lines.columns:
        contam_mask = (detected_lines['disposition'] == 1) & (detected_lines['species'] != '12C16O')
        n_contam_detected = np.sum(contam_mask)

print(f"\n[DEBUG] Line Detection Summary:")
print(f"  Noise level (std): {noise_std:.4e} {bunit}")
print(f"  SNR threshold: 3.0 sigma")
print(f"  Detection threshold: {3.0 * noise_std:.4e} {bunit}")
print(f"  Total lines detected: {n_total_detected}")
print(f"  CO lines (12C16O): {n_co_detected}")
print(f"  Contaminating lines: {n_contam_detected}")

# Check contaminating lines in wavelength range
if len(contam_lines) > 0 and 'line_wave_um' in contam_lines.columns:
    contam_in_range = contam_lines[
        (contam_lines['line_wave_um'] >= WAVE_MIN) & 
        (contam_lines['line_wave_um'] <= WAVE_MAX)
    ]
    print(f"  Contaminating lines in range ({WAVE_MIN}-{WAVE_MAX} μm): {len(contam_in_range)}")
else:
    print(f"  No contaminating line catalog loaded")

# =============================================================================
# STEP 7: PRINT RESULTS
# =============================================================================
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"CO Flux:            {result['co_flux']:.6e} {flux_unit}")
print(f"Contaminant Flux:   {result['contam_flux']:.6e} {flux_unit}")

# Check if fluxes are identical (suspicious)
if abs(result['co_flux'] - result['contam_flux']) < 1e-10:
    print("  ⚠️  WARNING: CO and Contaminant fluxes are identical!")
    print("  → This likely means no contaminating lines were fitted")

print(f"Mean Continuum:     {result['mean_continuum']:.6e} {bunit}")

if n_co_detected > 0:
    print(f"Detected CO Lines:  {n_co_detected}")
else:
    print(f"Detected CO Lines:  0")

if n_contam_detected > 0:
    print(f"Detected Contam:    {n_contam_detected}")
    
print(f"Continuum Applied:  {'✓ Yes' if continuum_applied else '✗ No'}")
print("="*70)

# Validation
if np.isfinite(result['co_flux']) and result['co_flux'] > 0:
    print("\n✓ SUCCESS: Valid positive CO flux extracted!")
elif result['co_flux'] == 0:
    print("\n⚠ WARNING: CO flux is exactly 0 (no lines detected/fitted)")
else:
    print(f"\n❌ ERROR: Invalid CO flux: {result['co_flux']}")

# =============================================================================
# STEP 7: VISUALIZATION
# =============================================================================
print("\nGenerating visualization...")

fig = plt.figure(figsize=(16, 10))

# Panel 1: Original spectrum with continuum
ax1 = plt.subplot(3, 1, 1)
ax1.plot(spectrum_wave, spectrum_flux, 'k-', linewidth=1.5, label='Original Spectrum', alpha=0.7)
if continuum_applied:
    ax1.plot(spectrum_wave, continuum, 'orange', linewidth=2, label='Fitted Continuum', linestyle='--')
ax1.axvspan(WAVE_MIN, WAVE_MAX, alpha=0.15, color='blue', label='Analysis Range')
ax1.set_ylabel(f'Flux ({bunit})', fontsize=12, fontweight='bold')
ax1.set_title(f'Step 1: Original Spectrum | Pixel ({PIXEL_X}, {PIXEL_Y})', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(spectrum_wave.min(), spectrum_wave.max())

# Panel 2: Continuum-subtracted spectrum with detected CO lines
ax2 = plt.subplot(3, 1, 2)
ax2.plot(spectrum_wave, spectrum_flux_subtracted, 'b-', linewidth=1.5, label='Continuum-Subtracted', alpha=0.8)
ax2.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax2.axvspan(WAVE_MIN, WAVE_MAX, alpha=0.15, color='blue')

# Mark detected CO lines
if len(detected_lines) > 0:
    co_lines = detected_lines[detected_lines['disposition'] == 1]
    for idx, line in co_lines.iterrows():
        wave = line['line']
        ax2.axvline(wave, color='red', alpha=0.4, linewidth=1.2, linestyle=':')
    ax2.plot([], [], 'r:', linewidth=1.2, label=f'{len(co_lines)} CO Lines Detected')

ax2.set_ylabel(f'Flux ({bunit})', fontsize=12, fontweight='bold')
ax2.set_title('Step 2: Continuum-Subtracted Spectrum with Detected CO Lines', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(spectrum_wave.min(), spectrum_wave.max())

# Panel 3: Zoomed view of CO line region
ax3 = plt.subplot(3, 1, 3)

# Filter to analysis range
mask = (spectrum_wave >= WAVE_MIN) & (spectrum_wave <= WAVE_MAX)
wave_zoom = spectrum_wave[mask]
flux_zoom = spectrum_flux_subtracted[mask]

ax3.plot(wave_zoom, flux_zoom, 'b-', linewidth=2, label='Continuum-Subtracted', alpha=0.8)
ax3.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

# Mark CO lines in zoomed region
if len(detected_lines) > 0:
    co_lines = detected_lines[detected_lines['disposition'] == 1]
    for idx, line in co_lines.iterrows():
        wave = line['line']
        if WAVE_MIN <= wave <= WAVE_MAX:
            ax3.axvline(wave, color='red', alpha=0.6, linewidth=2, linestyle=':')
            # Label some lines
            if idx < 5:  # Label first 5 lines only
                ax3.text(wave, ax3.get_ylim()[1]*0.9, f'{wave:.4f}', 
                        rotation=90, fontsize=8, ha='right', va='top', color='red', alpha=0.8)

ax3.set_xlabel('Wavelength (μm)', fontsize=12, fontweight='bold')
ax3.set_ylabel(f'Flux ({bunit})', fontsize=12, fontweight='bold')
ax3.set_title(f'Step 3: Zoomed CO Region ({WAVE_MIN} - {WAVE_MAX} μm)', fontsize=13, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(WAVE_MIN, WAVE_MAX)

plt.tight_layout()
plt.savefig('co_flux_test_with_continuum.png', dpi=150, bbox_inches='tight')
print("✓ Saved plot: co_flux_test_with_continuum.png")

plt.show()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
