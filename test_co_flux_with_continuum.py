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
# DIAGNOSTICS FUNCTION
# =============================================================================
def diagnose_pixel_extraction(spectrum_wave, spectrum_flux, spectrum_subtracted, continuum, 
                              co_lines, contam_lines, detected_result, noise_std,
                              wave_min, wave_max, verbose=True):
    """Detailed diagnostics of what happened during extraction"""
    from scipy.signal import find_peaks
    
    if not verbose:
        return
    
    print("\n" + "="*70)
    print("DETAILED PEAK DETECTION DIAGNOSTICS")
    print("="*70)
    
    # Extract wavelength range of interest
    mask = (spectrum_wave >= wave_min) & (spectrum_wave <= wave_max)
    wave_roi = spectrum_wave[mask]
    flux_roi = spectrum_subtracted[mask]
    
    # Find all local peaks in the residuals
    peaks_idx, peaks_props = find_peaks(flux_roi, height=noise_std * 1.0, distance=3)
    
    print(f"\nPeak Detection:")
    print(f"  Noise std: {noise_std:.4e}")
    print(f"  Detection threshold (1σ): {1.0 * noise_std:.4e}")
    print(f"  Detection threshold (2σ): {2.0 * noise_std:.4e}")
    print(f"  Detection threshold (3σ): {3.0 * noise_std:.4e}")
    print(f"  Local peaks found (>1σ): {len(peaks_idx)}")
    
    # Find peaks at 3σ threshold
    peaks_idx_3sig, _ = find_peaks(flux_roi, height=noise_std * 3.0, distance=3)
    print(f"  Peaks above 3σ: {len(peaks_idx_3sig)}")
    
    if len(peaks_idx) > 0:
        print(f"\n  All detected peaks ({len(peaks_idx)} total):")
        print(f"  {'#':<4} {'Wave (μm)':<12} {'Height':<12} {'σ (noise)':<12} {'Match CO':<10}")
        print(f"  {'-'*62}")
        
        for i, pidx in enumerate(peaks_idx):
            peak_wave = wave_roi[pidx]
            peak_height = flux_roi[pidx]
            peak_sigma = peak_height / noise_std if noise_std > 0 else 0
            
            # Check if matches any CO line
            co_matches = co_lines[np.abs(co_lines['line_wave_um'] - peak_wave) < 0.0005]
            match_str = f"[YES] {len(co_matches)}" if len(co_matches) > 0 else "[NO]"
            
            print(f"  {i:<4} {peak_wave:<12.5f} {peak_height:<12.4e} {peak_sigma:<12.2f} {match_str:<10}")
    
    # Show what was detected by the extraction
    detected = detected_result['detected_lines']
    print(f"\nExtraction Results:")
    print(f"  Total lines detected: {len(detected)}")
    if len(detected) > 0:
        co_detected = detected[(detected['disposition'] == 1) & (detected['species'] == '12C16O')]
        print(f"  CO lines detected: {len(co_detected)}")
        if len(co_detected) > 0:
            print(f"\n  Detected CO lines table:")
            print(f"  {'Wave (μm)':<12} {'Max Amp':<14} {'Line Flux':<14} {'Disposition':<12}")
            print(f"  {'-'*52}")
            for _, row in co_detected.iterrows():
                wave = row['line'] if 'line' in row else row.get('line_wave_um', np.nan)
                amp = row.get('max_amplitude', np.nan)
                flx = row.get('line_flux', np.nan)
                disp = row.get('disposition', 0)
                print(f"  {wave:<12.5f} {amp:<14.4e} {flx:<14.4e} {disp:<12}")

# =============================================================================
# PRESET PARAMETERS
# =============================================================================
WAVE_MIN = 4.7  # μm
WAVE_MAX = 4.8  # μm
MOLECULE = 'CO'
ISO_ID = 1  # 12C16O
V_UPPER = 1  # Transition upper vibrational level
V_LOWER = 0  # Transition lower vibrational level

# Default cube file (can override with command line)
DEFAULT_CUBE = r"data\NIRSpec\jw01621-o001_t001_nirspec_g395h-f290lp_s3d.fits"

# Region to test
X_MIN = 0
X_MAX = 50
Y_MIN = 0
Y_MAX = 50

# Enable detailed diagnostics
VERBOSE_DIAGNOSTICS = True

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
print(f"[OK] Cube loaded: {cube_data.cube.shape} (spectral, y, x)")

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
    
    print(f"[OK] Fetched {len(CO_lines_hitran)} CO lines from HITRAN")
    
    # Load contaminating lines catalog
    contam_file = r"data/CO/ContaminatingLines.csv"
    try:
        contam_lines = pd.read_csv(contam_file)
        print(f"[OK] Loaded {len(contam_lines)} contaminating lines")
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
    print(f"[OK] Loaded {len(CO_lines_hitran)} CO lines from local catalog")

# =============================================================================
# STEP 3: PROCESS PIXEL REGION
# =============================================================================
print(f"\nProcessing pixel region ({X_MIN}:{X_MAX}, {Y_MIN}:{Y_MAX})...")

spectrum_wave = cube_data.wavelength
if hasattr(spectrum_wave, 'value'):
    spectrum_wave = spectrum_wave.value

# Storage for results
n_x = X_MAX - X_MIN + 1
n_y = Y_MAX - Y_MIN + 1
co_flux_map = np.full((n_y, n_x), np.nan)
contam_flux_map = np.full((n_y, n_x), np.nan)
detected_lines_map = np.full((n_y, n_x), 0)

print(f"  Processing {n_x} × {n_y} = {n_x * n_y} pixels...")

# Get flux unit from FITS
bunit = cube_data.header.get('BUNIT', 'MJy/sr')
flux_unit = f"{bunit} × μm"

# Loop over region
for iy, py in enumerate(range(Y_MIN, Y_MAX + 1)):
    for ix, px in enumerate(range(X_MIN, X_MAX + 1)):
        # Extract spectrum
        spectrum_flux = cube_data.get_pixel_spectrum(px, py)
        
        # Extract CO flux with continuum subtraction
        result = extract_co_flux_from_spectrum(
            spectrum_wave,
            spectrum_flux,
            CO_lines_hitran,
            contam_lines,
            wave_min=WAVE_MIN,
            wave_max=WAVE_MAX,
            do_duplicate_removal=True,
            subtract_continuum=True,
            continuum_poly_deg=3
        )
        
        # Store results
        co_flux_map[iy, ix] = result['co_flux']
        contam_flux_map[iy, ix] = result['contam_flux']
        
        # Count detected lines
        if len(result['detected_lines']) > 0:
            n_detected = np.sum(result['detected_lines']['disposition'] == 1)
            detected_lines_map[iy, ix] = n_detected

print(f"[OK] Processed {n_x * n_y} pixels")

# Compute statistics
valid_mask = np.isfinite(co_flux_map) & (co_flux_map > 0)
n_valid = np.sum(valid_mask)
if n_valid > 0:
    co_mean = np.mean(co_flux_map[valid_mask])
    co_std = np.std(co_flux_map[valid_mask])
    co_median = np.median(co_flux_map[valid_mask])
    co_min = np.min(co_flux_map[valid_mask])
    co_max = np.max(co_flux_map[valid_mask])
else:
    co_mean = co_std = co_median = co_min = co_max = np.nan

print(f"\n[STATISTICS] CO Flux over region:")
print(f"  Valid pixels: {n_valid}/{n_x * n_y}")
print(f"  Mean:   {co_mean:.4e} {flux_unit}")
print(f"  Median: {co_median:.4e} {flux_unit}")
print(f"  Std:    {co_std:.4e} {flux_unit}")
print(f"  Range:  [{co_min:.4e}, {co_max:.4e}] {flux_unit}")
print(f"  Detected lines range: [{np.nanmin(detected_lines_map):.0f}, {np.nanmax(detected_lines_map):.0f}]")

# =============================================================================
# STEP 4: EXTRACT SINGLE EXAMPLE PIXEL FOR DETAILED VISUALIZATION
# =============================================================================
print(f"\nExtracting example pixel for detailed visualization...")

# Use center pixel of region
EXAMPLE_X = (X_MIN + X_MAX) // 2
EXAMPLE_Y = (Y_MIN + Y_MAX) // 2
print(f"  Example pixel: ({EXAMPLE_X}, {EXAMPLE_Y})")

spectrum_flux = cube_data.get_pixel_spectrum(EXAMPLE_X, EXAMPLE_Y)

print(f"  Spectrum: {spectrum_flux.shape[0]} wavelength points")
print(f"  Valid values: {np.sum(~np.isnan(spectrum_flux))}")
print(f"  Flux range: {np.nanmin(spectrum_flux):.4e} to {np.nanmax(spectrum_flux):.4e}")

# Apply continuum subtraction for visualization
print(f"  Applying automatic continuum subtraction...")
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
    print(f"  [OK] Continuum fitted (mean: {np.nanmean(continuum):.4e})")
    continuum_applied = True
else:
    spectrum_flux_subtracted = spectrum_flux
    continuum = np.zeros_like(spectrum_flux)
    print("  Warning: Continuum fitting failed")
    continuum_applied = False

# Extract CO flux for this pixel
result = extract_co_flux_from_spectrum(
    spectrum_wave,
    spectrum_flux,
    CO_lines_hitran,
    contam_lines,
    wave_min=WAVE_MIN,
    wave_max=WAVE_MAX,
    do_duplicate_removal=True,
    subtract_continuum=True,
    continuum_poly_deg=3
)

# =============================================================================
# STEP 5: ANALYZE EXAMPLE PIXEL RESULTS
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
# STEP 6: PRINT EXAMPLE PIXEL RESULTS
# =============================================================================
print("\n" + "="*70)
print(f"EXAMPLE PIXEL ({EXAMPLE_X}, {EXAMPLE_Y}) RESULTS")
print("="*70)
print(f"CO Flux:            {result['co_flux']:.6e} {flux_unit}")
print(f"Contaminant Flux:   {result['contam_flux']:.6e} {flux_unit}")

# Check if fluxes are identical (suspicious)
if abs(result['co_flux'] - result['contam_flux']) < 1e-10:
    print("  [WARN] WARNING: CO and Contaminant fluxes are identical!")
    print("  → This likely means no contaminating lines were fitted")

print(f"Mean Continuum:     {result['mean_continuum']:.6e} {bunit}")

if n_co_detected > 0:
    print(f"Detected CO Lines:  {n_co_detected}")
else:
    print(f"Detected CO Lines:  0")

if n_contam_detected > 0:
    print(f"Detected Contam:    {n_contam_detected}")
    
print(f"Continuum Applied:  {'[YES]' if continuum_applied else '[NO]'}")
print("="*70)

# Validation
if np.isfinite(result['co_flux']) and result['co_flux'] > 0:
    print("\n[OK] SUCCESS: Valid positive CO flux extracted!")
elif result['co_flux'] == 0:
    print("\n[WARN] WARNING: CO flux is exactly 0 (no lines detected/fitted)")
else:
    print(f"\n[ERR] ERROR: Invalid CO flux: {result['co_flux']}")

# Run detailed diagnostics
if VERBOSE_DIAGNOSTICS:
    diagnose_pixel_extraction(
        spectrum_wave, spectrum_flux, spectrum_flux_subtracted, continuum,
        CO_lines_hitran, contam_lines, result, noise_std,
        WAVE_MIN, WAVE_MAX, verbose=True
    )

# =============================================================================
# STEP 7: VISUALIZATION
# =============================================================================
print("\nGenerating visualization...")

fig = plt.figure(figsize=(20, 12))

# Panel 1: 2D CO Flux Map
ax0 = plt.subplot(2, 3, 1)
im = ax0.imshow(co_flux_map, origin='lower', cmap='viridis', aspect='auto',
                extent=[X_MIN-0.5, X_MAX+0.5, Y_MIN-0.5, Y_MAX+0.5])
ax0.set_xlabel('X pixel', fontsize=11, fontweight='bold')
ax0.set_ylabel('Y pixel', fontsize=11, fontweight='bold')
ax0.set_title(f'CO Flux Map\n({X_MIN}:{X_MAX}, {Y_MIN}:{Y_MAX})', fontsize=12, fontweight='bold')
ax0.plot(EXAMPLE_X, EXAMPLE_Y, 'r*', markersize=15, label='Example')
ax0.legend(loc='upper right', fontsize=9)
cbar0 = plt.colorbar(im, ax=ax0)
cbar0.set_label(flux_unit, fontsize=10)

# Panel 1b: Detected Lines Map
ax0b = plt.subplot(2, 3, 2)
im1b = ax0b.imshow(detected_lines_map, origin='lower', cmap='plasma', aspect='auto',
                   extent=[X_MIN-0.5, X_MAX+0.5, Y_MIN-0.5, Y_MAX+0.5])
ax0b.set_xlabel('X pixel', fontsize=11, fontweight='bold')
ax0b.set_ylabel('Y pixel', fontsize=11, fontweight='bold')
ax0b.set_title(f'Detected Lines Map\n({X_MIN}:{X_MAX}, {Y_MIN}:{Y_MAX})', fontsize=12, fontweight='bold')
ax0b.plot(EXAMPLE_X, EXAMPLE_Y, 'r*', markersize=15)
cbar1b = plt.colorbar(im1b, ax=ax0b)
cbar1b.set_label('N lines', fontsize=10)

# Panel 1c: CO Flux Histogram
ax0c = plt.subplot(2, 3, 3)
valid_fluxes = co_flux_map[valid_mask]
if len(valid_fluxes) > 0:
    ax0c.hist(valid_fluxes, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax0c.axvline(co_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {co_mean:.2e}')
    ax0c.axvline(co_median, color='orange', linestyle='--', linewidth=2, label=f'Median: {co_median:.2e}')
    ax0c.legend(loc='upper right', fontsize=9)
ax0c.set_xlabel(flux_unit, fontsize=11, fontweight='bold')
ax0c.set_ylabel('Count', fontsize=11, fontweight='bold')
ax0c.set_title(f'CO Flux Distribution\n(n={n_valid})', fontsize=12, fontweight='bold')
ax0c.grid(True, alpha=0.3)

# Panel 2: Original spectrum with continuum (example pixel)
ax1 = plt.subplot(2, 3, 4)
ax1.plot(spectrum_wave, spectrum_flux, 'k-', linewidth=1.5, label='Original Spectrum', alpha=0.7)
if continuum_applied:
    ax1.plot(spectrum_wave, continuum, 'orange', linewidth=2, label='Fitted Continuum', linestyle='--')
ax1.axvspan(WAVE_MIN, WAVE_MAX, alpha=0.15, color='blue', label='Analysis Range')
ax1.set_ylabel(f'Flux ({bunit})', fontsize=12, fontweight='bold')
ax1.set_title(f'Example: Original Spectrum | Pixel ({EXAMPLE_X}, {EXAMPLE_Y})', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(spectrum_wave.min(), spectrum_wave.max())

# Panel 3: Continuum-subtracted spectrum with detected CO lines
ax2 = plt.subplot(2, 3, 5)
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
ax2.set_title('Example: Continuum-Subtracted Spectrum', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(spectrum_wave.min(), spectrum_wave.max())

# Panel 4: Zoomed view of CO line region
ax3 = plt.subplot(2, 3, 6)

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
ax3.set_title(f'Example: Zoomed CO Region ({WAVE_MIN} - {WAVE_MAX} μm)', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(WAVE_MIN, WAVE_MAX)

plt.tight_layout()
plt.savefig('co_flux_region_test.png', dpi=150, bbox_inches='tight')
print("[OK] Saved plot: co_flux_region_test.png")

plt.show()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
