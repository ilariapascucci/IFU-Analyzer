"""
Quick test: Extract CO flux from a single spectrum with visualization
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cube_explorer.data_loader import CubeData
from cube_explorer.line_catalog import load_CO_line_catalog, load_contaminating_lines_catalog
from cube_explorer.co_flux_extractor import extract_co_flux_from_spectrum

# Get cube file and pixel from command line
cube_file = sys.argv[1] if len(sys.argv) > 1 else r"data/NIRSpec/jw01621-o001_t001_nirspec_g395h-f290lp_s3d.fits"
pixel_x = int(sys.argv[2]) if len(sys.argv) > 2 else 19
pixel_y = int(sys.argv[3]) if len(sys.argv) > 3 else 19
use_plot = not (len(sys.argv) > 4 and sys.argv[4] == '--no-plot')

# Load cube
print("Loading cube...")
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
print(f"\nExtracting spectrum from pixel ({pixel_x}, {pixel_y})...")
spectrum_wave = cube_data.wavelength
if hasattr(spectrum_wave, 'unit'):
    spectrum_wave = spectrum_wave.value
spectrum_flux = cube_data.get_pixel_spectrum(pixel_x, pixel_y)
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
#print(f"Contaminent Flux: {result['contam_flux']:.6e} {flux_unit}")
#print(f"Mean Continuum: {result['mean_continuum']:.6e} {bunit}")
#print(f"Detected lines: {len(result['detected_lines'])}")
#n_co = len(result['detected_lines'][result['detected_lines']['disposition'] == 1])
#print(f"Detected CO lines (disposition=1): {n_co}")
#residuals = result['residuals']['residual'].values
#print(f"Residuals std: {np.std(residuals):.4e}")
#print(f"Residuals mean: {np.mean(residuals):.4e}")
#print(f"Residuals min/max: {np.min(residuals):.4e} / {np.max(residuals):.4e}")
print("="*60)

# Check if flux is valid
if np.isfinite(result['co_flux']) and result['co_flux'] > 0:
    print("✓ SUCCESS: Got valid, positive CO flux!")
elif result['co_flux'] == 0:
    print("⚠ WARNING: CO flux is exactly 0 (no lines fitted?)")
else:
    print(f"❌ ERROR: CO flux is invalid: {result['co_flux']}")

# Generate visualization if requested
if use_plot:
    print("\nGenerating visualization...")
    fig = plt.figure(figsize=(15, 11))
    
    # ===== PANEL 1: Full spectrum =====
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(spectrum_wave, spectrum_flux, 'k-', linewidth=1.2, alpha=0.8, label='Observed spectrum')
    
    # Mark CO wavelength range
    ax1.axvspan(wave_min, wave_max, alpha=0.1, color='red', label='CO line region')
    
    # Mark detected CO lines
    detected_lines = result['detected_lines']
    if len(detected_lines) > 0:
        co_lines = detected_lines[detected_lines['disposition'] == 1]
        for _, line in co_lines.iterrows():
            ax1.axvline(line['line'], color='red', alpha=0.25, linewidth=0.6)
    
    ax1.set_ylabel(f'Flux ({bunit})', fontsize=11, fontweight='bold')
    ax1.set_title(
        f'CO Rovibrational Flux Extraction\nPixel ({pixel_x}, {pixel_y}) | '
        f'CO Flux: {result["co_flux"]:.4e} {flux_unit} | ',
        #f'{n_co} lines detected',
        fontsize=13, fontweight='bold'
    )
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.set_xlim([spectrum_wave[0], spectrum_wave[-1]])
    
    '''
    # ===== PANEL 2: Zoomed CO region with continuum =====
    ax2 = plt.subplot(3, 1, 2)
    mask = (spectrum_wave >= wave_min) & (spectrum_wave <= wave_max)
    wave_zoom = spectrum_wave[mask]
    flux_zoom = spectrum_flux[mask]
    
    ax2.plot(wave_zoom, flux_zoom, 'k-', linewidth=1.5, label='Data', zorder=2)
    ax2.axhline(result['mean_continuum'], color='blue', linestyle='--', linewidth=2, 
                label=f'Continuum: {result["mean_continuum"]:.1f} {bunit}', alpha=0.8)
    
    # Show line detections in red
    if len(detected_lines) > 0:
        co_lines = detected_lines[detected_lines['disposition'] == 1]
        first_line = True
        for _, line in co_lines.iterrows():
            line_mask = (wave_zoom >= line['line']-0.015) & (wave_zoom <= line['line']+0.015)
            if np.any(line_mask):
                label = 'Detected CO lines' if first_line else None
                ax2.plot(wave_zoom[line_mask], flux_zoom[line_mask], 'r-', linewidth=2.5, alpha=0.8, zorder=3, label=label)
                ax2.axvline(line['line'], color='red', alpha=0.15, linewidth=1)
                first_line = False
    
    ax2.fill_between(wave_zoom, result['mean_continuum'], 0, alpha=0.05, color='blue')
    ax2.set_ylabel(f'Flux ({bunit})', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Wavelength (μm)', fontsize=11, fontweight='bold')
    ax2.set_title(f'CO Region Detail - {n_co} detected lines', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax2.set_xlim([wave_min-0.02, wave_max+0.02])
    '''
    '''
    # ===== PANEL 3: Residuals =====
    ax3 = plt.subplot(3, 1, 3)
    residuals_std = np.std(residuals)
    residuals_mean = np.mean(residuals)
    
    ax3.plot(result['residuals']['wave'].values, residuals, 'b-', linewidth=0.8, 
             label='Residuals', alpha=0.8, zorder=2)
    ax3.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
    ax3.axhline(residuals_std, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    ax3.axhline(-residuals_std, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    ax3.fill_between(result['residuals']['wave'].values, -residuals_std, residuals_std, 
                      alpha=0.1, color='red', label=f'±1σ = ±{residuals_std:.1f} {bunit}', zorder=0)
    
    ax3.set_ylabel(f'Residual ({bunit})', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Wavelength (μm)', fontsize=11, fontweight='bold')
    ax3.set_title(
        f'Fit Quality - RMS: {residuals_std:.2f} MJy/sr | '
        f'Mean: {residuals_mean:.4f} MJy/sr | '
        f'Range: [{np.min(residuals):.1f}, {np.max(residuals):.1f}]',
        fontsize=11, fontweight='bold'
    )
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    '''
    plt.tight_layout()
    
    # Save figure
    output_file = f'CO_flux_extraction_pixel_{pixel_x}_{pixel_y}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved: {output_file}")
    
    plt.show()
