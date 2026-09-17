#!/usr/bin/env python
"""
Basic IFU Analyzer Example - Using Existing GUI Functions
===========================================================

This example uses the SAME functions that work in the GUI.
No API wrapper - just direct imports of proven code.
"""

import glob
import numpy as np
import os
from astropy.io import fits

# Direct imports from GUI modules - use what already works
from cube_explorer.data_loader import CubeData
from cube_explorer.gaussian_fitter import fit_gaussian_to_line
from cube_explorer.flux_map_extractor import fit_hitran_lines_collective
from cube_explorer.line_catalog import fetch_hitran_data
from cube_explorer.continuum_dialog import ContinuumParametersDialog  # Get default params
from cube_explorer.gui_continuum import GuiContinuumMixin
from cube_explorer.visualization import plot_line_decomposition


def fit_continuum_directly(wavelength, flux):
    """Fit continuum using the GUI's exact algorithm.

    Args:
        wavelength (ndarray): Wavelength array for the spectrum.
        flux (ndarray): Flux array matching wavelength.

    Returns:
        continuum_values (ndarray): Smoothed continuum, same shape as wavelength, NaN where masked out.
    """
    from scipy.ndimage import median_filter
    from scipy.interpolate import UnivariateSpline
    from scipy.signal import find_peaks
    
    # GUI default parameters
    params = {
        'window_size': 51,
        'sigma_threshold': 2.0,
        'peak_width': 1,
        'peak_distance': 3,
        'peak_exclusion': 3,
        'iterations': 2,
        'smoothing_factor': 5
    }
    
    mask = np.isfinite(wavelength) & np.isfinite(flux)
    wavelength_clean = wavelength[mask]
    flux_clean = flux[mask]
    
    if len(wavelength_clean) < 10:
        return np.full_like(wavelength, np.nan)
    
    window_size = params['window_size']
    flux_smoothed = median_filter(flux_clean, size=window_size, mode='reflect')
    peaks, properties = find_peaks(
        flux_clean, 
        prominence=np.std(flux_clean) * params['sigma_threshold'],
        width=params['peak_width'], 
        distance=params['peak_distance']
    )
    
    exclude_mask = np.zeros(len(flux_clean), dtype=bool)
    for peak_idx in peaks:
        exclusion = params['peak_exclusion']
        left_edge = max(0, peak_idx - exclusion)
        right_edge = min(len(flux_clean), peak_idx + exclusion + 1)
        exclude_mask[left_edge:right_edge] = True
    
    good_indices = np.where(~exclude_mask)[0]
    if len(good_indices) < 10:
        exclude_mask = np.zeros(len(flux_clean), dtype=bool)
    
    flux_good = flux_clean.copy()
    flux_good[exclude_mask] = np.nan
    
    if np.any(exclude_mask):
        valid_mask = ~exclude_mask
        flux_good[exclude_mask] = np.interp(
            wavelength_clean[exclude_mask], 
            wavelength_clean[valid_mask], 
            flux_clean[valid_mask]
        )
    
    continuum = median_filter(flux_good, size=window_size, mode='reflect')
    
    for iteration in range(params['iterations']):
        residuals = flux_clean - continuum
        sigma = np.std(residuals[~exclude_mask])
        outliers = (residuals > 2.0 * sigma) & (~exclude_mask)
        if np.sum(outliers) == 0:
            break
        exclude_mask |= outliers
        flux_good = flux_clean.copy()
        valid_mask = ~exclude_mask
        flux_good[exclude_mask] = np.interp(
            wavelength_clean[exclude_mask],
            wavelength_clean[valid_mask],
            flux_clean[valid_mask]
        )
        continuum = median_filter(flux_good, size=window_size, mode='reflect')
    
    try:
        valid_mask = ~exclude_mask
        spline = UnivariateSpline(
            wavelength_clean[valid_mask],
            continuum[valid_mask],
            s=len(wavelength_clean) / params['smoothing_factor'],
            k=3
        )
        continuum_smooth = spline(wavelength_clean)
    except:
        continuum_smooth = continuum
    
    continuum_values = np.full_like(wavelength, np.nan)
    continuum_values[mask] = continuum_smooth
    return continuum_values


def main():
    """Run the end-to-end example: load a cube, extract a spectrum, and fit CO lines.

    Args:
        None

    Returns:
        None
    """
    print("\n" + "=" * 70)
    print(" IFU Analyzer - Direct GUI Function Usage")
    print("=" * 70 + "\n")

    # STEP 1: Load cube
    print("[STEP 1] Loading FITS cube...")
    filepath = "data/NIRSpec/jw01621-o001_t001_nirspec_g395h-f290lp_s3d.fits"
    
    if not os.path.exists(filepath):
        for f in sorted(glob.glob("data/**/*.fits", recursive=True)):
            try:
                with fits.open(f) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None and hdu.data.ndim >= 3 and hdu.data.shape[0] > 100:
                            filepath = f
                            break
                    if filepath != "data/NIRSpec/jw01621-o001_t001_nirspec_g395h-f290lp_s3d.fits":
                        break
            except:
                pass

    try:
        cube = CubeData()
        cube.load_from_file(filepath)
        print(f"  SUCCESS! Loaded {filepath}")
        print(f"  Wavelength range: {cube.wavelength.min():.4f} - {cube.wavelength.max():.4f} um")
        print(f"  Cube shape: {cube.cube.shape}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # STEP 2: Extract spectrum
    print("\n[STEP 2] Extracting spectrum...")
    try:
        if hasattr(cube.wavelength, 'value'):
            wavelength = cube.wavelength.value
        else:
            wavelength = cube.wavelength
        
        spectrum = cube.spectrum.copy()
        if hasattr(spectrum, 'value'):
            spectrum = spectrum.value
        
        print(f"  SUCCESS! Extracted spectrum ({len(spectrum)} points)")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

        # STEP 3: Fit CO v=1->0 lines from HITRAN 
    print("\n[STEP 3] Fitting CO v=1->0 lines from HITRAN...")
    try:
        # Fetch HITRAN data - ALREADY FILTERED by wavelength
        hitran_lines, _ = fetch_hitran_data(
            4.7, 4.8,
            'CO',
            isotopologue=1,
            v_upper=1,
            v_lower=0
        )
        
        # Test each line individually to diagnose fit issues
        print(f"\n  Testing spectrum quality at each line:")
        line_results = []
        for i, line in enumerate(hitran_lines, 1):
            wl_center = line['wavelength']
            
            # Find closest index to this wavelength
            idx = np.argmin(np.abs(wavelength - wl_center))
            flux_at_center = spectrum[idx]
            
            # Extract small window around this line (+/-20 pixels)
            window_idx = slice(max(0, idx-20), min(len(spectrum), idx+20))
            flux_window = spectrum[window_idx]
            
            # Calculate stats
            peak_flux = np.min(flux_window)  # Absorption = dip
            local_continuum = np.mean([flux_window[0], flux_window[-1]])
            depth = local_continuum - peak_flux
            
            status = "[DEEP]" if depth > local_continuum * 0.05 else "[SHALLOW]"
            print(f"    {i:2d}. {wl_center:.9f} um  {status}  Continuum={local_continuum:.1f}, Depth={depth:.1f} (at idx={idx})")
        
        # Now fit all lines simultaneously, in shared wavelength chunks (the same
        # collective approach Flux Map uses) rather than one independent fit per line.
        print(f"\n  Fitting all lines with fit_hitran_lines_collective...")
        line_pairs = fit_hitran_lines_collective(
            wavelength, spectrum,
            hitran_lines,
            continuum=np.zeros_like(spectrum),
            header=None,
            wave_unit=None
        )

        print(f"\n  {'#':<3} | {'HITRAN':<11} | {'Fitted':<11} | {'Transition':<20}")
        print(f"  {'-'*60}")
        for i, (hitran_line, result) in enumerate(line_pairs, 1):
            hitran_wl = hitran_line['wavelength']
            transition = hitran_line['transition']
            fitted_wl = result.center if result.success else float('nan')
            print(f"  {i:<3} | {hitran_wl:<11.7f} | {fitted_wl:<11.7f} | {transition:<20}")

        # Filter to 4.7-4.8 range
        all_lines = [r for _, r in line_pairs if 4.7 <= r.center <= 4.8]
        detected = [r for r in all_lines if r.success]
        print(f"\n  Lines in 4.7-4.8 um range: {len(all_lines)}")
        print(f"  Successfully fitted (success=True): {len(detected)} lines")
        
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # STEP 4: Fit continuum - using exact GUI function
    print("\n[STEP 4] Fitting continuum (GUI algorithm)...")
    try:
        continuum = fit_continuum_directly(wavelength, spectrum)
        print(f"  SUCCESS! Continuum fitted")
    except Exception as e:
        print(f"  ERROR: {e}")
        continuum = None

    # STEP 5: Plot using GUI's visualization function
    print("\n[STEP 5] Creating visualization...")
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot raw spectrum
        ax.plot(wavelength, spectrum, 'b-', linewidth=0.8, label='Total Spectrum', alpha=0.7)
        
        # Plot continuum
        if continuum is not None:
            ax.plot(wavelength, continuum, 'orange', linewidth=2.5, label='Continuum', zorder=10)
        
        # Mark individual lines - only show lines that passed fitting
        line_positions = []
        for line in detected:
            wl = line['wavelength']
            ax.axvline(wl, color='red', alpha=0.3, linewidth=0.9, linestyle=':')
            line_positions.append(wl)
        
        ax.set_xlabel('Wavelength (um)', fontsize=11)
        ax.set_ylabel('Flux', fontsize=11)
        ax.set_title(f'Spectrum + Continuum: {len(line_positions)} CO v=1->0 Lines (4.7-4.8 um)', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim(wavelength.min(), wavelength.max())
        
        plt.tight_layout()
        
        output_file = 'co_analysis_nirspec.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  SUCCESS! Plot saved: {output_file}")
        print(f"  Detected {len(line_positions)} lines in 4.7-4.8 um:")
        for i, wl in enumerate(sorted(line_positions), 1):
            print(f"    {i:2d}. {wl:.9f} um")
        plt.close(fig)
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print(" DONE - Using direct GUI functions (no complex API wrapper)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
