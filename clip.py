import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import median_filter, rotate
from scipy.interpolate import UnivariateSpline, CubicSpline, interp1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

try:
    from cube_explorer.data_loader import CubeData
except ImportError:
    print("Warning: Could not import CubeData. Ensure clip.py is outside the cube_explorer folder.")

def save_output(cube, wavelength, data, data_type="continuum", output_name=None):
    """
    Saves spectral data. For subtracted spectra, it saves the original 
    flux alongside the fit to ensure unit consistency (MJy/sr).

    Parameters
    ----------
    output_name : str, optional
        Filename to write (without extension).  When provided the
        interactive ``input()`` prompt is skipped — useful when calling
        this function from a Jupyter notebook or script.
    """
    if output_name is not None:
        fname = str(output_name).strip()
    else:
        fname = input(f"\nEnter filename for the {data_type.upper()} (without extension): ").strip()
    out_name = f"{fname}.txt"

    wave_unit = getattr(cube, 'wave_unit', 'Unknown')
    # Force flux unit from the FITS header to prevent MJy vs Jy mismatch
    flux_unit = cube.header.get('BUNIT', 'MJy/sr') 
    instrument = getattr(cube, 'instrument', 'Unknown')
    
    # Get original flux for comparison if we are saving a subtraction
    original_flux = cube.spectrum if hasattr(cube, 'spectrum') else None

    with open(out_name, 'w') as f:
        f.write(f"# Export from IFU-Analyzer CLI\n")
        f.write(f"# Instrument: {instrument} | Unit: {flux_unit}\n")
        
        if data_type == "continuum":
            f.write(f"# Type: fitted_continuum\n")
            f.write(f"# Wavelength ({wave_unit})\tContinuum_Fit\n")
            for w, val in zip(wavelength, data):
                if np.isfinite(val):
                    f.write(f"{w:.6e}\t{val:.6e}\n")
                    
        elif data_type == "subtracted spectrum":
            f.write(f"# Type: continuum_subtracted\n")
            f.write(f"# Wavelength ({wave_unit})\tOriginal_Flux\tContinuum_Fit\tSubtracted_Result\n")
            # Calculate the fit used by working backward (Original - Result = Fit)
            for i, (w, sub_val) in enumerate(zip(wavelength, data)):
                if np.isfinite(sub_val) and original_flux is not None:
                    orig_val = original_flux[i]
                    fit_val = orig_val - sub_val
                    f.write(f"{w:.6e}\t{orig_val:.6e}\t{fit_val:.6e}\t{sub_val:.6e}\n")

    print(f"[OK] Task: Exported {data_type} to {out_name} using {instrument} calibration.")


def run_subtract_continuum(args):
    """Subtract a constant value, a loaded continuum file, or calculate from raw points."""
    print(f"Loading cube: {args.input}")
    cube = CubeData()
    if not cube.load_from_file(args.input):
        print("Error loading cube.")
        return

    wavelength = cube.wavelength.value if hasattr(cube.wavelength, 'value') else cube.wavelength
    flux = cube.spectrum

    if args.continuum.endswith('.txt') or args.continuum.endswith('.csv'):
        print(f"Analyzing file type for {args.continuum}...")
        
        is_fitted_continuum = False
        try:
            with open(args.continuum, 'r') as f:
                for _ in range(15):
                    line = f.readline()
                    if not line: break
                    if "Type: fitted_continuum" in line or "Continuum Export" in line:
                        is_fitted_continuum = True
                        break
        except Exception as e:
            print(f"Error reading file: {e}")
            return
            
        if is_fitted_continuum:
            print("[OK] Detected fitted continuum file. Mapping values...")
            try:
                cont_data = np.genfromtxt(args.continuum, delimiter='\t', comments='#')
                if cont_data.shape[1] >= 2:
                    interp = interp1d(cont_data[:, 0], cont_data[:, 1], bounds_error=False, fill_value=np.nan)
                    continuum = interp(wavelength)
                else:
                    raise ValueError("File does not have enough columns.")
            except Exception as e:
                print(f"Error extracting data: {e}")
                return
                
        else:
            print("[OK] No headers detected. Assuming raw custom points file...")
            print("Fitting Cubic Spline to the provided points...")
            try:
                points = np.genfromtxt(args.continuum, delimiter=',', comments='#')
                if np.all(np.isnan(points)):
                    points = np.genfromtxt(args.continuum, comments='#')
                    
                if points.ndim == 1: 
                    points = points.reshape(1, -1)
                
                points = points[np.argsort(points[:, 0])]
                x_pts, y_pts = points[:, 0], points[:, 1]
                
                spline_model = CubicSpline(x_pts, y_pts, bc_type='natural')
                continuum = spline_model(wavelength)
            except Exception as e:
                print(f"Error parsing points file: {e}")
                return
    else:
        try:
            val = float(args.continuum)
            continuum = np.full_like(flux, val)
            print(f"Applying constant continuum baseline of {val}")
        except ValueError:
            print("Error: --continuum must be a valid number or a path to a .txt file")
            return

    print("Subtraction complete!")
    subtracted_flux = flux - continuum
    save_output(cube, wavelength, subtracted_flux, data_type="subtracted spectrum")


def run_auto_continuum(args):
    """Run automatic continuum subtraction using space-separated parameters."""
    print(f"Loading cube: {args.input}")
    cube = CubeData()
    if not cube.load_from_file(args.input):
        print("Error loading cube.")
        return

    params = args.params
    window = int(params[0]) if len(params) > 0 else 51
    sigma_val = float(params[1]) if len(params) > 1 else 2.0
    width = int(params[2]) if len(params) > 2 else 1
    distance = int(params[3]) if len(params) > 3 else 3
    exclusion = int(params[4]) if len(params) > 4 else 3
    iters = int(params[5]) if len(params) > 5 else 2
    smoothing = int(params[6]) if len(params) > 6 else 5

    wavelength = cube.wavelength.value if hasattr(cube.wavelength, 'value') else cube.wavelength
    flux = cube.spectrum
    mask = np.isfinite(wavelength) & np.isfinite(flux)
    wave_clean, flux_clean = wavelength[mask], flux[mask]

    print(f"Running median filter and peak detection (Window: {window}, Sigma: {sigma_val})...")
    flux_smoothed = median_filter(flux_clean, size=window, mode='reflect')
    peaks, _ = find_peaks(flux_clean, prominence=np.std(flux_clean) * sigma_val, width=width, distance=distance)

    exclude_mask = np.zeros(len(flux_clean), dtype=bool)
    for peak_idx in peaks:
        left = max(0, peak_idx - exclusion)
        right = min(len(flux_clean), peak_idx + exclusion + 1)
        exclude_mask[left:right] = True

    flux_good = flux_clean.copy()
    flux_good[exclude_mask] = np.nan
    if np.any(exclude_mask):
        flux_good[exclude_mask] = np.interp(wave_clean[exclude_mask], wave_clean[~exclude_mask], flux_clean[~exclude_mask])

    continuum = median_filter(flux_good, size=window, mode='reflect')
    
    for _ in range(iters):
        residuals = flux_clean - continuum
        current_sigma = np.std(residuals[~exclude_mask])
        outliers = (residuals > 2.0 * current_sigma) & (~exclude_mask)
        if np.sum(outliers) == 0: break
        exclude_mask |= outliers
        flux_good = flux_clean.copy()
        flux_good[exclude_mask] = np.interp(wave_clean[exclude_mask], wave_clean[~exclude_mask], flux_clean[~exclude_mask])
        continuum = median_filter(flux_good, size=window, mode='reflect')

    spline = UnivariateSpline(wave_clean[~exclude_mask], continuum[~exclude_mask], s=len(wave_clean)/smoothing, k=3)
    continuum_smooth = spline(wave_clean)
    
    print("Calculation complete!")
    save_output(cube, wave_clean, continuum_smooth, data_type="continuum")


def run_custom_continuum(args):
    """Run custom continuum fitting using a saved TXT file of points."""
    print(f"Loading cube: {args.input}")
    cube = CubeData()
    if not cube.load_from_file(args.input):
        print("Error loading cube.")
        return

    wavelength = cube.wavelength.value if hasattr(cube.wavelength, 'value') else cube.wavelength
    flux = cube.spectrum

    print(f"Loading points from {args.points}...")
    try:
        points = np.genfromtxt(args.points, delimiter=',', comments='#')
        if points.ndim == 1: points = points.reshape(1, -1)
        
        points = points[np.argsort(points[:, 0])]
        x_pts, y_pts = points[:, 0], points[:, 1]
    except Exception as e:
        print(f"Error loading points file: {e}")
        return

    print("Fitting Cubic Spline and subtracting...")
    spline_model = CubicSpline(x_pts, y_pts, bc_type='natural')
    continuum = spline_model(wavelength)
    subtracted_flux = flux - continuum

    print("Calculation complete!")
    save_output(cube, wavelength, subtracted_flux, data_type="subtracted spectrum")


def run_2d_rotate(args):
    """Rotate a 2D FITS file and prompt to save it."""
    print(f"Loading 2D FITS: {args.input}")
    with fits.open(args.input) as hdul:
        data = np.squeeze(hdul[0].data)
        header = hdul[0].header

    theta = np.deg2rad(args.angle)
    print(f"Rotating by {args.angle} degrees...")
    
    array_out = rotate(data, args.angle, reshape=True, cval=np.nan, order=1)
    
    if 'CD1_1' in header:
        cos_t, sin_t = np.cos(-theta), np.sin(-theta)
        R_inv = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        cd_mat = np.array([[header.get('CD1_1', 1), header.get('CD1_2', 0)],
                           [header.get('CD2_1', 0), header.get('CD2_2', 1)]])
        new_cd = np.dot(cd_mat, R_inv)
        header['CD1_1'], header['CD1_2'] = new_cd[0, 0], new_cd[0, 1]
        header['CD2_1'], header['CD2_2'] = new_cd[1, 0], new_cd[1, 1]

    # Prompt user for output filename
    fname = input("\nEnter filename to save ROTATED FITS (without extension): ").strip()
    out_name = f"{fname}.fits"
    
    fits.writeto(out_name, array_out, header, overwrite=True)
    print(f"[OK] Saved rotated FITS to {out_name}")


def run_2d_contour(args):
    """Generate contour coordinates, calculate RA/Dec via WCS, and prompt to save."""
    print(f"Loading 2D FITS: {args.input}")
    with fits.open(args.input) as hdul:
        data = np.squeeze(hdul[0].data)
        header = hdul[0].header
        
    # Set up WCS to calculate the RA/Dec accurately
    wcs = WCS(header)
    if wcs.pixel_n_dim > 2:
        wcs = wcs.celestial
    
    valid = data[np.isfinite(data)]
    _, median_val, astropy_std = sigma_clipped_stats(valid, sigma=3.0)
    
    print(f"Background Median: {median_val:.4f}, Std: {astropy_std:.4f}")
    band_levels = sorted([median_val + n * astropy_std for n in args.levels])
    
    fig, ax = plt.subplots()
    cset = ax.contour(data, levels=band_levels)

    # Show the figure now so Jupyter / interactive sessions render it
    # before the blocking input() prompt below.
    plt.tight_layout()
    plt.show()

    rows = ["# IFU Analyzer 2D - Contour Export\n", "# level, x_pixel, y_pixel, RA_deg, Dec_deg\n"]
    
    # FIX: Use .allsegs which is compatible with newer versions of matplotlib (3.8+)
    for i, level_segs in enumerate(cset.allsegs):
        level = cset.levels[i] if i < len(cset.levels) else i
        for seg in level_segs:
            # seg is an array of (x, y) coordinates for a specific contour line
            for x, y in seg:
                # Calculate the exact world coordinates for every single pixel path point
                ra, dec = wcs.pixel_to_world_values(x, y)
                rows.append(f"{level:.6e}, {x:.4f}, {y:.4f}, {float(ra):.8f}, {float(dec):.8f}\n")
                
    # Prompt user for output filename
    fname = input("\nEnter filename to save CONTOUR MAP (without extension): ").strip()
    out_name = f"{fname}.txt"
    
    with open(out_name, 'w') as f:
        f.writelines(rows)
    print(f"[OK] Saved contour map to {out_name}")
    plt.close(fig)  # Release figure; prevents stale figures in Jupyter

# =============================================================================
# Jupyter / script convenience wrappers
# =============================================================================
# The run_*() functions above require an argparse.Namespace object, which is
# awkward to build interactively.  The functions below accept plain keyword
# arguments and are meant to be called directly from a Jupyter notebook:
#
#   from clip import auto_continuum, subtract_continuum, custom_continuum
#   from clip import rotate_2d, contour_2d
#
# All output_name parameters default to None, which triggers the usual
# interactive input() prompt.  Pass a string to skip the prompt:
#
#   auto_continuum("mycube.fits", output_name="my_continuum")
# =============================================================================

def auto_continuum(input_file, window=51, sigma=2.0, width=1, distance=3,
                   exclusion=3, iters=2, smoothing=5, output_name=None):
    """Compute an automatic continuum fit and save it.

    Parameters match the --params order of the CLI:
    window, sigma, width, distance, exclusion, iters, smoothing.
    Pass output_name to skip the interactive filename prompt.
    """
    from argparse import Namespace
    args = Namespace(
        input=input_file,
        params=[window, sigma, width, distance, exclusion, iters, smoothing],
    )
    # Temporarily patch save_output to inject output_name when provided
    if output_name is not None:
        import functools
        orig = save_output
        globals()['save_output'] = functools.partial(orig, output_name=output_name)
        try:
            run_auto_continuum(args)
        finally:
            globals()['save_output'] = orig
    else:
        run_auto_continuum(args)


def subtract_continuum(input_file, continuum, output_name=None):
    """Subtract a continuum from a cube spectrum and save the result.

    continuum can be a numeric constant, a fitted-continuum .txt file, or
    a raw points .txt/.csv file (same rules as the CLI --continuum flag).
    Pass output_name to skip the interactive filename prompt.
    """
    from argparse import Namespace
    args = Namespace(input=input_file, continuum=str(continuum))
    if output_name is not None:
        import functools
        orig = save_output
        globals()['save_output'] = functools.partial(orig, output_name=output_name)
        try:
            run_subtract_continuum(args)
        finally:
            globals()['save_output'] = orig
    else:
        run_subtract_continuum(args)


def custom_continuum(input_file, points_file, output_name=None):
    """Fit a cubic spline to custom continuum anchor points and subtract.

    points_file is a comma-separated text file of (wavelength, flux) pairs.
    Pass output_name to skip the interactive filename prompt.
    """
    from argparse import Namespace
    args = Namespace(input=input_file, points=points_file)
    if output_name is not None:
        import functools
        orig = save_output
        globals()['save_output'] = functools.partial(orig, output_name=output_name)
        try:
            run_custom_continuum(args)
        finally:
            globals()['save_output'] = orig
    else:
        run_custom_continuum(args)


def rotate_2d(input_file, angle, output_name=None):
    """Rotate a 2D FITS image by angle degrees and save.

    Pass output_name to skip the interactive filename prompt.
    """
    from argparse import Namespace
    import functools

    args = Namespace(input=input_file, angle=float(angle))
    if output_name is not None:
        # Patch the input() call inside run_2d_rotate by temporarily
        # replacing the built-in so the filename is injected non-interactively.
        import builtins
        orig_input = builtins.input
        builtins.input = lambda _prompt='': output_name
        try:
            run_2d_rotate(args)
        finally:
            builtins.input = orig_input
    else:
        run_2d_rotate(args)


def contour_2d(input_file, levels, output_name=None):
    """Generate contour coordinates for a 2D FITS image and save.

    levels is a list of sigma multiples, e.g. [3, 5, 10].
    Pass output_name to skip the interactive filename prompt.
    """
    from argparse import Namespace
    import builtins

    args = Namespace(input=input_file, levels=[float(l) for l in levels])
    if output_name is not None:
        orig_input = builtins.input
        builtins.input = lambda _prompt='': output_name
        try:
            run_2d_contour(args)
        finally:
            builtins.input = orig_input
    else:
        run_2d_contour(args)


def main():
    parser = argparse.ArgumentParser(description="IFU Analyzer Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Automatic Continuum
    parser_auto = subparsers.add_parser("auto-continuum")
    parser_auto.add_argument("-i", "--input", required=True)
    parser_auto.add_argument("-p", "--params", nargs='+', type=float, default=[51, 2.0, 1, 3, 3, 2, 5],
                             help="Provide space-separated values: window sigma width distance exclusion iters smoothing")

    # 2. Custom Continuum
    parser_custom = subparsers.add_parser("custom-continuum")
    parser_custom.add_argument("-i", "--input", required=True)
    parser_custom.add_argument("-p", "--points", required=True)

    # 3. Specific Subtraction
    parser_sub = subparsers.add_parser("subtract-continuum")
    parser_sub.add_argument("-i", "--input", required=True, help="Input FITS file")
    parser_sub.add_argument("-c", "--continuum", required=True, help="Constant numeric value OR a path to a continuum .txt file")

    # 4. 2D Rotate (Output argument removed to allow interactive prompting)
    parser_rot = subparsers.add_parser("2d-rotate")
    parser_rot.add_argument("-i", "--input", required=True)
    parser_rot.add_argument("-a", "--angle", type=float, required=True)

    # 5. 2D Contour (Output argument removed to allow interactive prompting)
    parser_contour = subparsers.add_parser("2d-contour")
    parser_contour.add_argument("-i", "--input", required=True)
    parser_contour.add_argument("-l", "--levels", nargs='+', type=float, required=True, 
                                help="Provide space-separated sigma levels (e.g., 3 5 10)")

    args = parser.parse_args()
    if args.command == "auto-continuum": run_auto_continuum(args)
    elif args.command == "custom-continuum": run_custom_continuum(args)
    elif args.command == "subtract-continuum": run_subtract_continuum(args)
    elif args.command == "2d-rotate": run_2d_rotate(args)
    elif args.command == "2d-contour": run_2d_contour(args)

if __name__ == "__main__":
    main()