"""
Pixel-by-pixel rotational diagram map.

Runs the region-level rotational diagram fit at every spatial pixel of one cube,
or of several co-aligned cubes with their spectra stitched. Fits detections only:
a line below the 3-sigma floor at a pixel is dropped, not made an upper limit.
"""

import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from threading import Lock

import numpy as np

from .unit_converter import extract_units
from .analyzer_2d import _get_pixel_size_arcsec, _make_2d_wcs
from .rotational_diagram_h2_fitter import (
    compute_line_fluxes_collective, get_gu, pc,
    rotational_diagram as rotational_diagram_h2,
)
from .rotational_diagram_general_fitter import (
    rotational_diagram as rotational_diagram_general,
)

# Status codes for the per-pixel status map.
STATUS_OK = 0
STATUS_NO_DATA = 1           # pixel spectrum is empty or all-NaN
STATUS_TOO_FEW_LINES = 2     # fewer detections than min_lines
STATUS_FIT_FAILED = 3        # the rotational_diagram call raised
STATUS_UNCONSTRAINED = 4     # fit hit the T ceiling or the error is >= T

# T ceilings inside the fitters. These must match rotational_diagram_h2_fitter
# (3000) and rotational_diagram_general_fitter (4000).
T_CAP_H2 = 3000.0
T_CAP_GENERAL = 4000.0

MAP_KEYS = ("T", "T_err", "Ncol", "Ncol_err")


class _NullWriter:
    """
    Sink that discards writes, for redirect_stdout over the pixel loop.

    Args:
        None

    Returns:
        None
    """

    def write(self, _text):
        """
        Discard the text.

        Args:
            _text (str): Text to drop

        Returns:
            None
        """

    def flush(self):
        """
        Do nothing.

        Args:
            None

        Returns:
            None
        """


class _StitchedCube:
    """
    Present several co-aligned cubes as one cube with a concatenated spectrum.

    Args:
        cubes (list): CubeData objects that pass _cubes_coaligned

    Returns:
        None
    """

    def __init__(self, cubes):
        self._cubes = cubes
        waves = [np.asarray(extract_units(c.wavelength)[0], dtype=float) for c in cubes]
        cat = np.concatenate(waves)
        self._order = np.argsort(cat)
        self.wavelength = cat[self._order]
        self.header = cubes[0].header
        _, n_y, n_x = cubes[0].cube.shape
        self.cube = _Shape((self.wavelength.size, n_y, n_x))

    def get_pixel_spectrum(self, x, y):
        """
        Concatenated flux at pixel (x, y), sorted to match self.wavelength.

        Args:
            x (int): Spatial x index
            y (int): Spatial y index

        Returns:
            spectrum (np.ndarray) or None: Flux values, or None if every cube
                returns None
        """
        parts = []
        for cube in self._cubes:
            s = cube.get_pixel_spectrum(x, y)
            if s is None:
                s = np.full(cube.cube.shape[0], np.nan)
            parts.append(np.asarray(s, dtype=float))
        cat = np.concatenate(parts)
        if np.all(np.isnan(cat)):
            return None
        return cat[self._order]


class _Shape:
    """
    Minimal stand-in exposing only `.shape`, for _StitchedCube.cube.

    Args:
        shape (tuple): (n_wave, n_y, n_x)

    Returns:
        None
    """

    def __init__(self, shape):
        self.shape = shape


def _cubes_coaligned(cubes, tol_frac=0.1):
    """
    Check that every cube shares one spatial grid, so no reprojection is needed.

    Args:
        cubes (list): CubeData objects, each with cube.shape and header
        tol_frac (float): Allowed corner mismatch, as a fraction of a pixel

    Returns:
        result (tuple): (ok, message). ok is True when the grids match; message
            names the first mismatch otherwise
    """
    if len(cubes) < 2:
        return True, ""
    ref = cubes[0]
    ref_ny, ref_nx = ref.cube.shape[1], ref.cube.shape[2]
    ref_scale = _get_pixel_size_arcsec(ref.header)
    ref_wcs = _make_2d_wcs(ref.header)
    for cube in cubes[1:]:
        ny, nx = cube.cube.shape[1], cube.cube.shape[2]
        if (ny, nx) != (ref_ny, ref_nx):
            return False, f"spatial shape {ny}x{nx} != {ref_ny}x{ref_nx}"
        scale = _get_pixel_size_arcsec(cube.header)
        if ref_scale and scale and abs(scale - ref_scale) > tol_frac * ref_scale:
            return False, f"pixel scale {scale:.4f} != {ref_scale:.4f} arcsec"
        wcs = _make_2d_wcs(cube.header)
        if ref_wcs is not None and wcs is not None and ref_scale:
            for px, py in ((0, 0), (ref_nx - 1, ref_ny - 1)):
                sep = ref_wcs.pixel_to_world(px, py).separation(
                    wcs.pixel_to_world(px, py)).arcsec
                if sep > tol_frac * ref_scale:
                    return False, f"corner offset {sep:.4f} arcsec at ({px},{py})"
    return True, ""


def _resolve_cube(cube_or_list):
    """
    Accept one cube or a co-aligned list, and return one cube-like object.

    Args:
        cube_or_list (CubeData or list): A single cube, or several co-aligned cubes

    Returns:
        cube (object): The cube itself, or a _StitchedCube over the list

    Raises:
        ValueError: if a multi-cube list is not on one spatial grid
    """
    if not isinstance(cube_or_list, (list, tuple)):
        return cube_or_list
    if len(cube_or_list) == 1:
        return cube_or_list[0]
    ok, msg = _cubes_coaligned(list(cube_or_list))
    if not ok:
        raise ValueError(f"Cubes are not on a common spatial grid ({msg}); "
                         f"reprojection is not supported yet.")
    return _StitchedCube(list(cube_or_list))


def _dedupe_colocated(lines, tol_um):
    """
    Drop the weaker line of each near-coincident pair.

    Args:
        lines (list): Line dicts, each with 'wavelength' (um) and 'a_coeff' (s^-1)
        tol_um (float): Two lines are the same feature within this many microns

    Returns:
        result (tuple): (kept, dropped) lists of the input dicts, kept sorted by
            wavelength
    """
    ordered = sorted(lines, key=lambda r: r["wavelength"])
    kept, dropped = [], []
    for line in ordered:
        if kept and abs(line["wavelength"] - kept[-1]["wavelength"]) <= tol_um:
            weaker, stronger = sorted(
                (line, kept[-1]), key=lambda r: r.get("a_coeff", 0.0)
            )
            kept[-1] = stronger
            dropped.append(weaker)
        else:
            kept.append(line)
    return kept, dropped


def _build_molecular_data(lines, molecule):
    """
    Build the molecular_data dict, the same way run_rotational_fit does.

    Args:
        lines (list): Line dicts sorted by wavelength, each with 'wavelength',
            'a_coeff', 'e_upper', 'Gp', and 'branch'/'j_lower' for H2
        molecule (str): Species name; 'H2' S-branch lines use the get_gu rule

    Returns:
        molecular_data (dict): 'wavelengths', 'Aul', 'Eu', 'gu', each {index: value}
            keyed by a 1-based index over `lines`
    """
    data = {"wavelengths": {}, "Aul": {}, "Eu": {}, "gu": {}}
    for i, line in enumerate(lines, start=1):
        data["wavelengths"][i] = line["wavelength"]
        data["Aul"][i] = line.get("a_coeff", 0.0)
        data["Eu"][i] = line.get("e_upper", 0.0)
        j_lower = line.get("j_lower")
        if molecule == "H2" and line.get("branch") == "S" and j_lower is not None:
            data["gu"][i] = get_gu(j_lower)
        else:
            data["gu"][i] = line.get("Gp", 1)
    return data


def _run_fitter(fluxes, flux_errors, molecular_data, molecule, isotopologue,
                opr, fit_opr, opr_grid, dist_pc, r_au):
    """
    Call the right rotational_diagram fitter (H2 or general), 1T only.

    Args:
        fluxes (dict): {index: flux_cgs} for kept detections
        flux_errors (dict): {index: flux_error_cgs}
        molecular_data (dict): Output of _build_molecular_data
        molecule (str): Species name; 'H2' uses the OPR-capable fitter
        isotopologue (int): HITRAN isotopologue number, non-H2 only
        opr (float): Fixed ortho-para ratio, or None for the fitter default
        fit_opr (bool): Grid-search OPR, H2 only
        opr_grid (array): OPR grid for the search, or None
        dist_pc (float): Source distance in parsecs
        r_au (float): Emitting radius in AU, or None

    Returns:
        results (dict): The fitter's result dict, including '1T' and '_plot_data'
    """
    if molecule == "H2":
        return rotational_diagram_h2(
            fluxes, flux_errors, molecular_data=molecular_data, OPR=opr,
            fit_opr=fit_opr, opr_grid=opr_grid, fit1T=True, fit2T=False,
            R=r_au, dist=dist_pc, upper_limits=None,
        )
    return rotational_diagram_general(
        fluxes, flux_errors, molecular_data=molecular_data, fit1T=True,
        fit2T=False, R=r_au, dist=dist_pc, upper_limits=None,
        molecule=molecule, isotopologue=isotopologue,
    )


def _collect_detections(per_line):
    """
    Turn compute_line_fluxes_collective output into fluxes/errors dicts.

    Args:
        per_line (list): One result dict per line, in the line-list order

    Returns:
        result (tuple): (fluxes, flux_errors), each {1-based index: value}, over
            successful non-upper-limit lines only
    """
    fluxes, flux_errors = {}, {}
    for idx, res in enumerate(per_line, start=1):
        if not res.get("success") or res.get("is_upper_limit"):
            continue
        fluxes[idx] = res["flux_cgs"]
        if res.get("flux_error_cgs") is not None:
            flux_errors[idx] = res["flux_error_cgs"]
    return fluxes, flux_errors


def _prepare_lines(line_table, molecule, dedup_tol_um):
    """
    Dedupe co-located lines and build the fit inputs from the kept ones.

    Args:
        line_table (list): Line dicts with 'wavelength', 'a_coeff', 'e_upper', 'Gp'
        molecule (str): Species name, passed to _build_molecular_data
        dedup_tol_um (float): Co-located-line merge tolerance in microns

    Returns:
        result (tuple): (kept, dropped, line_wavelengths, molecular_data)
    """
    kept, dropped = _dedupe_colocated(line_table, dedup_tol_um)
    line_wavelengths = [line["wavelength"] for line in kept]
    molecular_data = _build_molecular_data(kept, molecule)
    return kept, dropped, line_wavelengths, molecular_data


def _fit_one_pixel(spectrum, wavelength, line_wavelengths, molecular_data, header,
                   contam_lines, molecule, isotopologue, opr, fit_opr, opr_grid,
                   dist_pc, r_au, min_lines):
    """
    Fit one pixel's rotational diagram (1T, detections only).

    Args:
        spectrum (array): Pixel flux values, native units
        wavelength (array): Wavelength values (um) for `spectrum`
        line_wavelengths (list): Target line centres (um), wavelength-sorted
        molecular_data (dict): Output of _build_molecular_data for the same lines
        header (astropy.io.fits.Header): Cube header, for instrument and CGS conversion
        contam_lines (pd.DataFrame): Contaminating lines, or None
        molecule (str): Species name; 'H2' uses the OPR-capable fitter
        isotopologue (int): HITRAN isotopologue number, non-H2 only
        opr (float): Fixed ortho-para ratio, or None
        fit_opr (bool): Grid-search OPR, H2 only
        opr_grid (array): OPR grid, or None
        dist_pc (float): Source distance in parsecs
        r_au (float): Emitting radius in AU, or None
        min_lines (int): Fewest detections a 1T fit needs

    Returns:
        row (tuple): (T, dT, Ncol, dNcol, n_lines, status); the first four are NaN
            unless status is STATUS_OK
    """
    nan = float("nan")
    if spectrum is None or np.all(np.isnan(spectrum)):
        return nan, nan, nan, nan, 0, STATUS_NO_DATA

    per_line = compute_line_fluxes_collective(
        wavelength, spectrum, line_wavelengths, header, contam_lines=contam_lines
    )
    fluxes, flux_errors = _collect_detections(per_line)
    if len(fluxes) < min_lines:
        return nan, nan, nan, nan, len(fluxes), STATUS_TOO_FEW_LINES

    try:
        result = _run_fitter(fluxes, flux_errors, molecular_data, molecule,
                             isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au)
    except Exception:
        traceback.print_exc()
        return nan, nan, nan, nan, len(fluxes), STATUS_FIT_FAILED

    one_t = result["1T"]
    t_val, dt_val = float(one_t["T"]), float(one_t["dT"])
    ncol, dncol = float(one_t["Ncol"]), float(one_t["dNcol"])

    # A railed T, or an error at least as large as T, means the lines do not
    # constrain the slope. Report nothing, not the ceiling value.
    t_cap = T_CAP_H2 if molecule == "H2" else T_CAP_GENERAL
    if t_val >= 0.99 * t_cap or not np.isfinite(dt_val) or dt_val >= t_val:
        return nan, nan, nan, nan, len(fluxes), STATUS_UNCONSTRAINED

    return t_val, dt_val, ncol, dncol, len(fluxes), STATUS_OK


def fit_pixel(cube_or_list, x, y, line_table, molecule="H2", isotopologue=1,
              opr=None, fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None,
              min_lines=3, dedup_tol_um=1e-5, contam_lines=None):
    """
    Fit one pixel and return the full rotational_diagram result for display.

    Args:
        cube_or_list (CubeData or list): The cube, or co-aligned cubes to stitch
        x (int): Spatial x index
        y (int): Spatial y index
        line_table (list): Line dicts, see create_rotational_diagram_map
        molecule (str): Species name
        isotopologue (int): HITRAN isotopologue number, non-H2 only
        opr (float): Fixed ortho-para ratio, or None
        fit_opr (bool): Grid-search OPR, H2 only
        opr_grid (array): OPR grid, or None
        dist_pc (float): Source distance in parsecs
        r_au (float): Emitting radius in AU, or None
        min_lines (int): Fewest detections a 1T fit needs
        dedup_tol_um (float): Co-located-line merge tolerance in microns
        contam_lines (pd.DataFrame): Contaminating lines, or None

    Returns:
        out (dict): 'status' (int), 'n_lines' (int), and on STATUS_OK also
            'results' (the fitter's dict with '1T'/'_plot_data'); otherwise 'reason'
    """
    cube = _resolve_cube(cube_or_list)
    wavelength, _unit = extract_units(cube.wavelength)
    kept, _dropped, line_wavelengths, molecular_data = _prepare_lines(
        line_table, molecule, dedup_tol_um)

    spectrum = cube.get_pixel_spectrum(x, y)
    if spectrum is None or np.all(np.isnan(spectrum)):
        return {"status": STATUS_NO_DATA, "n_lines": 0, "reason": "pixel has no data"}

    per_line = compute_line_fluxes_collective(
        wavelength, spectrum, line_wavelengths, cube.header, contam_lines=contam_lines)
    fluxes, flux_errors = _collect_detections(per_line)
    if len(fluxes) < min_lines:
        return {"status": STATUS_TOO_FEW_LINES, "n_lines": len(fluxes),
                "reason": f"{len(fluxes)} detections, need {min_lines}"}

    try:
        results = _run_fitter(fluxes, flux_errors, molecular_data, molecule,
                              isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au)
    except Exception as exc:
        return {"status": STATUS_FIT_FAILED, "n_lines": len(fluxes), "reason": str(exc)}

    return {"status": STATUS_OK, "n_lines": len(fluxes), "results": results}


def create_rotational_diagram_map(cube_or_list, line_table, molecule="H2",
                                  isotopologue=1, opr=None, fit_opr=False,
                                  opr_grid=None, dist_pc=140.0, r_au=None,
                                  min_lines=3, dedup_tol_um=1e-5, n_jobs=8,
                                  contam_lines=None, progress_callback=None,
                                  cancel_flag=None):
    """
    Fit a 1T rotational diagram at every spatial pixel of the cube.

    Args:
        cube_or_list (CubeData or list): The cube, or co-aligned cubes to stitch
        line_table (list): Line dicts with 'wavelength', 'a_coeff', 'e_upper', 'Gp'
            and 'branch'/'j_lower' for H2 S-branch lines
        molecule (str): Species name; 'H2' uses the OPR-capable fitter
        isotopologue (int): HITRAN isotopologue number, non-H2 only
        opr (float): Fixed ortho-para ratio, or None for the fitter default
        fit_opr (bool): Grid-search OPR per pixel, H2 only
        opr_grid (array): OPR grid for the search, or None
        dist_pc (float): Source distance in parsecs, passed straight to the fitter
        r_au (float): Emitting radius in AU, or None (total-molecules mode)
        min_lines (int): Fewest detections a pixel needs for a 1T fit
        dedup_tol_um (float): Co-located-line merge tolerance in microns
        n_jobs (int): Worker threads, one cube row per task
        contam_lines (pd.DataFrame): Contaminating lines, or None
        progress_callback (callable): progress_callback(done_pixels, total_pixels)
        cancel_flag (object): Object with a `.cancelled` attribute, or None

    Returns:
        result (dict): 'T', 'T_err', 'Ncol', 'Ncol_err' (2D float32), 'n_lines',
            'status' (2D int), plus 'header', 'wave_range', 'molecule', 'dist_pc',
            'r_au', 'opr', 'kept_lines', 'dropped_lines'

    Raises:
        ValueError: if a multi-cube list is not on one spatial grid
    """
    cube = _resolve_cube(cube_or_list)
    if cube.cube is None:
        print("[ERROR] No cube data loaded")
        return None

    _n_wave, n_y, n_x = cube.cube.shape
    wavelength, _unit = extract_units(cube.wavelength)
    kept, dropped, line_wavelengths, molecular_data = _prepare_lines(
        line_table, molecule, dedup_tol_um)
    for weak in dropped:
        near = min(kept, key=lambda k: abs(k["wavelength"] - weak["wavelength"]))
        print(f"[dedupe] dropped {weak['wavelength']:.6f} um (A={weak.get('a_coeff', 0):.3e}) "
              f"-- co-located with {near['wavelength']:.6f} um (A={near.get('a_coeff', 0):.3e})")

    print(f">>> Rotational diagram map: {n_y}x{n_x} pixels, {len(kept)} lines, "
          f"molecule={molecule}, min_lines={min_lines}, R={r_au}")

    t_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    dt_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    n_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    dn_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    nlines_map = np.zeros((n_y, n_x), dtype=np.int16)
    status_map = np.full((n_y, n_x), STATUS_NO_DATA, dtype=np.int16)

    counter = {"done": 0}
    lock = Lock()
    total = n_y * n_x
    start = time.time()
    quiet = _NullWriter()

    def process_row(y):
        """
        Fit every pixel in one cube row.

        Args:
            y (int): Row index on the spatial y-axis

        Returns:
            row (tuple): (y, T, dT, Ncol, dNcol, n_lines, status), each array
                being length n_x
        """
        t_row = np.full(n_x, np.nan, dtype=np.float32)
        dt_row = np.full(n_x, np.nan, dtype=np.float32)
        n_row = np.full(n_x, np.nan, dtype=np.float32)
        dn_row = np.full(n_x, np.nan, dtype=np.float32)
        nl_row = np.zeros(n_x, dtype=np.int16)
        st_row = np.full(n_x, STATUS_NO_DATA, dtype=np.int16)

        for x in range(n_x):
            if cancel_flag is not None and cancel_flag.cancelled:
                break
            spectrum = cube.get_pixel_spectrum(x, y)
            t, dt, ncol, dncol, nlines, status = _fit_one_pixel(
                spectrum, wavelength, line_wavelengths, molecular_data,
                cube.header, contam_lines, molecule, isotopologue,
                opr, fit_opr, opr_grid, dist_pc, r_au, min_lines,
            )
            t_row[x], dt_row[x] = t, dt
            n_row[x], dn_row[x] = ncol, dncol
            nl_row[x], st_row[x] = nlines, status
            with lock:
                counter["done"] += 1
        return y, t_row, dt_row, n_row, dn_row, nl_row, st_row

    # compute_line_fluxes_collective and the fitter print per line. Send that to a
    # buffer for the whole run; progress still goes to the real stdout.
    real_stdout = sys.__stdout__
    with redirect_stdout(quiet), ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(process_row, y) for y in range(n_y)]
        for done_count, future in enumerate(as_completed(futures), start=1):
            if cancel_flag is not None and cancel_flag.cancelled:
                for f in futures:
                    f.cancel()
                break
            y, t_row, dt_row, n_row, dn_row, nl_row, st_row = future.result()
            t_map[y], dt_map[y] = t_row, dt_row
            n_map[y], dn_map[y] = n_row, dn_row
            nlines_map[y], status_map[y] = nl_row, st_row
            with lock:
                done = counter["done"]
            if progress_callback is not None:
                progress_callback(done, total)
            elapsed = time.time() - start
            real_stdout.write(f"    row {done_count}/{n_y}  ({done}/{total} px, {elapsed:.1f}s)\n")
            real_stdout.flush()

    ok = int(np.sum(status_map == STATUS_OK))
    with np.errstate(all="ignore"):
        t_lo, t_hi = np.nanmin(t_map), np.nanmax(t_map)
    print(f">>> Done ({time.time() - start:.1f}s). Fitted {ok}/{total} pixels; "
          f"T range {t_lo:.1f}-{t_hi:.1f} K")

    return {
        "T": t_map, "T_err": dt_map, "Ncol": n_map, "Ncol_err": dn_map,
        "n_lines": nlines_map, "status": status_map,
        "header": cube.header,
        "wave_range": (float(np.nanmin(wavelength)), float(np.nanmax(wavelength))),
        "molecule": molecule, "dist_pc": dist_pc, "r_au": r_au, "opr": opr,
        "kept_lines": kept, "dropped_lines": dropped,
    }


def save_rotational_diagram_map_fits(result, output_path):
    """
    Write the map result to a multi-extension FITS file.

    Args:
        result (dict): Output of create_rotational_diagram_map
        output_path (str): Destination path

    Returns:
        None
    """
    from astropy.io import fits

    hdus = [fits.PrimaryHDU()]
    for key in ("T", "T_err", "Ncol", "Ncol_err", "n_lines", "status"):
        hdus.append(fits.ImageHDU(np.asarray(result[key]), name=key.upper()))
    hdus[1].header["BUNIT"] = "K"
    hdus[3].header["BUNIT"] = "cm-2"
    hdus[0].header["MOLECULE"] = result["molecule"]
    hdus[0].header["DIST_PC"] = result["dist_pc"]
    if result.get("r_au") is not None:
        hdus[0].header["R_AU"] = result["r_au"]
    fits.HDUList(hdus).writeto(output_path, overwrite=True)
    print(f"[OK] wrote {output_path}")
