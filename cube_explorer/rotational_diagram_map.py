"""
Pixel-by-pixel rotational diagram map: runs the region-level fit at every
spatial pixel of one cube. Several cubes always go through the _multi
functions' convolve+reproject path onto a common grid -- registration is
never assumed from a tolerance check, only verified by construction.
Detections only -- a line below the 3-sigma floor is dropped.
"""

import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np

from .unit_converter import extract_units
from .line_catalog import dedupe_colocated_lines as _dedupe_colocated
from .analyzer_2d import (
    _get_pixel_size_arcsec, _make_2d_wcs, _wcs_footprints_overlap,
    build_target_wcs_for_batch, gaussian_convolve_match, reproject_exact,
)
from .rotational_diagram_h2_fitter import (
    compute_line_fluxes_collective, get_gu, pc, T_CAP as T_CAP_H2,
    rotational_diagram as rotational_diagram_h2,
)
from .rotational_diagram_general_fitter import (
    rotational_diagram as rotational_diagram_general, T_CAP as T_CAP_GENERAL,
)

# Status codes for the per-pixel status map. Every fit the fitter returns is
# STATUS_OK, even one pinned at the T ceiling -- nothing is masked.
STATUS_OK = 0
STATUS_NO_DATA = 1           # pixel spectrum is empty or all-NaN
STATUS_TOO_FEW_LINES = 2     # fewer detections than min_lines
STATUS_FIT_FAILED = 3        # the rotational_diagram call raised

MAP_KEYS = ("T", "T_err", "Ncol", "Ncol_err")

def _resolve_cube(cube_or_list):
    """
    Accept one cube, or a one-element list holding it, and return the cube.

    Args:
        cube_or_list (CubeData or list): A single cube, or a one-element list

    Returns:
        cube (CubeData): The cube itself

    Raises:
        ValueError: if more than one cube is given -- fit_pixel/fit_region/
            create_rotational_diagram_map take exactly one; use
            fit_pixel_multi/fit_region_multi/create_rotational_diagram_map_multi
            for several cubes. Those always convolve and reproject onto a
            common grid rather than assuming several cubes are already
            aligned, so there is no "cubes share one grid" fast path here.
    """
    if not isinstance(cube_or_list, (list, tuple)):
        return cube_or_list
    if len(cube_or_list) == 1:
        return cube_or_list[0]
    raise ValueError(
        f"Got {len(cube_or_list)} cubes; this function takes exactly one. "
        f"Use fit_pixel_multi/fit_region_multi/create_rotational_diagram_map_multi "
        f"for several cubes."
    )


def _line_keys(lines, molecule):
    """
    Assign each line the fit index apply_OPR's is_ortho(J) check needs: the
    line's own upper-J-derived number, matching the notebook's convention,
    not a plain sequential 1, 2, 3... which only shares J's parity by luck.

    Args:
        lines (list): Line dicts, wavelength-sorted
        molecule (str): Species name; only H2 uses j_lower as the key

    Returns:
        keys (list): One integer key per line, same order as `lines`

    Raises:
        ValueError: if two lines land on the same key (e.g. the same j_lower
            from two different vibrational bands)
    """
    if molecule == "H2":
        keys = [line["j_lower"] if line.get("j_lower") is not None else i
                for i, line in enumerate(lines, start=1)]
    else:
        keys = list(range(1, len(lines) + 1))
    if len(set(keys)) != len(keys):
        raise ValueError("Two selected lines share the same fit index (same "
                         "j_lower) -- they cannot be fit in the same diagram.")
    return keys


def _build_molecular_data(lines, keys, molecule):
    """
    Build the molecular_data dict, the same way run_rotational_fit does.

    Args:
        lines (list): Line dicts sorted by wavelength, each with 'wavelength',
            'a_coeff', 'e_upper', 'Gp', and 'branch'/'j_lower' for H2
        keys (list): Output of _line_keys for the same `lines`
        molecule (str): Species name; 'H2' S-branch lines use the get_gu rule

    Returns:
        molecular_data (dict): 'wavelengths', 'Aul', 'Eu', 'gu', each {key: value}
    """
    data = {"wavelengths": {}, "Aul": {}, "Eu": {}, "gu": {}}
    for key, line in zip(keys, lines):
        data["wavelengths"][key] = line["wavelength"]
        data["Aul"][key] = line.get("a_coeff", 0.0)
        data["Eu"][key] = line.get("e_upper", 0.0)
        j_lower = line.get("j_lower")
        if molecule == "H2" and line.get("branch") == "S" and j_lower is not None:
            data["gu"][key] = get_gu(j_lower)
        else:
            data["gu"][key] = line.get("Gp", 1)
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


def _collect_detections(per_line, keys):
    """
    Turn compute_line_fluxes_collective output into fluxes/errors dicts.

    Args:
        per_line (list): One result dict per line, in the line-list order
        keys (list): Output of _line_keys for the same line list

    Returns:
        result (tuple): (fluxes, flux_errors), each {key: value}, over
            successful non-upper-limit lines only
    """
    fluxes, flux_errors = {}, {}
    for key, res in zip(keys, per_line):
        if not res.get("success") or res.get("is_upper_limit"):
            continue
        fluxes[key] = res["flux_cgs"]
        if res.get("flux_error_cgs") is not None:
            flux_errors[key] = res["flux_error_cgs"]
    return fluxes, flux_errors


def _prepare_lines(line_table, molecule, dedup_tol_um):
    """
    Dedupe co-located lines and build the fit inputs from the kept ones.

    Args:
        line_table (list): Line dicts with 'wavelength', 'a_coeff', 'e_upper', 'Gp'
        molecule (str): Species name, passed to _build_molecular_data
        dedup_tol_um (float): Co-located-line merge tolerance in microns

    Returns:
        result (tuple): (kept, dropped, line_wavelengths, molecular_data, keys)
    """
    kept, dropped = _dedupe_colocated(line_table, dedup_tol_um)
    line_wavelengths = [line["wavelength"] for line in kept]
    keys = _line_keys(kept, molecule)
    molecular_data = _build_molecular_data(kept, keys, molecule)
    return kept, dropped, line_wavelengths, molecular_data, keys


def _fit_spectrum(spectrum, wavelength, line_wavelengths, molecular_data, keys, header,
                  contam_lines, molecule, isotopologue, opr, fit_opr, opr_grid,
                  dist_pc, r_au, min_lines):
    """
    Fit one spectrum's rotational diagram (1T, detections only). Shared by
    _fit_one_pixel (the bulk loop, flat-tuple return) and fit_pixel (one
    GUI-triggered pixel, dict return) -- same extract/detect/fit sequence.

    Args:
        spectrum (array): Flux values, native units
        wavelength (array): Wavelength values (um) for `spectrum`
        line_wavelengths (list): Target line centres (um), wavelength-sorted
        molecular_data (dict): Output of _build_molecular_data for the same lines
        keys (list): Output of _line_keys for the same lines
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
        out (dict): 'status' (int), 'n_lines' (int), and on STATUS_OK also
            'results' (the fitter's dict with '1T'/'_plot_data'); otherwise 'reason'
    """
    if spectrum is None or np.all(np.isnan(spectrum)):
        return {"status": STATUS_NO_DATA, "n_lines": 0, "reason": "no data"}

    per_line = compute_line_fluxes_collective(
        wavelength, spectrum, line_wavelengths, header, contam_lines=contam_lines
    )
    fluxes, flux_errors = _collect_detections(per_line, keys)
    if len(fluxes) < min_lines:
        return {"status": STATUS_TOO_FEW_LINES, "n_lines": len(fluxes),
                "reason": f"{len(fluxes)} detections, need {min_lines}"}

    try:
        results = _run_fitter(fluxes, flux_errors, molecular_data, molecule,
                              isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au)
    except Exception as exc:
        traceback.print_exc()
        return {"status": STATUS_FIT_FAILED, "n_lines": len(fluxes), "reason": str(exc)}

    return {"status": STATUS_OK, "n_lines": len(fluxes), "results": results}


def _fit_one_pixel(spectrum, wavelength, line_wavelengths, molecular_data, keys, header,
                   contam_lines, molecule, isotopologue, opr, fit_opr, opr_grid,
                   dist_pc, r_au, min_lines):
    """
    Fit one pixel and flatten the result for the bulk map loop's arrays.

    Args:
        spectrum (array): Pixel flux values, native units
        wavelength (array): Wavelength values (um) for `spectrum`
        line_wavelengths (list): Target line centres (um), wavelength-sorted
        molecular_data (dict): Output of _build_molecular_data for the same lines
        keys (list): Output of _line_keys for the same lines
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
    out = _fit_spectrum(spectrum, wavelength, line_wavelengths, molecular_data, keys, header,
                        contam_lines, molecule, isotopologue, opr, fit_opr, opr_grid,
                        dist_pc, r_au, min_lines)
    nan = float("nan")
    if out["status"] != STATUS_OK:
        return nan, nan, nan, nan, out["n_lines"], out["status"]
    one_t = out["results"]["1T"]
    return (float(one_t["T"]), float(one_t["dT"]),
            float(one_t["Ncol"]), float(one_t["dNcol"]),
            out["n_lines"], STATUS_OK)


def fit_pixel(cube_or_list, x, y, line_table, molecule="H2", isotopologue=1,
              opr=None, fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None,
              min_lines=3, dedup_tol_um=1e-5, contam_lines=None):
    """
    Fit one pixel and return the full rotational_diagram result for display.

    Args:
        cube_or_list (CubeData or list): One cube, or a one-element list holding it
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
    kept, _dropped, line_wavelengths, molecular_data, keys = _prepare_lines(
        line_table, molecule, dedup_tol_um)
    spectrum = cube.get_pixel_spectrum(x, y)
    return _fit_spectrum(spectrum, wavelength, line_wavelengths, molecular_data, keys,
                         cube.header, contam_lines, molecule, isotopologue,
                         opr, fit_opr, opr_grid, dist_pc, r_au, min_lines)


def fit_region(cube_or_list, line_table, mask=None, molecule="H2", isotopologue=1,
               opr=None, fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None,
               min_lines=3, dedup_tol_um=1e-5, contam_lines=None):
    """
    Fit one rotational diagram to the summed spectrum over `mask`.

    Args:
        cube_or_list (CubeData or list): One cube, or a one-element list holding it
        line_table (list): Line dicts, see create_rotational_diagram_map
        mask (np.ndarray): 2D bool of pixels to sum. None sums every pixel
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, min_lines,
            dedup_tol_um, contam_lines: Same meaning as in fit_pixel

    Returns:
        out (dict): 'status' (int), 'n_pixels' (int), 'n_lines' (int), and on
            STATUS_OK also 'results'; otherwise 'reason'
    """
    cube = _resolve_cube(cube_or_list)
    wavelength, _unit = extract_units(cube.wavelength)
    kept, _dropped, line_wavelengths, molecular_data, keys = _prepare_lines(
        line_table, molecule, dedup_tol_um)

    _n_wave, n_y, n_x = cube.cube.shape
    if mask is None:
        coords = [(x, y) for y in range(n_y) for x in range(n_x)]
    else:
        coords = [(int(x), int(y)) for y, x in zip(*np.where(mask))]

    specs = [np.asarray(s, dtype=float) for s in
             (cube.get_pixel_spectrum(x, y) for x, y in coords) if s is not None]
    if not specs:
        return {"status": STATUS_NO_DATA, "n_pixels": 0, "n_lines": 0,
                "reason": "no pixels to combine"}
    total = np.nansum(np.array(specs), axis=0)

    per_line = compute_line_fluxes_collective(
        wavelength, total, line_wavelengths, cube.header, contam_lines=contam_lines)
    fluxes, flux_errors = _collect_detections(per_line, keys)
    if len(fluxes) < min_lines:
        return {"status": STATUS_TOO_FEW_LINES, "n_pixels": len(specs),
                "n_lines": len(fluxes),
                "reason": f"{len(fluxes)} detections in the summed spectrum, need {min_lines}"}

    try:
        results = _run_fitter(fluxes, flux_errors, molecular_data, molecule,
                              isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au)
    except Exception as exc:
        return {"status": STATUS_FIT_FAILED, "n_pixels": len(specs),
                "n_lines": len(fluxes), "reason": str(exc)}

    return {"status": STATUS_OK, "n_pixels": len(specs), "n_lines": len(fluxes),
            "results": results}


def create_rotational_diagram_map(cube_or_list, line_table, molecule="H2",
                                  isotopologue=1, opr=None, fit_opr=False,
                                  opr_grid=None, dist_pc=140.0, r_au=None,
                                  min_lines=3, dedup_tol_um=1e-5, n_jobs=8,
                                  contam_lines=None, region_fit=None,
                                  progress_callback=None, cancel_flag=None):
    """
    Fit a 1T rotational diagram at every spatial pixel of the cube.

    Args:
        cube_or_list (CubeData or list): One cube, or a one-element list holding it
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
        ValueError: if more than one cube is given -- use
            create_rotational_diagram_map_multi instead
    """
    cube = _resolve_cube(cube_or_list)
    if cube.cube is None:
        print("[ERROR] No cube data loaded")
        return None

    _n_wave, n_y, n_x = cube.cube.shape
    wavelength, _unit = extract_units(cube.wavelength)
    kept, dropped, line_wavelengths, molecular_data, keys = _prepare_lines(
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
                spectrum, wavelength, line_wavelengths, molecular_data, keys,
                cube.header, contam_lines, molecule, isotopologue,
                opr, fit_opr, opr_grid, dist_pc, r_au, min_lines,
            )
            t_row[x], dt_row[x] = t, dt
            n_row[x], dn_row[x] = ncol, dncol
            nl_row[x], st_row[x] = nlines, status
            with lock:
                counter["done"] += 1
        return y, t_row, dt_row, n_row, dn_row, nl_row, st_row

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
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
            print(f"    row {done_count}/{n_y}  ({done}/{total} px, {elapsed:.1f}s)")
            sys.stdout.flush()

    ok = int(np.sum(status_map == STATUS_OK))
    with np.errstate(all="ignore"):
        t_lo, t_hi = np.nanmin(t_map), np.nanmax(t_map)
    print(f">>> Done ({time.time() - start:.1f}s). Fitted {ok}/{total} pixels; "
          f"T range {t_lo:.1f}-{t_hi:.1f} K")

    # Whole-cube-summed anchor fit, the same thing Run Fit does. Reused if
    # the caller already ran it.
    if region_fit is None:
        region_fit = fit_region(
            cube, line_table, mask=None, molecule=molecule,
            isotopologue=isotopologue, opr=opr, fit_opr=fit_opr, opr_grid=opr_grid,
            dist_pc=dist_pc, r_au=r_au, min_lines=min_lines,
            dedup_tol_um=dedup_tol_um, contam_lines=contam_lines,
        )

    return {
        "T": t_map, "T_err": dt_map, "Ncol": n_map, "Ncol_err": dn_map,
        "n_lines": nlines_map, "status": status_map,
        "header": cube.header,
        "wave_range": (float(np.nanmin(wavelength)), float(np.nanmax(wavelength))),
        "molecule": molecule, "dist_pc": dist_pc, "r_au": r_au, "opr": opr,
        "min_lines": min_lines, "t_cap": T_CAP_H2 if molecule == "H2" else T_CAP_GENERAL,
        "region_fit": region_fit, "kept_lines": kept, "dropped_lines": dropped,
    }


def _native_line_map(cube, line_wavelength, contam_lines=None, n_jobs=8,
                     progress_callback=None, cancel_flag=None):
    """
    Fit one line at every pixel of one cube's native grid.

    Args:
        cube (CubeData): One loaded cube
        line_wavelength (float): The line's catalog wavelength (um)
        contam_lines (pd.DataFrame): Contaminating lines, or None
        n_jobs (int): Worker threads, one cube row per task
        progress_callback (callable): progress_callback(done_pixels, total_pixels)
        cancel_flag (object): Object with a `.cancelled` attribute, or None

    Returns:
        maps (tuple): (flux_map, flux_error_map), float32, NaN where there is
            no data or the fit did not succeed
    """
    wavelength, _unit = extract_units(cube.wavelength)
    _n_wave, n_y, n_x = cube.cube.shape
    flux_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    err_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    counter = {"done": 0}
    lock = Lock()
    total = n_y * n_x

    def process_row(y):
        """
        Fit this line in every pixel of one row.

        Args:
            y (int): Row index on the spatial y-axis

        Returns:
            row (tuple): (y, flux_row, err_row), each length n_x
        """
        flux_row = np.full(n_x, np.nan, dtype=np.float32)
        err_row = np.full(n_x, np.nan, dtype=np.float32)
        for x in range(n_x):
            if cancel_flag is not None and cancel_flag.cancelled:
                break
            spectrum = cube.get_pixel_spectrum(x, y)
            if spectrum is None or np.all(np.isnan(spectrum)):
                with lock:
                    counter["done"] += 1
                continue
            per_line = compute_line_fluxes_collective(
                wavelength, spectrum, [line_wavelength], cube.header,
                contam_lines=contam_lines)
            res = per_line[0]
            if res.get("success"):
                flux_row[x] = res["flux_cgs"]
                err_row[x] = res.get("flux_error_cgs", np.nan)
            with lock:
                counter["done"] += 1
        return y, flux_row, err_row

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(process_row, y) for y in range(n_y)]
        for future in as_completed(futures):
            if cancel_flag is not None and cancel_flag.cancelled:
                for f in futures:
                    f.cancel()
                break
            y, flux_row, err_row = future.result()
            flux_map[y], err_map[y] = flux_row, err_row
            if progress_callback is not None:
                with lock:
                    progress_callback(counter["done"], total)
    print(f"    line {line_wavelength:.4f} um: native map done "
          f"({int(np.sum(np.isfinite(flux_map)))}/{total} px fit)")
    return flux_map, err_map


def _convolve_reproject_pair(flux_map, err_map, src_wcs, src_pix_arcsec,
                             dst_wcs, dst_shape, worst_pix_arcsec):
    """
    Match one line's flux/error maps to the common grid: convolve to the
    worst resolution, then reproject flux-conservingly onto dst_wcs.

    Args:
        flux_map (np.ndarray): Native-grid flux map
        err_map (np.ndarray): Native-grid flux-error map
        src_wcs (astropy.wcs.WCS): This map's own celestial WCS
        src_pix_arcsec (float): This map's native pixel scale (arcsec)
        dst_wcs (astropy.wcs.WCS): The common output WCS
        dst_shape (tuple): (ny, nx) of the common grid
        worst_pix_arcsec (float): The coarsest pixel scale among all inputs

    Returns:
        maps (tuple): (flux_common, err_common), NaN outside this map's own
            reprojected footprint
    """
    # Same recipe as gui_2d.py's batch convolution -- resolution-match then
    # flux-conserving reprojection. Both maps get identical treatment.
    sigma_arcsec = (max(0.0, worst_pix_arcsec ** 2 - src_pix_arcsec ** 2)) ** 0.5 / 2.3548
    sigma_pix = sigma_arcsec / src_pix_arcsec if src_pix_arcsec > 0 else 0.0

    flux_blur = gaussian_convolve_match(flux_map, sigma_pix=sigma_pix)
    err_blur = gaussian_convolve_match(err_map, sigma_pix=sigma_pix)

    flux_common, cov1, _ = reproject_exact(flux_blur, src_wcs, dst_wcs, dst_shape, conserve_flux=True)
    err_common, cov2, _ = reproject_exact(err_blur, src_wcs, dst_wcs, dst_shape, conserve_flux=True)
    flux_common[cov1 < 1.0 - 1e-6] = np.nan
    err_common[cov2 < 1.0 - 1e-6] = np.nan
    return flux_common, err_common


def fit_region_multi(cubes_and_lines, molecule="H2", isotopologue=1, opr=None,
                     fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None,
                     min_lines=3, dedup_tol_um=1e-5, contam_lines=None):
    """
    Fit one rotational diagram from the whole-cube summed spectrum of each
    (cube, line) pair. The multi-cube analogue of fit_region: every pixel of
    every cube is summed anyway, so no convolution or reprojection is needed.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs, one per input cube
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, min_lines,
            dedup_tol_um, contam_lines: Same meaning as in fit_pixel

    Returns:
        out (dict): 'status' (int), 'n_lines' (int), 'n_pixels' (total spaxels
            summed, across every cube), 'n_cubes', and on STATUS_OK also
            'results'; otherwise 'reason'
    """
    kept_pairs, _dropped = _dedupe_colocated(list(cubes_and_lines), dedup_tol_um, line_of=lambda p: p[1])
    kept_lines = [l for _, l in kept_pairs]
    keys = _line_keys(kept_lines, molecule)
    molecular_data = _build_molecular_data(kept_lines, keys, molecule)

    fluxes, flux_errors, n_pixels_total = {}, {}, 0
    for key, (cube, line) in zip(keys, kept_pairs):
        wavelength, _unit = extract_units(cube.wavelength)
        _n_wave, n_y, n_x = cube.cube.shape
        specs = [np.asarray(s, dtype=float) for s in
                 (cube.get_pixel_spectrum(x, y) for y in range(n_y) for x in range(n_x))
                 if s is not None]
        if not specs:
            continue
        n_pixels_total += len(specs)
        total = np.nansum(np.array(specs), axis=0)
        per_line = compute_line_fluxes_collective(
            wavelength, total, [line["wavelength"]], cube.header, contam_lines=contam_lines)
        res = per_line[0]
        if res.get("success") and not res.get("is_upper_limit"):
            fluxes[key] = res["flux_cgs"]
            flux_errors[key] = res.get("flux_error_cgs", np.nan)

    if len(fluxes) < min_lines:
        return {"status": STATUS_TOO_FEW_LINES, "n_lines": len(fluxes),
                "n_pixels": n_pixels_total, "n_cubes": len(kept_pairs),
                "reason": f"{len(fluxes)} detections across the cubes, need {min_lines}"}
    try:
        results = _run_fitter(fluxes, flux_errors, molecular_data, molecule, isotopologue,
                              opr, fit_opr, opr_grid, dist_pc, r_au)
    except Exception as exc:
        return {"status": STATUS_FIT_FAILED, "n_lines": len(fluxes),
                "n_pixels": n_pixels_total, "n_cubes": len(kept_pairs), "reason": str(exc)}
    return {"status": STATUS_OK, "n_lines": len(fluxes), "results": results,
            "n_pixels": n_pixels_total, "n_cubes": len(kept_pairs)}


def convolution_plan(cubes_and_lines, dedup_tol_um=1e-5):
    """
    Cheap, fit-free preview of what create_rotational_diagram_map_multi will
    do to each cube: the same pixel-scale/WCS arithmetic it runs up front,
    without any per-pixel fitting.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs, one per input cube
        dedup_tol_um (float): Co-located-line merge tolerance in microns

    Returns:
        plan (dict): 'worst_pix_arcsec', 'dst_shape' (ny, nx), 'per_cube' (list
            of dicts: 'wavelength', 'filename', 'native_pix_arcsec',
            'sigma_arcsec', 'sigma_pix'), 'dropped_lines'

    Raises:
        ValueError: if a cube has no readable celestial WCS/pixel scale, or
            two cubes share no sky coverage
    """
    kept_pairs, dropped = _dedupe_colocated(list(cubes_and_lines), dedup_tol_um, line_of=lambda p: p[1])
    cubes = [c for c, _ in kept_pairs]
    lines = [l for _, l in kept_pairs]

    wcses = [_make_2d_wcs(c.header) for c in cubes]
    shapes = [(c.cube.shape[1], c.cube.shape[2]) for c in cubes]
    scales = [_get_pixel_size_arcsec(c.header) for c in cubes]
    if any(w is None for w in wcses) or any(s is None for s in scales):
        raise ValueError("Could not read a celestial WCS/pixel scale from every cube's header.")
    for i in range(len(cubes)):
        for j in range(i + 1, len(cubes)):
            if not _wcs_footprints_overlap(wcses[i], shapes[i], wcses[j], shapes[j]):
                raise ValueError(
                    f"The cube for {lines[i]['wavelength']:.4f} um and the cube "
                    f"for {lines[j]['wavelength']:.4f} um share no sky coverage.")

    worst_pix = max(scales)
    dst_wcs, dst_shape = build_target_wcs_for_batch(list(zip(wcses, shapes)), worst_pix)

    per_cube = []
    for cube, line, scale in zip(cubes, lines, scales):
        sigma_arcsec = (max(0.0, worst_pix ** 2 - scale ** 2)) ** 0.5 / 2.3548
        sigma_pix = sigma_arcsec / scale if scale > 0 else 0.0
        per_cube.append({
            "wavelength": line["wavelength"],
            "filename": getattr(cube, "filename", None),
            "native_pix_arcsec": scale,
            "sigma_arcsec": sigma_arcsec,
            "sigma_pix": sigma_pix,
        })
    return {"worst_pix_arcsec": worst_pix, "dst_shape": dst_shape,
            "per_cube": per_cube, "dropped_lines": dropped}


def create_rotational_diagram_map_multi(cubes_and_lines, molecule="H2", isotopologue=1,
                                        opr=None, fit_opr=False, opr_grid=None,
                                        dist_pc=140.0, r_au=None, min_lines=3,
                                        dedup_tol_um=1e-5, n_jobs=8, contam_lines=None,
                                        region_fit=None, progress_callback=None,
                                        cancel_flag=None):
    """
    Fit a 1T rotational diagram at every pixel of a common grid built from
    several cubes at different pixel scales, one line per cube. Convolves
    each cube's flux map to the worst resolution, then reprojects before fitting.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs, one per input cube
        molecule (str): Species name; 'H2' uses the OPR-capable fitter
        isotopologue (int): HITRAN isotopologue number, non-H2 only
        opr (float): Fixed ortho-para ratio, or None for the fitter default
        fit_opr (bool): Grid-search OPR per pixel, H2 only
        opr_grid (array): OPR grid for the search, or None
        dist_pc (float): Source distance in parsecs
        r_au (float): Emitting radius in AU, or None (total-molecules mode)
        min_lines (int): Fewest detections a pixel needs for a 1T fit
        dedup_tol_um (float): Co-located-line merge tolerance in microns
        n_jobs (int): Worker threads for each native-grid extraction
        contam_lines (pd.DataFrame): Contaminating lines, or None
        region_fit (dict): Output of fit_region_multi, reused if the caller
            already ran it (the preview does), or None to skip it
        progress_callback (callable): progress_callback(done_units, total_units)
        cancel_flag (object): Object with a `.cancelled` attribute, or None

    Returns:
        result (dict): 'T', 'T_err', 'Ncol', 'Ncol_err' (2D float32), 'n_lines',
            'status' (2D int), plus 'dst_wcs', 'molecule', 'dist_pc', 'r_au',
            'opr', 'min_lines', 't_cap', 'worst_pix_arcsec', 'kept_lines',
            'dropped_lines', 'region_fit'

    Raises:
        ValueError: if fewer than 2 pairs are given, a cube has no readable
            celestial WCS/pixel scale, or two inputs share no sky coverage
    """
    if len(cubes_and_lines) < 2:
        raise ValueError("Need at least 2 (cube, line) pairs -- use "
                         "create_rotational_diagram_map for a single cube.")

    kept_pairs, dropped = _dedupe_colocated(list(cubes_and_lines), dedup_tol_um, line_of=lambda p: p[1])
    for weak_cube, weak_line in dropped:
        print(f"[dedupe] dropped {weak_line['wavelength']:.6f} um "
              f"(A={weak_line.get('a_coeff', 0):.3e})")

    cubes = [c for c, _ in kept_pairs]
    kept_lines = [l for _, l in kept_pairs]
    keys = _line_keys(kept_lines, molecule)
    molecular_data = _build_molecular_data(kept_lines, keys, molecule)

    wcses = [_make_2d_wcs(c.header) for c in cubes]
    shapes = [(c.cube.shape[1], c.cube.shape[2]) for c in cubes]
    scales = [_get_pixel_size_arcsec(c.header) for c in cubes]
    if any(w is None for w in wcses) or any(s is None for s in scales):
        raise ValueError("Could not read a celestial WCS/pixel scale from every cube's header.")
    for i in range(len(cubes)):
        for j in range(i + 1, len(cubes)):
            if not _wcs_footprints_overlap(wcses[i], shapes[i], wcses[j], shapes[j]):
                raise ValueError(
                    f"The cube for {kept_lines[i]['wavelength']:.4f} um and the cube "
                    f"for {kept_lines[j]['wavelength']:.4f} um share no sky coverage.")

    worst_pix = max(scales)
    dst_wcs, dst_shape = build_target_wcs_for_batch(list(zip(wcses, shapes)), worst_pix)
    n_y, n_x = dst_shape
    print(f">>> Multi-cube rotational diagram map: {len(cubes)} cubes, "
          f"worst scale {worst_pix:.4f} arcsec/px, common grid {n_y}x{n_x}")

    # Progress spans every native-grid extraction plus the final per-pixel fit.
    total_units = sum(c.cube.shape[1] * c.cube.shape[2] for c in cubes) + n_y * n_x
    progress_state = {"done": 0}
    progress_lock = Lock()

    def _bump(count):
        """
        Advance the shared progress counter by `count` units.

        Args:
            count (int): Units just completed

        Returns:
            None
        """
        with progress_lock:
            progress_state["done"] += count
            done = progress_state["done"]
        if progress_callback is not None:
            progress_callback(done, total_units)

    flux_maps, err_maps = [], []
    for cube, line, wcs, scale in zip(cubes, kept_lines, wcses, scales):
        seen = {"n": 0}

        def _line_progress(cur, _tot, _seen=seen):
            """
            Convert one cube's own (current, total) into a shared-counter delta.

            Args:
                cur (int): Pixels done in this cube so far
                _tot (int): This cube's own pixel total, unused

            Returns:
                None
            """
            _bump(cur - _seen["n"])
            _seen["n"] = cur

        flux_native, err_native = _native_line_map(
            cube, line["wavelength"], contam_lines=contam_lines, n_jobs=n_jobs,
            progress_callback=_line_progress, cancel_flag=cancel_flag)
        flux_common, err_common = _convolve_reproject_pair(
            flux_native, err_native, wcs, scale, dst_wcs, dst_shape, worst_pix)
        flux_maps.append(flux_common)
        err_maps.append(err_common)

    t_map = np.full(dst_shape, np.nan, dtype=np.float32)
    dt_map = np.full(dst_shape, np.nan, dtype=np.float32)
    n_map = np.full(dst_shape, np.nan, dtype=np.float32)
    dn_map = np.full(dst_shape, np.nan, dtype=np.float32)
    nlines_map = np.zeros(dst_shape, dtype=np.int16)
    status_map = np.full(dst_shape, STATUS_NO_DATA, dtype=np.int16)

    for y in range(n_y):
        if cancel_flag is not None and cancel_flag.cancelled:
            break
        for x in range(n_x):
            fluxes, flux_errors = {}, {}
            for i in range(len(kept_lines)):
                f, e = flux_maps[i][y, x], err_maps[i][y, x]
                # The same 3-sigma detection idea used everywhere else in this
                # feature, applied to the convolved flux/error pair.
                if np.isfinite(f) and np.isfinite(e) and e > 0 and f > 3 * e:
                    fluxes[keys[i]] = float(f)
                    flux_errors[keys[i]] = float(e)

            if len(fluxes) < min_lines:
                status_map[y, x] = STATUS_NO_DATA if not fluxes else STATUS_TOO_FEW_LINES
                nlines_map[y, x] = len(fluxes)
            else:
                try:
                    result = _run_fitter(fluxes, flux_errors, molecular_data, molecule,
                                         isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au)
                    one_t = result["1T"]
                    t_map[y, x] = float(one_t["T"])
                    dt_map[y, x] = float(one_t["dT"])
                    n_map[y, x] = float(one_t["Ncol"])
                    dn_map[y, x] = float(one_t["dNcol"])
                    status_map[y, x] = STATUS_OK
                except Exception:
                    traceback.print_exc()
                    status_map[y, x] = STATUS_FIT_FAILED
                nlines_map[y, x] = len(fluxes)
            _bump(1)

    ok = int(np.sum(status_map == STATUS_OK))
    print(f">>> Done. Fitted {ok}/{n_y * n_x} pixels on the common grid.")

    # Whole-cube-summed anchor fit, the same thing Run Fit does. Reused if
    # the caller already ran it.
    if region_fit is None:
        region_fit = fit_region_multi(
            kept_pairs, molecule=molecule, isotopologue=isotopologue, opr=opr,
            fit_opr=fit_opr, opr_grid=opr_grid, dist_pc=dist_pc, r_au=r_au,
            min_lines=min_lines, dedup_tol_um=dedup_tol_um, contam_lines=contam_lines,
        )

    return {
        "T": t_map, "T_err": dt_map, "Ncol": n_map, "Ncol_err": dn_map,
        "n_lines": nlines_map, "status": status_map,
        "dst_wcs": dst_wcs,
        "molecule": molecule, "dist_pc": dist_pc, "r_au": r_au, "opr": opr,
        "min_lines": min_lines, "t_cap": T_CAP_H2 if molecule == "H2" else T_CAP_GENERAL,
        "worst_pix_arcsec": worst_pix, "kept_lines": kept_lines, "dropped_lines": dropped,
        "region_fit": region_fit,
        # Each line's convolved/reprojected map, kept so fit_pixel_multi can
        # re-fit one pixel from a cheap lookup instead of redoing convolution.
        "_flux_maps": flux_maps, "_err_maps": err_maps, "_keys": keys,
        "_molecular_data": molecular_data,
    }


def fit_pixel_multi(result, x, y, molecule="H2", isotopologue=1, opr=None,
                    fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None,
                    min_lines=3):
    """
    Re-fit one pixel of a convolved/reprojected map from its cached per-line
    flux/error maps instead of redoing convolution -- a blurred pixel depends
    on its native neighbours, so it can't be looked up alone.

    Args:
        result (dict): Output of create_rotational_diagram_map_multi
        x (int): Spatial x index on the common grid
        y (int): Spatial y index on the common grid
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au,
            min_lines: Same meaning as in fit_pixel

    Returns:
        out (dict): Same shape as fit_pixel's return -- 'status', 'n_lines',
            and on STATUS_OK also 'results' (with '1T' and '_plot_data')

    Raises:
        KeyError: if result has no cached per-line maps (predates this
            function, or came from a different map type)
    """
    flux_maps = result["_flux_maps"]
    err_maps = result["_err_maps"]
    keys = result["_keys"]
    molecular_data = result["_molecular_data"]

    fluxes, flux_errors = {}, {}
    for i, key in enumerate(keys):
        f, e = flux_maps[i][y, x], err_maps[i][y, x]
        # Same 3-sigma detection rule used everywhere else in this feature,
        # on the same convolved flux/error pair the map loop already used.
        if np.isfinite(f) and np.isfinite(e) and e > 0 and f > 3 * e:
            fluxes[key] = float(f)
            flux_errors[key] = float(e)

    if len(fluxes) < min_lines:
        status = STATUS_NO_DATA if not fluxes else STATUS_TOO_FEW_LINES
        return {"status": status, "n_lines": len(fluxes),
                "reason": f"{len(fluxes)} detections, need {min_lines}"}

    try:
        results = _run_fitter(fluxes, flux_errors, molecular_data, molecule,
                              isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au)
    except Exception as exc:
        return {"status": STATUS_FIT_FAILED, "n_lines": len(fluxes), "reason": str(exc)}

    return {"status": STATUS_OK, "n_lines": len(fluxes), "results": results}


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

    # Multi-cube has a real common WCS (dst_wcs); single-cube only has its
    # own header. Either way it goes on every image HDU, not just the primary.
    wcs_header = None
    if result.get("dst_wcs") is not None:
        wcs_header = result["dst_wcs"].to_header()
    elif result.get("header") is not None:
        wcs_header = result["header"]

    hdus = [fits.PrimaryHDU()]
    for key in ("T", "T_err", "Ncol", "Ncol_err", "n_lines", "status"):
        hdu = fits.ImageHDU(np.asarray(result[key]), name=key.upper())
        if wcs_header is not None:
            hdu.header.update(wcs_header)
        hdus.append(hdu)
    hdus[1].header["BUNIT"] = "K"
    hdus[3].header["BUNIT"] = "cm-2"
    hdus[0].header["MOLECULE"] = result["molecule"]
    hdus[0].header["DIST_PC"] = result["dist_pc"]
    if result.get("r_au") is not None:
        hdus[0].header["R_AU"] = result["r_au"]
    if result.get("worst_pix_arcsec") is not None:
        hdus[0].header["PIXSCALE"] = (result["worst_pix_arcsec"], "arcsec/pixel, worst input")
    fits.HDUList(hdus).writeto(output_path, overwrite=True)
    print(f"[OK] wrote {output_path}")
