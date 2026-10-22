"""
Pixel-by-pixel rotational diagram map: runs the region-level fit at every
spatial pixel of one cube. Several cubes always go through the _multi
functions' convolve+reproject path onto a common grid -- registration is
never assumed from a tolerance check, only verified by construction.
Detections only -- a line below the n-sigma floor (3 by default) is dropped.
"""

import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from threading import Lock

import numpy as np
from astropy.io import fits

from .unit_converter import extract_units
from .utils import safe_worker_count, detect_instrument, get_pixel_size_arcsec, make_2d_wcs, wcs_footprints_overlap
from .line_catalog import dedupe_colocated_lines as _dedupe_colocated
from .convolve_reproject import convolve_reproject_maps
from .fits_export import build_2d_export_header, build_line_map_header, write_fits_product
from .rotational_diagram_h2_fitter import (
    compute_line_fluxes_collective, get_gu, T_CAP as T_CAP_H2,
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

# Per-pixel codes in a native line map's status map (Calculate Maps). Same
# 0/1/2 order as the Gaussian map's Fit Status, so both read alike.
NATIVE_PX_NONE = 0           # no fit: empty spectrum, failed fit, or outside a region
NATIVE_PX_UL = 1             # fit succeeded but is below the n-sigma floor
NATIVE_PX_DETECTION = 2      # fit succeeded and is at or above the n-sigma floor

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
                opr, fit_opr, opr_grid, dist_pc, r_au, fit2t=False):
    """
    Call the right rotational_diagram fitter (H2 or general).

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
        fit2t (bool): Report the two-temperature fit instead of 1T, same
            default Run Fit uses (report 1T only when 2T wasn't requested)

    Returns:
        results (dict): The fitter's result dict, with '1T' or '2T' (whichever
            was requested) plus '_plot_data'
    """
    if molecule == "H2":
        return rotational_diagram_h2(
            fluxes, flux_errors, molecular_data=molecular_data, OPR=opr,
            fit_opr=fit_opr, opr_grid=opr_grid, fit1T=not fit2t, fit2T=fit2t,
            R=r_au, dist=dist_pc, upper_limits=None,
        )
    return rotational_diagram_general(
        fluxes, flux_errors, molecular_data=molecular_data, fit1T=not fit2t,
        fit2T=fit2t, R=r_au, dist=dist_pc, upper_limits=None,
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
                  dist_pc, r_au, min_lines, n_sigma=3.0, fit2t=False):
    """
    Fit one spectrum's rotational diagram (detections only). Shared by
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
        min_lines (int): Fewest detections this fit needs
        n_sigma (float): Detection threshold in noise sigmas, 3.0 by default
        fit2t (bool): Report the two-temperature fit instead of 1T

    Returns:
        out (dict): 'status' (int), 'n_lines' (int), and on STATUS_OK also
            'results' (the fitter's dict with '1T' or '2T' plus '_plot_data');
            otherwise 'reason'. Also has 'per_line' (compute_line_fluxes_collective's
            own per-line output, same order as line_wavelengths) whenever a fit was
            attempted at all, even if it didn't reach STATUS_OK -- callers that
            want each line's own flux (not just the combined T fit) read it
            from there instead of re-fitting.
    """
    if spectrum is None or np.all(np.isnan(spectrum)):
        return {"status": STATUS_NO_DATA, "n_lines": 0, "reason": "no data"}

    per_line = compute_line_fluxes_collective(
        wavelength, spectrum, line_wavelengths, header, contam_lines=contam_lines,
        n_sigma=n_sigma,
    )
    fluxes, flux_errors = _collect_detections(per_line, keys)
    if len(fluxes) < min_lines:
        return {"status": STATUS_TOO_FEW_LINES, "n_lines": len(fluxes),
                "reason": f"{len(fluxes)} detections, need {min_lines}", "per_line": per_line}

    try:
        results = _run_fitter(fluxes, flux_errors, molecular_data, molecule,
                              isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au,
                              fit2t=fit2t)
    except Exception as exc:
        traceback.print_exc()
        return {"status": STATUS_FIT_FAILED, "n_lines": len(fluxes), "reason": str(exc), "per_line": per_line}

    return {"status": STATUS_OK, "n_lines": len(fluxes), "results": results, "per_line": per_line}


def _fit_one_pixel(spectrum, wavelength, line_wavelengths, molecular_data, keys, header,
                   contam_lines, molecule, isotopologue, opr, fit_opr, opr_grid,
                   dist_pc, r_au, min_lines, n_sigma=3.0, fit2t=False):
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
        min_lines (int): Fewest detections this pixel's fit needs
        n_sigma (float): Detection threshold in noise sigmas, 3.0 by default
        fit2t (bool): Report the two-temperature fit instead of 1T

    Returns:
        row (tuple): (T, dT, Ncol, dNcol, T2, dT2, Ncol2, dNcol2, n_lines,
            status, line_fluxes, line_flux_errors). T/dT/Ncol/dNcol hold
            component 1 (1T's only component, or 2T's T1); T2/dT2/
            Ncol2/dNcol2 hold 2T's T2 component, NaN when fit2t is False.
            All eight are NaN unless status is STATUS_OK. line_fluxes/
            line_flux_errors are arrays, one entry per line_wavelengths entry
            (same order), each line's own flux at this pixel -- NaN wherever
            that particular line's fit didn't succeed or came back below the
            n_sigma detection floor (an upper limit, not a real value for
            this pixel), independent of whether the pixel had enough
            detections for a T fit.
    """
    nan = float("nan")
    line_fluxes = np.full(len(line_wavelengths), nan)
    line_flux_errors = np.full(len(line_wavelengths), nan)

    out = _fit_spectrum(spectrum, wavelength, line_wavelengths, molecular_data, keys, header,
                        contam_lines, molecule, isotopologue, opr, fit_opr, opr_grid,
                        dist_pc, r_au, min_lines, n_sigma, fit2t=fit2t)
    per_line = out.get("per_line")
    if per_line is not None:
        for i, res in enumerate(per_line):
            # Detections only -- a fit that merely converged on noise (an
            # upper limit at this pixel) must not leave a value in the map.
            if res.get("success") and not res.get("is_upper_limit"):
                line_fluxes[i] = res["flux_cgs"]
                line_flux_errors[i] = res.get("flux_error_cgs", nan)

    if out["status"] != STATUS_OK:
        return (nan, nan, nan, nan, nan, nan, nan, nan,
                out["n_lines"], out["status"], line_fluxes, line_flux_errors)
    if fit2t:
        two_t = out["results"]["2T"]
        return (float(two_t["T1"]), float(two_t["dT1"]),
                float(two_t["Ncol1"]), float(two_t["dNcol1"]),
                float(two_t["T2"]), float(two_t["dT2"]),
                float(two_t["Ncol2"]), float(two_t["dNcol2"]),
                out["n_lines"], STATUS_OK, line_fluxes, line_flux_errors)
    one_t = out["results"]["1T"]
    return (float(one_t["T"]), float(one_t["dT"]),
            float(one_t["Ncol"]), float(one_t["dNcol"]),
            nan, nan, nan, nan,
            out["n_lines"], STATUS_OK, line_fluxes, line_flux_errors)


def _pin_worker_math_threads():
    """
    Limit every future child process's math library (OpenBLAS/OMP/MKL) to one
    thread each, so N worker processes don't each also fan out to every CPU
    core (measured 5-6x slower than threads when left at the default). Must
    run before a ProcessPoolExecutor is created -- with spawn, each child is a
    fresh interpreter that reads these at its own numpy import, and setting
    them this late no longer affects the already-running parent process's own
    BLAS (it was configured once, at the parent's own first numpy import).

    Args:
        None

    Returns:
        None
    """
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _fit_row_worker(row_idx, spectra_row, wavelength, line_wavelengths, molecular_data, keys,
                    header, contam_lines, molecule, isotopologue, opr, fit_opr, opr_grid,
                    dist_pc, r_au, min_lines, n_lines_total, row_mask=None, n_sigma=3.0,
                    fit2t=False):
    """
    Fit every pixel in one cube row, in a worker process. Module-level so it
    can be pickled for ProcessPoolExecutor; everything process_row's closure
    used to capture is passed explicitly instead.

    Args:
        row_idx (int): Row index on the spatial y-axis
        spectra_row (np.ndarray): This row's pixel spectra, shape (n_wave, n_x),
            float32 -- matches CubeData.get_pixel_spectrum's own dtype exactly
        wavelength, line_wavelengths, molecular_data, keys, header, contam_lines,
            molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au,
            min_lines: Same meaning as create_rotational_diagram_map's own args
        n_lines_total (int): len(line_wavelengths), for preallocating the
            per-line arrays
        row_mask (np.ndarray): Boolean mask, length n_x, True where this row's
            pixels should be fit. None fits every pixel (no region imported).
        n_sigma (float): Detection threshold in noise sigmas, 3.0 by default
        fit2t (bool): Report the two-temperature fit instead of 1T

    Returns:
        row (tuple): (row_idx, T, dT, Ncol, dNcol, T2, dT2, Ncol2, dNcol2,
            n_lines, status, line_fluxes, line_flux_errors) -- T/dT/Ncol/dNcol/
            T2/dT2/Ncol2/dNcol2/n_lines/status length n_x, line_fluxes/
            line_flux_errors shape (n_lines_total, n_x). T2/dT2/Ncol2/dNcol2
            stay all-NaN when fit2t is False.
    """
    n_x = spectra_row.shape[1]
    t_row = np.full(n_x, np.nan, dtype=np.float32)
    dt_row = np.full(n_x, np.nan, dtype=np.float32)
    n_row = np.full(n_x, np.nan, dtype=np.float32)
    dn_row = np.full(n_x, np.nan, dtype=np.float32)
    t2_row = np.full(n_x, np.nan, dtype=np.float32)
    dt2_row = np.full(n_x, np.nan, dtype=np.float32)
    n2_row = np.full(n_x, np.nan, dtype=np.float32)
    dn2_row = np.full(n_x, np.nan, dtype=np.float32)
    nl_row = np.zeros(n_x, dtype=np.int16)
    st_row = np.full(n_x, STATUS_NO_DATA, dtype=np.int16)
    line_flux_row = np.full((n_lines_total, n_x), np.nan, dtype=np.float32)
    line_flux_err_row = np.full((n_lines_total, n_x), np.nan, dtype=np.float32)

    for x in range(n_x):
        if row_mask is not None and not row_mask[x]:
            continue
        spectrum = spectra_row[:, x]
        (t, dt, ncol, dncol, t2, dt2, ncol2, dncol2, nlines, status,
         line_fluxes, line_flux_errors) = _fit_one_pixel(
            spectrum, wavelength, line_wavelengths, molecular_data, keys,
            header, contam_lines, molecule, isotopologue,
            opr, fit_opr, opr_grid, dist_pc, r_au, min_lines, n_sigma,
            fit2t=fit2t,
        )
        t_row[x], dt_row[x] = t, dt
        n_row[x], dn_row[x] = ncol, dncol
        t2_row[x], dt2_row[x] = t2, dt2
        n2_row[x], dn2_row[x] = ncol2, dncol2
        nl_row[x], st_row[x] = nlines, status
        line_flux_row[:, x] = line_fluxes
        line_flux_err_row[:, x] = line_flux_errors
    return (row_idx, t_row, dt_row, n_row, dn_row, t2_row, dt2_row, n2_row, dn2_row,
            nl_row, st_row, line_flux_row, line_flux_err_row)


def fit_pixel(cube_or_list, x, y, line_table, molecule="H2", isotopologue=1,
              opr=None, fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None,
              min_lines=3, dedup_tol_um=1e-5, contam_lines=None, n_sigma=3.0,
              fit2t=False):
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
        min_lines (int): Fewest detections this fit needs
        dedup_tol_um (float): Co-located-line merge tolerance in microns
        contam_lines (pd.DataFrame): Contaminating lines, or None
        n_sigma (float): Detection threshold in noise sigmas, 3.0 by default
        fit2t (bool): Report the two-temperature fit instead of 1T

    Returns:
        out (dict): 'status' (int), 'n_lines' (int), and on STATUS_OK also
            'results' (the fitter's dict with '1T' or '2T' plus '_plot_data');
            otherwise 'reason'
    """
    cube = _resolve_cube(cube_or_list)
    wavelength, _unit = extract_units(cube.wavelength)
    kept, _dropped, line_wavelengths, molecular_data, keys = _prepare_lines(
        line_table, molecule, dedup_tol_um)
    spectrum = cube.get_pixel_spectrum(x, y)
    return _fit_spectrum(spectrum, wavelength, line_wavelengths, molecular_data, keys,
                         cube.header, contam_lines, molecule, isotopologue,
                         opr, fit_opr, opr_grid, dist_pc, r_au, min_lines, n_sigma,
                         fit2t=fit2t)


def fit_region(cube_or_list, line_table, mask=None, molecule="H2", isotopologue=1,
               opr=None, fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None,
               min_lines=3, dedup_tol_um=1e-5, contam_lines=None, fit2t=False):
    """
    Fit one rotational diagram to the summed spectrum over `mask`.

    Args:
        cube_or_list (CubeData or list): One cube, or a one-element list holding it
        line_table (list): Line dicts, see create_rotational_diagram_map
        mask (np.ndarray): 2D bool of pixels to sum. None sums every pixel
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, min_lines,
            dedup_tol_um, contam_lines: Same meaning as in fit_pixel
        fit2t (bool): Report the two-temperature fit instead of 1T

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
                              isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au,
                              fit2t=fit2t)
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
                                  progress_callback=None, cancel_flag=None, region_mask=None,
                                  n_sigma=3.0, fit2t=False):
    """
    Fit a rotational diagram at every spatial pixel of the cube.

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
        min_lines (int): Fewest detections a pixel needs for a fit. Callers
            must raise this for fit2t (a 2T fit has 4 free parameters, not 2)
        dedup_tol_um (float): Co-located-line merge tolerance in microns
        n_jobs (int): Worker processes, one cube row per task
        contam_lines (pd.DataFrame): Contaminating lines, or None
        progress_callback (callable): progress_callback(done_pixels, total_pixels)
        cancel_flag (object): Object with a `.cancelled` attribute, or None
        region_mask (np.ndarray): Boolean mask, same (ny, nx) shape as the
            cube, True where a pixel should be fit. None fits the whole cube
            (no region imported) -- the previous, unrestricted behavior. Also
            restricts region_fit's own summed spectrum to the same pixels.
        n_sigma (float): Detection threshold in noise sigmas, 3.0 by default.
            A line below it is an upper limit and stays NaN in the maps. Does
            not affect region_fit, which keeps a fixed 3-sigma rule.
        fit2t (bool): Report the two-temperature fit instead of 1T, at every pixel

    Returns:
        result (dict): 'T', 'T_err', 'Ncol', 'Ncol_err' (2D float32, component
            1 -- 1T's only component, or 2T's T1), 'T2', 'T2_err',
            'Ncol2', 'Ncol2_err' (2D float32, 2T's T2 component, all-NaN
            when fit2t is False), 'fit2t', 'n_lines', 'status' (2D int), plus
            'header', 'wave_range', 'molecule', 'dist_pc', 'r_au', 'opr',
            'kept_lines', 'dropped_lines', 'line_flux_maps',
            'line_flux_error_maps' (each a list of 2D float32 arrays, one per
            kept_lines entry, same order -- that line's own flux/error at every
            pixel, independent of whether the pixel got a T fit)

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

    n_lines_total = len(kept)
    print(f">>> Rotational diagram map: {n_y}x{n_x} pixels, {len(kept)} lines, "
          f"molecule={molecule}, min_lines={min_lines}, R={r_au}, n_sigma={n_sigma:g}")

    t_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    dt_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    n_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    dn_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    t2_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    dt2_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    n2_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    dn2_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    nlines_map = np.zeros((n_y, n_x), dtype=np.int16)
    status_map = np.full((n_y, n_x), STATUS_NO_DATA, dtype=np.int16)
    # Per-line flux/error maps: each line's own AUC at every pixel, the same
    # value _fit_one_pixel already fits along the way to the T map -- not
    # recomputed, just kept instead of discarded.
    line_flux_maps = np.full((n_lines_total, n_y, n_x), np.nan, dtype=np.float32)
    line_flux_error_maps = np.full((n_lines_total, n_y, n_x), np.nan, dtype=np.float32)

    total = n_y * n_x
    completed = 0
    done_count = 0
    start = time.time()

    # Rows run in separate processes -- the per-pixel work (continuum fit,
    # window auto-detection, curve_fit) is mostly Python code holding the
    # GIL, so threads measured no speedup (same finding as the Gaussian map's
    # own switch to processes).
    _pin_worker_math_threads()
    n_jobs = safe_worker_count(n_jobs)
    max_in_flight = 2 * n_jobs  # a large cube is never copied out all at once
    cancelled = False
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = set()
        next_row = 0
        while (next_row < n_y or pending) and not cancelled:
            while next_row < n_y and len(pending) < max_in_flight:
                spectra_row = cube.cube[:, next_row, :].astype(np.float32, copy=False)
                row_mask = region_mask[next_row, :] if region_mask is not None else None
                pending.add(executor.submit(
                    _fit_row_worker, next_row, spectra_row, wavelength, line_wavelengths,
                    molecular_data, keys, cube.header, contam_lines, molecule, isotopologue,
                    opr, fit_opr, opr_grid, dist_pc, r_au, min_lines, n_lines_total, row_mask,
                    n_sigma, fit2t,
                ))
                next_row += 1
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                (y, t_row, dt_row, n_row, dn_row, t2_row, dt2_row, n2_row, dn2_row,
                 nl_row, st_row, line_flux_row, line_flux_err_row) = fut.result()
                t_map[y], dt_map[y] = t_row, dt_row
                n_map[y], dn_map[y] = n_row, dn_row
                t2_map[y], dt2_map[y] = t2_row, dt2_row
                n2_map[y], dn2_map[y] = n2_row, dn2_row
                nlines_map[y], status_map[y] = nl_row, st_row
                line_flux_maps[:, y, :] = line_flux_row
                line_flux_error_maps[:, y, :] = line_flux_err_row
                done_count += 1
                completed += n_x
                if progress_callback is not None:
                    progress_callback(completed, total)
                elapsed = time.time() - start
                print(f"    row {done_count}/{n_y}  ({completed}/{total} px, {elapsed:.1f}s)")
                sys.stdout.flush()

            if cancel_flag is not None and cancel_flag.cancelled:
                cancelled = True
                # Drops rows that have not started yet; the pool's own __exit__
                # still waits for in-flight rows to finish first.
                for f in pending:
                    f.cancel()

    ok = int(np.sum(status_map == STATUS_OK))
    with np.errstate(all="ignore"):
        t_lo, t_hi = np.nanmin(t_map), np.nanmax(t_map)
    print(f">>> Done ({time.time() - start:.1f}s). Fitted {ok}/{total} pixels; "
          f"T range {t_lo:.1f}-{t_hi:.1f} K")

    # Whole-cube/region anchor fit, same mask as the per-pixel maps above, so
    # it stays "same as Run Fit" either way. Reused if the caller already ran it.
    if region_fit is None:
        region_fit = fit_region(
            cube, line_table, mask=region_mask, molecule=molecule,
            isotopologue=isotopologue, opr=opr, fit_opr=fit_opr, opr_grid=opr_grid,
            dist_pc=dist_pc, r_au=r_au, min_lines=min_lines,
            dedup_tol_um=dedup_tol_um, contam_lines=contam_lines, fit2t=fit2t,
        )

    return {
        "T": t_map, "T_err": dt_map, "Ncol": n_map, "Ncol_err": dn_map,
        "T2": t2_map, "T2_err": dt2_map, "Ncol2": n2_map, "Ncol2_err": dn2_map,
        "fit2t": fit2t, "n_lines": nlines_map, "status": status_map,
        "header": cube.header,
        "cube_filename": getattr(cube, "filename", None),
        "wave_range": (float(np.nanmin(wavelength)), float(np.nanmax(wavelength))),
        "molecule": molecule, "dist_pc": dist_pc, "r_au": r_au, "opr": opr,
        "min_lines": min_lines, "t_cap": T_CAP_H2 if molecule == "H2" else T_CAP_GENERAL,
        "region_fit": region_fit, "kept_lines": kept, "dropped_lines": dropped,
        "line_flux_maps": [line_flux_maps[i] for i in range(n_lines_total)],
        "line_flux_error_maps": [line_flux_error_maps[i] for i in range(n_lines_total)],
    }


def _native_line_row_worker(row_idx, spectra_row, wavelength, line_wavelength, header, contam_lines,
                            row_mask=None, n_sigma=3.0):
    """
    Fit one line in every pixel of one cube row, in a worker process.
    Module-level so it can be pickled for ProcessPoolExecutor.

    Args:
        row_idx (int): Row index on the spatial y-axis
        spectra_row (np.ndarray): This row's pixel spectra, shape (n_wave, n_x),
            float32 -- matches CubeData.get_pixel_spectrum's own dtype exactly
        wavelength (array): Wavelength values (um) for spectra_row
        line_wavelength (float): The line's catalog wavelength (um)
        header (astropy.io.fits.Header): Cube header, for instrument and CGS conversion
        contam_lines (pd.DataFrame): Contaminating lines, or None
        row_mask (np.ndarray): Boolean mask, length n_x, True where this row's
            pixels should be fit. None fits every pixel (no region imported).
        n_sigma (float): Detection threshold in noise sigmas, 3.0 by default

    Returns:
        row (tuple): (row_idx, flux_row, err_row, fit_row, status_row,
            ceiling_row), each length n_x. flux_row/err_row are NaN wherever
            the fit didn't succeed, came back below the n_sigma detection
            floor (an upper limit, not a real value at that pixel), or fell
            outside an imported region. fit_row holds the fitted flux at
            every successful fit, upper limits included; status_row holds
            the NATIVE_PX_* code (uint8); ceiling_row holds the n_sigma flux
            ceiling wherever a fit succeeded
    """
    n_x = spectra_row.shape[1]
    flux_row = np.full(n_x, np.nan, dtype=np.float32)
    err_row = np.full(n_x, np.nan, dtype=np.float32)
    fit_row = np.full(n_x, np.nan, dtype=np.float32)
    ceiling_row = np.full(n_x, np.nan, dtype=np.float32)
    status_row = np.full(n_x, NATIVE_PX_NONE, dtype=np.uint8)
    for x in range(n_x):
        if row_mask is not None and not row_mask[x]:
            continue
        spectrum = spectra_row[:, x]
        if np.all(np.isnan(spectrum)):
            continue
        per_line = compute_line_fluxes_collective(
            wavelength, spectrum, [line_wavelength], header, contam_lines=contam_lines,
            n_sigma=n_sigma)
        res = per_line[0]
        if not res.get("success"):
            continue
        # The fitted value of an upper limit goes only in fit_row; flux_row
        # stays detections-only, since Convolve Maps and Run Map read it.
        fit_row[x] = res["flux_cgs"]
        ceiling_row[x] = res["F_ul_cgs"]
        if res.get("is_upper_limit"):
            status_row[x] = NATIVE_PX_UL
        else:
            status_row[x] = NATIVE_PX_DETECTION
            flux_row[x] = res["flux_cgs"]
            err_row[x] = res.get("flux_error_cgs", np.nan)
    return row_idx, flux_row, err_row, fit_row, status_row, ceiling_row


def _native_line_map(cube, line_wavelength, contam_lines=None, n_jobs=8,
                     progress_callback=None, cancel_flag=None, executor=None, region_mask=None,
                     n_sigma=3.0):
    """
    Fit one line at every pixel of one cube's native grid.

    Args:
        cube (CubeData): One loaded cube
        line_wavelength (float): The line's catalog wavelength (um)
        contam_lines (pd.DataFrame): Contaminating lines, or None
        n_jobs (int): Worker processes, one cube row per task -- ignored when
            `executor` is given (its own worker count applies instead)
        progress_callback (callable): progress_callback(done_pixels, total_pixels)
        cancel_flag (object): Object with a `.cancelled` attribute, or None
        executor (ProcessPoolExecutor): Shared pool to submit this line's row
            tasks to, so compute_native_line_maps can fit every line through
            one set of worker processes instead of spawning a fresh set per
            line -- each spawn re-imports the whole module chain (including
            pymc, which is slow and prints its own startup warnings on every
            import). None creates and tears down a private pool for this one
            line, for standalone/backward-compat callers.
        region_mask (np.ndarray): Boolean mask, same (ny, nx) shape as this
            cube, True where a pixel should be fit. None fits the whole cube
            (no region imported) -- the previous, unrestricted behavior.
        n_sigma (float): Detection threshold in noise sigmas, 3.0 by default

    Returns:
        maps (tuple): (flux_map, flux_error_map, fit_flux_map, status_map,
            ceiling_map). flux_map/flux_error_map are float32, NaN where
            there is no data, the fit did not succeed, the line is below
            n_sigma, or the pixel is outside an imported region.
            fit_flux_map (float32) also holds the fitted flux of the
            below-n_sigma pixels; status_map (uint8) holds the NATIVE_PX_*
            code; ceiling_map (float32) holds the n_sigma flux ceiling
    """
    wavelength, _unit = extract_units(cube.wavelength)
    _n_wave, n_y, n_x = cube.cube.shape
    flux_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    err_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    fit_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    ceiling_map = np.full((n_y, n_x), np.nan, dtype=np.float32)
    status_map = np.full((n_y, n_x), NATIVE_PX_NONE, dtype=np.uint8)
    total = n_y * n_x
    completed = 0

    # Same reasoning as create_rotational_diagram_map: threads gave no
    # speedup for this per-pixel curve_fit work, so this uses processes too.
    own_executor = executor is None
    if own_executor:
        _pin_worker_math_threads()
        n_jobs = safe_worker_count(n_jobs)
        executor = ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn"))
    max_in_flight = 2 * n_jobs
    cancelled = False
    try:
        pending = set()
        next_row = 0
        while (next_row < n_y or pending) and not cancelled:
            while next_row < n_y and len(pending) < max_in_flight:
                spectra_row = cube.cube[:, next_row, :].astype(np.float32, copy=False)
                row_mask = region_mask[next_row, :] if region_mask is not None else None
                pending.add(executor.submit(
                    _native_line_row_worker, next_row, spectra_row, wavelength,
                    line_wavelength, cube.header, contam_lines, row_mask, n_sigma,
                ))
                next_row += 1
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                y, flux_row, err_row, fit_row, status_row, ceiling_row = fut.result()
                flux_map[y], err_map[y] = flux_row, err_row
                fit_map[y], status_map[y], ceiling_map[y] = fit_row, status_row, ceiling_row
                completed += n_x
                if progress_callback is not None:
                    progress_callback(completed, total)

            if cancel_flag is not None and cancel_flag.cancelled:
                cancelled = True
                for f in pending:
                    f.cancel()
    finally:
        if own_executor:
            executor.shutdown()
    print(f"    line {line_wavelength:.4f} um: native map done "
          f"({int(np.sum(np.isfinite(flux_map)))}/{total} px fit)")
    return flux_map, err_map, fit_map, status_map, ceiling_map


def fit_region_multi(cubes_and_lines, molecule="H2", isotopologue=1, opr=None,
                     fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None,
                     min_lines=3, dedup_tol_um=1e-5, contam_lines=None, region_masks=None,
                     fit2t=False):
    """
    Fit one rotational diagram from the whole-cube (or region-restricted)
    summed spectrum of each (cube, line) pair. The multi-cube analogue of
    fit_region: every pixel of every cube (or every pixel inside its mask) is
    summed anyway, so no convolution or reprojection is needed.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs, one per input cube
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, min_lines,
            dedup_tol_um, contam_lines: Same meaning as in fit_pixel
        region_masks (dict): {CubeData: boolean mask (ny, nx)}, restricting
            which pixels get summed for that cube -- from an imported region.
            A cube missing from this dict, or region_masks=None, sums every
            pixel (the previous, unrestricted behavior).
        fit2t (bool): Report the two-temperature fit instead of 1T

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
        mask = region_masks.get(cube) if region_masks else None
        if mask is None:
            coords = [(x, y) for y in range(n_y) for x in range(n_x)]
        else:
            coords = [(int(x), int(y)) for y, x in zip(*np.where(mask))]
        specs = [np.asarray(s, dtype=float) for s in
                 (cube.get_pixel_spectrum(x, y) for x, y in coords)
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
                              opr, fit_opr, opr_grid, dist_pc, r_au, fit2t=fit2t)
    except Exception as exc:
        return {"status": STATUS_FIT_FAILED, "n_lines": len(fluxes),
                "n_pixels": n_pixels_total, "n_cubes": len(kept_pairs), "reason": str(exc)}
    return {"status": STATUS_OK, "n_lines": len(fluxes), "results": results,
            "n_pixels": n_pixels_total, "n_cubes": len(kept_pairs)}


def fit_region_mixed(cubes_and_lines, imported_lines, molecule="H2", isotopologue=1, opr=None,
                     fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None,
                     min_lines=3, dedup_tol_um=1e-5, contam_lines=None, region_masks=None,
                     imported_region_masks=None, fit2t=False):
    """
    fit_region_multi's mixed-source sibling: cube-backed pairs are fit the
    same way; each imported line's already-reduced map is summed directly
    instead, since there's nothing left to fit.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs, one per
            cube-backed line. May be empty (a pure-import run).
        imported_lines (list): Line dicts with an 'imported_map' payload.
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au,
            min_lines, dedup_tol_um, contam_lines: Same meaning as fit_region_multi
        region_masks (dict): {CubeData: boolean mask}, same as fit_region_multi
        imported_region_masks (dict): {id(line): boolean mask (ny, nx)},
            restricting which pixels of an imported line's own map get
            summed. A line missing from this dict, or None, sums the whole map.
        fit2t (bool): Report the two-temperature fit instead of 1T

    Returns:
        out (dict): Same shape as fit_region_multi's own return
    """
    items = [(cube, line) for cube, line in cubes_and_lines] + \
            [(None, line) for line in imported_lines]
    kept_pairs, _dropped = _dedupe_colocated(items, dedup_tol_um, line_of=lambda p: p[1])
    kept_lines = [l for _, l in kept_pairs]
    keys = _line_keys(kept_lines, molecule)
    molecular_data = _build_molecular_data(kept_lines, keys, molecule)

    fluxes, flux_errors, n_pixels_total = {}, {}, 0
    for key, (cube, line) in zip(keys, kept_pairs):
        if cube is not None:
            wavelength, _unit = extract_units(cube.wavelength)
            _n_wave, n_y, n_x = cube.cube.shape
            mask = region_masks.get(cube) if region_masks else None
            if mask is None:
                coords = [(x, y) for y in range(n_y) for x in range(n_x)]
            else:
                coords = [(int(x), int(y)) for y, x in zip(*np.where(mask))]
            specs = [np.asarray(s, dtype=float) for s in
                     (cube.get_pixel_spectrum(x, y) for x, y in coords)
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
        else:
            imp = line["imported_map"]
            flux_map = imp["flux_map"]
            err_map = (imp["flux_error_map"] if imp["error_pct"] is None
                      else np.abs(flux_map) * (imp["error_pct"] / 100.0))
            mask = imported_region_masks.get(id(line)) if imported_region_masks else None
            if mask is None:
                flux_sum = float(np.nansum(flux_map))
                err_sum = float(np.sqrt(np.nansum(err_map ** 2)))
                n_pix = int(np.sum(np.isfinite(flux_map)))
            else:
                flux_sum = float(np.nansum(flux_map[mask]))
                err_sum = float(np.sqrt(np.nansum(err_map[mask] ** 2)))
                n_pix = int(np.sum(mask & np.isfinite(flux_map)))
            if n_pix > 0 and flux_sum > 0:
                n_pixels_total += n_pix
                fluxes[key] = flux_sum
                flux_errors[key] = err_sum

    if len(fluxes) < min_lines:
        return {"status": STATUS_TOO_FEW_LINES, "n_lines": len(fluxes),
                "n_pixels": n_pixels_total, "n_cubes": len(kept_pairs),
                "reason": f"{len(fluxes)} detections across the cubes, need {min_lines}"}
    try:
        results = _run_fitter(fluxes, flux_errors, molecular_data, molecule, isotopologue,
                              opr, fit_opr, opr_grid, dist_pc, r_au, fit2t=fit2t)
    except Exception as exc:
        return {"status": STATUS_FIT_FAILED, "n_lines": len(fluxes),
                "n_pixels": n_pixels_total, "n_cubes": len(kept_pairs), "reason": str(exc)}
    return {"status": STATUS_OK, "n_lines": len(fluxes), "results": results,
            "n_pixels": n_pixels_total, "n_cubes": len(kept_pairs)}


def compute_native_line_maps(cubes_and_lines, contam_lines=None, n_jobs=8, dedup_tol_um=1e-5,
                             progress_callback=None, cancel_flag=None, region_masks=None,
                             n_sigma=3.0):
    """
    Fit each line at its own cube's native resolution -- stage 1 of the
    multi-cube map pipeline (see convolve_reproject_maps, combine_line_maps).
    Exposed separately so Calculate Maps can cache this expensive stage alone.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs, one per input cube
        contam_lines (pd.DataFrame): Contaminating lines, or None
        n_jobs (int): Worker processes for each native-grid extraction
        dedup_tol_um (float): Co-located-line merge tolerance in microns
        region_masks (dict): {CubeData: boolean mask (ny, nx)}, restricting
            which pixels get fit for that cube -- from an imported region.
            Keyed by cube identity so it survives co-located-line dedup. A
            cube missing from this dict, or region_masks=None, fits every
            pixel (the previous, unrestricted behavior).
        progress_callback (callable): progress_callback(done_pixels, total_pixels)
        cancel_flag (object): Object with a `.cancelled` attribute, or None
        n_sigma (float): Detection threshold in noise sigmas, 3.0 by default.
            The only place a threshold is applied: later stages use these maps as they are

    Returns:
        result (dict): 'kept_lines', 'dropped_lines' (line dicts),
            'native_flux_maps', 'native_flux_error_maps' (lists of 2D float32
            arrays, one per kept_lines entry, each cube's own native shape),
            'native_fit_flux_maps', 'native_status_maps', 'native_ul_ceiling_maps'
            (same layout; fitted flux including upper limits, NATIVE_PX_* codes,
            n_sigma flux ceiling -- what Select Region shows),
            'native_wcs', 'native_shapes', 'native_pix_arcsec', 'native_instrument'
            (one per kept_lines entry), 'cube_filenames'

    Raises:
        ValueError: if no pairs are given, a cube has no readable celestial
            WCS/pixel scale, or two inputs share no sky coverage
    """
    if len(cubes_and_lines) < 1:
        # 1 pair is now valid -- merge_native_results may add imported
        # lines on top of it; a no-import single cube still uses create_rotational_diagram_map.
        raise ValueError("Need at least 1 (cube, line) pair.")

    kept_pairs, dropped = _dedupe_colocated(list(cubes_and_lines), dedup_tol_um, line_of=lambda p: p[1])
    for weak_cube, weak_line in dropped:
        print(f"[dedupe] dropped {weak_line['wavelength']:.6f} um "
              f"(A={weak_line.get('a_coeff', 0):.3e})")

    cubes = [c for c, _ in kept_pairs]
    kept_lines = [l for _, l in kept_pairs]

    wcses = [make_2d_wcs(c.header) for c in cubes]
    shapes = [(c.cube.shape[1], c.cube.shape[2]) for c in cubes]
    scales = [get_pixel_size_arcsec(c.header) for c in cubes]
    if any(w is None for w in wcses) or any(s is None for s in scales):
        raise ValueError("Could not read a celestial WCS/pixel scale from every cube's header.")
    for i in range(len(cubes)):
        for j in range(i + 1, len(cubes)):
            if not wcs_footprints_overlap(wcses[i], shapes[i], wcses[j], shapes[j]):
                raise ValueError(
                    f"The cube for {kept_lines[i]['wavelength']:.4f} um and the cube "
                    f"for {kept_lines[j]['wavelength']:.4f} um share no sky coverage.")

    print(f">>> Calculate Maps: {len(cubes)} cubes, {len(kept_lines)} lines, native resolution")

    total_units = sum(c.cube.shape[1] * c.cube.shape[2] for c in cubes)
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

    native_flux_maps, native_err_maps = [], []
    native_fit_maps, native_status_maps, native_ceiling_maps = [], [], []
    # One pool for every line, not one per line: each fresh ProcessPoolExecutor
    # spawn re-imports the whole module chain (including pymc), which is slow
    # and prints its own startup warnings every time. Sharing it here means
    # that import cost is paid once for the whole Calculate Maps run.
    _pin_worker_math_threads()
    n_jobs = safe_worker_count(n_jobs)
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn")) as executor:
        for cube, line in zip(cubes, kept_lines):
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

            region_mask = region_masks.get(cube) if region_masks else None
            flux_native, err_native, fit_native, status_native, ceiling_native = _native_line_map(
                cube, line["wavelength"], contam_lines=contam_lines, n_jobs=n_jobs,
                progress_callback=_line_progress, cancel_flag=cancel_flag, executor=executor,
                region_mask=region_mask, n_sigma=n_sigma)
            native_flux_maps.append(flux_native)
            native_err_maps.append(err_native)
            native_fit_maps.append(fit_native)
            native_status_maps.append(status_native)
            native_ceiling_maps.append(ceiling_native)
            # Without this, a cancelled run kept marching through every
            # remaining line (each one still paying its own submit/wait
            # overhead) instead of actually stopping -- this is also why a
            # cancelled run could stay alive in the background long enough to
            # race a second run started right after it.
            if cancel_flag is not None and cancel_flag.cancelled:
                break

    return {
        "kept_lines": kept_lines, "dropped_lines": dropped,
        "native_flux_maps": native_flux_maps, "native_flux_error_maps": native_err_maps,
        "native_fit_flux_maps": native_fit_maps, "native_status_maps": native_status_maps,
        "native_ul_ceiling_maps": native_ceiling_maps,
        "native_wcs": wcses, "native_shapes": shapes, "native_pix_arcsec": scales,
        "cube_filenames": [getattr(c, "filename", None) for c in cubes],
        "native_instrument": [c.instrument for c in cubes],
        # The cube's own real header, not native_wcs's WCS-only one -- the
        # Map/Convolved Map FITS export needs this for INSTRUME/TARGNAME/etc.
        "native_headers": [c.header for c in cubes],
    }


def build_imported_map_payload(fits_path, error_pct):
    """
    Load an externally-produced 2D flux map as an Import Map payload, in
    place of fitting one from a cube. Same 2D-HDU scan as gui_2d.py's
    load_2d_fits, reimplemented GUI-free to avoid importing that module.

    Args:
        fits_path (str): Path to the flux-map FITS file.
        error_pct (float): Flat percent error; the error map this returns is
            abs(flux_map) * error_pct / 100.

    Returns:
        payload (dict): 'flux_map', 'flux_error_map' (both 2D float32),
            'error_pct' (float, or None if 'flux_error_map' came from a real
            ERR extension instead of a flat percent), 'wcs' (astropy.wcs.WCS,
            2D celestial), 'pix_arcsec' (float), 'shape' (ny, nx),
            'instrument' (str or None), 'filename' (str, basename of
            fits_path), 'header' (cleaned 2D header, NAXIS3/CTYPE3/etc
            stripped), 'global_fit' (dict with 'area'/'area_error'/
            'r_squared'/'x_data'/'y_data'/'baseline' from a Gaussian Map
            export's own point fit, or None if absent).

    Raises:
        ValueError: no 2D image data found, or no usable celestial WCS/pixel
            scale -- an unprojectable map can never be reprojected onto the
            common grid, so this is a hard reject at import time.
    """
    data, header, instrument, real_err = None, None, None, None
    spec_x, spec_y, spec_base = None, None, None
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "size", 0) > 0:
                squeezed = np.squeeze(hdu.data)
                if squeezed.ndim == 2:
                    data, header = squeezed, hdu.header.copy()
                    break
        if data is None:
            raise ValueError(f"No 2D image data found in {fits_path}.")
        instrument = detect_instrument(hdul)
        # A real per-pixel error (Save Maps/Gaussian Map's own ERR
        # extension) takes priority over the flat percent given below.
        for hdu in hdul:
            if (getattr(hdu, "name", "") or "").strip().upper() == "ERR" and hdu.data is not None:
                squeezed_err = np.squeeze(hdu.data)
                if squeezed_err.shape == data.shape:
                    real_err = squeezed_err
                break
        # A Gaussian Map export's global (stacked-spectrum) fit -- the
        # spectrum behind GFITAREA/GFITAERR/GFITR2, for a real Fit view.
        for hdu in hdul:
            name = (getattr(hdu, "name", "") or "").strip().upper()
            if name == "SPECX" and hdu.data is not None:
                spec_x = np.asarray(hdu.data, dtype=float)
            elif name == "SPECY" and hdu.data is not None:
                spec_y = np.asarray(hdu.data, dtype=float)
            elif name == "SPECBASE" and hdu.data is not None:
                spec_base = np.asarray(hdu.data, dtype=float)
        global_fit = None
        if "GFITAREA" in header:
            global_fit = {
                "area": float(header["GFITAREA"]),
                "area_error": float(header["GFITAERR"]) if "GFITAERR" in header else None,
                "r_squared": float(header["GFITR2"]) if "GFITR2" in header else None,
                "amplitude": float(header["GFITAMP"]) if "GFITAMP" in header else None,
                "center": float(header["GFITCTR"]) if "GFITCTR" in header else None,
                "sigma": float(header["GFITSIG"]) if "GFITSIG" in header else None,
                "x_data": spec_x, "y_data": spec_y, "baseline": spec_base,
            }

    header["NAXIS"] = 2
    if "WCSAXES" in header:
        header["WCSAXES"] = 2
    for kw in ("NAXIS3", "NAXIS4", "CTYPE3", "CRVAL3", "CRPIX3", "CDELT3", "CUNIT3",
               "PC1_3", "PC2_3", "PC3_1", "PC3_2", "PC3_3",
               "CD1_3", "CD2_3", "CD3_1", "CD3_2", "CD3_3"):
        if kw in header:
            del header[kw]

    wcs2d = make_2d_wcs(header)
    if wcs2d is None or not getattr(wcs2d, "has_celestial", False):
        raise ValueError(f"{fits_path} has no usable celestial WCS -- it can't be reprojected.")
    pix_arcsec = get_pixel_size_arcsec(header)
    if pix_arcsec is None:
        raise ValueError(f"Could not read a pixel scale from {fits_path}'s header.")

    flux_map = np.asarray(data, dtype=np.float32)
    if real_err is not None:
        err_map, stored_pct = np.asarray(real_err, dtype=np.float32), None
    else:
        err_map = (np.abs(flux_map) * (error_pct / 100.0)).astype(np.float32)
        stored_pct = float(error_pct)
    return {
        "flux_map": flux_map, "flux_error_map": err_map, "error_pct": stored_pct,
        "wcs": wcs2d, "pix_arcsec": pix_arcsec, "shape": flux_map.shape,
        "instrument": instrument, "filename": os.path.basename(fits_path), "header": header,
        "global_fit": global_fit,
    }


def _native_entry_from_imported_map(line):
    """
    Repackage a line's imported_map payload into a compute_native_line_maps-
    shaped per-line entry -- no I/O or fitting, just rebuilding the error
    map from the CURRENT error_pct unless a real ERR extension exists.

    Args:
        line (dict): Line dict with an 'imported_map' payload (see
            build_imported_map_payload).

    Returns:
        entry (dict): 'flux_map', 'flux_error_map', 'wcs', 'shape',
            'pix_arcsec', 'instrument', 'filename', 'header' -- one element
            of each of compute_native_line_maps's own list outputs. Also
            'fit_flux_map', 'status_map', 'ul_ceiling_map', all None: an
            imported map has no fit, so no upper-limit data.
    """
    imp = line["imported_map"]
    flux_map = imp["flux_map"]
    if imp["error_pct"] is None:
        # A real per-pixel error (Save Maps/Gaussian Map's ERR extension) --
        # already final, not a percent to recompute from.
        err_map = imp["flux_error_map"]
    else:
        err_map = (np.abs(flux_map) * (imp["error_pct"] / 100.0)).astype(np.float32)
    return {
        "flux_map": flux_map, "flux_error_map": err_map, "wcs": imp["wcs"],
        "shape": imp["shape"], "pix_arcsec": imp["pix_arcsec"],
        "instrument": imp["instrument"], "filename": imp["filename"], "header": imp["header"],
        "fit_flux_map": None, "status_map": None, "ul_ceiling_map": None,
    }


def merge_native_results(native_cube_result, imported_lines, dedup_tol_um=1e-5):
    """
    Merge compute_native_line_maps's cube-backed output with imported-map
    lines into one native_result-shaped dict, so Convolve Maps/combine_line_maps
    treat every checked line identically regardless of source.

    Args:
        native_cube_result (dict): Output of compute_native_line_maps, or
            None if no cube-backed line is checked (a pure-import run).
        imported_lines (list): Line dicts with an 'imported_map' payload.
        dedup_tol_um (float): Co-located-line merge tolerance in microns,
            applied across the whole combined set (cube-backed + imported).

    Returns:
        result (dict): Same shape as compute_native_line_maps's own return.
    """
    pairs = []
    dropped = list(native_cube_result["dropped_lines"]) if native_cube_result is not None else []
    if native_cube_result is not None:
        for i, line in enumerate(native_cube_result["kept_lines"]):
            entry = {
                "flux_map": native_cube_result["native_flux_maps"][i],
                "flux_error_map": native_cube_result["native_flux_error_maps"][i],
                "wcs": native_cube_result["native_wcs"][i],
                "shape": native_cube_result["native_shapes"][i],
                "pix_arcsec": native_cube_result["native_pix_arcsec"][i],
                "instrument": native_cube_result["native_instrument"][i],
                "filename": native_cube_result["cube_filenames"][i],
                "header": native_cube_result["native_headers"][i],
                "fit_flux_map": native_cube_result["native_fit_flux_maps"][i],
                "status_map": native_cube_result["native_status_maps"][i],
                "ul_ceiling_map": native_cube_result["native_ul_ceiling_maps"][i],
            }
            pairs.append((entry, line))
    for line in imported_lines:
        pairs.append((_native_entry_from_imported_map(line), line))

    kept_pairs, newly_dropped = _dedupe_colocated(pairs, dedup_tol_um, line_of=lambda p: p[1])
    kept_entries = [e for e, _ in kept_pairs]
    kept_lines = [l for _, l in kept_pairs]

    return {
        "kept_lines": kept_lines, "dropped_lines": dropped + newly_dropped,
        "native_flux_maps": [e["flux_map"] for e in kept_entries],
        "native_flux_error_maps": [e["flux_error_map"] for e in kept_entries],
        "native_wcs": [e["wcs"] for e in kept_entries],
        "native_shapes": [e["shape"] for e in kept_entries],
        "native_pix_arcsec": [e["pix_arcsec"] for e in kept_entries],
        "cube_filenames": [e["filename"] for e in kept_entries],
        "native_instrument": [e["instrument"] for e in kept_entries],
        "native_headers": [e["header"] for e in kept_entries],
        "native_fit_flux_maps": [e["fit_flux_map"] for e in kept_entries],
        "native_status_maps": [e["status_map"] for e in kept_entries],
        "native_ul_ceiling_maps": [e["ul_ceiling_map"] for e in kept_entries],
    }


def combine_line_maps(convolved_result, molecule="H2", isotopologue=1, opr=None, fit_opr=False,
                      opr_grid=None, dist_pc=140.0, r_au=None, min_lines=3, region_fit=None,
                      progress_callback=None, cancel_flag=None, fit2t=False):
    """
    Combine already-convolved per-line flux maps into a rotational diagram at
    every pixel of their common grid -- stage 3 (final) of the multi-cube
    pipeline. No detection threshold of its own (that's compute_native_line_maps's n_sigma).

    Args:
        convolved_result (dict): Output of convolve_reproject_maps
        molecule (str): Species name; 'H2' uses the OPR-capable fitter
        isotopologue (int): HITRAN isotopologue number, non-H2 only
        opr (float): Fixed ortho-para ratio, or None for the fitter default
        fit_opr (bool): Grid-search OPR per pixel, H2 only
        opr_grid (array): OPR grid for the search, or None
        dist_pc (float): Source distance in parsecs
        r_au (float): Emitting radius in AU, or None (total-molecules mode)
        min_lines (int): Fewest detections a pixel needs for a fit
        region_fit (dict): Output of fit_region_multi. This stage has no cube
            access to compute it itself -- the caller supplies it (see
            create_rotational_diagram_map_multi), or leaves it None to omit
            it from the result (e.g. a quick what-if re-combination)
        progress_callback (callable): progress_callback(done_pixels, total_pixels)
        cancel_flag (object): Object with a `.cancelled` attribute, or None
        fit2t (bool): Report the two-temperature fit instead of 1T, at every pixel

    Returns:
        result (dict): 'T', 'T_err', 'Ncol', 'Ncol_err' (2D float32, component
            1), 'T2', 'T2_err', 'Ncol2', 'Ncol2_err' (2D float32, 2T's T2
            component, all-NaN when fit2t is False), 'fit2t', 'n_lines',
            'status' (2D int), plus 'dst_wcs', 'molecule', 'dist_pc', 'r_au',
            'opr', 'min_lines', 't_cap', 'worst_pix_arcsec', 'kept_lines',
            'dropped_lines', 'region_fit', 'native_flux_maps', 'native_flux_error_maps',
            'native_wcs', 'native_shapes', 'native_pix_arcsec', 'cube_filenames'
    """
    kept_lines = convolved_result["kept_lines"]
    keys = _line_keys(kept_lines, molecule)
    molecular_data = _build_molecular_data(kept_lines, keys, molecule)
    flux_maps = convolved_result["line_flux_maps"]
    err_maps = convolved_result["line_flux_error_maps"]
    dst_shape = convolved_result["dst_shape"]
    n_y, n_x = dst_shape

    t_map = np.full(dst_shape, np.nan, dtype=np.float32)
    dt_map = np.full(dst_shape, np.nan, dtype=np.float32)
    n_map = np.full(dst_shape, np.nan, dtype=np.float32)
    dn_map = np.full(dst_shape, np.nan, dtype=np.float32)
    t2_map = np.full(dst_shape, np.nan, dtype=np.float32)
    dt2_map = np.full(dst_shape, np.nan, dtype=np.float32)
    n2_map = np.full(dst_shape, np.nan, dtype=np.float32)
    dn2_map = np.full(dst_shape, np.nan, dtype=np.float32)
    nlines_map = np.zeros(dst_shape, dtype=np.int16)
    status_map = np.full(dst_shape, STATUS_NO_DATA, dtype=np.int16)

    total = n_y * n_x
    done = 0
    for y in range(n_y):
        if cancel_flag is not None and cancel_flag.cancelled:
            break
        for x in range(n_x):
            fluxes, flux_errors = {}, {}
            for i in range(len(kept_lines)):
                f, e = flux_maps[i][y, x], err_maps[i][y, x]
                # Detections were already decided per line in Calculate Maps; the convolved
                # pair only has to be a usable positive flux and error for the fit.
                if np.isfinite(f) and np.isfinite(e) and e > 0 and f > 0:
                    fluxes[keys[i]] = float(f)
                    flux_errors[keys[i]] = float(e)

            if len(fluxes) < min_lines:
                status_map[y, x] = STATUS_NO_DATA if not fluxes else STATUS_TOO_FEW_LINES
                nlines_map[y, x] = len(fluxes)
            else:
                try:
                    result = _run_fitter(fluxes, flux_errors, molecular_data, molecule,
                                         isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au,
                                         fit2t=fit2t)
                    if fit2t:
                        two_t = result["2T"]
                        t_map[y, x] = float(two_t["T1"])
                        dt_map[y, x] = float(two_t["dT1"])
                        n_map[y, x] = float(two_t["Ncol1"])
                        dn_map[y, x] = float(two_t["dNcol1"])
                        t2_map[y, x] = float(two_t["T2"])
                        dt2_map[y, x] = float(two_t["dT2"])
                        n2_map[y, x] = float(two_t["Ncol2"])
                        dn2_map[y, x] = float(two_t["dNcol2"])
                    else:
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
            done += 1
            if progress_callback is not None:
                progress_callback(done, total)

    ok = int(np.sum(status_map == STATUS_OK))
    print(f">>> Done. Fitted {ok}/{n_y * n_x} pixels on the common grid.")

    return {
        "T": t_map, "T_err": dt_map, "Ncol": n_map, "Ncol_err": dn_map,
        "T2": t2_map, "T2_err": dt2_map, "Ncol2": n2_map, "Ncol2_err": dn2_map,
        "fit2t": fit2t, "n_lines": nlines_map, "status": status_map,
        "dst_wcs": convolved_result["dst_wcs"],
        "molecule": molecule, "dist_pc": dist_pc, "r_au": r_au, "opr": opr,
        "min_lines": min_lines, "t_cap": T_CAP_H2 if molecule == "H2" else T_CAP_GENERAL,
        "worst_pix_arcsec": convolved_result["worst_pix_arcsec"],
        "kept_lines": kept_lines, "dropped_lines": convolved_result["dropped_lines"],
        "region_fit": region_fit,
        # Each line's convolved/reprojected map, kept so fit_pixel_multi can
        # re-fit one pixel from a cheap lookup instead of redoing convolution.
        "_flux_maps": flux_maps, "_err_maps": err_maps, "_keys": keys,
        "_molecular_data": molecular_data,
        # Same arrays, same order as kept_lines, under the public name the
        # single-cube path also uses -- already convolved/reprojected here.
        "line_flux_maps": flux_maps, "line_flux_error_maps": err_maps,
        # Passthrough so a caller that only ever calls create_rotational_diagram_map_multi
        # (never touching the staged functions directly) can still populate a
        # pre-convolution view of each line's map, with its own cube's real
        # native WCS/shape/scale (not the common dst_wcs -- wrong resolution
        # for a native-grid map) and originating filename.
        "native_flux_maps": convolved_result["native_flux_maps"],
        "native_flux_error_maps": convolved_result["native_flux_error_maps"],
        "native_wcs": convolved_result["native_wcs"],
        "native_shapes": convolved_result["native_shapes"],
        "native_pix_arcsec": convolved_result["native_pix_arcsec"],
        "cube_filenames": convolved_result["cube_filenames"],
    }


def create_rotational_diagram_map_multi(cubes_and_lines, molecule="H2", isotopologue=1,
                                        opr=None, fit_opr=False, opr_grid=None,
                                        dist_pc=140.0, r_au=None, min_lines=3,
                                        dedup_tol_um=1e-5, n_jobs=8, contam_lines=None,
                                        region_fit=None, progress_callback=None,
                                        cancel_flag=None, region_masks=None, n_sigma=3.0,
                                        fit2t=False):
    """
    Fit a rotational diagram at every pixel of a common grid, one line per
    cube, convolved onto the longest-wavelength line's own native grid -- a
    thin wrapper chaining compute_native_line_maps -> convolve_reproject_maps -> combine_line_maps.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs, one per input cube
        molecule (str): Species name; 'H2' uses the OPR-capable fitter
        isotopologue (int): HITRAN isotopologue number, non-H2 only
        opr (float): Fixed ortho-para ratio, or None for the fitter default
        fit_opr (bool): Grid-search OPR per pixel, H2 only
        opr_grid (array): OPR grid for the search, or None
        dist_pc (float): Source distance in parsecs
        r_au (float): Emitting radius in AU, or None (total-molecules mode)
        min_lines (int): Fewest detections a pixel needs for a fit
        dedup_tol_um (float): Co-located-line merge tolerance in microns
        n_jobs (int): Worker processes for each native-grid extraction
        contam_lines (pd.DataFrame): Contaminating lines, or None
        region_fit (dict): Output of fit_region_multi, reused if the caller
            already ran it (the preview does), or None to skip it
        progress_callback (callable): progress_callback(done_units, total_units)
        cancel_flag (object): Object with a `.cancelled` attribute, or None
        region_masks (dict): {CubeData: boolean mask (ny, nx)}, restricting
            which pixels get natively fit for that cube -- from an imported
            region. Does not affect region_fit's own whole-cube sum. None
            fits every pixel (the previous, unrestricted behavior).
        n_sigma (float): Detection threshold in noise sigmas, 3.0 by default.
            Applies to the native fits only (the convolved maps get no second
            test); like the region mask, it does not affect region_fit, which keeps 3 sigma.
        fit2t (bool): Report the two-temperature fit instead of 1T, at every pixel

    Returns:
        result (dict): 'T', 'T_err', 'Ncol', 'Ncol_err' (2D float32, component
            1), 'T2', 'T2_err', 'Ncol2', 'Ncol2_err' (2D float32, 2T's T2
            component, all-NaN when fit2t is False), 'fit2t', 'n_lines',
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

    # Precompute the common grid size purely so progress_callback's total
    # matches what it always has (native extraction + final combination) --
    # compute_native_line_maps/convolve_reproject_maps redo this cheap WCS
    # work themselves, since each stage must stand on its own when called directly.
    kept_pairs_peek, _ = _dedupe_colocated(list(cubes_and_lines), dedup_tol_um, line_of=lambda p: p[1])
    cubes_peek = [c for c, _ in kept_pairs_peek]
    total_native = sum(c.cube.shape[1] * c.cube.shape[2] for c in cubes_peek)
    wcses_peek = [make_2d_wcs(c.header) for c in cubes_peek]
    shapes_peek = [(c.cube.shape[1], c.cube.shape[2]) for c in cubes_peek]
    scales_peek = [get_pixel_size_arcsec(c.header) for c in cubes_peek]
    if any(w is None for w in wcses_peek) or any(s is None for s in scales_peek):
        raise ValueError("Could not read a celestial WCS/pixel scale from every cube's header.")
    # Reference grid = longest-wavelength line's own native shape (the
    # convention convolve_reproject_maps uses for real).
    ref_idx_peek = int(np.argmax([l["wavelength"] for _, l in kept_pairs_peek]))
    dst_shape_peek = shapes_peek[ref_idx_peek]
    total_units = total_native + dst_shape_peek[0] * dst_shape_peek[1]

    def _native_progress(done, _total):
        """
        Forward compute_native_line_maps's own progress onto the combined total.

        Args:
            done (int): Native-extraction pixels done so far
            _total (int): compute_native_line_maps's own total, unused

        Returns:
            None
        """
        if progress_callback is not None:
            progress_callback(done, total_units)

    def _combine_progress(done, _total):
        """
        Forward combine_line_maps's own progress onto the combined total.

        Args:
            done (int): Common-grid pixels done so far
            _total (int): combine_line_maps's own total, unused

        Returns:
            None
        """
        if progress_callback is not None:
            progress_callback(total_native + done, total_units)

    native = compute_native_line_maps(
        cubes_and_lines, contam_lines=contam_lines, n_jobs=n_jobs, dedup_tol_um=dedup_tol_um,
        progress_callback=_native_progress, cancel_flag=cancel_flag, region_masks=region_masks,
        n_sigma=n_sigma)
    convolved = convolve_reproject_maps(native)

    # Whole-cube-summed anchor fit, the same thing Run Fit does. Reused if
    # the caller already ran it.
    if region_fit is None:
        kept_pairs, _ = _dedupe_colocated(list(cubes_and_lines), dedup_tol_um, line_of=lambda p: p[1])
        region_fit = fit_region_multi(
            kept_pairs, molecule=molecule, isotopologue=isotopologue, opr=opr,
            fit_opr=fit_opr, opr_grid=opr_grid, dist_pc=dist_pc, r_au=r_au,
            min_lines=min_lines, dedup_tol_um=dedup_tol_um, contam_lines=contam_lines,
            fit2t=fit2t,
        )

    return combine_line_maps(
        convolved, molecule=molecule, isotopologue=isotopologue, opr=opr, fit_opr=fit_opr,
        opr_grid=opr_grid, dist_pc=dist_pc, r_au=r_au, min_lines=min_lines, region_fit=region_fit,
        progress_callback=_combine_progress, cancel_flag=cancel_flag, fit2t=fit2t)


def create_rotational_diagram_map_mixed(cubes_and_lines, imported_lines, molecule="H2",
                                        isotopologue=1, opr=None, fit_opr=False, opr_grid=None,
                                        dist_pc=140.0, r_au=None, min_lines=3,
                                        dedup_tol_um=1e-5, n_jobs=8, contam_lines=None,
                                        region_fit=None, progress_callback=None,
                                        cancel_flag=None, region_masks=None,
                                        imported_region_masks=None, n_sigma=3.0, fit2t=False):
    """
    create_rotational_diagram_map_multi's mixed-source sibling (some lines
    are cube-backed, others already have their map via Import Map): chains
    compute_native_line_maps -> merge_native_results -> convolve_reproject_maps -> combine_line_maps.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs, one per
            cube-backed line. May be empty (a pure-import run).
        imported_lines (list): Line dicts with an 'imported_map' payload.
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au,
            min_lines, dedup_tol_um, n_jobs, contam_lines, region_fit,
            progress_callback, cancel_flag, region_masks, n_sigma, fit2t:
            Same meaning as create_rotational_diagram_map_multi
        imported_region_masks (dict): {id(line): boolean mask}, restricting
            which pixels of an imported line's own map count toward
            region_fit. None sums the whole map.

    Returns:
        result (dict): Same shape as create_rotational_diagram_map_multi's
            own return.

    Raises:
        ValueError: if neither a cube-backed nor an imported line is given,
            a cube/imported map has no readable celestial WCS/pixel scale,
            or two inputs share no sky coverage
    """
    if not cubes_and_lines and not imported_lines:
        raise ValueError("Need at least one cube-backed or imported line.")

    total_native = sum(c.cube.shape[1] * c.cube.shape[2] for c, _ in cubes_and_lines)

    # Cheap peek at the final common-grid size (both sources), so
    # progress_callback's total is accurate from the start.
    wcses_peek, shapes_peek, scales_peek, wavelengths_peek = [], [], [], []
    for c, line in cubes_and_lines:
        wcses_peek.append(make_2d_wcs(c.header))
        shapes_peek.append((c.cube.shape[1], c.cube.shape[2]))
        scales_peek.append(get_pixel_size_arcsec(c.header))
        wavelengths_peek.append(line["wavelength"])
    for line in imported_lines:
        imp = line["imported_map"]
        wcses_peek.append(imp["wcs"])
        shapes_peek.append(imp["shape"])
        scales_peek.append(imp["pix_arcsec"])
        wavelengths_peek.append(line["wavelength"])
    if any(w is None for w in wcses_peek) or any(s is None for s in scales_peek):
        raise ValueError("Could not read a celestial WCS/pixel scale from every line.")
    # Reference grid = longest-wavelength line's own native shape (the
    # convention convolve_reproject_maps uses for real).
    ref_idx_peek = int(np.argmax(wavelengths_peek))
    dst_shape_peek = shapes_peek[ref_idx_peek]
    total_units = total_native + dst_shape_peek[0] * dst_shape_peek[1]

    def _native_progress(done, _total):
        """
        Forward compute_native_line_maps's own progress onto the combined total.

        Args:
            done (int): Native-extraction pixels done so far
            _total (int): compute_native_line_maps's own total, unused

        Returns:
            None
        """
        if progress_callback is not None:
            progress_callback(done, total_units)

    def _combine_progress(done, _total):
        """
        Forward combine_line_maps's own progress onto the combined total.

        Args:
            done (int): Common-grid pixels done so far
            _total (int): combine_line_maps's own total, unused

        Returns:
            None
        """
        if progress_callback is not None:
            progress_callback(total_native + done, total_units)

    native = None
    if cubes_and_lines:
        native = compute_native_line_maps(
            cubes_and_lines, contam_lines=contam_lines, n_jobs=n_jobs, dedup_tol_um=dedup_tol_um,
            progress_callback=_native_progress, cancel_flag=cancel_flag, region_masks=region_masks,
            n_sigma=n_sigma)
    merged = merge_native_results(native, imported_lines, dedup_tol_um=dedup_tol_um)
    convolved = convolve_reproject_maps(merged)

    if region_fit is None:
        region_fit = fit_region_mixed(
            cubes_and_lines, imported_lines, molecule=molecule, isotopologue=isotopologue,
            opr=opr, fit_opr=fit_opr, opr_grid=opr_grid, dist_pc=dist_pc, r_au=r_au,
            min_lines=min_lines, dedup_tol_um=dedup_tol_um, contam_lines=contam_lines,
            region_masks=region_masks, imported_region_masks=imported_region_masks, fit2t=fit2t,
        )

    return combine_line_maps(
        convolved, molecule=molecule, isotopologue=isotopologue, opr=opr, fit_opr=fit_opr,
        opr_grid=opr_grid, dist_pc=dist_pc, r_au=r_au, min_lines=min_lines, region_fit=region_fit,
        progress_callback=_combine_progress, cancel_flag=cancel_flag, fit2t=fit2t)


def fit_pixel_multi(result, x, y, molecule="H2", isotopologue=1, opr=None,
                    fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None,
                    min_lines=3, fit2t=False):
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
        fit2t (bool): Report the two-temperature fit instead of 1T

    Returns:
        out (dict): Same shape as fit_pixel's return -- 'status', 'n_lines',
            and on STATUS_OK also 'results' (with '1T' or '2T' plus '_plot_data')

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
        # Same rule as combine_line_maps: no threshold here, only a usable flux/error pair.
        if np.isfinite(f) and np.isfinite(e) and e > 0 and f > 0:
            fluxes[key] = float(f)
            flux_errors[key] = float(e)

    if len(fluxes) < min_lines:
        status = STATUS_NO_DATA if not fluxes else STATUS_TOO_FEW_LINES
        return {"status": status, "n_lines": len(fluxes),
                "reason": f"{len(fluxes)} detections, need {min_lines}"}

    try:
        results = _run_fitter(fluxes, flux_errors, molecular_data, molecule,
                              isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au,
                              fit2t=fit2t)
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
    # Multi-cube has a real common WCS (dst_wcs); single-cube only has its
    # own header. Either way it's reduced to a clean 2D WCS -- the raw
    # header's own EXTNAME can't leak through and clobber the one set below.
    source_header = result["dst_wcs"].to_header() if result.get("dst_wcs") is not None else result.get("header")
    shape = np.asarray(result["T"]).shape

    # Multi-cube: dst_wcs carries no INSTRUME/TARGNAME -- fold them in from
    # the coarsest-scale cube's real header, same choice Convolve Maps uses.
    identity_header = None
    native_headers, native_scales = result.get("native_headers"), result.get("native_pix_arcsec")
    if result.get("dst_wcs") is not None and native_headers and native_scales:
        identity_header = native_headers[int(np.argmax(native_scales))]

    primary_header, _ = build_line_map_header(
        source_header, shape, molecule=result["molecule"], identity_header=identity_header,
        extra={'DIST_PC': result["dist_pc"]},
    )
    if result.get("r_au") is not None:
        primary_header["R_AU"] = result["r_au"]
    if result.get("worst_pix_arcsec") is not None:
        primary_header["PIXSCALE"] = (result["worst_pix_arcsec"], "arcsec/pixel, worst input")

    # Only the two temperature/column keys get BUNIT, not their _err counterparts -- matches before
    bunit_by_key = {"T": "K", "Ncol": "cm-2", "T2": "K", "Ncol2": "cm-2"}
    keys = ("T", "T_err", "Ncol", "Ncol_err")
    if result.get("fit2t"):
        keys += ("T2", "T2_err", "Ncol2", "Ncol2_err")
    keys += ("n_lines", "status")

    extensions = []
    for key in keys:
        extra = {'BUNIT': bunit_by_key[key]} if key in bunit_by_key else None
        ext_header = build_2d_export_header(source_header, shape, extra=extra)
        extensions.append((key.upper(), np.asarray(result[key]), ext_header))

    write_fits_product(output_path, None, primary_header, extensions=extensions)
    print(f"[OK] wrote {output_path}")
