"""
Per-pixel continuum fitting for a whole cube.

Fits a continuum at every pixel, either with the automatic method or from
anchor wavelengths, and writes two cubes: the continuum, and the
continuum-subtracted (line-only) cube.
"""

import multiprocessing
import os
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
from astropy.io import fits
from scipy.interpolate import CubicSpline

from .flux_map_extractor import DEFAULT_SIGMA_MULTIPLIER, fit_auto_continuum
from .unit_converter import convert_wavelength
from .utils import get_spectral_resolution, safe_worker_count

# Same minimum as the interactive custom continuum tool.
MIN_ANCHORS = 3
CLIP_SIGMA = 3.0
# A window with fewer valid samples than this gets a plain median, with no clipping.
MIN_CLIP_SAMPLES = 5


def _wavelength_values(wavelength):
    """
    Return a wavelength array as plain floats, without astropy units.

    Args:
        wavelength (array or Quantity): Wavelength values

    Returns:
        values (array): Float array of the same length
    """
    return np.asarray(wavelength.value if hasattr(wavelength, 'value') else wavelength, dtype=float)


def _continuum_row_worker(row_idx, spectra_row, wavelength, continuum_params):
    """
    Fit the automatic continuum to every pixel in one cube row.
    Module-level so ProcessPoolExecutor can pickle it.
    Each pixel uses the same fit_auto_continuum call as the Flux Map.

    Args:
        row_idx (int): Row index in the cube's spatial y-axis
        spectra_row (array): Spectra of this row, shape (n_wave, n_x)
        wavelength (array): 1D wavelength array, length n_wave
        continuum_params (dict): fit_auto_continuum() keyword arguments, or None for the defaults

    Returns:
        row (tuple): (row_idx, row_continuum, n_failed). row_continuum has shape (n_wave, n_x).
            n_failed is the number of pixels that have data but got no continuum.
    """
    params = continuum_params or {}
    n_wave, n_x = spectra_row.shape
    row_continuum = np.full((n_wave, n_x), np.nan, dtype=np.float32)
    n_failed = 0
    for ix in range(n_x):
        spectrum = spectra_row[:, ix]
        if not np.any(np.isfinite(spectrum)):
            continue
        continuum = fit_auto_continuum(wavelength, spectrum, **params)
        if np.all(np.isnan(continuum)):
            n_failed += 1
            continue
        row_continuum[:, ix] = continuum
    return row_idx, row_continuum, n_failed


def _anchor_values(block):
    """
    Get one value per pixel from a window of slices around an anchor.
    Samples more than CLIP_SIGMA from the window median are dropped, then the median of the rest is used.
    A window with fewer than MIN_CLIP_SAMPLES valid samples uses its plain median.

    Args:
        block (array): Flux in the anchor window, shape (n_window, n_x)

    Returns:
        values (array): Anchor value of each pixel, shape (n_x,). NaN where the window has no valid sample
    """
    block = np.asarray(block, dtype=float)
    with warnings.catch_warnings():
        # An all-NaN window gives NaN on purpose; the caller counts those pixels.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        median = np.nanmedian(block, axis=0)
        sigma = 1.4826 * np.nanmedian(np.abs(block - median), axis=0)
        keep = np.isfinite(block) & ((np.abs(block - median) <= CLIP_SIGMA * sigma) | (sigma == 0))
        clipped = np.nanmedian(np.where(keep, block, np.nan), axis=0)
    n_valid = np.isfinite(block).sum(axis=0)
    return np.where(n_valid >= MIN_CLIP_SAMPLES, clipped, median)


def _custom_row_worker(row_idx, spectra_row, wavelength, anchors):
    """
    Fit the anchor-based continuum to every pixel in one cube row.
    Module-level so ProcessPoolExecutor can pickle it.
    Each pixel gets a cubic spline through its own anchor values, held flat beyond the outer anchors.

    Args:
        row_idx (int): Row index in the cube's spatial y-axis
        spectra_row (array): Spectra of this row, shape (n_wave, n_x)
        wavelength (array): 1D wavelength array, length n_wave
        anchors (dict): Anchor set from build_custom_anchors()

    Returns:
        row (tuple): (row_idx, row_continuum, n_failed). row_continuum has shape (n_wave, n_x).
            n_failed is the number of pixels that have data but fewer than MIN_ANCHORS valid anchors.
    """
    n_wave, n_x = spectra_row.shape
    centers = anchors['wavelengths']
    values = np.stack([_anchor_values(spectra_row[lo:hi]) for lo, hi in anchors['windows']])
    has_data = np.any(np.isfinite(spectra_row), axis=0)
    row_continuum = np.full((n_wave, n_x), np.nan, dtype=np.float32)
    n_failed = 0
    for ix in range(n_x):
        if not has_data[ix]:
            continue
        valid = np.isfinite(values[:, ix])
        if valid.sum() < MIN_ANCHORS:
            n_failed += 1
            continue
        x_valid = centers[valid]
        spline = CubicSpline(x_valid, values[valid, ix], bc_type='natural')
        # Clipping the evaluation points holds the first and last anchor value flat beyond the anchors.
        row_continuum[:, ix] = spline(np.clip(wavelength, x_valid[0], x_valid[-1]))
    return row_idx, row_continuum, n_failed


def build_custom_anchors(cube_data, anchor_wavelengths, half_width_sigma=DEFAULT_SIGMA_MULTIPLIER):
    """
    Turn clicked anchor wavelengths into the slice windows used at every pixel.
    Each anchor snaps to the nearest slice. Its window spans +/- half_width_sigma instrumental
    sigma, sigma = wavelength / R / 2.355, with R from the instrument. The default is the Flux Map's.

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        anchor_wavelengths (array): Anchor wavelengths, in the units of cube_data.wavelength
        half_width_sigma (float): Half-width of each window, in instrumental sigma

    Returns:
        anchors (dict): 'wavelengths' (snapped anchor wavelengths, ascending), 'windows' (a (lo, hi)
            slice range per anchor) and 'description' (HISTORY lines that describe the set)

    Raises:
        ValueError: if the wavelengths are not ascending, an anchor is outside the spectrum, fewer than
            MIN_ANCHORS different slices remain, or the instrument has no resolution formula
    """
    wavelength = _wavelength_values(cube_data.wavelength)
    anchor_wavelengths = np.asarray(anchor_wavelengths, dtype=float)
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("The cube wavelengths must increase for the anchor windows.")
    half_step = 0.5 * np.median(np.diff(wavelength))
    outside = anchor_wavelengths[(anchor_wavelengths < wavelength[0] - half_step) |
                                 (anchor_wavelengths > wavelength[-1] + half_step)]
    if outside.size:
        raise ValueError(f"Anchors outside the spectrum ({wavelength[0]:.5g} to {wavelength[-1]:.5g}): "
                         + ", ".join(f"{w:.5g}" for w in outside))

    slices = np.unique(np.abs(wavelength[None, :] - anchor_wavelengths[:, None]).argmin(axis=1))
    if len(slices) < MIN_ANCHORS:
        raise ValueError(f"Need at least {MIN_ANCHORS} different anchors, got {len(slices)}.")

    to_um = float(convert_wavelength(1.0, cube_data.wave_unit, 'um'))
    windows = []
    for c in slices:
        resolution = get_spectral_resolution(wavelength[c] * to_um, cube_data.instrument, cube_data.header)
        half_width = half_width_sigma * (wavelength[c] / resolution) / 2.355
        windows.append((int(np.searchsorted(wavelength, wavelength[c] - half_width, side='left')),
                        int(np.searchsorted(wavelength, wavelength[c] + half_width, side='right'))))

    centers = wavelength[slices]
    description = [
        f"Custom continuum: {len(slices)} anchors, window +/-{half_width_sigma:g} sigma (lambda/R/2.355), "
        f"{CLIP_SIGMA:g}-sigma clipped median, cubic spline, flat beyond the outer anchors",
        "Anchor wavelengths: " + ", ".join(f"{w:.6g}" for w in centers),
    ]
    return {'wavelengths': centers, 'windows': windows, 'description': description}


def fit_continuum_cube(cube, wavelength, continuum_params=None, anchors=None, n_jobs=None,
                       progress_callback=None, cancel_event=None):
    """
    Fit a continuum at every pixel of a cube, automatic or from anchors.
    Rows run in separate processes. A pixel without enough valid data stays NaN.
    Scripts that call this need an if __name__ == "__main__": guard on Windows/macOS.

    Args:
        cube (array): 3D data cube with shape (n_wave, n_y, n_x)
        wavelength (array): 1D wavelength array, length n_wave
        continuum_params (dict): fit_auto_continuum() keyword arguments, or None for the defaults.
            Not used when anchors is given
        anchors (dict): Anchor set from build_custom_anchors(), or None for the automatic fit
        n_jobs (int): Number of worker processes. Defaults to auto-detect CPUs, capped by memory
        progress_callback (callable): Optional progress_callback(current, total), in pixels
        cancel_event (threading.Event): Optional event. When set, the run stops and returns None

    Returns:
        result (tuple or None): (continuum_cube, n_failed), or None if cancelled.
            continuum_cube is float32 with the same shape as cube.
            n_failed is the number of pixels that have data but got no continuum.

    Raises:
        ValueError: if cube is not 3D, or wavelength does not match its first axis
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got {cube.ndim}D")
    wavelength = _wavelength_values(wavelength)
    n_wave, n_y, n_x = cube.shape
    if len(wavelength) != n_wave:
        raise ValueError(f"Wavelength length {len(wavelength)} does not match cube spectral axis {n_wave}")

    if anchors is not None:
        row_worker, worker_arg = _custom_row_worker, anchors
    else:
        row_worker, worker_arg = _continuum_row_worker, continuum_params

    continuum_cube = np.full(cube.shape, np.nan, dtype=np.float32)
    total_pixels = n_y * n_x
    completed = 0
    n_failed = 0
    cancelled = False

    if n_jobs is None:
        n_jobs = max(1, min(os.cpu_count() or 1, 8))
    n_jobs = safe_worker_count(n_jobs)

    print(f"Fitting the continuum at {total_pixels:,} pixels with {n_jobs} worker processes")

    # A few rows in flight at a time, so a large cube is never copied out all at once.
    max_in_flight = 2 * n_jobs
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
        pending = set()
        next_row = 0
        while (next_row < n_y or pending) and not cancelled:
            while next_row < n_y and len(pending) < max_in_flight:
                spectra_row = np.asarray(cube[:, next_row, :])
                pending.add(ex.submit(row_worker, next_row, spectra_row, wavelength, worker_arg))
                next_row += 1
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                row_idx, row_continuum, row_failed = fut.result()
                continuum_cube[:, row_idx, :] = row_continuum
                n_failed += row_failed
                completed += n_x
                if progress_callback:
                    progress_callback(completed, total_pixels)
                if completed % (n_x * 5) == 0:
                    print(f"  Progress: {completed}/{total_pixels} pixels ({100*completed/total_pixels:.1f}%)")
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                # Rows that have not started yet are dropped. The pool still waits for rows in flight.
                for f in pending:
                    f.cancel()

    if cancelled:
        print(f"Continuum fit cancelled: {completed}/{total_pixels} pixels completed before stopping")
        return None

    print(f"Continuum fit complete: {total_pixels:,} pixels, {n_failed} with data but no continuum")
    return continuum_cube, n_failed


def _write_cube_fits(data, header, instrument, output_path, history):
    """
    Write a cube to FITS with the source header, in the layout the loader reads back.
    MUSE data goes in the first extension. MIRI and NIRSpec data goes in the primary HDU.
    Each string in history becomes a HISTORY card.

    Args:
        data (array): 3D cube to write
        header (astropy.io.fits.Header): Header of the source cube
        instrument (str): Instrument name from CubeData.instrument
        output_path (str): Destination FITS path
        history (list): Strings that describe what the cube is and how it was made

    Returns:
        None: Writes output_path as a side effect
    """
    out_header = header.copy()
    for line in history:
        out_header['HISTORY'] = line

    if instrument == "MUSE":
        # TELESCOPE_LOADERS reads MUSE data from hdul[1] positionally.
        hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=data, header=out_header)])
    else:
        hdul = fits.HDUList([fits.PrimaryHDU(data=data, header=out_header)])
    hdul.writeto(output_path, overwrite=True)


def export_continuum_cubes(cube_data, continuum_path, contsub_path, continuum_params=None,
                           anchor_wavelengths=None, n_jobs=None, progress_callback=None,
                           cancel_event=None):
    """
    Fit the continuum at every pixel, then write the continuum cube and the continuum-subtracted cube.
    The subtracted cube is the original cube minus the continuum cube, so the two always sum to the original.
    A pixel that has data but got no continuum is NaN in both outputs. n_failed counts those pixels.

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        continuum_path (str): Destination FITS path of the continuum cube
        contsub_path (str): Destination FITS path of the continuum-subtracted cube
        continuum_params (dict): fit_auto_continuum() keyword arguments, or None for the defaults.
            Only for the automatic fit
        anchor_wavelengths (array): Anchor wavelengths for the custom fit, or None for the automatic fit
        n_jobs (int): Number of worker processes. Defaults to auto-detect CPUs, capped by memory
        progress_callback (callable): Optional progress_callback(current, total), in pixels
        cancel_event (threading.Event): Optional event. When set, the run stops and nothing is written

    Returns:
        n_failed (int or None): Number of pixels that have data but got no continuum.
            None if cancelled.

    Raises:
        ValueError: if no cube is loaded, or both continuum_params and anchor_wavelengths are given.
            build_custom_anchors() also raises for bad anchors
    """
    if cube_data.cube is None:
        raise ValueError("No cube loaded")
    if continuum_params and anchor_wavelengths is not None:
        raise ValueError("Give continuum_params or anchor_wavelengths, not both.")

    anchors = None
    if anchor_wavelengths is not None:
        anchors = build_custom_anchors(cube_data, anchor_wavelengths)
        method_history = anchors['description']
    elif continuum_params:
        method_history = ["Automatic continuum parameters: "
                          + ", ".join(f"{k}={v}" for k, v in continuum_params.items())]
    else:
        method_history = ["Automatic continuum parameters: defaults"]

    result = fit_continuum_cube(cube_data.cube, cube_data.wavelength, continuum_params, anchors,
                                n_jobs, progress_callback, cancel_event)
    if result is None:
        return None
    continuum_cube, n_failed = result

    _write_cube_fits(continuum_cube, cube_data.header, cube_data.instrument, continuum_path,
                     ["IFU Analyzer: continuum cube (continuum fitted at every pixel)"] + method_history)
    print(f"Continuum cube written to {continuum_path}")

    # In place, so the second cube needs no extra memory. The continuum cube is already on disk.
    contsub_cube = continuum_cube
    np.subtract(cube_data.cube, contsub_cube, out=contsub_cube, dtype=np.float32)
    _write_cube_fits(contsub_cube, cube_data.header, cube_data.instrument, contsub_path,
                     ["IFU Analyzer: continuum-subtracted cube (cube minus the continuum cube)"] + method_history)
    print(f"Continuum-subtracted cube written to {contsub_path}")
    return n_failed
