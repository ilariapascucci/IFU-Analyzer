"""
Write and read a Gaussian fit file.
The file holds the spectrum, the fit curve, the baseline, and every fitted line with all its
numbers. A fit can be built again from it without a new fit. This module has no GUI code.
"""

import re
from dataclasses import dataclass, field, replace

import numpy as np

from .gaussian_fitter import GaussianFitResult, build_gaussian_fit_curve, gaussian
from .spectrum_file import _read_lines

FIRST_LINE = "IFU Analyzer Gaussian fit file, format 1"

# Numbers of a line row. A number that is not available is written as nan.
_FLOAT_FIELDS = (
    "catalog_wavelength", "a_coeff", "e_upper", "center", "amplitude", "sigma", "fwhm", "area", "area_native",
    "area_error", "amplitude_error", "sigma_error", "catalog_sigma", "noise_std",
    "baseline_slope", "baseline_intercept", "r_squared",
)
# These can be None in a fit result. They are None again when the file is read.
_OPTIONAL_FIELDS = ("area_error", "amplitude_error", "sigma_error", "catalog_sigma", "noise_std")
# These belong to the catalog line, not to the fit. A line dict has them only if they are known.
_CATALOG_FIELDS = ("catalog_wavelength", "a_coeff", "e_upper")

LINE_COLUMNS = (("kind", "species", "transition", "source") + _FLOAT_FIELDS
                + ("area_unit", "first_index", "last_index", "success", "chunk_model", "message"))

MARKER_NAMES = ("fit_left", "peak", "fit_right", "noise_left", "noise_right")


@dataclass
class FitFile:
    """The content of a fit file.

    Attributes:
        fit_type (str): 'single' or 'multi'
        tab (str): Spectrum tab that was fitted, or None
        wave_unit (str): Wavelength unit as written in the file, or None
        flux_unit (str): Flux unit as written in the file, or None
        instrument (str): Instrument name as written in the file, or None
        baseline_source (str): Where the baseline came from, as written in the file, or None
        markers (dict): The fit markers that were set, with keys from MARKER_NAMES. Empty for a multi fit.
        lines (list): One dict for each line, with the keys of LINE_COLUMNS (numbers as float or None)
        wavelength (numpy.ndarray): Wavelength column
        original (numpy.ndarray): Original spectrum column
        fit (numpy.ndarray): Fit column. NaN where no fit was made.
        baseline (numpy.ndarray): Baseline column. NaN where no fit was made.
        comments (list): The comment lines of the file
        continuum_lines (list): The lines of the file that say which continuum the fit used (without "#"),
            or an empty list. A loaded fit writes them again when it is saved.
    """
    fit_type: str
    tab: str
    wave_unit: str
    flux_unit: str
    instrument: str
    baseline_source: str
    markers: dict
    lines: list
    wavelength: np.ndarray
    original: np.ndarray
    fit: np.ndarray
    baseline: np.ndarray
    comments: list = field(default_factory=list)
    continuum_lines: list = field(default_factory=list)


def _num(value):
    """Write a number so that it reads back as the same number.

    Args:
        value (float): The number, or None

    Returns:
        text (str): The number, or "nan" if it is None or not finite
    """
    if value is None or not np.isfinite(value):
        return "nan"
    return repr(float(value))


def _text(value):
    """Write text on one line, with no tab.

    Args:
        value (str): The text, or None

    Returns:
        text (str): The text with tabs and line breaks changed to spaces, or "-" if it is empty
    """
    cleaned = " ".join(str(value).split()) if value is not None else ""
    return cleaned or "-"


def _chunk_model_text(model):
    """Write the chunk model of a line (the local continuum and every Gaussian that was fitted with it).

    Args:
        model (dict): 'a', 'b', 'x0' and 'gaussians' (list of amplitude, center, sigma), or None

    Returns:
        text (str): The model on one cell, or "-" if there is none
    """
    if not model:
        return "-"
    gaussians = "|".join(",".join(repr(float(v)) for v in g) for g in model["gaussians"])
    return f"{repr(float(model['a']))};{repr(float(model['b']))};{repr(float(model['x0']))};{gaussians}"


def _parse_chunk_model(text):
    """Read the text that _chunk_model_text wrote.

    Args:
        text (str): The cell

    Returns:
        model (dict): 'a', 'b', 'x0' and 'gaussians', or None if the cell is "-"
    """
    if text == "-":
        return None
    a, b, x0, gaussians = text.split(";")
    return {"a": float(a), "b": float(b), "x0": float(x0),
            "gaussians": [tuple(float(v) for v in g.split(",")) for g in gaussians.split("|") if g]}


def _line_row(entry, wavelength):
    """Turn one fitted line into the cells of a line row.

    Args:
        entry (dict): 'kind' ('line', 'contaminant' or 'skipped'), 'line' (catalog dict, may be empty)
            and 'result' (GaussianFitResult)
        wavelength (numpy.ndarray): Wavelength grid of the spectrum

    Returns:
        cells (list): The text of each cell, in the order of LINE_COLUMNS

    Raises:
        ValueError: If the window of a fitted line is not on the wavelength grid
    """
    result, line = entry["result"], entry.get("line") or {}
    first = last = -1
    if result.success:
        found = np.flatnonzero(np.isin(wavelength, result.x_data))
        if len(found) != len(result.x_data):
            raise ValueError("A fitted line does not match the wavelength grid. "
                             "Use the same wavelength array that was used for the fit.")
        first, last = int(found[0]), int(found[-1])
    values = {
        "catalog_wavelength": line.get("lambda_micron", line.get("wavelength")),
        "a_coeff": line.get("a_coeff"), "e_upper": line.get("e_upper"),
        "center": result.center, "amplitude": result.amplitude, "sigma": result.sigma, "fwhm": result.fwhm,
        "area": result.area, "area_native": result.area_native, "area_error": result.area_error,
        "amplitude_error": result.amplitude_error, "sigma_error": result.sigma_error,
        "catalog_sigma": result.catalog_sigma, "noise_std": result.noise_std,
        "baseline_slope": result.baseline_slope, "baseline_intercept": result.baseline_intercept,
        "r_squared": result.r_squared,
    }
    return ([_text(entry["kind"]), _text(line.get("species")), _text(line.get("transition")), _text(line.get("source"))]
            + [_num(values[name]) for name in _FLOAT_FIELDS]
            + [_text(result.area_unit), str(first), str(last), "1" if result.success else "0",
               _chunk_model_text(result.chunk_model), _text(result.message)])


def write_fit_file(path, wavelength, spectrum, entries, fit_type, wave_unit=None, flux_unit=None,
                   header_lines=(), markers=None):
    """Save a Gaussian fit to a text file that read_fit_file can read again.
    The columns are separated by tabs. The comment lines at the top start with "#" and hold the
    fit type, the units, the markers, and one row for each fitted line.

    Args:
        path (str): Output file path
        wavelength (numpy.ndarray): Wavelength grid that the lines were fitted on
        spectrum (numpy.ndarray): Original spectrum on the same grid
        entries (list): Dicts with 'kind' ('line', 'contaminant' or 'skipped'), 'line' (the catalog dict,
            may be empty) and 'result' (GaussianFitResult)
        fit_type (str): 'single' or 'multi'
        wave_unit (str): Wavelength unit for the header, or None for "Unknown"
        flux_unit (str): Flux unit for the header, or None for "Unknown"
        header_lines (list): Extra comment lines (str) to write below the fit type
        markers (dict): The fit markers, with keys from MARKER_NAMES. A marker that is None is left out.

    Returns:
        None

    Raises:
        ValueError: If no line was fitted, fit_type is not valid, the arrays differ in length,
            or the window of a fitted line is not on the wavelength grid
    """
    if fit_type not in ("single", "multi"):
        raise ValueError(f"fit_type must be 'single' or 'multi', got {fit_type!r}.")
    wavelength = np.asarray(wavelength, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    if len(wavelength) != len(spectrum):
        raise ValueError("wavelength and spectrum must have the same length.")
    successful = [entry["result"] for entry in entries if entry["result"].success]
    if not successful:
        raise ValueError("No successful fit to save.")

    fit, baseline = build_gaussian_fit_curve(wavelength, successful, return_baseline=True)
    rows = [_line_row(entry, wavelength) for entry in entries]
    marker_text = ", ".join(f"{name}={repr(float(markers[name]))}" for name in MARKER_NAMES
                            if markers and markers.get(name) is not None)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {FIRST_LINE}\n")
        f.write(f"# Fit type: {fit_type}\n")
        for line in header_lines:
            f.write(f"# {line}\n")
        f.write(f"# Wavelength unit: {wave_unit or 'Unknown'}\n")
        f.write(f"# Flux unit: {flux_unit or 'Unknown'}\n")
        if marker_text:
            f.write(f"# Markers: {marker_text}\n")
        f.write("# Fit and Baseline are NaN where no fit was made. Original-Fit is NaN there too.\n")
        f.write("# Line columns: " + "\t".join(LINE_COLUMNS) + "\n")
        for row in rows:
            f.write("# Line: " + "\t".join(row) + "\n")
        f.write("# Columns: wavelength, original, fit, original-fit, baseline\n")
        for w, o, m, b in zip(wavelength, spectrum, fit, baseline):
            f.write(f"{w:.10e}\t{o:.10e}\t{m:.10e}\t{o - m:.10e}\t{b:.10e}\n")


def _parse_line_row(cells):
    """Turn the cells of a line row into a dict with typed values.

    Args:
        cells (list): The text of each cell, in the order of LINE_COLUMNS

    Returns:
        line (dict): The keys of LINE_COLUMNS. A number is a float, or None for an optional number
            that is not available. first_index and last_index are int. success is bool.

    Raises:
        ValueError: If the row has the wrong number of cells
    """
    if len(cells) != len(LINE_COLUMNS):
        raise ValueError(f"A line row has {len(cells)} cells, but {len(LINE_COLUMNS)} are needed.")
    row = dict(zip(LINE_COLUMNS, cells))
    for name in _FLOAT_FIELDS:
        value = float(row[name])
        row[name] = None if (np.isnan(value) and (name in _OPTIONAL_FIELDS or name in _CATALOG_FIELDS)) else value
    for name in ("species", "transition", "source", "area_unit", "message"):
        row[name] = None if row[name] == "-" else row[name]
    row["first_index"], row["last_index"] = int(row["first_index"]), int(row["last_index"])
    row["success"] = row["success"] == "1"
    row["chunk_model"] = _parse_chunk_model(row["chunk_model"])
    return row


def read_fit_file(path):
    """Read a Gaussian fit file that write_fit_file wrote.

    Args:
        path (str): Path of the file

    Returns:
        fit_file (FitFile): The content of the file

    Raises:
        OSError: If the file cannot be opened
        ValueError: If the file has no fit parameters (for example a fit file from an older version),
            or a line row is not valid
    """
    meta = {}
    comments, rows, data = [], [], []
    for line in _read_lines(path):
        if not line.strip():
            continue
        if line.startswith("#"):
            comments.append(line)
            if line.startswith("# Line:"):
                rows.append(_parse_line_row(line[len("# Line:"):].strip("\n").lstrip(" ").split("\t")))
            elif not line.startswith("# Line columns:"):
                match = re.match(r"^#\s*([^:]+?)\s*:\s*(.*?)\s*$", line)
                if match:
                    meta[match.group(1).lower()] = match.group(2)
        else:
            data.append([float(cell) for cell in line.split("\t")])
    if "fit type" not in meta or not rows:
        raise ValueError("This file has no fit parameters. Only a fit file that was saved with "
                         "this version of the program can be loaded.")
    if not data or any(len(r) != 5 for r in data):
        raise ValueError("The data rows of the file must have 5 columns: wavelength, original, fit, original-fit, baseline.")

    markers = {}
    for part in meta.get("markers", "").split(","):
        if "=" in part:
            name, value = part.split("=", 1)
            markers[name.strip()] = float(value)
    table = np.array(data, dtype=float)
    return FitFile(
        fit_type=meta["fit type"], tab=meta.get("spectrum tab"), wave_unit=meta.get("wavelength unit"),
        flux_unit=meta.get("flux unit"), instrument=meta.get("instrument"), baseline_source=meta.get("baseline"),
        markers=markers, lines=rows, wavelength=table[:, 0], original=table[:, 1], fit=table[:, 2],
        baseline=table[:, 4], comments=comments,
        continuum_lines=[line[1:].strip() for line in comments if line.startswith("# Continuum")])


def fit_type_of_comments(comments):
    """Tell if the comment lines of a text file are the header of a fit file, and of which type.
    A fit file starts with the line that write_fit_file writes and has a "Fit type" line.
    Load Spectrum uses this to tell the user that the file also holds a fit.

    Args:
        comments (list): The comment lines of the file, each with its "#" at the start

    Returns:
        fit_type (str): 'single' or 'multi', or None if the file is not a fit file of this version
    """
    if not any(line.lstrip("#").strip() == FIRST_LINE for line in comments):
        return None
    for line in comments:
        match = re.match(r"^#\s*Fit type\s*:\s*(single|multi)\s*$", line)
        if match:
            return match.group(1)
    return None


def _without_empty_rows(fit_file):
    """Make the fit file again without the rows that have no wavelength or no flux.
    Load Spectrum removes these rows, so a spectrum that it loaded from the file is shorter.
    The first and last index of each line are changed to the new row numbers.

    Args:
        fit_file (FitFile): The file that read_fit_file gives

    Returns:
        reduced (FitFile): The file with only the rows that have both numbers, or None if no row is
            removed, or if the rows cannot be removed without a change to a fitted window
    """
    keep = np.isfinite(fit_file.wavelength) & np.isfinite(fit_file.original)
    if keep.all():
        return None
    new_index = np.cumsum(keep) - 1
    kept = np.flatnonzero(keep)
    lines = []
    for row in fit_file.lines:
        row = dict(row)
        if row["success"]:
            inside = kept[(kept >= row["first_index"]) & (kept <= row["last_index"])]
            # A multi fit uses only the points with flux. A single fit needs a window with no gap.
            if inside.size == 0 or (row["chunk_model"] is None and inside.size != row["last_index"] - row["first_index"] + 1):
                return None
            row["first_index"], row["last_index"] = int(new_index[inside[0]]), int(new_index[inside[-1]])
        lines.append(row)
    return replace(fit_file, lines=lines, wavelength=fit_file.wavelength[keep], original=fit_file.original[keep],
                   fit=fit_file.fit[keep], baseline=fit_file.baseline[keep])


def check_same_wavelengths(file_wavelength, wavelength):
    """Check that the wavelengths of a fit file are the wavelengths of the spectrum.
    The fit was made on these wavelengths. On other wavelengths, the saved fit would be wrong.

    Args:
        file_wavelength (numpy.ndarray): Wavelengths in the fit file
        wavelength (numpy.ndarray): Wavelengths of the spectrum that the fit goes on

    Returns:
        None

    Raises:
        ValueError: If the wavelengths are not the same. The message says if they differ by one factor,
            which is what a Doppler shift does.
    """
    file_wavelength = np.asarray(file_wavelength, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)
    if len(file_wavelength) != len(wavelength):
        raise ValueError(f"The file has {len(file_wavelength)} wavelengths, but the spectrum has {len(wavelength)}. "
                         "The fit was made on a different spectrum.")
    if np.allclose(file_wavelength, wavelength, rtol=1e-8, atol=0):
        return
    ratio = wavelength / file_wavelength
    if np.allclose(ratio, ratio[0], rtol=1e-8, atol=0):
        raise ValueError(f"The wavelengths of the spectrum are the wavelengths of the file times {ratio[0]:.9f}. "
                         "That is a Doppler shift: the fit was made with another shift than the spectrum has now. "
                         "Apply or revert the shift, then load the fit again.")
    raise ValueError("The wavelengths of the file are not the wavelengths of the spectrum. "
                     "The fit was made on a different spectrum.")


def rebuild_fit(fit_file, wavelength, spectrum):
    """Build the fit results of a fit file again, on the spectrum that they were fitted on.
    This does not fit again. It evaluates the saved Gaussians and the saved baseline.

    Args:
        fit_file (FitFile): The file that read_fit_file gives
        wavelength (numpy.ndarray): Wavelengths of the spectrum. They must be the wavelengths of the file.
        spectrum (numpy.ndarray): Flux of the spectrum

    Returns:
        result (dict): 'entries' (list of dicts with 'kind', 'line' and 'result', in the order of the file)
            and 'notes' (list of str: warnings about the data, empty if none)

    Raises:
        ValueError: If the wavelengths are not the wavelengths of the file, or a line cannot be built
    """
    wavelength = np.asarray(wavelength, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    if len(fit_file.wavelength) != len(wavelength):
        reduced = _without_empty_rows(fit_file)
        if reduced is not None and len(reduced.wavelength) == len(wavelength):
            fit_file = reduced
    check_same_wavelengths(fit_file.wavelength, wavelength)
    if len(spectrum) != len(wavelength):
        raise ValueError("The spectrum and its wavelengths have different lengths.")

    notes = []
    both = np.isfinite(spectrum) & np.isfinite(fit_file.original)
    scale = np.nanmax(np.abs(spectrum)) if np.isfinite(spectrum).any() else 1.0
    if both.any() and not np.allclose(spectrum[both], fit_file.original[both], rtol=1e-6, atol=1e-8 * scale):
        notes.append("The flux of the spectrum is not the flux that the fit was made on. "
                     "The fit curve may not follow the data.")

    entries = []
    for row in fit_file.lines:
        line = {name: row[name] for name in ("species", "transition", "source") if row[name] is not None}
        for name in _CATALOG_FIELDS:
            if row[name] is not None:
                if name == "catalog_wavelength":
                    line["lambda_micron"] = line["wavelength"] = row[name]
                else:
                    line[name] = row[name]
        entries.append({"kind": row["kind"], "line": line, "result": _rebuild_result(row, fit_file, wavelength, spectrum)})
    return {"entries": entries, "notes": notes}


def _rebuild_result(row, fit_file, wavelength, spectrum):
    """Build one GaussianFitResult from a line row.

    Args:
        row (dict): A line of the fit file, as _parse_line_row gives it
        fit_file (FitFile): The file that the row is from
        wavelength (numpy.ndarray): Wavelengths of the spectrum
        spectrum (numpy.ndarray): Flux of the spectrum

    Returns:
        result (GaussianFitResult): The result. A line that did not fit has empty arrays and success False.
    """
    numbers = dict(
        amplitude=row["amplitude"], center=row["center"], sigma=row["sigma"], fwhm=row["fwhm"], area=row["area"],
        area_native=row["area_native"], baseline_slope=row["baseline_slope"],
        baseline_intercept=row["baseline_intercept"], r_squared=row["r_squared"],
        amplitude_error=row["amplitude_error"], sigma_error=row["sigma_error"], area_error=row["area_error"],
        catalog_sigma=row["catalog_sigma"], noise_std=row["noise_std"], area_unit=row["area_unit"] or "",
        success=row["success"], message=row["message"] or "")
    if not row["success"]:
        empty = np.array([])
        return GaussianFitResult(x_data=empty, y_data=empty, baseline=empty, y_corrected=empty, y_fit=empty,
                                 residuals=empty, **numbers)

    window = np.arange(row["first_index"], row["last_index"] + 1)
    x_data, flux, baseline = wavelength[window], spectrum[window], fit_file.baseline[window]
    model = row["chunk_model"]
    if model is None:
        # A single fit: the baseline is a straight line or the fitted continuum. The Gaussian is the model.
        y_data, y_corrected = flux, flux - baseline
        y_fit = gaussian(x_data, row["amplitude"], row["center"], row["sigma"])
        y_fit_own = None
    else:
        # A multi fit works on the flux minus the continuum, in the points where that is finite.
        keep = np.isfinite(x_data) & np.isfinite(flux - baseline)
        x_data, baseline = x_data[keep], baseline[keep]
        y_data = (flux - fit_file.baseline[window])[keep]
        y_corrected = y_data
        local = model["a"] + model["b"] * (x_data - model["x0"])
        y_fit = local + sum(gaussian(x_data, *g) for g in model["gaussians"])
        y_fit_own = local + gaussian(x_data, row["amplitude"], row["center"], row["sigma"])
    return GaussianFitResult(
        x_data=x_data, y_data=y_data, baseline=baseline, y_corrected=y_corrected, y_fit=y_fit,
        residuals=y_corrected - y_fit, y_fit_own=y_fit_own, chunk_model=model, **numbers)
