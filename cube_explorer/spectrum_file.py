"""
Read a spectrum from a text file (txt, csv, tsv) or from a 1D FITS table (for example a JWST x1d file).
The file has a wavelength column, a flux column, and an optional flux error column.
This module also writes the file that "Save Spectrum" makes (spectrum_header_lines, write_spectrum_file),
and the reader above reads that file again.
This module has no GUI code. The GUI and the API both use it.
"""

import os
import re
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits

from .unit_converter import doppler_factor_to_velocity, spectral_axis_medium

# The fewest points that the spectrum tools can work with.
MIN_POINTS = 10

# Comment lines of the form "# Key: value". Each key that the Save Spectrum file writes is listed here.
_HEADER_KEYS = {
    "wavelength unit": "wave_unit",
    "flux unit": "flux_unit",
    "wavelength medium": "medium",
    "instrument": "instrument",
    "grating": "grating",
    "object": "object",
    "source file": "source_file",
    "columns": "columns",
}

_X_NAMES = ("wavelength", "wave", "lambda", "wl", "lam", "x")
_Y_NAMES = ("original", "flux", "spectrum", "intensity", "signal", "y", "f")
_ERROR_WORDS = ("err", "sigma", "unc", "std")

# Spellings of the wavelength units. Each maps to the unit name that the cubes use,
# and to the number that turns a value in this spelling into that unit.
_WAVE_UNITS = {
    "um": ("um", 1.0), "µm": ("um", 1.0), "μm": ("um", 1.0), "micron": ("um", 1.0),
    "microns": ("um", 1.0), "micrometer": ("um", 1.0), "micrometre": ("um", 1.0),
    "nm": ("um", 1.0e-3), "nanometer": ("um", 1.0e-3), "nanometre": ("um", 1.0e-3),
    "å": ("Å", 1.0), "a": ("Å", 1.0), "angstrom": ("Å", 1.0), "angstroms": ("Å", 1.0),
}

_FITS_EXTENSIONS = (".fits", ".fit", ".fts")
_FITS_WAVE_NAMES = ("wavelength", "wave", "lambda", "wl", "lam")
_INSTRUMENT_NAMES = {"MIRI": "MIRI", "NIRSPEC": "NIRSpec", "MUSE": "MUSE"}
# Bit 0 of the DQ column of a JWST file. It marks a value that must not be used.
_JWST_DO_NOT_USE = 1


@dataclass
class SpectrumTable:
    """The numbers and the header facts of one spectrum file.

    Attributes:
        path (str): Path of the file
        values (numpy.ndarray): 2D array with one row for each data line. A value that is not a number is NaN.
        column_names (list): Name of each column. "Column 1", "Column 2" and so on if the file gives none.
        names_found (bool): True if the file gives the column names
        metadata (dict): Facts from the comment lines or the FITS header: wave_unit, flux_unit, medium,
            instrument, grating, object, source_file, shift_factor
        skipped_rows (int): Number of data lines that had the wrong number of columns
        comments (list): The comment lines of the file, as written, each with its "#" at the start.
            For a FITS file, these are the header cards.
        column_units (list): The unit of each column, as a FITS table gives it. Empty for a text file.
        notes (list): Sentences (str) that say what the reader removed from the data
    """
    path: str
    values: np.ndarray
    column_names: list
    names_found: bool
    metadata: dict = field(default_factory=dict)
    skipped_rows: int = 0
    comments: list = field(default_factory=list)
    column_units: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _read_lines(path):
    """Read the lines of a text file. UTF-8 is tried first, then Windows-1252, because
    older files can write the angstrom sign in Windows-1252.

    Args:
        path (str): Path of the file

    Returns:
        lines (list): The lines of the file, with no line break at the end

    Raises:
        OSError: If the file cannot be opened
    """
    with open(path, 'rb') as handle:
        raw = handle.read()
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = raw.decode('cp1252', errors='replace')
    return text.splitlines()


def _split(line, delimiter):
    """Split one line into cells.

    Args:
        line (str): One line of the file
        delimiter (str): The separator, or None to split on any run of white space

    Returns:
        cells (list): The cells, with the white space around each cell removed
    """
    return [cell.strip() for cell in (line.split(delimiter) if delimiter else line.split())]


def _find_delimiter(lines):
    """Find the separator of the data lines. A tab, a comma and a semicolon are tried in this
    order. A separator is right if it gives two or more cells on every line, the same number each time.

    Args:
        lines (list): The data lines (no comment lines, no empty lines)

    Returns:
        delimiter (str): The separator, or None if the cells are separated by white space
    """
    sample = lines[:30]
    for delimiter in ('\t', ',', ';'):
        counts = {len(line.strip().split(delimiter)) for line in sample}
        if len(counts) == 1 and min(counts) >= 2:
            return delimiter
    return None


def _to_float(cell):
    """Turn one cell into a number.

    Args:
        cell (str): Text of the cell

    Returns:
        value (float): The number, or NaN if the cell is empty or not a number
    """
    try:
        return float(cell)
    except ValueError:
        return float('nan')


def _parse_comment(line, metadata):
    """Read one comment line. A line of the form "# Key: value" with a known key adds a fact to metadata.

    Args:
        line (str): A comment line, with the "#" at the start
        metadata (dict): The facts found so far. This function adds to it.

    Returns:
        None
    """
    match = re.match(r'^#\s*([^:]+?)\s*:\s*(.*?)\s*$', line)
    if match is None:
        return
    key, value = match.group(1).lower(), match.group(2)
    if key in _HEADER_KEYS and value:
        metadata[_HEADER_KEYS[key]] = value
    elif key == "wavelength shift factor":
        try:
            metadata["shift_factor"] = float(value.split()[0])
        except (ValueError, IndexError):
            pass
    elif key == "wavelength shift":
        # Older files give the factor on the same line, as "... km/s, factor 0.9998".
        factor = re.search(r'factor\s+([-+0-9.eE]+)', value)
        if factor is not None:
            metadata["shift_factor"] = float(factor.group(1))


def read_spectrum_table(path):
    """Read a text file into a table of numbers, with the facts that the comment lines give.
    The separator can be a tab, a comma, a semicolon or white space. A first line of column names is allowed.
    Lines that start with "#" are comments. The Save Spectrum file of IFU Analyzer gives its units here.

    Args:
        path (str): Path of the text file

    Returns:
        table (SpectrumTable): The numbers, the column names and the facts of the header

    Raises:
        OSError: If the file cannot be opened
        ValueError: If the file has no data lines, or fewer than 2 columns
    """
    lines = _read_lines(path)
    metadata = {}
    comments = []
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            _parse_comment(stripped, metadata)
            comments.append(stripped)
        else:
            data_lines.append(stripped)
    if not data_lines:
        raise ValueError("The file has no data lines.")

    delimiter = _find_delimiter(data_lines)
    rows = [_split(line, delimiter) for line in data_lines]

    # A first line with no numbers at all is a row of column names.
    names = None
    if all(np.isnan(_to_float(cell)) for cell in rows[0]):
        names = rows.pop(0)
    n_columns = len(rows[0]) if rows else len(names or [])
    if n_columns < 2:
        raise ValueError("The file needs at least 2 columns: wavelength and flux.")

    if names is None and "columns" in metadata:
        listed = [name.strip() for name in metadata["columns"].split(',')]
        if len(listed) == n_columns:
            names = listed
    if names is None and comments:
        # A column-name line can be a comment, for example "# wavelength flux error".
        last = [cell for cell in _split(comments[-1].lstrip('#'), delimiter) if cell]
        if len(last) == n_columns and all(np.isnan(_to_float(cell)) for cell in last):
            names = last
    names_found = names is not None and len(names) == n_columns
    if not names_found:
        names = [f"Column {i + 1}" for i in range(n_columns)]

    good = [row for row in rows if len(row) == n_columns]
    values = np.array([[_to_float(cell) for cell in row] for row in good], dtype=float).reshape(-1, n_columns)
    return SpectrumTable(path=path, values=values, column_names=list(names), names_found=names_found,
                         metadata=metadata, skipped_rows=len(rows) - len(good), comments=comments)


def is_fits_file(path):
    """Tell if a file is a FITS file. The name ends with .fits, .fit or .fts, or the file starts with "SIMPLE".

    Args:
        path (str): Path of the file

    Returns:
        is_fits (bool): True if the file is a FITS file

    Raises:
        OSError: If the file cannot be opened
    """
    if os.path.splitext(path)[1].lower() in _FITS_EXTENSIONS:
        return True
    with open(path, 'rb') as handle:
        return handle.read(6) == b'SIMPLE'


def _header_comment_lines(label, header):
    """Write a FITS header as comment lines, one for each card, so the Header button can show them.

    Args:
        label (str): Name of the header, for example "PRIMARY header"
        header (astropy.io.fits.Header): The header

    Returns:
        lines (list): The comment lines (str), each with its "#" at the start
    """
    return [f"# --- {label} ---"] + ["# " + card.image.strip() for card in header.cards]


def read_spectrum_fits(path):
    """Read a 1D spectrum from a FITS table, for example the EXTRACT1D table of a JWST x1d file.
    The first table that has a wavelength column is used. Each column of numbers becomes a column of the
    table, with the unit that the file gives. Rows that a JWST DQ column marks as DO_NOT_USE are removed.

    Args:
        path (str): Path of the FITS file

    Returns:
        table (SpectrumTable): The numbers, the column names, the column units and the facts of the header

    Raises:
        OSError: If the file cannot be opened
        ValueError: If the file has no table with a wavelength column, holds a cube or an image,
            has a wavelength unit that this program does not read, or has fewer than 2 columns of numbers
    """
    with fits.open(path) as hdul:
        primary = hdul[0].header
        table_hdu = None
        image_shapes = []
        for hdu in hdul:
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)) and hdu.data is not None:
                if any(column.name.lower() in _FITS_WAVE_NAMES for column in hdu.columns):
                    table_hdu = hdu
                    break
            elif isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)) and len(hdu.shape) >= 2:
                image_shapes.append(hdu.shape)
        if table_hdu is None:
            if image_shapes:
                raise ValueError(f"This FITS file holds an image or a cube {image_shapes[0]}, not a 1D spectrum. "
                                 "Open a cube with Open FITS.")
            raise ValueError("This FITS file has no table with a wavelength column (WAVELENGTH, WAVE or LAMBDA).")

        columns = []
        for column in table_hdu.columns:
            values = np.asarray(table_hdu.data[column.name])
            if values.ndim == 2 and values.shape[0] == 1:
                values = values[0]  # One row, and each cell holds a whole spectrum
            if values.ndim == 1 and values.dtype.kind in "fiu":
                columns.append((column.name, str(column.unit or "").strip(), values.astype(float)))
        wave_index = next((i for i, (name, _, _) in enumerate(columns) if name.lower() in _FITS_WAVE_NAMES), None)
        if wave_index is None:
            raise ValueError("The wavelength column of this FITS table does not hold numbers.")
        n_rows = columns[wave_index][2].size
        columns = [entry for entry in columns if entry[2].size == n_rows]
        wave_index = next(i for i, (name, _, _) in enumerate(columns) if name.lower() in _FITS_WAVE_NAMES)
        if len(columns) < 2:
            raise ValueError("The FITS table needs at least 2 columns of numbers: wavelength and flux.")

        wave_name, wave_unit, _ = columns[wave_index]
        if wave_unit:
            try:
                normalize_wave_unit(wave_unit)
            except ValueError:
                raise ValueError(f"The wavelength column {wave_name} is in '{wave_unit}'. "
                                 "This program reads um, nm and Angstrom.") from None

        names = [entry[0] for entry in columns]
        arrays = [entry[2] for entry in columns]
        notes = []
        telescope = str(primary.get("TELESCOP", "")).strip().upper()
        if telescope == "JWST" and "DQ" in names:
            flagged = (np.nan_to_num(arrays[names.index("DQ")]).astype(np.int64) & _JWST_DO_NOT_USE) != 0
            if flagged.any():
                arrays = [array[~flagged] for array in arrays]
                notes.append(f"{int(flagged.sum())} row(s) that the DQ column marks as DO_NOT_USE were removed.")

        metadata = {}
        if wave_unit:
            metadata["wave_unit"] = wave_unit
        instrument = str(primary.get("INSTRUME", "")).strip()
        if instrument:
            metadata["instrument"] = _INSTRUMENT_NAMES.get(instrument.upper(), instrument)
        grating = str(primary.get("GRATING", "")).strip()
        if grating:
            metadata["grating"] = grating
        target = str(primary.get("TARGNAME", primary.get("OBJECT", ""))).strip()
        if target:
            metadata["object"] = target
        if telescope == "JWST":
            metadata["medium"] = "vacuum"

        comments = (_header_comment_lines("PRIMARY header", primary)
                    + _header_comment_lines(f"{table_hdu.name} header", table_hdu.header))
    return SpectrumTable(path=path, values=np.column_stack(arrays), column_names=names, names_found=True,
                         metadata=metadata, comments=comments, column_units=[entry[1] for entry in columns],
                         notes=notes)


def read_spectrum_file(path):
    """Read a spectrum file: a FITS table, or a text file (txt, csv, tsv).

    Args:
        path (str): Path of the file

    Returns:
        table (SpectrumTable): The numbers, the column names and the facts of the header

    Raises:
        OSError: If the file cannot be opened
        ValueError: If the file is not a valid spectrum, see read_spectrum_fits and read_spectrum_table
    """
    return read_spectrum_fits(path) if is_fits_file(path) else read_spectrum_table(path)


def column_flux_unit(table, y_index):
    """Give the unit that a FITS table states for one column. A text file states none.

    Args:
        table (SpectrumTable): The table that read_spectrum_file gives
        y_index (int): Index of the column

    Returns:
        unit (str): The unit, for example "Jy" or "MJy/sr". An empty string if the table gives none.
    """
    if 0 <= y_index < len(table.column_units):
        return table.column_units[y_index]
    return ""


def suggest_error_column(table, y_index):
    """Find the error column that goes with a flux column of a FITS table. It is the first column
    with an error word in its name and the same unit as the flux. For FLUX in Jy this is FLUX_ERROR.
    For SURF_BRIGHT in MJy/sr this is SB_ERROR.

    Args:
        table (SpectrumTable): The table that read_spectrum_file gives
        y_index (int): Index of the flux column

    Returns:
        error_index (int): Index of the error column, or None if the table gives no units or no column fits
    """
    unit = column_flux_unit(table, y_index)
    if not unit:
        return None
    for index, name in enumerate(table.column_names):
        if (index != y_index and table.column_units[index] == unit
                and any(word in name.lower() for word in _ERROR_WORDS)):
            return index
    return None


def is_save_spectrum_file(comments):
    """Tell if the comment lines of a text file are the header of a file that Save Spectrum wrote.
    That file has the columns wavelength, original, continuum, subtracted and plotted.

    Args:
        comments (list): The comment lines of the file, each with its "#" at the start

    Returns:
        is_spectrum_file (bool): True if the first comment line is the first line of a Save Spectrum file
    """
    return bool(comments) and comments[0].lstrip("#").strip().startswith("IFU Analyzer spectrum file")


def suggest_columns(table):
    """Guess which column is the wavelength, which is the flux, and which is the flux error.
    The guess uses the column names. Without names, the columns are wavelength, flux and error, in that order.
    In a Save Spectrum file, the flux is the "Original" column.

    Args:
        table (SpectrumTable): The table that read_spectrum_table gives

    Returns:
        columns (tuple): (x_index, y_index, error_index). error_index is None if no column looks like an error.
    """
    n_columns = table.values.shape[1]
    lowered = [name.lower().strip() for name in table.column_names]

    def first_match(candidates, exclude):
        """Find the first column whose name starts with one of the candidate words.

        Args:
            candidates (tuple): Words to look for at the start of a name
            exclude (list): Column indexes that are not allowed

        Returns:
            index (int): Column index, or None if no name matches
        """
        for word in candidates:
            for index, name in enumerate(lowered):
                token = re.split(r'[\s_(\[]', name)[0]
                same = token == word or (len(word) > 2 and token.startswith(word))
                if index not in exclude and same:
                    return index
        return None

    if table.names_found:
        x_index = first_match(_X_NAMES, [])
        x_index = 0 if x_index is None else x_index
        error_index = next((i for i, name in enumerate(lowered)
                            if i != x_index and any(word in name for word in _ERROR_WORDS)), None)
        exclude = [x_index] + ([error_index] if error_index is not None else [])
        y_index = first_match(_Y_NAMES, exclude)
        if y_index is None:
            y_index = next(i for i in range(n_columns) if i not in exclude)
        return x_index, y_index, error_index

    return 0, 1, (2 if n_columns == 3 else None)


def normalize_wave_unit(text):
    """Turn a wavelength unit, as a person writes it, into the unit name that the cubes use.

    Args:
        text (str): The unit, for example "um", "µm", "micron", "nm" or "Angstrom"

    Returns:
        result (tuple): (unit, factor). unit is "um" or "Å". factor is the number that turns a value
            in the given unit into that unit. A nanometer value becomes micron, so factor is 0.001.

    Raises:
        ValueError: If the unit is not a unit of length that this program uses
    """
    key = str(text).strip().lower()
    if key not in _WAVE_UNITS:
        raise ValueError(f"Wavelength unit '{text}' is not known. Use um, nm, or Angstrom.")
    return _WAVE_UNITS[key]


def prepare_spectrum(table, x_index, y_index, error_index=None):
    """Take the three columns out of a table and make them ready to use.
    A row with no wavelength or no flux is removed. The rows go in order of increasing wavelength.

    Args:
        table (SpectrumTable): The table that read_spectrum_table gives
        x_index (int): Index of the wavelength column
        y_index (int): Index of the flux column
        error_index (int): Index of the flux error column, or None for no error

    Returns:
        result (dict): wavelength (numpy.ndarray), flux (numpy.ndarray), error (numpy.ndarray or None),
            notes (list of str). A note says what was changed in the data.

    Raises:
        ValueError: If two columns are the same one, an index is not in the table, an error is negative,
            a wavelength is repeated, or fewer than MIN_POINTS rows are left
    """
    n_columns = table.values.shape[1]
    chosen = [x_index, y_index] + ([error_index] if error_index is not None else [])
    if len(set(chosen)) != len(chosen):
        raise ValueError("The wavelength, flux and error must be three different columns.")
    if any(not 0 <= index < n_columns for index in chosen):
        raise ValueError(f"The file has {n_columns} columns. A chosen column is not one of them.")

    x = table.values[:, x_index]
    y = table.values[:, y_index]
    error = table.values[:, error_index] if error_index is not None else None

    notes = list(table.notes)
    if table.skipped_rows:
        notes.append(f"{table.skipped_rows} line(s) had the wrong number of columns and were skipped.")
    keep = np.isfinite(x) & np.isfinite(y)
    if not keep.all():
        notes.append(f"{int((~keep).sum())} row(s) with no wavelength or no flux were removed.")
    x, y = x[keep], y[keep]
    if error is not None:
        error = error[keep]
        if np.any(error[np.isfinite(error)] < 0):
            raise ValueError("The error column has negative values. An error cannot be negative.")
    if x.size < MIN_POINTS:
        raise ValueError(f"Only {x.size} usable rows. The tools need at least {MIN_POINTS}.")

    order = np.argsort(x, kind='stable')
    if not np.array_equal(order, np.arange(x.size)):
        notes.append("The rows were not in order of increasing wavelength. They were sorted.")
        x, y = x[order], y[order]
        if error is not None:
            error = error[order]
    if np.any(np.diff(x) == 0):
        raise ValueError("The wavelength column has a value that appears more than once.")
    if error is not None and not np.isfinite(error).any():
        error = None
        notes.append("The error column has no numbers, so it was not used.")
    return {"wavelength": x, "flux": y, "error": error, "notes": notes}


def default_settings(table):
    """Give the settings for a table from what the file itself says. The GUI dialog starts from these
    values. A file that Save Spectrum wrote gives every setting, so it needs no question.

    Args:
        table (SpectrumTable): The table that read_spectrum_table gives

    Returns:
        settings (dict): x_index, y_index, error_index, wave_unit (str, as written), flux_unit (str),
            medium ('air', 'vacuum' or None), instrument (str or None), grating (str or None),
            object (str or None), complete (bool). complete is True if the file gives both units
            and the column names, so nothing needs to be asked.
    """
    x_index, y_index, error_index = suggest_columns(table)
    paired = suggest_error_column(table, y_index)
    error_index = error_index if paired is None else paired
    meta = table.metadata
    medium = meta.get("medium", "").lower()
    instrument = meta.get("instrument", "")
    flux_unit = meta["flux_unit"] if "flux_unit" in meta else column_flux_unit(table, y_index)
    return {
        "x_index": x_index, "y_index": y_index, "error_index": error_index,
        "wave_unit": meta.get("wave_unit", "um"),
        "flux_unit": "" if flux_unit.lower() == "unknown" else flux_unit,
        "medium": medium if medium in ("air", "vacuum") else None,
        "instrument": None if instrument.lower() in ("", "unknown", "none") else instrument,
        "grating": meta.get("grating") or None,
        "object": meta.get("object") or None,
        "complete": "wave_unit" in meta and ("flux_unit" in meta or bool(flux_unit)) and table.names_found,
    }


def plotted_description(subtracted, divisor, divisor_note, normalization_on):
    """Say how the "plotted" column of a saved spectrum was made, as a sentence for the file header.

    Args:
        subtracted (bool): True if the continuum was subtracted from the plotted column
        divisor (float): The number that the plotted column was divided by, 1.0 if it was not divided
        divisor_note (str): Where the divisor comes from, for example "peak of the Full spectrum"
        normalization_on (bool): True if a normalization was asked for. It is used only when
            divisor is 1.0, because then the normalization did nothing

    Returns:
        text (str): The steps, separated by semicolons
    """
    steps = []
    if subtracted:
        steps.append("continuum subtracted")
    if divisor != 1.0:
        steps.append(f"divided by {divisor:.6e} ({divisor_note})")
    elif normalization_on:
        steps.append("normalization is on, but the peak is not a positive number, so the plot is not normalized")
    return "; ".join(steps) if steps else "same as Original (no subtraction, no normalization)"


def spectrum_header_lines(tab, spaxels, wave_unit, flux_unit, plotted_text, divisor, shift_factor,
                          continuum_lines, has_error=False, header=None, instrument=None, filename=None):
    """Build the comment lines at the top of a saved spectrum file.
    The lines give the units, instrument, wavelength frame, continuum and column names. They hold
    all the facts that the spectrum tools need to work on the file without the FITS cube.

    Args:
        tab (str): Spectrum tab, 'full', 'left' or 'right'
        spaxels (str): Sentence that says which spaxels the spectrum sums over
        wave_unit (str): Wavelength unit of the saved wavelengths
        flux_unit (str): Flux unit of the Original, Continuum and Subtracted columns
        plotted_text (str): Sentence that says how the Plotted column was made
        divisor (float): Normalization divisor of the Plotted column, 1.0 if not normalized
        shift_factor (float): Doppler shift factor of the wavelength axis, 1.0 if not shifted
        continuum_lines (list): Lines about the continuum, each without "#"
        has_error (bool): True if the file has an Error column after the Plotted column
        header (astropy.io.fits.Header): Header of the cube, or None. It gives the object name and
            the wavelength medium (air or vacuum)
        instrument (str): Instrument name, or None
        filename (str): Path or name of the cube file, or None

    Returns:
        lines (list): Header lines. Each line starts with '#', with no line break at the end.
    """
    instrument = instrument or "Unknown"
    medium = spectral_axis_medium(header) or "unknown"

    shift_velocity = doppler_factor_to_velocity(shift_factor)

    lines = ["# IFU Analyzer spectrum file, format 1"]
    if filename:
        lines.append(f"# Source file: {os.path.basename(filename)}")
    if header is not None and str(header.get('OBJECT', '')).strip():
        lines.append(f"# Object: {str(header.get('OBJECT')).strip()}")
    lines.append(f"# Instrument: {instrument}")
    if header is not None and str(instrument).upper() == "NIRSPEC" and str(header.get('GRATING', '')).strip():
        lines.append(f"# Grating: {str(header.get('GRATING')).strip().upper()}")
    lines.append(f"# Spectrum tab: {tab}")
    lines.append(f"# Spaxels: {spaxels}")
    lines.append(f"# Wavelength unit: {wave_unit}")
    lines.append(f"# Wavelength medium: {medium}")
    # 0.0 stops a rounding error from printing "-0".
    shift_velocity = 0.0 if abs(shift_velocity) < 5e-7 else shift_velocity
    lines.append(f"# Wavelength shift: {shift_velocity:.6f}".rstrip('0').rstrip('.') + " km/s")
    lines.append(f"# Wavelength shift factor: {shift_factor:.12g}")
    lines.append(f"# Flux unit: {flux_unit}")
    lines.extend(f"# {text}" for text in continuum_lines)
    lines.append(f"# Plotted: {plotted_text}")
    if divisor != 1.0:
        lines.append("# Plotted is dimensionless because it is normalized.")
    lines.append("# Columns: wavelength, original, continuum, subtracted, plotted" + (", error" if has_error else ""))
    return lines


def write_spectrum_file(path, header_lines, wavelength, original, continuum, plotted, error=None):
    """Write a spectrum file in the format of "Save Spectrum". The columns are wavelength, original,
    continuum, subtracted and plotted, and error if it is given. The subtracted column is
    original minus continuum where the continuum is a number, and NaN where it is not.

    Args:
        path (str): Path of the file to write
        header_lines (list): Comment lines, as spectrum_header_lines gives them
        wavelength (numpy.ndarray): Wavelength values
        original (numpy.ndarray): Flux values, with the same length
        continuum (numpy.ndarray): The fitted continuum, or None if there is none. It is written as NaN.
        plotted (numpy.ndarray): The plotted column, with the same length
        error (numpy.ndarray): Flux error of each point, or None for no error column

    Returns:
        None

    Raises:
        OSError: if the file cannot be written
    """
    if continuum is None:
        continuum = np.full_like(original, np.nan, dtype=float)
    subtracted = np.where(np.isfinite(continuum), original - continuum, np.nan)
    with open(path, 'w') as f:
        for line in header_lines:
            f.write(line + "\n")
        for i, (w, o, c, s, p) in enumerate(zip(wavelength, original, continuum, subtracted, plotted)):
            row = f"{w:.10e}\t{o:.6e}\t{c:.6e}\t{s:.6e}\t{p:.6e}"
            if error is not None:
                row += f"\t{error[i]:.6e}"
            f.write(row + "\n")
