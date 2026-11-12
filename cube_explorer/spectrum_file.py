"""
Read a spectrum from a text file (txt, csv, tsv).
The file has a wavelength column, a flux column, and an optional flux error column.
This module has no GUI code. The GUI and the API both use it.
"""

import re
from dataclasses import dataclass, field

import numpy as np

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


@dataclass
class SpectrumTable:
    """The numbers and the header facts of one text file.

    Attributes:
        path (str): Path of the file
        values (numpy.ndarray): 2D array with one row for each data line. A value that is not a number is NaN.
        column_names (list): Name of each column. "Column 1", "Column 2" and so on if the file gives none.
        names_found (bool): True if the file gives the column names
        metadata (dict): Facts from the comment lines: wave_unit, flux_unit, medium, instrument,
            grating, object, source_file, shift_factor
        skipped_rows (int): Number of data lines that had the wrong number of columns
        comments (list): The comment lines of the file, as written, each with its "#" at the start
    """
    path: str
    values: np.ndarray
    column_names: list
    names_found: bool
    metadata: dict = field(default_factory=dict)
    skipped_rows: int = 0
    comments: list = field(default_factory=list)


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

    notes = []
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
    meta = table.metadata
    medium = meta.get("medium", "").lower()
    instrument = meta.get("instrument", "")
    return {
        "x_index": x_index, "y_index": y_index, "error_index": error_index,
        "wave_unit": meta.get("wave_unit", "um"),
        "flux_unit": "" if meta.get("flux_unit", "").lower() == "unknown" else meta.get("flux_unit", ""),
        "medium": medium if medium in ("air", "vacuum") else None,
        "instrument": None if instrument.lower() in ("", "unknown", "none") else instrument,
        "grating": meta.get("grating") or None,
        "object": meta.get("object") or None,
        "complete": "wave_unit" in meta and "flux_unit" in meta and table.names_found,
    }
