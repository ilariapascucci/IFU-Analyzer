"""
IFU Analyzer API - Non-GUI programmatic interface

This module provides a clean API for using IFU analysis functions without the GUI.
All functions in this module work as pure Python functions and can be imported
and used in scripts, Jupyter notebooks, or other Python applications.

Main Categories:
1. Data Loading        - Load FITS cubes and extract spectra
2. Gaussian Fitting    - Fit spectral lines with Gaussian profiles
3. Flux Analysis       - Extract target-line fluxes with continuum handling
4. Rotational Diagram  - Analyze H2 rotational temperatures from molecular transitions
5. Unit Conversion     - Convert between wavelength/velocity/frequency units
6. Line Catalogs       - Query and filter spectral line databases

Example Usage
=============

Basic data loading and spectrum extraction:
    >>> from cube_explorer.api import load_cube
    >>> cube = load_cube("my_data.fits")
    >>> spectrum = cube.spectrum

Gaussian line fitting:
    >>> from cube_explorer.api import fit_line
    >>> result = fit_line(
    ...     wavelength=cube.wavelength,
    ...     spectrum=spectrum,
    ...     x1=2.292, x2=2.300
    ... )
    >>> print(f"Line center: {result.center} um")
    >>> print(f"FWHM: {result.fwhm} um")

Flux extraction:
    >>> from cube_explorer.api import extract_flux
    >>> flux_result = extract_flux(
    ...     wavelength=cube.wavelength,
    ...     spectrum=spectrum,
    ...     target_catalog_path="path/to/CO_catalog.ecsv"
    ... )
    >>> print(f"Total flux: {flux_result['flux']} erg/s/cm^2")

Rotational diagram analysis (H2):
    Needs at least 5 lines to meaningfully constrain a 2-temperature fit (4 free
    parameters) -- fewer than that and fit2T=True is technically callable but gives
    degenerate, meaningless uncertainties. molecular_data below is the standard H2
    v=0->0 S(1)-S(7) pure-rotational set (5.5-17 um), matching data/H2_lines.csv:

    >>> from cube_explorer.api import rotational_diagram
    >>> molecular_data = {
    ...     'wavelengths': {1: 17.034846, 2: 12.278612, 3: 9.664911, 4: 8.025041,
    ...                     5: 6.909509, 6: 6.108564, 7: 5.511183},
    ...     'Aul': {1: 4.758e-10, 2: 2.753e-9, 3: 9.83e-9, 4: 2.641e-8,
    ...             5: 5.876e-8, 6: 1.141e-7, 7: 2.0e-7},
    ...     'Eu': {1: 1015.947, 2: 1683.068, 3: 2505.873, 4: 3477.452,
    ...            5: 4589.958, 6: 5834.798, 7: 7202.827},
    ...     'gu': {1: 7, 2: 9, 3: 11, 4: 13, 5: 15, 6: 17, 7: 19},  # bare 2*J_upper+1;
    ...         # apply_OPR() applies the ortho/para nuclear-spin weighting on top of this
    ... }
    >>> result = rotational_diagram(
    ...     fluxes={1: 178e-16, 2: 55.9e-16, 3: 122e-16, 4: 28.4e-16,
    ...             5: 72.3e-16, 6: 14.4e-16, 7: 22.2e-16},
    ...     flux_errors={1: 9e-16, 2: 3.2e-16, 3: 6.3e-16, 4: 2.4e-16,
    ...                  5: 4e-16, 6: 3.2e-16, 7: 3.8e-16},
    ...     molecular_data=molecular_data,
    ...     R=8.5,     # AU
    ...     dist=71.5, # pc
    ...     fit2T=True
    ... )
    >>> print(f"1T Temperature: {result['1T']['T']:.1f} K")
    >>> print(f"2T Temperatures: {result['2T']['T1']:.1f} K, {result['2T']['T2']:.1f} K")
    1T Temperature: 535.6 K
    2T Temperatures: 273.4 K, 863.4 K

Unit conversion (using astropy):
    >>> import astropy.units as u
    >>> wavelength = 2.293 * u.um
    >>> frequency = wavelength.to(u.Hz, equivalencies=u.spectral_density(wavelength))
    >>> print(f"Frequency: {frequency}")
"""

import time
import warnings

import numpy as np
from astropy import units as u
from astropy.io import fits

from .unit_converter import _string_to_astropy_unit


def _spectrum_and_header(spectrum, header):
    """Split a spectrum into plain numbers and the header to fit with.
    A spectrum that carries a unit (an astropy Quantity, like cube.spectrum) gives its own unit,
    and the header gets that unit as BUNIT. A spectrum in Jy is then never read as MJy/sr
    because it came with the header of a cube.

    Args:
        spectrum (array-like or Quantity): Flux values, with or without a unit.
        header (astropy.io.fits.Header or None): FITS header that the user gave.

    Returns:
        result (tuple): (values, header). values is a float array. header is the given header,
            a copy of it with BUNIT set to the unit of the spectrum, or (if none was given)
            a new header that holds only that unit. A spectrum with no unit leaves the header as it is.
    """
    values = _as_float_array(spectrum)
    unit = getattr(spectrum, "unit", None)
    if unit is None or unit == u.dimensionless_unscaled:
        return values, header
    if header is None:
        header = fits.Header()
    elif _string_to_astropy_unit(header.get("BUNIT")) == unit:
        return values, header
    else:
        header = header.copy()
    header["BUNIT"] = unit.to_string()
    return values, header


def _as_float_array(values):
    """Return a plain float ndarray, stripping astropy units if present.
    load_cube returns cube.wavelength as a Quantity; arithmetic on a united
    array can silently produce NaN, so every entry point normalizes through this.

    Args:
        values: Quantity, scalar, list, or array to convert.

    Returns:
        arr (numpy.ndarray): Float array with units stripped.
    """
    return np.asarray(getattr(values, "value", values), dtype=float)


# --- Data Loading API ---

from .data_loader import CubeData

def load_cube(filename):
    """
    Load a FITS data cube from file.

    This function handles multi-instrument FITS files (MUSE, MIRI, NIRSpec)
    with automatic instrument detection.

    Args:
        filename (str): Path to FITS file

    Returns:
        CubeData: Object containing cube data, wavelength, WCS, and metadata

    Raises:
        ValueError: If instrument cannot be detected or file is invalid

    Example:
        >>> cube = load_cube("Level3_ch1-long_s3d.fits")
        >>> print(cube.instrument)  # "MIRI"
        >>> print(cube.wavelength.shape)  # (1000,)
        >>> print(cube.cube.shape)  # (1000, 512, 512)
    """
    cube = CubeData()
    cube.load_from_file(filename)
    return cube


def load_spectrum(filename, x_column=None, y_column=None, error_column=None, wave_unit=None,
                  flux_unit=None, medium=None, instrument=None, grating=None):
    """
    Load one spectrum from a text file (txt or csv) or a 1D FITS table, for example a JWST x1d file.
    This is the same as the GUI's "Load Spectrum" button.

    The file has a wavelength column, a flux column and an optional flux error column. The separator can
    be a tab, a comma, a semicolon or white space. Lines that start with "#" are comments. A file that
    "Save Spectrum" wrote gives its own units, instrument and column names. Any other text file needs wave_unit.
    A FITS table gives the unit of each column. For an x1d file the flux is FLUX in Jy, unless y_column is
    SURF_BRIGHT (MJy/sr). The error column and the flux unit then follow the flux column.
    The object has no cube. cube.spectrum, cube.wavelength, cube.header and cube.spectrum_error work as
    they do for a cube, so fit_continuum and fit_line take them in the same way.

    Args:
        filename (str): Path to the text file or the FITS file
        x_column (str or int): Name or index of the wavelength column. None uses the guess from the column names.
        y_column (str or int): Name or index of the flux column. None uses the guess.
        error_column (str, int or False): Name or index of the flux error column. None uses the guess.
            False means that no column is the error.
        wave_unit (str): 'um', 'nm' or 'Angstrom'. Needed if the file does not give it. nm is converted to um.
        flux_unit (str): Flux unit, for example 'Jy'. None uses the file's unit, or leaves it unknown.
        medium (str): 'air' or 'vacuum'. None uses the file's value, or leaves it unknown.
        instrument (str): 'MIRI', 'NIRSpec' or 'MUSE'. It gives the resolving power for the multi-Gaussian fit.
        grating (str): NIRSpec grating, for example 'G395H'. Needed for the resolving power of NIRSpec.

    Returns:
        CubeData: Object with spectrum, wavelength, wave_unit, flux_unit, header and spectrum_error.
        cube is None.

    Raises:
        ValueError: If a column or unit is not valid, wave_unit is missing, or fewer than 10 usable rows are left
        OSError: If the file cannot be opened

    Example:
        >>> spec = load_spectrum("my_spectrum.csv", wave_unit="um", flux_unit="Jy")
        >>> result = fit_line(spec.wavelength, spec.spectrum, x1=6.90, x2=6.92)
    """
    from .spectrum_file import column_flux_unit, default_settings, read_spectrum_file, suggest_error_column

    table = read_spectrum_file(filename)
    settings = default_settings(table)
    settings.pop("complete")

    def column_index(column):
        """Turn a column name or index into an index.

        Args:
            column (str or int): Column name, or index counted from 0

        Returns:
            index (int): Column index

        Raises:
            ValueError: If no column has this name
        """
        if isinstance(column, str):
            names = [name.lower() for name in table.column_names]
            if column.lower() not in names:
                raise ValueError(f"No column is named {column!r}. The columns are {table.column_names}.")
            return names.index(column.lower())
        return int(column)

    if x_column is not None:
        settings["x_index"] = column_index(x_column)
    if y_column is not None:
        settings["y_index"] = column_index(y_column)
        # A FITS table gives a unit for each column, so the unit and the error column follow the flux column.
        settings["flux_unit"] = column_flux_unit(table, settings["y_index"]) or settings["flux_unit"]
        paired = suggest_error_column(table, settings["y_index"])
        if paired is not None:
            settings["error_index"] = paired
    if error_column is False:
        settings["error_index"] = None
    elif error_column is not None:
        settings["error_index"] = column_index(error_column)

    if wave_unit is not None:
        settings["wave_unit"] = wave_unit
    elif "wave_unit" not in table.metadata:
        raise ValueError("The file does not give its wavelength unit. Give wave_unit: 'um', 'nm' or 'Angstrom'.")
    if flux_unit is not None:
        settings["flux_unit"] = flux_unit
    if medium is not None:
        settings["medium"] = medium
    if instrument is not None:
        settings["instrument"] = instrument
    if grating is not None:
        settings["grating"] = grating

    cube = CubeData()
    for note in cube.load_from_table(table, settings):
        warnings.warn(note)
    return cube


# --- Fit File API ---

def load_fit(filename, wavelength=None, spectrum=None):
    """
    Load a Gaussian fit file that "Save Fit" (or the save_path of fit_line and fit_hitran_lines) wrote.
    It gives the fit back as it was, with no new fit. This is the same as the GUI's "Load Fit"
    (single line) and "Load Fits" (multi-Gaussian) buttons.

    Args:
        filename (str): Path to the fit file
        wavelength (array-like): Wavelengths of the spectrum that the fit goes on. None uses the
            wavelengths in the file. Given wavelengths must be the wavelengths of the file.
        spectrum (array-like or Quantity): Flux of the spectrum. None uses the flux in the file. A Quantity gives
            its unit, which must be the unit of the file.

    Returns:
        dict: 'fit_type' ('single' or 'multi'), 'lines' (list of dicts with 'kind' ('line', 'contaminant' or
            'skipped'), 'line' (catalog data) and 'result' (GaussianFitResult), in the order of the file),
            'markers' (the fit markers of a single fit), 'tab', 'wave_unit', 'flux_unit', 'baseline_source',
            'wavelength', 'spectrum', 'fit' and 'baseline' (arrays), and 'notes' (warnings, may be empty).

    Raises:
        ValueError: If the file has no fit parameters (a file from an older version), the wavelengths
            are not the wavelengths of the file (a Doppler shift gives this error), or the flux unit of
            the spectrum is not the flux unit of the file.
        OSError: If the file cannot be opened

    Example:
        >>> fit = load_fit("gaussian_fit.txt")
        >>> print(fit['lines'][0]['result'].center)
    """
    from .fit_file import read_fit_file, rebuild_fit

    fit_file = read_fit_file(filename)
    wl = fit_file.wavelength if wavelength is None else _as_float_array(wavelength)
    sp = fit_file.original if spectrum is None else _as_float_array(spectrum)
    unit = getattr(spectrum, "unit", None)
    rebuilt = rebuild_fit(fit_file, wl, sp, flux_unit=None if unit is None else unit.to_string())
    for note in rebuilt['notes']:
        warnings.warn(note)
    return {
        'fit_type': fit_file.fit_type, 'lines': rebuilt['entries'], 'markers': fit_file.markers,
        'tab': fit_file.tab, 'wave_unit': fit_file.wave_unit, 'flux_unit': fit_file.flux_unit,
        'baseline_source': fit_file.baseline_source, 'wavelength': wl, 'spectrum': sp,
        'fit': fit_file.fit, 'baseline': fit_file.baseline, 'notes': rebuilt['notes'],
    }


# --- Region Extraction API ---

def extract_pixel_spectrum(cube, x, y):
    """Extract the 1D spectrum at a single spatial pixel. This is the same
    lookup the GUI uses; prefer it over indexing cube.cube directly, since
    it always applies the correct (wavelength, y, x) axis order.

    Args:
        cube (CubeData): Loaded cube, as returned by load_cube()
        x (int): X pixel coordinate (0-based)
        y (int): Y pixel coordinate (0-based)

    Returns:
        astropy.units.Quantity: 1D spectrum at (x, y), same length as cube.wavelength. It is the flux
        of that pixel in Jy for a cube in MJy/sr (the cube value times the pixel solid angle), the same
        as the GUI shows, and the unit goes with the data.

    Example:
        >>> cube = load_cube("data.fits")
        >>> spectrum = extract_pixel_spectrum(cube, x=20, y=25)
    """
    native = cube.get_pixel_spectrum(x, y)
    return None if native is None else cube.with_spectrum_unit(cube.to_spectrum_units(native))


def extract_region_spectrum(cube, mask):
    """Extract the summed spectrum over an arbitrary spatial region, the same
    sum the GUI shows for region selections (each reduces to a 2D
    boolean mask). A NaN pixel adds 0 to the sum instead of giving NaN.

    Args:
        cube (CubeData): Loaded cube, as returned by load_cube()
        mask (numpy.ndarray): 2D boolean array matching the cube's spatial
            shape (cube.cube.shape[1:]); True marks pixels inside the region

    Returns:
        astropy.units.Quantity: 1D summed spectrum of the masked pixels. It is a flux in Jy for a cube in MJy/sr
        (the sum times the pixel solid angle), the same as the GUI shows, and the unit goes with the data.

    Example:
        >>> cube = load_cube("data.fits")
        >>> mask = np.zeros(cube.cube.shape[1:], dtype=bool)
        >>> mask[15:35, 10:30] = True  # a rectangular region, [y, x] order
        >>> spectrum = extract_region_spectrum(cube, mask)
    """
    native = cube.compute_region_spectrum(mask)
    return None if native is None else cube.with_spectrum_unit(cube.to_spectrum_units(native))


# --- Continuum Fitting API ---

from .flux_map_extractor import fit_auto_continuum, automatic_lines, CALLER_LINES

def fit_continuum(wavelength, flux, method='auto', **kwargs):
    """Fit continuum to a spectrum using automatic peak detection: the same
    method the GUI uses, detecting emission/absorption peaks and fitting a
    smooth continuum through the remaining points.

    Args:
        wavelength (array-like): Wavelength/frequency values
        flux (array-like): Flux values
        method (str): Fitting method - currently only 'auto' is supported

    Keyword Args:
        window_size (int): Size of median filter window (default 51, must be odd)
        sigma_threshold (float): Prominence threshold in sigma (default 2.0)
        peak_width (int): Minimum width of peaks in pixels (default 1)
        peak_distance (int): Minimum distance between peaks (default 3)
        peak_exclusion (int): Pixels to exclude on each side of peak (default 3)
        iterations (int): Iterative clipping iterations (default 2)
        smoothing_factor (float): Spline smoothing divisor (default 5). A higher value follows the
            data more closely. The smoothing follows the noise of the spectrum, so it does not
            depend on the flux unit.
        mask_ranges (list): (low, high) wavelength pairs, in the unit of wavelength (default None).
            The points in these ranges are not used to fit the continuum. The continuum crosses
            a masked range as a smooth curve between its two sides. The same key works in the
            continuum_params of save_continuum_cubes, extract_flux and create_flux_map.

    Returns:
        array or Quantity: Fitted continuum values matching input wavelength array. If flux is an
        astropy Quantity (like cube.spectrum, which is in Jy), the continuum has the same unit, so
        spectrum - continuum works. A plain array in gives a plain array out.

    Raises:
        ValueError: if a masked range is not valid, or if fewer than 10 points are outside the masked ranges

    Example:
        >>> wavelength = cube.wavelength
        >>> spectrum = cube.spectrum
        >>> continuum = fit_continuum(wavelength, spectrum, sigma_threshold=2.0)
        >>> line_only = spectrum - continuum
    """
    if method != 'auto':
        raise ValueError(f"Unknown continuum fitting method: {method}. Use 'auto'.")
    continuum = fit_auto_continuum(
        _as_float_array(wavelength), _as_float_array(flux), **kwargs
    )
    unit = getattr(flux, "unit", None)
    if unit is None or unit == u.dimensionless_unscaled:
        return continuum
    return continuum * unit


from .flux_map_extractor import (
    mask_file_lines as _mask_file_lines, read_mask_file as _read_mask_file,
    move_masks_to_axis as _move_masks_to_axis, merge_mask_ranges as _merge_mask_ranges,
)

def save_continuum_masks(filename, mask_ranges, wave_unit=None, shift_factor=1.0):
    """Save masked wavelength ranges of the automatic continuum to a tab-separated text file.
    This is the same file as the GUI's Save Masks button writes. The file gives the wavelength unit
    and the Doppler shift factor, so load_continuum_masks can put the ranges on the right axis.

    Args:
        filename (str): Path of the text file to write
        mask_ranges (list): (low, high) wavelength pairs, in wave_unit
        wave_unit (str, optional): Wavelength unit of the pairs, for example 'um' or 'Angstrom'.
            None writes "Unknown", and a load then cannot convert units.
        shift_factor (float, optional): Doppler shift factor of the wavelength axis. It is 1.0 if the
            axis is not shifted. A Doppler shift of v km/s in the GUI gives a factor of 1 / (1 + v/c).

    Returns:
        None

    Raises:
        ValueError: if a masked range is not valid
        OSError: if the file cannot be written

    Example:
        >>> save_continuum_masks("masks.txt", [(6.90, 6.93), (7.10, 7.15)], wave_unit="um")
    """
    lines = _mask_file_lines(mask_ranges, wave_unit, float(shift_factor))
    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")


def load_continuum_masks(filename, wave_unit=None, shift_factor=1.0):
    """Load masked wavelength ranges from a text file that save_continuum_masks or the GUI wrote.
    The ranges are moved onto the axis you give. If the file was saved in another unit, or with
    another Doppler shift, the ranges stay on the same part of the spectrum.

    Args:
        filename (str): Path of the masks file
        wave_unit (str, optional): Wavelength unit of the spectrum that you use the ranges on, for
            example 'um'. None keeps the unit of the file.
        shift_factor (float, optional): Doppler shift factor of the wavelength axis of that spectrum.
            It is 1.0 if the axis is not shifted. A file that gives no shift factor is not moved.

    Returns:
        list: Sorted (low, high) tuples of float on your axis, ready for mask_ranges in fit_continuum.

    Raises:
        OSError: if the file cannot be opened
        ValueError: if the file has no range or a bad line, or if its unit is not the same kind of
            unit as wave_unit

    Example:
        >>> masks = load_continuum_masks("masks.txt", wave_unit="um")
        >>> continuum = fit_continuum(cube.wavelength, cube.spectrum, mask_ranges=masks)
    """
    ranges, file_unit, file_shift = _read_mask_file(filename)
    moved = _move_masks_to_axis(ranges, file_unit, file_shift,
                                file_unit if wave_unit is None else wave_unit, float(shift_factor))
    return _merge_mask_ranges(moved)


from .continuum_cube import export_continuum_cubes as _export_continuum_cubes_internal

def save_continuum_cubes(cube_data, continuum_path, contsub_path, continuum_params=None,
                         anchor_wavelengths=None, n_jobs=None):
    """Fit the continuum at every pixel, then save two cubes: the continuum
    and the continuum-subtracted (line-only) cube. The same as the GUI's
    Save Continuum Cubes button. The two cubes always sum to the original.

    Two methods: automatic (the default), or anchor_wavelengths, the same
    wavelengths you would click in the GUI's Custom Continuum tool. With
    anchors, each pixel gets a cubic spline through its own flux at the
    anchors (median of +/- 7 instrumental sigma, clipped at 3 sigma), held
    flat beyond the first and last anchor. Scripts that call this need an
    ``if __name__ == "__main__":`` guard on Windows/macOS.

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        continuum_path (str): Output FITS path of the continuum cube
        contsub_path (str): Output FITS path of the continuum-subtracted cube
        continuum_params (dict, optional): Override the fit_continuum() keyword
            arguments (window_size, sigma_threshold, peak_width, peak_distance,
            peak_exclusion, iterations, smoothing_factor). The same values
            apply to every pixel. If None, uses the defaults. Not allowed
            together with anchor_wavelengths.
        anchor_wavelengths (array-like, optional): At least 3 wavelengths, in the
            units of cube_data.wavelength. If given, uses the anchor method
            instead of the automatic one.
        n_jobs (int, optional): Number of worker processes (default: auto-detect
            CPUs, capped by available memory)

    Returns:
        int: Number of pixels that have data but got no continuum (automatic: too
            few valid points; anchors: fewer than 3 valid anchors). They are
            NaN in both cubes.

    Example:
        >>> cube = load_cube("data.fits")
        >>> n_failed = save_continuum_cubes(cube, "cube_continuum.fits", "cube_contsub.fits")
        >>> n_failed = save_continuum_cubes(
        ...     cube, "cube_continuum.fits", "cube_contsub.fits",
        ...     anchor_wavelengths=[6.55, 6.70, 6.95, 7.20, 7.35])
    """
    return _export_continuum_cubes_internal(
        cube_data, continuum_path, contsub_path,
        continuum_params=continuum_params, anchor_wavelengths=anchor_wavelengths, n_jobs=n_jobs,
    )


# --- Gaussian Fitting API ---

from .gaussian_fitter import (
    DEFAULT_DETECTION_SIGMA,
    fit_gaussian_to_line,
    calculate_equivalent_width,
    fit_gaussians_per_pixel as _fit_gaussians_per_pixel_internal,
    write_residual_cube_fits as _write_residual_cube_fits,
    save_gaussian_fit as _save_gaussian_fit_internal,
)
from .flux_map_extractor import (
    fit_hitran_lines_collective as _fit_hitran_lines_internal,
    save_gaussian_parameters as _save_gaussian_parameters_internal,
)
from .line_catalog import fetch_hitran_data

def fit_line(wavelength, spectrum, x1, x2, line_name=None, peak_x=None,
             header=None, wave_unit=None, save_path=None, parameters_path=None,
             continuum=None):
    """Fit a Gaussian to an emission line in a spectrum. The fitter needs a
    starting guess for the line center; here it defaults to the brightest
    sample between x1 and x2, or pass peak_x explicitly.

    Args:
        wavelength (array-like): Wavelength/frequency values. Astropy
            Quantities are accepted; their units are stripped.
        spectrum (array-like or Quantity): Flux values. A Quantity (like ``cube.spectrum``,
            which is in Jy for a cube in MJy/sr) gives its own unit, and that unit is used
            instead of the ``BUNIT`` of ``header``.
        x1 (float): Left boundary of the fitting region.
        x2 (float): Right boundary of the fitting region.
        line_name (str, optional): Descriptive label for the line. Not stored
            on the result; it exists for readability at the call site.
        peak_x (float, optional): Initial guess for the line center. Defaults
            to the brightest wavelength between ``x1`` and ``x2``.
        header (optional): FITS header used for unit conversion. Pass
            ``cube.header`` to match what the GUI produces.
        wave_unit (str, optional): Spectral axis unit, e.g. ``cube.wave_unit``.
        save_path (str, optional): If given, write a tab-separated text file with
            the columns wavelength, original spectrum, fit and original minus fit (the GUI's "Save Fit"
            file). The fit is NaN outside ``x1`` to ``x2``. Nothing is written, and a
            warning is raised, if the fit failed.
        parameters_path (str, optional): If given, write a tab-separated text file
            with the fitted parameters (the GUI's "Save Parameters" file). The same
            rule applies if the fit failed.
        continuum (array-like, optional): Pre-fitted continuum, same length as
            ``wavelength``. When given, it is the baseline of the fit, as in the GUI
            when a continuum is fitted for the spectrum. If None (default), the
            baseline is the straight line between the flux values at ``x1`` and ``x2``.
            The fit fails if the continuum has no value where the flux has data.

    Returns:
        GaussianFitResult: Object with ``amplitude``, ``center``, ``sigma``,
        ``fwhm``, ``area``, ``area_unit``, ``area_native``, ``baseline_slope``,
        ``baseline_intercept``, ``r_squared``, ``success`` and ``message``.

        ``area`` is in erg/s/cm^2 when the spectrum is in Jy (``header`` is not needed), or
        when it is in MJy/sr and ``header`` is supplied (MIRI, NIRSpec); otherwise it
        stays in flux*wavelength units.
        Check ``area_unit`` rather than assuming; it always states which.
        ``area_native`` is the unconverted value, which is what you want when
        dividing by a continuum measured in the cube's own units.

    Raises:
        ValueError: If no wavelength samples lie between ``x1`` and ``x2``, or
            if every flux value in that window is NaN.

    Example:
        >>> result = fit_line(
        ...     wavelength=cube.wavelength,
        ...     spectrum=cube.spectrum,
        ...     x1=6.90,
        ...     x2=6.95,
        ...     line_name="H2 S(5)"
        ... )
        >>> if result.success:
        ...     print(f"Line center: {result.center} um")
        ...     print(f"FWHM: {result.fwhm:.6f} um")
    """
    wl = _as_float_array(wavelength)
    fl, header = _spectrum_and_header(spectrum, header)

    # Boundaries are not reordered for you -- the GUI rejects a mis-ordered
    # selection, and this must behave the same way.
    if x1 >= x2:
        raise ValueError(
            f"x1 ({x1}) must be less than x2 ({x2})."
        )

    if peak_x is None:
        in_window = (wl >= x1) & (wl <= x2)
        if not np.any(in_window):
            raise ValueError(
                f"No wavelength samples between {x1} and {x2}. "
                f"Spectrum covers {wl.min():.4f} to {wl.max():.4f}."
            )
        window_flux = fl[in_window]
        if np.all(np.isnan(window_flux)):
            raise ValueError(
                f"All flux values between {x1} and {x2} are NaN."
            )
        peak_x = float(wl[in_window][np.nanargmax(window_flux)])

    result = fit_gaussian_to_line(x1, x2, wl, fl, peak_x=peak_x,
                                  header=header, wave_unit=wave_unit,
                                  continuum=None if continuum is None else _as_float_array(continuum))

    if save_path is not None or parameters_path is not None:
        if not result.success:
            warnings.warn(f"The fit failed ({result.message}), so no file was written")
        else:
            flux_unit = header.get('BUNIT') if header is not None else None
            baseline_text = ("fitted continuum" if continuum is not None
                             else "straight line between the window edges")
            header_lines = [f"Fit window: {x1:.6f} - {x2:.6f}", f"Baseline: {baseline_text}"]
            if continuum is not None:
                # Only the array is given here, so the settings that made it are not known.
                header_lines += CALLER_LINES
            if save_path is not None:
                _save_gaussian_fit_internal(save_path, wl, fl, [result], wave_unit=wave_unit,
                                            flux_unit=flux_unit, header_lines=header_lines, fit_type='single',
                                            markers={'fit_left': x1, 'peak': peak_x, 'fit_right': x2})
                print(f"[OK] Saved the fit to {save_path}")
            if parameters_path is not None:
                _save_gaussian_parameters_internal(
                    parameters_path, [{'kind': 'line', 'line': {}, 'result': result}],
                    wave_unit=wave_unit, flux_unit=flux_unit, header_lines=header_lines)
                print(f"[OK] Saved the parameters to {parameters_path}")
    return result


def fit_hitran_lines(wavelength, spectrum, molecule, wavelength_min=None, wavelength_max=None,
                     isotopologue=1, refresh_cache=False, v_upper=None, v_lower=None,
                     header=None, wave_unit=None, continuum=None, contam_catalog=None,
                     save_path=None, parameters_path=None):
    """Fit HITRAN spectral lines automatically using database lookup. Same
    algorithm as the GUI's Multi-Gaussian Fit: every requested line and any
    contaminant sharing its chunk fit simultaneously, so they don't double-claim flux.

    Args:
        wavelength (array-like): Wavelength/frequency values
        spectrum (array-like): Flux values
        molecule (str): Molecule name (e.g., 'CO', 'H2O', 'CH4', '[Fe II]')
        wavelength_min (float, optional): Minimum wavelength to search. If None, uses spectrum min.
        wavelength_max (float, optional): Maximum wavelength to search. If None, uses spectrum max.
        isotopologue (int): Isotopologue number (default 1 for main isotope)
        refresh_cache (bool): Force refresh HITRAN cache (default False)
        v_upper (int, optional): Filter by upper vibrational level (e.g., 1 for v=1)
        v_lower (int, optional): Filter by lower vibrational level (e.g., 0 for v=0)
        header: FITS header. Required whenever any line is actually found -- each
            line's fit width comes from the instrument's real spectral resolution at
            that wavelength (MIRI/NIRSpec only), not a guess, so a missing header
            raises ValueError rather than silently fitting with the wrong width.
            Also used for the erg/s/cm2 conversion.
        wave_unit: Wavelength unit string, needed alongside header for the conversion.
        continuum (array-like, optional): Pre-fitted continuum, same length as
            wavelength, subtracted before fitting each line. If None, one is fitted
            automatically via ``fit_continuum()`` -- the same fallback the GUI uses
            when you have not run Automatic Continuum interactively first.
        contam_catalog (pd.DataFrame, optional): Non-target lines (a 'line_wave_um'
            column, same format as ``load_contaminating_lines_catalog()``) to give their
            own component in the fit so they don't distort a nearby requested line's
            amplitude -- the same optional catalog ``extract_flux``/Flux Map
            already supports. None (default) skips this.
        save_path (str, optional): If given, write a tab-separated text file with
            the columns wavelength, original spectrum, fit and original minus fit (the GUI's "Save Fit"
            file). The fit includes the contaminant lines and is NaN where no line
            was fitted. Nothing is written, and a warning is raised, if no line fit
            succeeded.
        parameters_path (str, optional): If given, write a tab-separated text file
            with one row for each line: the fitted parameters, the catalog data, and
            a flag for an unreliable area error (the GUI's "Save Parameters" file).
            Lines that could not be fit are listed as ``skipped`` with the reason.
            The same rule applies if no line fit succeeded.

    Returns:
        list: List of dicts with fit results for each line:
            - wavelength: Fitted line center wavelength
            - species: Molecule name
            - transition: Transition description
            - amplitude: Peak height above the local continuum
            - sigma: Gaussian sigma width
            - fwhm: Full width at half maximum
            - area: Integrated line flux, in the units given by ``area_unit``
            - area_native: Integrated line flux before any CGS conversion
            - area_unit: Units of ``area`` -- ``'erg/s/cm^2'`` only when header/wave_unit
              let it convert; always check this rather than assuming
            - r_squared: Fit quality
            - success: Whether fit succeeded
            - message: Status message

    Raises:
        RuntimeError: if the HITRAN fetch fails, or HITRAN does not know the molecule

    Example:
        >>> # Fit CO v=1->0 lines only
        >>> results = fit_hitran_lines(
        ...     wavelength=cube.wavelength,
        ...     spectrum=spectrum,
        ...     molecule='CO',
        ...     wavelength_min=4.7,
        ...     wavelength_max=4.8,
        ...     v_upper=1,
        ...     v_lower=0,
        ...     header=cube.header,
        ...     wave_unit=cube.wave_unit,
        ... )
        >>> detected = [r for r in results if r['success']]
        >>> print(f"Detected {len(detected)} CO v=1->0 lines")
    """
    wl = _as_float_array(wavelength)
    sp, header = _spectrum_and_header(spectrum, header)

    # Use spectrum edges if not provided
    if wavelength_min is None:
        wavelength_min = float(np.nanmin(wl))
    if wavelength_max is None:
        wavelength_max = float(np.nanmax(wl))

    # Fetch HITRAN data for this molecule
    hitran_lines, fetch_meta = fetch_hitran_data(
        wavelength_min, wavelength_max,
        molecule, isotopologue,
        refresh_cache=refresh_cache,
        v_upper=v_upper,
        v_lower=v_lower
    )

    if not hitran_lines and fetch_meta.get('message'):
        raise RuntimeError(f"Could not get {molecule} lines from HITRAN: {fetch_meta['message']}")
    if not hitran_lines:
        warnings.warn(
            f"No HITRAN lines found for {molecule} in range {wavelength_min:.6f}-{wavelength_max:.6f} um")
        return []

    if continuum is None:
        continuum_arr = fit_continuum(wl, sp)
        continuum_lines = automatic_lines(default=True)
    else:
        continuum_arr = _as_float_array(continuum)
        continuum_lines = list(CALLER_LINES)

    # Contaminant fits (if given) aren't surfaced; they only keep requested
    # lines' amplitudes honest. Center/sigma are free within a bound, not fixed.
    start_time = time.time()
    fit_pairs, contam_pairs, _ = _fit_hitran_lines_internal(
        wl, sp, hitran_lines, continuum_arr,
        header=header, wave_unit=wave_unit,
        contam_lines=contam_catalog)

    n_fitted = sum(1 for _line, result in fit_pairs if result.success)
    print(f"[OK] Fitted {n_fitted} of {len(fit_pairs)} {molecule} lines between "
          f"{wavelength_min:.6f} and {wavelength_max:.6f} um in {time.time() - start_time:.1f} s")

    if save_path is not None or parameters_path is not None:
        if n_fitted == 0:
            warnings.warn(f"No {molecule} line fit succeeded, so no file was written")
        else:
            flux_unit = header.get('BUNIT') if header is not None else None
            if save_path is not None:
                _save_gaussian_fit_internal(
                    save_path, wl, sp, [], wave_unit=wave_unit, flux_unit=flux_unit, fit_type='multi',
                    header_lines=continuum_lines,
                    entries=([{'kind': 'line', 'line': line, 'result': result}
                              for line, result in fit_pairs if result.success]
                             + [{'kind': 'contaminant', 'line': line, 'result': result}
                                for line, result in contam_pairs]
                             + [{'kind': 'skipped', 'line': line, 'result': result}
                                for line, result in fit_pairs if not result.success]))
                print(f"[OK] Saved the fit to {save_path}")
            if parameters_path is not None:
                entries = (
                    [{'kind': 'line', 'line': line, 'result': result}
                     for line, result in fit_pairs if result.success]
                    + [{'kind': 'contaminant', 'line': line, 'result': result}
                       for line, result in contam_pairs]
                )
                entries.sort(key=lambda entry: entry['result'].center)
                entries += [{'kind': 'skipped', 'line': line, 'result': result}
                            for line, result in fit_pairs if not result.success]
                _save_gaussian_parameters_internal(
                    parameters_path, entries, wave_unit=wave_unit, flux_unit=flux_unit,
                    header_lines=continuum_lines)
                print(f"[OK] Saved the parameters to {parameters_path}")

    return [
        {
            'wavelength': result.center if result.success else (line.get('lambda_micron') or line.get('wavelength')),
            'species': line.get('species', 'Unknown'),
            'transition': line.get('transition', 'N/A'),
            'amplitude': result.amplitude,
            'sigma': result.sigma,
            'fwhm': result.fwhm,
            'area': result.area,
            'area_native': result.area_native,
            'area_unit': result.area_unit,
            'r_squared': result.r_squared,
            'success': result.success,
            'message': result.message,
        }
        for line, result in fit_pairs
    ]


def measure_equivalent_width(fit_result, wavelength=None, continuum_spectrum=None):
    """Calculate equivalent width (the width of an imaginary rectangular
    line with the same area) from a Gaussian fit result. A full continuum
    array is averaged over the wavelengths actually fitted.

    Args:
        fit_result (GaussianFitResult): Result from :func:`fit_line`.
        wavelength (array-like, optional): Wavelength array matching
            ``continuum_spectrum``. Required when passing a continuum array.
        continuum_spectrum (array-like or float, optional): Continuum-only
            spectrum, or a single continuum level. If omitted, the fit's own
            baseline is used.

    Returns:
        float: Equivalent width in the same units as ``wavelength``. Returns
        0.0 if the fit failed or the continuum level is non-positive.

    Raises:
        ValueError: If a continuum array is given without a wavelength array.

    Example:
        >>> continuum = fit_continuum(cube.wavelength, cube.spectrum)
        >>> ew = measure_equivalent_width(result, cube.wavelength, continuum)
        >>> print(f"Equivalent width: {ew:.6f} um")
    """
    if continuum_spectrum is None:
        # Fall back to the baseline the fit already determined.
        return calculate_equivalent_width(fit_result)

    continuum = _as_float_array(continuum_spectrum)

    if continuum.ndim == 0:
        return calculate_equivalent_width(fit_result, float(continuum))

    if wavelength is None:
        raise ValueError(
            "wavelength is required when continuum_spectrum is an array, "
            "so the continuum can be averaged over the fitted region."
        )

    wl = _as_float_array(wavelength)

    # Average the continuum across the wavelengths that were actually fitted.
    lo = float(np.nanmin(fit_result.x_data))
    hi = float(np.nanmax(fit_result.x_data))
    in_fit = (wl >= lo) & (wl <= hi)
    level = float(np.nanmean(continuum[in_fit] if np.any(in_fit) else continuum))

    return calculate_equivalent_width(fit_result, level)


def create_gaussian_map(cube_data, lam_min=None, lam_max=None, n_jobs=None, **kwargs):
    """
    Fit a Gaussian to every spatial pixel's spectrum in a wavelength window.

    This is the same full-cube fitting used by the GUI's Gaussian mapping
    dialog. Iterative sigma-clipping outlier rejection is always enabled.

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        lam_min (float, optional): Minimum wavelength for the fit window (default: full range)
        lam_max (float, optional): Maximum wavelength for the fit window (default: full range)
        n_jobs (int, optional): Number of worker processes (default: auto-detect CPUs)
        **kwargs: Passed through to fit_gaussians_per_pixel (e.g. fac_above_stdres,
            noise_min, noise_max, lambda_0, progress_callback)

    Returns:
        dict: 2D parameter maps - 'amplitude', 'continuum', 'center', 'sigma', 'fwhm', 'area',
            'r_squared', 'success', 'is_detection', 'is_upper_limit'. 'continuum' is the baseline of
            each pixel at 'continuum_wavelength', in the flux unit of the cube

    Example:
        >>> cube = load_cube("data.fits")
        >>> maps = create_gaussian_map(cube, lam_min=4.65, lam_max=4.70)
        >>> print(f"Detections: {maps['is_detection'].sum()}")
    """
    wl = cube_data.wavelength
    if hasattr(wl, 'value'):
        wl = wl.value
    return _fit_gaussians_per_pixel_internal(
        cube=cube_data.cube,
        wavelength=wl,
        lam_min=lam_min,
        lam_max=lam_max,
        n_jobs=n_jobs,
        header=cube_data.header,
        wave_unit=cube_data.wave_unit,
        flux_unit=cube_data.header.get('BUNIT') if cube_data.header else None,
        **kwargs
    )


from .line_catalog import (
    collect_rest_line_candidates as _collect_rest_line_candidates,
    load_line_list as _load_line_list,
)
from .unit_converter import (
    spectral_axis_medium as _spectral_axis_medium,
    vacuum_um_to_cube_wavelength as _vacuum_um_to_cube_wavelength,
    wavelength_to_doppler_velocity as _wavelength_to_doppler_velocity,
    remove_systemic_velocity as _remove_systemic_velocity,
)


def find_rest_lines(molecule, lam_min, lam_max, line_list_path=None):
    """
    List the lines of one molecule that can set zero velocity, in a wavelength window.
    It reads HITRAN and the LST catalog, and every line stays in the list.
    Pass the 'wavelength' of the line you pick to create_velocity_map().

    Args:
        molecule (str): Molecule or atom name, for example 'CO', 'H2' or 'He I'
        lam_min (float): Lower end of the window, in um (vacuum)
        lam_max (float): Upper end of the window, in um (vacuum)
        line_list_path (str, optional): CSV file, HITRAN-format .par file (for example
            HITEMP), or LST .lst file with more lines of this molecule. Its lines have a
            'source' that starts with 'File'.

    Returns:
        list: Line dicts sorted by wavelength. 'wavelength' is in um (vacuum), and 'source'
            is 'HITRAN', 'LST' or 'File: <name>'.

    Raises:
        ValueError: if the window is not 0 < lam_min < lam_max, or line_list_path is not
            a readable line list of this molecule
    """
    rows, notes = _collect_rest_line_candidates(molecule, lam_min, lam_max)
    for note in notes:
        print(f"[WARN] {note}")
    if line_list_path is not None:
        rows = rows + _load_line_list(line_list_path, lam_min, lam_max, molecule)
    return sorted(rows, key=lambda row: row['wavelength'])


def create_velocity_map(cube_data, maps, rest_wavelength, systemic_velocity=0.0):
    """
    Make a Doppler velocity map from Gaussian map line centers and a rest wavelength.
    Zero velocity is the rest wavelength, in um (vacuum), as find_rest_lines() gives it.
    The function converts it to the air or vacuum wavelengths that the cube uses.

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        maps (dict): Result of create_gaussian_map()
        rest_wavelength (float): Rest wavelength of the line, in um (vacuum)
        systemic_velocity (float, optional): Systemic velocity in km/s. The map is relative to it.
            This is the "Systemic velocity" box of the Velocity map in the GUI. Only this map changes.

    Returns:
        numpy.ndarray: Velocity map in km/s. A pixel without a fitted center is NaN.

    Raises:
        ValueError: if the cube header does not say if wavelengths are in air or in vacuum,
            the spectral axis is not a wavelength, or systemic_velocity is not a finite number
            greater than -c
    """
    medium = _spectral_axis_medium(cube_data.header)
    if medium is None:
        raise ValueError("The cube header does not say if its wavelengths are in air or in vacuum "
                         "(CTYPE3 must be WAVE or AWAV), so a rest wavelength cannot be compared.")
    rest = float(_vacuum_um_to_cube_wavelength(rest_wavelength, cube_data.wave_unit, medium))
    velocity = _wavelength_to_doppler_velocity(maps['center'], rest, cube_data.wave_unit)
    return _remove_systemic_velocity(velocity, systemic_velocity)


def export_residual_cube(cube_data, maps, out_path="residual_cube.fits"):
    """
    Save the residual cube of a Gaussian map to a FITS file. This is the "Residuals" export of the
    Gaussian map window. The residuals are the ones that the map kept when it fitted each pixel:
    (data - fit) / noise, inside the fit window. Nothing is fitted again.

    Args:
        cube_data (CubeData): Loaded cube that the map was fitted on. It gives the header.
        maps (dict): Result of create_gaussian_map(). It must have 'residuals_cube'.
        out_path (str): Output FITS filename (default "residual_cube.fits")

    Returns:
        None: The FITS file is written to out_path. To use the residuals as an array, read
        maps['residuals_cube'].

    Raises:
        ValueError: if maps has no 'residuals_cube', for example a map that was cancelled

    Example:
        >>> cube = load_cube("data.fits")
        >>> maps = create_gaussian_map(cube, lam_min=4.65, lam_max=4.70)
        >>> export_residual_cube(cube, maps, out_path="residual.fits")
    """
    residuals = maps.get('residuals_cube') if maps is not None else None
    if residuals is None:
        raise ValueError("maps has no 'residuals_cube'. Pass the result of create_gaussian_map().")
    _write_residual_cube_fits(residuals, out_path=out_path, header=cube_data.header,
                              instrument_hint=cube_data.instrument)


# --- CO Flux Extraction API ---

import pandas as pd
from .flux_map_extractor import (
    extract_flux_from_spectrum as _extract_flux_internal,
    create_flux_map as _create_flux_map_internal,
)
from .line_catalog import load_CO_line_catalog, load_contaminating_lines_catalog

def extract_flux(wavelength, spectrum, target_catalog_path, contamination_catalog_path=None,
                     continuum_params=None,
                     header=None, wave_unit=None):
    """Extract total flux by fitting all target lines in a spectrum
    simultaneously with contaminating lines and continuum; the same
    algorithm as the GUI's Flux Map button, run on a single spectrum.

    Args:
        wavelength (array-like): Wavelength values in same units as catalog
        spectrum (array-like): Flux values
        target_catalog_path (str): Path to target line catalog (ECSV format)
        contamination_catalog_path (str, optional): Path to contaminant lines catalog
        continuum_params (dict, optional): Override the default continuum-fitting
            parameters (window_size, sigma_threshold, peak_width, peak_distance,
            peak_exclusion, iterations, smoothing_factor). If None, uses the defaults,
            which work well for typical spectra.
        header: FITS header. Required whenever any line is actually detected -- each
            line's fit width comes from the instrument's real spectral resolution at
            that wavelength (MIRI/NIRSpec only), not a guess, so a missing header
            raises ValueError as soon as a line is found rather than silently fitting
            with the wrong width. Also used to convert 'flux'/'contam_flux' to
            erg/s/cm2 -- 'flux_unit' in the result always tells you which you got.
        wave_unit (str, optional): Wavelength unit (e.g. 'um'). Used alongside
            header for the CGS conversion.

    Returns:
        dict: Results containing:
            - 'flux': Total flux, in erg/s/cm2 if header/wave_unit were given
              and the conversion is supported for this instrument, else native units
            - 'contam_flux': Total flux attributed to contaminating lines, same units
            - 'flux_unit': unit of 'flux'/'contam_flux'
            - 'mean_continuum': Mean continuum level used for the fit (native units)
            - 'detected_lines': DataFrame of individual detected line fits
            - 'residuals': DataFrame of fit residuals
            - 'continuum_curve': DataFrame of the fitted continuum vs wavelength

    Example:
        >>> result = extract_flux(
        ...     wavelength=cube.wavelength,
        ...     spectrum=cube.spectrum,
        ...     target_catalog_path="CO/CO_comprehensive_linelist_with_g.ecsv",
        ...     header=cube.header,
        ...     wave_unit=cube.wave_unit,
        ... )
        >>> print(f"Total flux: {result['flux']:.4e} {result['flux_unit']}")
        >>> print(f"Detected {len(result['detected_lines'])} lines")
    """
    # A spectrum with a unit (like cube.spectrum) gives its own unit to the conversion
    spectrum, header = _spectrum_and_header(spectrum, header)

    # Load catalogs from files
    target_catalog = load_CO_line_catalog(target_catalog_path)
    contam_catalog = None
    if contamination_catalog_path:
        contam_catalog = load_contaminating_lines_catalog(contamination_catalog_path)

    # Call internal function
    result = _extract_flux_internal(
        wavelength, spectrum,
        target_catalog, contam_catalog,
        do_duplicate_removal=True,
        continuum_params=continuum_params,
        header=header, wave_unit=wave_unit,
    )

    lines = result['detected_lines']
    n_detected = int((lines['disposition'] == 1).sum()) if len(lines) else 0
    print(f"[OK] Extracted flux {result['flux']:.4e} {result['flux_unit']} from {n_detected} detected line(s), "
          f"contaminating flux {result['contam_flux']:.4e} {result['flux_unit']}")
    return result


def create_flux_map(cube_data, target_catalog_path=None, contamination_catalog_path=None, n_jobs=8,
                        wavelength_min=None, wavelength_max=None, continuum_params=None):
    """Create a 2D map of target-line flux across the entire data cube:
    HITRAN CO by default, or a custom target_catalog_path. Same algorithm
    as the GUI's Flux Map button (continuum subtraction, then per-pixel fit).

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        target_catalog_path (str, optional): Path to target line catalog (ECSV format).
                                         If None, automatically fetches CO from HITRAN.
        contamination_catalog_path (str, optional): Path to contaminant lines
        n_jobs (int): Number of parallel jobs (default 8)
        wavelength_min (float, optional): Min wavelength to fit, and for the HITRAN
            search when target_catalog_path is None (auto-detect if None)
        wavelength_max (float, optional): Max wavelength to fit, and for the HITRAN
            search when target_catalog_path is None (auto-detect if None)
        continuum_params (dict, optional): Override the default continuum-fitting
            parameters, applied identically to every pixel. If None, uses the defaults,
            which work well for typical spectra.

    Returns:
        dict: Map data containing:
            - 'flux_map': 2D array of flux per pixel, in erg/s/cm2 if cube_data.header
              supports the conversion (see 'flux_unit'), else the cube's native flux unit.
              A pixel with no detected line is NaN
            - 'contam_flux_map': 2D array of contaminating-line flux per pixel, same units
              as 'flux_map'. A pixel with no detected contaminating line is NaN
            - 'detection_map': 2D array with number of detected target lines per pixel
            - 'contam_detection_map': 2D array with number of detected contaminating lines
              per pixel
            - 'wavelength': 1D wavelength array used for fitting
            - 'header': FITS header from the original cube
            - 'wave_range': (wavelength_min, wavelength_max) actually used for the fit
            - 'flux_unit': unit of 'flux_map'/'contam_flux_map'

    Raises:
        RuntimeError: if the CO line list for the cube's wavelength range cannot be
            fetched from HITRAN
        ValueError: if cube_data has no cube loaded

    Example:
        >>> cube = load_cube("data.fits")
        >>> flux_map_result = create_flux_map(cube, n_jobs=4)
        >>> print(f"Map shape: {flux_map_result['flux_map'].shape}, unit: {flux_map_result['flux_unit']}")
    """
    # If no catalog provided, fetch CO from HITRAN automatically (the default target species --
    # pass target_catalog_path for any other species)
    if target_catalog_path is None:
        print("  Fetching CO data from HITRAN...")

        # Extract wavelength range, handling Quantity objects
        wl = cube_data.wavelength
        if hasattr(wl, 'value'):
            wl = wl.value

        wavelength_min = float(np.nanmin(wl)) if wavelength_min is None else wavelength_min
        wavelength_max = float(np.nanmax(wl)) if wavelength_max is None else wavelength_max

        hitran_lines, fetch_meta = fetch_hitran_data(
            wavelength_min, wavelength_max,
            'CO', isotopologue=1,
            refresh_cache=False
        )
        if not hitran_lines:
            reason = fetch_meta.get('message') or 'HITRAN has no CO line in this range'
            raise RuntimeError(
                f"Could not get CO lines from HITRAN for {wavelength_min:.6f}-{wavelength_max:.6f} um: {reason}")
        # flux_map_extractor expects a DataFrame with a 'line_wave_um' column;
        # fetch_hitran_data returns a list of dicts with 'wavelength' instead
        target_catalog = pd.DataFrame(hitran_lines).rename(columns={'wavelength': 'line_wave_um'})
        contam_catalog = None
    else:
        # Load catalogs from files (original behavior)
        target_catalog = load_CO_line_catalog(target_catalog_path)
        contam_catalog = None
        if contamination_catalog_path:
            contam_catalog = load_contaminating_lines_catalog(contamination_catalog_path)

    return _create_flux_map_internal(
        cube_data, target_catalog, contam_catalog,
        n_jobs=n_jobs,
        continuum_params=continuum_params,
        wave_min=wavelength_min, wave_max=wavelength_max,
    )


# --- Rotational Diagram API ---

from .rotational_diagram_h2_fitter import rotational_diagram as _rotational_diagram_h2
from .rotational_diagram_general_fitter import rotational_diagram as _rotational_diagram_general


def rotational_diagram(fluxes, flux_errors, molecular_data=None, OPR=None, fit_opr=False, opr_grid=None,
                        fit1T=True, fit2T=False, R=None, dist=140, upper_limits=None,
                        progress_callback=None, molecule='H2', isotopologue=1):
    """
    Fit a rotational diagram (1T and/or 2T) to a set of line fluxes. Dispatches to the
    H2-specific fitter (hardcoded partition function, OPR support) or the general
    HITRAN-based fitter (any molecule, e.g. CO) depending on `molecule`.
    R is optional. Without R, N is a total molecule count instead of a column density.

    Args:
        fluxes (dict): {J: flux_value}, J is the upper rotational level
        flux_errors (dict): {J: flux_error}
        molecular_data (dict): 'wavelengths', 'Aul', 'Eu', 'gu' keys, each {J: value}
        OPR (float): Fixed ortho-para ratio (H2 only; ignored otherwise)
        fit_opr (bool): Grid-search OPR to minimize chi-squared (H2 only; ignored otherwise)
        opr_grid (array): OPR values to search (H2 only). Defaults to np.linspace(0.1, 5.0, 200)
        fit1T (bool): Report the 1T fit (always computed regardless, as 2T's own seed)
        fit2T (bool): Perform the 2T fit
        R (float): Emitting radius (AU), or None
        dist (float): Distance (pc)
        upper_limits (dict): {J: True} for measurements marked as upper limits
        progress_callback (callable): (phase, current, total) -> None, for a GUI progress bar
        molecule (str): Species being fit, per get_available_molecules()
        isotopologue (int): HITRAN isotopologue number for `molecule` (ignored for H2)

    Returns:
        results (dict): Fit results. "OPR"/"OPR_err"/"OPR_err_pos"/"OPR_err_neg" keys
            are present only when molecule == 'H2'

    Raises:
        ValueError: if neither fit1T nor fit2T is True
    """
    if molecule == 'H2':
        return _rotational_diagram_h2(
            fluxes, flux_errors, molecular_data=molecular_data, OPR=OPR, fit_opr=fit_opr,
            opr_grid=opr_grid, fit1T=fit1T, fit2T=fit2T, R=R, dist=dist,
            upper_limits=upper_limits, progress_callback=progress_callback,
        )
    return _rotational_diagram_general(
        fluxes, flux_errors, molecular_data=molecular_data, fit1T=fit1T, fit2T=fit2T,
        R=R, dist=dist, upper_limits=upper_limits, progress_callback=progress_callback,
        molecule=molecule, isotopologue=isotopologue,
    )


# --- Rotational Diagram Map API ---
# Per-pixel maps, one cube or several at different pixel scales (rotational_diagram_map.py).

from .rotational_diagram_map import (
    fit_pixel as _rdm_fit_pixel,
    fit_region as _rdm_fit_region,
    create_rotational_diagram_map as _rdm_create_map,
    fit_region_multi as _rdm_fit_region_multi,
    create_rotational_diagram_map_multi as _rdm_create_map_multi,
    fit_pixel_multi as _rdm_fit_pixel_multi,
    save_rotational_diagram_map_fits as _rdm_save_fits,
    STATUS_OK, STATUS_NO_DATA, STATUS_TOO_FEW_LINES, STATUS_FIT_FAILED,
)


def fit_region(cube_data, line_table, mask=None, molecule='H2', isotopologue=1, opr=None,
               fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None, min_lines=3,
               dedup_tol_um=1e-5, contam_lines=None, fit2t=False):
    """Fit one rotational diagram to a cube's summed spectrum over a region.
    This is the same fit the GUI's Run Fit button runs. Pass a mask to sum
    only some pixels, or omit it to sum the whole cube.

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        line_table (list): Line dicts with 'wavelength', 'a_coeff', 'e_upper', 'Gp',
            and 'branch'/'j_lower' for H2 S-branch lines -- the same rows the GUI's
            HITRAN fetch or CSV import populate
        mask (np.ndarray, optional): 2D boolean array, same spatial shape as the
            cube. True pixels are summed; None sums every pixel
        molecule (str): Species name; 'H2' uses the OPR-capable fitter
        isotopologue (int): HITRAN isotopologue number, ignored for H2
        opr (float, optional): Fixed ortho-para ratio, H2 only
        fit_opr (bool): Grid-search OPR to minimize chi-squared, H2 only
        opr_grid (array, optional): OPR values to search, H2 only
        dist_pc (float): Source distance in parsecs
        r_au (float, optional): Emitting region radius in AU. None reports Ncol
            as a total molecule count instead of a column density
        min_lines (int): Fewest detected lines needed before fitting
        dedup_tol_um (float): Co-located-line merge tolerance in microns
        contam_lines (pd.DataFrame, optional): Contaminating lines to give their
            own component in the fit, same format create_flux_map accepts
        fit2t (bool): Report the two-temperature fit instead of 1T

    Returns:
        dict: 'status' (int, one of STATUS_OK/STATUS_NO_DATA/STATUS_TOO_FEW_LINES/
            STATUS_FIT_FAILED), 'n_pixels', 'n_lines', and on STATUS_OK also
            'results' (rotational_diagram()'s own return value, with a '1T' or
            '2T' entry depending on fit2t); otherwise 'reason' explains why no
            fit was attempted

    Example:
        >>> cube = load_cube("data.fits")
        >>> result = fit_region(cube, line_table)
        >>> if result['status'] == STATUS_OK:
        ...     print(result['results']['1T']['T'])
    """
    return _rdm_fit_region(cube_data, line_table, mask=mask, molecule=molecule,
                           isotopologue=isotopologue, opr=opr, fit_opr=fit_opr,
                           opr_grid=opr_grid, dist_pc=dist_pc, r_au=r_au,
                           min_lines=min_lines, dedup_tol_um=dedup_tol_um,
                           contam_lines=contam_lines, fit2t=fit2t)


def fit_pixel(cube_data, x, y, line_table, molecule='H2', isotopologue=1, opr=None,
              fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None, min_lines=3,
              dedup_tol_um=1e-5, contam_lines=None, n_sigma=DEFAULT_DETECTION_SIGMA, fit2t=False):
    """Fit one rotational diagram to a single pixel's own spectrum.
    This is fit_region for one pixel instead of a summed region.

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        x (int): Spatial x pixel index
        y (int): Spatial y pixel index
        line_table (list): Line dicts, see fit_region
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, min_lines, dedup_tol_um, contam_lines, fit2t: Same meaning as in fit_region
        n_sigma (float): Detection threshold in noise sigmas. A line below it
            is an upper limit and is left out of the fit. 3.0 by default

    Returns:
        dict: Same shape as fit_region's return, for this one pixel

    Example:
        >>> cube = load_cube("data.fits")
        >>> result = fit_pixel(cube, 21, 21, line_table)
    """
    return _rdm_fit_pixel(cube_data, x, y, line_table, molecule=molecule,
                          isotopologue=isotopologue, opr=opr, fit_opr=fit_opr,
                          opr_grid=opr_grid, dist_pc=dist_pc, r_au=r_au,
                          min_lines=min_lines, dedup_tol_um=dedup_tol_um,
                          contam_lines=contam_lines, n_sigma=n_sigma, fit2t=fit2t)


def create_rotational_diagram_map(cube_data, line_table, molecule='H2', isotopologue=1,
                                   opr=None, fit_opr=False, opr_grid=None, dist_pc=140.0,
                                   r_au=None, min_lines=3, dedup_tol_um=1e-5, n_jobs=8,
                                   contam_lines=None, progress_callback=None, cancel_flag=None,
                                   n_sigma=DEFAULT_DETECTION_SIGMA, fit2t=False):
    """Fit a rotational diagram at every pixel of one cube.
    Same fit as the GUI's Run Map button, run once per pixel.

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        line_table (list): Line dicts, see fit_region
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, dedup_tol_um, contam_lines: Same meaning as in fit_region
        min_lines (int): Fewest lines a pixel needs, checked separately at each
            pixel -- one pixel can use different lines than its neighbor. Raise
            this for fit2t (a 2T fit has 4 free parameters, not 2)
        n_jobs (int): Worker threads, one cube row per task
        progress_callback (callable, optional): progress_callback(done_pixels, total_pixels)
        cancel_flag (object, optional): Object with a `.cancelled` attribute, checked
            between rows so a long-running map can be stopped early
        n_sigma (float): Detection threshold in noise sigmas, 3.0 by default. A line
            below it is an upper limit and stays NaN in the maps. Not applied to
            'region_fit', whose whole-cube sum keeps the 3-sigma rule
        fit2t (bool): Report the two-temperature fit instead of 1T, at every pixel

    Returns:
        dict: 'T', 'T_err' (2D float32, K), 'Ncol', 'Ncol_err' (2D float64, cm^-2, or molecules if r_au is None) --
            component 1 (1T's only component, or 2T's T1); 'T2', 'T2_err',
            'Ncol2', 'Ncol2_err' (2T's T2 component, all-NaN when fit2t is False);
            'fit2t', 'n_lines' (2D int, lines used per pixel), 'status' (2D int, one
            of STATUS_OK/STATUS_NO_DATA/STATUS_TOO_FEW_LINES/STATUS_FIT_FAILED),
            'region_fit' (the whole-cube-summed anchor fit, same as fit_region's
            return), plus 'header', 'wave_range', 'molecule', 'dist_pc', 'r_au',
            'opr', 'kept_lines', 'dropped_lines'

    Raises:
        ValueError: if a multi-cube list is not on one spatial grid

    Example:
        >>> cube = load_cube("data.fits")
        >>> result = create_rotational_diagram_map(cube, line_table, n_jobs=4)
        >>> print(f"Fitted {(result['status'] == STATUS_OK).sum()} pixels")
    """
    return _rdm_create_map(cube_data, line_table, molecule=molecule,
                           isotopologue=isotopologue, opr=opr, fit_opr=fit_opr,
                           opr_grid=opr_grid, dist_pc=dist_pc, r_au=r_au,
                           min_lines=min_lines, dedup_tol_um=dedup_tol_um, n_jobs=n_jobs,
                           contam_lines=contam_lines, progress_callback=progress_callback,
                           cancel_flag=cancel_flag, n_sigma=n_sigma, fit2t=fit2t)


def fit_region_multi(cubes_and_lines, molecule='H2', isotopologue=1, opr=None,
                      fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None,
                      min_lines=3, dedup_tol_um=1e-5, contam_lines=None, fit2t=False):
    """Fit one rotational diagram from several cubes at once, one line per cube.
    This is fit_region for many cubes instead of one. No convolution or
    reprojection is needed here, since every pixel of every cube is summed.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs, one per input cube
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, min_lines, dedup_tol_um, contam_lines, fit2t: Same meaning as in fit_region

    Returns:
        dict: 'status' (int, one of STATUS_OK/STATUS_TOO_FEW_LINES/STATUS_FIT_FAILED),
            'n_lines', 'n_pixels' (total spaxels summed, across every cube), 'n_cubes',
            and on STATUS_OK also 'results'; otherwise 'reason'

    Example:
        >>> cubes_and_lines = [(load_cube("h2_s1.fits"), {"wavelength": 17.034846}),
        ...                    (load_cube("h2_s2.fits"), {"wavelength": 12.278612})]
        >>> result = fit_region_multi(cubes_and_lines)
    """
    return _rdm_fit_region_multi(cubes_and_lines, molecule=molecule,
                                 isotopologue=isotopologue, opr=opr, fit_opr=fit_opr,
                                 opr_grid=opr_grid, dist_pc=dist_pc, r_au=r_au,
                                 min_lines=min_lines, dedup_tol_um=dedup_tol_um,
                                 contam_lines=contam_lines, fit2t=fit2t)


def create_rotational_diagram_map_multi(cubes_and_lines, molecule='H2', isotopologue=1,
                                         opr=None, fit_opr=False, opr_grid=None,
                                         dist_pc=140.0, r_au=None, min_lines=3,
                                         dedup_tol_um=1e-5, n_jobs=8, contam_lines=None,
                                         progress_callback=None, cancel_flag=None, n_sigma=DEFAULT_DETECTION_SIGMA,
                                         fit2t=False):
    """Fit a rotational diagram at every pixel of a common grid built from
    several cubes at different pixel scales, one line per cube. Convolves
    and reprojects each cube onto that grid first, then fits like Run Map.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs, one per input cube,
            at least 2
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, min_lines, dedup_tol_um, contam_lines, fit2t: Same meaning as in fit_region
        n_jobs (int): Worker threads for each cube's native-grid extraction
        progress_callback (callable, optional): progress_callback(done_units, total_units),
            spanning every native-grid extraction plus the final per-pixel fit
        cancel_flag (object, optional): Object with a `.cancelled` attribute
        n_sigma (float): Detection threshold in noise sigmas, 3.0 by default. Applies
            to the native fits only; the convolved maps get no second test. Not applied
            to 'region_fit', whose whole-cube sum keeps the 3-sigma rule

    Returns:
        dict: 'T', 'T_err' (2D float32, K), 'Ncol', 'Ncol_err' (2D float64, cm^-2, or molecules if r_au is None) --
            component 1; 'T2', 'T2_err', 'Ncol2', 'Ncol2_err' (2T's T2 component,
            all-NaN when fit2t is False); 'fit2t', 'n_lines' (2D int), 'status' (2D
            int, see create_rotational_diagram_map), 'region_fit' (the whole-cube-
            summed anchor fit, same as fit_region_multi's return), plus 'dst_wcs'
            (the new common-grid WCS), 'molecule', 'dist_pc', 'r_au', 'opr',
            'min_lines', 't_cap', 'worst_pix_arcsec', 'kept_lines', 'dropped_lines'

    Raises:
        ValueError: if fewer than 2 pairs are given, a cube has no readable
            celestial WCS/pixel scale, or two inputs share no sky coverage

    Example:
        >>> cubes_and_lines = [(load_cube("h2_s1.fits"), {"wavelength": 17.034846}),
        ...                    (load_cube("h2_s2.fits"), {"wavelength": 12.278612})]
        >>> result = create_rotational_diagram_map_multi(cubes_and_lines, n_jobs=4)
        >>> print(f"Common grid: {result['T'].shape}, worst pixel scale: "
        ...       f"{result['worst_pix_arcsec']:.4f} arcsec")
    """
    return _rdm_create_map_multi(cubes_and_lines, molecule=molecule,
                                 isotopologue=isotopologue, opr=opr, fit_opr=fit_opr,
                                 opr_grid=opr_grid, dist_pc=dist_pc, r_au=r_au,
                                 min_lines=min_lines, dedup_tol_um=dedup_tol_um,
                                 n_jobs=n_jobs, contam_lines=contam_lines,
                                 progress_callback=progress_callback, cancel_flag=cancel_flag,
                                 n_sigma=n_sigma, fit2t=fit2t)


def fit_pixel_multi(result, x, y, molecule='H2', isotopologue=1, opr=None, fit_opr=False,
                     opr_grid=None, dist_pc=140.0, r_au=None, min_lines=3, fit2t=False):
    """Re-fit one pixel of a create_rotational_diagram_map_multi result.
    Uses its cached per-line maps instead of redoing the convolution.

    Args:
        result (dict): Output of create_rotational_diagram_map_multi
        x (int): Spatial x index on the result's common grid
        y (int): Spatial y index on the result's common grid
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, min_lines, fit2t: Same meaning as in fit_region

    Returns:
        dict: Same shape as fit_pixel's return, for this pixel of the common grid

    Raises:
        KeyError: if `result` has no cached per-line maps (predates this
            function, or came from a different map type)

    Example:
        >>> result = create_rotational_diagram_map_multi(cubes_and_lines)
        >>> refit = fit_pixel_multi(result, 10, 15, fit_opr=True)
    """
    return _rdm_fit_pixel_multi(result, x, y, molecule=molecule, isotopologue=isotopologue,
                                opr=opr, fit_opr=fit_opr, opr_grid=opr_grid,
                                dist_pc=dist_pc, r_au=r_au, min_lines=min_lines, fit2t=fit2t)


def save_rotational_diagram_map_fits(result, output_path):
    """Write a rotational-diagram-map result to a multi-extension FITS file:
    one image HDU each for T, T_err, Ncol, Ncol_err, n_lines, and status --
    plus T2/T2_err/Ncol2/Ncol2_err when the result is a 2T fit.

    Args:
        result (dict): Output of create_rotational_diagram_map or
            create_rotational_diagram_map_multi
        output_path (str): Destination FITS path

    Returns:
        None

    Example:
        >>> result = create_rotational_diagram_map(cube, line_table)
        >>> save_rotational_diagram_map_fits(result, "rot_diag_map.fits")
    """
    _rdm_save_fits(result, output_path)


# --- Rotational Diagram Map API: the steps of Run Map, one at a time ---

from .rotational_diagram_map import (
    compute_native_line_maps as _rdm_compute_native,
    combine_line_maps as _rdm_combine,
    fit_region_mixed as _rdm_fit_region_mixed,
    create_rotational_diagram_map_mixed as _rdm_create_map_mixed,
    build_imported_map_payload as _rdm_build_imported_payload,
)
from .convolve_reproject import (
    convolution_plan as _convolution_plan,
    convolve_reproject_maps as _convolve_reproject_maps,
)
from .import_map_dialog import _peek_header, _has_real_error
from .line_catalog import match_nearest_hitran_line as _match_nearest_hitran_line


def compute_native_line_maps(cubes_and_lines, contam_lines=None, n_jobs=8, dedup_tol_um=1e-5,
                             progress_callback=None, cancel_flag=None, region_masks=None, n_sigma=DEFAULT_DETECTION_SIGMA):
    """Fit each line at the native resolution of its own cube. This is the Calculate Maps button of the
    rotational-diagram window, and the first step of Run Map. The next step is convolve_line_maps.
    Scripts that call this need an ``if __name__ == "__main__":`` guard on Windows/macOS.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs, one per input cube
        contam_lines (pandas.DataFrame, optional): Contaminating lines, or None
        n_jobs (int): Number of worker processes for each cube
        dedup_tol_um (float): Co-located-line merge tolerance in microns
        progress_callback (callable, optional): progress_callback(done_pixels, total_pixels)
        cancel_flag (object, optional): Object with a `.cancelled` attribute
        region_masks (dict, optional): {CubeData: 2D boolean mask}. Only the pixels in the mask are fitted
            for that cube. A cube that is not in the dict is fitted at every pixel.
        n_sigma (float): Detection threshold in noise sigmas (default 3.0). This is the only step that
            applies a threshold. The later steps use the maps as they are.

    Returns:
        dict: 'kept_lines', 'dropped_lines', 'native_flux_maps', 'native_flux_error_maps',
        'native_fit_flux_maps', 'native_status_maps', 'native_ul_ceiling_maps' (lists of 2D arrays, one
        for each kept line), 'native_wcs', 'native_headers', 'native_shapes', 'native_pix_arcsec',
        'native_instrument' and 'cube_filenames'. Pass it to convolution_plan, convolve_line_maps
        or save_line_maps.

    Raises:
        ValueError: if no pair is given, a cube has no celestial WCS or pixel scale, or two cubes
            share no sky coverage

    Example:
        >>> native = compute_native_line_maps([(load_cube("s1.fits"), {"wavelength": 17.0348, ...}),
        ...                                    (load_cube("s2.fits"), {"wavelength": 12.2786, ...})])
        >>> save_line_maps(native, "native_maps")
    """
    return _rdm_compute_native(cubes_and_lines, contam_lines=contam_lines, n_jobs=n_jobs,
                               dedup_tol_um=dedup_tol_um, progress_callback=progress_callback,
                               cancel_flag=cancel_flag, region_masks=region_masks, n_sigma=n_sigma)


def convolution_plan(native_result):
    """Show what convolve_line_maps will do, without doing it. This is the table that the GUI shows
    before Convolve Maps and Run Map. The line with the longest wavelength is the reference.

    Args:
        native_result (dict): Result of compute_native_line_maps

    Returns:
        dict: 'worst_pix_arcsec' (pixel scale of the reference line), 'dst_shape' (shape of the output
        grid), 'per_cube' (one dict for each line: 'wavelength', 'instrument', 'filename',
        'native_pix_arcsec', 'sigma_arcsec', 'sigma_pix') and 'dropped_lines'

    Raises:
        ValueError: if a line is from an instrument that has no PSF formula (only MIRI and NIRSpec have one)
    """
    return _convolution_plan(native_result)


def convolve_line_maps(native_result):
    """Convolve every native line map to the resolution of the line with the longest wavelength, and
    reproject it onto the grid of that line. This is the Convolve Maps button of the rotational-diagram
    window, and the second step of Run Map. The error maps go through the same steps.

    Args:
        native_result (dict): Result of compute_native_line_maps (or of merge of imported maps, see
            create_rotational_diagram_map_mixed)

    Returns:
        dict: 'kept_lines', 'line_flux_maps', 'line_flux_error_maps' (the convolved maps, all on one grid),
        'dst_wcs', 'dst_shape', 'worst_pix_arcsec', 'reference_wavelength', and the native maps that
        went in. Pass it to combine_line_maps or save_line_maps.

    Raises:
        ValueError: if a line is from an instrument that has no PSF formula (only MIRI and NIRSpec have one)

    Example:
        >>> convolved = convolve_line_maps(native)
        >>> save_line_maps(convolved, "convolved_maps.zip")
    """
    return _convolve_reproject_maps(native_result)


def combine_line_maps(convolved_result, molecule='H2', isotopologue=1, opr=None, fit_opr=False,
                      opr_grid=None, dist_pc=140.0, r_au=None, min_lines=3, region_fit=None,
                      progress_callback=None, cancel_flag=None, fit2t=False):
    """Fit a rotational diagram at every pixel of the convolved line maps. This is the last step of
    Run Map. It has no detection threshold of its own. The threshold was set in compute_native_line_maps.

    Args:
        convolved_result (dict): Result of convolve_line_maps
        molecule (str): Species name. 'H2' uses the fitter with the ortho-para ratio.
        isotopologue (int): HITRAN isotopologue number, not used for H2
        opr (float, optional): Fixed ortho-para ratio, H2 only
        fit_opr (bool): Search the ortho-para ratio at every pixel, H2 only
        opr_grid (array, optional): Ratios to search, H2 only
        dist_pc (float): Distance in parsecs
        r_au (float, optional): Emitting radius in AU. A map does not use the value: with a radius,
            N is per cm^2 of the area of each pixel. None gives molecules per pixel.
        min_lines (int): Fewest detected lines that a pixel needs for a fit
        region_fit (dict, optional): Result of fit_region_multi, to add the fit of the summed region
        progress_callback (callable, optional): progress_callback(done_pixels, total_pixels)
        cancel_flag (object, optional): Object with a `.cancelled` attribute
        fit2t (bool): Report the two-temperature fit instead of the one-temperature fit

    Returns:
        dict: The same as create_rotational_diagram_map_multi: 'T', 'T_err', 'Ncol', 'Ncol_err', 'T2',
        'T2_err', 'Ncol2', 'Ncol2_err', 'n_lines', 'status', 'dst_wcs' and more. Save it with
        save_rotational_diagram_map_fits.
    """
    return _rdm_combine(convolved_result, molecule=molecule, isotopologue=isotopologue, opr=opr,
                        fit_opr=fit_opr, opr_grid=opr_grid, dist_pc=dist_pc, r_au=r_au,
                        min_lines=min_lines, region_fit=region_fit, progress_callback=progress_callback,
                        cancel_flag=cancel_flag, fit2t=fit2t)


def import_line_map(fits_path, species=None, wavelength=None, error_pct=None):
    """Read a flux map that was made before, for example by Export of the Gaussian map, as one line
    of a rotational diagram. This is the Load Maps (Import Map) button. The line has no cube. Its map
    goes straight into create_rotational_diagram_map_mixed.

    The species and the wavelength come from the FITS header (the MOLECULE and LINEWAVE keywords) when
    it has them. A value that you give replaces the header value. The Einstein A, the upper energy and
    the statistical weight come from the header, or from the HITRAN line nearest to the wavelength.

    Args:
        fits_path (str): Path of the FITS file of the flux map
        species (str, optional): Species, for example 'H2'. Needed if the header has no MOLECULE keyword.
        wavelength (float, optional): Line wavelength in um. Needed if the header has no LINEWAVE keyword.
        error_pct (float, optional): Flat error in percent of the flux. Needed if the file has no error
            map (an ERR extension). It is not used if the file has one.

    Returns:
        dict: A line dict with 'wavelength', 'species', 'transition', 'hitran_matched' and 'imported_map'
        (the map, its error, its WCS and its pixel scale). If HITRAN has no match, 'hitran_matched' is
        False and the line has no A, E_u or g_u, so it is not fitted until you add them.

    Raises:
        ValueError: if the species, the wavelength or the error percent is needed and not given,
            or the file has no 2D image with a celestial WCS

    Example:
        >>> line = import_line_map("s1_gaussian_area_map.fits", error_pct=20.0)
        >>> result = create_rotational_diagram_map_mixed([(cube, other_line)], [line])
    """
    found = _peek_header(fits_path)
    if species is None and found.get("MOLECULE"):
        species = str(found["MOLECULE"])
    if not species or not str(species).strip():
        raise ValueError(f"{fits_path} has no MOLECULE keyword. Give species, for example 'H2'.")
    species = str(species).strip().upper()
    if wavelength is None and found.get("LINEWAVE") is not None:
        try:
            wavelength = float(found["LINEWAVE"])
        except (TypeError, ValueError):
            wavelength = None
    if wavelength is None:
        raise ValueError(f"{fits_path} has no usable LINEWAVE keyword. Give wavelength, in um.")
    wavelength = float(wavelength)
    if _has_real_error(fits_path):
        error_pct = 0.0
    elif error_pct is None:
        raise ValueError(f"{fits_path} has no error map. Give error_pct, the flat error in percent of the flux.")
    payload = _rdm_build_imported_payload(fits_path, error_pct)

    if all(key in found for key in ("A_UL", "E_UPPER", "G_U", "TRANSITN")):
        best_match = {'a_coeff': found['A_UL'], 'e_upper': found['E_UPPER'], 'Gp': found['G_U'],
                      'transition': found['TRANSITN'], 'j_lower': found.get('JLOWER'),
                      'branch': found.get('BRANCH')}
    else:
        try:
            hitran_lines, _meta = fetch_hitran_data(
                wavelength - 0.01, wavelength + 0.01, molecule=species, isotopologue=1)
        except Exception as error:
            warnings.warn(f"Could not fetch HITRAN data for {species} at {wavelength:.6f} um: {error}")
            hitran_lines = []
        best_match = _match_nearest_hitran_line(hitran_lines, wavelength)

    if best_match:
        line = {'wavelength': wavelength, 'hitran_matched': True,
                'a_coeff': best_match.get('a_coeff', 0.0), 'e_upper': best_match.get('e_upper', 0.0),
                'Gp': best_match.get('Gp', 1.0), 'j_lower': best_match.get('j_lower'),
                'branch': best_match.get('branch'), 'transition': best_match.get('transition', 'User Data'),
                'species': species}
    else:
        warnings.warn(f"No HITRAN match for {species} at {wavelength:.6f} um. A_ul, E_u and g_u are blank, "
                      "so this line is not fitted until you add them.")
        line = {'wavelength': wavelength, 'hitran_matched': False, 'transition': 'NA', 'species': species}
    line['imported_map'] = payload
    return line


def fit_region_mixed(cubes_and_lines, imported_lines, molecule='H2', isotopologue=1, opr=None,
                     fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None, min_lines=3,
                     dedup_tol_um=1e-5, contam_lines=None, region_masks=None,
                     imported_region_masks=None, fit2t=False):
    """Fit one rotational diagram to the summed region, with some lines from cubes and some from
    imported maps (see import_line_map). The same as fit_region_multi, plus the imported lines.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs for the lines that come from cubes. Can be empty.
        imported_lines (list): Line dicts from import_line_map
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, min_lines, dedup_tol_um, contam_lines, fit2t: The same as in fit_region_multi
        region_masks (dict, optional): {CubeData: 2D boolean mask}. Only these pixels are summed for that cube.
        imported_region_masks (dict, optional): {id(line): 2D boolean mask} for an imported line. A line
            that is not in the dict sums its whole map.

    Returns:
        dict: The same as fit_region_multi
    """
    return _rdm_fit_region_mixed(cubes_and_lines, imported_lines, molecule=molecule,
                                 isotopologue=isotopologue, opr=opr, fit_opr=fit_opr, opr_grid=opr_grid,
                                 dist_pc=dist_pc, r_au=r_au, min_lines=min_lines, dedup_tol_um=dedup_tol_um,
                                 contam_lines=contam_lines, region_masks=region_masks,
                                 imported_region_masks=imported_region_masks, fit2t=fit2t)


def create_rotational_diagram_map_mixed(cubes_and_lines, imported_lines, molecule='H2', isotopologue=1,
                                        opr=None, fit_opr=False, opr_grid=None, dist_pc=140.0, r_au=None,
                                        min_lines=3, dedup_tol_um=1e-5, n_jobs=8, contam_lines=None,
                                        region_fit=None, progress_callback=None, cancel_flag=None,
                                        region_masks=None, imported_region_masks=None, n_sigma=DEFAULT_DETECTION_SIGMA,
                                        fit2t=False):
    """Make a rotational-diagram map from lines that come from cubes and lines that come from imported
    maps (see import_line_map). The same as create_rotational_diagram_map_multi, plus the imported lines.
    Scripts that call this need an ``if __name__ == "__main__":`` guard on Windows/macOS.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs for the lines that come from cubes. Can be empty.
        imported_lines (list): Line dicts from import_line_map
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, min_lines, dedup_tol_um, n_jobs, contam_lines, progress_callback, cancel_flag, n_sigma, fit2t: The same as in create_rotational_diagram_map_multi
        region_fit (dict, optional): Result of fit_region_mixed, to add the fit of the summed region
        region_masks (dict, optional): {CubeData: 2D boolean mask}. Only these pixels are fitted for that cube.
        imported_region_masks (dict, optional): {id(line): 2D boolean mask} for an imported line

    Returns:
        dict: The same as create_rotational_diagram_map_multi

    Raises:
        ValueError: if no line is given, a map has no celestial WCS or pixel scale, or two inputs
            share no sky coverage
    """
    return _rdm_create_map_mixed(cubes_and_lines, imported_lines, molecule=molecule,
                                 isotopologue=isotopologue, opr=opr, fit_opr=fit_opr, opr_grid=opr_grid,
                                 dist_pc=dist_pc, r_au=r_au, min_lines=min_lines, dedup_tol_um=dedup_tol_um,
                                 n_jobs=n_jobs, contam_lines=contam_lines, region_fit=region_fit,
                                 progress_callback=progress_callback, cancel_flag=cancel_flag,
                                 region_masks=region_masks, imported_region_masks=imported_region_masks,
                                 n_sigma=n_sigma, fit2t=fit2t)


# --- Map Saving API: the FITS files of the Save and Export buttons ---

from .map_export import (
    save_gaussian_map as _save_gaussian_map,
    save_gaussian_maps as _save_gaussian_maps,
    save_flux_map as _save_flux_map,
    save_line_maps as _save_line_maps,
)


def save_gaussian_map(cube_data, maps, parameter, out_path, lam_min, lam_max, peak_wavelength=None,
                      rest_wavelength=None, rest_label=None, systemic_velocity=0.0, species=None):
    """Save one parameter of a Gaussian map to a FITS file. This is the Export button of the Gaussian map
    window. The header has the fit window, the unit and the keywords that the parameter needs
    (for example VSYS for the Velocity map and CONTWAV for the Continuum map).

    Args:
        cube_data (CubeData): Loaded cube that the map was fitted on
        maps (dict): Result of create_gaussian_map()
        parameter (str): One of 'area', 'area_error', 'amplitude', 'continuum', 'center', 'velocity',
            'sigma', 'fwhm', 'chi2', 'fit_status', 'success_mask'
        out_path (str): Path of the FITS file to write
        lam_min (float): Start of the fit window, in the wavelength unit of the cube
        lam_max (float): End of the fit window, in the wavelength unit of the cube
        peak_wavelength (float, optional): Wavelength of the line peak. None uses the middle of the window.
        rest_wavelength (float, optional): Rest wavelength in um (vacuum). The Velocity map needs it.
        rest_label (str, optional): Short text that names the rest line, for the header
        systemic_velocity (float): Systemic velocity in km/s that the Velocity map is relative to
        species (str, optional): Species of the line, for example 'H2'. For the Area map only. It adds the
            HITRAN constants of the nearest line, so import_line_map can read the file.

    Returns:
        None

    Raises:
        ValueError: if the parameter is not known, or the Velocity map has no rest_wavelength

    Example:
        >>> maps = create_gaussian_map(cube, lam_min=17.02, lam_max=17.05, lambda_0=17.0348)
        >>> save_gaussian_map(cube, maps, 'area', 's1_area.fits', 17.02, 17.05,
        ...                   peak_wavelength=17.0348, species='H2')
    """
    _save_gaussian_map(cube_data, maps, parameter, out_path, lam_min, lam_max, peak_wavelength=peak_wavelength,
                       rest_wavelength=rest_wavelength, rest_label=rest_label,
                       systemic_velocity=systemic_velocity, species=species)


def save_gaussian_maps(cube_data, maps, out_dir, lam_min, lam_max, parameters=None, peak_wavelength=None,
                       rest_wavelength=None, rest_label=None, systemic_velocity=0.0, species=None,
                       base_name=None):
    """Save several parameters of a Gaussian map, one FITS file for each, in one folder. The names are
    <cube name>_gaussian_<parameter>.fits. The Velocity map is left out when there is no rest_wavelength.

    Args:
        cube_data (CubeData): Loaded cube that the map was fitted on
        maps (dict): Result of create_gaussian_map()
        out_dir (str): Folder for the files. It is made if it does not exist.
        lam_min (float): Start of the fit window, in the wavelength unit of the cube
        lam_max (float): End of the fit window, in the wavelength unit of the cube
        parameters (list, optional): Parameter names, see save_gaussian_map. None writes all of them.
        peak_wavelength, rest_wavelength, rest_label, systemic_velocity, species: See save_gaussian_map
        base_name (str, optional): First part of each file name. None uses the name of the cube file.

    Returns:
        list: The paths of the files that were written
    """
    return _save_gaussian_maps(cube_data, maps, out_dir, lam_min, lam_max, parameters=parameters,
                               peak_wavelength=peak_wavelength, rest_wavelength=rest_wavelength,
                               rest_label=rest_label, systemic_velocity=systemic_velocity,
                               species=species, base_name=base_name)


def save_flux_map(result, out_path, species=None, cube_data=None):
    """Save a Flux Map result to a FITS file. This is the Export button of the Flux Map window. The first
    image is the flux map. The extensions hold the detection counts and the contaminating lines.

    Args:
        result (dict): Result of create_flux_map()
        out_path (str): Path of the FITS file to write
        species (str, optional): Species of the target lines, for example 'CO'. None writes 'Target'.
        cube_data (CubeData, optional): Loaded cube. It gives the instrument name if the header has none.

    Returns:
        None

    Example:
        >>> flux = create_flux_map(cube, target_catalog_path="co_lines.ecsv")
        >>> save_flux_map(flux, "co_flux_map.fits", species="CO", cube_data=cube)
    """
    _save_flux_map(result, out_path, species=species, cube_data=cube_data)


def save_line_maps(result, output_path, molecule=None):
    """Save the flux map of every line of a rotational-diagram result. This is the Save Maps button of the
    rotational-diagram window, without the PNG files. The native maps go in 'Flux Maps/fits' and the
    convolved maps in 'Convolved Maps/fits'. Each file is <species>_<flux or convolved>_<wavelength>um.fits.

    Args:
        result (dict): Result of compute_native_line_maps (native maps), convolve_line_maps (convolved maps),
            create_rotational_diagram_map (native maps) or create_rotational_diagram_map_multi (both)
        output_path (str): A folder, or a path that ends in .zip to write one zip file
        molecule (str, optional): Species name for the file names and headers. None uses the one in the
            result, or 'H2'.

    Returns:
        dict: {'Flux Maps': number of native maps written, 'Convolved Maps': number of convolved maps
        written}. A group that the result does not have is not in the dict.

    Raises:
        ValueError: if the result has no line maps

    Example:
        >>> native = compute_native_line_maps(cubes_and_lines)
        >>> convolved = convolve_line_maps(native)
        >>> save_line_maps(convolved, "convolved_maps.zip")
    """
    return _save_line_maps(result, output_path, molecule=molecule)


# --- Regions and Tables API: the region files and the CSV files of the GUI ---

from .table_export import (
    make_region_mask as _make_region_mask,
    save_region_file as _save_region_file,
    load_region_file as _load_region_file,
    save_line_fluxes_csv as _save_line_fluxes_csv,
    load_line_fluxes_csv as _load_line_fluxes_csv,
    save_rotational_fit_csv as _save_rotational_fit_csv,
)
from .unit_converter import doppler_velocity_to_wavelength_factor as _doppler_factor


def make_region_mask(shape, coords, target_shape, verts=None):
    """Make a boolean mask of a region from its shape and its pixel coordinates. The mask has True for the
    pixels that the region touches, the same as a region that you draw in the GUI. Use it as mask in
    extract_region_spectrum or fit_region.

    Args:
        shape (str): 'POINT', 'RECTANGLE', 'CIRCLE', 'ELLIPSE' or 'POLYGON'
        coords (tuple): (x, y) for 'POINT'. (xmin, xmax, ymin, ymax) in pixels for 'RECTANGLE', 'CIRCLE' and
            'ELLIPSE'. The circle and the ellipse fill that box. Not used for 'POLYGON'.
        target_shape (tuple): (height, width) of the image, for example cube.cube.shape[1:]
        verts (array-like, optional): (N, 2) polygon vertices as (x, y) pixels. Needed for 'POLYGON'.

    Returns:
        numpy.ndarray: 2D boolean array of shape target_shape

    Raises:
        ValueError: if the shape is not known, or a polygon has no vertices

    Example:
        >>> mask = make_region_mask('CIRCLE', (20, 30, 15, 25), cube.cube.shape[1:])
        >>> spectrum = extract_region_spectrum(cube, mask)
    """
    return _make_region_mask(shape, coords, target_shape, verts=verts)


def save_region(path, shape, coords, cube_data, verts=None, source_desc=None):
    """Save a region to a text file, as RA and Dec sky coordinates. This is Save Region in the GUI.
    The file works on any image of the same field. Read it with load_region.

    Args:
        path (str): Path of the text file to write
        shape (str): 'POINT', 'RECTANGLE', 'CIRCLE', 'ELLIPSE' or 'POLYGON'
        coords (tuple): Pixel coordinates, see make_region_mask
        cube_data (CubeData): Loaded cube. Its WCS turns the pixels into sky coordinates.
        verts (array-like, optional): (N, 2) polygon vertices as (x, y) pixels. Needed for 'POLYGON'.
        source_desc (str, optional): Note for the header of the file. None uses the name of the cube file.

    Returns:
        None

    Raises:
        ValueError: if the shape is not known, a polygon has no vertices, or the cube has no celestial WCS
    """
    _save_region_file(path, shape, coords, cube_data, verts=verts, source_desc=source_desc)


def load_region(path, cube_data):
    """Read a region file and turn it into a boolean mask on the pixel grid of a cube. This is Import Region
    in the GUI. The file can come from another image of the same field, because it holds sky coordinates.

    Args:
        path (str): Path of a region file that Save Region wrote
        cube_data (CubeData): Loaded cube that the mask is for

    Returns:
        numpy.ndarray: 2D boolean array, shape (ny, nx) of the cube. Use it as mask in
        extract_region_spectrum or fit_region, or in region_masks.

    Raises:
        ValueError: if the file is not a region file, the cube has no celestial WCS, or the region
            is outside the cube

    Example:
        >>> mask = load_region("my_region.txt", cube)
        >>> spectrum = extract_region_spectrum(cube, mask)
    """
    return _load_region_file(path, cube_data)


def save_line_fluxes(path, lines):
    """Save the line fluxes to a CSV file with the columns species, wavelength, flux, flux_error and
    upper_limit. This is Save Fluxes in the rotational-diagram window. Read it with load_line_fluxes.
    A line with no flux is not written.

    Args:
        path (str): Path of the CSV file to write
        lines (list): One dict for each line, with 'species', 'wavelength' (um), 'flux' and 'flux_error'
            (erg/s/cm2). A line with 'upper_limit' True is written as an upper limit.

    Returns:
        int: Number of lines written

    Raises:
        ValueError: if no line has a flux
    """
    return _save_line_fluxes_csv(path, lines)


def load_line_fluxes(path):
    """Read a CSV file of lines and add the HITRAN constants that the rotational diagram needs. This is
    Import from CSV in the rotational-diagram window. The file needs the columns species and wavelength
    (um). The columns flux, an error column and upper_limit are optional. Each line is matched to the
    nearest HITRAN line. A line with no match is flagged ('hitran_matched' False) and has no A, E_u and g_u.

    Args:
        path (str): Path of the CSV file

    Returns:
        list: One dict for each valid row. It is the line_table of fit_region and
        create_rotational_diagram_map. A row with flux and flux_error also gives the fluxes for
        rotational_diagram.

    Raises:
        ValueError: if the file is empty, has no species or wavelength column, or has no valid row

    Example:
        >>> lines = load_line_fluxes("H2_lines.csv")
        >>> result = fit_region(cube, lines)
    """
    return _load_line_fluxes_csv(path)


def save_rotational_fit(path, results, header_text=None):
    """Save a rotational-diagram fit to a CSV file: one row for each point of the diagram, then the fit
    line. This is Save Fit in the rotational-diagram window. The lines that start with "#" at the top
    hold the results.

    Args:
        path (str): Path of the CSV file to write
        results (dict): The 'results' of fit_region or fit_pixel, or the result of rotational_diagram.
            It must have '_plot_data'.
        header_text (str, optional): Text for the "#" lines at the top. None writes the single numbers
            of the '1T' and '2T' entries.

    Returns:
        None

    Raises:
        ValueError: if results has no '_plot_data'

    Example:
        >>> fit = fit_region(cube, lines)
        >>> save_rotational_fit("fit.csv", fit['results'])
    """
    _save_rotational_fit_csv(path, results, header_text=header_text)


def doppler_shift_factor(velocity):
    """Give the number that moves observed wavelengths to rest wavelengths for a systemic velocity.
    This is the multiplicative Doppler shift of the GUI: rest = observed / (1 + v/c).

    Args:
        velocity (float): Systemic velocity in km/s. Positive means moving away (redshift).

    Returns:
        float: The factor. Multiply the wavelengths by it.

    Example:
        >>> rest_wavelength = cube.wavelength * doppler_shift_factor(17.5)
    """
    return _doppler_factor(velocity)


def fetch_hitran_lines(lam_min, lam_max, molecule, isotopologue=1, refresh_cache=False, v_upper=None,
                       v_lower=None):
    """Get the lines of one molecule from HITRAN, or from the local copy when there is one. This is
    Fetch HITRAN in the GUI. Each line is a dict with 'wavelength' (um, vacuum), 'a_coeff', 'e_upper',
    'Gp', 'transition' and more.

    Args:
        lam_min (float): Start of the wavelength range in um
        lam_max (float): End of the wavelength range in um
        molecule (str): Molecule name, for example 'H2' or 'CO'
        isotopologue (int): HITRAN isotopologue number (default 1). None gives all of them.
        refresh_cache (bool): True fetches the lines again, even if a copy is stored
        v_upper (int, optional): Keep only lines with this upper vibrational level
        v_lower (int, optional): Keep only lines with this lower vibrational level

    Returns:
        list: The line dicts, sorted by wavelength. Empty if HITRAN has no line in the range.

    Raises:
        RuntimeError: if the lines could not be fetched (for example, no network and no local copy)

    Example:
        >>> lines = fetch_hitran_lines(4.6, 5.3, 'CO', v_upper=1, v_lower=0)
    """
    rows, meta = fetch_hitran_data(lam_min, lam_max, molecule, isotopologue=isotopologue,
                                   refresh_cache=refresh_cache, v_upper=v_upper, v_lower=v_lower)
    if meta.get('error'):
        raise RuntimeError(f"Could not get {molecule} lines from HITRAN: {meta.get('message', 'unknown error')}")
    return rows


# --- Unit Conversion API: no wrapper functions here, use astropy.units
# directly, e.g. (2.293*u.um).to(u.Hz, equivalencies=u.brightness_temperature(...)). ---
__all__ = [
    # Data Loading
    'load_cube',
    'load_spectrum',
    'load_fit',
    # Region Extraction
    'extract_pixel_spectrum',
    'extract_region_spectrum',
    # Continuum
    'fit_continuum',
    'save_continuum_masks',
    'load_continuum_masks',
    'save_continuum_cubes',
    # Gaussian Fitting
    'fit_line',
    'fit_hitran_lines',
    'measure_equivalent_width',
    'create_gaussian_map',
    'find_rest_lines',
    'create_velocity_map',
    'export_residual_cube',
    # Flux
    'extract_flux',
    'create_flux_map',
    # Rotational Diagram
    'rotational_diagram',
    # Rotational Diagram Map
    'fit_region',
    'fit_pixel',
    'create_rotational_diagram_map',
    'fit_region_multi',
    'create_rotational_diagram_map_multi',
    'fit_pixel_multi',
    'save_rotational_diagram_map_fits',
    # Rotational Diagram Map, one step at a time
    'compute_native_line_maps',
    'convolution_plan',
    'convolve_line_maps',
    'combine_line_maps',
    'import_line_map',
    'fit_region_mixed',
    'create_rotational_diagram_map_mixed',
    # Map Saving
    'save_gaussian_map',
    'save_gaussian_maps',
    'save_flux_map',
    'save_line_maps',
    # Regions and Tables
    'make_region_mask',
    'save_region',
    'load_region',
    'save_line_fluxes',
    'load_line_fluxes',
    'save_rotational_fit',
    # Helpers
    'doppler_shift_factor',
    'fetch_hitran_lines',
    'STATUS_OK',
    'STATUS_NO_DATA',
    'STATUS_TOO_FEW_LINES',
    'STATUS_FIT_FAILED',
]
