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

import numpy as np


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
        numpy.ndarray: 1D spectrum at (x, y), same length as cube.wavelength

    Example:
        >>> cube = load_cube("data.fits")
        >>> spectrum = extract_pixel_spectrum(cube, x=20, y=25)
    """
    return cube.get_pixel_spectrum(x, y)


def extract_region_spectrum(cube, mask):
    """Extract the mean spectrum over an arbitrary spatial region, the same
    averaging the GUI uses for region selections (each reduces to a 2D
    boolean mask). Excludes NaN pixels instead of propagating NaN.

    Args:
        cube (CubeData): Loaded cube, as returned by load_cube()
        mask (numpy.ndarray): 2D boolean array matching the cube's spatial
            shape (cube.cube.shape[1:]); True marks pixels inside the region

    Returns:
        numpy.ndarray: 1D mean spectrum over the masked pixels

    Example:
        >>> cube = load_cube("data.fits")
        >>> mask = np.zeros(cube.cube.shape[1:], dtype=bool)
        >>> mask[15:35, 10:30] = True  # a rectangular region, [y, x] order
        >>> spectrum = extract_region_spectrum(cube, mask)
    """
    return cube.compute_region_spectrum(mask)


# --- Continuum Fitting API ---

from .flux_map_extractor import fit_auto_continuum

def fit_continuum(wavelength, flux, method='auto', **kwargs):
    """Fit continuum to a spectrum using automatic peak detection: the same
    method the GUI uses, detecting emission/absorption peaks and fitting a
    smooth continuum through the remaining points.

    Args:
        wavelength (array-like): Wavelength/frequency values
        flux (array-like): Flux values
        method (str): Fitting method - currently only 'auto' is supported
        **kwargs: Additional arguments for fine-tuning:
            window_size (int): Size of median filter window (default 51, must be odd)
            sigma_threshold (float): Prominence threshold in sigma (default 2.0)
            peak_width (int): Minimum width of peaks in pixels (default 1)
            peak_distance (int): Minimum distance between peaks (default 3)
            peak_exclusion (int): Pixels to exclude on each side of peak (default 3)
            iterations (int): Iterative clipping iterations (default 2)
            smoothing_factor (float): Spline smoothing parameter (default 5)

    Returns:
        array: Fitted continuum values matching input wavelength array

    Example:
        >>> wavelength = cube.wavelength
        >>> spectrum = cube.spectrum
        >>> continuum = fit_continuum(wavelength, spectrum, sigma_threshold=2.0)
        >>> line_only = spectrum - continuum
    """
    if method != 'auto':
        raise ValueError(f"Unknown continuum fitting method: {method}. Use 'auto'.")
    return fit_auto_continuum(
        _as_float_array(wavelength), _as_float_array(flux), **kwargs
    )


# --- Gaussian Fitting API ---

from .gaussian_fitter import (
    fit_gaussian_to_line,
    calculate_equivalent_width,
    fit_gaussians_per_pixel as _fit_gaussians_per_pixel_internal,
    export_residual_cube_fits as _export_residual_cube_fits_internal,
)
from .flux_map_extractor import fit_hitran_lines_collective as _fit_hitran_lines_internal
from .line_catalog import fetch_hitran_data

def fit_line(wavelength, spectrum, x1, x2, line_name=None, peak_x=None,
             header=None, wave_unit=None):
    """Fit a Gaussian to an emission line in a spectrum. The fitter needs a
    starting guess for the line center; here it defaults to the brightest
    sample between x1 and x2, or pass peak_x explicitly.

    Args:
        wavelength (array-like): Wavelength/frequency values. Astropy
            Quantities are accepted; their units are stripped.
        spectrum (array-like): Flux values.
        x1 (float): Left boundary of the fitting region.
        x2 (float): Right boundary of the fitting region.
        line_name (str, optional): Descriptive label for the line. Not stored
            on the result; it exists for readability at the call site.
        peak_x (float, optional): Initial guess for the line center. Defaults
            to the brightest wavelength between ``x1`` and ``x2``.
        header (optional): FITS header used for unit conversion. Pass
            ``cube.header`` to match what the GUI produces.
        wave_unit (str, optional): Spectral axis unit, e.g. ``cube.wave_unit``.

    Returns:
        GaussianFitResult: Object with ``amplitude``, ``center``, ``sigma``,
        ``fwhm``, ``area``, ``area_unit``, ``area_native``, ``baseline_slope``,
        ``baseline_intercept``, ``r_squared``, ``success`` and ``message``.

        ``area`` is in erg/s/cm^2 when ``header`` is supplied and the cube is
        in MJy/sr (MIRI, NIRSpec); otherwise it stays in flux*wavelength units.
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
    fl = _as_float_array(spectrum)

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

    return fit_gaussian_to_line(x1, x2, wl, fl, peak_x=peak_x,
                                header=header, wave_unit=wave_unit)


def fit_hitran_lines(wavelength, spectrum, molecule, wavelength_min=None, wavelength_max=None,
                     isotopologue=1, refresh_cache=False, v_upper=None, v_lower=None,
                     header=None, wave_unit=None, continuum=None, contam_catalog=None):
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
    sp = _as_float_array(spectrum)

    # Use spectrum edges if not provided
    if wavelength_min is None:
        wavelength_min = float(np.nanmin(wl))
    if wavelength_max is None:
        wavelength_max = float(np.nanmax(wl))

    # Fetch HITRAN data for this molecule
    hitran_lines, _ = fetch_hitran_data(
        wavelength_min, wavelength_max,
        molecule, isotopologue,
        refresh_cache=refresh_cache,
        v_upper=v_upper,
        v_lower=v_lower
    )

    if not hitran_lines:
        print(f"Warning: No HITRAN lines found for {molecule} in range {wavelength_min:.6f}-{wavelength_max:.6f} um")
        return []

    if continuum is None:
        continuum_arr = fit_continuum(wl, sp)
    else:
        continuum_arr = _as_float_array(continuum)

    # Contaminant fits (if given) aren't surfaced; they only keep requested
    # lines' amplitudes honest. Center/sigma are free within a bound, not fixed.
    fit_pairs, _contam_pairs, _ = _fit_hitran_lines_internal(
        wl, sp, hitran_lines, continuum_arr,
        header=header, wave_unit=wave_unit,
        contam_lines=contam_catalog)

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
        dict: 2D parameter maps - 'amplitude', 'center', 'sigma', 'fwhm', 'area',
            'r_squared', 'success', 'is_detection', 'is_upper_limit'

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


def export_residual_cube(cube_data, out_path="residual_cube.fits", lam_min=None,
                          lam_max=None, n_jobs=None, return_cube=False):
    """
    Fit and subtract a Gaussian at every pixel, writing the residual cube to FITS.

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        out_path (str): Output FITS filename (default "residual_cube.fits")
        lam_min (float, optional): Minimum wavelength for the fit window
        lam_max (float, optional): Maximum wavelength for the fit window
        n_jobs (int, optional): Number of worker threads (default: auto-detect CPUs)
        return_cube (bool): If True, return the residual array instead of writing
            to out_path -- the two are exclusive, not additive: out_path is
            ignored when return_cube is True.

    Returns:
        array or None: The residual cube if return_cube is True; otherwise None
            (the FITS file is written to out_path as a side effect instead).

    Example:
        >>> cube = load_cube("data.fits")
        >>> export_residual_cube(cube, out_path="residual.fits", lam_min=4.65, lam_max=4.70)
    """
    wl = cube_data.wavelength
    if hasattr(wl, 'value'):
        wl = wl.value
    return _export_residual_cube_fits_internal(
        cube=cube_data.cube,
        wavelength=wl,
        lam_min=lam_min,
        lam_max=lam_max,
        n_jobs=n_jobs,
        out_path=out_path,
        return_cube=return_cube,
        header=cube_data.header,
    )


# --- CO Flux Extraction API ---

import pandas as pd
from .flux_map_extractor import (
    extract_flux_from_spectrum as _extract_flux_internal,
    create_flux_map as _create_flux_map_internal,
)
from .line_catalog import load_CO_line_catalog, load_contaminating_lines_catalog

def extract_flux(wavelength, spectrum, target_catalog_path, contamination_catalog_path=None,
                     snr_threshold=3.0, continuum_params=None,
                     header=None, wave_unit=None):
    """Extract total flux by fitting all target lines in a spectrum
    simultaneously with contaminating lines and continuum; the same
    algorithm as the GUI's Flux Map button, run on a single spectrum.

    Args:
        wavelength (array-like): Wavelength values in same units as catalog
        spectrum (array-like): Flux values
        target_catalog_path (str): Path to target line catalog (ECSV format)
        contamination_catalog_path (str, optional): Path to contaminant lines catalog
        snr_threshold (float): Minimum signal-to-noise for line detection (default 3.0)
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
    # Load catalogs from files
    target_catalog = load_CO_line_catalog(target_catalog_path)
    contam_catalog = None
    if contamination_catalog_path:
        contam_catalog = load_contaminating_lines_catalog(contamination_catalog_path)

    # Call internal function
    return _extract_flux_internal(
        wavelength, spectrum,
        target_catalog, contam_catalog,
        do_duplicate_removal=True,
        snr_threshold=snr_threshold,
        continuum_params=continuum_params,
        header=header, wave_unit=wave_unit,
    )


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
              supports the conversion (see 'flux_unit'), else the cube's native flux unit
            - 'contam_flux_map': 2D array of contaminating-line flux per pixel, same units
              as 'flux_map'
            - 'detection_map': 2D array with number of detected target lines per pixel
            - 'contam_detection_map': 2D array with number of detected contaminating lines
              per pixel
            - 'wavelength': 1D wavelength array used for fitting
            - 'header': FITS header from the original cube
            - 'wave_range': (wavelength_min, wavelength_max) actually used for the fit
            - 'flux_unit': unit of 'flux_map'/'contam_flux_map'

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

        hitran_lines, _ = fetch_hitran_data(
            wavelength_min, wavelength_max,
            'CO', isotopologue=1,
            refresh_cache=False
        )
        if not hitran_lines:
            print("  ERROR: Could not fetch CO data from HITRAN")
            return None
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

from .rotational_diagram_h2_fitter import rotational_diagram as _rotational_diagram_h2, pc as _rot_diag_pc
from .rotational_diagram_general_fitter import rotational_diagram as _rotational_diagram_general


def rotational_diagram(fluxes, flux_errors, molecular_data=None, OPR=None, fit_opr=False, opr_grid=None,
                        fit1T=True, fit2T=False, R=None, dist=140*_rot_diag_pc, upper_limits=None,
                        progress_callback=None, molecule='H2', isotopologue=1):
    """
    Fit a rotational diagram (1T and/or 2T) to a set of line fluxes. Dispatches to the
    H2-specific fitter (hardcoded partition function, OPR support) or the general
    HITRAN-based fitter (any molecule, e.g. CO) depending on `molecule`.

    Args:
        fluxes (dict): {J: flux_value}, J is the upper rotational level
        flux_errors (dict): {J: flux_error}
        molecular_data (dict): 'wavelengths', 'Aul', 'Eu', 'gu' keys, each {J: value}
        OPR (float): Fixed ortho-para ratio (H2 only; ignored otherwise)
        fit_opr (bool): Grid-search OPR to minimize chi-squared (H2 only; ignored otherwise)
        opr_grid (array): OPR values to search (H2 only). Defaults to np.linspace(0.1, 5.0, 200)
        fit1T (bool): Report the 1T fit (always computed regardless, as 2T's own seed)
        fit2T (bool): Perform the 2T fit
        R (float): Emitting radius (AU or cm)
        dist (float): Distance (pc or cm)
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
              dedup_tol_um=1e-5, contam_lines=None, n_sigma=3.0, fit2t=False):
    """Fit one rotational diagram to a single pixel's own spectrum.
    This is fit_region for one pixel instead of a summed region.

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        x (int): Spatial x pixel index
        y (int): Spatial y pixel index
        line_table (list): Line dicts, see fit_region
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, min_lines,
            dedup_tol_um, contam_lines, fit2t: Same meaning as in fit_region
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
                                   n_sigma=3.0, fit2t=False):
    """Fit a rotational diagram at every pixel of one cube.
    Same fit as the GUI's Run Map button, run once per pixel.

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        line_table (list): Line dicts, see fit_region
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, dedup_tol_um,
            contam_lines: Same meaning as in fit_region
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
        dict: 'T', 'T_err' (2D float32, K), 'Ncol', 'Ncol_err' (2D float32, cm^-2) --
            component 1 (1T's only component, or 2T's "Warm" one); 'T2', 'T2_err',
            'Ncol2', 'Ncol2_err' (2T's "Hot" component, all-NaN when fit2t is False);
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
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, min_lines,
            dedup_tol_um, contam_lines, fit2t: Same meaning as in fit_region

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
                                         progress_callback=None, cancel_flag=None, n_sigma=3.0,
                                         fit2t=False):
    """Fit a rotational diagram at every pixel of a common grid built from
    several cubes at different pixel scales, one line per cube. Convolves
    and reprojects each cube onto that grid first, then fits like Run Map.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs, one per input cube,
            at least 2
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, min_lines,
            dedup_tol_um, contam_lines, fit2t: Same meaning as in fit_region
        n_jobs (int): Worker threads for each cube's native-grid extraction
        progress_callback (callable, optional): progress_callback(done_units, total_units),
            spanning every native-grid extraction plus the final per-pixel fit
        cancel_flag (object, optional): Object with a `.cancelled` attribute
        n_sigma (float): Detection threshold in noise sigmas, 3.0 by default. Applies
            to the native fits only; the convolved maps get no second test. Not applied
            to 'region_fit', whose whole-cube sum keeps the 3-sigma rule

    Returns:
        dict: 'T', 'T_err' (2D float32, K), 'Ncol', 'Ncol_err' (2D float32, cm^-2) --
            component 1; 'T2', 'T2_err', 'Ncol2', 'Ncol2_err' (2T's "Hot" component,
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
        molecule, isotopologue, opr, fit_opr, opr_grid, dist_pc, r_au, min_lines,
            fit2t: Same meaning as in fit_region

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


# --- Unit Conversion API: no wrapper functions here, use astropy.units
# directly, e.g. (2.293*u.um).to(u.Hz, equivalencies=u.brightness_temperature(...)). ---
__all__ = [
    # Data Loading
    'load_cube',
    # Region Extraction
    'extract_pixel_spectrum',
    'extract_region_spectrum',
    # Continuum
    'fit_continuum',
    # Gaussian Fitting
    'fit_line',
    'fit_hitran_lines',
    'measure_equivalent_width',
    'create_gaussian_map',
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
    'STATUS_OK',
    'STATUS_NO_DATA',
    'STATUS_TOO_FEW_LINES',
    'STATUS_FIT_FAILED',
]
