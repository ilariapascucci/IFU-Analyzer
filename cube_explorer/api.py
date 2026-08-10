"""
IFU Analyzer API - Non-GUI programmatic interface

This module provides a clean API for using IFU analysis functions without the GUI.
All functions in this module work as pure Python functions and can be imported
and used in scripts, Jupyter notebooks, or other Python applications.

Main Categories:
1. Data Loading        - Load FITS cubes and extract spectra
2. Gaussian Fitting    - Fit spectral lines with Gaussian profiles
3. CO Flux Analysis    - Extract CO line fluxes with continuum handling
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

CO flux extraction:
    >>> from cube_explorer.api import extract_co_flux
    >>> co_result = extract_co_flux(
    ...     wavelength=cube.wavelength,
    ...     spectrum=spectrum,
    ...     co_catalog_path="path/to/CO_catalog.ecsv"
    ... )
    >>> print(f"Total CO flux: {co_result['co_flux']} erg/s/cm^2")

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
    """
    Return a plain float ndarray, stripping astropy units if present.

    ``load_cube`` returns ``cube.wavelength`` as an astropy Quantity. Several
    of the underlying algorithms do arithmetic that silently produces NaN when
    handed a united array, so every public entry point normalises its inputs
    through this helper first.
    """
    return np.asarray(getattr(values, "value", values), dtype=float)


# ============================================================================
# DATA LOADING API
# ============================================================================

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


# ============================================================================
# CONTINUUM FITTING API
# ============================================================================

from .co_flux_extractor import fit_continuum_median_peaks

def fit_continuum(wavelength, flux, method='auto', **kwargs):
    """
    Fit continuum to a spectrum using automatic peak detection.

    This is the automated continuum subtraction method used in the GUI.
    It detects emission/absorption peaks and fits a smooth continuum
    through the remaining points.

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
    return fit_continuum_median_peaks(
        _as_float_array(wavelength), _as_float_array(flux), **kwargs
    )


# ============================================================================
# GAUSSIAN FITTING API
# ============================================================================

from .gaussian_fitter import (
    fit_gaussian_to_line,
    calculate_equivalent_width,
    fit_gaussians_per_pixel as _fit_gaussians_per_pixel_internal,
    export_residual_cube_fits as _export_residual_cube_fits_internal,
)
from .co_flux_extractor import fit_hitran_lines_collective as _fit_hitran_lines_internal
from .line_catalog import fetch_hitran_data

def fit_line(wavelength, spectrum, x1, x2, line_name=None, peak_x=None,
             header=None, wave_unit=None):
    """
    Fit a Gaussian to an emission line in a spectrum.

    The underlying fitter needs a starting guess for the line center. In the
    GUI that comes from the wavelength you click on; here it defaults to the
    brightest sample between ``x1`` and ``x2``, or you can pass ``peak_x``
    explicitly.

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

        ``area`` is in erg/s/cm² when ``header`` is supplied and the cube is in
        MJy/sr (MIRI, NIRSpec); otherwise it stays in flux×wavelength units.
        Check ``area_unit`` rather than assuming — it always states which.
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
    """
    Fit HITRAN spectral lines automatically using database lookup.

    This is the same algorithm as the GUI's **Multi-Gaussian Fit** button, not a
    separate approximation of it: every line's width comes from the instrument's
    spectral resolution (not fitted), and its center is fixed at its catalog
    wavelength plus one shared velocity shift estimated from these lines' own peaks
    (see ``fit_hitran_lines_collective``'s docstring), which turns fitting all of them
    into a single linear least-squares solve (``np.linalg.lstsq``) instead of an
    iterative nonlinear fit. Lines that overlap are disentangled by the same solve
    automatically -- no independent-fit double-claiming of shared flux, and no
    per-selection cost blowup from fitting many lines together.

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
        header: Optional FITS header for the erg/s/cm2 conversion, and for looking up
            the instrument's spectral resolution (INSTRUME/GRATING). Without it, area
            stays in the cube's native units -- same rule as fit_line.
        wave_unit: Wavelength unit string, needed alongside header for the conversion.
        continuum (array-like, optional): Pre-fitted continuum, same length as
            wavelength, subtracted before fitting each line. If None, one is fitted
            automatically via ``fit_continuum()`` -- the same fallback the GUI uses
            when you have not run Automatic Continuum interactively first.
        contam_catalog (pd.DataFrame, optional): Non-CO/target lines (a 'line_wave_um'
            column, same format as ``load_contaminating_lines_catalog()``) to give their
            own component in the fit so they don't distort a nearby requested line's
            amplitude -- the same optional catalog ``extract_co_flux``/CO Flux Map
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
            - area_unit: Units of ``area`` -- ``'erg/s/cm²'`` only when header/wave_unit
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

    # Fit Gaussian profiles to all lines at once via linear least squares. Contaminant fits
    # (if contam_catalog was given) aren't surfaced here -- they exist to keep the requested
    # lines' amplitudes honest, not as an additional reportable result.
    fit_pairs, _contam_pairs, velocity_shift_kms = _fit_hitran_lines_internal(
        wl, sp, hitran_lines, continuum_arr,
        header=header, wave_unit=wave_unit,
        contam_lines=contam_catalog)
    if velocity_shift_kms is not None:
        print(f"Applied a {velocity_shift_kms:+.2f} km/s wavelength shift to every line's fixed "
              f"center (estimated from these lines' own peaks) -- each result's 'wavelength' "
              f"reflects this, not the unshifted catalog value.")

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
    """
    Calculate equivalent width from a Gaussian fit result.

    Equivalent width is the width of an imaginary rectangular line with the
    same total area as the observed line.

    The underlying calculation needs a single continuum level. If you pass a
    full continuum array, it is averaged over the wavelengths that were
    actually fitted, so the level reflects the line's own neighbourhood
    rather than the whole spectrum.

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
    dialog. RANSAC outlier rejection is always enabled.

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        lam_min (float, optional): Minimum wavelength for the fit window (default: full range)
        lam_max (float, optional): Maximum wavelength for the fit window (default: full range)
        n_jobs (int, optional): Number of worker threads (default: auto-detect CPUs)
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
        return_cube (bool): If True, also return the residual array in memory

    Returns:
        str, or (str, array) if return_cube=True: Path to the written FITS file

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


# ============================================================================
# CO FLUX EXTRACTION API
# ============================================================================

import pandas as pd
from .co_flux_extractor import (
    extract_co_flux_from_spectrum as _extract_co_flux_internal,
    create_co_flux_map as _create_co_flux_map_internal,
)
from .line_catalog import load_CO_line_catalog, load_contaminating_lines_catalog

def extract_co_flux(wavelength, spectrum, co_catalog_path, contamination_catalog_path=None,
                     snr_threshold=3.0, subtract_continuum=True, continuum_params=None,
                     header=None, wave_unit=None):
    """
    Extract total CO flux by fitting all CO lines in a spectrum.

    Fits multiple CO rovibrational lines simultaneously with contaminating
    lines and continuum, then integrates total flux. This is the same algorithm
    as the GUI's CO Flux Map button, run on a single spectrum instead of every pixel.

    Args:
        wavelength (array-like): Wavelength values in same units as catalog
        spectrum (array-like): Flux values
        co_catalog_path (str): Path to CO line catalog (ECSV format)
        contamination_catalog_path (str, optional): Path to contaminant lines catalog
        snr_threshold (float): Minimum signal-to-noise for line detection (default 3.0)
        subtract_continuum (bool): Fit and subtract a continuum before fitting lines,
            using the same method as the GUI's Automatic Continuum (default True --
            matches the GUI, which always does this)
        continuum_params (dict, optional): Override the default continuum-fitting
            parameters (window_size, sigma_threshold, peak_width, peak_distance,
            peak_exclusion, iterations, smoothing_factor). If None, uses the defaults,
            which work well for typical spectra.
        header: FITS header, used to convert 'co_flux'/'contam_flux' to erg/s/cm2.
            Without it, they stay in the cube's native flux units -- 'flux_unit' in
            the result always tells you which one you got.
        wave_unit (str, optional): Wavelength unit (e.g. 'um'). Required alongside
            header for the CGS conversion.

    Returns:
        dict: Results containing:
            - 'co_flux': Total CO flux, in erg/s/cm2 if header/wave_unit were given
              and the conversion is supported for this instrument, else native units
            - 'contam_flux': Total flux attributed to contaminating lines, same units
            - 'flux_unit': unit of 'co_flux'/'contam_flux'
            - 'mean_continuum': Mean continuum level used for the fit (native units)
            - 'detected_lines': DataFrame of individual detected line fits
            - 'residuals': DataFrame of fit residuals
            - 'continuum_curve': DataFrame of the fitted continuum vs wavelength

    Example:
        >>> result = extract_co_flux(
        ...     wavelength=cube.wavelength,
        ...     spectrum=cube.spectrum,
        ...     co_catalog_path="CO/CO_comprehensive_linelist_with_g.ecsv",
        ...     header=cube.header,
        ...     wave_unit=cube.wave_unit,
        ... )
        >>> print(f"Total CO flux: {result['co_flux']:.4e} {result['flux_unit']}")
        >>> print(f"Detected {len(result['detected_lines'])} lines")
    """
    # Load catalogs from files
    co_catalog = load_CO_line_catalog(co_catalog_path)
    contam_catalog = None
    if contamination_catalog_path:
        contam_catalog = load_contaminating_lines_catalog(contamination_catalog_path)

    # Call internal function
    return _extract_co_flux_internal(
        wavelength, spectrum,
        co_catalog, contam_catalog,
        do_duplicate_removal=True,
        snr_threshold=snr_threshold,
        subtract_continuum=subtract_continuum,
        continuum_params=continuum_params,
        header=header, wave_unit=wave_unit,
    )


def create_co_flux_map(cube_data, co_catalog_path=None, contamination_catalog_path=None, n_jobs=8,
                        wavelength_min=None, wavelength_max=None, subtract_continuum=True, continuum_params=None):
    """
    Create a 2D map of CO flux across the entire data cube.

    Fits CO lines in every pixel and generates output FITS maps. Can use either:
    - Automatic HITRAN fetching (default, like fit_hitran_lines)
    - Custom CO catalog file (if co_catalog_path provided)

    This is the same algorithm as the GUI's CO Flux Map button: continuum subtraction
    (default parameters, or your own via continuum_params) followed by a simultaneous
    multi-Gaussian fit per pixel.

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        co_catalog_path (str, optional): Path to CO line catalog (ECSV format).
                                         If None, automatically fetches from HITRAN.
        contamination_catalog_path (str, optional): Path to contaminant lines
        n_jobs (int): Number of parallel jobs (default 8)
        wavelength_min (float, optional): Min wavelength to fit, and for the HITRAN
            search when co_catalog_path is None (auto-detect if None)
        wavelength_max (float, optional): Max wavelength to fit, and for the HITRAN
            search when co_catalog_path is None (auto-detect if None)
        subtract_continuum (bool): Fit and subtract a continuum per pixel before fitting
            lines, using the same method as the GUI's Automatic Continuum (default True
            -- matches the GUI, which always does this)
        continuum_params (dict, optional): Override the default continuum-fitting
            parameters, applied identically to every pixel. If None, uses the defaults,
            which work well for typical spectra.

    Returns:
        dict: Map data containing:
            - 'flux_map': 2D array of CO flux per pixel, in erg/s/cm2 if cube_data.header
              supports the conversion (see 'flux_unit'), else the cube's native flux unit
            - 'detection_map': 2D array with number of detected CO lines per pixel
            - 'wavelength': 1D wavelength array used for fitting
            - 'header': FITS header from the original cube
            - 'wave_range': (wavelength_min, wavelength_max) actually used for the fit
            - 'flux_unit': unit of 'flux_map'

    Example:
        >>> cube = load_cube("data.fits")
        >>> co_map = create_co_flux_map(cube, n_jobs=4)
        >>> print(f"Map shape: {co_map['flux_map'].shape}, unit: {co_map['flux_unit']}")
    """
    # If no catalog provided, fetch from HITRAN automatically
    if co_catalog_path is None:
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
        # co_flux_extractor expects a DataFrame with a 'line_wave_um' column;
        # fetch_hitran_data returns a list of dicts with 'wavelength' instead
        co_catalog = pd.DataFrame(hitran_lines).rename(columns={'wavelength': 'line_wave_um'})
        contam_catalog = None
    else:
        # Load catalogs from files (original behavior)
        co_catalog = load_CO_line_catalog(co_catalog_path)
        contam_catalog = None
        if contamination_catalog_path:
            contam_catalog = load_contaminating_lines_catalog(contamination_catalog_path)

    return _create_co_flux_map_internal(
        cube_data, co_catalog, contam_catalog,
        n_jobs=n_jobs,
        subtract_continuum=subtract_continuum,
        continuum_params=continuum_params,
        wave_min=wavelength_min, wave_max=wavelength_max,
    )


# ============================================================================
# ROTATIONAL DIAGRAM API
# ============================================================================

from .rotational_diagram_fitter import rotational_diagram


# ============================================================================
# UNIT CONVERSION API
# ============================================================================

# Note: Standard unit conversion functions like wavelength_to_velocity()
# are available but not exported here. Use astropy.units for unit conversions:
#   from astropy.units import Quantity
#   wavelength = 2.293 * u.um
#   frequency = wavelength.to(u.Hz, equivalencies=u.brightness_temperature(wavelength))
__all__ = [
    # Data Loading
    'load_cube',
    # Continuum
    'fit_continuum',
    # Gaussian Fitting
    'fit_line',
    'fit_hitran_lines',
    'measure_equivalent_width',
    'create_gaussian_map',
    'export_residual_cube',
    # CO Flux
    'extract_co_flux',
    'create_co_flux_map',
    # Rotational Diagram
    'rotational_diagram',
]
