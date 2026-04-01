"""
IFU Analyzer API - Non-GUI programmatic interface

This module provides a clean API for using IFU analysis functions without the GUI.
All functions in this module work as pure Python functions and can be imported
and used in scripts, Jupyter notebooks, or other Python applications.

Main Categories:
1. Data Loading     - Load FITS cubes and extract spectra
2. Gaussian Fitting - Fit spectral lines with Gaussian profiles
3. CO Flux Analysis - Extract CO line fluxes with continuum handling
4. Unit Conversion  - Convert between wavelength/velocity/frequency units
5. Line Catalogs    - Query and filter spectral line databases

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
    >>> print(f"Line center: {result.center} µm")
    >>> print(f"FWHM: {result.fwhm} µm")

CO flux extraction:
    >>> from cube_explorer.api import extract_co_flux
    >>> co_result = extract_co_flux(
    ...     wavelength=cube.wavelength,
    ...     spectrum=spectrum,
    ...     co_catalog_path="path/to/CO_catalog.ecsv"
    ... )
    >>> print(f"Total CO flux: {co_result['co_flux']} erg/s/cm²")

Unit conversion (using astropy):
    >>> import astropy.units as u
    >>> wavelength = 2.293 * u.um
    >>> frequency = wavelength.to(u.Hz, equivalencies=u.spectral_density(wavelength))
    >>> print(f"Frequency: {frequency}")
"""

# ============================================================================
# DATA LOADING API
# ============================================================================

from .data_loader import CubeData

def load_cube(filename):
    """
    Load a FITS data cube from file.

    This function handles multi-instrument FITS files (MUSE, ALMA, MIRI, NIRSpec)
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
    return fit_continuum_median_peaks(wavelength, flux, **kwargs)


# ============================================================================
# GAUSSIAN FITTING API
# ============================================================================

from .gaussian_fitter import (
    fit_gaussian_to_line,
    calculate_equivalent_width,
    fit_hitran_lines as _fit_hitran_lines_internal,
)
from .line_catalog import fetch_hitran_data

def fit_line(wavelength, spectrum, x1, x2, line_name=None, auto_boundaries=True):
    """
    Fit a Gaussian to an emission line in a spectrum.

    Automatically detects line center and estimates parameters using
    smart peak detection and noise estimation.

    Args:
        wavelength (array-like): Wavelength/frequency values
        spectrum (array-like): Flux values
        x1 (float): Left boundary of fitting region
        x2 (float): Right boundary of fitting region
        line_name (str, optional): Name of line for documentation
        auto_boundaries (bool): Auto-detect boundaries from peak (default True)

    Returns:
        GaussianFitResult: Object containing:
            - amplitude: Peak height
            - center: Line center wavelength
            - sigma: Gaussian sigma width
            - continuum: Fitted continuum baseline
            - fwhm: Full width at half maximum
            - success: Whether fit was successful
            - message: Fitting status message

    Example:
        >>> result = fit_line(
        ...     wavelength=cube.wavelength,
        ...     spectrum=spectrum,
        ...     x1=2.292,
        ...     x2=2.300,
        ...     line_name="CO(5-4)"
        ... )
        >>> if result.success:
        ...     print(f"Line center: {result.center} µm")
        ...     print(f"FWHM: {result.fwhm:.6f} µm")
    """
    return fit_gaussian_to_line(x1, x2, wavelength, spectrum)


def fit_multiple_lines(wavelength, spectrum, line_list, auto_boundaries=False):
    """
    Fit multiple spectral lines simultaneously.

    Args:
        wavelength (array-like): Wavelength/frequency values
        spectrum (array-like): Flux values
        line_list (list): List of (x1, x2) tuples defining fitting regions
        auto_boundaries (bool): Auto-detect boundaries (default False)

    Returns:
        list: List of GaussianFitResult objects

    Example:
        >>> lines = [(2.292, 2.300), (2.310, 2.318), (2.405, 2.413)]
        >>> results = fit_multiple_lines(cube.wavelength, spectrum, lines)
        >>> for i, result in enumerate(results):
        ...     print(f"Line {i}: center={result.center:.6f}, FWHM={result.fwhm:.6f}")
    """
    return [fit_line(wavelength, spectrum, x1, x2) for x1, x2 in line_list]


def fit_hitran_lines(wavelength, spectrum, molecule, wavelength_min=None, wavelength_max=None, isotopologue=1, refresh_cache=False, v_upper=None, v_lower=None):
    """
    Fit HITRAN spectral lines automatically using database lookup.

    This is the same workflow as the GUI's HITRAN line fitting. It automatically
    fetches line data from the HITRAN database (or local cache), then fits
    Gaussian profiles to all lines in the wavelength range.

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

    Returns:
        list: List of dicts with fit results for each line:
            - wavelength: Line center wavelength
            - species: Molecule name
            - transition: Transition description
            - amplitude: Peak height
            - sigma: Gaussian sigma width
            - fwhm: Full width at half maximum
            - area: Integrated line flux
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
        ...     v_lower=0
        ... )
        >>> detected = [r for r in results if r['success']]
        >>> print(f"Detected {len(detected)} CO v=1->0 lines")
    """
    import numpy as np
    
    # Use spectrum edges if not provided
    if wavelength_min is None:
        wavelength_min = float(np.nanmin(wavelength))
    if wavelength_max is None:
        wavelength_max = float(np.nanmax(wavelength))
    
    # Fetch HITRAN data for this molecule
    hitran_lines, _ = fetch_hitran_data(
        wavelength_min, wavelength_max,
        molecule, isotopologue,
        refresh_cache=refresh_cache,
        v_upper=v_upper,
        v_lower=v_lower
    )
    
    if not hitran_lines:
        print(f"Warning: No HITRAN lines found for {molecule} in range {wavelength_min:.3f}-{wavelength_max:.3f} µm")
        return []
    
    # Fit Gaussian profiles to all lines
    return _fit_hitran_lines_internal(wavelength, spectrum, hitran_lines)


def measure_equivalent_width(fit_result, wavelength, continuum_spectrum):
    """
    Calculate equivalent width from a Gaussian fit result.

    Equivalent width is the width of an imaginary rectangular line with the
    same total area as the observed line.

    Args:
        fit_result (GaussianFitResult): Result from fit_line()
        wavelength (array-like): Wavelength array
        continuum_spectrum (array-like): Continuum-only spectrum

    Returns:
        float: Equivalent width in same units as wavelength

    Example:
        >>> ew = measure_equivalent_width(result, wavelength, continuum)
        >>> print(f"Equivalent width: {ew:.6f} µm")
    """
    return calculate_equivalent_width(fit_result, wavelength, continuum_spectrum)


# ============================================================================
# CO FLUX EXTRACTION API
# ============================================================================

from .co_flux_extractor import (
    extract_co_flux_from_spectrum as _extract_co_flux_internal,
    create_co_flux_map as _create_co_flux_map_internal,
)
from .line_catalog import load_CO_line_catalog, load_contaminating_lines_catalog

def extract_co_flux(wavelength, spectrum, co_catalog_path, contamination_catalog_path=None, snr_threshold=3.0):
    """
    Extract total CO flux by fitting all CO lines in a spectrum.

    Fits multiple CO rovibrational lines simultaneously with contaminating
    lines and continuum, then integrates total flux.

    Args:
        wavelength (array-like): Wavelength values in same units as catalog
        spectrum (array-like): Flux values
        co_catalog_path (str): Path to CO line catalog (ECSV format)
        contamination_catalog_path (str, optional): Path to contaminant lines catalog
        snr_threshold (float): Minimum signal-to-noise for line detection (default 3.0)

    Returns:
        dict: Results containing:
            - 'co_flux': Total CO flux (erg/s/cm²)
            - 'line_fluxes': Dict of individual line fluxes
            - 'continuum': Fitted continuum spectrum
            - 'residual': Spectrum after line removal
            - 'n_detected': Number of detected CO lines

    Example:
        >>> result = extract_co_flux(
        ...     wavelength=cube.wavelength,
        ...     spectrum=spectrum,
        ...     co_catalog_path="CO/CO_comprehensive_linelist_with_g.ecsv",
        ...     snr_threshold=3.0
        ... )
        >>> print(f"Total CO flux: {result['co_flux']:.4e} erg/s/cm²")
        >>> print(f"Detected {result['n_detected']} lines")
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
        do_duplicate_removal=True
    )


def create_co_map(cube_data, co_catalog_path=None, contamination_catalog_path=None, n_jobs=8, wavelength_min=None, wavelength_max=None):
    """
    Create a 2D map of CO flux across the entire data cube.

    Fits CO lines in every pixel and generates output FITS maps. Can use either:
    - Automatic HITRAN fetching (default, like fit_hitran_lines)
    - Custom CO catalog file (if co_catalog_path provided)

    Args:
        cube_data (CubeData): Loaded cube from load_cube()
        co_catalog_path (str, optional): Path to CO line catalog (ECSV format).
                                         If None, automatically fetches from HITRAN.
        contamination_catalog_path (str, optional): Path to contaminant lines
        n_jobs (int): Number of parallel jobs (default 8)
        wavelength_min (float, optional): Min wavelength for HITRAN search (auto-detect if None)
        wavelength_max (float, optional): Max wavelength for HITRAN search (auto-detect if None)

    Returns:
        dict: Map data containing:
            - 'flux_map': 2D array of CO fluxes
            - 'amplitude_map': Peak amplitudes
            - 'sigma_map': Gaussian sigma widths
            - 'center_map': Line center shifts
            - 'success_mask': Boolean mask of successful fits

    Example:
        >>> cube = load_cube("data.fits")
        >>> co_map = create_co_map(cube, n_jobs=4)
        >>> print(f"Map shape: {co_map['flux_map'].shape}")
    """
    # If no catalog provided, fetch from HITRAN automatically
    if co_catalog_path is None:
        print("  Fetching CO data from HITRAN...")
        import numpy as np
        
        # Extract wavelength range, handling Quantity objects
        wl = cube_data.wavelength
        if hasattr(wl, 'value'):
            wl = wl.value
        
        wavelength_min = float(np.nanmin(wl)) if wavelength_min is None else wavelength_min
        wavelength_max = float(np.nanmax(wl)) if wavelength_max is None else wavelength_max
        
        co_catalog, _ = fetch_hitran_data(
            wavelength_min, wavelength_max,
            'CO', isotopologue=1,
            refresh_cache=False
        )
        if not co_catalog:
            print("  ERROR: Could not fetch CO data from HITRAN")
            return None
        contam_catalog = None
    else:
        # Load catalogs from files (original behavior)
        co_catalog = load_CO_line_catalog(co_catalog_path)
        contam_catalog = None
        if contamination_catalog_path:
            contam_catalog = load_contaminating_lines_catalog(contamination_catalog_path)
    
    return _create_co_flux_map_internal(
        cube_data, co_catalog, contam_catalog,
        n_jobs=n_jobs
    )


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
    'fit_multiple_lines',
    'fit_hitran_lines',
    'measure_equivalent_width',
    # CO Flux
    'extract_co_flux',
    'create_co_map',
]
