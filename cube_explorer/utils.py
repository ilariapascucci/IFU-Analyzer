"""
Utility functions and constants for the Cube Explorer
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.io import fits
from .unit_converter import convert_wavelength

# Resolving-power tables from the JWST docs (Jakobsen et al. 2022), one file per disperser.
NIRSPEC_DISP_DIR = Path(__file__).resolve().parent.parent / 'nirspec_disp'
NIRSPEC_GRATINGS = ('G140H', 'G140M', 'G235H', 'G235M', 'G395H', 'G395M', 'PRISM')

# MUSE line spread function FWHM in Angstrom, a polynomial in wavelength in Angstrom, from sky
# lines in the udf-10 datacube (Bacon et al. 2017, Eq. 8). Highest power first. Measured from
# 4750 to 9350 Angstrom.
MUSE_LSF_FWHM_COEFFS = (5.866e-8, -9.187e-4, 6.040)
MUSE_LSF_RANGE_ANGSTROM = (4750.0, 9350.0)

# Keywords that only the PRIMARY header of a MUSE file has. The loader copies them to the cube header.
MUSE_PRIMARY_KEYS = ('INSTRUME', 'TELESCOP', 'ESO INS MODE')


def safe_worker_count(requested_jobs, per_worker_mb=400, usable_fraction=0.6):
    """
    Cap a worker-process count to fit in available memory. Each spawned
    process needs its own numpy/scipy import, independent of the parent's.

    Args:
        requested_jobs (int): Worker count requested before the memory check
        per_worker_mb (int): Estimated startup memory per worker, in MB
        usable_fraction (float): Fraction of current available memory this check can use

    Returns:
        n_jobs (int): requested_jobs, or fewer if available memory is tight
    """
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 ** 2)
    except Exception:
        print("psutil not available -- skipping the memory-based worker cap.")
        return requested_jobs

    max_by_memory = max(1, int((available_mb * usable_fraction) // per_worker_mb))
    n_jobs = max(1, min(requested_jobs, max_by_memory))
    if n_jobs < requested_jobs:
        print(f"Capping worker processes to {n_jobs} (requested {requested_jobs}): "
              f"only {available_mb:.0f} MB available.")
    return n_jobs


def _merge_primary_into_sci(hdul, sci_key):
    """
    Return the SCI header with PRIMARY keywords added as a fallback.

    Args:
        hdul (astropy.io.fits.HDUList): Open FITS file
        sci_key (int or str): Index or name of the SCI header

    Returns:
        sci_hdr (astropy.io.fits.Header): Copy of the SCI header, with PRIMARY keys added

    Raises:
        KeyError: if sci_key is not in hdul
    """

    sci_hdr = hdul[sci_key].header.copy()
    primary_hdr = hdul[0].header
    # Only copy PRIMARY keys that are missing from SCI
    for key, value, comment in primary_hdr.cards:
        if key and key not in sci_hdr and key not in ('SIMPLE', 'EXTEND', 'END'):
            try:
                sci_hdr[key] = (value, comment)
            except Exception:
                pass
    return sci_hdr


def _muse_header(hdul):
    """
    Return the DATA header of a MUSE cube with three PRIMARY keywords added.
    The keywords name the instrument, the telescope and the observing mode.
    Only these three are copied, because the full PRIMARY header changes the WCS frame.

    Args:
        hdul (astropy.io.fits.HDUList): Open MUSE FITS file, with the cube in HDU 1

    Returns:
        data_hdr (astropy.io.fits.Header): Copy of the DATA header, plus the keywords
            that PRIMARY has and DATA does not

    Raises:
        IndexError: if hdul has no HDU 1
    """
    data_hdr = hdul[1].header.copy()
    primary_hdr = hdul[0].header
    for key in MUSE_PRIMARY_KEYS:
        if key in primary_hdr and key not in data_hdr:
            # A HIERARCH card keeps the spaces in the keyword and gives no warning
            card_key = f"HIERARCH {key}" if " " in key else key
            data_hdr[card_key] = (primary_hdr[key], primary_hdr.comments[key])
    return data_hdr


"""
Where the main info lies on each telescope.
"""

TELESCOPE_LOADERS = {
    "MUSE": lambda hdul: (_muse_header(hdul), hdul[1].data),
    "MIRI": lambda hdul: (
        (_merge_primary_into_sci(hdul, "SCI"), hdul["SCI"].data)
        if "SCI" in hdul else (hdul[0].header, hdul[0].data)
    ),
    "NIRSpec": lambda hdul: (
        (_merge_primary_into_sci(hdul, "SCI"), hdul["SCI"].data)
        if "SCI" in hdul else (hdul[0].header, hdul[0].data)
    ),
}


def detect_instrument_from_header(header):
    """
    Detect the instrument or telescope from one header's own keywords, with
    no other HDUs to fall back on.

    Args:
        header (astropy.io.fits.Header): A single header.

    Returns:
        instrument (str or None): 'MUSE', 'MIRI', or 'NIRSpec', or None if not found
    """
    instrument = header.get("INSTRUME", "").strip().upper()
    telescope = header.get("TELESCOP", "").strip().upper()
    if "MUSE" in instrument or "MUSE" in telescope:
        return "MUSE"
    if "MIRI" in instrument or "MIRI" in telescope:
        return "MIRI"
    if "NIRSPEC" in instrument or "NIRSPEC" in telescope:
        return "NIRSpec"
    return None


def detect_instrument(hdul):
    """
    Detect the instrument or telescope from FITS header keywords. Checks every HDU, since
    some files keep the instrument name outside the primary header.

    Args:
        hdul (astropy.io.fits.HDUList): Open FITS file

    Returns:
        instrument (str or None): 'MUSE', 'MIRI', or 'NIRSpec', or None if not found
    """
    for hdu in hdul:
        instrument = detect_instrument_from_header(hdu.header)
        if instrument is not None:
            return instrument
    return None  # if no match found


@lru_cache(maxsize=None)
def _load_nirspec_curve(grating):
    """
    Read the resolving-power table of one NIRSpec disperser from its FITS file.
    The result is cached, so each file is read one time only.
    The table has wavelength in microns and R = lambda/delta_lambda (no unit).

    Args:
        grating (str): Disperser name in upper case, one of NIRSPEC_GRATINGS

    Returns:
        curve (tuple): Two numpy arrays, (wavelength in microns, resolving power).
            The wavelength array is strictly increasing.

    Raises:
        FileNotFoundError: if the FITS file for this grating is not in NIRSPEC_DISP_DIR
        ValueError: if the table has NaN values or the wavelength does not increase
    """
    path = NIRSPEC_DISP_DIR / f'jwst_nirspec_{grating.lower()}_disp.fits'
    if not path.is_file():
        raise FileNotFoundError(
            f"Resolving-power table for NIRSpec {grating} not found: {path}")

    with fits.open(path) as hdul:
        table = hdul['TABLE'].data
        wavelength = np.array(table['WAVELENGTH'], dtype=float)
        resolution = np.array(table['R'], dtype=float)

    if not (np.all(np.isfinite(wavelength)) and np.all(np.isfinite(resolution))):
        raise ValueError(f"Resolving-power table {path.name} has NaN or infinite values.")
    if np.any(np.diff(wavelength) <= 0):
        raise ValueError(f"Wavelength in {path.name} does not increase. Interpolation needs it to.")
    return wavelength, resolution


def get_spectral_resolution(wavelength, instrument, header=None):
    """
    Calculate spectral resolution R(lambda) = lambda/delta_lambda for different instruments.
    MIRI and MUSE use formulas. NIRSpec uses the JWST table of its disperser, read from FITS files.
    For NIRSpec, R at a wavelength between two table rows is a straight line between them.

    Args:
        wavelength (float): Wavelength, in microns
        instrument (str): Instrument name ('NIRSPEC', 'MIRI' or 'MUSE')
        header (astropy.io.fits.Header): FITS header. Required for NIRSpec, to read the grating.
            For MUSE, it gives the observing mode for a warning. It can be None.

    Returns:
        resolution (float): Spectral resolution R(lambda) at this wavelength. For a NIRSpec or
            MUSE wavelength outside the table or the measured range, this is R at the nearest end.

    Raises:
        ValueError: If instrument cannot be detected, grating is invalid, or the NIRSpec or
            MUSE wavelength is NaN or infinite
        FileNotFoundError: If the table file of the NIRSpec grating is missing

    Formulas:
        MIRI: R(lambda) = 4603 - 128 * lambda
        NIRSpec: R from the table of the GRATING header value (nirspec_disp/ folder)
        MUSE: R(lambda) = lambda / FWHM(lambda), with FWHM = 5.866e-8 * lambda^2 - 9.187e-4 * lambda
            + 6.040, lambda and FWHM in Angstrom (Bacon et al. 2017, Eq. 8, udf-10 datacube).
            The paper measured this on two wide-field datacubes, and it does not name the observing
            mode. The formula is the same for every MUSE cube and does not use the header.
    """

    instrument = instrument.upper()

    if instrument == 'MIRI':
        # MIRI spectral resolution formula
        return 4603 - 128 * wavelength

    elif instrument == 'MUSE':
        if not np.isfinite(wavelength):
            raise ValueError(f"MUSE wavelength must be finite, got {wavelength}.")

        mode = header.get('ESO INS MODE', None) if header is not None else None
        mode = str(mode).strip().upper() if mode is not None else None
        if mode is None or not (mode.startswith('WFM') and mode.endswith('-N')):
            print(f"[WARNING] MUSE observing mode is {mode or 'unknown'}. The spectral resolution "
                  f"formula was measured on wide-field nominal-range data. It may be less accurate here.")

        low, high = MUSE_LSF_RANGE_ANGSTROM
        wavelength_angstrom = float(wavelength) * 1.0e4
        if not low <= wavelength_angstrom <= high:
            # The formula was not measured at this wavelength. Use R at the nearest end and say so.
            end = low if wavelength_angstrom < low else high
            print(f"[WARNING] MUSE: {wavelength_angstrom:.1f} Angstrom is outside the measured range "
                  f"({low:.0f}-{high:.0f} Angstrom). Using R at {end:.0f} Angstrom instead.")
            wavelength_angstrom = end
        fwhm = np.polyval(MUSE_LSF_FWHM_COEFFS, wavelength_angstrom)
        return float(wavelength_angstrom / fwhm)

    elif instrument == 'NIRSPEC':
        if header is None:
            raise ValueError("NIRSpec requires FITS header for grating information")

        grating = header.get('GRATING', '').upper()
        if grating not in NIRSPEC_GRATINGS:
            raise ValueError(f"Unknown NIRSpec grating: {grating}")
        if not np.isfinite(wavelength):
            raise ValueError(f"NIRSpec wavelength must be finite, got {wavelength}.")

        table_wavelength, table_resolution = _load_nirspec_curve(grating)
        low, high = table_wavelength[0], table_wavelength[-1]
        if not low <= wavelength <= high:
            # The table does not reach this wavelength. Use R at the nearest end and say so.
            end = low if wavelength < low else high
            print(f"[WARNING] NIRSpec {grating}: {wavelength:.4f} um is outside the resolving-power "
                  f"table ({low:.3f}-{high:.3f} um). Using R at {end:.3f} um instead.")
        return float(np.interp(wavelength, table_wavelength, table_resolution))

    else:
        raise ValueError(f"Instrument {instrument} not yet supported for spectral resolution")


def infer_spec_axis(header, cube_shape):
    """
    Find which axis holds the spectral data, using the header's NAXIS3 value.

    Args:
        header (astropy.io.fits.Header): FITS header for the cube
        cube_shape (tuple): Shape of the loaded data cube

    Returns:
        axis (int): Index of the spectral axis (0 = first, 2 = last, for a 3D cube).
            Returns 0 if the header does not give a clear answer.
    """
    try:
        naxis3 = header.get('NAXIS3', None)
        if naxis3 is not None:
            if naxis3 == cube_shape[0]:
                return 0
            elif naxis3 == cube_shape[-1]:
                return len(cube_shape) - 1
    except Exception as e:
        print(f"Error inferring spectral axis: {e}")  # Debug info
    return 0


def get_wavelength(header, instrument, cube_shape):
    """
    Calculate the wavelength array from a FITS header. Returns Angstroms for MUSE and
    microns for MIRI/NIRSpec, and checks CTYPE3 for the axis type.

    Args:
        header (astropy.io.fits.Header): FITS header for the cube
        instrument (str): Detected instrument name
        cube_shape (tuple): Shape of the loaded data cube

    Returns:
        result (dict): Has 'wavelength' (Quantity) and 'unit' (str). Also has 'flux_unit'
            and 'wave_error' for MUSE, or 'flux_unit', 'mjd_avg', 'exposure_time', and
            'pa_aper' for MIRI/NIRSpec. The caller sets these extra keys onto the cube
            with setattr. On error, returns a dummy dict instead of raising.
    """

    try:
        # Handle a genuine single-slice cube (NAXIS3 == 1) - dummy wavelength
        if cube_shape[0] == 1:
            print("Single-slice cube detected: creating single dummy wavelength point")
            return {'wavelength': np.array([0.0]) * u.dimensionless_unscaled, 'unit': "pixel"}

        # Get axis type
        ctype3 = header.get("CTYPE3", "").strip().upper()

        if instrument == "MUSE":
            # Verify MUSE wavelength axis configuration
            if (header.get("CTYPE3", "").strip().upper() == "AWAV" and
                header.get("CUNIT3", "").strip().upper() == "ANGSTROM"):

                # Get wavelength calibration
                crval3 = float(header["CRVAL3"])  # Starting wavelength
                crpix3 = float(header["CRPIX3"])  # Reference pixel
                cd33 = float(header["CD3_3"])     # Step size
                n_slices = int(header["NAXIS3"])  # Number of slices

                # Calculate wavelength array
                wavelengths = crval3 + (np.arange(1, n_slices + 1) - crpix3) * cd33

                extra = {'flux_unit': header.get("BUNIT", "")}
                if "CRDER3" in header:
                    extra['wave_error'] = float(header["CRDER3"])

                return {'wavelength': wavelengths * u.AA, 'unit': "Å", **extra}
            else:
                raise ValueError("Unexpected MUSE wavelength axis configuration")

        elif instrument in ["MIRI", "NIRSpec"]:
            # By this point, TELESCOPE_LOADERS already resolved `header` to the SCI
            # extension, or fell back to the primary HDU. No further lookup is needed here.
            crval3 = float(header["CRVAL3"])
            cdelt3 = float(header["CDELT3"])
            crpix3 = float(header["CRPIX3"])
            n_slices = int(header["NAXIS3"])

            # Check units and convert if needed
            cunit3 = header.get("CUNIT3", "").strip().upper()
            wavelengths = crval3 + (np.arange(1, n_slices + 1) - crpix3) * cdelt3

            extra = {
                # Store MJD for potential time-series analysis
                'mjd_avg': float(header.get("MJD-AVG", 0.0)),
                'exposure_time': float(header.get("XPOSURE", 0.0)),
                # Store flux units (typically MJy/sr for JWST)
                'flux_unit': header.get("BUNIT", "MJy/sr"),
                # Store aperture position angle for orientation
                'pa_aper': float(header.get("PA_APER", 0.0)),
            }

            # Convert wavelengths to microns if needed, via the shared
            # unit-conversion module rather than ad hoc factors.
            if cunit3 == "ANGSTROM" or "ANGSTROM" in ctype3:
                source_unit = "angstrom"
            elif cunit3 == "NM" or "NANOMETER" in ctype3:
                source_unit = "nm"
            elif cunit3 == "M":
                source_unit = "m"
            else:
                source_unit = "um"
            wavelengths = convert_wavelength(wavelengths, source_unit, "um")

            return {'wavelength': wavelengths * u.um, 'unit': "um", **extra}

    except Exception as e:
        print(f"Error calculating wavelength array: {e}")
        return {'wavelength': None * u.dimensionless_unscaled, 'unit': ""}


