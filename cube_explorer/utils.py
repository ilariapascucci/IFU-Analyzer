"""
Utility functions and constants for the Cube Explorer
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import rotate
from .unit_converter import convert_wavelength, sky_wcs_problem

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

# Resolution clamp warnings kept since the last take_resolution_notes() call.
_resolution_notes = []


def _report_resolution_clamp(message):
    """
    Print a resolution clamp warning to the console and keep it for the GUI.
    The GUI shows the kept warnings when a run ends.
    The same message is kept one time only.

    Args:
        message (str): Which wavelength is outside the range, and which R is used instead

    Returns:
        None
    """
    print(f"[WARNING] {message}")
    if message not in _resolution_notes:
        _resolution_notes.append(message)


def take_resolution_notes():
    """
    Return the kept resolution clamp warnings as one text, and forget them.
    A GUI run calls this when it ends, to show the text in a warning dialog.
    A GUI run also calls it when it starts, to drop the warnings of an earlier run.

    Args:
        None

    Returns:
        note (str): Text for a warning dialog, or an empty string if nothing was kept
    """
    if not _resolution_notes:
        return ""
    lines = "\n".join(f"- {message}" for message in _resolution_notes)
    _resolution_notes.clear()
    return ("The spectral resolution is not known at some wavelengths of this run. "
            "R was held at the value of the nearest end, so the widths and windows there "
            f"are approximate:\n\n{lines}")


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
            _report_resolution_clamp(
                f"MUSE: {wavelength_angstrom:.1f} Angstrom is outside the measured range "
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
            _report_resolution_clamp(
                f"NIRSpec {grating}: {wavelength:.4f} um is outside the resolving-power "
                f"table ({low:.3f}-{high:.3f} um). Using R at {end:.3f} um instead.")
        return float(np.interp(wavelength, table_wavelength, table_resolution))

    else:
        raise ValueError(f"Instrument {instrument} not yet supported for spectral resolution")


def resolving_power(wavelength, wave_unit, instrument, header=None):
    """
    Give the spectral resolution R(lambda) at a wavelength that is in any unit of length.
    The wavelength is turned into micron first, because get_spectral_resolution works in micron.

    Args:
        wavelength (float): Wavelength, in wave_unit
        wave_unit (str): Unit of the wavelength, for example 'um' or 'Angstrom'
        instrument (str): Instrument name ('NIRSPEC', 'MIRI' or 'MUSE')
        header (astropy.io.fits.Header): FITS header. Required for NIRSpec, to read the grating.

    Returns:
        resolution (float): Spectral resolution R(lambda) at this wavelength

    Raises:
        ValueError: If wave_unit is not a unit of length, or for the errors of get_spectral_resolution
    """
    to_um = float(convert_wavelength(1.0, wave_unit, 'um'))
    return get_spectral_resolution(wavelength * to_um, instrument, header)


def psf_fwhm_arcsec(wavelength, instrument):
    """
    Calculate the FWHM of the point spread function, in arcsec, at a wavelength.
    MIRI uses a straight line in wavelength. NIRSpec uses an approximate quadratic fit.
    Other instruments have no PSF formula here, so they raise an error.

    Args:
        wavelength (float): Wavelength, in microns
        instrument (str): Instrument name ('MIRI' or 'NIRSpec')

    Returns:
        fwhm (float): PSF FWHM, in arcsec

    Raises:
        ValueError: If the instrument has no PSF formula

    Formulas:
        MIRI: FWHM(") = 0.033 * lambda + 0.106
        NIRSpec: FWHM(") = 0.1 * (0.031406393 * lambda^2 + 0.019719141 * lambda + 1.5167115).
            The fit gives the FWHM in pixels. NIRSpec has a constant scale of 0.1 arcsec per pixel.
            It is an approximate fit, and a higher order polynomial might fit better.
    """
    name = str(instrument).upper()

    if name == 'MIRI':
        return 0.033 * wavelength + 0.106

    elif name == 'NIRSPEC':
        return 0.1 * (0.031406393 * wavelength ** 2 + 0.019719141 * wavelength + 1.5167115)

    else:
        raise ValueError(f"No PSF formula for instrument {instrument}. Only MIRI and NIRSpec are supported.")


def infer_spec_axis(header, cube_shape):
    """
    Find which axis holds the spectral data, using the header's NAXIS3 value.

    Args:
        header (astropy.io.fits.Header): FITS header for the cube
        cube_shape (tuple): Shape of the loaded data cube

    Returns:
        axis (int): Index of the spectral axis (0 = first, 2 = last, for a 3D cube)

    Raises:
        ValueError: The header has no NAXIS3, or NAXIS3 is not the length of the first or the last axis
    """
    naxis3 = header.get('NAXIS3', None)
    if naxis3 is None:
        raise ValueError("The header has no NAXIS3, so the spectral axis cannot be found.")
    if naxis3 == cube_shape[0]:
        return 0
    if naxis3 == cube_shape[-1]:
        return len(cube_shape) - 1
    raise ValueError(f"NAXIS3 is {naxis3}, but the cube has the shape {tuple(cube_shape)}. "
                     "The spectral axis cannot be found.")


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
            with setattr.

    Raises:
        ValueError: The header lacks a keyword of the wavelength axis, or the axis or its unit is not valid.
            A missing CUNIT3 is an error. A missing BUNIT is not: the flux unit is then an empty string.
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
                # An empty text means the flux unit is not known. No unit is assumed.
                'flux_unit': str(header.get("BUNIT", "")).strip(),
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
            elif cunit3 in ("UM", "MICRON", "MICRONS", "MICROMETER", "MICROMETERS"):
                source_unit = "um"
            elif not cunit3:
                raise ValueError("The header has no CUNIT3, so the wavelength unit is not known.")
            else:
                raise ValueError(f"CUNIT3 is '{cunit3.lower()}'. This program reads um, nm, Angstrom and m.")
            wavelengths = convert_wavelength(wavelengths, source_unit, "um")

            return {'wavelength': wavelengths * u.um, 'unit': "um", **extra}

    except Exception as e:
        raise ValueError(f"The wavelength axis of the cube is incomplete or not valid: {e}") from e


# --- 2D WCS and BUNIT helpers ---

def get_pixel_size_arcsec(header):
    """Return the celestial pixel scale (arcsec/pixel) from a FITS header.
    Uses proj_plane_pixel_scales so rotated CD/PC matrices (e.g. JWST IFU)
    are handled correctly; falls back to raw CDELT if the WCS fails.

    Args:
        header (astropy.io.fits.Header): FITS header to read the WCS from.

    Returns:
        scale (float): Pixel scale in arcsec/pixel, or None if the header has no complete sky coordinates.
    """
    # A pixel scale from an incomplete WCS would be a default value, for example 3600 arcsec.
    if sky_wcs_problem(header) is not None:
        return None
    try:
        w = WCS(header)
        if w.has_celestial:
            return abs(float(proj_plane_pixel_scales(w.celestial)[0])) * 3600.0
    except Exception:
        pass
    for k in ('CDELT1', 'CD1_1'):
        if k in header:
            try:
                return abs(float(header[k])) * 3600.0
            except Exception:
                pass
    return None


def make_2d_wcs(header):
    """Build a clean 2-D celestial WCS from a header (drops trivial axes).

    Args:
        header (astropy.io.fits.Header): FITS header to build the WCS from.

    Returns:
        wcs (astropy.wcs.WCS): The 2D WCS, or None if it could not be built. A header without
            complete sky coordinates gives a plain pixel WCS, and its has_celestial is False.
    """
    try:
        if sky_wcs_problem(header) is not None:
            # Astropy would fill the missing keywords with defaults and report false sky positions.
            return WCS(naxis=2)
        w = WCS(header)
        if w.pixel_n_dim > 2 and w.has_celestial:
            return w.celestial
        while w.pixel_n_dim > 2:
            w = w.dropaxis(w.pixel_n_dim - 1)
        return w
    except Exception:
        return None


def rotate_image_with_wcs(image, wcs_in, total_angle):
    """Turn a 2D image and its WCS together, so the sky position of every pixel stays the same.

    The array grows to hold the turned image, and the new pixels are NaN. The 2D viewer and the
    superimposition tables both use this function. A positive angle turns the image clockwise as drawn
    (origin='lower').

    Args:
        image (numpy.ndarray): 2D array of pixel values, with the image as it was loaded.
        wcs_in (astropy.wcs.WCS): 2D celestial WCS of image.
        total_angle (float): Rotation in degrees from the loaded image. A positive value is clockwise.

    Returns:
        array_out (numpy.ndarray): The turned image, with the new shape.
        wcs_out (astropy.wcs.WCS): The WCS of array_out. wcs_in does not change.
    """
    wcs_out = wcs_in.deepcopy()
    theta = np.deg2rad(total_angle)

    array_out = rotate(image, total_angle, reshape=True, cval=np.nan, order=1)

    # Both rotation signs here were checked against a point source with a known sky position.
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R_inv = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    if hasattr(wcs_out.wcs, 'cd'):
        wcs_out.wcs.cd = np.dot(wcs_out.wcs.cd, R_inv)
    elif hasattr(wcs_out.wcs, 'pc'):
        wcs_out.wcs.pc = np.dot(wcs_out.wcs.pc, R_inv)
    else:
        wcs_out.wcs.pc = R_inv

    ny, nx = image.shape
    new_ny, new_nx = array_out.shape

    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    new_cx, new_cy = (new_nx - 1) / 2.0, (new_ny - 1) / 2.0

    crpix_0 = wcs_in.wcs.crpix - 1
    dx, dy = crpix_0[0] - cx, crpix_0[1] - cy

    cos_r, sin_r = np.cos(-theta), np.sin(-theta)
    new_dx = dx * cos_r - dy * sin_r
    new_dy = dx * sin_r + dy * cos_r

    wcs_out.wcs.crpix = np.array([new_cx + new_dx, new_cy + new_dy]) + 1
    return array_out, wcs_out


def rotate_image_with_header(image, header, total_angle):
    """Turn a 2D image and write the matching header, like the Rotate button of the 2D viewer.

    The WCS keywords of the header are replaced with the keywords of the turned WCS, and the
    size keywords follow the new array. The other keywords stay. The 2D viewer calls this
    function, and so does the API function rotate_image.

    Args:
        image (numpy.ndarray): 2D array of pixel values, with the image as it was loaded.
        header (astropy.io.fits.Header): FITS header of image, with complete sky coordinates.
        total_angle (float): Rotation in degrees from the loaded image. A positive value is clockwise.

    Returns:
        array_out (numpy.ndarray): The turned image, with the new shape.
        header_out (astropy.io.fits.Header): A copy of header for array_out. header does not change.

    Raises:
        ValueError: The header has no complete sky coordinates.
    """
    problem = sky_wcs_problem(header)
    if problem is not None:
        raise ValueError(f"Rotate needs complete sky coordinates. {problem}")
    wcs_in_full = WCS(header)

    if wcs_in_full.pixel_n_dim > 2:
        wcs_in = wcs_in_full.celestial
    else:
        wcs_in = wcs_in_full

    array_out, wcs_out = rotate_image_with_wcs(image, wcs_in, total_angle)
    new_ny, new_nx = array_out.shape

    new_header = header.copy()
    for key in list(wcs_in_full.to_header().keys()):
        if key in new_header:
            del new_header[key]

    wcs_header = wcs_out.to_header()
    for key in wcs_header:
        new_header[key] = wcs_header[key]

    new_header['NAXIS'] = 2
    new_header['NAXIS1'] = new_nx
    new_header['NAXIS2'] = new_ny
    if 'NAXIS3' in new_header:
        del new_header['NAXIS3']
    if 'WCSAXES' in new_header:
        new_header['WCSAXES'] = 2
    return array_out, new_header


class CubeFileError(ValueError):
    """The FITS file holds a data cube and no 2D image. Callers can show their own advice for it."""


def select_2d_image(hdul, hdu_index=None):
    """Pick the first 2D image of an open FITS file, with its header and coverage map.

    Without an HDU index, the first HDU with data that squeezes to 2D is used. The header loses
    the third axis keywords of JWST files. The data is not copied, so it is valid only while
    the file is open.

    Args:
        hdul (astropy.io.fits.HDUList): Open FITS file.
        hdu_index (int): Index of the HDU to use, or None for the first 2D image.

    Returns:
        found (dict): The keys image (numpy.ndarray), header (astropy.io.fits.Header),
            coverage (numpy.ndarray or None) and hdu_index (int).

    Raises:
        CubeFileError: No HDU holds a 2D image, and one holds a data cube.
        ValueError: No HDU holds image data.
    """
    data = None
    header = None
    coverage = None
    loaded_index = None
    # Accurately find the correct 2D extension (fixes JWST empty primary HDUs)
    for index, hdu in enumerate(hdul):
        if hdu_index is not None and index != hdu_index:
            continue
        if hdu.data is not None and getattr(hdu.data, 'size', 0) > 0:
            sq_data = np.squeeze(hdu.data)
            if sq_data.ndim == 2:
                data = sq_data
                header = hdu.header.copy()
                loaded_index = index
                break

    # "COVERAGE" extension holds the per-pixel coverage fraction (0-1)
    # from the reproject pipeline, used to weight partial-overlap pixels.
    for hdu in hdul:
        name = (getattr(hdu, 'name', '') or '').strip().upper()
        if name == 'COVERAGE' and hdu.data is not None:
            try:
                cov_arr = np.squeeze(hdu.data).astype(float)
                if data is not None and cov_arr.shape == data.shape:
                    coverage = cov_arr
                    break
            except Exception:
                pass

    if data is None:
        # No HDU squeezed to 2D; distinguish a genuine multi-plane
        # cube from a file with no image data.
        cube_shape = None
        for hdu in hdul:
            if hdu.data is None or getattr(hdu.data, 'size', 0) == 0:
                continue
            sq = np.squeeze(hdu.data)
            if sq.ndim > 2:
                cube_shape = tuple(sq.shape)
                break
        if cube_shape is not None:
            raise CubeFileError(f"This file holds a {cube_shape} data cube, not a 2D image.")
        raise ValueError("No image data found in FITS file.")

    # data is the squeezed 2D array from the loop above (degenerate
    # axes accepted); true multi-plane cubes are rejected earlier.

    # Remove dangling 3D JWST keys
    header['NAXIS'] = 2
    if 'WCSAXES' in header:
        header['WCSAXES'] = 2

    bad_keys = ['NAXIS3', 'NAXIS4','CTYPE3', 'CRVAL3', 'CRPIX3', 'CDELT3', 'CUNIT3',
                'PC1_3', 'PC2_3', 'PC3_1', 'PC3_2', 'PC3_3',
                'CD1_3', 'CD2_3', 'CD3_1', 'CD3_2', 'CD3_3']
    for k in bad_keys:
        if k in header:
            del header[k]
    return {"image": data, "header": header, "coverage": coverage, "hdu_index": loaded_index}


def read_2d_fits(path, hdu_index=None):
    """Read one 2D image of a FITS file into memory, with its header and coverage map.

    This is select_2d_image on a file that this function opens and closes. The arrays are
    copies, so they stay valid after the file is closed.

    Args:
        path (str): Path of the FITS file.
        hdu_index (int): Index of the HDU to use, or None for the first 2D image.

    Returns:
        found (dict): The keys image (numpy.ndarray), header (astropy.io.fits.Header),
            coverage (numpy.ndarray or None) and hdu_index (int).

    Raises:
        CubeFileError: The file holds a data cube and no 2D image.
        ValueError: The file holds no image data.
        OSError: The file cannot be read as FITS.
    """
    with fits.open(path) as hdul:
        found = select_2d_image(hdul, hdu_index)
        found["image"] = np.array(found["image"])
        if found["coverage"] is not None:
            found["coverage"] = np.array(found["coverage"])
    return found



def contour_levels(image, levels_text, sigma_text):
    """Turn the level entries into the pixel values where contours are drawn.

    A number with a % sign is a percent of the peak: p / 100 * the largest finite pixel.
    Any other number N is a multiple of sigma: median + N * sigma. The median and the sigma
    come from a sigma-clipped fit of the image. A typed sigma replaces the fitted sigma.

    Args:
        image (numpy.ndarray): 2D array of pixel values.
        levels_text (str): Comma-separated levels, for example '3, 5, 50%'.
        sigma_text (str): Sigma as text, or '' for the fitted value. Not used when
            every level is a percent.

    Returns:
        levels (list): The pixel values, sorted from low to high.

    Raises:
        ValueError: A level or the sigma is not a number, no level is given, a percent is
            not above 0, the image has no finite pixel, a percent level needs a peak that
            is not above 0, or the sigma is not above 0.
    """
    entries = [x.strip() for x in levels_text.split(",") if x.strip()]
    if not entries:
        raise ValueError("Enter at least one level.")
    multiples, percents = [], []
    for entry in entries:
        is_percent = entry.endswith("%")
        try:
            number = float(entry[:-1] if is_percent else entry)
        except ValueError:
            number = float("nan")
        if not np.isfinite(number):
            raise ValueError("Levels must be comma-separated numbers, for example 3, 5, 50%.")
        if is_percent and not number > 0:
            raise ValueError("A percent level must be above 0%.")
        (percents if is_percent else multiples).append(number)
    valid = image[np.isfinite(image)]
    if valid.size == 0:
        raise ValueError("The source image has no finite pixel.")
    levels = []
    if percents:
        peak = valid.max()
        if not peak > 0:
            raise ValueError("A percent level needs a source image with a peak above 0.")
        levels += [percent / 100 * peak for percent in percents]
    if multiples:
        _, median_val, fitted_std = sigma_clipped_stats(valid, sigma=3.0)
        if sigma_text.strip():
            try:
                sigma = float(sigma_text)
            except ValueError:
                raise ValueError("Sigma must be a number, or empty for the automatic value.")
        else:
            sigma = fitted_std
        if not sigma > 0:
            raise ValueError("Sigma must be above 0.")
        levels += [median_val + n * sigma for n in multiples]
    return sorted(levels)


def levels_inside_data(image, levels):
    """Split contour levels into the ones that have a line and the ones that have none.

    A level has a line only when it is between the lowest and the highest finite pixel.
    Astropy fails on a contour set with no level inside the data.

    Args:
        image (numpy.ndarray): 2D array of pixel values, with at least one finite pixel.
        levels (list): Pixel values, as contour_levels gives them.

    Returns:
        inside (list): The levels between the lowest and the highest finite pixel.
        skipped (str): The other levels as text, for example '1.000e-05, 2.500e+03'. It is
            empty when no level is skipped.
    """
    finite = image[np.isfinite(image)]
    inside = [level for level in levels if finite.min() < level < finite.max()]
    skipped = ", ".join(f"{level:.3e}" for level in levels if level not in inside)
    return inside, skipped


def viewer_wcs(header):
    """Build the 2D WCS that the 2D viewer uses for a panel, from the header of its image.

    The third axis keywords and the distortion keywords are removed first, because they
    break the axes of the plot. A header with incomplete sky coordinates gives a plain
    pixel WCS, and the problem is returned as text.

    Args:
        header (astropy.io.fits.Header): FITS header of a 2D image.

    Returns:
        wcs (astropy.wcs.WCS): The WCS. A plain pixel WCS when the sky coordinates are not complete.
        problem (str): What is missing in the sky coordinates, or None when they are complete.
    """
    ch = header.copy()
    ch['NAXIS']   = 2
    ch['WCSAXES'] = 2

    # Remove every 3-D / spectral / distortion key that breaks WCSAxes
    bad_keys = [
        'NAXIS3', 'NAXIS4',
        'CTYPE3', 'CRVAL3', 'CRPIX3', 'CDELT3', 'CUNIT3',
        'PC1_3',  'PC2_3',  'PC3_1',  'PC3_2',  'PC3_3',
        'CD1_3',  'CD2_3',  'CD3_1',  'CD3_2',  'CD3_3',
        'A_ORDER','B_ORDER','AP_ORDER','BP_ORDER',
    ]
    # Dynamically collect SIP coefficient keys (A_i_j, B_i_j, AP_i_j, BP_i_j)
    sip_keys = [k for k in ch.keys()
                if k.startswith(('A_', 'B_', 'AP_', 'BP_')) and k != 'BUNIT']
    bad_keys.extend(sip_keys)
    for k in bad_keys:
        if k in ch:
            del ch[k]

    # Strip '-SIP' suffix from CTYPE1 / CTYPE2 if present
    for ct in ('CTYPE1', 'CTYPE2'):
        if ct in ch and '-SIP' in str(ch[ct]):
            ch[ct] = str(ch[ct]).replace('-SIP', '')

    problem = sky_wcs_problem(ch)
    if problem is None:
        wcs_obj = WCS(ch, naxis=2)
        wcs_for_projection = wcs_obj.celestial if wcs_obj.has_celestial else wcs_obj
    else:
        # Plain pixel axes. Astropy would fill the missing keywords with default values.
        wcs_for_projection = WCS(naxis=2)
    return wcs_for_projection, problem


# --- BUNIT helper: per-pixel-flux detection ---

def is_per_pixel_flux_unit(bunit):
    """Heuristic: does this BUNIT indicate per-pixel total flux (vs SB)?
    Per-pixel units (erg/s/cm^2, counts, DN, electrons, plain Jy) return True.
    Surface-brightness units (MJy/sr, Jy/beam) return False.

    Args:
        bunit (str): FITS BUNIT header value, or None.

    Returns:
        is_flux (bool): True if bunit looks like a per-pixel flux unit.
    """
    if bunit is None:
        return False
    s = str(bunit).strip().lower().replace(' ', '')
    sb_markers = ['/sr', 'sr-1', 'sr^-1',
                  '/arcsec2', 'arcsec-2', 'arcsec^-2',
                  '/beam', 'beam-1', 'beam^-1',
                  '/pix', 'pix-1', 'pix^-1',
                  '/deg2', 'deg-2']
    if any(m in s for m in sb_markers):
        return False
    per_pix_markers = [
        'erg/(s*cm2)', 'erg/s/cm2', 'erg.s-1.cm-2', 'ergs-1cm-2',
        'jy', 'mjy', 'ujy', 'njy',
        'counts', 'count', 'cts', 'ct',
        'dn', 'electrons', 'electron', 'e-', 'photons', 'photon',
        'w/m2', 'wm-2', 'w.m-2',
    ]
    return any(s == m or s.endswith(m) for m in per_pix_markers)


# --- WCS footprint overlap check ---

def wcs_footprints_overlap(wcs1, shape1, wcs2, shape2):
    """Return True if the two image footprints share any sky coverage.
    Projects each image's corners and centre into the other's pixel frame
    (covers partial overlap, containment, and rotated/thin-strip cases).

    Args:
        wcs1 (astropy.wcs.WCS): WCS of the first image.
        shape1 (tuple): (ny, nx) shape of the first image.
        wcs2 (astropy.wcs.WCS): WCS of the second image.
        shape2 (tuple): (ny, nx) shape of the second image.

    Returns:
        overlap (bool): True if any sample point falls inside the other image

    Raises:
        ValueError: A WCS cannot be evaluated, so the sky coverage of the two images is not known
    """
    try:
        ny1, nx1 = shape1
        ny2, nx2 = shape2

        # Sample 5 points per image: 4 corners + centre
        xs1 = np.array([-0.5, nx1 - 0.5, nx1 - 0.5, -0.5, (nx1 - 1) / 2.0])
        ys1 = np.array([-0.5, -0.5,      ny1 - 0.5, ny1 - 0.5, (ny1 - 1) / 2.0])
        xs2 = np.array([-0.5, nx2 - 0.5, nx2 - 0.5, -0.5, (nx2 - 1) / 2.0])
        ys2 = np.array([-0.5, -0.5,      ny2 - 0.5, ny2 - 0.5, (ny2 - 1) / 2.0])

        # Project image-1 points into image-2 pixel space
        sky1 = wcs1.pixel_to_world(xs1, ys1)
        px_1in2, py_1in2 = wcs2.world_to_pixel(sky1)
        px_1in2 = np.asarray(px_1in2, dtype=float)
        py_1in2 = np.asarray(py_1in2, dtype=float)
        if np.any(
            np.isfinite(px_1in2) & np.isfinite(py_1in2)
            & (px_1in2 >= -0.5) & (px_1in2 <= nx2 - 0.5)
            & (py_1in2 >= -0.5) & (py_1in2 <= ny2 - 0.5)
        ):
            return True

        # Project image-2 points into image-1 pixel space
        sky2 = wcs2.pixel_to_world(xs2, ys2)
        px_2in1, py_2in1 = wcs1.world_to_pixel(sky2)
        px_2in1 = np.asarray(px_2in1, dtype=float)
        py_2in1 = np.asarray(py_2in1, dtype=float)
        if np.any(
            np.isfinite(px_2in1) & np.isfinite(py_2in1)
            & (px_2in1 >= -0.5) & (px_2in1 <= nx1 - 0.5)
            & (py_2in1 >= -0.5) & (py_2in1 <= ny1 - 0.5)
        ):
            return True

        return False
    except Exception as error:
        raise ValueError(f"The sky coverage of the two images is not known, because a WCS cannot be "
                         f"evaluated: {error}") from error
