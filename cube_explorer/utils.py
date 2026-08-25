"""
Utility functions and constants for the Cube Explorer
"""

import numpy as np
import astropy.units as u
from .unit_converter import convert_wavelength


def _merge_primary_into_sci(hdul, sci_key):
    """
    Return SCI header with PRIMARY keywords added as fallback.

    Args:
        hdul (astropy.io.fits.Header) : FITS file header
        sci_key (int): index of SCI header
    
    Returns:
        sci_hdr (astropy.io.fits.Header): copy of SCI header

    Raises:
        Exception: If no key was found
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


"""
Where the main info lies on each telescope.
"""

TELESCOPE_LOADERS = {
    "MUSE": lambda hdul: (hdul[1].header, hdul[1].data),
    "MIRI": lambda hdul: (
        (_merge_primary_into_sci(hdul, "SCI"), hdul["SCI"].data)
        if "SCI" in hdul else (hdul[0].header, hdul[0].data)
    ),
    "NIRSpec": lambda hdul: (
        (_merge_primary_into_sci(hdul, "SCI"), hdul["SCI"].data)
        if "SCI" in hdul else (hdul[0].header, hdul[0].data)
    ),
}


def detect_instrument(hdul):
    """
    Detect the instrument/telescope used to obtain the data from FITS headers
    using header keywords.

    Args:
        hdul (astropy.io.fits.Header): FITS file header

    Returns:
        instrument (str): Name of detected instrument ('MUSE', 'MIRI', 'NIRSpec')
                          None if no known instrument is detected

    Note:
        - Checks all HDUs in case metadata is not in primary header
        - Uses case-insensitive matching for robustness
        - Currently supports MUSE, MIRI, and NIRSpec
        - Returns None rather than raising an exception if no match
    """

    for hdu in hdul:
        hdr = hdu.header
        instrument = hdr.get("INSTRUME", "").strip().upper()
        telescope = hdr.get("TELESCOP", "").strip().upper()

        # Match to known telescope keywords
        if "MUSE" in instrument or "MUSE" in telescope:
            return "MUSE"
        if "MIRI" in instrument or "MIRI" in telescope:
            return "MIRI"
        if "NIRSPEC" in instrument or "NIRSPEC" in telescope:
            return "NIRSpec"

    return None  # if no match found


def get_spectral_resolution(wavelength, instrument, header=None):
    """
    Calculate spectral resolution R(λ) = λ/Δλ for different instruments.

    Args:
        wavelength (float): wavelength in units
        instrument (str): instrument name ('NIRSPEC', 'MIRI')
        header (astropy.io.fits.Header): FITS file header. Required for NIRSpec to get grating info

    Returns:
        spectral resolution (float): R(λ)

    Raises:
        ValueError: If instrument cannot be detected or grating is invalid

    Formulas:
        MIRI: R(λ) = 4603 - 128 * λ
        NIRSpec (from grating):
            - G140H, G235H, G395H: R ~ 2700
            - G140M, G235M, G395M: R ~ 1000
            - PRISM: R ~ 100
    """
    
    instrument = instrument.upper()

    if instrument == 'MIRI':
        # MIRI spectral resolution formula
        return 4603 - 128 * wavelength

    elif instrument == 'NIRSPEC':
        if header is None:
            raise ValueError("NIRSpec requires FITS header for grating information")

        grating = header.get('GRATING', '').upper()

        # High resolution gratings: R ~ 2700
        if grating in ['G140H', 'G235H', 'G395H']:
            return 2700

        # Medium resolution gratings: R ~ 1000
        elif grating in ['G140M', 'G235M', 'G395M']:
            return 1000

        # Low resolution prism: R ~ 100
        elif grating == 'PRISM':
            return 100

        else:
            raise ValueError(f"Unknown NIRSpec grating: {grating}")

    else:
        raise ValueError(f"Instrument {instrument} not yet supported for spectral resolution")


def infer_spec_axis(header, cube_shape):
    """
    Determine which axis contains spectral information based on FITS header by using NAXIS3.

    Args:
        header (astropy.io.fits.Header): FITS header for the cube
        cube_shape (tuple): Shape of the loaded data cube

    Returns:
        int: Index of spectral axis (0 = first axis, 2 = last axis for 3D cube)
             Returns 0 by default if cannot determine from header.

    Raises:
        Exception: If any error occurs
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
    Calculate wavelength array from FITS header based on instrument type.

    Uses instrument-specific calculations and checks CTYPE3 to determine proper units:
    - MUSE: Wavelength in Angstroms (Å) as astropy Quantity
    - MIRI/NIRSpec: Wavelength in microns (µm) as astropy Quantity

    Args:
        header (astropy.io.fits.Header): FITS header for the cube
        instrument (str): Detected instrument name
        cube_shape (tuple): Shape of the loaded data cube

    Returns:
        dict: 'wavelength' (Quantity) and 'unit' (str), plus 'flux_unit'/'wave_error'
              for MUSE or 'flux_unit'/'mjd_avg'/'exposure_time'/'pa_aper' for
              MIRI/NIRSpec. Callers apply the extra keys onto the cube with setattr.

    Raises:
        ValueError: If wavelength is invalid
        Exception: If any other error occurs during calculations
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
            # Get wavelength solution. By this point TELESCOPE_LOADERS has
            # already resolved `header` to the SCI extension (or fallen back
            # to the primary HDU if there wasn't one), so no further lookup
            # is needed here.
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

            return {'wavelength': wavelengths * u.um, 'unit': "µm", **extra}

    except Exception as e:
        print(f"Error calculating wavelength array: {e}")
        return {'wavelength': None * u.dimensionless_unscaled, 'unit': ""}


