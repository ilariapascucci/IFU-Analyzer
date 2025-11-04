"""
Utility functions and constants for the Cube Explorer
"""

import numpy as np

# Dictionary mapping telescope/instrument names to their data loading functions
# Each function takes an HDUList and returns (header, data) tuple
# The returned header contains metadata and WCS info
# The returned data is the 3D data cube

TELESCOPE_LOADERS = {
    "MUSE": lambda hdul: (hdul[1].header, hdul[1].data),
    "ALMA": lambda hdul: (hdul[0].header, hdul[0].data),
    "MIRI": lambda hdul: (hdul["SCI"].header, hdul["SCI"].data),
    "NIRSpec": lambda hdul: (hdul["SCI"].header, hdul["SCI"].data),
}

def detect_instrument(hdul):
    """
    Detect the instrument/telescope used to obtain the data from FITS headers.

    Searches through all HDUs in the file looking for INSTRUME or TELESCOP
    keywords. Matches against known instruments in a case-insensitive way.

    Args:
        hdul (astropy.io.fits.HDUList): Open FITS file HDU list

    Returns:
        str or None: Name of detected instrument ('MUSE', 'ALMA', 'MIRI', 'NIRSpec')
                    None if no known instrument is detected

    Note:
        - Checks all HDUs in case metadata is not in primary header
        - Uses case-insensitive matching for robustness
        - Currently supports MUSE, ALMA, MIRI, and NIRSpec
        - Returns None rather than raising an exception if no match
    """
    for hdu in hdul:
        hdr = hdu.header
        instrument = hdr.get("INSTRUME", "").strip().upper()
        telescope = hdr.get("TELESCOP", "").strip().upper()

        # Match to known telescope keywords
        if "MUSE" in instrument or "MUSE" in telescope:
            return "MUSE"
        if "ALMA" in instrument or "ALMA" in telescope:
            return "ALMA"
        if "MIRI" in instrument or "MIRI" in telescope:
            return "MIRI"
        if "NIRSPEC" in instrument or "NIRSPEC" in telescope:
            return "NIRSpec"

    return None  # if no match found