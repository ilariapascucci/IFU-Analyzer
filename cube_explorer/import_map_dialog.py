"""
Header-peeking helpers for Import Map: best-effort species/wavelength/
identity lookup and real-error detection for an externally-produced 2D flux
map (FITS), so gui_rotational_diagram.py's import_map_line can decide what
(if anything) it still needs to ask the user for one file at a time.
"""

from astropy.io import fits

_IDENTITY_KEYWORDS = ("MOLECULE", "LINEWAVE", "A_UL", "E_UPPER", "G_U",
                      "TRANSITN", "JLOWER", "BRANCH")


def _peek_header(path):
    """Best-effort peek at a FITS header for round-tripped identity keywords.
    Never raises -- this is a cosmetic shortcut, not validation; the real
    read happens on import via build_imported_map_payload.

    Args:
        path (str): FITS file path.

    Returns:
        found (dict): Whichever of _IDENTITY_KEYWORDS are present.
    """
    try:
        with fits.open(path) as hdul:
            found = {}
            for hdu in hdul:
                for kw in _IDENTITY_KEYWORDS:
                    if kw not in found and kw in hdu.header:
                        found[kw] = hdu.header[kw]
            return found
    except Exception:
        return {}


def _has_real_error(path):
    """Check for a real per-pixel ERR extension matching this file's own 2D
    image shape, so the caller only asks for a flat error percent when
    there's genuinely no error data to fall back on. Never raises.

    Args:
        path (str): FITS file path.

    Returns:
        has_err (bool): True if a shape-matching ERR extension was found.
    """
    try:
        with fits.open(path) as hdul:
            data_shape = None
            for hdu in hdul:
                if hdu.data is not None and getattr(hdu.data, "size", 0) > 0:
                    squeezed = hdu.data.squeeze()
                    if squeezed.ndim == 2:
                        data_shape = squeezed.shape
                        break
            if data_shape is None:
                return False
            for hdu in hdul:
                if (getattr(hdu, "name", "") or "").strip().upper() == "ERR" and hdu.data is not None:
                    if hdu.data.squeeze().shape == data_shape:
                        return True
            return False
    except Exception:
        return False
