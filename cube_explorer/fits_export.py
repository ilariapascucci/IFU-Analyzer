"""
Shared FITS export helpers: a common 2D header (WCS + descriptive metadata)
and a common file writer, so every "save this map as FITS" site builds its
output the same way instead of repeating the same header-building logic.

GUI-free, mirrors analyzer_2d.py's and region_io.py's pattern. Only for 2D
derived products (slices, parameter/flux/temperature maps) -- a 3D cube
export (the region-cube export, the Gaussian residuals cube) needs its full
original header kept intact, not reduced to a clean 2D WCS, so those stay
separate rather than being forced through this module.
"""

from datetime import datetime, timezone

from astropy.io import fits

from .analyzer_2d import _make_2d_wcs

# Copied onto every 2D export unmodified. BUNIT is deliberately not here --
# every export site either keeps the cube's own unit or computes its own
# derived one, so it belongs in each call's own `extra`, not this shared set.
_COMMON_KEYWORDS = ('INSTRUME', 'TELESCOP', 'OBSERVER', 'OBJECT', 'DATE-OBS', 'EXPTIME')

# This app's own unit/results labels use Unicode glyphs (mu, superscripts,
# sigma, plus-minus...) for on-screen display; FITS header values/comments
# must be plain ASCII, so any such string needs this before it can be
# written. Individual characters, not fixed combos, so any combination
# (e.g. any superscript digit after a minus sign) comes out right. Aligned
# with gui_rotational_diagram.py's own console-text replacement chain,
# which is the confirmed set of symbols this app actually generates.
_ASCII_REPLACEMENTS = {
    "μ": "u", "µ": "u", "⁻": "-", "¹": "1", "²": "2", "³": "3", "°": "deg", "×": "x", "·": ".",
    "±": "+/-", "σ": "sigma", "≈": "~=", "⚠": "!",
}


def _ascii_safe(value):
    """Make a header value FITS-safe: plain ASCII only.

    Args:
        value: Any header value; non-strings pass through unchanged.

    Returns:
        value: The same value, or an ASCII-safe copy if it was a string.
    """
    if not isinstance(value, str):
        return value
    for bad, good in _ASCII_REPLACEMENTS.items():
        value = value.replace(bad, good)
    return value.encode("ascii", "replace").decode("ascii")


def _ascii_safe_card(value):
    """Apply _ascii_safe to a header value, or to both halves of a
    (value, comment) card tuple.

    Args:
        value: A plain header value, or a (value, comment) tuple.

    Returns:
        value: The same shape, ASCII-safe.
    """
    if isinstance(value, tuple) and len(value) == 2:
        return _ascii_safe(value[0]), _ascii_safe(value[1])
    return _ascii_safe(value)


def build_2d_export_header(source_header, data_shape, extra=None):
    """
    Build a standalone 2D FITS header: a clean WCS from source_header, plus
    the descriptive keywords every 2D export shares.

    Args:
        source_header (astropy.io.fits.Header or None): Header of the cube,
            or an already-built WCS's own to_header(), for this product.
        data_shape (tuple): (ny, nx) of the 2D data this header describes.
        extra (dict, optional): Export-specific keywords to add, as
            {key: value} or {key: (value, comment)}. Not for COMMENT/HISTORY
            -- FITS allows more than one of those, but a dict can only hold
            one value per key, so add those to the returned header directly.

    Returns:
        header (astropy.io.fits.Header): Ready for fits.PrimaryHDU/ImageHDU.
    """
    wcs2d = _make_2d_wcs(source_header) if source_header is not None else None
    header = wcs2d.to_header() if wcs2d is not None else fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = int(data_shape[1])
    header['NAXIS2'] = int(data_shape[0])

    if source_header is not None:
        for kw in _COMMON_KEYWORDS:
            if kw in source_header:
                header[kw] = _ascii_safe(source_header[kw])

    header['ORIGIN'] = 'IFU Analyzer'
    header['DATE'] = (datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'), 'File creation date (UTC)')

    if extra:
        for key, value in extra.items():
            header[key] = _ascii_safe_card(value)

    return header


def write_fits_product(path, primary_data, primary_header, extensions=None, output_verify='exception'):
    """
    Write a PRIMARY HDU plus optional named extensions to path.

    Args:
        path (str): Destination FITS path.
        primary_data (numpy.ndarray or None): Data for the PRIMARY HDU, or
            None for an empty primary (e.g. every real array is its own
            named extension instead).
        primary_header (astropy.io.fits.Header): Header for the PRIMARY HDU.
        extensions (list, optional): (name, data, header) tuples, one per
            extra ImageHDU. header may be None for a blank one.
        output_verify (str): astropy's writeto output_verify mode.

    Returns:
        None: Writes path as a side effect.
    """
    hdul = fits.HDUList([fits.PrimaryHDU(data=primary_data, header=primary_header)])
    for name, data, header in (extensions or []):
        ext_hdu = fits.ImageHDU(data=data, header=header if header is not None else fits.Header())
        ext_hdu.header['EXTNAME'] = name  # set last -- header's own EXTNAME (if any) must not win
        hdul.append(ext_hdu)
    hdul.writeto(path, overwrite=True, output_verify=output_verify)
