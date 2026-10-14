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

import numpy as np
from astropy.io import fits

from .analyzer_2d import _make_2d_wcs

# Copied onto every 2D export unmodified. BUNIT is deliberately not here --
# every export site either keeps the cube's own unit or computes its own
# derived one, so it belongs in each call's own `extra`, not this shared set.
# TARGNAME/CHANNEL/BAND/DETECTOR are JWST-pipeline keywords (target/channel
# identity); same graceful-miss handling as the rest of this tuple.
_COMMON_KEYWORDS = ('INSTRUME', 'TELESCOP', 'OBSERVER', 'OBJECT', 'DATE-OBS', 'EXPTIME',
                     'TARGNAME', 'CHANNEL', 'BAND', 'DETECTOR')

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

    Raises:
        ValueError: if source_header is None or has no sky (RA/Dec) WCS. The 2D viewer
            cannot open a file without one, so the export stops here instead.
    """
    wcs2d = _make_2d_wcs(source_header) if source_header is not None else None
    if wcs2d is None or not wcs2d.has_celestial:
        raise ValueError(
            "Cannot write a 2D FITS export: the source header has no sky (RA/Dec) WCS. "
            "Only data with a sky WCS, such as MIRI, NIRSpec and MUSE cubes, can be exported.")
    header = wcs2d.to_header()
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


def build_line_map_header(source_header, shape, wavelength=None, molecule=None, flux_map=None,
                          error_map=None, src_filename=None, fit_method=None,
                          identity_header=None, extra=None):
    """
    Shared header for any per-line flux/fit-parameter 2D map -- Rotational
    Diagram, Gaussian Map, and Flux Map exports all route through this, so
    the same line gets the same keyword vocabulary everywhere.

    Args:
        source_header: Real header, or a common grid's dst_wcs.to_header()
            -- whichever defines this map's own pixel grid.
        shape (tuple): (ny, nx) of the map.
        wavelength (float): Line center wavelength (um), or None if this map
            isn't tied to one line (e.g. a combined multi-line result).
        molecule (str): Species name, or None.
        flux_map (array): This map's own data, for SUMFLUX -- None skips it.
        error_map (array): Per-pixel error, for SUMFERR and the ERR
            extension -- None skips both.
        src_filename (str): Originating file, for SRCFILE -- None skips it.
        fit_method (str): Free-text method label, for FITMETHD -- None skips it.
        identity_header: Real cube/source header to pull INSTRUME/TARGNAME/etc
            from when source_header is a bare common-grid WCS with no
            identity of its own -- None trusts source_header as-is.
        extra (dict): Additional feature-specific keywords, merged in last.

    Returns:
        result (tuple): (header, extensions) -- header is the PRIMARY HDU
            header; extensions has one ("ERR", error_map, header_copy) entry
            if error_map was given and has finite values, else [].
    """
    fields = {}
    if wavelength is not None:
        fields["LINEWAVE"] = (float(wavelength), "Line center wavelength (um)")
    if molecule is not None:
        fields["MOLECULE"] = molecule
    if src_filename is not None:
        fields["SRCFILE"] = src_filename
    if fit_method is not None:
        fields["FITMETHD"] = fit_method
    if flux_map is not None:
        fields["SUMFLUX"] = (float(np.nansum(flux_map)), "Sum of this map's finite pixels")
    if error_map is not None:
        fields["SUMFERR"] = (float(np.sqrt(np.nansum(np.asarray(error_map, dtype=float) ** 2))),
                             "Propagated error of SUMFLUX")
    if extra:
        fields.update(extra)

    header = build_2d_export_header(source_header, shape, extra=fields)
    # identity_header fills in what source_header couldn't -- a common grid's
    # own WCS header never carries INSTRUME/TARGNAME, only the real one does.
    if identity_header is not None:
        for kw in _COMMON_KEYWORDS:
            if kw in identity_header and kw not in header:
                header[kw] = _ascii_safe(identity_header[kw])

    extensions = []
    if error_map is not None and np.any(np.isfinite(error_map)):
        extensions.append(("ERR", np.asarray(error_map, dtype=np.float32), header.copy()))
    return header, extensions


def build_resample_header(dst_wcs_or_header, shape, bunit=None, history=None, identity_header=None):
    """
    Shared header for a resolution-matched (convolved + reprojected) 2D
    output -- the 2D viewer's Pair/Batch convolve exports all route
    through this instead of near-identical copies.

    Args:
        dst_wcs_or_header: The output grid's astropy.wcs.WCS, or an
            already-built fits.Header for it.
        shape (tuple): (ny, nx) of the output image.
        bunit (str): Output unit, or None to leave BUNIT unset.
        history (str): One HISTORY line describing the convolution/
            reprojection (sigma, PSF model, target grid) -- None adds none.
        identity_header: Real source header to pull INSTRUME/TARGNAME/etc
            from -- None trusts dst_wcs_or_header as-is (it usually won't
            have them, being grid-only).

    Returns:
        header (astropy.io.fits.Header): Ready for the PRIMARY HDU. Pair
            with a separate fits.ImageHDU(data=coverage, header=Header())
            whose header gets 'EXTNAME'='COVERAGE' and 'BUNIT'='fraction'.
    """
    source_header = (dst_wcs_or_header.to_header()
                     if hasattr(dst_wcs_or_header, "to_header") else dst_wcs_or_header)
    extra = {}
    if bunit is not None:
        extra["BUNIT"] = bunit
    header = build_2d_export_header(source_header, shape, extra=extra)
    if identity_header is not None:
        for kw in _COMMON_KEYWORDS:
            if kw in identity_header and kw not in header:
                header[kw] = _ascii_safe(identity_header[kw])
    if history:
        header["HISTORY"] = _ascii_safe(history)
    return header


def build_coverage_extension(coverage):
    """
    The COVERAGE extension every resample export pairs with its PRIMARY --
    one shared definition so all three call sites agree on what it means.

    Args:
        coverage (array): Per-pixel input/output overlap fraction, 0-1.

    Returns:
        entry (tuple): ("COVERAGE", data, header) for write_fits_product's
            extensions list.
    """
    header = fits.Header()
    header["BUNIT"] = "fraction"
    header["COMMENT"] = "Per-pixel input/output overlap fraction (0..1)."
    return ("COVERAGE", np.asarray(coverage, dtype=np.float32), header)


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
