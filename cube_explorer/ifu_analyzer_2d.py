"""
Main GUI class for the Dual 2D FITS Viewer
"""

import os
import gc
import zipfile
import tempfile
import warnings
import traceback
import numpy as np
import matplotlib.patches as patches
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales  # used by: _get_pixel_size_arcsec
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve_fft, Gaussian2DKernel  # used by: gaussian_convolve_match
from astropy.coordinates import SkyCoord  # used by: build_target_wcs_for_batch, _apply_loaded_contour
import astropy.units as u  # used by: build_target_wcs_for_batch, _apply_loaded_contour
from scipy.ndimage import rotate

# Suppress FITS warning messages about header formatting
warnings.filterwarnings('ignore', category=Warning, module='astropy.wcs.wcs')
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QMessageBox, QComboBox, QDialog, QDoubleSpinBox, QApplication, QTextEdit,
    QInputDialog, QLineEdit, QListWidget, QListWidgetItem
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm, PowerNorm  # used by: display_image, _change_scale
from PyQt5.QtCore import Qt
from .visualization import SlicePlotter
from .colormap_dialog import ColorMapDialog
from .region_selector import RegionSelector

# ==============================================================================
# RESOLUTION MATCHING HELPERS  (v3, clean implementation)
# ==============================================================================

def _get_pixel_size_arcsec(header):
    """Return the celestial pixel scale (arcsec/pixel) from a FITS header.

    Uses ``astropy.wcs.utils.proj_plane_pixel_scales`` so that rotated CD
    / PC matrices (e.g. JWST IFU) are handled correctly.  Falls back to
    raw CDELT keywords if the WCS cannot be built.
    """
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


def _make_2d_wcs(header):
    """Build a clean 2-D celestial WCS from a header (drops trivial axes)."""
    try:
        w = WCS(header)
        if w.pixel_n_dim > 2 and w.has_celestial:
            return w.celestial
        while w.pixel_n_dim > 2:
            w = w.dropaxis(w.pixel_n_dim - 1)
        return w
    except Exception:
        return None
    
# ----------------------------------------------------------------------
# 2. Gaussian convolution -- pure astropy, NaN-aware, flux-preserving
# ----------------------------------------------------------------------

def gaussian_convolve_match(image, sigma_pix):
    """
    Smooth ``image`` with a Gaussian of width ``sigma_pix`` (in pixels)
    using ``astropy.convolution.convolve_fft``.

    * NaN pixels in the input are interpolated through during the
      convolution and then put back as NaN (the convolution itself sees
      a continuous image, so flux from real pixels does not "leak" into
      NaN regions in the output).

    * After convolution, total flux over the originally-finite pixel
      mask is renormalised to match the input total.  This compensates
      for kernel-spill across the image edge -- without it, a Gaussian
      whose tails reach beyond the array boundary would silently lose
      a fraction of the flux.

    Returns a new array; ``image`` is not modified.

    If ``sigma_pix <= 0`` (the two scales already match), this returns
    ``image.copy()`` unchanged -- the function is a no-op in that case,
    so calling it repeatedly on a same-resolution pair is idempotent.
    """
    image = np.asarray(image, dtype=float)
    if not np.isfinite(sigma_pix) or sigma_pix <= 0.05:
        return image.copy()

    kernel = Gaussian2DKernel(x_stddev=float(sigma_pix))

    finite_mask = np.isfinite(image)
    sum_in = float(np.nansum(image[finite_mask]))

    # nan_treatment='interpolate': during convolution NaNs are replaced
    # by the local kernel-weighted mean of finite neighbours, so the
    # convolution computes correctly across NaN gaps.
    # preserve_nan=True: original NaN positions are restored at the end.
    out = convolve_fft(
        image, kernel,
        nan_treatment='interpolate',
        preserve_nan=True,
        boundary='fill', fill_value=0.0,
        normalize_kernel=True,
        allow_huge=True,
    )

    out = np.array(out, dtype=float)
    out = np.where(finite_mask, out, np.nan)
    sum_out = float(np.nansum(out[finite_mask]))
    if sum_in != 0.0 and sum_out != 0.0:
        scale = sum_in / sum_out
        if 0.5 < scale < 2.0:
            out = np.where(finite_mask, out * scale, np.nan)

    return out


# ----------------------------------------------------------------------
# 3. Building output WCS at the worst pixel scale
# ----------------------------------------------------------------------

def _decompose_cd(wcs):
    """Return (rotation_unit_vectors, x_scale_deg, y_scale_deg) for a WCS.

    We split the CD matrix into rotation columns (unit vectors) and the
    per-axis scales.  This lets us re-build a CD matrix with the same
    ROTATION but a DIFFERENT (coarser) pixel scale.
    """
    try:
        cd = wcs.pixel_scale_matrix
    except Exception:
        cd = np.diag(np.asarray(wcs.wcs.cdelt, dtype=float))
    sx = float(np.hypot(cd[0, 0], cd[1, 0]))   # |column 1|
    sy = float(np.hypot(cd[0, 1], cd[1, 1]))   # |column 2|
    ux = cd[:, 0] / sx if sx > 0 else np.array([1.0, 0.0])
    uy = cd[:, 1] / sy if sy > 0 else np.array([0.0, 1.0])
    sign_x = -1.0 if cd[0, 0] < 0 else 1.0   # follow input parity
    sign_y =  1.0 if cd[1, 1] > 0 else -1.0
    return ux, uy, sx, sy, sign_x, sign_y


def build_target_wcs_for_pair(fine_wcs, fine_shape, target_pix_arcsec):
    """
    Output WCS for a single fine image being matched to a coarser scale.

    The output WCS:
      - has the same celestial CENTRE as the fine image,
      - inherits the fine image's ROTATION,
      - uses ``target_pix_arcsec`` as its pixel scale,
      - has shape large enough to fully contain the fine image.

    Returns ``(out_wcs, (ny, nx))``.
    """
    ny_in, nx_in = fine_shape
    cx_in = (nx_in - 1) / 2.0
    cy_in = (ny_in - 1) / 2.0
    sky_centre = fine_wcs.pixel_to_world(cx_in, cy_in)

    ux, uy, sx_deg, sy_deg, sign_x, sign_y = _decompose_cd(fine_wcs)
    pix_deg = target_pix_arcsec / 3600.0

    out = WCS(naxis=2)
    out.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    out.wcs.crval = [float(sky_centre.ra.deg), float(sky_centre.dec.deg)]
    out.wcs.cunit = ['deg', 'deg']
    out.wcs.cd = np.column_stack([ux * pix_deg * sign_x,
                                  uy * pix_deg * sign_y])

    # Output shape: scale up the fine FOV by (fine_pix / target_pix), pad +2
    fine_pix_x_arcsec = sx_deg * 3600.0
    fine_pix_y_arcsec = sy_deg * 3600.0
    nx_out = int(np.ceil(nx_in * fine_pix_x_arcsec / target_pix_arcsec)) + 2
    ny_out = int(np.ceil(ny_in * fine_pix_y_arcsec / target_pix_arcsec)) + 2
    if nx_out % 2 == 0:
        nx_out += 1
    if ny_out % 2 == 0:
        ny_out += 1
    out.wcs.crpix = [(nx_out + 1) / 2.0, (ny_out + 1) / 2.0]
    return out, (ny_out, nx_out)


def build_target_wcs_for_batch(input_wcss_and_shapes, target_pix_arcsec):
    """
    Output WCS that covers the UNION of every input image's footprint
    at the coarse target scale, oriented to match the LARGEST-AREA input.

    Returns ``(out_wcs, (ny, nx))``.
    """
    inputs = list(input_wcss_and_shapes)

    # Find the largest-area image (proxy: just shape * pix^2 in arcsec^2).
    # This is the "reference" for orientation and centre.
    best_idx = 0
    best_area = -1.0
    all_corners_ra = []
    all_corners_dec = []
    for idx, (w, shape) in enumerate(inputs):
        ny, nx = shape
        xs = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
        ys = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
        sky = w.pixel_to_world(xs, ys)
        all_corners_ra.append(np.asarray(sky.ra.deg))
        all_corners_dec.append(np.asarray(sky.dec.deg))
        # Approx tangent-plane area using small-angle on cos(dec)
        ra = np.asarray(sky.ra.deg); dec = np.asarray(sky.dec.deg)
        dec0 = float(np.mean(dec))
        x = (ra - np.mean(ra)) * np.cos(np.deg2rad(dec0))
        y = (dec - np.mean(dec))
        a = 0.5 * abs((x[0]*y[1] - x[1]*y[0]) + (x[1]*y[2] - x[2]*y[1])
                      + (x[2]*y[3] - x[3]*y[2]) + (x[3]*y[0] - x[0]*y[3]))
        if a > best_area:
            best_area = a
            best_idx = idx

    ref_wcs, ref_shape = inputs[best_idx]
    ref_cx = (ref_shape[1] - 1) / 2.0
    ref_cy = (ref_shape[0] - 1) / 2.0
    sky_centre = ref_wcs.pixel_to_world(ref_cx, ref_cy)
    ux, uy, _, _, sign_x, sign_y = _decompose_cd(ref_wcs)
    pix_deg = target_pix_arcsec / 3600.0

    out = WCS(naxis=2)
    out.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    out.wcs.crval = [float(sky_centre.ra.deg), float(sky_centre.dec.deg)]
    out.wcs.cunit = ['deg', 'deg']
    out.wcs.cd = np.column_stack([ux * pix_deg * sign_x,
                                  uy * pix_deg * sign_y])
    # Provisional CRPIX at (1,1); we'll re-centre below
    out.wcs.crpix = [1.0, 1.0]

    # Project every input corner into the (provisional) target pixel grid
    ra = np.concatenate(all_corners_ra)
    dec = np.concatenate(all_corners_dec)
    sky = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    px, py = out.world_to_pixel(sky)
    px = np.asarray(px); py = np.asarray(py)

    pad = 2
    nx_out = int(np.ceil(np.nanmax(px) - np.nanmin(px))) + 2 * pad + 1
    ny_out = int(np.ceil(np.nanmax(py) - np.nanmin(py))) + 2 * pad + 1
    if nx_out % 2 == 0:
        nx_out += 1
    if ny_out % 2 == 0:
        ny_out += 1

    # Re-centre: place ref centre at the middle of the new grid
    px0, py0 = out.world_to_pixel(sky_centre)
    out.wcs.crpix = [
        float(out.wcs.crpix[0] + ((nx_out + 1) / 2.0 - float(px0))),
        float(out.wcs.crpix[1] + ((ny_out + 1) / 2.0 - float(py0))),
    ]
    return out, (ny_out, nx_out)


# ----------------------------------------------------------------------
# 4. Flux-conserving exact reprojection (reproject_exact equivalent)
# ----------------------------------------------------------------------

def _polygon_area(xs, ys):
    """Shoelace area (always non-negative)."""
    n = len(xs)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) * 0.5


def _ensure_ccw(xs, ys):
    """Force a polygon to be counter-clockwise."""
    n = len(xs)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    if s < 0:
        return list(reversed(xs)), list(reversed(ys))
    return list(xs), list(ys)


def _sutherland_hodgman(subj_x, subj_y, clip_x, clip_y):
    """Clip the (CCW) subject polygon by the (CCW) clip polygon."""
    out_x = list(subj_x)
    out_y = list(subj_y)
    n_clip = len(clip_x)
    for c in range(n_clip):
        if not out_x:
            break
        cx1, cy1 = clip_x[c], clip_y[c]
        cx2, cy2 = clip_x[(c + 1) % n_clip], clip_y[(c + 1) % n_clip]
        ex, ey = cx2 - cx1, cy2 - cy1
        in_x, in_y = out_x, out_y
        out_x, out_y = [], []
        if not in_x:
            break
        sx = in_x[-1]
        sy = in_y[-1]
        for i in range(len(in_x)):
            ex_pt = in_x[i]
            ey_pt = in_y[i]
            inside_e = (ex * (ey_pt - cy1) - ey * (ex_pt - cx1)) >= 0.0
            inside_s = (ex * (sy - cy1) - ey * (sx - cx1)) >= 0.0
            if inside_e:
                if not inside_s:
                    # Compute intersection
                    dx, dy = ex_pt - sx, ey_pt - sy
                    denom = ex * dy - dx * ey
                    if denom != 0:
                        t = (ex * (cy1 - sy) - ey * (cx1 - sx)) / denom
                        out_x.append(sx + t * dx)
                        out_y.append(sy + t * dy)
                out_x.append(ex_pt)
                out_y.append(ey_pt)
            elif inside_s:
                dx, dy = ex_pt - sx, ey_pt - sy
                denom = ex * dy - dx * ey
                if denom != 0:
                    t = (ex * (cy1 - sy) - ey * (cx1 - sx)) / denom
                    out_x.append(sx + t * dx)
                    out_y.append(sy + t * dy)
            sx, sy = ex_pt, ey_pt
    return out_x, out_y


def _project_to_tan(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """Gnomonic (TAN) projection of (ra, dec) onto a tangent plane at
    (ra0, dec0).  Inputs/outputs are arrays in degrees / radians."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    ra0 = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)
    cos_c = (np.sin(dec0) * np.sin(dec)
             + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0))
    cos_c = np.where(np.abs(cos_c) < 1e-12, 1e-12, cos_c)
    xi = np.cos(dec) * np.sin(ra - ra0) / cos_c
    eta = (np.cos(dec0) * np.sin(dec)
           - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cos_c
    return xi, eta


def reproject_exact(image, src_wcs, dst_wcs, dst_shape, conserve_flux=False):
    """
    Flux-conserving reprojection by spherical-polygon intersection.

    Implements the same algorithm as ``reproject.reproject_exact`` from
    the reproject package -- but as a self-contained function with no
    external reproject dependency.

    For every output pixel:
      1. Project its 4 corners to sky.
      2. Build a local TAN tangent plane at its centre.
      3. Find input pixels whose centres lie close to the output pixel
         (within a small bounding-box margin).
      4. For each candidate input pixel: project its 4 corners into the
         same tangent plane, clip against the output pixel polygon,
         compute the overlap area.
      5. Surface-brightness mode (default):
            output = Σ(input * overlap) / Σ(overlap)
         Flux mode:
            output = Σ(input * overlap / input_full_area)
         The latter conserves the simple sum (Σ output == Σ input) for
         per-pixel-flux units (erg/s/cm², counts, ...).

    Returns (out, coverage, area_acc):
       out        - reprojected image (NaN where no input overlaps)
       coverage   - fractional output-pixel coverage in [0, 1]
       area_acc   - absolute overlap area per output pixel, in steradians
    """
    image = np.asarray(image, dtype=float)
    ny_in, nx_in = image.shape
    ny_out, nx_out = dst_shape

    # Sky positions of every INPUT pixel CORNER
    yy, xx = np.indices((ny_in + 1, nx_in + 1), dtype=float)
    xx -= 0.5; yy -= 0.5
    in_sky = src_wcs.pixel_to_world(xx.ravel(), yy.ravel())
    in_ra = np.asarray(in_sky.ra.deg).reshape(ny_in + 1, nx_in + 1)
    in_dec = np.asarray(in_sky.dec.deg).reshape(ny_in + 1, nx_in + 1)

    # Sky positions of every OUTPUT pixel CORNER
    yy, xx = np.indices((ny_out + 1, nx_out + 1), dtype=float)
    xx -= 0.5; yy -= 0.5
    out_sky = dst_wcs.pixel_to_world(xx.ravel(), yy.ravel())
    out_ra = np.asarray(out_sky.ra.deg).reshape(ny_out + 1, nx_out + 1)
    out_dec = np.asarray(out_sky.dec.deg).reshape(ny_out + 1, nx_out + 1)

    # Locate every INPUT pixel CENTRE in OUTPUT pixel coords (for
    # candidate selection -- O(N) speedup).
    yy, xx = np.indices((ny_in, nx_in), dtype=float)
    in_centres = src_wcs.pixel_to_world(xx.ravel(), yy.ravel())
    x_in_out, y_in_out = dst_wcs.world_to_pixel(in_centres)
    x_in_out = np.asarray(x_in_out).reshape(ny_in, nx_in)
    y_in_out = np.asarray(y_in_out).reshape(ny_in, nx_in)

    finite_mask = np.isfinite(image) & np.isfinite(x_in_out) & np.isfinite(y_in_out)
    flat_y = y_in_out[finite_mask]
    flat_x = x_in_out[finite_mask]
    flat_jj, flat_ii = np.where(finite_mask)
    flat_v = image[finite_mask]
    # Sort by y for fast row-window lookup
    order = np.argsort(flat_y, kind='stable')
    flat_y = flat_y[order]; flat_x = flat_x[order]
    flat_jj = flat_jj[order]; flat_ii = flat_ii[order]
    flat_v = flat_v[order]

    # Margin: cover one full output pixel + one input-pixel size in
    # output-pixel units, so we never miss a partially-overlapping pixel.
    try:
        dx = x_in_out[:, 1:] - x_in_out[:, :-1]
        dy = y_in_out[1:, :] - y_in_out[:-1, :]
        in_in_out_size = float(np.nanmedian(np.abs(np.concatenate(
            [dx.ravel(), dy.ravel()]))))
        if not np.isfinite(in_in_out_size) or in_in_out_size <= 0:
            in_in_out_size = 1.0
    except Exception:
        in_in_out_size = 1.0
    margin = max(1.5, in_in_out_size + 1.0)

    flux_acc = np.zeros(dst_shape, dtype=float)
    area_acc = np.zeros(dst_shape, dtype=float)
    out_pix_area = np.zeros(dst_shape, dtype=float)

    for j in range(ny_out):
        # Candidate input pixels are those whose y-in-output is within margin of j
        i_lo = np.searchsorted(flat_y, j - margin, side='left')
        i_hi = np.searchsorted(flat_y, j + margin, side='right')
        cand_x = flat_x[i_lo:i_hi]
        cand_jj = flat_jj[i_lo:i_hi]
        cand_ii = flat_ii[i_lo:i_hi]
        cand_v = flat_v[i_lo:i_hi]
        for i in range(nx_out):
            o_ra = (out_ra[j, i], out_ra[j, i+1],
                    out_ra[j+1, i+1], out_ra[j+1, i])
            o_dec = (out_dec[j, i], out_dec[j, i+1],
                     out_dec[j+1, i+1], out_dec[j+1, i])
            if not all(np.isfinite(o_ra)) or not all(np.isfinite(o_dec)):
                continue
            ra0 = sum(o_ra) * 0.25
            dec0 = sum(o_dec) * 0.25
            ox_arr, oy_arr = _project_to_tan(
                np.asarray(o_ra), np.asarray(o_dec), ra0, dec0)
            ox_l, oy_l = _ensure_ccw(ox_arr.tolist(), oy_arr.tolist())
            out_pix_area[j, i] = _polygon_area(ox_l, oy_l)
            if i_lo == i_hi:
                continue
            # Filter candidates by x as well
            sel = (cand_x >= i - margin) & (cand_x <= i + margin)
            if not np.any(sel):
                continue
            sjj = cand_jj[sel]; sii = cand_ii[sel]; svv = cand_v[sel]
            ns = len(sii)
            # Project all candidate input corners into this TAN plane
            ira = np.empty((ns, 4)); idec = np.empty((ns, 4))
            ira[:, 0] = in_ra[sjj,     sii]
            ira[:, 1] = in_ra[sjj,     sii + 1]
            ira[:, 2] = in_ra[sjj + 1, sii + 1]
            ira[:, 3] = in_ra[sjj + 1, sii]
            idec[:, 0] = in_dec[sjj,     sii]
            idec[:, 1] = in_dec[sjj,     sii + 1]
            idec[:, 2] = in_dec[sjj + 1, sii + 1]
            idec[:, 3] = in_dec[sjj + 1, sii]
            ix_4, iy_4 = _project_to_tan(ira, idec, ra0, dec0)
            for k in range(ns):
                if not np.all(np.isfinite(ira[k])) or not np.all(np.isfinite(idec[k])):
                    continue
                ix_l, iy_l = _ensure_ccw(ix_4[k].tolist(), iy_4[k].tolist())
                px, py = _sutherland_hodgman(ix_l, iy_l, ox_l, oy_l)
                if len(px) < 3:
                    continue
                a = _polygon_area(px, py)
                if a <= 0.0:
                    continue
                if conserve_flux:
                    in_full = _polygon_area(ix_l, iy_l)
                    if in_full <= 0:
                        continue
                    flux_acc[j, i] += svv[k] * (a / in_full)
                else:
                    flux_acc[j, i] += svv[k] * a
                area_acc[j, i] += a

    coverage = np.zeros(dst_shape, dtype=float)
    valid = out_pix_area > 0
    coverage[valid] = area_acc[valid] / out_pix_area[valid]
    coverage = np.clip(coverage, 0.0, 1.0)

    out = np.full(dst_shape, np.nan)
    has_overlap = coverage >= (1.0 - 1e-6)
    
    if conserve_flux:
        out[has_overlap] = flux_acc[has_overlap]
    else:
        out[has_overlap] = flux_acc[has_overlap] / area_acc[has_overlap]
    return out, coverage, area_acc


# ----------------------------------------------------------------------
# 5. BUNIT helpers (unit-aware flux audit and per-pixel-flux detection)
# ----------------------------------------------------------------------

def _is_per_pixel_flux_unit(bunit):
    """Heuristic: does this BUNIT indicate per-pixel total flux (vs SB)?

    Per-pixel flux units (erg/s/cm², counts, DN, electrons, plain Jy)
    require ``conserve_flux=True`` in :func:`reproject_exact` so that
    Σ output == Σ input.  Surface-brightness units (MJy/sr, Jy/beam,
    Jy/arcsec², ...) need the default SB mode.
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


# ----------------------------------------------------------------------
# 5b. WCS footprint overlap check
# ----------------------------------------------------------------------

def _wcs_footprints_overlap(wcs1, shape1, wcs2, shape2):
    """Return True if the two image footprints share ANY sky coverage.

    Projects the corners + centre of each image into the other image's
    pixel frame.  If at least one sample point lands inside the other
    image's pixel bounds the two footprints overlap.

    The check covers four cases correctly:
      * Partial overlap (corners cross the boundary)
      * One image entirely inside the other (inner corners land inside)
      * Thin-strip / rotated overlaps (extra midpoint samples help)
      * Zero overlap (all samples land outside → returns False)

    Returns True on any exception so the caller is never blocked by a
    WCS that cannot be evaluated.
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
    except Exception:
        # Cannot evaluate the WCS — assume overlap so the caller is not
        # blocked by a degenerate header; the reprojection will produce
        # an all-NaN result if there really is no overlap.
        return True


def _log_norm_vmin(data):
    """Return the ``vmin`` to use for a ``LogNorm`` over ``data``.

    Uses the smallest strictly-positive finite value so the log stretch
    spans the decades the data actually occupies, instead of an arbitrary
    floor like ``1e-10`` (which — if the true positive minimum is much
    higher — wastes most of the color range on values that don't exist
    and washes out the display). Falls back to ``1e-5`` when there is no
    positive data at all.
    """
    finite = data[np.isfinite(data)]
    positive = finite[finite > 0]
    return float(np.nanmin(positive)) if positive.size else 1e-5


# ----------------------------------------------------------------------
# 6. Backwards-compatibility aliases for the rest of the file
# ----------------------------------------------------------------------
# The GUI code calls into these names; keep them all pointing at the new
# clean implementation.

def _nan_safe_gaussian_filter(image, sigma=None, sigma_pix=None):
    """Backwards-compatible alias accepting either keyword name."""
    if sigma_pix is None:
        sigma_pix = sigma
    return gaussian_convolve_match(image, sigma_pix=sigma_pix)


_build_matched_output_wcs = build_target_wcs_for_pair
_build_common_output_wcs = build_target_wcs_for_batch
_reproject_exact_polygon = reproject_exact


def _reproject_exact_flux(image, src_wcs, dst_wcs, dst_shape, subgrid=None,
                          conserve_flux=False):
    out, fp, _ = reproject_exact(image, src_wcs, dst_wcs, dst_shape,
                                 conserve_flux=conserve_flux)
    return out, fp


def _reproject_wcs_to_wcs(image, src_wcs, dst_wcs, dst_shape, order=1):
    out, fp, _ = reproject_exact(image, src_wcs, dst_wcs, dst_shape)
    return out, fp


def _reproject_flux_conserving(image, src_wcs, dst_wcs, dst_shape,
                               conserve_flux=False, subgrid=None):
    out, fp, _ = reproject_exact(image, src_wcs, dst_wcs, dst_shape,
                                 conserve_flux=conserve_flux)
    return out, fp

# BATCH MATCH RESOLUTION DIALOG
# ==============================================================================

class FileListItem(QWidget):
    """Custom widget to display a filename and a remove button in the QListWidget"""
    def __init__(self, filepath, remove_callback, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.lbl = QLabel(os.path.basename(filepath))
        self.lbl.setToolTip(filepath)
        
        self.remove_btn = QPushButton("✖")
        self.remove_btn.setFixedSize(24, 24)
        self.remove_btn.setStyleSheet("""
            QPushButton { color: red; font-weight: bold; border: none; } 
            QPushButton:hover { background-color: #ffe0e0; border-radius: 12px; }
        """)
        self.remove_btn.clicked.connect(lambda: remove_callback(self))
        
        layout.addWidget(self.lbl)
        layout.addStretch()
        layout.addWidget(self.remove_btn)

class BatchMatchResolutionDialog(QDialog):
    """Standalone dialog for processing multiple FITS files into a ZIP"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Match Resolution & Reproject")
        self.setMinimumSize(500, 400)
        self.filepaths = []
        
        layout = QVBoxLayout(self)
        
        info_lbl = QLabel("Select multiple 2D FITS files. The app will find the worst resolution and largest\n"
                          "sky footprint among them, convolve all images to match, and package them in a ZIP.")
        layout.addWidget(info_lbl)
        
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("+ Add FITS Files")
        self.add_btn.clicked.connect(self.add_files)
        
        self.process_btn = QPushButton("Process & Download ZIP")
        self.process_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px;")
        self.process_btn.clicked.connect(self.process_and_zip)
        
        btn_layout.addWidget(self.add_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.process_btn)
        
        layout.addLayout(btn_layout)
        
    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select 2D FITS Files", filter="FITS files (*.fits *.fz *.fit)")
        for f in files:
            if f not in self.filepaths:
                self.filepaths.append(f)
                
                item = QListWidgetItem(self.list_widget)
                item_widget = FileListItem(f, self.remove_file)
                item.setSizeHint(item_widget.sizeHint())
                self.list_widget.addItem(item)
                self.list_widget.setItemWidget(item, item_widget)
                
    def remove_file(self, item_widget):
        self.filepaths.remove(item_widget.filepath)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if self.list_widget.itemWidget(item) == item_widget:
                self.list_widget.takeItem(i)
                break
                
    def process_and_zip(self):
        """
        Batch Convolution: takes N >= 2 input FITS files, finds the worst
        (coarsest) pixel scale among them, and convolves+reprojects every
        input onto a SINGLE common destination WCS at that scale.

        The destination WCS:
          - has the worst pixel scale (degrades nothing further);
          - inherits the rotation and centre of the LARGEST-area input;
          - is sized to cover the UNION of every input footprint.

        Each output FITS is a 2-HDU file (image + COVERAGE fraction).
        Inputs that are already at the worst scale are passed through with
        sigma=0 (no smoothing), and reprojected only when their WCS differs
        from the common one.  This keeps repeated batch runs idempotent.
        """
        if len(self.filepaths) < 2:
            QMessageBox.warning(
                self, "Need more files",
                "Please select at least 2 FITS files for batch processing.")
            return

        zip_path, _ = QFileDialog.getSaveFileName(
            self, "Save ZIP File", "Convolution_Images.zip", "ZIP Files (*.zip)")
        if not zip_path:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # ---- 1. Load every input ------------------------------------
            images_data = []
            for fp in self.filepaths:
                with fits.open(fp) as hdul:
                    data, header = None, None
                    for hdu in hdul:
                        if hdu.data is not None and getattr(hdu.data, 'size', 0) > 0:
                            sq = np.squeeze(hdu.data)
                            if sq.ndim == 2:
                                data = sq.astype(float)
                                header = hdu.header.copy()
                                break
                    if data is None:
                        raise ValueError(
                            f"No 2D image data in {os.path.basename(fp)}")
                    pix = _get_pixel_size_arcsec(header)
                    if pix is None:
                        raise ValueError(
                            f"Could not find pixel scale in "
                            f"{os.path.basename(fp)}")
                    wcs = _make_2d_wcs(header)
                    if wcs is None:
                        raise ValueError(
                            f"Could not build WCS for {os.path.basename(fp)}")
                    images_data.append({
                        'filename': os.path.basename(fp),
                        'data': data, 'header': header,
                        'pix_scale': pix, 'wcs': wcs, 'shape': data.shape,
                    })
                    disjoint_pairs = []
                    for i in range(len(images_data)):
                        for j in range(i + 1, len(images_data)):
                            a, b = images_data[i], images_data[j]
                            if not _wcs_footprints_overlap(
                                    a['wcs'], a['shape'], b['wcs'], b['shape']):
                                disjoint_pairs.append((a['filename'], b['filename']))

                    if disjoint_pairs:
                        QApplication.restoreOverrideCursor()
                        msg_lines = [
                            "Some of the selected images cover completely "
                            "different regions of the sky:\n"
                        ]
                        for fa, fb in disjoint_pairs[:10]:
                            msg_lines.append(f"   • {fa}  ↔  {fb}")
                        if len(disjoint_pairs) > 10:
                            msg_lines.append(
                                f"   …and {len(disjoint_pairs) - 10} more pair(s)")
                        msg_lines.append(
                            "\nBatch Convolution requires every input to share "
                            "sky coverage with the others, so they can be "
                            "matched onto a common WCS grid.\n\n"
                            "Please remove the off-field file(s) and try again.")
                        QMessageBox.warning(
                            self, "No Sky Overlap",
                            "\n".join(msg_lines))
                        return

            # ---- 2. Find the worst (coarsest) scale ---------------------
            worst_pix = max(info['pix_scale'] for info in images_data)

            # ---- 3. Build a single common destination WCS ---------------
            dst_wcs, dst_shape = build_target_wcs_for_batch(
                [(info['wcs'], info['shape']) for info in images_data],
                worst_pix,
            )

            # ---- 4. For each input, convolve + reproject ---------------
            SR_PER_ARC2 = (np.pi / (180.0 * 3600.0)) ** 2
            with tempfile.TemporaryDirectory() as tmpdir:
                flux_audit = []
                for info in images_data:
                    fine_pix = info['pix_scale']
                    fine_data = info['data']
                    fine_wcs = info['wcs']
                    bunit = info['header'].get('BUNIT', '')
                    per_pix = _is_per_pixel_flux_unit(bunit)

                    sigma_arcsec = (
                        max(0.0, worst_pix**2 - fine_pix**2) ** 0.5) / 2.3548
                    sigma_fine_pix = (sigma_arcsec / fine_pix
                                      if fine_pix > 0 else 0.0)

                    convolved = gaussian_convolve_match(
                        fine_data, sigma_pix=sigma_fine_pix)

                    # Reproject onto common destination
                    result_img, coverage, _ = reproject_exact(
                        convolved, fine_wcs, dst_wcs, dst_shape,
                        conserve_flux=per_pix)
                    result_img[coverage < 1.0 - 1e-6] = np.nan
                    if not np.any(np.isfinite(result_img)):
                        if not np.any(np.isfinite(result_img)):
                            QMessageBox.warning(
                                self, "Empty Result",
                                "Slides Convolute completed but the output has no valid pixels.\n\n"
                                "This usually means the two images overlap only at their edges, "
                                "so no output pixel is fully covered by the input.\n\n"
                                "Try a closer-aligned pair of images.")
                            return

                    out_header = dst_wcs.to_header()
                    out_header['NAXIS'] = 2
                    out_header['NAXIS1'] = dst_shape[1]
                    out_header['NAXIS2'] = dst_shape[0]
                    if bunit:
                        out_header['BUNIT'] = bunit
                    out_header['HISTORY'] = (
                        f"Batch Convolution: sigma={sigma_arcsec:.4f}\" "
                        f"({sigma_fine_pix:.3f} fine-px); reprojected to "
                        f"{worst_pix:.4f}\"/px "
                        f"({'flux' if per_pix else 'SB'} mode)"
                    )

                    out_name = f"matched_{info['filename']}"
                    out_filepath = os.path.join(tmpdir, out_name)
                    cov_header = fits.Header()
                    cov_header['EXTNAME'] = 'COVERAGE'
                    cov_header['BUNIT'] = 'fraction'
                    cov_header['COMMENT'] = (
                        'Per-pixel input/output overlap fraction (0..1).')
                    fits.HDUList([
                        fits.PrimaryHDU(data=result_img, header=out_header),
                        fits.ImageHDU(data=coverage.astype(np.float32),
                                      header=cov_header),
                    ]).writeto(out_filepath, overwrite=True)

                    # Flux audit
                    if per_pix:
                        in_total = float(np.nansum(fine_data))
                        out_total = float(np.nansum(
                            result_img[np.isfinite(result_img)]))
                    else:
                        pix_sr = (fine_pix ** 2) * SR_PER_ARC2
                        out_pix_sr = (worst_pix ** 2) * SR_PER_ARC2
                        in_total = float(np.nansum(fine_data)) * pix_sr
                        good = np.isfinite(result_img) & np.isfinite(coverage)
                        out_total = float(np.nansum(
                            result_img[good] * coverage[good])) * out_pix_sr
                    flux_audit.append((info['filename'], in_total, out_total))

                # ---- 5. Pack everything into the ZIP -------------------
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(tmpdir):
                        for file in files:
                            zipf.write(os.path.join(root, file), arcname=file)

            QApplication.restoreOverrideCursor()

            audit_lines = []
            for name, fi, fo in flux_audit:
                ratio = (fo / fi) if fi != 0 else float('nan')
                audit_lines.append(f"  {name}: ratio = {ratio:.6f}")

            final_msg = (
                f"Batch Convolution complete!\n\n"
                f"All {len(images_data)} images standardized to:\n"
                f"  Pixel size : {worst_pix:.4f} arcsec/px\n"
                f"  Output grid: {dst_shape[1]} × {dst_shape[0]} px "
                f"(union of FOVs, aligned to largest input)\n\n"
                f"Flux audit (output_total / input_total):\n"
                + "\n".join(audit_lines)
                + f"\n\nSaved to:\n{zip_path}")
            QMessageBox.information(self, "Success", final_msg)
            self.accept()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Processing Error", f"An error occurred:\n{e}")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

class Ifu_analyzer_2d_window(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("2D FITS Dual Viewer")
        
        self.setWindowFlags(Qt.Window)
        self.setMinimumSize(1200, 600)
        
        self.image_left = None
        self.header_left = None
        self.image_right = None
        self.header_right = None
        # Optional per-pixel coverage fraction for reprojected images (0..1).
        # Saved alongside the image as a 2nd HDU named "COVERAGE" by the
        # convolution pipeline.  When present the selected-area integrated
        # flux uses ``Σ value × pixel_area × coverage`` so that partial-overlap
        # edge pixels do not over-count.
        self.coverage_left = None
        self.coverage_right = None
        
        self._init_state()
        self._init_ui()

    def closeEvent(self, event):
        self.image_left = None
        self.image_right = None
        self.base_image_left = None
        self.base_image_right = None
        gc.collect()
        event.accept()

    def _init_state(self):
        """Initialize state variables"""
        self.active_shape = None
        self.active_target = None
        self.last_selection_left = None
        self.last_selection_right = None
        self.last_contour_right_wcs = None
        self.last_contour_right_level = None
        self.scale_left = 'Linear'
        self.scale_right = 'Linear'
        self.slice1_cmap = 'magma'
        self.slice2_cmap = 'magma'
        
        self.wcs_left = None
        self.wcs_right = None
        self.base_image_left = None
        self.base_header_left = None
        self.cum_angle_left = 0.0
        
        self.base_image_right = None
        self.base_header_right = None
        self.cum_angle_right = 0.0
        
        # Contour tracking variables for saving
        self.last_contour_left_img = None
        self.last_contour_left_wcs = None
        self.last_contour_left_level = None
        
        self.last_contour_right_img = None
        self.last_contour_right_wcs = None
        self.last_contour_right_level = None

    def _init_ui(self):
        # Stats labels - WordWrap enabled to prevent text cutoff
        self.stats1_label = QLabel("Image Stats:\n-")
        self.stats1_label.setMinimumWidth(150)
        self.stats1_label.setWordWrap(True)
        self.stats1_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        
        self.stats2_label = QLabel("Image Stats:\n-")
        self.stats2_label.setMinimumWidth(150)
        self.stats2_label.setWordWrap(True)
        self.stats2_label.setStyleSheet("font-size: 10pt; font-weight: bold;")

        self.hover1_label = QLabel("Cursor Info:\n-")
        self.hover1_label.setMinimumWidth(150)
        self.hover1_label.setWordWrap(True)
        self.hover1_label.setStyleSheet("font-size: 10pt; color: #333;")
        
        self.hover2_label = QLabel("Cursor Info:\n-")
        self.hover2_label.setMinimumWidth(150)
        self.hover2_label.setWordWrap(True)
        self.hover2_label.setStyleSheet("font-size: 10pt; color: #333;")

        self._init_plots()
        self._init_shape_controls()
        self._setup_main_layout()

    def _init_plots(self):
        self.slice1_fig = Figure(figsize=(5, 5), dpi=100)
        self.slice1_canvas = FigureCanvas(self.slice1_fig)
        self.slice1_plotter = SlicePlotter(self.slice1_fig)
        self.slice1_toolbar = NavigationToolbar(self.slice1_canvas, self)
        self.slice1_canvas.mpl_connect("button_press_event", lambda e: self._on_slice_click(e, "left"))
        self.slice1_canvas.mpl_connect("motion_notify_event", lambda e: self._on_mouse_hover(e, "left"))
        
        self.slice2_fig = Figure(figsize=(5, 5), dpi=100)
        self.slice2_canvas = FigureCanvas(self.slice2_fig)
        self.slice2_plotter = SlicePlotter(self.slice2_fig)
        self.slice2_toolbar = NavigationToolbar(self.slice2_canvas, self)
        self.slice2_canvas.mpl_connect("button_press_event", lambda e: self._on_slice_click(e, "right"))
        self.slice2_canvas.mpl_connect("motion_notify_event", lambda e: self._on_mouse_hover(e, "right"))

        self.region_selector_left = RegionSelector(
            self.slice1_plotter.ax,
            self.slice1_canvas,
            lambda mask, patch, coords: self._handle_region_select(mask, patch, coords, "left")
        )
        self.region_selector_right = RegionSelector(
            self.slice2_plotter.ax,
            self.slice2_canvas,
            lambda mask, patch, coords: self._handle_region_select(mask, patch, coords, "right")
        )

    def _open_colormap_dialog(self, target):
        """Opens the colormap dialog and applies the selection to the chosen panel."""
        # Open the custom dialog imported at the top of the file
        dialog = ColorMapDialog(self)
        
        if dialog.exec_():
            # Safely extract the chosen colormap string from the dialog
            # (Checking the most common variable names used in custom PyQt dialogs)
            selected_cmap = getattr(dialog, 'selected_cmap', None)
            if selected_cmap is None and hasattr(dialog, 'get_colormap'):
                selected_cmap = dialog.get_colormap()
            if selected_cmap is None:
                selected_cmap = getattr(dialog, 'cmap', 'magma') # fallback

            if selected_cmap:
                # Apply the colormap to the correct panel variable
                if target == 'left':
                    self.slice1_cmap = selected_cmap
                else:
                    self.slice2_cmap = selected_cmap
                    
                # Force the image to redraw with the new colors (if an image is loaded)
                if target == 'left' and self.image_left is not None:
                    self.display_image('left')
                elif target == 'right' and self.image_right is not None:
                    self.display_image('right')

    def _init_shape_controls(self):
        def make_shape_button(symbol, target, callback=None):
            btn = QPushButton(symbol)
            btn.setCheckable(True)
            btn.setFixedSize(40, 40)
            btn.setStyleSheet("QPushButton { font-size: 18pt; }")
            if callback is None:
                btn.clicked.connect(lambda _, s=symbol: self._set_shape(s, target=target))
            else:
                btn.clicked.connect(callback)
            return btn

        shapes = ["*", "▭", "◯", "⬭", "✏"]
        self.shape_buttons = {
            "left": {shape: make_shape_button(shape, "left") for shape in shapes},
            "right": {shape: make_shape_button(shape, "right") for shape in shapes}
        }

        self.color_btn_left = QPushButton("🌈")
        self.color_btn_left.setFixedSize(40, 40)
        self.color_btn_left.clicked.connect(lambda: self._open_colormap_dialog('left'))

        self.color_btn_right = QPushButton("🌈")
        self.color_btn_right.setFixedSize(40, 40)
        self.color_btn_right.clicked.connect(lambda: self._open_colormap_dialog('right'))

        self.scale_combo_left = QComboBox()
        self.scale_combo_left.addItems(["Linear", "Log", "Sqrt", "Power (γ=2)"])
        self.scale_combo_left.currentTextChanged.connect(lambda: self._change_scale('left'))
        
        self.scale_combo_right = QComboBox()
        self.scale_combo_right.addItems(["Linear", "Log", "Sqrt", "Power (γ=2)"])
        self.scale_combo_right.currentTextChanged.connect(lambda: self._change_scale('right'))

        self.copy_btn_l = QPushButton("⎘")
        self.copy_btn_l.setFixedSize(40, 40)
        self.copy_btn_l.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.copy_btn_l.setToolTip("Copy selection from left slice to right")
        self.copy_btn_l.clicked.connect(lambda: self.copy_selection(source="left", target="right"))

        self.copy_btn_r = QPushButton("⎘")
        self.copy_btn_r.setFixedSize(40, 40)
        self.copy_btn_r.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.copy_btn_r.setToolTip("Copy selection from right slice to left")
        self.copy_btn_r.clicked.connect(lambda: self.copy_selection(source="right", target="left"))

        self.erase_btn_l = QPushButton("✖")
        self.erase_btn_l.setFixedSize(40, 40)
        self.erase_btn_l.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.erase_btn_l.clicked.connect(lambda: self.clear_region("left"))

        self.erase_btn_r = QPushButton("✖")
        self.erase_btn_r.setFixedSize(40, 40)
        self.erase_btn_r.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.erase_btn_r.clicked.connect(lambda: self.clear_region("right"))

        # =========================================================
        # BOTTOM PANEL CONTROLS (STYLED LARGER, NO BOLD)
        # =========================================================
        control_style = "font-size: 10pt; padding: 4px;"
        btn_style = "QPushButton { font-size: 11pt; padding: 8px 16px; }"

        # ----- UNIFIED ROTATION CONTROLS -----
        self.rot_dir = QComboBox()
        self.rot_dir.addItems(["Counter-Clockwise", "Clockwise"])
        self.rot_dir.setStyleSheet(control_style)
        
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(0.0, 360.0)
        self.angle_spin.setValue(90.0)
        self.angle_spin.setSuffix(" °")
        self.angle_spin.setStyleSheet(control_style)
        
        self.rotate_btn = QPushButton("Rotate")
        self.rotate_btn.setStyleSheet(btn_style)
        self.rotate_btn.clicked.connect(self._prompt_and_rotate)
        
        self.save_btn = QPushButton("Save FITS")
        self.save_btn.setStyleSheet(btn_style)
        self.save_btn.clicked.connect(self._prompt_and_save)

        # ----- NEW CONTOUR CONTROLS -----
        self.contour_btn = QPushButton("Plot Contour")
        self.contour_btn.setStyleSheet(btn_style)
        self.contour_btn.clicked.connect(self._prompt_and_contour)

        self.superimpose_btn = QPushButton("Superimpose WCS")
        self.superimpose_btn.setStyleSheet(btn_style)
        self.superimpose_btn.clicked.connect(self._prompt_and_superimpose)
        
        self.save_contour_btn = QPushButton("Save Contour Mask")
        self.save_contour_btn.setStyleSheet(btn_style)
        self.save_contour_btn.clicked.connect(self._prompt_and_save_contour)

        self.load_contour_btn = QPushButton("Load Saved Contour")
        self.load_contour_btn.setStyleSheet(btn_style)
        self.load_contour_btn.clicked.connect(self._prompt_and_load_contour)

        self.clear_contour_btn = QPushButton("Clear Contours")
        self.clear_contour_btn.setStyleSheet(btn_style)
        self.clear_contour_btn.clicked.connect(self._prompt_and_clear_contours)

        self.match_res_btn = QPushButton("Slides Convolute")
        self.match_res_btn.setStyleSheet(btn_style)
        self.match_res_btn.setToolTip(
            "Convolve the higher-resolution slide (left/right) to match the\n"
            "coarser pixel size of the other, then reproject onto a common\n"
            "WCS grid. Flux is conserved via spherical-polygon intersection."
        )
        self.match_res_btn.clicked.connect(self.prompt_and_match_resolution)

        self.batch_match_btn = QPushButton("Batch Convolution")
        self.batch_match_btn.setStyleSheet(btn_style)
        self.batch_match_btn.setToolTip(
            "Select multiple 2D FITS files. All are convolved to the worst\n"
            "(largest) pixel scale among them and reprojected onto a common\n"
            "WCS grid covering the union footprint. Result is downloaded as a ZIP."
        )
        self.batch_match_btn.clicked.connect(self._open_batch_match_dialog)

    def copy_selection(self, source, target):
        """Copy region selection from one side to the other"""
        source_selection = self.last_selection_right if source == "right" else self.last_selection_left
        if not source_selection:
            return

        if source_selection["shape"] == "*":
            # Handle pixel selection
            x, y = source_selection["coords"]
            self._handle_pixel_click(x, y, target)
        elif source_selection["shape"] == "✏":
            # Handle lasso selection - need to recreate from vertices
            if "patch" in source_selection and hasattr(source_selection["patch"], "lasso_verts"):
                verts = source_selection["patch"].lasso_verts
                selector = self.region_selector_left if target == "left" else self.region_selector_right
                selector.active_shape = "✏"
                # Call on_lasso_select directly with the stored vertices
                selector.on_lasso_select(verts)
            else:
                # Fallback: try to create from bounding box if vertices not available
                selector = self.region_selector_left if target == "left" else self.region_selector_right
                selector.active_shape = source_selection["shape"]
                try:
                    # Convert coords to integers for safety
                    coords = tuple(int(c) for c in source_selection["coords"])
                    selector.create_region(coords)
                except Exception as e:
                    pass
        else:
            # Handle regular region selection (rectangle, circle, ellipse)
            selector = self.region_selector_left if target == "left" else self.region_selector_right
            selector.active_shape = source_selection["shape"]
            try:
                # Convert coords to integers for safety
                coords = tuple(int(c) for c in source_selection["coords"])
                selector.create_region(coords)
            except Exception as e:
                pass

    def _open_batch_match_dialog(self):
        dialog = BatchMatchResolutionDialog(self)
        dialog.exec_()

    def _prompt_and_rotate(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Select Target")
        msg.setText("Which slice would you like to rotate?")
        btn_left = msg.addButton("Left", QMessageBox.ActionRole)
        btn_right = msg.addButton("Right", QMessageBox.ActionRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        
        msg.exec_()
        
        if msg.clickedButton() == btn_left:
            self.rotate_image("left")
        elif msg.clickedButton() == btn_right:
            self.rotate_image("right")

    def _prompt_and_save(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Select Target")
        msg.setText("Which slice would you like to save?")
        btn_left = msg.addButton("Left", QMessageBox.ActionRole)
        btn_right = msg.addButton("Right", QMessageBox.ActionRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        
        msg.exec_()
        
        if msg.clickedButton() == btn_left:
            self.save_fits("left")
        elif msg.clickedButton() == btn_right:
            self.save_fits("right")

    def _prompt_and_contour(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Plot WCS Contour")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        # 1. Target Slice
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Plot contour ON:"))
        target_combo = QComboBox()
        target_combo.addItems(["Left Slice", "Right Slice"])
        target_layout.addWidget(target_combo)
        layout.addLayout(target_layout)

        # 2. Source Image
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Generate contour FROM:"))
        source_combo = QComboBox()
        source_combo.addItems(["Same Slice (Self)", "Other Slice (Match WCS)"])
        source_layout.addWidget(source_combo)
        layout.addLayout(source_layout)

        # 3. Background Std — LOCAL variable, not self.sigma_input
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Background Std (blank = Astropy auto):"))
        sigma_input = QLineEdit()          # ← local, not self.
        sigma_input.setPlaceholderText("Auto")
        sigma_input.setMinimumWidth(120)
        sigma_layout.addWidget(sigma_input)
        layout.addLayout(sigma_layout)

        # 4. Sigma levels — comma-separated multipliers
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Sigma levels (e.g. 3, 5, 10):"))
        level_input = QLineEdit()          # ← local, not self.
        level_input.setText("3, 5, 10")
        level_input.setPlaceholderText("e.g. 3, 5, 10")
        level_input.setMinimumWidth(120)
        level_layout.addWidget(level_input)
        layout.addLayout(level_layout)

        layout.addWidget(QLabel(
            "<i>Each level N draws a filled band at N × σ ± 0.05σ above background median.</i>"
        ))

        # 5. Color
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Contour Color:"))
        color_combo = QComboBox()
        color_combo.addItems(["white", "red", "green", "cyan", "magenta", "yellow", "black"])
        color_layout.addWidget(color_combo)
        layout.addLayout(color_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Plot")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        if dialog.exec_() == QDialog.Accepted:
            target      = "left" if target_combo.currentIndex() == 0 else "right"
            source      = "self" if source_combo.currentIndex() == 0 else "other"
            sigma_str   = sigma_input.text().strip()    # ← read from local var
            levels_str  = level_input.text().strip()    # ← read from local var
            color       = color_combo.currentText()
            self._apply_contour(target, source, sigma_str, levels_str, color)

    def _apply_contour(self, target, source, sigma_str, levels_str, color):
        target_plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter

        def get_2d_wcs(header):
            if header is None: return None
            w = WCS(header)
            while w.pixel_n_dim > 2:
                w = w.dropaxis(2)
            return w

        target_wcs = get_2d_wcs(self.header_left if target == 'left' else self.header_right)

        if source == 'self':
            source_img = self.image_left  if target == 'left' else self.image_right
            source_wcs = target_wcs
        else:
            source_img = self.image_right if target == 'left' else self.image_left
            source_wcs = get_2d_wcs(self.header_right if target == 'left' else self.header_left)

        if source_img is None:
            QMessageBox.warning(self, "Missing Data", "Source image is not loaded.")
            return
        if source == 'other' and (self.header_left is None or self.header_right is None):
            QMessageBox.warning(self, "Missing Data", "Both images must be loaded.")
            return

        # ── Parse sigma levels ────────────────────────────────────────────────
        try:
            sigma_levels = [float(x.strip()) for x in levels_str.split(',') if x.strip()]
        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                                "Levels must be comma-separated numbers (e.g. 3, 5, 10).")
            return
        if not sigma_levels:
            QMessageBox.warning(self, "Invalid Input", "Enter at least one sigma level.")
            return

        # ── Calculate background stats ────────────────────────────────────────
        valid = source_img[np.isfinite(source_img)]
        if valid.size == 0:
            return
        _, median_val, astropy_std = sigma_clipped_stats(valid, sigma=3.0)

        if sigma_str:
            try:
                bkg_std = float(sigma_str)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input",
                                    "Background Std must be a valid number.")
                return
        else:
            bkg_std = astropy_std

        # ── Build band-pairs: each level N → [N-0.05, N+0.05] × std + median ─
        band_levels = sorted([median_val + n * bkg_std for n in sigma_levels])

        ax = target_plotter.ax
        if not hasattr(target_plotter, 'contours'):
            target_plotter.contours = []

        # Track for save/clear
        if target == 'left':
            self.last_contour_left_img   = source_img
            self.last_contour_left_wcs   = source_wcs
            self.last_contour_left_level = band_levels[0]
        else:
            self.last_contour_right_img   = source_img
            self.last_contour_right_wcs   = source_wcs
            self.last_contour_right_level = band_levels[0]

        try:
            kwargs = {'levels': band_levels, 'alpha': 0.8}
            if source == 'other':
                kwargs['transform'] = ax.get_transform(source_wcs)
            cset = ax.contour(source_img, colors=[color], linewidths=1.5, **kwargs)
            
            target_plotter.contours.append(cset)
            target_plotter.current_contour = cset

            if target == 'left': self.slice1_canvas.draw()
            else:                 self.slice2_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Contour Error", f"Could not generate contours:\n{str(e)}")
            traceback.print_exc()

    def _prompt_and_clear_contours(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Clear Contours")
        msg.setText("Which slice's contours would you like to completely remove?")
        btn_left = msg.addButton("Left", QMessageBox.ActionRole)
        btn_right = msg.addButton("Right", QMessageBox.ActionRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        
        msg.exec_()
        
        if msg.clickedButton() == btn_left:
            self.clear_contours("left")
        elif msg.clickedButton() == btn_right:
            self.clear_contours("right")

    def clear_contours(self, target):
        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
        
        if hasattr(plotter, 'contours'):
            for contour_set in plotter.contours:
                if hasattr(contour_set, 'collections'):
                    for c in contour_set.collections:
                        try: c.remove()
                        except Exception: pass
                else:
                    try: contour_set.remove()
                    except Exception: pass
            plotter.contours = []
            
        if hasattr(plotter, 'current_contour') and plotter.current_contour:
            if hasattr(plotter.current_contour, 'collections'):
                for c in plotter.current_contour.collections:
                    try: c.remove()
                    except Exception: pass
            else:
                try: plotter.current_contour.remove()
                except Exception: pass
            plotter.current_contour = None
            
        if target == "left":
            self.slice1_canvas.draw()
            self.last_contour_left_img = None
            self.last_contour_left_level = None
        else:
            self.slice2_canvas.draw()
            self.last_contour_right_img = None
            self.last_contour_right_level = None

    def _prompt_and_save_contour(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Save Contour")
        msg.setText("Which slice's contour would you like to save?")
        btn_left  = msg.addButton("Left",   QMessageBox.ActionRole)
        btn_right = msg.addButton("Right",  QMessageBox.ActionRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.exec_()
        if msg.clickedButton() == btn_left:
            self._save_contour_txt("left")
        elif msg.clickedButton() == btn_right:
            self._save_contour_txt("right")

    def _save_contour_txt(self, target):
        plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter
        wcs_obj = self.wcs_left       if target == 'left' else self.wcs_right
        cset    = getattr(plotter, 'current_contour', None)

        if cset is None:
            QMessageBox.warning(self, "No Contour",
                                f"No active contour found on the {target} panel.")
            return

        fname, _ = QFileDialog.getSaveFileName(
            self, f"Save Contour ({target.capitalize()})",
            filter="Text files (*.txt);;CSV files (*.csv)")
        if not fname:
            return
        if not (fname.endswith('.txt') or fname.endswith('.csv')):
            fname += '.txt'

        try:
            rows = ["# IFU Analyzer 2D — Contour Export\n",
                    "# level, x_pixel, y_pixel, RA_deg, Dec_deg\n"]

            collections = cset.collections if hasattr(cset, 'collections') else [cset]
            levels      = list(cset.levels) if hasattr(cset, 'levels') else []

            for i, coll in enumerate(collections):
                level = levels[i] if i < len(levels) else i
                for path in coll.get_paths():
                    for x, y in path.vertices:
                        ra_str = dec_str = ''
                        if wcs_obj is not None:
                            try:
                                sky    = wcs_obj.pixel_to_world(x, y)
                                ra_str  = f"{sky.ra.deg:.8f}"
                                dec_str = f"{sky.dec.deg:.8f}"
                            except Exception:
                                pass
                        rows.append(f"{level:.6e}, {x:.4f}, {y:.4f}, {ra_str}, {dec_str}\n")

            with open(fname, 'w') as f:
                f.writelines(rows)

            QMessageBox.information(self, "Saved", f"Contour coordinates saved to:\n{fname}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save contour:\n{str(e)}")
            traceback.print_exc()

    def _prompt_and_load_contour(self):
        """Prompt user for target slice, mask file, and color to overlay a saved contour."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Load Saved Contour")
        msg.setText("Which slice would you like to plot the loaded contour ON?")
        btn_left = msg.addButton("Left", QMessageBox.ActionRole)
        btn_right = msg.addButton("Right", QMessageBox.ActionRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        
        msg.exec_()
        
        if msg.clickedButton() == btn_left:
            target = "left"
        elif msg.clickedButton() == btn_right:
            target = "right"
        else:
            return
            
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Contour Text File", filter="Text files (*.txt *.csv)"
        )
        if not fname:
            return
            
        # Ask for contour color
        colors = ["white", "red", "green", "cyan", "magenta", "yellow", "black"]
        color, ok = QInputDialog.getItem(
            self, "Contour Color", "Select contour line color:", colors, 0, False
        )
        
        if ok and color:
            self._apply_loaded_contour(target, fname, color)

    def _apply_loaded_contour(self, target, fname, color):
        """Loads a saved contour and plots it, reprojected onto this panel's own WCS."""
        target_plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter
        target_img = self.image_left if target == 'left' else self.image_right
        target_wcs = self.wcs_left if target == 'left' else self.wcs_right

        if target_img is None:
            QMessageBox.warning(self, "No Target Image", f"Load an image in the {target} panel first.")
            return

        try:
            data = np.genfromtxt(fname, delimiter=',', comments='#')
            if data.ndim == 1: data = data.reshape(1, -1)

            x_pix_saved = data[:, 1]
            y_pix_saved = data[:, 2]

            # Prefer RA/Dec (columns 3, 4) converted into THIS panel's pixel
            # grid, so a contour saved from one image lines up correctly on
            # a different image (different rotation, shape, or reprojection).
            # Only fall back to the raw saved pixel coordinates when the file
            # has no RA/Dec (old export, or no WCS at save time) or this
            # panel itself has no WCS to convert into — in that case the
            # contour is only valid if both panels already share a pixel grid.
            use_sky = data.shape[1] >= 5 and target_wcs is not None
            if use_sky:
                ra_pts = data[:, 3]
                dec_pts = data[:, 4]
                use_sky = bool(np.all(np.isfinite(ra_pts)) and np.all(np.isfinite(dec_pts)))

            if use_sky:
                sky = SkyCoord(ra=ra_pts * u.deg, dec=dec_pts * u.deg, frame='icrs')
                x_pts, y_pts = target_wcs.world_to_pixel(sky)
                x_pts = np.asarray(x_pts, dtype=float)
                y_pts = np.asarray(y_pts, dtype=float)
            else:
                QMessageBox.warning(
                    self, "No WCS Alignment",
                    "This contour file has no RA/Dec, or this panel has no WCS, "
                    "so it will be plotted at its saved pixel coordinates.\n\n"
                    "It will only line up correctly if this panel uses the exact "
                    "same pixel grid as the panel it was saved from.")
                x_pts, y_pts = x_pix_saved, y_pix_saved

            dist = np.sqrt(np.diff(x_pts)**2 + np.diff(y_pts)**2)
            break_indices = np.where(dist > 5.0)[0] + 1 
            
            x_plot = np.insert(x_pts, break_indices, np.nan)
            y_plot = np.insert(y_pts, break_indices, np.nan)
            
            ax = target_plotter.ax

            if not hasattr(target_plotter, 'contours'):
                target_plotter.contours = []
            
            contour_line, = ax.plot(x_plot, y_plot, color=color, linewidth=1.5, alpha=0.8, zorder=10)
            
            class DummyContourSet:
                def __init__(self, line):
                    self.collections = [line]
            
            target_plotter.contours.append(DummyContourSet(contour_line))
            target_plotter.current_contour = DummyContourSet(contour_line)
            
            if target == 'left': self.slice1_canvas.draw()
            else: self.slice2_canvas.draw()
            
            QMessageBox.information(self, "Loaded", f"Successfully loaded contour from {os.path.basename(fname)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to plot saved contour:\n{str(e)}")
            traceback.print_exc()

    def _prompt_and_superimpose(self):
        if self.image_left is None or self.image_right is None:
            QMessageBox.warning(self, "Missing Images", "Load an image in BOTH panels before superimposing.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Superimpose / WCS Match")
        dialog.setMinimumWidth(380)
        layout = QVBoxLayout(dialog)

        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("Background (reference) image:"))
        bg_combo = QComboBox()
        bg_combo.addItems(["Left", "Right"])
        bg_layout.addWidget(bg_combo)
        layout.addLayout(bg_layout)

        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Overlay colormap:"))
        cmap_combo = QComboBox()
        cmap_combo.addItems(["viridis", "plasma", "hot", "cool", "Blues", "Greens", "Reds", "magma"])
        cmap_layout.addWidget(cmap_combo)
        layout.addLayout(cmap_layout)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Overlay alpha (0.1 – 1.0):"))
        alpha_spin = QDoubleSpinBox()
        alpha_spin.setRange(0.1, 1.0)
        alpha_spin.setValue(0.5)
        alpha_spin.setSingleStep(0.1)
        alpha_layout.addWidget(alpha_spin)
        layout.addLayout(alpha_layout)

        layout.addWidget(QLabel("<i>The overlay image is reprojected onto the background\nWCS grid and blended with the selected alpha.</i>"))

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Superimpose")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        if dialog.exec_() != QDialog.Accepted:
            return

        bg          = bg_combo.currentText().lower()      
        ov          = 'right' if bg == 'left' else 'left'
        alpha       = alpha_spin.value()
        overlay_cmap = cmap_combo.currentText()

        bg_img    = self.image_left   if bg == 'left' else self.image_right
        bg_wcs    = self.wcs_left     if bg == 'left' else self.wcs_right
        ov_img    = self.image_left   if ov == 'left' else self.image_right
        ov_wcs    = self.wcs_left     if ov == 'left' else self.wcs_right

        if bg_wcs is None or ov_wcs is None:
            QMessageBox.warning(self, "No WCS", "Both images must have valid WCS to superimpose.")
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            reprojected, footprint = _reproject_wcs_to_wcs(ov_img, ov_wcs, bg_wcs, bg_img.shape, order=1)

            plotter = self.slice1_plotter if bg == 'left' else self.slice2_plotter
            canvas  = self.slice1_canvas  if bg == 'left' else self.slice2_canvas
            ax      = plotter.ax

            if np.count_nonzero(footprint) == 0:
                QMessageBox.warning(self, "No Overlap", "The two images have no overlapping sky area.")
                return

            ov_finite = reprojected[np.isfinite(reprojected)]
            if ov_finite.size == 0:
                return

            ov_norm = Normalize(vmin=float(np.nanmin(ov_finite)), vmax=float(np.nanmax(ov_finite)))
            ax.imshow(reprojected, origin='lower', cmap=overlay_cmap,
                    norm=ov_norm, alpha=alpha, aspect='auto', interpolation='nearest')
            canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Superimpose Error", f"Failed to superimpose:\n{str(e)}")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

    def _setup_main_layout(self):
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        left_half = QHBoxLayout()
        
        self.open_left_btn = QPushButton("Open 2D FITS (Left)")
        self.open_left_btn.clicked.connect(lambda: self.load_2d_fits("left"))
        
        self.header_left_btn = QPushButton("Header")
        self.header_left_btn.clicked.connect(lambda: self.show_header("left"))
        
        # Left Stats strictly bounded so they don't stretch Canvas space
        left_stats_layout = QVBoxLayout()
        left_stats_layout.addWidget(self.open_left_btn)
        left_stats_layout.addWidget(self.header_left_btn)
        left_stats_layout.addWidget(self.stats1_label)
        left_stats_layout.addSpacing(15) 
        left_stats_layout.addWidget(self.hover1_label)
        left_stats_layout.addStretch()
        
        left_stats_widget = QWidget()
        left_stats_widget.setLayout(left_stats_layout)
        left_stats_widget.setFixedWidth(180)

        # Left Canvas allowed to freely resize
        left_slice_layout = QVBoxLayout()
        left_slice_layout.addWidget(self.slice1_canvas)
        left_slice_layout.addWidget(self.slice1_toolbar)
        
        left_slice_widget = QWidget()
        left_slice_widget.setLayout(left_slice_layout)

        # Left Sidebar strictly bounded
        left_sidebar = QVBoxLayout()
        left_sidebar.addWidget(self.color_btn_left)
        left_sidebar.addWidget(QLabel("Scale:"))
        left_sidebar.addWidget(self.scale_combo_left)
        for shape in ["*", "▭", "◯", "⬭", "✏"]:
            left_sidebar.addWidget(self.shape_buttons["left"][shape])
        left_sidebar.addWidget(self.erase_btn_l)
        left_sidebar.addStretch()

        left_sidebar_widget = QWidget()
        left_sidebar_widget.setLayout(left_sidebar)
        left_sidebar_widget.setFixedWidth(120)

        left_half.addWidget(left_stats_widget, 0)
        left_half.addWidget(left_slice_widget, 1)
        left_half.addWidget(left_sidebar_widget, 0)

        # ================= RIGHT HALF =================
        right_half = QHBoxLayout()
        
        self.open_right_btn = QPushButton("Open 2D FITS (Right)")
        self.open_right_btn.clicked.connect(lambda: self.load_2d_fits("right"))
        
        self.header_right_btn = QPushButton("Header")
        self.header_right_btn.clicked.connect(lambda: self.show_header("right"))

        # Right Stats strictly bounded
        right_stats_layout = QVBoxLayout()
        right_stats_layout.addWidget(self.open_right_btn)
        right_stats_layout.addWidget(self.header_right_btn)
        right_stats_layout.addWidget(self.stats2_label)
        right_stats_layout.addSpacing(15)
        right_stats_layout.addWidget(self.hover2_label)
        right_stats_layout.addStretch()
        
        right_stats_widget = QWidget()
        right_stats_widget.setLayout(right_stats_layout)
        right_stats_widget.setFixedWidth(180)

        # Right Canvas allowed to freely resize
        right_slice_layout = QVBoxLayout()
        right_slice_layout.addWidget(self.slice2_canvas)
        right_slice_layout.addWidget(self.slice2_toolbar)
        
        right_slice_widget = QWidget()
        right_slice_widget.setLayout(right_slice_layout)

        # Right Sidebar strictly bounded
        right_sidebar = QVBoxLayout()
        right_sidebar.addWidget(self.color_btn_right)
        right_sidebar.addWidget(QLabel("Scale:"))
        right_sidebar.addWidget(self.scale_combo_right)
        for shape in ["*", "▭", "◯", "⬭", "✏"]:
            right_sidebar.addWidget(self.shape_buttons["right"][shape])
        right_sidebar.addWidget(self.erase_btn_r)
        right_sidebar.addStretch()

        right_sidebar_widget = QWidget()
        right_sidebar_widget.setLayout(right_sidebar)
        right_sidebar_widget.setFixedWidth(120)

        right_half.addWidget(right_stats_widget, 0)
        right_half.addWidget(right_slice_widget, 1)
        right_half.addWidget(right_sidebar_widget, 0)

        top_layout.addLayout(left_half, 1)
        top_layout.addLayout(right_half, 1)

        self.bottom_panel = QWidget()
        bottom_layout = QHBoxLayout(self.bottom_panel)

        lbl_style = "font-size: 11pt; color: #333;"

        # Left controls (Rotate)
        rotate_widget = QWidget()
        rotate_layout = QVBoxLayout(rotate_widget)
        rotate_layout.setContentsMargins(0,0,0,0)
        rotate_lbl = QLabel("Actions:")
        rotate_lbl.setStyleSheet(lbl_style)
        rotate_layout.addWidget(rotate_lbl)
        rotate_layout.addWidget(self.rot_dir)
        rotate_layout.addWidget(self.angle_spin)
        rotate_layout.addWidget(self.rotate_btn)
        
        # Right controls (Contours)
        contour_widget = QWidget()
        contour_layout = QVBoxLayout(contour_widget)
        contour_layout.setContentsMargins(0,0,0,0)
        contour_lbl = QLabel("Contours:")
        contour_lbl.setStyleSheet(lbl_style)
        contour_layout.addWidget(contour_lbl)
        contour_layout.addWidget(self.contour_btn)
        contour_layout.addWidget(self.superimpose_btn)
        contour_layout.addWidget(self.save_contour_btn)
        contour_layout.addWidget(self.load_contour_btn)
        contour_layout.addWidget(self.clear_contour_btn)

        match_res_widget = QWidget()
        match_res_layout = QVBoxLayout(match_res_widget)
        match_res_layout.setContentsMargins(0, 0, 0, 0)
        match_res_lbl = QLabel("Resolution:")
        match_res_lbl.setStyleSheet(lbl_style)
        match_res_layout.addWidget(match_res_lbl)
        match_res_layout.addWidget(self.match_res_btn)
        match_res_layout.addWidget(self.batch_match_btn)

        save_widget = QWidget()
        save_layout = QVBoxLayout(save_widget)
        save_layout.setContentsMargins(0,0,0,0)
        save_layout.addStretch() 
        save_layout.addWidget(self.save_btn)

        bottom_layout.addStretch(1) 
        bottom_layout.addWidget(rotate_widget)
        bottom_layout.addSpacing(30)
        bottom_layout.addWidget(contour_widget)
        bottom_layout.addSpacing(30)
        bottom_layout.addWidget(match_res_widget)
        
        bottom_layout.addStretch(3) 
        bottom_layout.addWidget(save_widget, alignment=Qt.AlignBottom)
        bottom_layout.addStretch(6) 

        main_layout.addLayout(top_layout, 2)
        main_layout.addStretch(1) 
        main_layout.addWidget(self.bottom_panel, 0)

        self.setLayout(main_layout)

    def show_header(self, target):
        header = self.header_left if target == "left" else self.header_right
        
        if header is None:
            QMessageBox.warning(self, "No Image", "Please load an image first to view its header.")
            return

        header_text = "\n".join(
            f"{k:8} = {v!r} / {header.comments[k]}" if k in header.comments else f"{k:8} = {v!r}" 
            for k, v in header.items()
        )
        
        header_window = QWidget()
        header_window.setWindowTitle(f"FITS Header ({target.capitalize()})")
        header_window.setMinimumSize(600, 500)
        
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFontFamily("Courier")
        text_edit.setText(header_text)
        layout.addWidget(text_edit)
        
        header_window.setLayout(layout)
        header_window.show()
        
        if target == "left":
            self.header_window_left = header_window
        else:
            self.header_window_right = header_window

    def rebuild_axes(self, target, header):
        plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter
        canvas  = self.slice1_canvas  if target == 'left' else self.slice2_canvas

        plotter.fig.clf()
        ax_rect   = [0.25, 0.20, 0.50, 0.66]
        cbar_rect = [0.80, 0.20, 0.03, 0.66]

        wcs_for_projection = None

        # ── Step 1: Build a clean, strictly-2D, SIP-free WCS ──────────────────
        try:
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

            wcs_obj = WCS(ch, naxis=2)

            wcs_for_projection = wcs_obj.celestial if wcs_obj.has_celestial else wcs_obj
            print(f"DEBUG rebuild_axes [{target}]: WCS parsed OK → {wcs_for_projection}")

        except Exception as e:
            print(f"DEBUG rebuild_axes [{target}]: WCS parse FAILED → {e}")
            wcs_for_projection = None
            
        # Store so hover readout can use it
        if target == 'left':
            self.wcs_left  = wcs_for_projection
        else:
            self.wcs_right = wcs_for_projection

        # ── Step 2: Create axes (WCSAxes if possible, plain fallback) ──────────
        try:
            if wcs_for_projection is None:
                raise ValueError("No valid WCS available for projection")
            plotter.ax = plotter.fig.add_axes(ax_rect, projection=wcs_for_projection)
            print(f"DEBUG rebuild_axes [{target}]: WCSAxes created ✓")
        except Exception as e:
            print(f"DEBUG rebuild_axes [{target}]: WCSAxes FAILED, using plain axes → {e}")
            plotter.ax = plotter.fig.add_axes(ax_rect)

        plotter.cbar_ax = plotter.fig.add_axes(cbar_rect)
        plotter.cbar_ax.set_visible(False)

        if target == 'left':
            self.region_selector_left = RegionSelector(
                plotter.ax, canvas,
                lambda mask, patch, coords: self._handle_region_select(mask, patch, coords, 'left'))
        else:
            self.region_selector_right = RegionSelector(
                plotter.ax, canvas,
                lambda mask, patch, coords: self._handle_region_select(mask, patch, coords, 'right'))

    def load_2d_fits(self, target):
        fname, _ = QFileDialog.getOpenFileName(
            self, f"Open 2D FITS ({target.capitalize()})", filter="FITS files (*.fits *.fz *.fit)")
        
        if not fname:
            return

        try:
            self.clear_region(target)
            
            with fits.open(fname) as hdul:
                data = None
                header = None
                coverage = None
                # Accurately find the correct 2D extension (fixes JWST empty primary HDUs)
                for hdu in hdul:
                    if hdu.data is not None and getattr(hdu.data, 'size', 0) > 0:
                        sq_data = np.squeeze(hdu.data)
                        if sq_data.ndim == 2:
                            data = sq_data
                            header = hdu.header.copy()
                            break

                # If a "COVERAGE" extension exists, load it: this is the per-pixel
                # coverage fraction (0..1) written by the from-scratch reproject
                # pipeline.  It lets the selected-area integrated-flux readout
                # weight partial-overlap edge pixels correctly.
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
                    QMessageBox.warning(self, "Load Error", "No image data found in FITS file.")
                    return

                data = np.squeeze(data)
                if data.ndim != 2:
                    QMessageBox.warning(self, "Dimension Error", 
                                        f"Expected a 2D image, but got a {data.ndim}D array.\n"
                                        "Please select a 2D FITS file.")
                    return

                # --- NEW CLEANUP BLOCK: Remove dangling 3D JWST keys ---
                header['NAXIS'] = 2
                if 'WCSAXES' in header:
                    header['WCSAXES'] = 2
                
                bad_keys = ['NAXIS3', 'NAXIS4','CTYPE3', 'CRVAL3', 'CRPIX3', 'CDELT3', 'CUNIT3', 
                            'PC1_3', 'PC2_3', 'PC3_1', 'PC3_2', 'PC3_3',
                            'CD1_3', 'CD2_3', 'CD3_1', 'CD3_2', 'CD3_3']
                for k in bad_keys:
                    if k in header:
                        del header[k]
                # -------------------------------------------------------

                if target == "left":
                    self.base_image_left = data.copy()
                    self.base_header_left = header.copy()
                    self.cum_angle_left = 0.0
                    self.image_left = data
                    self.header_left = header
                    self.coverage_left = coverage
                else:
                    self.base_image_right = data.copy()
                    self.base_header_right = header.copy()
                    self.cum_angle_right = 0.0
                    self.image_right = data
                    self.header_right = header
                    self.coverage_right = coverage

                self.rebuild_axes(target, header)
                filename = os.path.basename(fname)
                self.display_image(target, title=filename)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading FITS file:\n{str(e)}")
            traceback.print_exc()

    def rotate_image(self, target):
        base_image = self.base_image_left if target == "left" else self.base_image_right
        base_header = self.base_header_left if target == "left" else self.base_header_right
        
        if base_image is None or base_header is None:
            QMessageBox.warning(self, "No Image", f"Please load an image into the {target} panel first.")
            return
            
        angle = self.angle_spin.value()
        if angle == 0.0:
            return
            
        if self.rot_dir.currentText() == "Clockwise":
            angle = -angle

        if target == "left":
            self.cum_angle_left += angle
            total_angle = self.cum_angle_left
        else:
            self.cum_angle_right += angle
            total_angle = self.cum_angle_right

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            wcs_in_full = WCS(base_header)
            if not wcs_in_full.has_celestial:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "WCS Error", "Image does not have a valid celestial WCS.")
                return
                
            if wcs_in_full.pixel_n_dim > 2:
                wcs_in = wcs_in_full.celestial
            else:
                wcs_in = wcs_in_full
                
            wcs_out = wcs_in.deepcopy()
            theta = np.deg2rad(total_angle)
            
            array_out = rotate(base_image, total_angle, reshape=True, cval=np.nan, order=1)
            
            cos_t, sin_t = np.cos(-theta), np.sin(-theta)
            R_inv = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            
            if hasattr(wcs_out.wcs, 'cd'):
                wcs_out.wcs.cd = np.dot(wcs_out.wcs.cd, R_inv)
            elif hasattr(wcs_out.wcs, 'pc'):
                wcs_out.wcs.pc = np.dot(wcs_out.wcs.pc, R_inv)
            else:
                wcs_out.wcs.pc = R_inv
                
            ny, nx = base_image.shape
            new_ny, new_nx = array_out.shape
            
            cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
            new_cx, new_cy = (new_nx - 1) / 2.0, (new_ny - 1) / 2.0
            
            crpix_0 = wcs_in.wcs.crpix - 1
            dx, dy = crpix_0[0] - cx, crpix_0[1] - cy
            
            cos_r, sin_r = np.cos(theta), np.sin(theta)
            new_dx = dx * cos_r - dy * sin_r
            new_dy = dx * sin_r + dy * cos_r
            
            wcs_out.wcs.crpix = np.array([new_cx + new_dx, new_cy + new_dy]) + 1
            
            new_header = base_header.copy()
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
                    
            if target == "left":
                self.image_left = array_out
                self.header_left = new_header
                # Coverage map (if any) is no longer valid after rotation,
                # because rotate() reshapes the array; user can re-load to get it back.
                self.coverage_left = None
            else:
                self.image_right = array_out
                self.header_right = new_header
                self.coverage_right = None
                
            self.rebuild_axes(target, new_header)
            self.clear_region(target)
            
            display_angle = total_angle % 360 if total_angle >= 0 else -(-total_angle % 360)
            self.display_image(target, title=f"Rotated {display_angle:.1f}° Total")
            
        except Exception as e:
            QMessageBox.critical(self, "Rotation Error", f"An error occurred during rotation:\n{str(e)}")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

    def save_fits(self, target):
        image = self.image_left if target == "left" else self.image_right
        header = self.header_left if target == "left" else self.header_right
        
        if image is None or header is None:
            QMessageBox.warning(self, "No Image", f"There is no image currently loaded in the {target} panel to save.")
            return
            
        fname, _ = QFileDialog.getSaveFileName(
            self, f"Save FITS ({target.capitalize()})", filter="FITS files (*.fits)"
        )
        if not fname:
            return
            
        if not fname.lower().endswith('.fits'):
            fname += '.fits'
            
        try:
            hdu = fits.PrimaryHDU(data=image, header=header)
            hdu.writeto(fname, overwrite=True)
            QMessageBox.information(self, "Success", f"FITS successfully saved to:\n{fname}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save FITS file:\n{str(e)}")

    def display_image(self, target, title=''):
        image  = self.image_left   if target == 'left' else self.image_right
        header = self.header_left  if target == 'left' else self.header_right
        if image is None:
            return

        plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter
        stats_label = self.stats1_label if target == 'left' else self.stats2_label
        cmap  = getattr(self, 'slice1_cmap' if target == 'left' else 'slice2_cmap', 'magma')
        scale = self.scale_left if target == 'left' else self.scale_right

        ax = plotter.ax
        for artist in list(ax.images):
            artist.remove()
        for coll in list(ax.collections):
            coll.remove()
        for line in list(ax.lines):
            line.remove()
        ax.set_title('')

        # 1. Calculate Intensity Normalization
        finite = image[np.isfinite(image)]
        vmin   = float(np.nanmin(finite)) if finite.size else 0.0
        vmax   = float(np.nanmax(finite)) if finite.size else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0

        if scale == 'Log':
            # Log scale requires strictly positive values
            vmin_log = _log_norm_vmin(finite)
            norm = LogNorm(vmin=vmin_log, vmax=vmax)
        elif scale == 'Sqrt':
            vmin_sqrt = max(vmin, 0.0)
            norm = PowerNorm(gamma=0.5, vmin=vmin_sqrt, vmax=vmax)
        elif scale == 'Power (γ=2)':
            norm = PowerNorm(gamma=2.0, vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        # 2. Render the Image
        im = ax.imshow(image, origin='lower', cmap=cmap, norm=norm,
               interpolation='nearest', aspect='auto')
        ax.set_title('', pad=4)
        # 3. WCS Grid and Labels
        unit_str = ""
        if header is not None:
            if 'BUNIT' in header:
                unit_str = f" [{header['BUNIT']}]"
                
            # Only attempt grid if the axes was built as a WCSAxes object
            if hasattr(ax, 'coords'):
                try:
                    ax.set_axis_on()
                    ax.coords.grid(color='white', alpha=0.5, linestyle='solid')

                    try:
                        ra  = ax.coords['ra']
                        dec = ax.coords['dec']
                    except (KeyError, TypeError):
                        ra, dec = ax.coords[0], ax.coords[1]

                    # Choose formatter precision based on FOV — let Astropy auto-pick spacing
                    cdelt    = abs(header.get('CDELT1', header.get('CD1_1', 0.0)))
                    ny, nx   = image.shape
                    fov_amin = cdelt * max(nx, ny) * 60.0   # arcmin

                    if fov_amin > 10.0:
                        ra.set_major_formatter('hh:mm:ss')
                        dec.set_major_formatter('dd:mm:ss')
                    elif fov_amin > 1.0:
                        ra.set_major_formatter('hh:mm:ss.s')
                        dec.set_major_formatter('dd:mm:ss')
                    else:
                        # Sub-arcmin field — needs sub-second RA precision
                        ra.set_major_formatter('hh:mm:ss.ss')
                        dec.set_major_formatter('dd:mm:ss.s')

                    # ← NO set_ticks() call — Astropy auto-locates ticks for the current FOV

                    ra.set_axislabel('RA',   minpad=0.5)
                    dec.set_axislabel('Dec', minpad=0.5)
                    ra.set_ticklabel_position('tblr')
                    dec.set_ticklabel_position('tblr')
                    ra.set_axislabel_position('b')
                    dec.set_axislabel_position('l')
                    ra.set_ticks_visible(True)
                    dec.set_ticks_visible(True)
                    ra.set_ticklabel_visible(True)
                    dec.set_ticklabel_visible(True)

                    ra.set_ticklabel(color='black', size=7, rotation='vertical', pad=2)
                    dec.set_ticklabel(color='black', size=7)
                    ra.set_ticks(color='black', width=1)
                    dec.set_ticks(color='black', width=1)
                    ra.set_axislabel('RA',  color='black', size=8)
                    dec.set_axislabel('Dec', color='black', size=8)

                    print(f"DEBUG WCS grid: fov={fov_amin:.3f} arcmin, auto-spacing enabled")

                except Exception as e:
                    QMessageBox.warning(self, "Grid Error", f"Failed to draw WCS grid:\n{str(e)}")
                    traceback.print_exc()
                print(f"DEBUG coords type: {type(ax.coords)}")
                print(f"DEBUG ax position: {ax.get_position()}")

        # 4. Colorbar Management
        if not hasattr(plotter, 'cbar_ax') or plotter.cbar_ax is None:
            # Defined placement: [left, bottom, width, height] in figure coordinates
            plotter.cbar_ax = plotter.fig.add_axes([0.84, 0.10, 0.04, 0.82])

        plotter.cbar_ax.cla()
        plotter.cbar_ax.set_visible(True)

        cbar = plotter.fig.colorbar(im, cax=plotter.cbar_ax)
        cbar.ax.tick_params(labelsize=7)

        if header is not None and 'BUNIT' in header:
            cbar.set_label(header['BUNIT'], size=9, weight='bold', labelpad=5)
            
        plotter.cbar = cbar

        # 5. Update Statistics Panel
        mn   = np.nanmin(image)
        mx   = np.nanmax(image)
        mean = np.nanmean(image)
        stats_label.setText(
            f"Image Stats{unit_str}\n"
            f"min={mn:.3e}\n"
            f"max={mx:.3e}\n"
            f"mean={mean:.3e}"
        )

        # 6. Refresh Canvas
        if target == 'left':
            self.slice1_canvas.draw()
        else:
            self.slice2_canvas.draw()

    def _on_mouse_hover(self, event, target):
        """Handle dynamic updates of mouse coordinate position and value mapping"""
        plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter
        image = self.image_left if target == 'left' else self.image_right
        header = self.header_left if target == 'left' else self.header_right
        hover_label = self.hover1_label if target == 'left' else self.hover2_label

        if image is None or event.inaxes != plotter.ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        ix, iy = int(round(x)), int(round(y))
        
        # Pull actual intensity level
        if 0 <= iy < image.shape[0] and 0 <= ix < image.shape[1]:
            val = float(image[iy, ix])
        else:
            val = float('nan')

        # Translate Matplotlib X,Y coordinate to World Coordinate RA/DEC
        wcs_str = ""
        wcs_obj = self.wcs_left if target == 'left' else self.wcs_right
        
        if wcs_obj is not None and wcs_obj.has_celestial:
            try:
                wcs2d = wcs_obj.celestial if wcs_obj.pixel_n_dim > 2 else wcs_obj
                world = wcs2d.pixel_to_world(x, y)
                ra = world.ra.to_string(unit='hour', sep='hms', precision=2, pad=True)
                dec = world.dec.to_string(unit='deg', sep='dms', precision=1, alwayssign=True, pad=True)
                wcs_str = f"RA: {ra}\nDec: {dec}\n"
            except Exception:
                pass

        unit_str = ""
        if header is not None and 'BUNIT' in header:
            unit_str = f" [{header['BUNIT']}]"

        # Update the live UI label directly
        text = f"Cursor Info:\nPixel: {ix}, {iy}\n{wcs_str}Val: {val:.4e}{unit_str}"
        hover_label.setText(text)

    def _set_shape(self, shape, target):
        if self.active_shape == shape and self.active_target == target:
            self.active_shape = None
            self.active_target = None
        else:
            self.active_shape = shape
            self.active_target = target

        for side in ["left", "right"]:
            for btn_shape, btn in self.shape_buttons[side].items():
                btn.setChecked(side == target and btn_shape == shape and self.active_shape is not None)

        selector = self.region_selector_left if target == "left" else self.region_selector_right
        selector.activate_selector(shape if self.active_shape else None)

    def _handle_region_select(self, mask, patch, coords, target):
        image = self.image_left if target == "left" else self.image_right
        header = self.header_left if target == "left" else self.header_right
        
        # FIXED: Changed 'xNone' to 'None' to resolve NameError
        if image is None: return

        xmin, xmax, ymin, ymax = coords

        # Build the full-size mask matching the image dimensions
        if mask.shape == image.shape:
            full_mask = mask
        else:
            full_mask = np.zeros_like(image, dtype=bool)
            h_mask, w_mask = mask.shape
            y0, x0 = int(ymin), int(xmin)
            y1, x1 = min(y0 + h_mask, image.shape[0]), min(x0 + w_mask, image.shape[1])
            full_mask[y0:y1, x0:x1] = mask[:(y1-y0), :(x1-x0)]

        selection = {"shape": self.active_shape, "coords": coords, "mask": full_mask, "patch": patch}
        
        if target == "left": self.last_selection_left = selection
        else: self.last_selection_right = selection

        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
        
        # Clean up previous area artists
        if hasattr(plotter.ax, 'area_artists'):
            for artist in getattr(plotter.ax, 'area_artists'):
                try: artist.remove()
                except Exception: pass
        plotter.ax.area_artists = []

        plotter.ax.add_patch(patch)
        plotter.ax.area_artists.append(patch)
        
        selector = self.region_selector_left if target == "left" else self.region_selector_right
        if selector.selector is not None:
            try: selector.selector.set_visible(False)
            except Exception: pass
        
        # Reset UI button states
        self.active_shape = None
        self.active_target = None
        for side in ["left", "right"]:
            for btn in self.shape_buttons[side].values():
                btn.setChecked(False)
        selector.activate_selector(None)

        selected_area = image[full_mask]
        if selected_area.size > 0:
            mn, mx, mean = np.nanmin(selected_area), np.nanmax(selected_area), np.nanmean(selected_area)
            total_sum, std = np.nansum(selected_area), np.nanstd(selected_area)

            stats_label = self.stats1_label if target == "left" else self.stats2_label
            current_text = stats_label.text().split("\n")
            slice_stats = [line for line in current_text if line.startswith("Image Stats")]

            bunit = ""
            unit_str = ""
            if header is not None and 'BUNIT' in header:
                bunit = str(header['BUNIT'])
                unit_str = f" [{bunit}]"

            region_stats = [
                f"Selected Area Stats{unit_str}:",
                f"min={mn:.3e}", f"max={mx:.3e}", f"mean={mean:.3e}",
                f"Sum={total_sum:.3e}",
                f"std={std:.3e}",
            ]

            combined_stats = slice_stats + [""] + region_stats if slice_stats else region_stats
            stats_label.setText("\n".join(combined_stats))

        if target == "left": self.slice1_canvas.draw()
        else: self.slice2_canvas.draw()

    def clear_region(self, target):
        if target == "left":
            self.last_selection_left = None
            plotter = self.slice1_plotter
        else:
            self.last_selection_right = None
            plotter = self.slice2_plotter

        # Clear physical shapes
        if hasattr(plotter.ax, 'area_artists'):
            for artist in plotter.ax.area_artists:
                try: artist.remove()
                except Exception: pass
            plotter.ax.area_artists = []

        # Clear pixel marks
        if hasattr(plotter.ax, 'pixel_marker'):
            for marker in plotter.ax.pixel_marker:
                try: marker.remove()
                except Exception: pass
            plotter.ax.pixel_marker = []
            
        if hasattr(plotter, 'current_contour') and plotter.current_contour:
            if hasattr(plotter.current_contour, 'collections'):
                for c in plotter.current_contour.collections:
                    try: c.remove()
                    except Exception: pass
            else:
                try: plotter.current_contour.remove()
                except Exception: pass
            plotter.current_contour = None
            
        if target == "left":
            self.last_contour_left_img = None
            self.last_contour_left_wcs = None
            self.last_contour_left_level = None
        else:
            self.last_contour_right_img = None
            self.last_contour_right_wcs = None
            self.last_contour_right_level = None

        if target == "left" and self.image_left is not None:
            self.display_image("left", "Left Image")
        elif target == "right" and self.image_right is not None:
            self.display_image("right", "Right Image")

    def _handle_pixel_click(self, x, y, target):
        image = self.image_left if target == "left" else self.image_right
        header = self.header_left if target == "left" else self.header_right
        if image is None: return
                
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            intensity = float(image[y, x])
            
            if target == "left": self.last_selection_left = {"shape": "*", "coords": (x, y), "mask": None}
            else: self.last_selection_right = {"shape": "*", "coords": (x, y), "mask": None}

            stats_label = self.stats1_label if target == "left" else self.stats2_label
            
            unit_str = ""
            if header is not None and 'BUNIT' in header:
                unit_str = f" {header['BUNIT']}"
                
            stats_label.setText(f"Pixel: (x={x}, y={y})\nIntensity={intensity:.3e}{unit_str}")

            plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
            
            if hasattr(plotter.ax, 'area_artists'):
                for artist in plotter.ax.area_artists:
                    try: artist.remove()
                    except Exception: pass
                plotter.ax.area_artists = []

            if hasattr(plotter.ax, 'pixel_marker'):
                for marker in plotter.ax.pixel_marker:
                    marker.remove()
                plotter.ax.pixel_marker = []

            plotter.update_pixel_marker(x, y)
            
            if target == "left": self.slice1_canvas.draw()
            else: self.slice2_canvas.draw()

    def _on_slice_click(self, event, target):
        if event.inaxes is None or self.active_shape != "*": return

        ax = self.slice1_plotter.ax if target == "left" else self.slice2_plotter.ax
        if event.inaxes != ax: return

        x, y = int(round(event.xdata)), int(round(event.ydata))
        self._handle_pixel_click(x, y, target)

    def prompt_and_match_resolution(self):
        """
        "Slides Convolute" button: convolve the higher-resolution panel to
        the worse pixel scale, then reproject onto a grid that preserves
        the full FOV of the fine image.

        Pipeline (uses the rewritten v3 helpers above):
          1. Pick whichever panel has the WORSE (larger) pixel scale as
             the reference; convolve the OTHER panel to that scale.
          2. Build the destination WCS at the worst scale, inheriting the
             fine image's rotation and centre, sized to fully cover the
             fine FOV.
          3. ``gaussian_convolve_match`` with sigma derived from
             quadrature: sigma = sqrt(p_target^2 - p_fine^2) / 2.3548
             (in arcsec), expressed in fine-image pixels.
          4. ``reproject_exact`` with ``conserve_flux`` chosen from BUNIT.
          5. Save 2-HDU FITS (image + COVERAGE).

        If the two images already have the same pixel scale, the
        convolution would be a no-op (sigma == 0).  Rather than silently
        re-saving a copy, we tell the user nothing needs doing and
        return without writing anything.
        """
        if self.image_left is None or self.image_right is None:
            QMessageBox.warning(self, "Missing Images",
                                "Load images in BOTH panels first.")
            return

        pix_l = _get_pixel_size_arcsec(self.header_left)
        pix_r = _get_pixel_size_arcsec(self.header_right)
        if pix_l is None or pix_r is None:
            QMessageBox.warning(
                self, "Missing Pixel Scale",
                "Could not determine pixel scale for one or both images.")
            return

        # Pick the coarser scale as the target; the finer image gets convolved
        if pix_l >= pix_r:
            ref_side, conv_side = "left", "right"
            target_pix, fine_pix = pix_l, pix_r
            ref_header = self.header_left
            ref_img    = self.image_left
            fine_header = self.header_right
            fine_img = self.image_right
        else:
            ref_side, conv_side = "right", "left"
            target_pix, fine_pix = pix_r, pix_l
            ref_header = self.header_right
            ref_img    = self.image_right
            fine_header = self.header_left
            fine_img = self.image_left

        # Same-resolution short-circuit: refuse politely instead of
        # producing a degraded copy when the user clicks again.
        same_scale_eps = 1e-4  # arcsec
        if abs(target_pix - fine_pix) < same_scale_eps:
            QMessageBox.information(
                self, "Same resolution",
                f"Both panels already have the same pixel scale "
                f"({target_pix:.4f} arcsec/px).\n\n"
                "Slides Convolute would be a no-op, so nothing was changed.")
            return

        fine_wcs = _make_2d_wcs(fine_header)
        if fine_wcs is None:
            QMessageBox.warning(self, "WCS Error",
                                "Could not build a 2D WCS for the fine image.")
            return

        # Guard: refuse to convolve when the two images share no sky
        # coverage at all.  Without this, the reprojection silently
        # produces an all-NaN result and the user gets no useful output.
        ref_wcs = _make_2d_wcs(ref_header)
        if ref_wcs is not None and ref_img is not None:
            if not _wcs_footprints_overlap(
                    fine_wcs, fine_img.shape, ref_wcs, ref_img.shape):
                QMessageBox.warning(
                    self, "No Sky Overlap",
                    "The two images cover completely different regions of the sky.\n\n"
                    "Slides Convolute requires the panels to share sky coverage so "
                    "the convolved image can be placed on a meaningful WCS grid.\n\n"
                    "Please load a matching pair of images of the same field.")
                return

        # Sigma for the matching kernel.  Pixel-size-limited PSF
        # assumption: sqrt(target^2 - fine^2) is the FWHM, /2.3548 is sigma.
        sigma_arcsec = (max(0.0, target_pix**2 - fine_pix**2) ** 0.5) / 2.3548
        sigma_fine_pix = sigma_arcsec / fine_pix if fine_pix > 0 else 0.0

        dst_wcs, dst_shape = build_target_wcs_for_pair(
            fine_wcs, fine_img.shape, target_pix)

        bunit = fine_header.get('BUNIT', '') if fine_header is not None else ''
        per_pixel_flux = _is_per_pixel_flux_unit(bunit)

        info = (
            f"<b>Slides Convolute: {conv_side.capitalize()} &rarr; "
            f"{target_pix:.4f}\"/px ({ref_side} scale)</b><br>"
            f"Fine scale     : {fine_pix:.4f}\"/px ({conv_side})<br>"
            f"Target scale   : {target_pix:.4f}\"/px ({ref_side})<br>"
            f"Smoothing &sigma; : {sigma_arcsec:.4f}\" ({sigma_fine_pix:.3f} fine-px)<br>"
            f"Output grid    : {dst_shape[1]} × {dst_shape[0]} px "
            f"(full fine FOV, fine rotation preserved)<br>"
            f"BUNIT          : {bunit} &rarr; "
            f"{'flux' if per_pixel_flux else 'surface-brightness'} mode<br><br>"
            "Algorithm: astropy.convolution.convolve_fft (NaN-aware Gaussian) "
            "&rarr; flux-conserving exact spherical-polygon reprojection. Proceed?"
        )
        if QMessageBox.question(self, "Slides Convolute", info) != QMessageBox.Yes:
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            convolved = gaussian_convolve_match(fine_img, sigma_pix=sigma_fine_pix)

            # 2. Flux-conserving reproject to the target WCS.
            result_img, coverage, _ = reproject_exact(
                convolved, fine_wcs, dst_wcs, dst_shape,
                conserve_flux=per_pixel_flux)

            result_img[coverage < 1.0 - 1e-6] = np.nan
            if not np.any(np.isfinite(result_img)):
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(
                    self, "Empty Result",
                    "Slides Convolute completed but the output has no "
                    "valid pixels.\n\n"
                    "This usually means the two images overlap only at "
                    "their edges, so no output pixel is fully covered "
                    "by the input.\n\n"
                    "Try a closer-aligned pair of images.")
                return

            # 3. Build a clean output header.
            out_header = dst_wcs.to_header()
            out_header['NAXIS'] = 2
            out_header['NAXIS1'] = dst_shape[1]
            out_header['NAXIS2'] = dst_shape[0]
            if bunit:
                out_header['BUNIT'] = bunit
            out_header['HISTORY'] = (
                f"Slides Convolute: sigma={sigma_arcsec:.4f}\" "
                f"({sigma_fine_pix:.3f} fine-px); reprojected to "
                f"{target_pix:.4f}\"/px "
                f"({'flux' if per_pixel_flux else 'SB'} mode)"
            )

            fname, _ = QFileDialog.getSaveFileName(
                self, "Save Convolved FITS", filter="FITS (*.fits)")
            if fname:
                if not fname.lower().endswith(".fits"):
                    fname += ".fits"
                cov_header = fits.Header()
                cov_header['EXTNAME'] = 'COVERAGE'
                cov_header['BUNIT'] = 'fraction'
                cov_header['COMMENT'] = (
                    'Per-pixel input/output overlap fraction (0..1).')
                fits.HDUList([
                    fits.PrimaryHDU(data=result_img, header=out_header),
                    fits.ImageHDU(data=coverage.astype(np.float32),
                                  header=cov_header),
                ]).writeto(fname, overwrite=True)

                # Flux audit
                if per_pixel_flux:
                    in_total = float(np.nansum(fine_img))
                    out_total = float(
                        np.nansum(result_img[np.isfinite(result_img)]))
                    audit_unit = bunit
                else:
                    SR_PER_ARC2 = (np.pi / (180.0 * 3600.0)) ** 2
                    pix_sr = (fine_pix ** 2) * SR_PER_ARC2
                    out_pix_sr = (target_pix ** 2) * SR_PER_ARC2
                    in_total = float(np.nansum(fine_img)) * pix_sr
                    good = np.isfinite(result_img) & np.isfinite(coverage)
                    out_total = float(np.nansum(
                        result_img[good] * coverage[good])) * out_pix_sr
                    audit_unit = f"{bunit} × sr"
                ratio = (out_total / in_total) if in_total != 0 else float('nan')
                QMessageBox.information(
                    self, "Saved",
                    f"Saved to:\n{fname}\n\n"
                    f"Flux audit ({audit_unit}):\n"
                    f"  Input  total : {in_total:.4e}\n"
                    f"  Output total : {out_total:.4e}\n"
                    f"  Ratio out/in : {ratio:.6f}\n"
                    f"  (1.000000 means flux is exactly conserved.)"
                )

        except Exception as e:
            QMessageBox.critical(self, "Slides Convolute Error",
                                 f"An error occurred:\n{e}")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

    def _change_scale(self, panel):
        """Updates the color scale (Linear, Log, Power, etc.) for the specified panel.

        IMPORTANT: this method must persist the user's choice into
        ``self.scale_left`` / ``self.scale_right`` so that subsequent calls to
        ``display_image`` (e.g. after a colormap change) preserve the scale
        instead of falling back to Linear.
        """
        # Use the exact variable names from _init_plots
        if panel == 'left':
            combo = self.scale_combo_left
            canvas = self.slice1_canvas
            ax = self.slice1_plotter.ax
        else:
            combo = self.scale_combo_right
            canvas = self.slice2_canvas
            ax = self.slice2_plotter.ax

        # Persist the choice immediately, BEFORE any early-return below, so
        # that even if no image is loaded yet, the next display_image call
        # honours the selection.
        scale_type = combo.currentText()
        if panel == 'left':
            self.scale_left = scale_type
        else:
            self.scale_right = scale_type

        # Ensure an image is actually loaded before trying to scale it
        if getattr(ax, 'images', None) is None or not ax.images:
            return

        im = ax.images[0]
        data = im.get_array()

        # Safely calculate min and max, ignoring NaNs
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return

        vmin, vmax = float(np.nanmin(valid_data)), float(np.nanmax(valid_data))
        if vmin == vmax:
            vmax = vmin + 1.0

        # Apply the selected mathematical normalization
        if scale_type == 'Log':
            # Log scale cannot handle zero or negative numbers
            vmin_log = _log_norm_vmin(valid_data)
            im.set_norm(LogNorm(vmin=vmin_log, vmax=vmax))
        elif scale_type == 'Power (γ=2)':
            im.set_norm(PowerNorm(gamma=2.0, vmin=vmin, vmax=vmax))
        elif scale_type in ['Square Root', 'Sqrt']:
            im.set_norm(PowerNorm(gamma=0.5, vmin=max(0.0, vmin), vmax=vmax))
        else:
            # Default fallback is Linear
            im.set_norm(Normalize(vmin=vmin, vmax=vmax))

        # Redraw the updated image
        canvas.draw()
