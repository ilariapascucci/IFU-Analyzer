"""
Resolution matching for 2D FITS: NaN-aware Gaussian convolution and a
self-contained flux-conserving spherical-polygon reprojection (a drop-in
equivalent of ``reproject.reproject_exact`` with no ``reproject`` dependency).

GUI-free - imported by ``gui_2d.py`` (the dual 2D viewer) and safe to use
from scripts/notebooks. Public helpers: ``gaussian_convolve_match``,
``reproject_exact``, ``build_target_wcs_for_pair`` / ``build_target_wcs_for_batch``,
``_get_pixel_size_arcsec``, ``_make_2d_wcs``, ``_wcs_footprints_overlap``,
``_is_per_pixel_flux_unit``, ``_log_norm_vmin``.
"""

import warnings

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales  # used by: _get_pixel_size_arcsec
from astropy.convolution import convolve_fft, Gaussian2DKernel  # used by: gaussian_convolve_match
from astropy.coordinates import SkyCoord  # used by: build_target_wcs_for_batch
import astropy.units as u  # used by: build_target_wcs_for_batch

# Suppress FITS warning messages about header formatting
warnings.filterwarnings('ignore', category=Warning, module='astropy.wcs.wcs')

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
    # [ux, uy] is already the correctly-signed rotation+reflection matrix -
    # do not re-apply a sign correction (it mirrors rotated CD matrices).
    ux = cd[:, 0] / sx if sx > 0 else np.array([1.0, 0.0])
    uy = cd[:, 1] / sy if sy > 0 else np.array([0.0, 1.0])
    return ux, uy, sx, sy


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

    ux, uy, sx_deg, sy_deg = _decompose_cd(fine_wcs)
    pix_deg = target_pix_arcsec / 3600.0

    out = WCS(naxis=2)
    out.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    out.wcs.crval = [float(sky_centre.ra.deg), float(sky_centre.dec.deg)]
    out.wcs.cunit = ['deg', 'deg']
    out.wcs.cd = np.column_stack([ux * pix_deg, uy * pix_deg])

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
    ux, uy, _, _ = _decompose_cd(ref_wcs)
    pix_deg = target_pix_arcsec / 3600.0

    out = WCS(naxis=2)
    out.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    out.wcs.crval = [float(sky_centre.ra.deg), float(sky_centre.dec.deg)]
    out.wcs.cunit = ['deg', 'deg']
    out.wcs.cd = np.column_stack([ux * pix_deg, uy * pix_deg])
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
