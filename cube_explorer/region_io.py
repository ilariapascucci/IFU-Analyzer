"""
GUI-free pixel<->sky conversion for region shapes, with a small text-file
format to save and load them. Lets a region drawn on one cube be placed on
a different cube, at the same sky position, instead of being re-measured
by eye.
"""

import numpy as np
from astropy.coordinates import SkyCoord

SHAPE_NAMES = {"*": "POINT", "▭": "RECTANGLE", "○": "CIRCLE", "⬭": "ELLIPSE", "✏": "POLYGON"}
SHAPE_SYMBOLS = {v: k for k, v in SHAPE_NAMES.items()}
N_ELLIPSE_SAMPLES = 64
_AXIS_STEP_PX = 1.0  # how far (in source pixels) the saved "axis marker" points sit from center


def region_shape_to_sky(shape, coords, wcs2d, lasso_verts=None):
    """
    Convert a region selection's boundary to sky coordinates. For a
    rectangle, circle, or ellipse, also add two points marking which way
    the source pixel grid pointed on the sky (for sky_to_region_shape's
    later rotation check).

    Args:
        shape (str): Shape symbol, one of "*", "▭", "○", "⬭", "✏".
        coords (tuple): (x, y) for a point, else (xmin, xmax, ymin, ymax).
        wcs2d (astropy.wcs.WCS): 2D celestial WCS of the source cube.
        lasso_verts (array, optional): Polygon vertices; required if shape is "✏".

    Returns:
        result (dict): {"shape": shape, "pixel_points": ndarray (N,2) columns
            (x, y), "sky_points": astropy.coordinates.SkyCoord (length N),
            "sky_center"/"sky_axis_x"/"sky_axis_y": SkyCoord or None (None
            for point/polygon, which need no rotation check)}

    Raises:
        ValueError: wcs2d is None/has no celestial axes, or shape is "✏"
            with lasso_verts not given.
    """
    if wcs2d is None or not getattr(wcs2d, "has_celestial", False):
        raise ValueError("No usable celestial WCS to convert this region to sky coordinates.")

    if shape == "*":
        x, y = coords
        pixel_points = np.array([[float(x), float(y)]])
    elif shape == "✏":
        if lasso_verts is None:
            raise ValueError("Polygon selection has no stored vertices to save.")
        pixel_points = np.asarray(lasso_verts, dtype=float)
    elif shape in ("○", "⬭"):
        xmin, xmax, ymin, ymax = coords
        width, height = xmax - xmin + 1, ymax - ymin + 1
        cx, cy = xmin + (width - 1) / 2.0, ymin + (height - 1) / 2.0
        if shape == "○":
            # Matches _any_touch_circle_mask: radius is the smaller half-extent.
            a = b = min(max((width - 1) / 2.0, 1), max((height - 1) / 2.0, 1))
        else:
            a, b = max((width - 1) / 2.0, 0.5), max((height - 1) / 2.0, 0.5)
        angles = np.linspace(0, 2 * np.pi, N_ELLIPSE_SAMPLES, endpoint=False)
        pixel_points = np.column_stack([cx + a * np.cos(angles), cy + b * np.sin(angles)])
    elif shape == "▭":
        xmin, xmax, ymin, ymax = coords
        pixel_points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float)
    else:
        raise ValueError(f"Unknown shape symbol: {shape!r}")

    sky_points = wcs2d.pixel_to_world(pixel_points[:, 0], pixel_points[:, 1])

    sky_center = sky_axis_x = sky_axis_y = None
    if shape in ("▭", "○", "⬭"):
        # These two extra points record the source grid's orientation on
        # the sky, so sky_to_region_shape can later check for a rotation.
        center = pixel_points.mean(axis=0)
        sky_center = wcs2d.pixel_to_world(center[0], center[1])
        sky_axis_x = wcs2d.pixel_to_world(center[0] + _AXIS_STEP_PX, center[1])
        sky_axis_y = wcs2d.pixel_to_world(center[0], center[1] + _AXIS_STEP_PX)

    return {"shape": shape, "pixel_points": pixel_points, "sky_points": sky_points,
            "sky_center": sky_center, "sky_axis_x": sky_axis_x, "sky_axis_y": sky_axis_y}


def sky_to_region_shape(shape, sky_points, sky_center, sky_axis_x, sky_axis_y, wcs2d,
                         target_shape=None, angle_tol_deg=5.0):
    """
    Convert saved sky points back to pixel coordinates on a target cube's
    WCS, ready for RegionSelector.create_region or on_lasso_select. Falls
    back to a polygon if the target grid is rotated relative to the source.

    Args:
        shape (str): Shape symbol read back from the saved file.
        sky_points (astropy.coordinates.SkyCoord): Boundary points, length N.
        sky_center, sky_axis_x, sky_axis_y (SkyCoord or None): The saved axis
            markers; required (non-None) when shape is "▭"/"○"/"⬭".
        wcs2d (astropy.wcs.WCS): 2D celestial WCS of the target cube.
        target_shape (tuple, optional): (height, width) of the target image,
            for the off-frame check and clipping. None skips both.
        angle_tol_deg (float): How far from due-east/due-north (degrees) the
            axis markers may drift and still count as "not rotated".

    Returns:
        result (dict): {"shape": shape (possibly changed to "✏"),
            "coords": (x, y) or (xmin, xmax, ymin, ymax) or None,
            "verts": ndarray (N,2) or None} -- exactly one of
            "coords"/"verts" is set, matching what create_region /
            on_lasso_select expect.

    Raises:
        ValueError: wcs2d is None/has no celestial axes; every reprojected
            point is non-finite; or (when target_shape is given) every point
            falls outside it -- i.e. no sky overlap with the target cube at all.
    """
    if wcs2d is None or not getattr(wcs2d, "has_celestial", False):
        raise ValueError("Target cube has no usable celestial WCS to place this region on.")

    px, py = wcs2d.world_to_pixel(sky_points)
    px = np.atleast_1d(np.asarray(px, dtype=float))
    py = np.atleast_1d(np.asarray(py, dtype=float))
    finite = np.isfinite(px) & np.isfinite(py)
    if not finite.any():
        raise ValueError("This region's sky position does not resolve to any pixel on the target cube.")

    if target_shape is not None:
        h, w = target_shape
        in_frame = finite & (px >= -0.5) & (px <= w - 0.5) & (py >= -0.5) & (py <= h - 0.5)
        if not in_frame.any():
            raise ValueError("This region falls entirely outside the target cube's frame.")
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)

    if shape == "*":
        return {"shape": shape, "coords": (float(px[0]), float(py[0])), "verts": None}
    if shape == "✏":
        return {"shape": shape, "coords": None, "verts": np.column_stack([px, py])}

    # Rectangle/circle/ellipse: check the axis markers to decide if the
    # shape stayed axis-aligned, or must fall back to a polygon.
    cx, cy = wcs2d.world_to_pixel(sky_center)
    axx, axy = wcs2d.world_to_pixel(sky_axis_x)
    ayx, ayy = wcs2d.world_to_pixel(sky_axis_y)
    cx, cy, axx, axy, ayx, ayy = (float(v) for v in (cx, cy, axx, axy, ayx, ayy))
    angle_x = np.degrees(np.arctan2(axy - cy, axx - cx))          # ~0 deg if still due east
    angle_y = np.degrees(np.arctan2(ayy - cy, ayx - cx))          # ~90 deg if still due north
    # Only a true match counts as "aligned" -- a flip or swap falls back to
    # a polygon instead of risking a box with the wrong width and height.
    aligned = abs(angle_x) <= angle_tol_deg and abs(angle_y - 90) <= angle_tol_deg

    if aligned:
        return {"shape": shape, "coords": (float(px.min()), float(px.max()), float(py.min()), float(py.max())),
                "verts": None}
    return {"shape": "✏", "coords": None, "verts": np.column_stack([px, py])}


def write_region_wcs_txt(path, shape, source_desc, pixel_points, sky_points,
                          sky_center=None, sky_axis_x=None, sky_axis_y=None):
    """
    Write a region's boundary (and, for rectangle/circle/ellipse, its axis
    markers -- see sky_to_region_shape) to a small CSV-with-comment-header
    text file, in the same style as gui_2d.py's contour export.

    Args:
        path (str): Destination file path.
        shape (str): Shape symbol being saved.
        source_desc (str): Free-text note (e.g. source FITS filename) for
            the file's header comment; purely informational.
        pixel_points (numpy.ndarray): (N,2) source-pixel coordinates, kept
            for reference only -- never read back for reconstruction.
        sky_points (astropy.coordinates.SkyCoord): (N,) boundary sky coordinates.
        sky_center, sky_axis_x, sky_axis_y (SkyCoord, optional): Axis
            markers; omitted for point/polygon shapes.

    Returns:
        None: Writes `path` as a side effect.
    """
    lines = [
        "# IFU Analyzer - Region Export (WCS)\n",
        f"# shape: {SHAPE_NAMES[shape]}\n",
        f"# source: {source_desc}\n",
        f"# n_points: {len(pixel_points)}\n",
        "# columns: x_pixel, y_pixel, RA_deg, Dec_deg\n",
    ]
    for (x, y), sky in zip(pixel_points, sky_points):
        lines.append(f"{x:.4f}, {y:.4f}, {sky.ra.deg:.8f}, {sky.dec.deg:.8f}\n")
    if sky_center is not None:
        lines.append("# axis_markers: center_RA_deg, center_Dec_deg, "
                      "x_marker_RA_deg, x_marker_Dec_deg, y_marker_RA_deg, y_marker_Dec_deg\n")
        lines.append(f"{sky_center.ra.deg:.8f}, {sky_center.dec.deg:.8f}, "
                      f"{sky_axis_x.ra.deg:.8f}, {sky_axis_x.dec.deg:.8f}, "
                      f"{sky_axis_y.ra.deg:.8f}, {sky_axis_y.dec.deg:.8f}\n")
    with open(path, "w") as f:
        f.writelines(lines)


def read_region_wcs_txt(path):
    """
    Inverse of write_region_wcs_txt.

    Args:
        path (str): File to read.

    Returns:
        result (dict): {"shape": shape symbol, "sky_points": SkyCoord (N,),
            "sky_center"/"sky_axis_x"/"sky_axis_y": SkyCoord or None}

    Raises:
        ValueError: missing/unrecognized shape header, wrong point count for
            that shape, a malformed row, or any RA/Dec value missing or
            non-finite. There is no pixel-only fallback -- a file without
            valid sky coordinates is unusable, not silently reprojectable.
    """
    shape_token = None
    boundary_rows, marker_row = [], None
    expect_marker_next = False
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                low = line.lower()
                if low.startswith("# shape:"):
                    shape_token = line.split(":", 1)[1].strip().upper()
                elif low.startswith("# axis_markers:"):
                    expect_marker_next = True
                continue
            if expect_marker_next:
                marker_row = line
                expect_marker_next = False
            else:
                boundary_rows.append(line)

    if shape_token not in SHAPE_SYMBOLS:
        raise ValueError(f"Unrecognized or missing shape in {path!r}: {shape_token!r}")
    shape = SHAPE_SYMBOLS[shape_token]

    if shape_token == "POINT" and len(boundary_rows) != 1:
        raise ValueError(f"{path!r}: a POINT region needs exactly 1 point, found {len(boundary_rows)}.")
    if shape_token == "RECTANGLE" and len(boundary_rows) != 4:
        raise ValueError(f"{path!r}: a RECTANGLE region needs exactly 4 points, found {len(boundary_rows)}.")
    if shape_token in ("CIRCLE", "ELLIPSE", "POLYGON") and len(boundary_rows) < 3:
        raise ValueError(f"{path!r}: too few points ({len(boundary_rows)}) for a {shape_token} region.")

    ra, dec = [], []
    for row in boundary_rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 4:
            raise ValueError(f"Malformed row in {path!r}: {row!r}")
        try:
            r, d = float(parts[2]), float(parts[3])
        except ValueError:
            raise ValueError(f"Missing/invalid RA-Dec in {path!r}: {row!r}")
        if not (np.isfinite(r) and np.isfinite(d)):
            raise ValueError(f"Missing/invalid RA-Dec in {path!r}: {row!r}")
        ra.append(r)
        dec.append(d)
    sky_points = SkyCoord(ra=ra, dec=dec, unit="deg")

    needs_markers = shape_token in ("RECTANGLE", "CIRCLE", "ELLIPSE")
    sky_center = sky_axis_x = sky_axis_y = None
    if needs_markers:
        if marker_row is None:
            raise ValueError(f"{path!r} is missing its axis-marker row (needed to check for rotation).")
        try:
            m = [float(p.strip()) for p in marker_row.split(",")]
        except ValueError:
            raise ValueError(f"Malformed axis-marker row in {path!r}: {marker_row!r}")
        if len(m) != 6 or not all(np.isfinite(m)):
            raise ValueError(f"Malformed axis-marker row in {path!r}: {marker_row!r}")
        sky_center = SkyCoord(ra=m[0], dec=m[1], unit="deg")
        sky_axis_x = SkyCoord(ra=m[2], dec=m[3], unit="deg")
        sky_axis_y = SkyCoord(ra=m[4], dec=m[5], unit="deg")

    return {"shape": shape, "sky_points": sky_points,
            "sky_center": sky_center, "sky_axis_x": sky_axis_x, "sky_axis_y": sky_axis_y}
