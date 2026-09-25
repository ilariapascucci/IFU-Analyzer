"""
Region selection and analysis functionality
"""

import numpy as np
from matplotlib.widgets import RectangleSelector, EllipseSelector, LassoSelector
from matplotlib.path import Path
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.lines import Line2D


def mask_from_shape_coords(shape, coords, verts, target_shape):
    """
    Build a full-image boolean mask from a shape symbol and its coords/verts,
    without needing a live interactive RegionSelector/canvas -- the headless
    equivalent of what create_region/on_lasso_select's on_select_callback
    receives (a bounding-box mask) and embeds into a full-image mask (see
    gui_region_selection.py's _handle_region_select). Used to apply an
    imported region to data with no on-screen selector at all, such as the
    rotational-diagram map's per-cube native grids.

    Args:
        shape (str): Shape symbol -- "*", "▭", "○", "⬭", or "✏".
        coords (tuple): (xmin, xmax, ymin, ymax) for "*"/"▭"/"○"/"⬭"; ignored
            for "✏". For "*", (x, y) is also accepted.
        verts (numpy.ndarray): (N,2) polygon vertices; required for "✏",
            ignored otherwise.
        target_shape (tuple): (height, width) of the full image to mask.

    Returns:
        mask (numpy.ndarray): Boolean mask, shape == target_shape.
    """
    h_img, w_img = target_shape
    full_mask = np.zeros((h_img, w_img), dtype=bool)

    if shape == "*":
        x, y = coords
        x, y = int(round(x)), int(round(y))
        if 0 <= y < h_img and 0 <= x < w_img:
            full_mask[y, x] = True
        return full_mask

    if shape == "✏":
        verts = np.asarray(verts, dtype=float)
        xmin, ymin = verts.min(axis=0)
        xmax, ymax = verts.max(axis=0)
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        width, height = xmax - xmin + 1, ymax - ymin + 1
        box_mask = _any_touch_polygon_mask(verts, xmin, ymin, width, height)
    else:
        xmin, xmax, ymin, ymax = (int(c) for c in coords)
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        height = max(1, ymax - ymin + 1)
        width = max(1, xmax - xmin + 1)
        if shape == "○":
            box_mask = _any_touch_circle_mask(width, height)
        elif shape == "⬭":
            box_mask = _any_touch_ellipse_mask(width, height)
        else:  # rectangle
            box_mask = np.ones((height, width), dtype=bool)

    # Same embed as _handle_region_select: clip the box mask into the full
    # image, since an imported region can extend past this cube's own edge.
    y0, x0 = max(int(ymin), 0), max(int(xmin), 0)
    y1 = min(y0 + box_mask.shape[0] - max(-int(ymin), 0), h_img)
    x1 = min(x0 + box_mask.shape[1] - max(-int(xmin), 0), w_img)
    by0, bx0 = max(-int(ymin), 0), max(-int(xmin), 0)
    if y1 > y0 and x1 > x0:
        full_mask[y0:y1, x0:x1] = box_mask[by0:by0 + (y1 - y0), bx0:bx0 + (x1 - x0)]
    return full_mask


def _any_touch_circle_mask(width, height):
    """Boolean mask for a circle centered in a (height, width) box: a pixel
    counts as in if its own unit square touches the circle at all, found via
    the closest point of that square (clamped toward the centre) to the edge.

    Args:
        width (int): Bounding-box width in pixels
        height (int): Bounding-box height in pixels

    Returns:
        mask (np.ndarray): 2D bool array, shape (height, width)
    """
    yy, xx = np.indices((height, width))
    cx_rel = (width - 1) / 2.0
    cy_rel = (height - 1) / 2.0
    r = min(max((width - 1) / 2.0, 1), max((height - 1) / 2.0, 1))
    closest_x = np.clip(cx_rel, xx - 0.5, xx + 0.5)
    closest_y = np.clip(cy_rel, yy - 0.5, yy + 0.5)
    return (closest_x - cx_rel) ** 2 + (closest_y - cy_rel) ** 2 <= r ** 2


def _any_touch_ellipse_mask(width, height):
    """Same any-touch test as _any_touch_circle_mask, done in a space
    rescaled by the ellipse's own (a, b) axes so it becomes a unit circle.

    Args:
        width (int): Bounding-box width in pixels
        height (int): Bounding-box height in pixels

    Returns:
        mask (np.ndarray): 2D bool array, shape (height, width)
    """
    yy, xx = np.indices((height, width))
    cx_rel = (width - 1) / 2.0
    cy_rel = (height - 1) / 2.0
    a = max((width - 1) / 2.0, 0.5)
    b = max((height - 1) / 2.0, 0.5)
    x_lo, x_hi = (xx - 0.5 - cx_rel) / a, (xx + 0.5 - cx_rel) / a
    y_lo, y_hi = (yy - 0.5 - cy_rel) / b, (yy + 0.5 - cy_rel) / b
    closest_x = np.clip(0.0, x_lo, x_hi)
    closest_y = np.clip(0.0, y_lo, y_hi)
    return closest_x ** 2 + closest_y ** 2 <= 1


def _segment_intersects_boxes_vec(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
    """Liang-Barsky test: does the segment (x0,y0)-(x1,y1) intersect each of
    many axis-aligned boxes? Tests each edge independently, so it stays
    correct at a self-intersecting polygon (a real risk for a freehand lasso).

    Args:
        x0, y0, x1, y1 (float): Segment endpoints
        xmin, xmax, ymin, ymax (np.ndarray): Same-shape box bounds, one entry per box

    Returns:
        touches (np.ndarray): Bool array, same shape as xmin
    """
    dx, dy = x1 - x0, y1 - y0
    t0 = np.zeros(xmin.shape, dtype=float)
    t1 = np.ones(xmin.shape, dtype=float)
    valid = np.ones(xmin.shape, dtype=bool)

    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if p == 0:
            valid &= (q >= 0)
            continue
        t = q / p
        if p < 0:
            valid &= (t <= t1)
            t0 = np.where(t > t0, t, t0)
        else:
            valid &= (t >= t0)
            t1 = np.where(t < t1, t, t1)
    return valid & (t0 <= t1)


def _any_touch_polygon_mask(verts, xmin, ymin, width, height):
    """Boolean mask for an arbitrary (possibly self-intersecting) polygon: a
    pixel counts as in if its own unit square touches the polygon at all.
    Starts from point-in-polygon, then ORs in an edge-vs-box test per edge.

    Args:
        verts (np.ndarray): Polygon vertices, absolute data coordinates
        xmin (int): Mask's column 0 in absolute data coordinates
        ymin (int): Mask's row 0 in absolute data coordinates
        width (int): Mask width in pixels
        height (int): Mask height in pixels

    Returns:
        mask (np.ndarray): 2D bool array, shape (height, width)
    """
    verts = np.asarray(verts, dtype=float)
    yy, xx = np.indices((height, width))
    box_xmin = (xx + xmin - 0.5).astype(float)
    box_xmax = (xx + xmin + 0.5).astype(float)
    box_ymin = (yy + ymin - 0.5).astype(float)
    box_ymax = (yy + ymin + 0.5).astype(float)

    path = Path(verts)
    centers = np.column_stack([(xx + xmin).ravel(), (yy + ymin).ravel()])
    mask = path.contains_points(centers).reshape(height, width)

    n = len(verts)
    for i in range(n):
        j = (i + 1) % n
        touched = _segment_intersects_boxes_vec(
            verts[i, 0], verts[i, 1], verts[j, 0], verts[j, 1],
            box_xmin, box_xmax, box_ymin, box_ymax)
        mask |= touched
    return mask


def _mask_boundary_path(mask, xmin, ymin):
    """Trace a boolean mask's own pixel edges into a stroke-only Path (a
    "staircase" outline), so the drawn shape always matches the mask exactly
    instead of an idealized smooth circle/ellipse standing in for it.

    Args:
        mask (np.ndarray): 2D bool array
        xmin (int): Mask's column 0 in absolute data coordinates
        ymin (int): Mask's row 0 in absolute data coordinates

    Returns:
        path (matplotlib.path.Path or None): None if mask has no True pixels
    """
    ny, nx = mask.shape
    verts, codes = [], []
    for iy in range(ny):
        for ix in range(nx):
            if not mask[iy, ix]:
                continue
            x0, x1 = xmin + ix - 0.5, xmin + ix + 0.5
            y0, y1 = ymin + iy - 0.5, ymin + iy + 0.5
            if iy == 0 or not mask[iy - 1, ix]:
                verts += [(x0, y0), (x1, y0)]
                codes += [Path.MOVETO, Path.LINETO]
            if iy == ny - 1 or not mask[iy + 1, ix]:
                verts += [(x0, y1), (x1, y1)]
                codes += [Path.MOVETO, Path.LINETO]
            if ix == 0 or not mask[iy, ix - 1]:
                verts += [(x0, y0), (x0, y1)]
                codes += [Path.MOVETO, Path.LINETO]
            if ix == nx - 1 or not mask[iy, ix + 1]:
                verts += [(x1, y0), (x1, y1)]
                codes += [Path.MOVETO, Path.LINETO]
    if not verts:
        return None
    return Path(verts, codes)


class InteractivePatch:
    """
    Makes patches draggable and resizable with visual handles.
    
    Supports dragging the entire shape and resizing from corners/edges.
    Works with Rectangle, Circle, Ellipse, and Polygon patches.
    For Polygon (lasso), only dragging is supported.
    
    Attributes:
        ax: matplotlib axes containing the patch
        patch: the matplotlib patch to make interactive
        shape_type: 'rectangle', 'circle', 'ellipse', or 'polygon'
        on_change_callback: function called when patch is modified
        handles: list of Line2D objects for resize handles
    """
    
    HANDLE_HIT_RADIUS_PX = 8  # click tolerance for resize handles, in screen pixels

    def __init__(self, ax, patch, shape_type, on_change_callback, coords=None):
        """Initialize an interactive patch. Does not connect event handlers
        or create handles yet; call connect() to make it interactive.

        Args:
            ax (matplotlib.axes.Axes): Axes the patch is drawn on.
            patch (matplotlib.patches.Patch): Patch object to make interactive.
            shape_type (str): 'rectangle', 'circle', 'ellipse', or 'polygon'.
            on_change_callback (callable): function(new_coords) to call on changes.
            coords (tuple): Initial (xmin, xmax, ymin, ymax), or None.

        Returns:
            None
        """
        self.ax = ax
        self.patch = patch
        self.shape_type = shape_type
        self.on_change_callback = on_change_callback
        self.coords = coords if coords else (0, 10, 0, 10)

        self.handles = []
        self.active_handle = None
        self.press_data = None
        self.handle_positions = []  # Store handle positions for click detection

        self.cidpress = None
        self.cidrelease = None
        self.cidmotion = None
        self._connected = False

    def connect(self):
        """Wire up event handlers and show resize handles.

        Args:
            None

        Returns:
            None
        """
        if self._connected:
            return
        ax = self.ax
        self.cidpress = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self._connected = True

        # Create resize handles
        if self.shape_type != 'polygon':
            self._create_handles()
            # Redraw canvas to show handles immediately
            ax.figure.canvas.draw_idle()

    def disconnect_handlers(self):
        """Unwire event handlers and hide resize handles, keeping the patch itself intact.

        Args:
            None

        Returns:
            None
        """
        if not self._connected:
            return
        try:
            self.ax.figure.canvas.mpl_disconnect(self.cidpress)
            self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
            self.ax.figure.canvas.mpl_disconnect(self.cidmotion)
        except Exception as e:
            print(f"[DEBUG] Error disconnecting event handlers: {e}")

        self.cidpress = None
        self.cidrelease = None
        self.cidmotion = None
        self.active_handle = None
        self.press_data = None

        for i, handle in enumerate(self.handles):
            try:
                handle.remove()
            except Exception as e:
                print(f"[DEBUG] Error removing handle {i}: {e}")
        self.handles = []
        self.handle_positions = []
        self._connected = False

        try:
            self.ax.figure.canvas.draw_idle()
        except Exception as e:
            print(f"[DEBUG] Error redrawing after disconnect_handlers: {e}")
    
    def _create_handles(self):
        """Create visual resize handles for the patch.

        Args:
            None

        Returns:
            None
        """
        xmin, xmax, ymin, ymax = self.coords
        
        if self.shape_type == 'rectangle':
            # 8 handles: 4 corners + 4 edges
            positions = [
                (xmin, ymin), (xmax, ymin),  # bottom left, bottom right
                (xmin, ymax), (xmax, ymax),  # top left, top right
                ((xmin+xmax)/2, ymin), ((xmin+xmax)/2, ymax),  # bottom mid, top mid
                (xmin, (ymin+ymax)/2), (xmax, (ymin+ymax)/2)  # left mid, right mid
            ]
        elif self.shape_type == 'circle':
            # Just one handle on the circle edge for resizing
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            r = max(xmax - xmin, ymax - ymin) / 2
            positions = [(cx + r, cy)]
        elif self.shape_type == 'ellipse':
            # 4 corner handles
            positions = [
                (xmin, ymin), (xmax, ymin),
                (xmin, ymax), (xmax, ymax)
            ]
        else:
            positions = []
        
        # Store handle positions for distance-based click detection
        self.handle_positions = positions
        
        print(f"[DEBUG] Creating {len(positions)} handles: {positions}")
        
        # Extract X and Y coordinates separately for scatter
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        try:
            # Create all handles in a single scatter plot for better control
            scatter_collection = self.ax.scatter(xs, ys, s=200, c='red', marker='o', 
                                              zorder=1000, edgecolors='darkred', linewidths=2)
            scatter_collection.handle_positions = positions
            # Mark this as a handle collection for later identification
            scatter_collection._is_handle_collection = True
            scatter_collection._associated_patch = self.patch
            self.handles = [scatter_collection]
            print(f"[DEBUG] Successfully created scatter collection with {len(positions)} points")
        except Exception as e:
            print(f"[DEBUG] Error creating scatter handles: {e}")
            self.handles = []

    
    def on_press(self, event):
        """Handle mouse press to start dragging.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Press event.

        Returns:
            None
        """
        if event.inaxes != self.ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        
        print(f"[DEBUG] on_press called: event at ({event.xdata:.1f}, {event.ydata:.1f}), shape type: {self.shape_type}")
        
        # Check if clicking on a handle (using screen-pixel distance, so the hit
        # radius is consistent regardless of the image's data pixel scale or zoom)
        click_x, click_y = event.xdata, event.ydata
        click_disp = self.ax.transData.transform((click_x, click_y))
        for i, (handle_x, handle_y) in enumerate(self.handle_positions):
            handle_disp = self.ax.transData.transform((handle_x, handle_y))
            distance = np.hypot(click_disp[0] - handle_disp[0], click_disp[1] - handle_disp[1])
            if distance < self.HANDLE_HIT_RADIUS_PX:
                self.active_handle = i
                self.press_data = {
                    'xpress': event.xdata,
                    'ypress': event.ydata,
                    'coords': self.coords
                }
                print(f"[DEBUG] Clicked on handle {i} at distance {distance}")
                return
        
        # Check if clicking on the patch itself (to move entire shape)
        # For rectangles, check if within bounds
        if self.shape_type == 'rectangle':
            xmin, xmax, ymin, ymax = self.coords
            if xmin <= click_x <= xmax and ymin <= click_y <= ymax:
                self.active_handle = -1  # -1 means move the whole patch
                self.press_data = {
                    'xpress': event.xdata,
                    'ypress': event.ydata,
                    'coords': self.coords
                }
                print(f"[DEBUG] Clicked inside rectangle to move at ({click_x:.1f}, {click_y:.1f}). Coords: {self.coords}")
                return
        
        # For circles, check distance from center
        elif self.shape_type == 'circle':
            xmin, xmax, ymin, ymax = self.coords
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            r = max(xmax - xmin, ymax - ymin) / 2
            distance = np.sqrt((click_x - cx)**2 + (click_y - cy)**2)
            if distance <= r:
                self.active_handle = -1
                self.press_data = {
                    'xpress': event.xdata,
                    'ypress': event.ydata,
                    'coords': self.coords
                }
                print(f"[DEBUG] Clicked inside circle to move at distance {distance} from center")
                return
        
        # For ellipses, check if within bounding box (simplified)
        elif self.shape_type == 'ellipse':
            xmin, xmax, ymin, ymax = self.coords
            if xmin <= click_x <= xmax and ymin <= click_y <= ymax:
                self.active_handle = -1
                self.press_data = {
                    'xpress': event.xdata,
                    'ypress': event.ydata,
                    'coords': self.coords
                }
                print(f"[DEBUG] Clicked inside ellipse to move")
                return
        
        # For polygons, check if within bounding box
        elif self.shape_type == 'polygon':
            xmin, xmax, ymin, ymax = self.coords
            if xmin <= click_x <= xmax and ymin <= click_y <= ymax:
                self.active_handle = -1
                self.press_data = {
                    'xpress': event.xdata,
                    'ypress': event.ydata,
                    'coords': self.coords,
                    # Snapshot the real vertices at press time so each motion
                    # tick applies total displacement, not compounding drift.
                    'verts': np.asarray(self.patch.lasso_verts).copy()
                }
                print(f"[DEBUG] Clicked inside polygon to move")
                return
        
        print(f"[DEBUG] Click not on patch or handles. Coords: {self.coords}")

    
    def on_release(self, event):
        """Handle mouse release.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Release event.

        Returns:
            None
        """
        self.active_handle = None
        self.press_data = None

    def on_motion(self, event):
        """Handle mouse motion - update patch while dragging.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Motion event.

        Returns:
            None
        """
        if self.press_data is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        # Snap every move/resize to a whole-pixel step, matching how the
        # mask itself is always defined on whole pixels.
        dx = round(event.xdata - self.press_data['xpress'])
        dy = round(event.ydata - self.press_data['ypress'])

        old_coords = self.press_data['coords']
        xmin, xmax, ymin, ymax = old_coords
        
        if self.active_handle == -1:
            # Move entire patch
            new_coords = (xmin + dx, xmax + dx, ymin + dy, ymax + dy)
            self._update_patch(new_coords)
        else:
            # Resize from handle
            new_coords = self._resize_from_handle(old_coords, dx, dy)
            if new_coords:
                self._update_patch(new_coords)
    
    def _resize_from_handle(self, coords, dx, dy):
        """Calculate new coordinates when resizing from a handle.

        Args:
            coords (tuple): Current (xmin, xmax, ymin, ymax).
            dx (float): Horizontal displacement since press.
            dy (float): Vertical displacement since press.

        Returns:
            coords (tuple): New (xmin, xmax, ymin, ymax), or None if the
                active handle/shape combination has no resize behavior.
        """
        xmin, xmax, ymin, ymax = coords
        h_idx = self.active_handle
        
        if self.shape_type == 'rectangle':
            # 8 handles for rectangle
            if h_idx == 0:  # bottom-left
                return (xmin + dx, xmax, ymin + dy, ymax)
            elif h_idx == 1:  # bottom-right
                return (xmin, xmax + dx, ymin + dy, ymax)
            elif h_idx == 2:  # top-left
                return (xmin + dx, xmax, ymin, ymax + dy)
            elif h_idx == 3:  # top-right
                return (xmin, xmax + dx, ymin, ymax + dy)
            elif h_idx == 4:  # bottom-mid
                return (xmin, xmax, ymin + dy, ymax)
            elif h_idx == 5:  # top-mid
                return (xmin, xmax, ymin, ymax + dy)
            elif h_idx == 6:  # left-mid
                return (xmin + dx, xmax, ymin, ymax)
            elif h_idx == 7:  # right-mid
                return (xmin, xmax + dx, ymin, ymax)
        
        elif self.shape_type == 'circle':
            # Resize circle from edge handle, rounded so the radius only
            # ever changes in whole-pixel steps.
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            old_r = max(xmax - xmin, ymax - ymin) / 2
            new_r = max(1, round(old_r + max(dx, dy) / 2))
            return (cx - new_r, cx + new_r, cy - new_r, cy + new_r)
        
        elif self.shape_type == 'ellipse':
            # Resize ellipse from corners
            if h_idx == 0:  # bottom-left
                return (xmin + dx, xmax, ymin + dy, ymax)
            elif h_idx == 1:  # bottom-right
                return (xmin, xmax + dx, ymin + dy, ymax)
            elif h_idx == 2:  # top-left
                return (xmin + dx, xmax, ymin, ymax + dy)
            elif h_idx == 3:  # top-right
                return (xmin, xmax + dx, ymin, ymax + dy)
        
        return None
    
    def _update_patch(self, new_coords):
        """Update the patch geometry and redraw handles.

        Args:
            new_coords (tuple): New (xmin, xmax, ymin, ymax) for the patch.

        Returns:
            None
        """
        xmin, xmax, ymin, ymax = new_coords
        
        # Update coordinates
        self.coords = new_coords
        
        # Update patch based on type
        if self.shape_type == 'rectangle':
            width = xmax - xmin
            height = ymax - ymin
            self.patch.set_xy((xmin, ymin))
            self.patch.set_width(width)
            self.patch.set_height(height)
        
        elif self.shape_type in ('circle', 'ellipse'):
            # The patch is a staircase outline of the any-touch mask, not an
            # idealized shape, so rebuild it rather than mutate the geometry.
            ixmin, ixmax = int(round(xmin)), int(round(xmax))
            iymin, iymax = int(round(ymin)), int(round(ymax))
            width = max(1, ixmax - ixmin + 1)
            height = max(1, iymax - iymin + 1)
            mask_fn = _any_touch_circle_mask if self.shape_type == 'circle' else _any_touch_ellipse_mask
            path = _mask_boundary_path(mask_fn(width, height), ixmin, iymin)
            if path is not None:
                self.patch.set_path(path)

        elif self.shape_type == 'polygon':
            # Apply total displacement-since-press to the vertex snapshot
            # from on_press; using the current vertices would double-count.
            if self.press_data and 'verts' in self.press_data:
                old_coords = self.press_data['coords']
                old_xmin, old_xmax, old_ymin, old_ymax = old_coords
                dx = xmin - old_xmin
                dy = ymin - old_ymin
                new_verts = self.press_data['verts'] + np.array([dx, dy])
                self.patch.lasso_verts = new_verts
                # Staircase outline of the actual any-touch mask, not the
                # raw vertices -- rebuild it for the moved position.
                ixmin, ixmax = int(round(xmin)), int(round(xmax))
                iymin, iymax = int(round(ymin)), int(round(ymax))
                width = max(1, ixmax - ixmin + 1)
                height = max(1, iymax - iymin + 1)
                mask = _any_touch_polygon_mask(new_verts, ixmin, iymin, width, height)
                path = _mask_boundary_path(mask, ixmin, iymin)
                if path is not None:
                    self.patch.set_path(path)
        
        # Update handle positions
        self._update_handles()
        
        # Redraw canvas
        self.ax.figure.canvas.draw_idle()
        
        # Call callback with new coordinates
        print(f"[DEBUG] Calling on_change_callback with coords: {new_coords}")
        self.on_change_callback(new_coords)
    
    def _update_handles(self):
        """Update visual positions of resize handles.

        Args:
            None

        Returns:
            None
        """
        xmin, xmax, ymin, ymax = self.coords
        
        if self.shape_type == 'rectangle':
            positions = [
                (xmin, ymin), (xmax, ymin),
                (xmin, ymax), (xmax, ymax),
                ((xmin+xmax)/2, ymin), ((xmin+xmax)/2, ymax),
                (xmin, (ymin+ymax)/2), (xmax, (ymin+ymax)/2)
            ]
        
        elif self.shape_type == 'circle':
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            r = max(xmax - xmin, ymax - ymin) / 2
            positions = [(cx + r, cy)]
        
        elif self.shape_type == 'ellipse':
            positions = [
                (xmin, ymin), (xmax, ymin),
                (xmin, ymax), (xmax, ymax)
            ]
        else:
            positions = []
        
        # Update stored handle positions
        self.handle_positions = positions
        
        # Update scatter collection positions
        if len(self.handles) > 0:
            try:
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                self.handles[0].set_offsets(np.c_[xs, ys])
            except Exception as e:
                print(f"[DEBUG] Could not update handle positions: {e}")

    
    def disconnect(self):
        """Fully tear down: unwire handlers, remove handles, and drop the patch
        reference. Use this when the shape itself is being removed (e.g. erased).
        Safe to call even if the patch was never connect()-ed.

        Args:
            None

        Returns:
            None
        """
        print(f"[DEBUG] Disconnecting InteractivePatch for {self.shape_type}")
        self.disconnect_handlers()
        self.patch = None
        print(f"[DEBUG] InteractivePatch disconnect complete")

class RegionSelector:
    """
    Handles interactive region selection on matplotlib plots.
    
    This class provides functionality to select regions using different shapes:
    - Points (*)
    - Rectangles (▭)
    - Circles (○)
    - Ellipses (⬭)
    - Custom/Lasso (✏)
    
    The selector maintains state about the current selection mode and
    handles the creation of appropriate matplotlib selector widgets.

    Attributes:
        ax (matplotlib.axes.Axes): The axes on which to draw selections
        canvas (matplotlib.backends.backend_qt5agg.FigureCanvas): The canvas containing the axes
        on_select_callback (callable): Function to call when selection is complete
        selector (matplotlib.widgets.RectangleSelector/EllipseSelector/LassoSelector): Active selector widget
        active_shape (str): Currently active selection shape type
    """

    def __init__(self, ax, canvas, on_select_callback):
        """
        Initialize the region selector.

        Args:
            ax (matplotlib.axes.Axes): The axes to draw on
            canvas (matplotlib.backends.backend_qt5agg.FigureCanvas): The containing canvas
            on_select_callback (callable): Function called after selection is made.
                Will receive (mask, patch, coords) where:
                - mask is a boolean array showing selected pixels
                - patch is a matplotlib patch object for the selection
                - coords is the selection bounds (xmin,xmax,ymin,ymax)

        Returns:
            None
        """
        self.ax = ax
        self.canvas = canvas
        self.on_select_callback = on_select_callback
        self.selector = None
        self.active_shape = None
        self.interactive_patch = None  # Track the current interactive patch
        self.image = None  # Will store the current image for mask generation

    def activate_selector(self, shape):
        """Activate a matplotlib selector widget matching `shape` (point,
        rectangle, ellipse/circle, or lasso), disconnecting any existing
        selector first. Point selection is handled separately via clicks.

        Args:
            shape (str): The shape to activate, or None to deactivate selection.

        Returns:
            None
        """
        if self.selector is not None:
            self.selector.disconnect_events()
            self.selector = None

        self.active_shape = shape

        if shape == "*":
            # Pixel selection is handled by click events
            return
        elif shape == "▭":  # rectangle
            self.selector = RectangleSelector(
                self.ax, 
                self.on_select,
                useblit=True, 
                button=[1], 
                minspanx=2, 
                minspany=2, 
                spancoords='pixels',
                interactive=True
            )
            # Make sure selector is visible and active
            if hasattr(self.selector, 'set_visible'):
                self.selector.set_visible(True)
            if hasattr(self.selector, 'set_active'):
                self.selector.set_active(True)
        elif shape in ["○", "⬭"]:  # circle or ellipse
            self.selector = EllipseSelector(
                self.ax,
                self.on_select,
                interactive=True
            )
            # Make sure selector is visible and active
            if hasattr(self.selector, 'set_visible'):
                self.selector.set_visible(True)
            if hasattr(self.selector, 'set_active'):
                self.selector.set_active(True)
        elif shape == "✏":  # custom
            self.selector = LassoSelector(
                self.ax,
                self.on_lasso_select
            )

    def on_select(self, eclick, erelease):
        """Handle a completed rectangle/ellipse selection: converts the
        coordinates to ordered integers and creates the region.

        Args:
            eclick (matplotlib.backend_bases.MouseEvent): Mouse press event.
            erelease (matplotlib.backend_bases.MouseEvent): Mouse release event.

        Returns:
            None
        """
        # Get selection bounds
        x0, y0 = int(eclick.xdata), int(eclick.ydata)
        x1, y1 = int(erelease.xdata), int(erelease.ydata)
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        
        coords = (xmin, xmax, ymin, ymax)
        self.create_region(coords)

    def create_region(self, coords):
        """Create a region mask and visual patch based on current shape
        type, via _generate_mask_from_coords, then make it interactive.

        Args:
            coords (tuple): (xmin, xmax, ymin, ymax) region bounds.

        Returns:
            None
        """
        if self.active_shape == "○":
            shape_type = 'circle'
        elif self.active_shape == "⬭":
            shape_type = 'ellipse'
        else:
            shape_type = 'rectangle'

        mask, patch, _ = self._generate_mask_from_coords(coords, shape_type)

        # Call the callback with the results
        self.on_select_callback(mask, patch, coords)

        # Make patch interactive (draggable and resizable)
        def on_patch_change(new_coords):
            """Regenerate the mask and re-run the selection callback.

            Args:
                new_coords (tuple): New (xmin, xmax, ymin, ymax) region bounds.

            Returns:
                None
            """
            new_mask, _, _ = self._generate_mask_from_coords(new_coords, shape_type)
            print(f"[DEBUG] Patch changed to coords: {new_coords}, regenerated mask shape: {new_mask.shape}")
            self.on_select_callback(new_mask, patch, new_coords)

        # Disconnect previous interactive patch if any
        if self.interactive_patch is not None:
            print(f"[DEBUG] Disconnecting previous interactive patch")
            try:
                self.interactive_patch.disconnect()
            except Exception as e:
                print(f"[DEBUG] Error disconnecting previous patch: {e}")
            self.interactive_patch = None

        # Create new interactive patch
        self.interactive_patch = InteractivePatch(
            self.ax, patch, shape_type, on_patch_change, coords
        )

    def _generate_mask_from_coords(self, coords, shape_type):
        """
        Generate a mask and matching outline patch from coordinates and shape
        type. The single implementation used both for a freshly-drawn shape
        (via create_region) and for every drag/resize update.

        Args:
            coords (tuple): (xmin, xmax, ymin, ymax) region bounds
            shape_type (str): 'circle', 'ellipse', or 'rectangle'

        Returns:
            result (tuple): (mask, patch, coords) -- mask is a 2D bool array;
                patch is a matplotlib Patch tracing the mask's real edges
        """
        xmin, xmax, ymin, ymax = coords
        # Convert to integers for mask generation
        xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

        # Ensure proper ordering
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)

        height = max(1, ymax - ymin + 1)
        width = max(1, xmax - xmin + 1)

        if shape_type == 'circle':
            mask = _any_touch_circle_mask(width, height)
        elif shape_type == 'ellipse':
            mask = _any_touch_ellipse_mask(width, height)
        else:  # rectangle
            mask = np.ones((height, width), dtype=bool)

        # Outline traces the mask's own pixel edges exactly (a staircase),
        # so what's shown always matches what's actually included.
        if shape_type in ('circle', 'ellipse'):
            path = _mask_boundary_path(mask, xmin, ymin)
            patch = PathPatch(path, edgecolor='lime', facecolor='none', lw=1.5)
        else:
            patch = Rectangle((xmin, ymin), width-1, height-1,
                            edgecolor='lime', facecolor='none', lw=1.5)

        return mask, patch, coords

    def on_lasso_select(self, verts):
        """Handle custom shape selection made with the lasso tool.

        Args:
            verts (list): List of (x,y) vertex coordinates of the selection.

        Returns:
            None
        """
        verts = np.asarray(verts)
        xmin, ymin = np.min(verts, axis=0)
        xmax, ymax = np.max(verts, axis=0)

        # Convert to integers to handle decimal pixel coordinates
        xmin, ymin = int(xmin), int(ymin)
        xmax, ymax = int(xmax), int(ymax)

        width = xmax - xmin + 1
        height = ymax - ymin + 1
        mask = _any_touch_polygon_mask(verts, xmin, ymin, width, height)

        path = _mask_boundary_path(mask, xmin, ymin)
        patch = PathPatch(path, edgecolor='lime', facecolor='none', lw=1.5)
        # Store the real vertices on the patch (separate from its staircase
        # outline) so they can be accessed when copying or recreating
        patch.lasso_verts = verts

        self.on_select_callback(mask, patch, (xmin, xmax, ymin, ymax))

        # Make lasso interactive (movement only)
        def on_lasso_change(new_coords):
            """Recompute the mask from the vertices _update_patch already
            moved and rebuilt the staircase outline for.

            Args:
                new_coords (tuple): New (xmin, xmax, ymin, ymax) region bounds.

            Returns:
                None
            """
            new_xmin, new_xmax, new_ymin, new_ymax = (int(round(v)) for v in new_coords)
            new_verts = np.asarray(patch.lasso_verts)

            new_width = new_xmax - new_xmin + 1
            new_height = new_ymax - new_ymin + 1
            new_mask = _any_touch_polygon_mask(new_verts, new_xmin, new_ymin, new_width, new_height)

            self.on_select_callback(new_mask, patch, new_coords)
        
        # Disconnect previous interactive patch if any
        if self.interactive_patch is not None:
            print(f"[DEBUG] Disconnecting previous interactive patch (lasso)")
            try:
                self.interactive_patch.disconnect()
            except Exception as e:
                print(f"[DEBUG] Error disconnecting previous lasso patch: {e}")
            self.interactive_patch = None
        
        # Create new interactive patch (polygon - only movement)
        self.interactive_patch = InteractivePatch(
            self.ax, patch, 'polygon', on_lasso_change, (xmin, xmax, ymin, ymax)
        )