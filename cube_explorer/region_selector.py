"""
Region selection and analysis functionality
"""

import numpy as np
from matplotlib.widgets import RectangleSelector, EllipseSelector, LassoSelector
from matplotlib.path import Path
from matplotlib.patches import Rectangle, Polygon, PathPatch
from matplotlib.lines import Line2D


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
        """
        Initialize an interactive patch. Does NOT connect event handlers or
        create handles yet - call connect() to make it interactive.

        Args:
            ax: matplotlib axes
            patch: matplotlib patch object
            shape_type: 'rectangle', 'circle', 'ellipse', or 'polygon'
            on_change_callback: function(new_coords) to call on changes
            coords: initial coordinates (xmin, xmax, ymin, ymax)
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
        """Wire up event handlers and show resize handles."""
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
        """Unwire event handlers and hide resize handles, keeping the patch itself intact."""
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
        """Create visual resize handles for the patch."""
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
        """Handle mouse press to start dragging."""
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
                    # Snapshot vertices at press time so each motion tick can
                    # apply the TOTAL displacement-since-press to a fixed
                    # baseline, instead of stacking onto the previous tick's
                    # already-moved vertices (which would compound/drift).
                    'verts': np.asarray(self.patch.get_xy()).copy()
                }
                print(f"[DEBUG] Clicked inside polygon to move")
                return
        
        print(f"[DEBUG] Click not on patch or handles. Coords: {self.coords}")

    
    def on_release(self, event):
        """Handle mouse release."""
        self.active_handle = None
        self.press_data = None
    
    def on_motion(self, event):
        """Handle mouse motion - update patch while dragging."""
        if self.press_data is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        # Snap every move/resize to a whole-pixel step, matching how the
        # mask itself is always defined on whole pixels -- what's dragged on
        # screen never implies a position finer than the mask can represent.
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
        """Calculate new coordinates when resizing from a handle."""
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
            # Resize circle from edge handle. Rounded so the radius (and so
            # the whole bounding box) only ever changes in whole-pixel
            # steps, even though halving dx/dy could land on a half-pixel.
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
        """Update the patch geometry and redraw handles."""
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
            # The patch is a staircase outline of the actual any-touch mask,
            # not an idealized smooth shape -- rebuild it for the new coords
            # rather than mutating Circle/Ellipse geometry that no longer applies.
            ixmin, ixmax = int(round(xmin)), int(round(xmax))
            iymin, iymax = int(round(ymin)), int(round(ymax))
            width = max(1, ixmax - ixmin + 1)
            height = max(1, iymax - iymin + 1)
            mask_fn = _any_touch_circle_mask if self.shape_type == 'circle' else _any_touch_ellipse_mask
            path = _mask_boundary_path(mask_fn(width, height), ixmin, iymin)
            if path is not None:
                self.patch.set_path(path)

        elif self.shape_type == 'polygon':
            # Move polygon by applying total displacement-since-press to the
            # vertex snapshot captured at press time (see on_press) - using
            # the patch's current (already-moved) vertices here would double
            # count displacement on every tick after the first.
            if self.press_data and 'verts' in self.press_data:
                old_coords = self.press_data['coords']
                old_xmin, old_xmax, old_ymin, old_ymax = old_coords
                dx = xmin - old_xmin
                dy = ymin - old_ymin
                new_verts = self.press_data['verts'] + np.array([dx, dy])
                self.patch.set_xy(new_verts)
        
        # Update handle positions
        self._update_handles()
        
        # Redraw canvas
        self.ax.figure.canvas.draw_idle()
        
        # Call callback with new coordinates
        print(f"[DEBUG] Calling on_change_callback with coords: {new_coords}")
        self.on_change_callback(new_coords)
    
    def _update_handles(self):
        """Update visual positions of resize handles."""
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
        Safe to call even if the patch was never connect()-ed."""
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
    - Circles (◯) 
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
        """
        self.ax = ax
        self.canvas = canvas
        self.on_select_callback = on_select_callback
        self.selector = None
        self.active_shape = None
        self.interactive_patch = None  # Track the current interactive patch
        self.image = None  # Will store the current image for mask generation

    def activate_selector(self, shape):
        """
        Activate a specific type of region selector.

        Sets up the appropriate matplotlib widget based on shape:
        - '*': Point selection (handled by click events)
        - '▭': Rectangle selector 
        - '◯'/'⬭': Ellipse selector
        - '✏': Lasso selector for custom shapes

        Args:
            shape (str): The shape to activate, or None to deactivate selection

        Note:
            - Disconnects any existing selector before creating new one
            - Sets up appropriate event handlers and selection parameters
            - Point selection ('*') is handled separately through click events
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
        elif shape in ["◯", "⬭"]:  # circle or ellipse
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
        """
        Handle rectangle/ellipse selection events.

        Called when a selection is completed with rectangle or ellipse selectors.
        Converts the selection coordinates to integers and creates the region.

        Args:
            eclick (matplotlib.backend_bases.MouseEvent): Mouse press event
            erelease (matplotlib.backend_bases.MouseEvent): Mouse release event

        Note:
            - Converts floating point coordinates to integers
            - Ensures coordinates are ordered (min, max)
            - Delegates region creation to create_region
        """
        # Get selection bounds
        x0, y0 = int(eclick.xdata), int(eclick.ydata)
        x1, y1 = int(erelease.xdata), int(erelease.ydata)
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        
        coords = (xmin, xmax, ymin, ymax)
        self.create_region(coords)

    def create_region(self, coords):
        """
        Create a region mask and visual patch based on current shape type.

        Delegates the actual mask/patch construction to
        _generate_mask_from_coords, so a freshly-drawn shape and one later
        dragged/resized always use the exact same rule.

        Args:
            coords (tuple): (xmin, xmax, ymin, ymax) region bounds

        Note:
            - Circle/ellipse pixels count as in if they touch the shape at
              all (not just if their centre does); the drawn outline traces
              that mask's real edges instead of an idealized smooth shape
            - Calls callback with created mask, patch and coordinates
            - Made interactive with dragging and resizing
        """
        if self.active_shape == "◯":
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
            """Callback when patch is dragged or resized."""
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
        """
        Handle custom shape selection made with the lasso tool.

        Creates a boolean mask from the vertices by:
        1. Converting vertices to numpy array
        2. Finding bounding box
        3. Creating a point grid
        4. Testing which points are inside the polygon

        Args:
            verts (list): List of (x,y) vertex coordinates of the selection

        Note:
            - Uses matplotlib Path for point-in-polygon testing
            - Creates mask and patch just like other shape types
            - Bounding box used for coordinate reference
            - Results in a mask that exactly matches drawn shape
            - Made movable (dragging the entire shape)
        """
        verts = np.asarray(verts)
        xmin, ymin = np.min(verts, axis=0)
        xmax, ymax = np.max(verts, axis=0)
        
        # Convert to integers to handle decimal pixel coordinates
        xmin, ymin = int(xmin), int(ymin)
        xmax, ymax = int(xmax), int(ymax)
        
        nx = int(xmax - xmin + 1)
        ny = int(ymax - ymin + 1)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        coords = np.vstack((x.flatten(), y.flatten())).T
        
        path = Path(verts - [xmin, ymin])
        mask = path.contains_points(coords).reshape(ny, nx)
        
        patch = Polygon(verts, edgecolor='lime', facecolor='none', lw=1.5)
        # Store vertices on the patch so they can be accessed when copying
        patch.lasso_verts = verts
        
        self.on_select_callback(mask, patch, (xmin, xmax, ymin, ymax))
        
        # Make lasso interactive (movement only)
        def on_lasso_change(new_coords):
            """Callback when lasso is moved.

            Reuses the same `patch` object InteractivePatch._update_patch
            already moved in place (rather than building a second, separately
            computed Polygon) - just recomputes the mask from its current
            vertices and keeps lasso_verts in sync so a later copy or
            slice-switch-recreate uses the shape's current position.
            """
            new_xmin, new_xmax, new_ymin, new_ymax = new_coords
            new_verts = np.asarray(patch.get_xy())
            patch.lasso_verts = new_verts

            nx_new = int(new_xmax - new_xmin + 1)
            ny_new = int(new_ymax - new_ymin + 1)
            x_new, y_new = np.meshgrid(np.arange(nx_new), np.arange(ny_new))
            coords_new = np.vstack((x_new.flatten(), y_new.flatten())).T

            path_new = Path(new_verts - [new_xmin, new_ymin])
            new_mask = path_new.contains_points(coords_new).reshape(ny_new, nx_new)

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