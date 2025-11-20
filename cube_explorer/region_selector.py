"""
Region selection and analysis functionality
"""

import numpy as np
from matplotlib.widgets import RectangleSelector, EllipseSelector, LassoSelector
from matplotlib.path import Path
from matplotlib.patches import Circle, Ellipse, Rectangle, Polygon

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

        Creates both:
        1. Boolean mask identifying selected pixels
        2. Matplotlib patch for visual display

        Handles different shapes:
        - Circle: Uses radius = min(width,height)/2
        - Ellipse: Uses full width/height for axes
        - Rectangle: Uses full width/height

        Args:
            coords (tuple): (xmin, xmax, ymin, ymax) region bounds

        Note:
            - Generates mask using efficient numpy operations
            - Creates appropriate matplotlib patch for visualization 
            - Calls callback with created mask, patch and coordinates
            - Patches use consistent style (cyan outline, no fill)
        """
        xmin, xmax, ymin, ymax = coords
        # dimensions of the bounding box
        height = ymax - ymin + 1
        width = xmax - xmin + 1
        yy, xx = np.indices((height, width))

        if self.active_shape == "◯":  # Circle
            # center relative to bbox
            cx_rel = (width - 1) / 2.0
            cy_rel = (height - 1) / 2.0
            r = min((width - 1) / 2.0, (height - 1) / 2.0)
            mask = (xx - cx_rel)**2 + (yy - cy_rel)**2 <= r**2
            # absolute center for plotting
            cx_abs = xmin + cx_rel
            cy_abs = ymin + cy_rel
            patch = Circle((cx_abs, cy_abs), r, edgecolor='cyan', facecolor='none', lw=1.5)

        elif self.active_shape == "⬭":  # Ellipse
            cx_rel = (width - 1) / 2.0
            cy_rel = (height - 1) / 2.0
            a = (width - 1) / 2.0
            b = (height - 1) / 2.0
            mask = ((xx - cx_rel) / a)**2 + ((yy - cy_rel) / b)**2 <= 1
            cx_abs = xmin + cx_rel
            cy_abs = ymin + cy_rel
            patch = Ellipse((cx_abs, cy_abs), 2*a, 2*b, edgecolor='cyan', facecolor='none', lw=1.5)

        else:  # Rectangle
            mask = np.ones((height, width), dtype=bool)
            patch = Rectangle((xmin, ymin), width-1, height-1,
                              edgecolor='cyan', facecolor='none', lw=1.5)

        # Call the callback with the results
        self.on_select_callback(mask, patch, coords)

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
        """
        verts = np.asarray(verts)
        xmin, ymin = np.min(verts, axis=0)
        xmax, ymax = np.max(verts, axis=0)
        
        nx = int(xmax - xmin + 1)
        ny = int(ymax - ymin + 1)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        coords = np.vstack((x.flatten(), y.flatten())).T
        
        path = Path(verts - [xmin, ymin])
        mask = path.contains_points(coords).reshape(ny, nx)
        
        patch = Polygon(verts, edgecolor='cyan', facecolor='none', lw=1.5)
        
        self.on_select_callback(mask, patch, (xmin, xmax, ymin, ymax))