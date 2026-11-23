"""
Custom continuum subtraction using user-selected points and spline fitting

IMPORTANT: This module supports both interactive GUI usage and non-GUI programmatic usage.
PyQt5 is optional and only loaded for interactive features.
The functions before the selector class (points_array, snap_points, spline_continuum,
write_points_file, read_points_file) have no GUI code. The selector, the Save Points and Load Points
buttons, and the functions of the API all use them.
"""

import numpy as np
from scipy.interpolate import CubicSpline

MIN_CONTINUUM_POINTS = 3


def points_array(points):
    """Turn continuum points into a float array and check them.

    Each point is a wavelength and a flux. The wavelength is in the unit of the spectrum axis.
    The flux is in the unit of the spectrum.

    Args:
        points (array-like): (wavelength, flux) pairs, as a list of pairs or an array with 2 columns.

    Returns:
        array (numpy.ndarray): Float array with shape (N, 2), in the order that was given.

    Raises:
        ValueError: if the points are not pairs, or a value is not a finite number
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must be a list of (wavelength, flux) pairs.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Every wavelength and flux of the points must be a finite number.")
    return array


def snap_points(wavelength, flux, wavelengths):
    """Find the spectrum samples nearest to some wavelengths, as a click on the plot does in the GUI.

    A point of the GUI is never free. A click gives the sample of the spectrum that is nearest in
    wavelength, with the flux of that sample. This function does the same for each wavelength.

    Args:
        wavelength (numpy.ndarray): Wavelength values of the spectrum.
        flux (numpy.ndarray): Flux values of the spectrum, with the same length.
        wavelengths (array-like): Wavelengths where you would click.

    Returns:
        points (list): (wavelength, flux) tuples of float, one for each wavelength you gave.

    Raises:
        ValueError: if wavelength and flux have different shapes, or the nearest sample has no flux
    """
    wl, fl = np.asarray(wavelength, dtype=float), np.asarray(flux, dtype=float)
    if wl.shape != fl.shape:
        raise ValueError(f"flux has the shape {fl.shape}, but wavelength has the shape {wl.shape}.")
    points = []
    for x in np.atleast_1d(np.asarray(wavelengths, dtype=float)):
        idx = int(np.argmin(np.abs(wl - x)))
        if not np.isfinite(fl[idx]):
            raise ValueError(f"The sample nearest to {x} (at {wl[idx]}) has no flux. Choose another wavelength.")
        points.append((float(wl[idx]), float(fl[idx])))
    return points


def spline_continuum(wavelength, points):
    """Fit a natural cubic spline through the points and evaluate it on a wavelength axis.

    The points are sorted by wavelength first. The spline goes beyond the first and the last point
    with the same cubic. At least 3 points are needed.

    Args:
        wavelength (numpy.ndarray): Wavelength values where the continuum is evaluated.
        points (array-like): (wavelength, flux) pairs, in the unit of wavelength and of the spectrum.

    Returns:
        continuum (numpy.ndarray): Continuum values, one for each wavelength value.

    Raises:
        ValueError: if there are fewer than 3 points, a value is not finite, or two points
            have the same wavelength
    """
    array = points_array(points)
    if len(array) < MIN_CONTINUUM_POINTS:
        raise ValueError(f"A custom continuum needs at least {MIN_CONTINUUM_POINTS} points, "
                         f"and {len(array)} were given.")
    array = array[np.argsort(array[:, 0])]
    if np.any(np.diff(array[:, 0]) <= 0):
        raise ValueError("Two points have the same wavelength. Remove one of them.")
    return CubicSpline(array[:, 0], array[:, 1], bc_type='natural')(wavelength)


def write_points_file(filename, points, flux_unit=None):
    """Write continuum points to a text file. This is the file that "Save Points" writes.

    The file has two comment lines and then one line for each point: the wavelength and the flux,
    separated by a comma. The points stay in the order that was given.

    Args:
        filename (str): Path of the file to write.
        points (array-like): (wavelength, flux) pairs.
        flux_unit (str): Flux unit of the points, for example 'Jy'. None or '' writes "Unknown".

    Returns:
        None

    Raises:
        ValueError: if there is no point, or the points are not finite pairs
        OSError: if the file cannot be written
    """
    if len(points) == 0:
        raise ValueError("There are no points to save.")
    array = points_array(points)
    with open(filename, 'w') as f:
        f.write("# Wavelength, Flux\n")
        f.write(f"# Flux unit: {flux_unit or 'Unknown'}\n")
        for x, y in array:
            f.write(f"{x:.6e}, {y:.6e}\n")


def read_points_file(filename):
    """Read continuum points from a text file that "Save Points" or write_points_file made.

    Lines that start with # are comments. The comment "# Flux unit: ..." gives the unit of the
    points. This function does not check the unit, so the caller can decide what to do with it.
    A line that is not 'wavelength, flux' is skipped. The caller gets the skipped lines and
    must tell the user, for example with skipped_lines_text.

    Args:
        filename (str): Path of the file to read.

    Returns:
        result (tuple): (points, file_unit, skipped). points is a list of (wavelength, flux)
            tuples of float, in the order of the file. file_unit is the unit text of the file,
            or None. skipped is a list of (line number, text) for each line that was not read.

    Raises:
        OSError: if the file cannot be opened
        ValueError: if the file has no valid point
    """
    points, file_unit, skipped = [], None, []
    with open(filename, 'r') as f:
        for number, raw in enumerate(f, start=1):
            line = raw.strip()
            if line.lower().startswith('# flux unit:'):
                file_unit = line.split(':', 1)[1].strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            try:
                if len(parts) < 2:
                    raise ValueError
                points.append((float(parts[0]), float(parts[1])))
            except ValueError:
                skipped.append((number, line))
    if not points:
        raise ValueError(f"No valid point found in {filename}.")
    return points, file_unit, skipped


def skipped_lines_text(skipped, limit=5):
    """Write the lines that read_points_file skipped as a message for the user.

    Args:
        skipped (list): (line number, text) pairs, as read_points_file returns them.
        limit (int): Number of lines to list. The message counts the others.

    Returns:
        text (str): The message, or '' if no line was skipped.
    """
    if not skipped:
        return ""
    lines = [f"{len(skipped)} line(s) are not 'wavelength, flux' and were not read:"]
    lines += [f"  line {number}: {text!r}" for number, text in skipped[:limit]]
    if len(skipped) > limit:
        lines.append(f"  ...and {len(skipped) - limit} more")
    return "\n".join(lines)


class CustomContinuumSelector:
    """
    Interactive selector for custom continuum subtraction.
    User clicks near the spectrum to select continuum points,
    points SNAP to the actual spectrum data for accuracy.
    """
    
    def __init__(self, ax, canvas, wavelength, flux, parent_widget=None):
        """Initialize the custom continuum selector.

        Args:
            ax (matplotlib.axes.Axes): The spectrum plot axes.
            canvas (FigureCanvas): The matplotlib canvas.
            wavelength (numpy.ndarray): Wavelength/frequency array.
            flux (numpy.ndarray): The spectrum data; points snap to this.
            parent_widget (QWidget): Parent widget for showing messages.

        Returns:
            None
        """
        self.ax = ax
        self.canvas = canvas
        self.wavelength = wavelength
        self.flux = flux
        self.parent_widget = parent_widget
        
        # Store selected continuum points
        self.continuum_points = []
        self.continuum_artists = []
        self.continuum_line_artist = None
        self.fitted_continuum = None
        
        # Connection IDs
        self.click_cid = None
        self.pick_cid = None
        
        # Active state
        self.is_active = False
        self.delete_mode = False
        # True while the points are for Save Continuum Cubes, so Fit Continuum saves the cubes.
        self.for_cubes = False

        # Point requirements
        self.min_points = MIN_CONTINUUM_POINTS
        
    def activate(self, for_cubes=False):
        """Activate the custom continuum selector.

        Args:
            for_cubes (bool): True when the points are for Save Continuum Cubes.
                Fit Continuum then asks for the file name and saves the cubes.

        Returns:
            None
        """
        if self.is_active:
            return

        self.is_active = True
        self.for_cubes = for_cubes

        # Clear any previous selections
        self._clear_points()

        # Connect mouse events
        self.click_cid = self.canvas.mpl_connect('button_press_event', self._on_click)
        self.pick_cid = self.canvas.mpl_connect('pick_event', self._on_pick)

    def deactivate(self):
        """Deactivate the custom continuum selector.

        Args:
            None

        Returns:
            None
        """
        if not self.is_active:
            return

        self.is_active = False
        self.for_cubes = False

        # Disconnect events
        if self.click_cid:
            self.canvas.mpl_disconnect(self.click_cid)
            self.click_cid = None
        if self.pick_cid:
            self.canvas.mpl_disconnect(self.pick_cid)
            self.pick_cid = None

    def set_delete_mode(self, state):
        """Toggle left-click delete mode.

        Args:
            state (bool): True to delete points on left-click, False to add.

        Returns:
            None
        """
        self.delete_mode = state

    def _on_click(self, event):
        """Handle mouse click events - SNAP to nearest spectrum point.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Click event.

        Returns:
            None
        """
        if event.inaxes != self.ax:
            return
            
        if event.button != 1:
            return
            
        # Check if toolbar is active (zoom/pan)
        toolbar = self.canvas.toolbar
        if toolbar and toolbar.mode != '':
            return
        
        if self.delete_mode:
            return
            
        x_click = event.xdata
        if x_click is None:
            return
            
        # SNAP to the actual spectrum data point. A sample with no flux is not a valid point.
        try:
            x_snap, y_snap = snap_points(self.wavelength, self.flux, [x_click])[0]
        except ValueError:
            return

        self._add_point(x_snap, y_snap)
        
    def _on_pick(self, event):
        """Handle pick events. A left click on a point removes it while Delete Point mode is on.

        Args:
            event (matplotlib.backend_bases.PickEvent): Pick event.

        Returns:
            None
        """
        if not (event.mouseevent.button == 1 and self.delete_mode):
            return

        if hasattr(event.artist, 'custom_continuum_point'):
            self._remove_point(event.artist)

    def _add_point(self, x, y):
        """Add a continuum point at (x, y).

        Args:
            x (float): Wavelength of the snapped point.
            y (float): Flux of the snapped point.

        Returns:
            None
        """
        # Check if point already exists at this x location
        for existing_x, existing_y in self.continuum_points:
            if np.abs(existing_x - x) < 1e-10:
                return
        
        # Store current axis limits BEFORE adding the point
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        
        # Plot the point - BLACK dots for continuum points
        marker = self.ax.plot(
            x, y,
            'o',
            color='black',
            ms=8,
            picker=5,
            markeredgecolor='white',
            markeredgewidth=1.5,
            markerfacecolor='black',
            zorder=100
        )[0]
        
        marker.custom_continuum_point = True
        
        self.continuum_points.append((x, y))
        self.continuum_artists.append(marker)
        
        # Restore the zoom level AFTER adding the point
        self.ax.set_xlim(current_xlim)
        self.ax.set_ylim(current_ylim)
        
        # Update plot without changing view
        self.canvas.draw_idle()

    def _remove_point(self, artist):
        """Remove a continuum point.

        Args:
            artist (matplotlib.lines.Line2D): Marker artist to remove.

        Returns:
            None
        """
        try:
            # Store current axis limits
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
            
            # Find the index of this artist
            idx = self.continuum_artists.index(artist)
            
            # Remove from lists
            self.continuum_points.pop(idx)
            self.continuum_artists.pop(idx)
            
            # Remove from plot
            artist.remove()
            
            # Restore the zoom level
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
            
            # Update plot
            self.canvas.draw_idle()

        except (ValueError, IndexError) as e:
            print(f"[DEBUG] Error removing point: {e}")

            
    def _clear_points(self):
        """Clear all continuum points.

        Args:
            None

        Returns:
            None
        """
        for artist in self.continuum_artists:
            try:
                artist.remove()
            except Exception:
                pass
                
        self.continuum_points = []
        self.continuum_artists = []
        
        if self.continuum_line_artist is not None:
            try:
                self.continuum_line_artist.remove()
            except Exception:
                pass
            self.continuum_line_artist = None
            
        self.fitted_continuum = None
        self.canvas.draw_idle()

    def fit_continuum(self):
        """Fit a cubic spline through the selected continuum points. The
        Fit Continuum caller already enforces the minimum-points check;
        this only does the math, with no dialogs of its own.

        Args:
            None

        Returns:
            continuum (numpy.ndarray): Fitted continuum evaluated at
                self.wavelength, or None if too few points or fitting fails.
        """
        if len(self.continuum_points) < self.min_points:
            return None
                    
        try:
            continuum = spline_continuum(self.wavelength, self.continuum_points)

            self.fitted_continuum = continuum

            return continuum

        except Exception as e:
            if self.parent_widget:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self.parent_widget,
                    "Fitting Error",
                    f"Error fitting continuum:\n{str(e)}\n\n"
                    f"Make sure your points are well-distributed across the spectrum."
                )
            import traceback
            traceback.print_exc()
            return None
            
    def clear_all(self):
        """Clear all points and fitted continuum.

        Args:
            None

        Returns:
            None
        """
        self._clear_points()

    def load_points(self, points):
        """Take a list of (x, y) tuples from a saved file, clear the
        current canvas, and draw the loaded points.

        Args:
            points (list): (x, y) tuples to load.

        Returns:
            None
        """
        self._clear_points()
        for x, y in points:
            self._add_point(x, y)
