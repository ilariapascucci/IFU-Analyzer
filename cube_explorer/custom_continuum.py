"""
Custom continuum subtraction using user-selected points and spline fitting
"""

import numpy as np
from scipy.interpolate import CubicSpline
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

class CustomContinuumSelector:
    """
    Interactive selector for custom continuum subtraction.
    User clicks near the spectrum to select continuum points,
    points SNAP to the actual spectrum data for accuracy.
    """
    
    def __init__(self, ax, canvas, wavelength, flux, parent_widget=None):
        """
        Initialize the custom continuum selector
        
        Parameters
        ----------
        ax : matplotlib axes
            The spectrum plot axes
        canvas : FigureCanvas
            The matplotlib canvas
        wavelength : array
            Wavelength/frequency array
        flux : array
            The actual spectrum data (blue line) - points will snap to this
        parent_widget : QWidget
            Parent widget for showing messages
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
        self.key_cid = None
        
        # Active state
        self.is_active = False
        
        # Point requirements
        self.min_points = 5
        self.recommended_points = 20
        
    def activate(self):
        """Activate the custom continuum selector"""
        if self.is_active:
            return
            
        self.is_active = True
        
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
        
        # Clear any previous selections
        self._clear_points()
        
        # Connect mouse events
        self.click_cid = self.canvas.mpl_connect('button_press_event', self._on_click)
        self.pick_cid = self.canvas.mpl_connect('pick_event', self._on_pick)
        self.key_cid = self.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Show instructions
        if self.parent_widget:
            QMessageBox.information(
                self.parent_widget,
                "Custom Continuum Subtraction",
                f"Instructions:\n"
                f"• Click on the spectrum to select continuum points\n"
                f"• Points will SNAP to the nearest spectrum data point\n"
                f"• Select regions WITHOUT emission/absorption lines\n"
                f"• Right-click on a black marker to remove it\n"
                f"• Press ENTER when done to fit and subtract\n"
                f"• Press ESC to cancel\n"
                f"\n"
                f"Click on the spectrum plot first if keys don't work\n"
                f"\n"
                f"Minimum: {self.min_points} points\n"
                f"Recommended: {self.recommended_points}+ points for best results\n"
                f"\n"
                f"Method: Cubic spline with smoothness constraints\n"
                f"Result: I_subtracted(λ) = I_observed(λ) - I_continuum(λ)"
            )
        
        print("[DEBUG] Custom continuum selector activated")
        print("[DEBUG] Click on spectrum - points will snap to data")
        
    def deactivate(self):
        """Deactivate the custom continuum selector"""
        if not self.is_active:
            return
            
        self.is_active = False
        
        # Disconnect events
        if self.click_cid:
            self.canvas.mpl_disconnect(self.click_cid)
            self.click_cid = None
        if self.pick_cid:
            self.canvas.mpl_disconnect(self.pick_cid)
            self.pick_cid = None
        if self.key_cid:
            self.canvas.mpl_disconnect(self.key_cid)
            self.key_cid = None
            
        print("[DEBUG] Custom continuum selector deactivated")
            
    def on_key_press(self, event):
        """Handle keyboard events"""
        print(f"[DEBUG] Key pressed: {event.key}")
        print(f"[DEBUG] Current points: {len(self.continuum_points)}")
        
        if event.key in ['enter', 'return']:
            print("[DEBUG] ENTER pressed - fitting continuum...")
            if self.parent_widget is not None:
                if hasattr(self.parent_widget, '_finalize_custom_continuum'):
                    self.parent_widget._finalize_custom_continuum()
                else:
                    print("[DEBUG] ERROR: _finalize_custom_continuum not found!")
            else:
                print("[DEBUG] ERROR: parent_widget is None!")
        
        elif event.key == 'escape':
            print("[DEBUG] ESCAPE pressed - cancelling...")
            if self.parent_widget is not None:
                if hasattr(self.parent_widget, '_cancel_custom_continuum'):
                    self.parent_widget._cancel_custom_continuum()
                else:
                    print("[DEBUG] ERROR: _cancel_custom_continuum not found!")

                    
    def _on_click(self, event):
        """Handle mouse click events - SNAP to nearest spectrum point"""
        if event.inaxes != self.ax:
            return
            
        if event.button != 1:
            return
            
        # Check if toolbar is active (zoom/pan)
        toolbar = self.canvas.toolbar
        if toolbar and toolbar.mode != '':
            return
            
        x_click = event.xdata
        if x_click is None:
            return
            
        # Find the nearest wavelength index
        idx = np.argmin(np.abs(self.wavelength - x_click))
        
        # SNAP to the actual spectrum data point
        x_snap = self.wavelength[idx]
        y_snap = self.flux[idx]
        
        # Check if point is valid (not NaN or Inf)
        if not np.isfinite(y_snap):
            print(f"[DEBUG] Invalid point at index {idx}, skipping")
            return
            
        self._add_point(x_snap, y_snap)
        
    def _on_pick(self, event):
        """Handle pick events (clicking on existing points to remove them)"""
        if event.mouseevent.button != 3:
            return
            
        if hasattr(event.artist, 'custom_continuum_point'):
            self._remove_point(event.artist)
        
    def _add_point(self, x, y):
        """Add a continuum point at (x, y)"""
        # Check if point already exists at this x location
        for existing_x, existing_y in self.continuum_points:
            if np.abs(existing_x - x) < 1e-10:
                print(f"[DEBUG] Point already exists at x={x:.3f}, skipping")
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
        
        print(f"[DEBUG] Added point at ({x:.3f}, {y:.3e}). Total: {len(self.continuum_points)}")
        
        if len(self.continuum_points) == self.min_points:
            print(f"[DEBUG] ✓ Minimum points reached ({self.min_points})")
        elif len(self.continuum_points) == self.recommended_points:
            print(f"[DEBUG] ✓✓ Recommended points reached ({self.recommended_points})")

            
    def _remove_point(self, artist):
        """Remove a continuum point"""
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
            
            print(f"[DEBUG] Removed point. Remaining: {len(self.continuum_points)}")
            
        except (ValueError, IndexError) as e:
            print(f"[DEBUG] Error removing point: {e}")

            
    def _clear_points(self):
        """Clear all continuum points"""
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
        print("[DEBUG] All points cleared")
        
    def fit_continuum(self):
        """
        Fit a cubic spline through the selected continuum points.
        
        Mathematical Process:
        1. For each pair of adjacent knots, calculate cubic polynomial: y = ax³ + bx² + cx + d
        2. Apply smoothness constraints at each knot:
           - Values match (continuity)
           - First derivatives match (smooth slope)
           - Second derivatives match (smooth curvature)
        3. Interpolate across entire wavelength range
        
        Returns the fitted continuum array or None if fitting fails.
        """
        if len(self.continuum_points) < self.min_points:
            if self.parent_widget:
                QMessageBox.warning(
                    self.parent_widget,
                    "Insufficient Points",
                    f"You need at least {self.min_points} points to fit a continuum.\n"
                    f"Currently selected: {len(self.continuum_points)} points\n"
                    f"Recommended: {self.recommended_points}+ points for best results"
                )
            return None
            
        if len(self.continuum_points) < self.recommended_points:
            if self.parent_widget:
                result = QMessageBox.question(
                    self.parent_widget,
                    "Few Points Selected",
                    f"You have selected {len(self.continuum_points)} points.\n"
                    f"For best results, {self.recommended_points}+ points are recommended.\n\n"
                    f"Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if result == QMessageBox.No:
                    return None
                    
        try:
            points_array = np.array(self.continuum_points)
            sort_indices = np.argsort(points_array[:, 0])
            x_points = points_array[sort_indices, 0]
            y_points = points_array[sort_indices, 1]
            
            print(f"[DEBUG] Fitting cubic spline through {len(x_points)} points...")
            print(f"[DEBUG] X range: {x_points.min():.2f} to {x_points.max():.2f}")
            print(f"[DEBUG] Y range: {y_points.min():.3e} to {y_points.max():.3e}")
            
            spline_model = CubicSpline(x_points, y_points, bc_type='natural')
            continuum = spline_model(self.wavelength)
            
            self.fitted_continuum = continuum
            
            print(f"[DEBUG] ✓ Cubic spline fitted successfully")
            print(f"[DEBUG] Continuum range: {np.nanmin(continuum):.3e} to {np.nanmax(continuum):.3e}")
            print(f"[DEBUG] Ready for subtraction: I_subtracted(λ) = I_observed(λ) - I_continuum(λ)")
            
            return continuum
            
        except Exception as e:
            if self.parent_widget:
                QMessageBox.critical(
                    self.parent_widget,
                    "Fitting Error",
                    f"Error fitting continuum:\n{str(e)}\n\n"
                    f"Make sure your points are well-distributed across the spectrum."
                )
            print(f"[DEBUG] Spline fitting error: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def show_fitted_line(self):
        """Show the fitted continuum line on the plot (RED spline)"""
        if self.fitted_continuum is None:
            return
        
        # Remove previous continuum line if exists
        if self.continuum_line_artist is not None:
            try:
                self.continuum_line_artist.remove()
            except Exception:
                pass
        
        # Plot the fitted spline in RED
        self.continuum_line_artist = self.ax.plot(
            self.wavelength, 
            self.fitted_continuum,
            color='red',
            linewidth=2.5,
            linestyle='-',
            label='Custom Spline Continuum',
            zorder=50,
            alpha=0.8
        )[0]
        
        # Update legend
        self.ax.legend(loc='upper right', fontsize=8)
        
        # Redraw
        self.canvas.draw_idle()
        
        print("[DEBUG] Custom continuum spline displayed (red line)")

        
    def get_fitted_continuum(self):
        """Return the fitted continuum array"""
        return self.fitted_continuum
        
    def clear_all(self):
        """Clear all points and fitted continuum"""
        self._clear_points()
        print("[DEBUG] Custom continuum selection cleared")
