"""
Visualization functions for spectrum and slice plotting

Supports both raw numeric arrays and astropy Quantity objects for wavelength.
Quantities are automatically handled for proper plotting and display.

NOTE: This module works with pure matplotlib Figure objects and does not require Qt.
It creates Figure objects that can be embedded in Qt GUIs or saved to disk.
"""

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.units import Quantity

from .scale_dialog import ScaleSettings, build_norm


def _extract_wavelength_values(wavelength):
    """
    Extract numeric values from wavelength array, handling both Quantity and raw arrays.
    
    Args:
        wavelength: Wavelength array (Quantity or numpy array)
    
    Returns:
        numpy array of wavelength values (units stripped if Quantity)
    """
    if isinstance(wavelength, Quantity):
        return wavelength.value
    else:
        return wavelength


class SpectrumPlotter:
    """
    Handles plotting and updating of spectral data visualization.

    This class manages the spectrum display including:
    - Main spectrum plot
    - Region selection overlays
    - Wavelength/velocity slice indicators
    - Legend and axis labels

    Attributes:
        fig (matplotlib.figure.Figure): Figure container for plots
        ax (matplotlib.axes.Axes): Axes for spectrum plotting
        wavelength (numpy.ndarray): Wavelength/velocity values
        wave_unit (str): Unit for spectral axis (Å, µm, or km/s)
        original_spectrum_plotted (bool): Track if main spectrum is plotted
        vline_left (matplotlib.lines.Line2D): Left slice indicator line
        vline_right (matplotlib.lines.Line2D): Right slice indicator line
    """

    def __init__(self, figure, wavelength=None, wave_unit=""):
        """Initialize the spectrum plotter.

        Args:
            figure (matplotlib.figure.Figure): Figure to plot on.
            wavelength (numpy.ndarray): Wavelength/velocity values, or None.
            wave_unit (str): Unit for spectral axis (Angstrom, micron, km/s).

        Returns:
            None
        """
        self.fig = figure
        self.ax = self.fig.add_subplot(111)
        self.wavelength = wavelength
        self.wave_unit = wave_unit
        self.original_spectrum_plotted = False
        self.vline_left = None
        self.vline_right = None
        self.cube_data = None  # Will be set later
        self.global_line = None

    def plot_spectrum(self, spectrum, ax=None):
        """Plot the main integrated spectrum: sets up the x-axis
        (wavelength/velocity, or index if unavailable), plots the data,
        adds axis labels, and creates hidden slice indicator lines.

        Args:
            spectrum (numpy.ndarray): 1D array of spectral values.
            ax: Unused; kept for backward compatibility.

        Returns:
            ax (matplotlib.axes.Axes): The axes the spectrum was plotted on.
        """

        print("DEBUG plot_spectrum called")
        print("Existing lines on axis before clear:", len(self.ax.lines))

        if spectrum is None:
            print("Warning: Cannot plot spectrum - data is None")
            return
            
        self.ax.clear()
        # Reset axis limits and ticks to clear old data artifacts
        self.ax.set_xlim(auto=True)
        self.ax.set_ylim(auto=True)

        print("DEBUG plot_spectrum cleared axis, lines now:", len(self.ax.lines))

        try:
            # X-axis values - extract from Quantity if needed
            if self.wavelength is not None and len(self.wavelength) == len(spectrum):
                x = _extract_wavelength_values(self.wavelength)
            else:
                x = np.arange(len(spectrum))

            # Reset line registries
            self.vline_left = None
            self.vline_right = None

            # Use raw spectrum data (no automatic normalization)
            # Normalization is now controlled by user via Normalize button
            y = np.asarray(spectrum).astype(np.float32)
            print(f"Global spectrum plotted (no auto-normalization)")

            # Plot main spectrum as steps (discrete wavelength bins)
            self.global_line, = self.ax.step(x, y, where='mid', color='blue', label='Total Spectrum', zorder=1)

            # Axis labels
            if self.wave_unit in ["Å", "um"]:
                from .unit_converter import clean_wave_unit
                xlabel = f"Wavelength [{clean_wave_unit(self.wave_unit)}]"
            elif self.wave_unit == "km/s":
                xlabel = f"Velocity [{self.wave_unit}]"
            else:
                xlabel = "Spectral axis"
            self.ax.set_xlabel(xlabel)

            if hasattr(self.cube_data, 'flux_unit') and self.cube_data.flux_unit:
                from .unit_converter import clean_flux_unit
                ylabel = f"Flux [{clean_flux_unit(self.cube_data.flux_unit)}]"
            else:
                # No flux_unit available - use generic label instead of guessing
                # (wavelength alone doesn't determine flux unit: micron could be Jy or MJy/sr)
                ylabel = "Flux [units unknown]"
            self.ax.set_ylabel(ylabel)

            # Slice indicator lines start hidden; shown only once the user
            # explicitly picks a wavelength via "Set Left/Right Slice".
            ymin, ymax = self.ax.get_ylim()
            self.vline_left = self.ax.plot([x[0], x[0]], [ymin, ymax], '--', color='cyan', lw=1.5, label='Left slice', zorder=3)[0]
            self.vline_right = self.ax.plot([x[0], x[0]], [ymin, ymax], '--', color='magenta', lw=1.5, label='Right slice', zorder=3)[0]
            self.vline_left.set_visible(False)
            self.vline_right.set_visible(False)
            self.ax.set_ylim(ymin, ymax)

            # Finalize legend
            self.ax.legend(loc='upper right', fontsize=8)
        except Exception as e:
            print(f"Error plotting spectrum: {e}")
            # Create an empty plot as fallback
            self.ax.plot([], [])
        self.fig.tight_layout()

        return self.ax

    def plot_gaussian_fit(self, result):
        """Plot Gaussian fit results on the spectrum.

        Args:
            result: GaussianFitResult object containing fit parameters and data.

        Returns:
            None
        """
        # Clear previous fit lines if any
        if not hasattr(self, 'gauss_fit_artists'):
            self.gauss_fit_artists = []
        
        for artist in self.gauss_fit_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.gauss_fit_artists = []
        
        # Get normalization factor from global spectrum if it was normalized
        if hasattr(self, 'global_line') and self.global_line is not None:
            # Check if spectrum was normalized (median is then ~1)
            global_y = self.global_line.get_ydata()
            global_median = np.nanmedian(global_y[np.isfinite(global_y)])
            
            # If normalized (median close to 1), we need to normalize the fit too
            if 0.5 < global_median < 2.0:
                # Calculate normalization factor from the fit region
                fit_median = np.nanmedian(result.y_data[np.isfinite(result.y_data)])
                if fit_median > 0:
                    norm_factor = fit_median
                    print(f"[DEBUG] Normalizing Gaussian fit by factor {norm_factor:.3e}")
                else:
                    norm_factor = 1.0
            else:
                norm_factor = 1.0
        else:
            norm_factor = 1.0
        
        # Normalize baseline and fit
        baseline_norm = result.baseline / norm_factor
        y_fit_norm = result.y_fit / norm_factor
        
        print(f"[DEBUG] Gaussian fit plotting:")
        print(f"  x_data range: {result.x_data.min():.4f} to {result.x_data.max():.4f}")
        print(f"  baseline range: {baseline_norm.min():.4e} to {baseline_norm.max():.4e}")
        print(f"  y_fit range: {y_fit_norm.min():.4e} to {y_fit_norm.max():.4e}")
        print(f"  total range: {(baseline_norm + y_fit_norm).min():.4e} to {(baseline_norm + y_fit_norm).max():.4e}")
        print(f"  Current ax ylim: {self.ax.get_ylim()}")
        
        # Plot baseline as smooth curve (mathematical function, not discrete data)
        baseline_line, = self.ax.plot(
            result.x_data, baseline_norm,
            'k--', linewidth=2, alpha=0.7, label='Baseline', zorder=10
        )
        self.gauss_fit_artists.append(baseline_line)
        
        # Plot fitted Gaussian as smooth curve (mathematical function, not discrete data)
        gauss_line, = self.ax.plot(
            result.x_data, baseline_norm + y_fit_norm,
            'r-', linewidth=3, alpha=0.9, label='Gaussian Fit', zorder=11
        )
        self.gauss_fit_artists.append(gauss_line)
        
        # Fill area under the Gaussian (above baseline)
        fill = self.ax.fill_between(
            result.x_data, baseline_norm, baseline_norm + y_fit_norm,
            color='red', alpha=0.2, label='Integrated Area', zorder=9
        )
        self.gauss_fit_artists.append(fill)
        
        # Add text annotation with key results
        # Get units from cube_data if available
        if hasattr(self, 'cube_data') and self.cube_data is not None:
            wave_unit = getattr(self.cube_data, 'wave_unit', '')
        else:
            wave_unit = self.wave_unit if hasattr(self, 'wave_unit') else ''
        
        text_str = (
            f"Center: {result.center:.4f} {wave_unit}\n"
            f"FWHM: {result.fwhm:.4g} {wave_unit}\n"
            f"R²: {result.r_squared:.3f}"
        )
        text_obj = self.ax.text(
            0.02, 0.98, text_str,
            transform=self.ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9,
            zorder=12
        )
        self.gauss_fit_artists.append(text_obj)
        
        # Update legend
        self.ax.legend(loc='upper right', fontsize=8)
        
        print(f"[DEBUG] Gaussian fit plotted, total lines now: {len(self.ax.lines)}")

    def update_slice_indicator(self, wavelength_idx, target="left"):
        """Move the vertical line showing the current slice position for
        either the left or right display, preserving y-axis limits.

        Args:
            wavelength_idx (int): Index into wavelength array for new position.
            target (str): Which line to update, 'left' or 'right'.

        Returns:
            None
        """
        if self.wavelength is None:
            return
            
        try:
            # Extract wavelength value - handle Quantity objecs
            wavelength_vals = _extract_wavelength_values(self.wavelength)
            wave = wavelength_vals[wavelength_idx]
            
            # Get current y-axis limits
            ymin, ymax = self.ax.get_ylim()
            
            # Update the appropriate line
            if target == "left":
                if self.vline_left:
                    self.vline_left.set_xdata([wave, wave])
                    self.vline_left.set_ydata([ymin, ymax])
            else:
                if self.vline_right:
                    self.vline_right.set_xdata([wave, wave])
                    self.vline_right.set_ydata([ymin, ymax])
                    
            self.ax.set_ylim(ymin, ymax)
            
        except (IndexError, AttributeError) as e:
            print(f"Error updating slice indicator: {e}")

    def shift_plot_elements(self, shift_factor):
        """Shift all plot elements (Gaussian fit, slice indicators, vertical
        line markers) by the wavelength shift factor, called when a Doppler
        shift is applied so all graphics move with the spectrum.

        Args:
            shift_factor (float): The wavelength shift factor (e.g., 1 / (1 + v/c)).

        Returns:
            None
        """
        # Shift Gaussian fit artists if they exist
        if hasattr(self, 'gauss_fit_artists') and self.gauss_fit_artists:
            for artist in self.gauss_fit_artists:
                try:
                    # Handle line objects (baseline and gaussian fit lines)
                    if hasattr(artist, 'get_xdata') and hasattr(artist, 'set_xdata'):
                        xdata = artist.get_xdata()
                        shifted_xdata = xdata * shift_factor
                        artist.set_xdata(shifted_xdata)
                    # Handle fill_between (PolyCollection)
                    elif hasattr(artist, 'get_paths'):
                        # For fill_between, we need to scale the vertices
                        for path in artist.get_paths():
                            vertices = path.vertices
                            vertices[:, 0] *= shift_factor
                            path.vertices = vertices
                except Exception as e:
                    print(f"[DEBUG] Could not shift artist: {e}")
        
        # Shift left slice indicator
        if self.vline_left:
            try:
                xdata = self.vline_left.get_xdata()
                shifted_xdata = xdata * shift_factor
                self.vline_left.set_xdata(shifted_xdata)
            except Exception as e:
                print(f"[DEBUG] Could not shift vline_left: {e}")
        
        # Shift right slice indicator
        if self.vline_right:
            try:
                xdata = self.vline_right.get_xdata()
                shifted_xdata = xdata * shift_factor
                self.vline_right.set_xdata(shifted_xdata)
            except Exception as e:
                print(f"[DEBUG] Could not shift vline_right: {e}")
        
        # Shift all vertical lines on the axis (Gaussian markers, LST lines, region markers, etc.)
        try:
            for line in self.ax.get_lines():
                # Check if it's a vertical line by looking at its xdata
                xdata = line.get_xdata()
                if len(xdata) == 2 and xdata[0] == xdata[1]:
                    # This is a vertical line (both x coordinates are the same)
                    shifted_x = xdata[0] * shift_factor
                    line.set_xdata([shifted_x, shifted_x])
                    print(f"[DEBUG] Shifted vertical line from {xdata[0]:.4f} to {shifted_x:.4f}")
        except Exception as e:
            print(f"[DEBUG] Could not process lines on axis: {e}")
        
        print(f"[DEBUG] Shifted all plot elements by factor {shift_factor}")



class SlicePlotter:
    """
    Handles plotting and updating of 2D slice data visualization.

    This class manages the display of wavelength/velocity slices including:
    - Image display with colormap
    - Colorbar for intensity scale
    - Title and axis labels
    - Pixel selection markers
    - Region selection overlays

    Attributes:
        fig (matplotlib.figure.Figure): Figure container for plots
        ax (matplotlib.axes.Axes): Axes for image plotting
        im (matplotlib.image.AxesImage): The displayed image
        cbar (matplotlib.colorbar.Colorbar): Colorbar for intensity scale
        pixel_marker (list): Storage for pixel selection markers
    """
    
    def __init__(self, figure):
        """Initialize the slice plotter with custom axes positioning.

        Args:
            figure (matplotlib.figure.Figure): Figure to plot on.

        Returns:
            None
        """
        self.fig = figure
        # Clear any existing axes
        self.fig.clear()
        # Create new axes with specific position
        self.ax = self.fig.add_subplot(111)
        self.ax.set_position([0.08, 0.1, 0.70, 0.8])  # Adjust to leave room for colorbar and label
        self.im = None
        self.cbar = None
        self.pixel_marker = None
        # Draw empty axis
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.canvas.draw()

    def plot_slice(self, image, title="", cmap=None, settings=None, flux_unit="", mask=None):
        """Plot a 2D slice image with colorbar, creating or updating it as
        needed (origin lower-left, 'magma' colormap by default).

        Args:
            image (numpy.ndarray): 2D array of intensity values.
            title (str): Title for the plot.
            cmap (str): Colormap name, or None to keep the current one.
            settings (ScaleSettings): Scale type, gamma, and color range, or
                None for Linear with an Auto range.
            flux_unit (str): Unit string for the flux/intensity values.
            mask (numpy.ndarray): Boolean array of the pixels that set the
                percentile range, or None for all finite pixels.

        Returns:
            None
        """
        # If axes are empty (or cleared), reset self.im
        if self.im is None or not self.ax.has_data():
            self.im = None

        norm, image = build_norm(image, settings or ScaleSettings(), mask)

        # First plot: create image and fixed colorbar
        if self.im is None:
            self.ax.clear()
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_title(title)
            # Choose colormap: provided cmap or existing or default
            if cmap is None:
                cmap = getattr(self, 'current_cmap', 'magma')
            self.current_cmap = cmap
            self.im = self.ax.imshow(image, origin='lower', interpolation='nearest', 
                                     cmap=cmap, norm=norm)

            # Create fixed colorbar using axes divider
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            try:
                self.cbar = self.fig.colorbar(self.im, cax=cax)
                # Set colorbar label with units if provided
                if flux_unit:
                    self.cbar.set_label(f"Flux / Intensity\n[{flux_unit}]")
                else:
                    self.cbar.set_label("Flux / Intensity")
            except Exception as e:
                print(f"Warning: Could not create colorbar: {e}")
                self.cbar = None

        else:
            # Update existing image
            self.im.set_data(image)
            self.im.set_norm(norm)
            # Update cmap if a new one is specified
            if cmap is not None:
                self.current_cmap = cmap
                try:
                    self.im.set_cmap(cmap)
                except Exception:
                    pass
            self.ax.set_title(title)
            if self.cbar is not None:
                self.cbar.update_normal(self.im)

        # Keep ticks off
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Redraw canvas
        self.fig.canvas.draw_idle()

    def set_cmap(self, cmap_name):
        """Update the colormap for the existing slice image.

        Args:
            cmap_name (str): Matplotlib colormap name to apply.

        Returns:
            None
        """
        self.current_cmap = cmap_name
        if self.im is not None:
            try:
                self.im.set_cmap(cmap_name)
                if self.cbar is not None:
                    self.cbar.update_normal(self.im)
            except Exception as e:
                print(f"Could not set colormap {cmap_name}: {e}")
        self.fig.canvas.draw_idle()

    def update_pixel_marker(self, x, y):
        """Update or create a marker (red +) showing the selected pixel,
        removing any existing marker first.

        Args:
            x (int): X coordinate of selected pixel.
            y (int): Y coordinate of selected pixel.

        Returns:
            None
        """
        if not hasattr(self.ax, 'pixel_marker'):
            self.ax.pixel_marker = []
        
        # Remove old markers
        for artist in self.ax.pixel_marker:
            artist.remove()
        self.ax.pixel_marker = []  # Clear the list after removing
        
        # Add new marker
        marker = self.ax.plot(x, y, 'r+', markersize=8, markeredgewidth=1.5)[0]
        self.ax.pixel_marker = [marker]  # Store only the new marker


def draw_map(fig, ax, shown, norm, cmap, title, cbar_label=None):
    """Draw one 2D map on an axes, with its title, pixel-axis labels and colorbar.
    Every map dialog and the PNG export use this function, so all maps look alike.

    Args:
        fig (matplotlib.figure.Figure): Figure that owns the axes and the colorbar.
        ax (matplotlib.axes.Axes): Axes to draw on.
        shown (numpy.ndarray): 2D array to display, from build_norm.
        norm (matplotlib.colors.Normalize): Norm from build_norm, or None.
        cmap (str or matplotlib.colors.Colormap): Colormap name or object.
        title (str): Axes title.
        cbar_label (str): Colorbar label, or None for no colorbar.

    Returns:
        im (matplotlib.image.AxesImage): The drawn image.
    """
    # 'nearest' keeps every data pixel a sharp square instead of blending neighbours.
    im = ax.imshow(shown, origin='lower', cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")
    if cbar_label is not None:
        fig.colorbar(im, ax=ax).set_label(cbar_label)
    return im
