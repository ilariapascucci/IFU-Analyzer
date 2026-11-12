"""
Visualization functions for spectrum and slice plotting

Supports both raw numeric arrays and astropy Quantity objects for wavelength.
Quantities are automatically handled for proper plotting and display.

NOTE: This module works with pure matplotlib Figure objects and does not require Qt.
It creates Figure objects that can be embedded in Qt GUIs or saved to disk.
"""

import numpy as np
import astropy.units as u
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.units import Quantity

from .scale_dialog import ScaleSettings, build_norm


def grid_step_arcsec(wcs, nx, ny):
    """Pick one grid step for both the RA and Dec axes.

    The step is 1, 2 or 5 times a power of 10 arcsec. It puts about 5 lines
    across the smaller side of the image on the sky.

    Args:
        wcs (astropy.wcs.WCS): Celestial WCS of the image.
        nx (int): Image width, in pixels.
        ny (int): Image height, in pixels.

    Returns:
        step (float): Grid step, in arcsec.
    """
    corners = wcs.pixel_to_world(np.array([-0.5, nx - 0.5, nx - 0.5, -0.5]),
                                 np.array([-0.5, -0.5, ny - 0.5, ny - 0.5]))
    center = wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)
    d_ra = ((corners.ra.deg - center.ra.deg + 180) % 360 - 180) * np.cos(center.dec.rad) * 3600
    d_dec = (corners.dec.deg - center.dec.deg) * 3600
    target = min(np.ptp(d_ra), np.ptp(d_dec)) / 5
    power_of_ten = 10 ** np.floor(np.log10(target))
    return float(power_of_ten * min((1, 2, 5, 10), key=lambda m: abs(m - target / power_of_ten)))


def apply_wcs_grid(ax, image_shape):
    """Draw the RA and Dec grid, tick labels and axis labels on a WCS axes.

    The grid lines are white. RA and Dec use the same step. Each label sits on
    the edge along which its coordinate changes fastest.

    Args:
        ax (astropy.visualization.wcsaxes.WCSAxes): Axes made with a celestial WCS.
        image_shape (tuple): Image height and width in pixels, as (ny, nx).

    Returns:
        None

    Raises:
        Exception: The error from astropy when the WCS cannot make the grid.
    """
    ax.set_axis_on()
    ax.coords.grid(color='white', alpha=0.5, linestyle='solid')

    try:
        ra  = ax.coords['ra']
        dec = ax.coords['dec']
    except (KeyError, TypeError):
        ra, dec = ax.coords[0], ax.coords[1]

    # RA is written as an angle, like Dec, with the same step on both axes,
    # so the labels step by the same amount (1.0" and 1.0", not 0.1 s and 1.0").
    ny, nx = image_shape
    step = grid_step_arcsec(ax.wcs, nx, ny)
    label_format = 'dd:mm:ss.s' if step < 1 else 'dd:mm:ss'
    ra.set_format_unit(u.deg)
    ra.set_major_formatter(label_format)
    dec.set_major_formatter(label_format)
    ra.set_ticks(spacing=step * u.arcsec)
    dec.set_ticks(spacing=step * u.arcsec)

    # RA goes on the bottom edge when it changes fastest along x.
    # A rotated image has RA changing along y, so RA goes left.
    cx, cy = (nx - 1) / 2, (ny - 1) / 2
    center = ax.wcs.pixel_to_world(cx, cy)
    ra_step_x = abs(center.spherical_offsets_to(ax.wcs.pixel_to_world(cx + 1, cy))[0].arcsec)
    ra_step_y = abs(center.spherical_offsets_to(ax.wcs.pixel_to_world(cx, cy + 1))[0].arcsec)
    ra_side, dec_side = ('b', 'l') if ra_step_x >= ra_step_y else ('l', 'b')

    ra.set_axislabel('RA',   minpad=0.5)
    dec.set_axislabel('Dec', minpad=0.5)
    ra.set_ticklabel_position(ra_side)
    dec.set_ticklabel_position(dec_side)
    ra.set_axislabel_position(ra_side)
    dec.set_axislabel_position(dec_side)
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


def hide_wcs_grid(ax):
    """Remove the grid, the tick marks and all labels from a WCS axes.

    The image and the frame stay. Only the sky-coordinate decoration goes.

    Args:
        ax (astropy.visualization.wcsaxes.WCSAxes): Axes made with a celestial WCS.

    Returns:
        None
    """
    ax.coords.grid(False)
    for index in range(2):
        coord = ax.coords[index]
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')


def format_sky_position(wcs, x, y):
    """Write the RA and Dec of a pixel as two text lines for the cursor readout.

    RA is in hours, minutes and seconds. Dec is in degrees, arcminutes and
    arcseconds. The text is empty when the WCS has no sky coordinates.

    Args:
        wcs (astropy.wcs.WCS): WCS of the image, or None.
        x (float): Pixel x coordinate.
        y (float): Pixel y coordinate.

    Returns:
        text (str): "RA: ...\\nDec: ...\\n", or "" when no position is available.
    """
    if wcs is None or not wcs.has_celestial:
        return ""
    try:
        wcs2d = wcs.celestial if wcs.pixel_n_dim > 2 else wcs
        world = wcs2d.pixel_to_world(x, y)
        ra = world.ra.to_string(unit='hour', sep='hms', precision=2, pad=True)
        dec = world.dec.to_string(unit='deg', sep='dms', precision=1, alwayssign=True, pad=True)
        return f"RA: {ra}\nDec: {dec}\n"
    except Exception:
        return ""


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


# The legend entries the user can hide, by group. The spectrum has no group, so it always stays.
# An entry is in a group when its label starts with one of the group's texts.
LEGEND_GROUPS = {
    "Continuum": ("Continuum", "Zero Level"),
    "Slice markers": ("Left slice", "Right slice"),
    "Fit markers": ("Fit Left", "Peak", "Fit Right"),
    "Noise region": ("Noise Left", "Noise Right"),
    "Gaussian fits": ("Baseline", "Gaussian Fit", "Integrated Area"),
    "Contaminants": ("Contaminant",),
    "Shift preview": ("Preview",),
}

# The groups the user hid. Every spectrum legend reads this one set.
HIDDEN_LEGEND_GROUPS = set()


def legend_group_of(label):
    """Give the legend group of one legend entry.

    Args:
        label (str): The label of the plotted artist.

    Returns:
        group (str): The group name, or None if the entry is in no group.
    """
    for group, prefixes in LEGEND_GROUPS.items():
        if label.startswith(prefixes):
            return group
    return None


def draw_spectrum_legend(ax, handles=None):
    """Draw the spectrum legend. An artist that is not visible, or is in a hidden group, gets no entry.
    With no handles, every labelled artist on the axes is a candidate.

    Args:
        ax (matplotlib.axes.Axes): The spectrum axes.
        handles (list): The artists to list. None uses every labelled artist on ax.

    Returns:
        None
    """
    if handles is None:
        handles = ax.get_legend_handles_labels()[0]
    shown = []
    for artist in handles:
        label = artist.get_label()
        if not label or label.startswith('_') or not artist.get_visible():
            continue
        if legend_group_of(label) in HIDDEN_LEGEND_GROUPS:
            continue
        shown.append(artist)
    if shown:
        ax.legend(handles=shown, loc='upper right', fontsize=8)
    elif ax.get_legend() is not None:
        ax.get_legend().remove()


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
        self.fig.canvas.mpl_connect('resize_event', self._on_resize)

    def _on_resize(self, event):
        """Fit the margins to the labels again after the canvas changes size.

        Args:
            event (matplotlib.backend_bases.ResizeEvent): Resize event of the canvas.

        Returns:
            None
        """
        self.fig.tight_layout()

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


        if spectrum is None:
            print("Warning: Cannot plot spectrum - data is None")
            return
            
        self.ax.clear()
        # Reset axis limits and ticks to clear old data artifacts
        self.ax.set_xlim(auto=True)
        self.ax.set_ylim(auto=True)


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
            draw_spectrum_legend(self.ax)
        except Exception as e:
            print(f"Error plotting spectrum: {e}")
            # Create an empty plot as fallback
            self.ax.plot([], [])
        self.fig.tight_layout()

        return self.ax

    def plot_gaussian_fit(self, result, to_display=None):
        """Plot Gaussian fit results on the spectrum.

        Args:
            result: GaussianFitResult object containing fit parameters and data.
            to_display (callable): Function (x, y) that converts a raw-flux curve to the
                units of the plotted spectrum. None draws the curve in raw flux.

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
        
        # The fit is in raw flux. Draw it in the units of the plotted spectrum.
        if to_display is None:
            to_display = lambda x, y: np.asarray(y, dtype=float)
        x_shown = result.x_data
        baseline_shown = to_display(x_shown, result.baseline)
        fit_shown = to_display(x_shown, result.baseline + result.y_fit)

        # Plot baseline as smooth curve (mathematical function, not discrete data)
        baseline_line, = self.ax.plot(
            x_shown, baseline_shown,
            'k--', linewidth=2, alpha=0.7, label='Baseline', zorder=10
        )
        self.gauss_fit_artists.append(baseline_line)

        # Plot fitted Gaussian as smooth curve (mathematical function, not discrete data)
        gauss_line, = self.ax.plot(
            x_shown, fit_shown,
            'r-', linewidth=3, alpha=0.9, label='Gaussian Fit', zorder=11
        )
        self.gauss_fit_artists.append(gauss_line)

        # Fill area under the Gaussian (above baseline)
        fill = self.ax.fill_between(
            x_shown, baseline_shown, fit_shown,
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
        # The box follows the "Gaussian fits" legend group, so that group's check box shows or hides it.
        text_obj.set_visible("Gaussian fits" not in HIDDEN_LEGEND_GROUPS)
        self.gauss_fit_artists.append(text_obj)
        
        # Update legend
        draw_spectrum_legend(self.ax)
        

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
        """Move the two slice indicators by the wavelength shift factor, when a Doppler shift is applied.
        The line markers (HITRAN, LST, contaminant) stay, because they are fixed wavelengths.

        Args:
            shift_factor (float): The wavelength shift factor (e.g., 1 / (1 + v/c)).

        Returns:
            None
        """
        for line in (self.vline_left, self.vline_right):
            if line:
                x = float(line.get_xdata()[0]) * shift_factor
                line.set_xdata([x, x])



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
        wcs (astropy.wcs.WCS): Sky WCS of the axes, or None for plain axes
        pixel_margins (bool): True after set_wcs, when the margins are fixed in pixels
        grid_on (bool): True while the RA/Dec grid is shown
        image_shape (tuple): (height, width) of the last image drawn, or None
    """

    # Axes position (left, bottom, width, height) as fractions of the figure, for plotters that
    # lay out their own axes (the 2D viewer and the map dialogs).
    _RECT_PLAIN = [0.08, 0.1, 0.70, 0.8]
    # Margins in pixels (left, bottom, right, top) for plotters that call set_wcs. The labels
    # need the same number of pixels at every figure size. The right margin holds the colorbar.
    _MARGINS_PLAIN = (10, 10, 120, 28)
    _MARGINS_GRID = (80, 40, 120, 28)

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
        self.ax.set_position(self._RECT_PLAIN)
        self.im = None
        self.cbar = None
        self.pixel_marker = None
        self.wcs = None
        self.grid_on = False
        self.image_shape = None
        self.pixel_margins = False
        # Draw empty axis
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.canvas.draw()
        self.fig.canvas.mpl_connect('resize_event', self._on_resize)

    def _rect(self):
        """Axes position that keeps the pixel margins of the current grid state.

        Args:
            None

        Returns:
            rect (list): Left, bottom, width and height as fractions of the figure.
        """
        left, bottom, right, top = self._MARGINS_GRID if self.grid_on else self._MARGINS_PLAIN
        width = max(self.fig.bbox.width, 1)
        height = max(self.fig.bbox.height, 1)
        # A figure smaller than its margins keeps a thin axes, not a negative size.
        return [left / width, bottom / height,
                max(1 - (left + right) / width, 0.01), max(1 - (bottom + top) / height, 0.01)]

    def _on_resize(self, event):
        """Keep the pixel margins after the canvas changes size.

        Args:
            event (matplotlib.backend_bases.ResizeEvent): Resize event of the canvas.

        Returns:
            None
        """
        if self.pixel_margins:
            self.ax.set_position(self._rect())

    def set_wcs(self, wcs):
        """Rebuild the axes so they can show the RA/Dec grid of a new cube.

        The old image, colorbar and grid are removed. The grid is off after this call.
        Plain axes are made when the WCS is missing or cannot make axes.

        Args:
            wcs (astropy.wcs.WCS): 2D celestial WCS of the cube, or None.

        Returns:
            error (str): Why no sky axes were made, or None when the axes show sky coordinates.
        """
        self.fig.clf()
        self.im = None
        self.cbar = None
        self.pixel_marker = None
        self.grid_on = False
        self.image_shape = None
        self.wcs = None
        self.pixel_margins = True

        error = None
        try:
            if wcs is None or not wcs.has_celestial:
                raise ValueError("This cube has no celestial WCS.")
            self.ax = self.fig.add_axes(self._rect(), projection=wcs)
            self.wcs = wcs
            hide_wcs_grid(self.ax)
        except Exception as exc:
            error = str(exc)
            self.ax = self.fig.add_axes(self._rect())
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        return error

    def set_grid(self, on):
        """Show or hide the RA/Dec grid.

        The grid shows at the next draw when no image has been plotted yet.

        Args:
            on (bool): True to show the grid, False to hide it.

        Returns:
            None

        Raises:
            Exception: The error from astropy when the grid cannot be drawn. The grid stays off.
        """
        if self.wcs is None:
            raise ValueError("These axes have no sky WCS.")
        self.grid_on = on
        if self.image_shape is not None:
            try:
                self._decorate()
            except Exception:
                self.grid_on = False
                self._decorate()
                raise
        self.fig.canvas.draw_idle()

    def _decorate(self):
        """Apply the grid state to the axes: grid and labels on, or all off.

        Args:
            None

        Returns:
            None
        """
        if self.wcs is None:
            return
        if self.grid_on:
            apply_wcs_grid(self.ax, self.image_shape)
        else:
            hide_wcs_grid(self.ax)
        self.ax.set_position(self._rect())

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
        self.image_shape = image.shape
        if self.im is None:
            # WCS axes are always new here (see set_wcs). Their clear() removes the frame and the grid.
            if self.wcs is None:
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
            self._decorate()

            # Create fixed colorbar using axes divider
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=Axes)
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
        if self.wcs is None:
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
