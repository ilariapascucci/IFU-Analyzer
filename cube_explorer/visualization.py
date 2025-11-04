"""
Visualization functions for spectrum and slice plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        region_lines (dict): Storage for plot lines by region label
        original_spectrum_plotted (bool): Track if main spectrum is plotted
        vline_left (matplotlib.lines.Line2D): Left slice indicator line
        vline_right (matplotlib.lines.Line2D): Right slice indicator line
    """

    def __init__(self, figure, wavelength=None, wave_unit=""):
        """
        Initialize the spectrum plotter.

        Args:
            figure (matplotlib.figure.Figure): Figure to plot on
            wavelength (numpy.ndarray, optional): Wavelength/velocity values
            wave_unit (str, optional): Unit for spectral axis (Å, µm, km/s)
        """
        self.fig = figure
        self.ax = self.fig.add_subplot(111)
        self.wavelength = wavelength
        self.wave_unit = wave_unit
        self.region_lines = {}
        self.original_spectrum_plotted = False
        self.vline_left = None
        self.vline_right = None
        self.cube_data = None  # Will be set later

    def plot_spectrum(self, spectrum):
        """
        Plot the main integrated spectrum of the data cube.

        Handles:
        - Setting up x-axis (wavelength/velocity)
        - Plotting the spectrum data
        - Adding appropriate axis labels
        - Creating slice indicator lines
        - Handling invalid data cases

        Args:
            spectrum (numpy.ndarray): 1D array of spectral values

        Note:
            - Uses wavelength values if available, otherwise index
            - Handles all-NaN spectra gracefully
            - Creates persistent vertical lines for slice indication
            - Sets up proper axis labels based on units
        """
        if spectrum is None:
            print("Warning: Cannot plot spectrum - data is None")
            return
            
        self.ax.clear()
        
        try:
            # Create x-axis values
            if self.wavelength is not None and len(self.wavelength) == len(spectrum):
                x = self.wavelength
            else:
                x = np.arange(len(spectrum))
            
            # Ensure data is valid
            if np.all(np.isnan(spectrum)):
                print("Warning: Spectrum contains all NaN values")
                spectrum = np.zeros_like(spectrum)
            
            # Clear any existing plot lines and reset storage
            self.ax.clear()
            self.region_lines = {}
            self.vline_left = None
            self.vline_right = None
            
            # Plot main spectrum
            self.ax.plot(x, spectrum, label='Total Spectrum')

            # X axis label depends on units
            if self.wave_unit in ["Å", "µm"]:
                xlabel = f"Wavelength [{self.wave_unit}]"
            elif self.wave_unit == "km/s":
                xlabel = f"Velocity [{self.wave_unit}]"
            else:
                xlabel = "Spectral axis"
    
            self.ax.set_xlabel(xlabel)
            # Get flux units from data if available
            if hasattr(self.cube_data, 'flux_unit') and self.cube_data.flux_unit:
                ylabel = f"Flux [{self.cube_data.flux_unit}]"
            else:
                # Fallback to standard units based on instrument type
                if self.wave_unit == "Å":
                    ylabel = "Flux [10⁻²⁰ erg s⁻¹ cm⁻² Å⁻¹]"  # Standard MUSE units
                elif self.wave_unit == "µm":
                    ylabel = "Flux [Jy]"  # Common unit for IR observations
                elif self.wave_unit == "km/s":
                    ylabel = "Flux [K]"   # Common for radio/velocity cubes
                else:
                    ylabel = "Flux [arbitrary units]"
            self.ax.set_ylabel(ylabel)
                        # Create vertical lines for slice indicators
            # Create vertical lines for slice indicators
            ymin, ymax = self.ax.get_ylim()
            # label these so they appear in the legend as 'Left slice' / 'Right slice'
            # Use cyan for left slice, magenta for right slice to match region colors
            self.vline_left = self.ax.plot([x[0], x[0]], [ymin, ymax], '--', color='cyan', 
                                         lw=1.5, label='Left slice', zorder=3)[0]
            self.vline_right = self.ax.plot([x[0], x[0]], [ymin, ymax], '--', color='magenta', 
                                          lw=1.5, label='Right slice', zorder=3)[0]
            self.ax.set_ylim(ymin, ymax)
            self.ax.legend(loc='upper right', fontsize=8)
            
        except Exception as e:
            print(f"Error plotting spectrum: {e}")
            # Create an empty plot as fallback
            self.ax.plot([], [])
        self.fig.tight_layout()

    def plot_region_spectrum(self, spectrum, color='cyan', label=None):
        """
        Plot the spectrum from a selected region.

        Handles:
        - Removing previous region line with same label
        - Plotting new region spectrum
        - Maintaining legend entries
        - Color coding for different regions

        Args:
            spectrum (numpy.ndarray): 1D array of spectral values for region
            color (str, optional): Color for the plot line. Defaults to 'cyan'
            label (str, optional): Label for legend. If None, uses 'Selection'

        Note:
            - Maintains dictionary of region lines for updating
            - Updates legend automatically
            - Preserves original spectrum in plot
            - Uses consistent color scheme for different sides
        """
        if not hasattr(self, "region_lines"):
            self.region_lines = {}

        # Use the provided label (e.g. 'Left Selection' or 'Right Selection') so
        # the legend shows which side the selection came from.
        plot_label = label if label is not None else 'Selection'

        # Remove previous line for this specific label if present
        if plot_label in self.region_lines:
            try:
                self.region_lines[plot_label].remove()
            except Exception:
                pass

        # Plot the provided spectrum values directly (no normalization).
        # The GUI will provide region spectra scaled appropriately (e.g. summed over selected pixels)
        x = self.wavelength if self.wavelength is not None else np.arange(len(spectrum))
        try:
            plot_values = np.array(spectrum, dtype='float32')
        except Exception:
            plot_values = np.zeros_like(x, dtype='float32')

        line, = self.ax.plot(x, plot_values, color=color, label=plot_label)
        self.region_lines[plot_label] = line

        # Update legend to include slice vlines and selection lines
        try:
            self.ax.legend(loc='upper right', fontsize=8)
        except Exception:
            pass

    def update_slice_indicator(self, wavelength_idx, target="left"):
        """
        Update the position of a slice indicator line.

        Moves the vertical line showing current slice position for either
        the left or right display.

        Args:
            wavelength_idx (int): Index into wavelength array for new position
            target (str, optional): Which line to update ('left' or 'right')

        Note:
            - Maintains y-axis limits when updating
            - Handles missing wavelength data gracefully
            - Creates new line if one doesn't exist
            - Color codes lines by side (red=left, blue=right)
            - Safe to call if lines haven't been created yet
        """
        if self.wavelength is None:
            return
            
        try:
            # Get the wavelength value
            wave = self.wavelength[wavelength_idx]
            
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
        """
        Initialize the slice plotter.

        Args:
            figure (matplotlib.figure.Figure): Figure to plot on

        Note:
            - Sets up axes with custom positioning
            - Initializes empty attributes for later use
        """
        self.fig = figure
        # Clear any existing axes
        self.fig.clear()
        # Create new axes with specific position
        self.ax = self.fig.add_subplot(111)
        self.ax.set_position([0.1, 0.1, 0.8, 0.8])  # Adjust to leave room for colorbar
        self.im = None
        self.cbar = None
        self.pixel_marker = None
        # Draw empty axis
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.canvas.draw()

    def plot_slice(self, image, title=""):
        """
        Plot a 2D slice image with colorbar.

        Args:
            image (numpy.ndarray): 2D array of intensity values
            title (str, optional): Title for the plot

        Note:
            - Uses 'magma' colormap for visualization
            - Origin at lower left (astronomical convention)
            - Nearest neighbor interpolation for pixel accuracy
            - Creates or updates colorbar as needed
            - Maintains consistent colorbar size and position
        """
        # First plot: create image and fixed colorbar
        if self.im is None:
            self.ax.clear()
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_title(title)

            self.im = self.ax.imshow(image, origin='lower', interpolation='nearest', cmap='magma')

            # Create fixed colorbar using axes divider
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            try:
                self.cbar = self.fig.colorbar(self.im, cax=cax)
                self.cbar.set_label("Flux / Intensity")
            except Exception as e:
                print(f"Warning: Could not create colorbar: {e}")
                self.cbar = None

        else:
            # Update existing image
            self.im.set_data(image)
            self.im.set_clim(np.nanmin(image), np.nanmax(image))
            self.ax.set_title(title)
            if self.cbar is not None:
                self.cbar.update_normal(self.im)

        # Keep ticks off
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Redraw canvas
        self.fig.canvas.draw_idle()

    def update_pixel_marker(self, x, y):
        """
        Update or create a marker showing selected pixel position.

        Args:
            x (int): X coordinate of selected pixel
            y (int): Y coordinate of selected pixel

        Note:
            - Removes any existing markers before adding new one
            - Uses red plus symbol (+) for visibility
            - Consistent marker size and line width
            - Maintains list of markers for cleanup
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