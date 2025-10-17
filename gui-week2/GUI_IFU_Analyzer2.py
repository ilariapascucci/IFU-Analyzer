# Week 3: Oct 13 - Oct 17

'''
Goals:
- Create GUI tool that shows a cube's spectrum and any slice at the same time.
'''

'''
Achievements:
Week 1.
- Create GUI template using PyQt.
- Develop a Cube class with all the necessary functions.
- Import FITS files and match them with predetermine telescope options.
- Create a spectrum given the cube.
- Show a specific slice by zooming and clicking on the spectrum.
- Display data of the specific slide (wavelength, intensity, RA, DEC).
- Implement process for different telescopes.

Week 2.
- Create GitHub repo to store weekly advancements and compare scripts.
- Move slice information to the left.
- RIGHT SIDE: Add second slice for comparison.
- Get instrument info from file and remove dropbox.
- Zoom in slices and click pixels.
- Get pixel info after click (RA/DEC, intensity, coordinates).
'''

'''
Next Steps:
- Test with more telescopes (need more FITS files).
- Display more cube information (e.g., pixel size and position angle in sky).
- Avoid unit misuse (especially with ALMA telescopes).
- Get new ALMA FITS file.
- README and additonal documentation (expected outputs and commands).
- Display pixel-specific data (unique RA/DEC?, pixel coordinate of reference point?)
'''

"""
Basic Cube Explorer:
- Open a FITS cube (first 3D HDU)
- Bottom: full-spectrum (sum over spatial pixels)
- Top: slice comparison (plot and stats)
- Stats: min/max/mean and RA/DEC
"""

# General libraries
import sys
import numpy as np

# Astronomy libraries
from astropy.io import fits
from astropy.wcs import WCS

# GUI libraries
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QFileDialog, QLabel, QFrame, QMessageBox
)

# Plot libraries
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


# Information from each telescope -- constants, so more telescopes can be added
TELESCOPE_LOADERS = {
    "MUSE": lambda hdul: (hdul[1].header, hdul[1].data),
    "ALMA": lambda hdul: (hdul[0].header, hdul[0].data),
    "MIRI": lambda hdul: (hdul["SCI"].header, hdul["SCI"].data),
    "NIRSpec": lambda hdul: (hdul["SCI"].header, hdul["SCI"].data),
}


# Cube class -- includes values and UI initialization
class CubeExplorer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cube Explorer")
        self.cube = None  # loaded 3d numpy array
        self.header = None  # fits header
        self.wavelength = None  # 1d array of wavelengths
        self.wave_unit = ""  # units
        self.spec_axis = 0  # axes
        self.spectrum = None  # 1d spectrum of sums
        self.wcs = WCS(self.header)
        self._init_ui()  # function to initialize GUI


    # INITIALIZE GUI -- LAYOUT AND WIDGETS
    def _init_ui(self):
        # File Controls
        open_btn = QPushButton("Open FITS")
        open_btn.clicked.connect(self.select_file)
        load_btn = QPushButton("Load Cube")
        load_btn.clicked.connect(self.load_cube)
        
        self.instrument_label = QLabel("Instrument: -")
        self.instrument_label.setStyleSheet("font-weight: bold;")

        controls = QHBoxLayout()
        controls.addWidget(open_btn)
        controls.addWidget(load_btn)
        controls.addWidget(self.instrument_label)
        controls.addStretch()

        # Slice 1 Stats
        self.slice1_fig = Figure(figsize=(4, 4))
        self.slice1_canvas = FigureCanvas(self.slice1_fig)
        self.slice1_ax = self.slice1_fig.add_subplot(111)
        self.stats1_label = QLabel("Slice 1 Stats:\n-")
        self.stats1_label.setMinimumWidth(150)

        # Slice 2 Stats
        self.slice2_fig = Figure(figsize=(4, 4))
        self.slice2_canvas = FigureCanvas(self.slice2_fig)
        self.slice2_ax = self.slice2_fig.add_subplot(111)
        self.stats2_label = QLabel("Slice 2 Stats:\n-")
        self.stats2_label.setMinimumWidth(150)

        # Slice 1 Plot
        self.slice1_fig = Figure(figsize=(4, 4))
        self.slice1_canvas = FigureCanvas(self.slice1_fig)
        self.slice1_ax = self.slice1_fig.add_subplot(111)
        self.slice1_toolbar = NavigationToolbar(self.slice1_canvas, self)
        self.slice1_canvas.mpl_connect("button_press_event", lambda e: self.on_slice_click(e, "left"))

        # Slice 2 Plot
        self.slice2_fig = Figure(figsize=(4, 4))
        self.slice2_canvas = FigureCanvas(self.slice2_fig)
        self.slice2_ax = self.slice2_fig.add_subplot(111)
        self.slice2_toolbar = NavigationToolbar(self.slice2_canvas, self)
        self.slice2_canvas.mpl_connect("button_press_event", lambda e: self.on_slice_click(e, "right"))

        # Top Layout: Stats and Slices
        top_layout = QHBoxLayout()

        # Left Stats + Slice
        left_stats = QVBoxLayout()
        left_stats.addWidget(self.stats1_label)
        left_stats.addStretch()  # keeps the label top-aligned

        left_slice = QVBoxLayout()
        left_slice.addWidget(self.slice1_canvas)
        left_slice.addWidget(self.slice1_toolbar)

        # Right Stats + Slice
        right_stats = QVBoxLayout()
        right_stats.addWidget(self.stats2_label)
        right_stats.addStretch()

        right_slice = QVBoxLayout()
        right_slice.addWidget(self.slice2_canvas)
        right_slice.addWidget(self.slice2_toolbar)

        # 1:3 ratio for both halves
        top_layout.addLayout(left_stats, 1)   # 12.5%
        top_layout.addLayout(left_slice, 3)   # 37.5%
        top_layout.addLayout(right_stats, 1)  # 12.5%
        top_layout.addLayout(right_slice, 3)  # 37.5%

        # Slice selection controls
        self.set_left_btn = QPushButton("Set Left Slice")
        self.set_right_btn = QPushButton("Set Right Slice")
        self.set_left_btn.setCheckable(True)
        self.set_right_btn.setCheckable(True)

        self.set_left_btn.clicked.connect(lambda: self._select_target("left"))
        self.set_right_btn.clicked.connect(lambda: self._select_target("right"))

        slice_select_layout = QHBoxLayout()
        slice_select_layout.addWidget(self.set_left_btn)
        slice_select_layout.addWidget(self.set_right_btn)
        slice_select_layout.addStretch()

        # Spectrum (Bottom half)
        self.spec_fig = Figure(figsize=(8, 3))
        self.spec_canvas = FigureCanvas(self.spec_fig)
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_toolbar = NavigationToolbar(self.spec_canvas, self)
        self.spec_canvas.mpl_connect("button_press_event", self.on_spec_click)

        # Final main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(controls)
        main_layout.addLayout(top_layout)
        main_layout.addLayout(slice_select_layout)
        main_layout.addWidget(self.spec_canvas)
        main_layout.addWidget(self.spec_toolbar)

        self.setLayout(main_layout)

        # Default state
        self.active_target = None


    # OPENING A FITS CUBE AFTER SELECTING A FILE
    def select_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open FITS", filter="FITS files (*.fits *.fz *.fit)")
        if fname:
            self.selected_file = fname  # store path
            QMessageBox.information(self, "File Selected", f"Selected: {fname}")

    def load_cube(self):
        if not hasattr(self, "selected_file"):
            QMessageBox.warning(self, "No File", "Please open a FITS file first.")
            return

        try:
            hdul = fits.open(self.selected_file, memmap=True)

            # get instrument
            instrument = self._detect_instrument(hdul)
            self.instrument_label.setText(f"Instrument: {instrument}")
            
            if instrument is None:
                QMessageBox.warning(self, "Unknown Instrument", "Could not detect instrument from FITS headers.")
                return
            
            self.instrument = instrument

            QMessageBox.information(self, "Instrument Detected", f"Detected instrument: {instrument}")

            # Find telescope in selection
            header, data = TELESCOPE_LOADERS[instrument](hdul)

            # Squeeze axes to get 3
            data = np.squeeze(data)

            if data.ndim != 3:
                QMessageBox.warning(self, "Invalid FITS data",
                                f"Expected 3D cube after squeezing, but got shape {data.shape}")
                return

            self.cube = np.array(data, dtype=float)  # ensure float
            self.cube = np.squeeze(self.cube)

            self.header = header
            self.spec_axis = self._infer_spec_axis(header, self.cube)

            if self.spec_axis != 0:
                self.cube = np.moveaxis(self.cube, self.spec_axis, 0)

            self.wavelength, self.wave_unit = self._get_wavelength(header, self.cube, self.spec_axis)

            if self.wavelength is not None and len(self.wavelength) != self.cube.shape[self.spec_axis]:
            # Trim or pad wavelength if necessary
                n_wave = self.cube.shape[self.spec_axis]
                self.wavelength = self.wavelength[:n_wave]

            self.spectrum = self._compute_spectrum(self.cube, self.spec_axis)
            
            # reverse wavelength and spectrum so axis increases left->right. ALMA, for example
            if self.wavelength is not None and self.wavelength[0] > self.wavelength[-1]:
                self.wavelength = self.wavelength[::-1]
                self.spectrum = self.spectrum[::-1]

            self._plot_spectrum()

            # initial slice in the middle
            self._display_slice(0, target="left")
            self._display_slice(0, target="right")

            print("Cube shape:", self.cube.shape)
            print("Spectrum shape:", self.spectrum.shape)
            print("Wavelength length:", len(self.wavelength))
            print("Wavelength min/max:", self.wavelength.min(), self.wavelength.max())

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error reading FITS: {e}")
            raise


    # Find telescope name in fits file
    def _detect_instrument(self, hdul):
        """Try to detect the instrument name from FITS headers."""
        for hdu in hdul:
            hdr = hdu.header
            instrument = hdr.get("INSTRUME", "").strip().upper()
            telescope = hdr.get("TELESCOP", "").strip().upper()

            # Match to known telescope keywords -- get by cases
            if "MUSE" in instrument or "MUSE" in telescope:
                return "MUSE"
            if "ALMA" in instrument or "ALMA" in telescope:
                return "ALMA"
            if "MIRI" in instrument or "MIRI" in telescope:
                return "MIRI"
            if "NIRSPEC" in instrument or "NIRSPEC" in telescope:
                return "NIRSpec"

        return None  # if no match found


    # Axis data
    def _infer_spec_axis(self, header, data):
        """Heuristic: try NAXIS3; otherwise assume axis 0 (common: (nw, ny, nx))."""
        try:
            naxis3 = header.get('NAXIS3', None)
            if naxis3 is not None:
                # mapping: FITS axis j corresponds to numpy axis data.ndim - j
                if naxis3 == data.shape[0]:
                    return 0
                elif naxis3 == data.shape[-1]:
                    return data.ndim - 1
        except Exception:
            pass
        return 0


    # UNIQUE FOR EACH TELESCOPE -- CALCULATE THEM INDIVIDUALLY
    def _get_wavelength(self, header, cube, spec_axis):
        """
        Try:
        1) CRVAL/CRPIX/CDELT for the header axis that corresponds to spec_axis
        2) WCS-based world coordinate calculation
        Returns (wavelength_array, unit_string)
        """

        instrument = getattr(self, "instrument", None)

        if instrument == "MUSE":
            crval3 = header["CRVAL3"]
            crpix3 = header["CRPIX3"]
            cd33   = header["CD3_3"]
            n_slices = header["NAXIS3"]
            wavelengths = crval3 + (np.arange(1, n_slices + 1) - crpix3) * cd33
            return wavelengths, "Å"

        elif instrument == "ALMA":
            crval3 = header["CRVAL3"]
            cdelt3 = header["CDELT3"]
            crpix3 = header["CRPIX3"]
            n_slices = header["NAXIS3"]
            #freqs = crval3 + (np.arange(1, n_slices + 1) - crpix3) * cdelt3
            #c = 2.99792458e8
            #wavelengths = c / freqs * 1e6  # microns
            #return wavelengths, "µm"
            
            # ALMA spectral axis is usually radial velocity (VRAD) in m/s
            velocities = crval3 + (np.arange(1, n_slices + 1) - crpix3) * cdelt3
            velocities /= 1000.0  # convert to km/s for plotting
            return velocities, "km/s"

        elif instrument in ["MIRI", "NIRSpec"]:
            # JWST instruments often have WCS in microns already
            crval3 = header["CRVAL3"]
            cdelt3 = header["CDELT3"]
            crpix3 = header["CRPIX3"]
            n_slices = header["NAXIS3"]
            wavelengths = crval3 + (np.arange(1, n_slices + 1) - crpix3) * cdelt3
            return wavelengths, "µm"


    # Sum of flux for spectrum
    def _compute_spectrum(self, cube, spec_axis):
        axes = tuple(i for i in range(cube.ndim) if i != spec_axis)
        spec = np.nansum(cube, axis=axes)
        return spec


    # Plot spectrum with appropritate units
    def _plot_spectrum(self):
        self.spec_ax.clear()

        x = self.wavelength if self.wavelength is not None else np.arange(len(self.spectrum))
        self.spec_ax.plot(x, self.spectrum)

        # X axis label depends on units
        if self.wave_unit in ["Å", "µm"]:
            xlabel = "Wavelength [" + self.wave_unit + "]"
        elif self.wave_unit == "km/s":
            xlabel = "Velocity [" + self.wave_unit + "]"
        else:
            xlabel = "Spectral axis"
    
        self.spec_ax.set_xlabel(xlabel)
        self.spec_ax.set_ylabel("Flux (sum over spatial pixels)")
        self.spec_canvas.draw()
        self.spec_fig.tight_layout()  # automatically adjusts spacing
        
        # Create persistent vertical lines for left and right slices
        # Remove old ones if they exist
        if hasattr(self, 'vline_left') and self.vline_left:
            self.vline_left.remove()
        if hasattr(self, 'vline_right') and self.vline_right:
            self.vline_right.remove()

        ymin, ymax = self.spec_ax.get_ylim()

        # Start both at the beginning of the spectrum
        self.vline_left = self.spec_ax.plot(
            [self.wavelength[0], self.wavelength[0]],
            [ymin, ymax], 'r--', lw=1.2, label='Left slice')[0]

        self.vline_right = self.spec_ax.plot(
            [self.wavelength[0], self.wavelength[0]],
            [ymin, ymax], 'b--', lw=1.2, label='Right slice')[0]
        
        self.spec_ax.set_ylim(ymin, ymax)

        self.spec_ax.legend(loc='upper right', fontsize=8)


    # Select which side to get slice from (left or right)
    def _select_target(self, target):
        """Handle slice selection mode (left/right)."""
        if target == "left":
            self.set_right_btn.setChecked(False)
            self.active_target = "left" if self.set_left_btn.isChecked() else None
        elif target == "right":
            self.set_left_btn.setChecked(False)
            self.active_target = "right" if self.set_right_btn.isChecked() else None

        print("Active target:", self.active_target)


    # Display slice when clicked
    def on_spec_click(self, event):
        """When clicking on the spectrum, update the corresponding slice."""
        if event.inaxes != self.spec_ax or self.active_target is None:
            return

        # Find nearest wavelength index
        if self.wavelength is None:
            return
        click_x = event.xdata
        if click_x is None:
            return

        idx = (np.abs(self.wavelength - click_x)).argmin()

        ymin, ymax = self.spec_ax.get_ylim()

        # Move the appropriate vertical line
        if self.active_target == "left":
            if hasattr(self, 'vline_left'):
                self.vline_left.set_xdata([self.wavelength[idx], self.wavelength[idx]])
            else:
                self.vline_left = self.spec_ax.plot(
                    [self.wavelength[idx], self.wavelength[idx]], [ymin, ymax], 'r--', lw=1.2)[0]
            self._display_slice(idx, target="left")

        elif self.active_target == "right":
            if hasattr(self, 'vline_right'):
                self.vline_right.set_xdata([self.wavelength[idx], self.wavelength[idx]])
            else:
                self.vline_right = self.spec_ax.plot(
                    [self.wavelength[idx], self.wavelength[idx]], [ymin, ymax], 'b--', lw=1.2)[0]
            self._display_slice(idx, target="right")

        self.spec_canvas.draw_idle()


    # Display work
    def _display_slice(self, idx, target="left"):
        """Display a slice image on the chosen side."""
        image = np.take(self.cube, indices=idx, axis=self.spec_axis)

        # Select which side to update
        if target == "left":
            ax = self.slice1_ax
            canvas = self.slice1_canvas
            stats_label = self.stats1_label
            fig = self.slice1_fig
        else:
            ax = self.slice2_ax
            canvas = self.slice2_canvas
            stats_label = self.stats2_label
            fig = self.slice2_fig

        # Clear and show the image
        ax.clear()
        im = ax.imshow(image, origin='lower', interpolation='nearest', cmap='magma')
        ax.set_title(f"{target.capitalize()} Slice @ idx={idx}")

        # Handle colorbar
        if target == "left":
            self.last_idx_left = idx
            # Create or update the LEFT colorbar
            if not hasattr(self, 'cbar_left') or self.cbar_left is None:
                self.cbar_left = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                self.cbar_left.set_label("Flux / Intensity")
            else:
                self.cbar_left.update_normal(im)
        else:
            self.last_idx_right = idx
            # Create or update the RIGHT colorbar
            if not hasattr(self, 'cbar_right') or self.cbar_right is None:
                self.cbar_right = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                self.cbar_right.set_label("Flux / Intensity")
            else:
                self.cbar_right.update_normal(im)

        fig.tight_layout(pad=2.0)   # padding

        # Stats display
        mn, mx, mean = np.nanmin(image), np.nanmax(image), np.nanmean(image)

        instrument = getattr(self, "instrument", None)
        if instrument == "MUSE":
            ra = self.header.get("CRVAL1", np.nan)
            dec = self.header.get("CRVAL2", np.nan)
        elif instrument == "ALMA":
            ra = self.header.get("RA", np.nan)
            dec = self.header.get("DEC", np.nan)
        elif instrument in ["MIRI", "NIRSpec"]:
            ra = self.header.get("RA_V1", np.nan)
            dec = self.header.get("DEC_V1", np.nan)
        else:
            ra = dec = np.nan

        stats_label.setText(
            f"Intensity:\n"
            f"min={mn:.3e}\n"
            f"max={mx:.3e}\n"
            f"mean={mean:.3e}\n"
            f"RA={ra:.6f}\n"
            f"DEC={dec:.6f}"
        )

        # Redraw
        canvas.draw_idle()

    # Pixel information
    def on_slice_click(self, event, target):
        """Handle clicks on left or right slice: get pixel intensity and show a single marker."""
        if event.inaxes is None:
            return

        # Determine which image axes were clicked
        ax = self.slice1_ax if target == "left" else self.slice2_ax
        canvas = self.slice1_canvas if target == "left" else self.slice2_canvas
        stats_label = self.stats1_label if target == "left" else self.stats2_label

        if event.inaxes != ax:
            return  # clicked outside the relevant subplot

        # Get pixel coordinates (rounded to nearest int)
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        idx = self.last_idx_left if target == "left" else self.last_idx_right

        # Bounds check
        if x < 0 or y < 0:
            return
        image = np.take(
            self.cube,
            indices=idx,
            axis=self.spec_axis)
        if y >= image.shape[0] or x >= image.shape[1]:
            return

        intensity = image[y, x]

        # Store last clicked pixel
        if target == "left":
            self.last_clicked_left = (x, y, intensity)
        else:
            self.last_clicked_right = (x, y, intensity)

        # Convert to world coordinates (RA, DEC, wavelength)
        if hasattr(self, "wcs"):
            try:
                # Only pass x, y for 2D WCS
                ra, dec = self.wcs.all_pix2world([[x, y]], 0)[0]
                # Wavelength from cube index
                lam = self.wavelength[idx] if self.wavelength is not None else None
            except Exception as e:
                print("WCS conversion error:", e)
                ra, dec, lam = None, None, None
        else:
            ra, dec, lam = None, None, None


        # Update stats label
        old_text = stats_label.text().split("\nPixel:")[0]  # remove old pixel info
        coord_text = f"Pixel: (x={x}, y={y})\nIntensity={intensity:.3e}"
        if ra is not None:
            coord_text += f"\nRA={ra:.5f}°\nDEC={dec:.5f}°"

        stats_label.setText(f"{old_text}\n{coord_text}")

        # Update visual marker
        for artist in getattr(ax, "pixel_marker", []):
            artist.remove()
        ax.pixel_marker = [ax.plot(x, y, 'r+', markersize=8, markeredgewidth=1.5)[0]]

        canvas = self.slice1_canvas if target == "left" else self.slice2_canvas
        canvas.draw_idle()


# Run script
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CubeExplorer()
    w.resize(1100, 800)
    w.show()
    sys.exit(app.exec_())