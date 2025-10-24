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

Week 3.
- Upgrade GUI placement and visualization.
- Update pixel RA/DEC using VAL, PIX, DELTA.
- Select region from different shapes (rectangle, circle, ellipse, point, custom).
- Upgrade slice comparison (copy settings)
'''

'''
Next Steps:
- Test with more telescopes (need more FITS files).
- Display more cube information (e.g., pixel size and position angle in sky).
- Avoid unit misuse (especially with ALMA telescopes).
- Get new ALMA FITS file.
- README and additonal documentation (expected outputs and commands).
- Display pixel-specific data (unique RA/DEC?, pixel coordinate of reference point?)

- Add scaling (function and percentage) and color options -- defaults stay.
- Check into matplotlib toolbar.
- Normalize spectrum and overlay region spectrum (normalize only when comparing).
- Reform file structure.
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
import matplotlib

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
from matplotlib.widgets import RectangleSelector, EllipseSelector, LassoSelector
from matplotlib.path import Path
from matplotlib.patches import Polygon


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
        self.wcs = None
        self._init_ui()  # function to initialize GUI

    def closeEvent(self, event):
        import gc, matplotlib.pyplot as plt
        try:
            del self.cube
            del self.spectrum
            del self.last_mask
        except AttributeError:
            pass
        plt.close('all')
        gc.collect()
        event.accept()

    def free_memory(self):
        import gc, matplotlib.pyplot as plt
        try: del self.cube
        except AttributeError: pass
        plt.close('all')
        gc.collect()

    # INITIALIZE GUI -- LAYOUT AND WIDGETS
    def _init_ui(self):
        # ---------------- FILE CONTROLS ----------------
        open_btn = QPushButton("Open FITS")
        open_btn.clicked.connect(self.select_file)
        load_btn = QPushButton("Load Cube")
        load_btn.clicked.connect(lambda: (self.free_memory(), self.load_cube()))
        
        self.instrument_label = QLabel("Instrument: -")
        self.instrument_label.setStyleSheet("font-weight: bold;")

        controls = QHBoxLayout()
        controls.addWidget(open_btn)
        controls.addWidget(load_btn)
        controls.addWidget(self.instrument_label)
        controls.addStretch()

        # ---------------- SLICE 1 ----------------
        self.slice1_fig = Figure(figsize=(4, 4))
        self.slice1_canvas = FigureCanvas(self.slice1_fig)
        self.slice1_ax = self.slice1_fig.add_subplot(111)
        self.slice1_ax.set_position([0.05, 0.05, 0.9, 0.9])
        self.slice1_toolbar = NavigationToolbar(self.slice1_canvas, self)
        self.slice1_canvas.mpl_connect("button_press_event", lambda e: self.on_slice_click(e, "left"))

        self.stats1_label = QLabel("Slice 1 Stats:\n-")
        self.stats1_label.setMinimumWidth(150)
        self.stats1_label.setStyleSheet("font-size: 10pt; font-weight: bold;")

        # ---------------- SLICE 2 ----------------
        self.slice2_fig = Figure(figsize=(4, 4))
        self.slice2_canvas = FigureCanvas(self.slice2_fig)
        self.slice2_ax = self.slice2_fig.add_subplot(111)
        self.slice2_ax.set_position([0.05, 0.05, 0.9, 0.9])
        self.slice2_toolbar = NavigationToolbar(self.slice2_canvas, self)
        self.slice2_canvas.mpl_connect("button_press_event", lambda e: self.on_slice_click(e, "right"))

        self.stats2_label = QLabel("Slice 2 Stats:\n-")
        self.stats2_label.setMinimumWidth(150)
        self.stats2_label.setStyleSheet("font-size: 10pt; font-weight: bold;")

        # ---------------- SHAPE BUTTONS (LEFT + RIGHT) ----------------
        # Shared style (compact icon buttons)
        def make_shape_button(symbol, target, callback=None):
            btn = QPushButton(symbol)
            btn.setCheckable(True)
            btn.setFixedSize(36, 36)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 18pt;
                    border: 1px solid #888;
                    border-radius: 6px;
                    background-color: #f0f0f0;
                }
                QPushButton:checked {
                    background-color: #4a90e2;
                    color: white;
                    border: 2px solid #2d73c5;
                }
            """)
            if callback is None:
                btn.clicked.connect(lambda _, s=symbol: self._set_shape(s, target=target))
            else:
                btn.clicked.connect(callback)
            return btn

        # --- Left-side shape buttons ---
        self.pixel_btn_l = make_shape_button("*", "left")
        self.rectangle_btn_l = make_shape_button("▭", "left")
        self.circle_btn_l = make_shape_button("◯", "left")
        self.ellipse_btn_l = make_shape_button("⬭", "left")
        self.custom_btn_l = make_shape_button("✏", "left")
        self.copy_btn_l = make_shape_button("⎘", "left", callback=lambda: self.copy_selection(source="right", target="left"))

        self.left_shape_layout = QVBoxLayout()
        for b in [self.pixel_btn_l, self.rectangle_btn_l, self.circle_btn_l, self.ellipse_btn_l, self.custom_btn_l, self.copy_btn_l]:
            self.left_shape_layout.addWidget(b)
        self.left_shape_layout.addStretch()

        # --- Right-side shape buttons ---
        self.pixel_btn_r = make_shape_button("*", "right")
        self.rectangle_btn_r = make_shape_button("▭", "right")
        self.circle_btn_r = make_shape_button("◯", "right")
        self.ellipse_btn_r = make_shape_button("⬭", "right")
        self.custom_btn_r = make_shape_button("✏", "right")
        self.copy_btn_r = make_shape_button("⎘", "right", callback=lambda: self.copy_selection(source="left", target="right"))

        self.right_shape_layout = QVBoxLayout()
        for b in [self.pixel_btn_r, self.rectangle_btn_r, self.circle_btn_r, self.ellipse_btn_r, self.custom_btn_r, self.copy_btn_r]:
            self.right_shape_layout.addWidget(b)
        self.right_shape_layout.addStretch()

        # ---------------- STATS AND SLICES ----------------
        left_stats = QVBoxLayout()
        left_stats.addWidget(self.stats1_label)
        left_stats.addStretch()

        right_stats = QVBoxLayout()
        right_stats.addWidget(self.stats2_label)
        right_stats.addStretch()

        left_slice = QVBoxLayout()
        left_slice.addWidget(self.slice1_canvas)
        left_slice.addWidget(self.slice1_toolbar)

        right_slice = QVBoxLayout()
        right_slice.addWidget(self.slice2_canvas)
        right_slice.addWidget(self.slice2_toolbar)

        # Combine slice + shape buttons in a row each
        left_slice_row = QHBoxLayout()
        left_slice_row.addLayout(left_slice, 10)
        left_slice_row.addLayout(self.left_shape_layout, 1)

        right_slice_row = QHBoxLayout()
        right_slice_row.addLayout(right_slice, 10)
        right_slice_row.addLayout(self.right_shape_layout, 1)

        # Final top layout (10/40 split)
        top_layout = QHBoxLayout()
        top_layout.addLayout(left_stats, 1)
        top_layout.addLayout(left_slice_row, 3)
        top_layout.addLayout(right_stats, 1)
        top_layout.addLayout(right_slice_row, 3)

        # ---------------- SLICE SELECTION CONTROLS ----------------
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

        # ---------------- SPECTRUM ----------------
        self.spec_fig = Figure(figsize=(8, 3))
        self.spec_canvas = FigureCanvas(self.spec_fig)
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_toolbar = NavigationToolbar(self.spec_canvas, self)
        self.spec_canvas.mpl_connect("button_press_event", self.on_spec_click)

        # ---------------- MAIN LAYOUT ----------------
        main_layout = QVBoxLayout()
        main_layout.addLayout(controls)
        main_layout.addLayout(top_layout)
        main_layout.addLayout(slice_select_layout)
        main_layout.addWidget(self.spec_canvas)
        main_layout.addWidget(self.spec_toolbar)
        self.setLayout(main_layout)

        # Default state
        self.active_shape = None
        self.active_target = None
        self.region_mode = None
        self.selector = None
        self.last_clicked_left = None
        self.last_clicked_right = None
        self.last_idx_left = 0
        self.last_idx_right = 0
        self.last_selection_left = None
        self.last_selection_right = None
        self.slice1_ax.area_artists = []
        self.slice2_ax.area_artists = []

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

            self.wcs = WCS(self.header)

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

        #fig.tight_layout(pad=2.0)   # padding

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
        
        # pixel specific task
        if self.active_shape != '*':
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
        image = np.take(self.cube, indices=idx, axis=self.spec_axis)
        if y >= image.shape[0] or x >= image.shape[1]:
            return

        intensity = image[y, x]

        # Store last clicked pixel
        if target == "left":
            self.last_clicked_left = (x, y, intensity)
        else:
            self.last_clicked_right = (x, y, intensity)

        # --- Coordinate conversion ---
        ra = dec = lam = None
        instrument = getattr(self, "instrument", None)

        try:
            if instrument == "MUSE":
                # TAN projection with CD matrix
                crpix1 = float(self.header["CRPIX1"])
                crpix2 = float(self.header["CRPIX2"])
                crval1 = float(self.header["CRVAL1"])
                crval2 = float(self.header["CRVAL2"])
                cd11 = float(self.header.get("CD1_1", 0.0))
                cd12 = float(self.header.get("CD1_2", 0.0))
                cd21 = float(self.header.get("CD2_1", 0.0))
                cd22 = float(self.header.get("CD2_2", 0.0))

                dx = x + 1 - crpix1
                dy = y + 1 - crpix2

                dra = cd11 * dx + cd12 * dy
                ddec = cd21 * dx + cd22 * dy

                ra = crval1 + dra
                dec = crval2 + ddec

            elif instrument == "ALMA":
                crval1 = float(self.header["CRVAL1"])
                crval2 = float(self.header["CRVAL2"])
                cdelt1 = float(self.header["CDELT1"])
                cdelt2 = float(self.header["CDELT2"])
                crpix1 = float(self.header["CRPIX1"])
                crpix2 = float(self.header["CRPIX2"])
                crota = float(self.header.get("CROTA2", 0.0))

                theta = np.deg2rad(crota)
                dx = x + 1 - crpix1
                dy = y + 1 - crpix2

                dra = dx * cdelt1 * np.cos(theta) - dy * cdelt2 * np.sin(theta)
                ddec = dx * cdelt1 * np.sin(theta) + dy * cdelt2 * np.cos(theta)

                ra = crval1 + dra
                dec = crval2 + ddec

            elif instrument in ["MIRI", "NIRSpec"]:
                crpix1 = float(self.header.get("CRPIX1", 0.0))
                crpix2 = float(self.header.get("CRPIX2", 0.0))
                crval1 = float(self.header.get("CRVAL1", np.nan))
                crval2 = float(self.header.get("CRVAL2", np.nan))
                cdelt1 = float(self.header.get("CDELT1", 0.0))
                cdelt2 = float(self.header.get("CDELT2", 0.0))
                crota = float(self.header.get("CROTA2", 0.0))
                theta = np.deg2rad(crota)

                dx = x + 1 - crpix1
                dy = y + 1 - crpix2

                dra = dx * cdelt1 * np.cos(theta) - dy * cdelt2 * np.sin(theta)
                ddec = dx * cdelt1 * np.sin(theta) + dy * cdelt2 * np.cos(theta)

                ra = crval1 + dra
                dec = crval2 + ddec

            elif hasattr(self, "wcs"):
                # Generic fallback: try WCS directly
                ra, dec = self.wcs.all_pix2world([[x, y]], 0)[0]

            # Wavelength (if available)
            if hasattr(self, "wavelength") and self.wavelength is not None:
                lam = self.wavelength[idx]

        except Exception as e:
            print("Coordinate conversion error:", e)

        # --- Update stats label ---
        old_text = stats_label.text().split("\nPixel:")[0]
        coord_text = f"Pixel: (x={x}, y={y})\nIntensity={intensity:.3e}"
        if ra is not None and dec is not None:
            coord_text += f"\nRA={ra:.6f}°\nDEC={dec:.6f}°"
        if lam is not None:
            coord_text += f"\nλ={lam:.3f}"

        stats_label.setText(f"{old_text}\n{coord_text}")

        # --- Update visual marker ---
        if not hasattr(ax, 'pixel_marker'):
            ax.pixel_marker = []
        for artist in ax.pixel_marker:
            artist.remove()
        
        ax.pixel_marker = [ax.plot(x, y, 'r+', markersize=8, markeredgewidth=1.5)[0]]
        canvas.draw_idle()

        if target == "left":
            self.last_selection_left = {
                "shape": "*",
                "coords": (x, y),
                "mask": None  # pixels don’t need a mask
            }
        else:
            self.last_selection_right = {
                "shape": "*",
                "coords": (x, y),
                "mask": None
            }


    # Options for shape figures
    def _set_shape(self, shape, target=None):
        """Handle shape button clicks — toggles on/off, ensures mutual exclusivity."""
        target = target or self.active_target
        if target not in ["left", "right"]:
            return

        # Identify the correct shape buttons for this side
        if target == "left":
            btns = [self.pixel_btn_l, self.rectangle_btn_l, self.circle_btn_l, self.ellipse_btn_l, self.custom_btn_l]
        else:
            btns = [self.pixel_btn_r, self.rectangle_btn_r, self.circle_btn_r, self.ellipse_btn_r, self.custom_btn_r]

        # Determine which button was clicked
        shape_map = {
            "*": self.pixel_btn_l if target=="left" else self.pixel_btn_r,
            "▭": self.rectangle_btn_l if target=="left" else self.rectangle_btn_r,
            "◯": self.circle_btn_l if target=="left" else self.circle_btn_r,
            "⬭": self.ellipse_btn_l if target=="left" else self.ellipse_btn_r,
            "✏": self.custom_btn_l if target=="left" else self.custom_btn_r,
        }
        clicked_btn = shape_map.get(shape)
        if not clicked_btn:
            return

        # --- TOGGLE LOGIC ---
        if self.active_shape == shape and self.active_target == target:
            # same button clicked again → turn OFF
            clicked_btn.setChecked(False)
            self.active_shape = None
            self.active_target = None

            # disconnect selector (so clicks do nothing)
            if hasattr(self, "selector") and self.selector is not None:
                self.selector.disconnect_events()
                self.selector = None
            print(f"{target} shape deactivated")
            return

        # otherwise, new shape → activate it and uncheck others
        for b in btns:
            b.setChecked(False)
        clicked_btn.setChecked(True)
        
        self.active_shape = shape
        self.active_target = target
        print(f"Active shape: {shape} ({target})")

        # disconnect old selector
        if hasattr(self, "selector") and self.selector is not None:
            self.selector.disconnect_events()
            self.selector = None

        # activate new one
        self._activate_selector()


    # BE ABLE TO DRAW GIVEN THE SHAPE SELECTION
    def _activate_selector(self):
        """Activate the appropriate matplotlib selector for the currently active target."""
        
        if not hasattr(self, "active_shape") or self.active_shape is None:
            return
        if not hasattr(self, "active_target") or self.active_target is None:
            return

        # Choose axes and canvas
        ax = self.slice1_ax if self.active_target == "left" else self.slice2_ax
        canvas = self.slice1_canvas if self.active_target == "left" else self.slice2_canvas

        # Disconnect old selector
        if hasattr(self, "selector") and self.selector is not None:
            self.selector.disconnect_events()
            self.selector = None

        # Create the right selector
        if self.active_shape == "*":    # pixel - already uses on slice click
            canvas.mpl_connect("button_press_event", lambda event: self.on_slice_click(event, self.active_target))

        if self.active_shape == "▭":  # rectangle
            self.selector = RectangleSelector(
                ax, lambda eclick, erelease: self._on_select(eclick, erelease, self.active_target),
                useblit=True, button=[1], minspanx=2, minspany=2, spancoords='pixels',
                interactive=True
            )
        elif self.active_shape == "◯" or self.active_shape == "⬭":  # circle or ellipse
            self.selector = EllipseSelector(
                ax, lambda eclick, erelease: self._on_select(eclick, erelease, self.active_target),
                interactive=True
            )
        elif self.active_shape == "✏":  # custom
            self.selector = LassoSelector(
                ax, lambda verts: self._on_lasso_select(verts, self.active_target)
            )

        canvas.draw_idle()


    # GET VALUES GIVEN A SHAPE
    def _on_select(self, eclick, erelease, target):
        # --- Clear previous region and stats ---
        ax = self.slice1_ax if target == "left" else self.slice2_ax
        for artist in getattr(ax, "area_artists", []):
            artist.remove()
        ax.area_artists = []

        stats_label = self.stats1_label if target == "left" else self.stats2_label

        # --- Get selection bounds ---
        x0, y0 = int(eclick.xdata), int(eclick.ydata)
        x1, y1 = int(erelease.xdata), int(erelease.ydata)
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])

        # --- Get image slice ---
        idx = self.last_idx_left if target == "left" else self.last_idx_right
        image = np.take(self.cube, indices=idx, axis=self.spec_axis)

        xmin, xmax = max(0, xmin), min(image.shape[1]-1, xmax)
        ymin, ymax = max(0, ymin), min(image.shape[0]-1, ymax)

        # --- Build mask depending on shape ---
        yy, xx = np.indices(image.shape)

        if self.active_shape == "◯":  # Circle
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            r = min((xmax - xmin)//2, (ymax - ymin)//2)
            mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
            patch = matplotlib.patches.Circle((cx, cy), r, edgecolor='cyan', facecolor='none', lw=1.5)

        elif self.active_shape == "⬭":  # Ellipse
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            a, b = (xmax - xmin) / 2, (ymax - ymin) / 2
            mask = ((xx - cx) / a)**2 + ((yy - cy) / b)**2 <= 1
            patch = matplotlib.patches.Ellipse((cx, cy), 2*a, 2*b, edgecolor='cyan', facecolor='none', lw=1.5)

        else:  # Rectangle
            mask = np.zeros_like(image, dtype=bool)
            mask[ymin:ymax+1, xmin:xmax+1] = True
            patch = matplotlib.patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                                edgecolor='cyan', facecolor='none', lw=1.5)

        # --- Save mask ---
        self.last_mask = mask
        if target == "left":
            self.last_selection_left = {"shape": self.active_shape, "coords": (xmin, xmax, ymin, ymax), "mask": mask}
            self._plot_region_spectrum(mask, color='cyan', label='Left Selection')
        else:
            self.last_selection_right = {"shape": self.active_shape, "coords": (xmin, xmax, ymin, ymax), "mask": mask}
            self._plot_region_spectrum(mask, color='magenta', label='Right Selection')

        # --- Compute stats ---
        selected_area = image[mask]
        mn, mx, mean = np.nanmin(selected_area), np.nanmax(selected_area), np.nanmean(selected_area)
        stats_label.setText(f"Selected Area Stats:\nmin={mn:.3e}\nmax={mx:.3e}\nmean={mean:.3e}")

        # --- Draw patch ---
        ax.add_patch(patch)
        ax.area_artists.append(patch)
        canvas = self.slice1_canvas if target == "left" else self.slice2_canvas
        canvas.draw_idle()


    # FREEFORM OPTION
    def _on_lasso_select(self, verts, target):        
        ax = self.slice1_ax if target == "left" else self.slice2_ax
        if hasattr(ax, "area_artists"):
            for artist in ax.area_artists:
                artist.remove()
            ax.area_artists.clear()
        else:
            ax.area_artists = []

        stats_label = self.stats1_label if target == "left" else self.stats2_label
        old_text = stats_label.text().split("\nPixel:")[0]
        stats_label.setText(old_text)

        idx = self.last_idx_left if target == "left" else self.last_idx_right
        image = np.take(self.cube, indices=idx, axis=self.spec_axis)
        ny, nx = image.shape

        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        coords = np.vstack((x.flatten(), y.flatten())).T
        path = Path(verts)
        mask = path.contains_points(coords).reshape(ny, nx)

        self.last_mask = mask

        # --- Plot spectrum ---
        if target == "left":
            self.last_selection_left = {"shape": "lasso", "coords": verts, "mask": mask}
            self._plot_region_spectrum(mask, color='cyan', label='Left Selection')
        else:
            self.last_selection_right = {"shape": "lasso", "coords": verts, "mask": mask}
            self._plot_region_spectrum(mask, color='magenta', label='Right Selection')

        selected_area = image[mask]
        mn, mx, mean = np.nanmin(selected_area), np.nanmax(selected_area), np.nanmean(selected_area)
        stats_label.setText(f"{old_text}\nSelected Area Stats:\nmin={mn:.3e}\nmax={mx:.3e}\nmean={mean:.3e}")

        poly_artist = Polygon(verts, edgecolor='magenta', facecolor='none', lw=1.5)
        ax.add_patch(poly_artist)
        ax.area_artists.append(poly_artist)
        canvas = self.slice1_canvas if target == "left" else self.slice2_canvas
        canvas.draw_idle()


    # COPY SLICE SHAPE REGIONS
    def copy_selection(self, source, target):
        # Choose the selection and axes
        sel = self.last_selection_right if source=="right" else self.last_selection_left
        if sel is None:
            return  # nothing to copy

        # Set the target slice active_shape temporarily
        self.active_shape = sel["shape"]

        # For pixel:
        if sel["shape"] == "*":
            x, y = sel["coords"]
            class Event: pass
            event = Event()
            event.inaxes = self.slice1_ax if target=="left" else self.slice2_ax
            event.xdata, event.ydata = x, y
            self.on_slice_click(event, target)
        else:
            # For region: compute the mask on the target slice using same coordinates
            xmin, xmax, ymin, ymax = sel["coords"]
            class EventClick: pass
            class EventRelease: pass
            eclick = EventClick(); eclick.xdata = xmin; eclick.ydata = ymin
            erelease = EventRelease(); erelease.xdata = xmax; erelease.ydata = ymax
            self._on_select(eclick, erelease, target)


    # CALCULATE SPECTRUM FOR REGION
    def compute_region_spectrum(self, mask):
        """
        Compute mean spectrum through wavelength for all pixels inside mask.
        """
        cube = self.cube
        spec_axis = self.spec_axis

        # Reorder so spec_axis is 0 for simplicity
        cube_reordered = np.moveaxis(cube, spec_axis, 0)  # shape (λ, y, x)

        # Apply mask
        masked_pixels = cube_reordered[:, mask]

        # Compute mean across selected pixels for each λ
        spectrum = np.nanmean(masked_pixels, axis=1)
        return spectrum
    

    # PLOT REGION
    def _plot_region_spectrum(self, mask, color='cyan', label=None):
        """Helper to plot region spectrum without clearing axes."""
        if not hasattr(self, "spectrum") or self.spectrum is None:
            return

        region_spectrum = self.compute_region_spectrum(mask)

        # Normalize if you want
        region_spectrum /= np.nanmax(region_spectrum)

        # Keep reference so we can remove old region lines
        if not hasattr(self, "region_lines"):
            self.region_lines = {}

        # Remove previous line for this color/label
        if label in self.region_lines:
            self.region_lines[label].remove()

        # Plot new region spectrum
        line, = self.spec_ax.plot(self.wavelength, region_spectrum, color=color, label=label)
        self.region_lines[label] = line

        # Always keep original spectrum (plot once)
        if not hasattr(self, "original_spectrum_plotted") or not self.original_spectrum_plotted:
            self.spec_ax.plot(self.wavelength, self.spectrum, color='gray', alpha=0.5, label='Original')
            self.original_spectrum_plotted = True

        self.spec_ax.set_xlabel("Wavelength")
        self.spec_ax.set_ylabel("Mean Intensity")
        self.spec_ax.legend()
        self.spec_canvas.draw_idle()


# Run script
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CubeExplorer()
    w.resize(1100, 800)
    w.show()
    sys.exit(app.exec_())