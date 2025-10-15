# Week 2: Oct 6 - Oct 10

'''
Goals:
- Create GUI tool that shows a cube's spectrum and any slice at the same time.
'''

'''
Achievements:
- Create GUI template using PyQt.
- Develop a Cube class with all the necessary functions.
- Import FITS files and match them with predetermine telescope options.
- Create a spectrum given the cube.
- Show a specific slice by zooming and clicking on the spectrum.
- Display data of the specific slide (wavelength, intensity, RA, DEC).
- Implement process for different telescopes.
'''

'''
Next Steps:
- Create GitHub repo to store weekly advancements and compare scripts.
- Test with more telescopes (need more FITS files).
- Display more cube information (e.g., pixel size and position angle in sky).
- Avoid unit misuse (especially with ALMA telescopes).
- Get instrument info from file and remove dropbox.
- Zoom in slices and click pixels.
- Move slice information to the left.
- Get pixel info after click (RA/DEC, intensity, coordinates).
- RIGHT SIDE: Add second slice for comparison.
- README and additonal documentation (expected outputs and commands).
'''

"""
Basic Cube Explorer:
- Open a FITS cube (first 3D HDU)
- Instrument dropdown (placeholder for now)
- Bottom: full-spectrum (sum over spatial pixels)
- Top-left: image slice at selected wavelength (click spectrum to pick)
- Top-left overlay: min/max/mean and RA/DEC
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
        self._init_ui()  # function to initialize GUI


    # INITIALIZE GUI -- LAYOUT AND WIDGETS
    def _init_ui(self):
        # Top controls
        open_btn = QPushButton("Open FITS")  # file dialog
        open_btn.clicked.connect(self.select_file)
        self.instrument_cb = QComboBox()  # dropdown
        self.instrument_cb.addItems(["MUSE", "ALMA", "MIRI", "NIRSpec"])

        # button to load cube
        load_btn = QPushButton("Load Cube")
        load_btn.clicked.connect(self.load_cube)

        # dropdown
        controls = QHBoxLayout()
        controls.addWidget(open_btn)
        controls.addWidget(QLabel("Instrument:"))
        controls.addWidget(self.instrument_cb)
        controls.addWidget(load_btn)
        controls.addStretch()

        # Top area: left = slice canvas, right = placeholder
        top_h = QHBoxLayout()
        self.slice_fig = Figure(figsize=(5,4))
        self.slice_canvas = FigureCanvas(self.slice_fig)
        self.slice_ax = self.slice_fig.add_subplot(111)
        top_h.addWidget(self.slice_canvas, 3)

        placeholder = QFrame()
        placeholder.setFrameShape(QFrame.StyledPanel)
        top_h.addWidget(placeholder, 1)

        # Bottom: spectrum
        self.spec_fig = Figure(figsize=(8,3))
        self.spec_canvas = FigureCanvas(self.spec_fig)
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_canvas.mpl_connect("button_press_event", self.on_spec_click)

        # toolbar
        self.spec_toolbar = NavigationToolbar(self.spec_canvas, self)

        # Layout
        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addLayout(top_h)
        layout.addWidget(self.spec_canvas)
        layout.addWidget(self.spec_toolbar)
        self.setLayout(layout)


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

        # get instrument value
        instrument = self.instrument_cb.currentText()

        try:
            hdul = fits.open(self.selected_file, memmap=True)

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

            # get global min and max for colorbar consistency
            #self.global_vmin = np.nanmin(self.cube)
            #self.global_vmax = np.nanmax(self.cube)

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
            if self.wavelength is not None and len(self.wavelength) > 0:
                idx = len(self.wavelength) // 2
            else:
                idx = 0
            self._display_slice(idx)

            print("Cube shape:", self.cube.shape)
            print("Spectrum shape:", self.spectrum.shape)
            print("Wavelength length:", len(self.wavelength))
            print("Wavelength min/max:", self.wavelength.min(), self.wavelength.max())

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error reading FITS: {e}")
            raise


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

        instrument = self.instrument_cb.currentText()

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
        
        # vertical line placeholder --  makes it better to see which slice is clicked
        if hasattr(self, 'vline') and self.vline:
            self.vline.remove()
        self.vline = self.spec_ax.plot([self.wavelength[0], self.wavelength[0]],
                               self.spec_ax.get_ylim(), 'r--', lw=1.2)[0]


    # Display slice when clicked
    def on_spec_click(self, event):
        # triggered when user clicks on the bottom spectrum
        if event.inaxes != self.spec_ax:
            return
        x = event.xdata
        if x is None or self.wavelength is None:
            return
        
        idx = int(np.argmin(np.abs(self.wavelength - x)))

        # update vertical line
        self.vline.set_xdata([self.wavelength[idx], self.wavelength[idx]])  # must be a list
        self.vline.set_ydata(self.spec_ax.get_ylim())  # keep it spanning full y-axis
        self.spec_canvas.draw_idle()
        self._display_slice(idx)


    # Display work
    def _display_slice(self, idx):
        image = np.take(self.cube, indices=idx, axis=self.spec_axis)
        self.slice_ax.clear()

        # show image
        im = self.slice_ax.imshow(image, origin='lower', interpolation='nearest')
        self.slice_ax.set_title(f"Slice at index={idx}")

        # handle colorbar -- flux values change for color consistency
        if not hasattr(self, 'cbar') or self.cbar is None:
            # Create the colorbar only once
            self.cbar = self.slice_fig.colorbar(im, ax=self.slice_ax, fraction=0.046, pad=0.04)
            self.cbar.set_label("Flux / Intensity")
        else:
            # Update existing colorbar for new data
            self.cbar.update_normal(im)

        # intensity stats (min, max, mean)
        mn = np.nanmin(image)
        mx = np.nanmax(image)
        mean = np.nanmean(image)

        stats_text = f"Intensity:\nmin={mn:.3e}\nmax={mx:.3e}\nmean={mean:.3e}"
        self.slice_ax.text(0.01, 0.99, stats_text, transform=self.slice_ax.transAxes,
                        va='top', ha='left', fontsize=9, bbox=dict(facecolor='white', alpha=0.75))
    
        # wavelength top-right
        wave_val = self.wavelength[idx] if self.wavelength is not None else idx
        self.slice_ax.text(0.99, 0.99, f"Wavelength: {wave_val:.6g} {self.wave_unit}",
                        transform=self.slice_ax.transAxes, va='top', ha='right', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.75))
    
        # RA/DEC (use WCS pix->world)
        ra_dec_text = ""

        instrument = self.instrument_cb.currentText()

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

        ra_dec_text = f"RA={ra:.6f}\nDEC={dec:.6f}"

        '''
        try:
            w = WCS(self.header)
            ctypes = w.wcs.ctype
            nax = self.header.get('NAXIS', self.cube.ndim)
            centers = [(self.cube.shape[i]-1)/2.0 for i in range(self.cube.ndim)]
            pix = np.zeros((1, nax))

            for j in range(nax):
                data_axis = self.cube.ndim - (j+1)
                if data_axis == self.spec_axis:
                    pix[0, j] = idx + 1
                else:
                    pix[0, j] = centers[data_axis] + 1
                    
            world = w.wcs_pix2world(pix, 1)[0]
            ra_idx = next((i for i, ct in enumerate(ctypes) if 'RA' in ct.upper()), None)
            dec_idx = next((i for i, ct in enumerate(ctypes) if 'DEC' in ct.upper()), None)
            
            if ra_idx is not None and dec_idx is not None:
                ra = world[ra_idx]
                dec = world[dec_idx]
                ra_dec_text = f"\nRA={ra:.6f}\nDEC={dec:.6f}"
            else:
                ra_dec_text = "\nRA/DEC: n/a"

        except Exception:
            ra_dec_text = "\nRA/DEC: n/a"
        '''
        self.slice_ax.text(0.01, 0.75, ra_dec_text, transform=self.slice_ax.transAxes,
                        va='top', ha='left', fontsize=9, bbox=dict(facecolor='white', alpha=0.75))
        self.slice_canvas.draw_idle()


# Run script
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CubeExplorer()
    w.resize(1100, 800)
    w.show()
    sys.exit(app.exec_())