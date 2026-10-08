"""
Gaussian Fitting Dialog and Worker Classes

Contains QThread workers and dialog classes for Gaussian map fitting
to keep the main GUI file cleaner and more maintainable.
"""

import numpy as np
import os
import threading
from .unit_converter import gaussian_map_unit_labels
from .gaussian_fitter import fit_status_map, STATUS_NO_DATA
from .fits_export import build_2d_export_header, build_line_map_header, write_fits_product, _ascii_safe
from .line_catalog import fetch_hitran_data, match_nearest_hitran_line

# Default colors for the categorical Fit Status map, in STATUS_* code order:
# neutral grey for pixels with nothing to fit, amber for limits, green for detections.
STATUS_COLORS = ['#e6e6e6', '#e8b021', '#2e9e5b']
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QInputDialog
)
from PyQt5.QtGui import QColor, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap


class GaussianMapWorker(QThread):
    """Background worker thread for Gaussian map fitting"""
    finished = pyqtSignal(dict)  # Emits results when done
    error = pyqtSignal(str)  # Emits error message if failed
    progress = pyqtSignal(int, int)  # (current, total) - emits progress updates
    
    def __init__(self, cube, wavelength, lam_min, lam_max, n_jobs=8, fac_above_stdres=3.0,
                 noise_min=None, noise_max=None, header=None, wave_unit=None, flux_unit=None, lambda_0=None):
        """
        Initialize Gaussian map fitting worker.
        
        Args:
            cube: 3D data cube (n_wavelength, n_y, n_x)
            wavelength: 1D wavelength array (numpy array or astropy Quantity)
                       If Quantity, it will be automatically handled by fit_gaussians_per_pixel
            lam_min, lam_max: Wavelength range bounds for fitting (scalars)
            n_jobs: Number of parallel jobs (default: 8)
            fac_above_stdres: Detection threshold (default: 3.0)
            noise_min, noise_max: Optional noise limits
            header: FITS header for unit/metadata information
            wave_unit: Wavelength unit string
            lambda_0: Rest wavelength for velocity calculations

        Returns:
            None
        """
        super().__init__()
        self.cube = cube
        self.wavelength = wavelength
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.n_jobs = n_jobs
        self.fac_above_stdres = fac_above_stdres
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.header = header
        self.wave_unit = wave_unit
        self.flux_unit = flux_unit
        self.lambda_0 = lambda_0
        self._cancel_event = threading.Event()

    def cancel(self):
        """Request that the run stop after the currently in-flight rows finish.
        Does not block; call wait() on this QThread to pause until it exits.

        Args:
            None

        Returns:
            None
        """
        self._cancel_event.set()

    def run(self):
        """Execute fitting in background thread.

        Args:
            None

        Returns:
            None
        """
        try:
            from .gaussian_fitter import fit_gaussians_per_pixel
            results = fit_gaussians_per_pixel(
                cube=self.cube,
                wavelength=self.wavelength,
                lam_min=self.lam_min,
                lam_max=self.lam_max,
                n_jobs=self.n_jobs,
                fac_above_stdres=self.fac_above_stdres,
                noise_min=self.noise_min,
                noise_max=self.noise_max,
                header=self.header,
                wave_unit=self.wave_unit,
                flux_unit=self.flux_unit,
                lambda_0=self.lambda_0,
                progress_callback=self._emit_progress,
                cancel_event=self._cancel_event,
            )
            # results is None when cancel() fired before completion; nothing
            # is listening by then, so just let the thread end.
            if results is not None:
                self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


    def _emit_progress(self, current, total):
        """Emit progress signal.

        Args:
            current (int): Pixels fit so far.
            total (int): Total pixels to fit.

        Returns:
            None
        """
        self.progress.emit(current, total)


class MultiGaussianFitWorker(QThread):
    """Background worker thread for Multi-Gaussian Fit (fit_hitran_lines_collective).

    Same shape as GaussianMapWorker above -- a single spectrum with a handful of chunks is
    much smaller than a full-cube map, but fit_glines' simultaneous curve_fit per chunk can
    still run for tens of seconds on a crowded line list, long enough that running it on the
    GUI thread would freeze the window with no feedback.
    """
    finished = pyqtSignal(list, list)  # (fit_pairs, contam_pairs)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # (current_chunk, total_chunks)

    def __init__(self, spectrum_x, spectrum_y, hitran_lines, continuum,
                 header=None, wave_unit=None, contam_lines=None):
        """Store the multi-Gaussian fit inputs for run() to use.

        Args:
            spectrum_x (numpy.ndarray): Wavelength array to fit against.
            spectrum_y (numpy.ndarray): Flux array to fit.
            hitran_lines (list): Detection line rows to fit.
            continuum (numpy.ndarray): Continuum to subtract before fitting.
            header (astropy.io.fits.Header): FITS header for unit handling.
            wave_unit (str): Wavelength unit string.
            contam_lines (pandas.DataFrame): Contaminating lines to exclude,
                or None.

        Returns:
            None
        """
        super().__init__()
        self.spectrum_x = spectrum_x
        self.spectrum_y = spectrum_y
        self.hitran_lines = hitran_lines
        self.continuum = continuum
        self.header = header
        self.wave_unit = wave_unit
        self.contam_lines = contam_lines

    def run(self):
        """Fit every detection line simultaneously and emit `finished` or `error`.

        Args:
            None

        Returns:
            None
        """
        try:
            from .flux_map_extractor import fit_hitran_lines_collective
            fit_pairs, contam_pairs, _ = fit_hitran_lines_collective(
                self.spectrum_x, self.spectrum_y, self.hitran_lines, self.continuum,
                header=self.header, wave_unit=self.wave_unit, contam_lines=self.contam_lines,
                progress_callback=self._emit_progress,
            )
            self.finished.emit(fit_pairs, contam_pairs)
        except Exception as e:
            self.error.emit(str(e))

    def _emit_progress(self, current, total):
        """Relay a fit-progress update through the `progress` signal.

        Args:
            current (int): Chunks fit so far.
            total (int): Total chunks to fit.

        Returns:
            None
        """
        self.progress.emit(current, total)


class GaussianMapResultsDialog(QDialog):
    """
    Dialog window for displaying Gaussian parameter maps.
    
    Shows 2D maps of fitted parameters (area, amplitude, center, etc.)
    with interactive parameter selection and visualization controls.
    """
    
    def __init__(self, parent, results, cube_data, gauss_left_x, gauss_right_x, wavelength=None,
                 noise_min=None, noise_max=None, gauss_peak_x=None):
        """Build the Gaussian map results dialog and its parameter maps.

        Args:
            parent (QWidget): Parent widget.
            results (dict): Per-pixel Gaussian map results to display.
            cube_data (CubeData): Loaded cube, for units/metadata.
            gauss_left_x (float): Left fit boundary used for the map.
            gauss_right_x (float): Right fit boundary used for the map.
            wavelength (numpy.ndarray): Wavelength array for the single-pixel view.
            noise_min (float): Noise region lower bound the fit used, or None.
            noise_max (float): Noise region upper bound the fit used, or None.
            gauss_peak_x (float): User-picked peak wavelength that seeded the fit.

        Returns:
            None
        """
        super().__init__(parent)
        self.parent_widget = parent
        self.results = results
        self.cube_data = cube_data
        self.gauss_left_x = gauss_left_x
        self.gauss_right_x = gauss_right_x
        self.gauss_peak_x = gauss_peak_x
        self.wavelength = wavelength
        # The noise region the fit used, so a single-pixel view can draw the same
        # detection threshold the map was decided with.
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.pixel_selection_enabled = True
        self._export_map_btn = None
        self._current_unit_label = None  # Store the actual displayed unit label
        
        # Extract units from data
        self.wave_unit = self._get_wavelength_unit()
        self.flux_unit = self._get_flux_unit()
        
        self.setWindowTitle("Gaussian Parameter Maps")
        self.resize(800, 700)
        
        self._setup_ui()
    
    def _get_wavelength_unit(self):
        """Extract wavelength unit from cube_data or results.

        Args:
            None

        Returns:
            unit (str): Wavelength unit string, defaulting to 'um'.
        """
        # Prioritize cube_data since it has the actual unit used for fitting
        if hasattr(self.cube_data, 'wave_unit') and self.cube_data.wave_unit:
            return self.cube_data.wave_unit
        # Fall back to results if available
        if 'wavelength_unit' in self.results and self.results['wavelength_unit'] and self.results['wavelength_unit'] != 'unknown':
            return self.results['wavelength_unit']
        return 'um'  # Default fallback
    
    def _get_flux_unit(self):
        """Extract flux unit from cube_data.

        Args:
            None

        Returns:
            unit (str): Flux unit string, defaulting to 'native'.
        """
        if hasattr(self.cube_data, 'flux_unit') and self.cube_data.flux_unit:
            return self.cube_data.flux_unit
        return 'native'  # Default fallback
    
    def _get_parameter_units(self):
        """Determine unit strings for all parameters based on instrument/data.

        Args:
            None

        Returns:
            units (dict): parameter -> (unit_label, colormap) pairs.
        """
        # Ask unit_converter what the maps are in, not pattern-matched BUNIT:
        # that previously mislabeled units (e.g. an uppercased micron symbol).
        header = getattr(self.cube_data, 'header', None)
        lam_mid = (min(self.gauss_left_x, self.gauss_right_x)
                   + max(self.gauss_left_x, self.gauss_right_x)) / 2.0
        labels = gaussian_map_unit_labels(
            self.flux_unit, self.wave_unit, header=header, wavelength=lam_mid)
        wave_display = labels['wave']
        area_unit = labels['area']

        return {
            'Area (AUC)': (f'Area [{area_unit}]', 'viridis'),
            'Area Error': (f'Area error [{area_unit}]', 'viridis'),
            # Amplitude stays in the cube's native flux unit -- not CGS-converted, unlike Area.
            'Amplitude': (f'Amplitude [{self.flux_unit}]', 'plasma'),
            'Center': (f'Center [{wave_display}]', 'twilight'),
            'Sigma': (f'Sigma [{wave_display}]', 'cividis'),
            'FWHM': (f'FWHM [{wave_display}]', 'cividis'),
            'chi2 (fit quality)': ('chi^2 [dimensionless]', 'RdYlGn_r'),
            # A ListedColormap rather than a name: three fixed colors for three states.
            # Picking a named colormap from the dropdown resamples it to three shades.
            'Fit Status': ('Outcome [0=no data, 1=upper limit, 2=detection]',
                           ListedColormap(STATUS_COLORS)),
            'Success Mask': ('Success [1=yes, 0=no]', 'gray'),
        }
    
    def _setup_ui(self):
        """Initialize dialog UI components.

        Args:
            None

        Returns:
            None
        """
        layout = QVBoxLayout()
        
        # Info label: just the headline count now. Everything else moved into
        # the Show Statistics dialog below (_show_statistics), per parameter.
        n_success = np.sum(self.results['success'])
        n_total = self.results['success'].size
        info_label = QLabel(
            f"Successfully fit {n_success:,} / {n_total:,} pixels ({100*n_success/n_total:.1f}%)"
        )
        info_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        layout.addWidget(info_label)
        
        # Parameter selector
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Parameter:"))
        self.param_combo = QComboBox()
        self.param_combo.addItems([
            "Area (AUC)", "Area Error", "Amplitude", "Center", "Sigma",
            "FWHM", "chi2 (fit quality)", "Fit Status", "Success Mask", "Residuals"
        ])
        param_layout.addWidget(self.param_combo)
        
        # Colormap selector
        param_layout.addWidget(QLabel("  Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            "Default", "viridis", "plasma", "inferno", "magma", "cividis",
            "coolwarm", "RdYlBu", "RdYlGn", "seismic", "twilight", "jet"
        ])
        param_layout.addWidget(self.cmap_combo)
        
        # Scale selector
        param_layout.addWidget(QLabel("  Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Linear", "Log", "Percentile (2-98%)"])
        param_layout.addWidget(self.scale_combo)

        param_layout.addStretch()
        layout.addLayout(param_layout)

        # Own row, below the selectors, so the window doesn't end up wider than
        # the map; same pixel-select control as the flux map.
        select_layout = QHBoxLayout()
        self.pixel_select_btn = QPushButton("✓ Pixel Select ON")
        self.pixel_select_btn.setCheckable(True)
        self.pixel_select_btn.setChecked(True)
        self.pixel_select_btn.setStyleSheet("background-color: #90EE90; font-weight: bold;")
        self.pixel_select_btn.clicked.connect(self._toggle_pixel_selection)
        select_layout.addWidget(self.pixel_select_btn)
        select_layout.addWidget(QLabel("Click a pixel to see its fit; turn OFF to zoom/pan"))
        select_layout.addStretch()
        layout.addLayout(select_layout)

        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Connect combo boxes to update function
        self.param_combo.currentIndexChanged.connect(self._update_plot)
        self.cmap_combo.currentIndexChanged.connect(self._update_plot)
        self.scale_combo.currentIndexChanged.connect(self._update_plot)
        
        # Initial plot
        self._update_plot()
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Export current parameter button
        export_btn = QPushButton("Save as FITS File")
        export_btn.clicked.connect(self._export_current_parameter)
        button_layout.addWidget(export_btn)

        # Detailed, copyable per-parameter statistics -- same pattern as the
        # flux map and rotational diagram map results dialogs.
        stats_btn = QPushButton("Show Statistics")
        stats_btn.clicked.connect(self._show_statistics)
        button_layout.addWidget(stats_btn)

        button_layout.addStretch()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _update_plot(self):
        """Update plot when parameter selection changes.

        Args:
            None

        Returns:
            None
        """
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        param_name = self.param_combo.currentText()
        
        # Remove export button if present
        if self._export_map_btn is not None:
            self.layout().removeWidget(self._export_map_btn)
            self._export_map_btn.deleteLater()
            self._export_map_btn = None
        
        # Format wavelength range for titles
        lam_min = min(self.gauss_left_x, self.gauss_right_x)
        lam_max = max(self.gauss_left_x, self.gauss_right_x)
        wave_range = f"({lam_min:.4f} - {lam_max:.4f})"
        
        # Get dynamic units for all parameters
        param_units = self._get_parameter_units()

        # Select data based on parameter
        if param_name == "Area (AUC)":
            data = self.results['area']
            title = f"Integrated Area Under Gaussian (AUC) {wave_range}"
            unit_label, default_cmap = param_units['Area (AUC)']
        elif param_name == "Area Error":
            data = self.results['area_error']
            title = f"Area Uncertainty (1-sigma) {wave_range}"
            unit_label, default_cmap = param_units['Area Error']
        elif param_name == "Amplitude":
            data = self.results['amplitude']
            title = f"Gaussian Peak Amplitude {wave_range}"
            unit_label, default_cmap = param_units['Amplitude']
        elif param_name == "Center":
            data = self.results['center']
            title = f"Gaussian Center Position {wave_range}"
            unit_label, default_cmap = param_units['Center']
        elif param_name == "Sigma":
            data = self.results['sigma']
            title = f"Gaussian Width (sigma) {wave_range}"
            unit_label, default_cmap = param_units['Sigma']
        elif param_name == "FWHM":
            data = self.results['fwhm']
            title = f"Full Width at Half Maximum {wave_range}"
            unit_label, default_cmap = param_units['FWHM']
        elif param_name == "chi2 (fit quality)":
            data = self.results['r_squared']
            title = f"Reduced chi^2 (approx 1 is good fit) {wave_range}"
            unit_label, default_cmap = param_units['chi2 (fit quality)']
        elif param_name == "Fit Status":
            # A Left/Right region fit NaNs out everything outside the selection; those
            # pixels were never evaluated, which is the same as having no data here.
            data = np.nan_to_num(fit_status_map(self.results), nan=float(STATUS_NO_DATA))
            title = f"Fit Status (0=no data, 1=upper limit, 2=detection) {wave_range}"
            unit_label, default_cmap = param_units['Fit Status']
        elif param_name == "Success Mask":
            data = self.results['success'].astype(float)
            title = f"Successful Fits (1=success, 0=failed) {wave_range}"
            unit_label, default_cmap = param_units['Success Mask']
        elif param_name == "Residuals":
            self._handle_residuals_display(ax)
            return
        
        # Get selected colormap (use default if "Default" selected)
        cmap_selection = self.cmap_combo.currentText()
        cmap = default_cmap if cmap_selection == "Default" else cmap_selection
        
        # Store the unit label that's being displayed (for export later)
        self._current_unit_label = unit_label
        
        # Apply scaling
        scale_selection = self.scale_combo.currentText()
        from matplotlib.colors import LogNorm, Normalize

        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            if scale_selection == "Log":
                # Log scale: requires positive values
                if np.min(valid_data) > 0:
                    norm = LogNorm(vmin=np.nanmin(valid_data), vmax=np.nanmax(valid_data))
                else:
                    # Shift data to make it positive
                    offset = abs(np.min(valid_data)) + 1e-10
                    data_shifted = data + offset
                    norm = LogNorm(vmin=np.nanmin(data_shifted), vmax=np.nanmax(data_shifted))
                    data = data_shifted
            elif scale_selection == "Percentile (2-98%)":
                # Use 2-98 percentile range to enhance contrast
                vmin = np.nanpercentile(valid_data, 2)
                vmax = np.nanpercentile(valid_data, 98)
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                # Linear (default)
                norm = None
        else:
            norm = None
        
        # Plot with NaN handling
        im = ax.imshow(data, origin='lower', cmap=cmap, interpolation='nearest', norm=norm)
        ax.set_title(title)
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")
        
        # Colorbar with units
        self.fig.colorbar(im, ax=ax, label=unit_label)

        self.fig.tight_layout()
        self.canvas.draw()

    def _resolve_param_array(self, param_name):
        """
        Get the 2D array for a Parameter dropdown selection, straight from self.results
        -- not whatever _update_plot last drew, which Log scale can shift positive.
        Keep this in sync with _update_plot's own data selection above.

        Args:
            param_name (str): One of the Parameter dropdown's exact option strings.

        Returns:
            data (numpy.ndarray or None): The array for this selection, or the
                mid-wavelength slice of residuals_cube for "Residuals". None if
                residuals were never computed.
        """
        if param_name == "Area (AUC)":
            return self.results['area']
        if param_name == "Area Error":
            return self.results['area_error']
        if param_name == "Amplitude":
            return self.results['amplitude']
        if param_name == "Center":
            return self.results['center']
        if param_name == "Sigma":
            return self.results['sigma']
        if param_name == "FWHM":
            return self.results['fwhm']
        if param_name == "chi2 (fit quality)":
            return self.results['r_squared']
        if param_name == "Fit Status":
            return np.nan_to_num(fit_status_map(self.results), nan=float(STATUS_NO_DATA))
        if param_name == "Success Mask":
            return self.results['success'].astype(float)
        if param_name == "Residuals":
            cube = self.results.get('residuals_cube')
            return cube[cube.shape[0] // 2] if cube is not None else None
        return None

    def _show_statistics(self):
        """
        Show the currently selected parameter's own values, with units, plus the
        fit-wide pixel breakdown, in one copyable dialog -- same pattern as the
        flux map and rotational diagram map results windows.

        Args:
            None

        Returns:
            None
        """
        param_name = self.param_combo.currentText()
        data = self._resolve_param_array(param_name)

        n_success = np.sum(self.results['success'])
        n_detections = np.sum(self.results.get('is_detection', np.zeros_like(self.results['success'])))
        n_upper_limits = np.sum(self.results.get('is_upper_limit', np.zeros_like(self.results['success'])))
        n_failed = np.sum(self.results.get('fit_failed', np.zeros_like(self.results['success'])))
        n_total = self.results['success'].size
        upper_text = f"Upper limits: {n_upper_limits:,}"
        if n_failed:
            upper_text += f" ({n_failed:,} from fits that never ran)"

        lines = [
            f"{param_name.upper()} STATISTICS", "=" * 50, "",
            "PIXEL TYPES:",
            f"  Successfully fit: {n_success:,} / {n_total:,} ({100*n_success/n_total:.1f}%)",
            f"  Detections:       {n_detections:,}",
            f"  {upper_text}",
            f"  No data:          {n_total - n_detections - n_upper_limits:,}",
            "",
        ]

        # Same labels the colorbar uses; pull out the bracketed unit. Residuals
        # has no entry (it is not a plotted parameter), so state its units directly.
        if param_name == "Residuals":
            units = "normalized: (data - fit) / noise, dimensionless"
        else:
            unit_label, _ = self._get_parameter_units()[param_name]
            units = unit_label[unit_label.index('[') + 1: unit_label.rindex(']')]
        lines.append(f"{param_name.upper()} VALUES:")
        lines.append(f"  Units:   {units}")
        if data is None:
            lines.append("  (not available for this fit)")
        else:
            valid = data[np.isfinite(data)]
            if len(valid) == 0:
                lines.append("  No valid (non-NaN) pixels in this map.")
            else:
                lines += [
                    f"  Count:   {len(valid):,} / {data.size:,} valid ({100*len(valid)/data.size:.1f}%)",
                    f"  Min/Max: {np.min(valid):.4e} / {np.max(valid):.4e}",
                    f"  Mean:    {np.mean(valid):.4e}",
                    f"  Median:  {np.median(valid):.4e}",
                    f"  Std:     {np.std(valid):.4e}",
                ]
                if param_name == "Area (AUC)":
                    # Area is the only field upper limits also populate, so it
                    # is the only map where the two sums can differ.
                    det_mask = self.results.get('is_detection')
                    if det_mask is not None and det_mask.shape == data.shape:
                        det_sum = np.nansum(np.where(det_mask, data, np.nan))
                        lines.append(f"  Sum (detections only):           {det_sum:.4e}")
                    lines.append(f"  Sum (detections + upper limits): {np.nansum(data):.4e}")
                elif param_name != "Residuals":
                    # Summing normalized residuals isn't meaningful (they cancel near zero).
                    lines.append(f"  Sum:     {np.sum(valid):.4e}")

        dlg = QDialog(self)
        dlg.setWindowTitle("Statistics")
        dlg.resize(560, 480)
        layout = QVBoxLayout()
        body = QLabel("\n".join(lines))
        body.setStyleSheet("font-family: monospace; white-space: pre;")
        body.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        layout.addWidget(body)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.setLayout(layout)
        dlg.exec_()

    def _toggle_pixel_selection(self):
        """Toggle click-a-pixel mode so the matplotlib toolbar can be used instead.

        Args:
            None

        Returns:
            None
        """
        self.pixel_selection_enabled = self.pixel_select_btn.isChecked()
        if self.pixel_selection_enabled:
            self.pixel_select_btn.setText("✓ Pixel Select ON")
            self.pixel_select_btn.setStyleSheet("background-color: #90EE90; font-weight: bold;")
        else:
            self.pixel_select_btn.setText("✗ Pixel Select OFF")
            self.pixel_select_btn.setStyleSheet("background-color: #FFB6C6; font-weight: bold;")

    def _on_canvas_click(self, event):
        """Open the single-pixel fit view for the clicked map pixel.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Click event.

        Returns:
            None
        """
        # A modal dialog opened mid-press would swallow the matching
        # release, leaving the toolbar stuck mid-pan -- skip while active.
        if self.toolbar.mode or not self.pixel_selection_enabled:
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        n_y, n_x = self.results['success'].shape
        col = int(np.round(event.xdata))
        row = int(np.round(event.ydata))
        if not (0 <= col < n_x and 0 <= row < n_y):
            return

        self._show_pixel_fit(row, col)

    def _show_pixel_fit(self, row, col):
        """Show the stored fit for one pixel (row = y index, col = x index).

        Args:
            row (int): Pixel row (y) index.
            col (int): Pixel column (x) index.

        Returns:
            None
        """
        if self.wavelength is None:
            QMessageBox.warning(self, "No Wavelength Data",
                                "Wavelength array not available for pixel inspection")
            return
        if self.cube_data is None or getattr(self.cube_data, 'cube', None) is None:
            QMessageBox.warning(self, "No Cube", "The cube is no longer available")
            return

        from .pixel_diagnostics import PixelFitDialog

        lam_min = min(self.gauss_left_x, self.gauss_right_x)
        lam_max = max(self.gauss_left_x, self.gauss_right_x)
        labels = gaussian_map_unit_labels(
            self.flux_unit, self.wave_unit,
            header=getattr(self.cube_data, 'header', None),
            wavelength=(lam_min + lam_max) / 2.0)

        try:
            dlg = PixelFitDialog(
                self,
                self.cube_data.cube,
                self.wavelength,
                lam_min,
                lam_max,
                self.results,
                row=row,
                col=col,
                fac_above_stdres=self.results.get('fac_above_stdres', 3.0),
                noise_min=self.noise_min,
                noise_max=self.noise_max,
                flux_unit=self.flux_unit,
                wave_unit=labels['wave'],
                area_unit=labels['area'],
            )
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Could not inspect pixel ({col}, {row}):\n{e}")
            return
        dlg.exec_()

    def _handle_residuals_display(self, ax):
        """Handle residuals display with export button.

        Args:
            ax (matplotlib.axes.Axes): Axes to draw the residuals map on.

        Returns:
            None
        """
        if 'residuals_cube' in self.results:
            residual_cube = self.results['residuals_cube']
            # Show middle wavelength slice
            mid_idx = residual_cube.shape[0] // 2
            data = residual_cube[mid_idx]
            
            # Format wavelength range
            lam_min = min(self.gauss_left_x, self.gauss_right_x)
            lam_max = max(self.gauss_left_x, self.gauss_right_x)
            wave_range = f"({lam_min:.4f} - {lam_max:.4f})"
            title = f"Normalized Residuals (Middle Wavelength) {wave_range}"
            default_cmap = 'seismic'
            
            # Get selected colormap (use default if "Default" selected)
            cmap_selection = self.cmap_combo.currentText()
            cmap = default_cmap if cmap_selection == "Default" else cmap_selection
            
            # Apply scaling
            scale_selection = self.scale_combo.currentText()
            from matplotlib.colors import LogNorm, Normalize
            
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                if scale_selection == "Log":
                    # Log scale: requires positive values
                    if np.min(valid_data) > 0:
                        norm = LogNorm(vmin=np.nanmin(valid_data), vmax=np.nanmax(valid_data))
                    else:
                        # Shift data to make it positive for residuals (which can be negative)
                        offset = abs(np.min(valid_data)) + 1e-10
                        data_shifted = data + offset
                        norm = LogNorm(vmin=np.nanmin(data_shifted), vmax=np.nanmax(data_shifted))
                        data = data_shifted
                elif scale_selection == "Percentile (2-98%)":
                    # Use 2-98 percentile range to enhance contrast
                    vmin = np.nanpercentile(valid_data, 2)
                    vmax = np.nanpercentile(valid_data, 98)
                    norm = Normalize(vmin=vmin, vmax=vmax)
                else:
                    # Linear (default)
                    norm = None
            else:
                norm = None
            
            # Plot
            im = ax.imshow(data, origin='lower', cmap=cmap, interpolation='nearest', norm=norm)
            ax.set_title(title)
            ax.set_xlabel("X Pixel")
            ax.set_ylabel("Y Pixel")
            self.fig.colorbar(im, ax=ax, label="Residuals")
        else:
            # Fallback if residuals not in results
            data = np.zeros((self.results['success'].shape[0], self.results['success'].shape[1]))
            title = "Residuals (not computed)"
            cmap = 'gray'
            ax.imshow(data, origin='lower', cmap=cmap, interpolation='nearest')
            ax.set_title(title)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _export_residuals(self):
        """Export residuals cube to FITS file.

        Args:
            None

        Returns:
            None
        """
        # Generate filename from original cube filename
        if hasattr(self.cube_data, 'filename') and self.cube_data.filename:
            base_name = os.path.splitext(os.path.basename(self.cube_data.filename))[0]
            default_filename = f"{base_name}_residual_cube.fits"
        else:
            default_filename = "residual_cube.fits"
        
        out_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Residuals FITS", 
            default_filename, 
            "FITS files (*.fits)"
        )
        if out_path:
            try:
                from .gaussian_fitter import export_residual_cube_fits
                residual_cube = self.results['residuals_cube']
                export_residual_cube_fits(
                    self.cube_data.cube, 
                    self.cube_data.wavelength,
                    min(self.gauss_left_x, self.gauss_right_x),
                    max(self.gauss_left_x, self.gauss_right_x),
                    out_path=out_path, 
                    header=self.cube_data.header,
                    precomputed_cube=residual_cube,
                    instrument_hint=self.cube_data.instrument
                )
                QMessageBox.information(
                    self, 
                    "Export Successful", 
                    f"Residuals cube saved to:\n{out_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Export Error", 
                    f"Failed to export residuals:\n{str(e)}"
                )
    
    def _export_current_parameter(self):
        """Export the currently displayed parameter as a 3D FITS map.

        Args:
            None

        Returns:
            None
        """
        param_name = self.param_combo.currentText()
        
        # Generate base filename from original cube filename
        if hasattr(self.cube_data, 'filename') and self.cube_data.filename:
            base_name = os.path.splitext(os.path.basename(self.cube_data.filename))[0]
        else:
            base_name = "gaussian"
        
        # Residuals already has its own correct export path -- reuse it
        # instead of duplicating it here (the old copy had NAXIS wrong).
        if param_name == "Residuals":
            if 'residuals_cube' not in self.results or self.results['residuals_cube'] is None:
                QMessageBox.warning(self, "No Data", "Residuals not available for export")
                return
            self._export_residuals()
            return
        
        # Map parameter names to data and default filenames (2D maps)
        param_map = {
            "Area (AUC)": (self.results['area'], f"{base_name}_gaussian_area_map.fits"),
            "Area Error": (self.results['area_error'], f"{base_name}_gaussian_area_error_map.fits"),
            "Amplitude": (self.results['amplitude'], f"{base_name}_gaussian_amplitude_map.fits"),
            "Center": (self.results['center'], f"{base_name}_gaussian_center_map.fits"),
            "Sigma": (self.results['sigma'], f"{base_name}_gaussian_sigma_map.fits"),
            "FWHM": (self.results['fwhm'], f"{base_name}_gaussian_fwhm_map.fits"),
            "chi2 (fit quality)": (self.results['r_squared'], f"{base_name}_gaussian_chi2_map.fits"),
            "Fit Status": (fit_status_map(self.results), f"{base_name}_gaussian_fit_status.fits"),
            "Success Mask": (self.results['success'].astype(float), f"{base_name}_gaussian_success_mask.fits"),
        }
        
        if param_name not in param_map:
            QMessageBox.warning(self, "Invalid Parameter", f"Cannot export {param_name}")
            return
        
        data, default_filename = param_map[param_name]
        
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {param_name} Map",
            default_filename,
            "FITS files (*.fits)"
        )
        
        if not out_path:
            return

        # Only Area (AUC) is a flux map (what Import Map wants); blank skips
        # the lookup, so plain feature-fitting is unchanged.
        species_text = ""
        if param_name == "Area (AUC)":
            species_text, _ok = QInputDialog.getText(
                self, "Species (optional)",
                "Species for HITRAN lookup, e.g. H2 (blank to skip):")
            species_text = species_text.strip().upper()

        try:
            # Add metadata about the Gaussian fit
            lam_min = min(self.gauss_left_x, self.gauss_right_x)
            lam_max = max(self.gauss_left_x, self.gauss_right_x)

            # Use the EXACT unit label that's displayed on the colorbar
            # This ensures consistency between visualization and FITS export
            if self._current_unit_label:
                # Extract unit from "Parameter [unit]" format
                if '[' in self._current_unit_label and ']' in self._current_unit_label:
                    bunit = self._current_unit_label[self._current_unit_label.index('[')+1:self._current_unit_label.index(']')]
                else:
                    bunit = self._current_unit_label
                # The on-screen label can carry Unicode (superscripts, mu);
                # a raw COMMENT line isn't run through build_2d_export_header.
                bunit = _ascii_safe(bunit)
                comment = f'{param_name}: {bunit}'
            else:
                # Fallback if no unit label stored
                bunit = param_name
                comment = f'Gaussian fit parameter: {param_name}'

            source_header = self.cube_data.header
            object_label = (source_header.get('OBJECT', 'Unknown') if source_header is not None else 'Unknown') \
                + f' ({param_name})'
            # The picked peak, not the window midpoint: picking one is required
            # before any fit runs, and the window need not sit symmetric around it.
            line_wave = self.gauss_peak_x if self.gauss_peak_x is not None else (lam_min + lam_max) / 2.0
            extra = {
                'BUNIT': bunit,
                'OBJECT': object_label,
                'PARAM': param_name,
                'LAMMIN': (lam_min, "Minimum wavelength of fit"),
                'LAMMAX': (lam_max, "Maximum wavelength of fit"),
                'NSUCC': (int(np.sum(self.results['success'])), "Number of successful fits"),
                'NTOTAL': (int(self.results['success'].size), "Total number of pixels"),
            }
            is_area = param_name == "Area (AUC)"
            if is_area and species_text:
                try:
                    hitran_lines, _meta = fetch_hitran_data(
                        line_wave - 0.01, line_wave + 0.01,
                        molecule=species_text, isotopologue=1)
                except Exception:
                    hitran_lines = []
                best_match = match_nearest_hitran_line(hitran_lines, line_wave)
                if best_match:
                    extra['A_UL'] = (best_match.get('a_coeff', 0.0), "Einstein A coefficient (1/s)")
                    extra['E_UPPER'] = (best_match.get('e_upper', 0.0), "Upper state energy (K)")
                    extra['G_U'] = (best_match.get('Gp', 1.0), "Upper state statistical weight")
                    extra['TRANSITN'] = best_match.get('transition', 'User Data')
                    # H2's own fit-index/OPR logic needs these, not just
                    # the physical constants above (see _line_keys).
                    if best_match.get('j_lower') is not None:
                        extra['JLOWER'] = int(best_match['j_lower'])
                    if best_match.get('branch'):
                        extra['BRANCH'] = best_match['branch']

            # The global (stacked-spectrum) fit is a real point-fit result --
            # same shape as Calculate Fluxes', so Import Map can show it.
            global_fit = self.results.get('global_fit') if is_area else None
            if global_fit:
                if global_fit.get('area') is not None:
                    extra['GFITAREA'] = (float(global_fit['area']), "Global fit area (flux)")
                if global_fit.get('area_error') is not None:
                    extra['GFITAERR'] = (float(global_fit['area_error']), "Global fit area error")
                if global_fit.get('r_squared') is not None:
                    extra['GFITR2'] = (float(global_fit['r_squared']), "Global fit R-squared")
                # Scalar fit params too, not just the curve -- the Fit viewer's
                # own-line-only option recomputes the Gaussian from these.
                if global_fit.get('amplitude') is not None:
                    extra['GFITAMP'] = (float(global_fit['amplitude']), "Global fit amplitude")
                if global_fit.get('center') is not None:
                    extra['GFITCTR'] = (float(global_fit['center']), "Global fit center (um)")
                if global_fit.get('sigma') is not None:
                    extra['GFITSIG'] = (float(global_fit['sigma']), "Global fit sigma (um)")

            err_map = self.results.get('area_error') if is_area else None
            out_header, extensions = build_line_map_header(
                source_header, data.shape, wavelength=line_wave,
                molecule=species_text if (is_area and species_text) else None,
                flux_map=data if is_area else None, error_map=err_map,
                src_filename=(getattr(self.cube_data, 'filename', None) or 'UNKNOWN') if is_area else None,
                fit_method='Single-Gaussian fit' if is_area else None, extra=extra,
            )
            if global_fit and global_fit.get('x_data') is not None:
                # None header is fine -- write_fits_product fills in a blank
                # one; these are small 1D arrays, not WCS-bearing images.
                extensions = list(extensions) + [
                    ("SPECX", np.asarray(global_fit['x_data'], dtype=np.float32), None),
                    ("SPECY", np.asarray(global_fit['y_data'], dtype=np.float32), None),
                    ("SPECBASE", np.asarray(global_fit['baseline'], dtype=np.float32), None),
                ]
            # Get instrument info -- normalized to uppercase, falling back to
            # cube_data.instrument, same as this dialog has always done.
            instr = out_header.get('INSTRUME', '').strip().upper() or self.cube_data.instrument or 'UNKNOWN'
            teles = out_header.get('TELESCOP', '').strip().upper() or 'UNKNOWN'
            if instr != 'UNKNOWN':
                out_header['INSTRUME'] = instr
            if teles != 'UNKNOWN':
                out_header['TELESCOP'] = teles
            out_header['COMMENT'] = comment
            out_header['COMMENT'] = f'Gaussian fit parameter: {param_name}'

            write_fits_product(out_path, data, out_header, extensions=extensions or None)

            print(f"Parameter map FITS written to {out_path}")
            QMessageBox.information(
                self,
                "Export Successful",
                f"{param_name} map saved to:\n{out_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export {param_name}:\n{str(e)}"
            )


class HITRANGaussianResultsDialog(QDialog):
    """
    Dialog for displaying multi-Gaussian fit results for HITRAN lines.
    
    Shows a searchable, sortable table with results for each fitted line:
    - Wavelength, Species, Transition
    - Amplitude, Sigma, FWHM
    - Area (native and/or CGS units)
    - R^2 and fit status
    """
    
    def __init__(self, parent, results, wave_unit='um', flux_unit=''):
        """Build the HITRAN multi-Gaussian fit results table dialog.

        Args:
            parent (QWidget): Parent widget.
            results (list): Per-line fit result dicts to display.
            wave_unit (str): Wavelength unit label for the table.
            flux_unit (str): Flux unit label for the table.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle("HITRAN Gaussian Fit Results")
        self.setMinimumSize(1000, 600)
        
        self.results = results
        self.wave_unit = wave_unit
        self.flux_unit = flux_unit
        
        layout = QVBoxLayout()
        
        # Summary statistics
        summary_layout = QHBoxLayout()
        
        n_total = len(results)
        n_success = sum(1 for r in results if r.get('success', False))
        
        summary_text = QLabel(
            f"<b>Summary:</b> {n_success} of {n_total} lines successfully fitted"
        )
        summary_layout.addWidget(summary_text)
        summary_layout.addStretch()
        
        layout.addLayout(summary_layout)
        
        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels([
            "lambda (" + wave_unit + ")",
            "Species",
            "Transition",
            "Amplitude",
            "sigma (" + wave_unit + ")",
            "FWHM (" + wave_unit + ")",
            "Area (native)",
            "Area (CGS)",
            "R^2",
            "Status",
            "Message"
        ])
        
        self.table.setRowCount(len(results))
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        
        # Populate table
        for row_idx, result in enumerate(results):
            items = [
                QTableWidgetItem(f"{result.get('wavelength', 0):.6f}"),
                QTableWidgetItem(result.get('species', '')),
                QTableWidgetItem(result.get('transition', '')),
                QTableWidgetItem(f"{result.get('amplitude', 0):.4e}"),
                QTableWidgetItem(f"{result.get('sigma', 0):.6f}"),
                QTableWidgetItem(f"{result.get('fwhm', 0):.6f}"),
                QTableWidgetItem(f"{result.get('area_native', 0):.4e}"),
                QTableWidgetItem(f"{result.get('area', 0):.4e}"),
                QTableWidgetItem(f"{result.get('r_squared', 0):.4f}"),
                QTableWidgetItem("Y" if result.get('success', False) else "N"),
                QTableWidgetItem(result.get('message', ''))
            ]
            
            for col_idx, item in enumerate(items):
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                
                # Color code based on success
                if col_idx == 9:  # Status column
                    if result.get('success', False):
                        item.setForeground(QColor('green'))
                        font = QFont()
                        font.setBold(True)
                        item.setFont(font)
                    else:
                        item.setForeground(QColor('red'))
                
                self.table.setItem(row_idx, col_idx, item)
        
        # Auto-resize columns
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)
        
        # Export button
        button_layout = QHBoxLayout()
        
        export_csv_btn = QPushButton("Export to CSV")
        export_csv_btn.clicked.connect(self._export_to_csv)
        button_layout.addWidget(export_csv_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _export_to_csv(self):
        """Export results to CSV file.

        Args:
            None

        Returns:
            None
        """
        from PyQt5.QtWidgets import QFileDialog
        import csv
        
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export HITRAN Results to CSV",
            "hitran_fit_results.csv",
            "CSV Files (*.csv)"
        )
        
        if not out_path:
            return
        
        try:
            with open(out_path, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'wavelength', 'species', 'transition',
                        'amplitude', 'sigma', 'fwhm',
                        'area_native', 'area', 'r_squared',
                        'success', 'message'
                    ]
                )
                writer.writeheader()
                writer.writerows(self.results)
            
            QMessageBox.information(
                self,
                "Export Successful",
                f"Results exported to:\n{out_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export results:\n{str(e)}"
            )

