"""
Flux Map Dialog and Worker Classes

Contains QThread workers and dialog classes for flux map extraction.
Simplified: no parameter dialog, automatic processing of entire spectrum.
"""

import numpy as np
import os
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QMessageBox, QComboBox, QTabWidget, QWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.cm as cm
from astropy import units as u


class FluxMapWorker(QThread):
    """Background worker for flux extraction"""
    progress = pyqtSignal(int, int)  # (current, total)
    finished = pyqtSignal(dict)       # Results dict
    error = pyqtSignal(str)           # Error message
    
    def __init__(self, cube_data, target_catalog, contam_catalog, n_jobs=4, wave_min=None, wave_max=None, excluded_wavelengths=None, continuum_params=None):
        """Store flux-map fit inputs for run() to use.

        Args:
            cube_data (CubeData): Loaded cube to extract flux from.
            target_catalog (pandas.DataFrame): Target line catalog.
            contam_catalog (pandas.DataFrame): Contaminating lines, or None.
            n_jobs (int): Number of parallel worker threads.
            wave_min (float): Minimum wavelength to fit, or None.
            wave_max (float): Maximum wavelength to fit, or None.
            excluded_wavelengths (set): Wavelengths excluded from the fit.
            continuum_params (dict): Continuum-fitting parameter overrides.

        Returns:
            None
        """
        super().__init__()
        self.cube_data = cube_data
        self.target_catalog = target_catalog
        self.contam_catalog = contam_catalog
        self.n_jobs = n_jobs
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.excluded_wavelengths = excluded_wavelengths
        self.continuum_params = continuum_params
        self.cancelled = False  # Flag to stop processing
    
    def run(self):
        """Execute extraction.

        Args:
            None

        Returns:
            None
        """
        try:
            if self.cancelled:
                self.error.emit("Extraction cancelled by user")
                return
            
            from .flux_map_extractor import create_flux_map
            
            results = create_flux_map(
                cube_data=self.cube_data,
                target_catalog=self.target_catalog,
                contam_catalog=self.contam_catalog,
                n_jobs=self.n_jobs,
                progress_callback=self._emit_progress,
                cancel_flag=self,  # Pass self so extraction can check cancelled flag
                wave_min=self.wave_min,
                wave_max=self.wave_max,
                excluded_wavelengths=self.excluded_wavelengths,
                continuum_params=self.continuum_params
            )
            
            if self.cancelled:
                self.error.emit("Extraction cancelled by user")
            else:
                self.finished.emit(results)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
    
    def _emit_progress(self, current, total):
        """Relay a fit-progress update through the `progress` signal.

        Args:
            current (int): Pixels fit so far.
            total (int): Total pixels to fit.

        Returns:
            None
        """
        self.progress.emit(current, total)

    def request_cancel(self):
        """Request cancellation.

        Args:
            None

        Returns:
            None
        """
        self.cancelled = True


class FluxMapResultsDialog(QDialog):
    """Display flux extraction results with interactive viewer"""

    @staticmethod
    def _species_label(catalog):
        """Species label for map titles/stats -- e.g. 'H2', 'CO'. Assumes a
        single target species per run, so this takes the first non-empty
        'species' value; 'Target' is the fallback when none can be determined.

        Args:
            catalog (pd.DataFrame or None): The target catalog, expected to have a
                'species' column (same DataFrame passed to create_flux_map())

        Returns:
            label (str): e.g. 'H2', 'CO', or 'Target'
        """
        if catalog is None or len(catalog) == 0 or 'species' not in catalog.columns:
            return 'Target'
        for s in catalog['species'].dropna():
            s = str(s).strip()
            if s:
                return s
        return 'Target'

    def __init__(self, flux_map, detection_map, header=None, wave_range=None,
                 wavelength=None, cube_data=None, target_catalog=None, contam_catalog=None,
                 parent=None, excluded_wavelengths=None, flux_unit=None,
                 continuum_params=None, contam_flux_map=None, contam_detection_map=None):
        """Build the flux map results dialog and its interactive viewer.

        Args:
            flux_map (numpy.ndarray): 2D map of integrated target-line flux.
            detection_map (numpy.ndarray): 2D map of detection counts.
            header (astropy.io.fits.Header): FITS header, for units/metadata.
            wave_range (tuple): (wavelength_min, wavelength_max) fit range.
            wavelength (numpy.ndarray): Wavelength array for pixel inspection.
            cube_data (CubeData): Loaded cube the map was fit from.
            target_catalog (pandas.DataFrame): Target line catalog used.
            contam_catalog (pandas.DataFrame): Contaminating lines used, or None.
            parent (QWidget): Parent GUI window.
            excluded_wavelengths (set): Wavelengths excluded from the fit.
            flux_unit (str): Output flux unit, or None to derive from header.
            continuum_params (dict): Continuum-fitting parameters used.
            contam_flux_map (numpy.ndarray): 2D contaminant flux map, or None.
            contam_detection_map (numpy.ndarray): 2D contaminant detection
                map, or None.

        Returns:
            None
        """
        super().__init__(parent)
        self.flux_map = flux_map
        self.detection_map = detection_map
        # None for results cached before these fields existed -- every use below guards on that.
        self.contam_flux_map = contam_flux_map
        self.contam_detection_map = contam_detection_map
        self.header = header
        self.wave_range = wave_range
        self.wavelength = wavelength
        self.cube_data = cube_data
        self.target_catalog = target_catalog
        self.contam_catalog = contam_catalog
        self.species_label = self._species_label(target_catalog)
        self.continuum_params = continuum_params
        self.selected_pixel = None  # For pixel inspection mode
        self.pixel_selection_enabled = True  # Toggle for pixel clicking
        self.excluded_wavelengths = set(excluded_wavelengths) if excluded_wavelengths else set()  # Track excluded lines
        self.parent_gui = parent  # Reference to parent GUI for syncing species dialog
        self.flux_unit_output = flux_unit or 'MJy/sr um'  # Track output units from conversion
        
        # Extract units from FITS header using astropy units
        try:
            # If flux_unit_output was provided (from conversion), use that directly
            if self.flux_unit_output and self.flux_unit_output != 'MJy/sr um':
                if self.flux_unit_output == 'erg/s/cm2':
                    self.flux_unit_str = "erg/s/cm2"
                    self.bunit_str = "erg/s/cm2"
                    self.flux_unit_base = u.erg / (u.s * u.cm**2)
                    self.flux_unit = self.flux_unit_base
                else:
                    self.flux_unit_str = self.flux_unit_output
                    self.bunit_str = self.flux_unit_output
                    self.flux_unit_base = self.flux_unit_output
                    self.flux_unit = self.flux_unit_output
            else:
                # Fall back to header-based units
                bunit_str = header.get('BUNIT', 'MJy/sr') if header else 'MJy/sr'
                # Try to parse as astropy unit, fallback to string if it fails
                try:
                    self.flux_unit_base = u.Unit(bunit_str)
                except (ValueError, TypeError):
                    # If it's not a valid astropy unit, use it as-is for display
                    self.flux_unit_base = bunit_str
                
                # Flux unit is base flux unit per wavelength
                if isinstance(self.flux_unit_base, u.UnitBase):
                    self.flux_unit = self.flux_unit_base * u.um
                    self.flux_unit_str = f"{self.flux_unit_base} um"
                    self.bunit_str = f"{self.flux_unit_base}"
                else:
                    self.flux_unit_str = f"{self.flux_unit_base} um"
                    self.bunit_str = self.flux_unit_base
        except Exception:
            # Fallback if header parsing fails
            self.flux_unit_str = self.flux_unit_output if self.flux_unit_output else "MJy/sr um"
            self.bunit_str = self.flux_unit_output if self.flux_unit_output else "MJy/sr"
            if self.flux_unit_output == 'erg/s/cm2':
                self.flux_unit_base = u.erg / (u.s * u.cm**2)
                self.flux_unit = self.flux_unit_base
            else:
                self.flux_unit_base = u.MJy / u.sr
                self.flux_unit = self.flux_unit_base * u.um
        
        self.setWindowTitle(f"{self.species_label} Flux Map Results")
        self.resize(1200, 700)
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the results dialog's tabs, plots, and controls.

        Args:
            None

        Returns:
            None
        """
        from PyQt5.QtWidgets import QTabWidget
        layout = QVBoxLayout()
        
        # Info section
        n_valid = np.sum(~np.isnan(self.flux_map))
        n_total = self.flux_map.size
        valid_fluxes = self.flux_map[~np.isnan(self.flux_map)]
        mean_flux = np.nanmean(self.flux_map) if n_valid > 0 else 0
        sum_flux = np.nansum(self.flux_map) if n_valid > 0 else 0
        max_flux = np.nanmax(self.flux_map) if n_valid > 0 else 0
        detect_with_lines = self.detection_map[self.detection_map > 0]
        mean_detections = np.mean(detect_with_lines) if len(detect_with_lines) > 0 else 0
        
        info_text = (
            f"Valid pixels: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%) | "
            f"Total {self.species_label} flux: {sum_flux:.4e} | Mean: {mean_flux:.4e} | Max: {max_flux:.4e} | "
            f"Mean detections: {mean_detections:.1f} lines/pixel"
        )
        if self.contam_flux_map is not None:
            sum_contam = np.nansum(self.contam_flux_map)
            info_text += f" | Total contam flux: {sum_contam:.4e}"
        if self.contam_detection_map is not None:
            n_contam_pixels = np.sum(self.contam_detection_map > 0)
            info_text += f" | Contam detections: {n_contam_pixels:,} px"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("background-color: #f0f0f0; padding: 6px; border-radius: 3px;")
        info_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        layout.addWidget(info_label)
        
        # Create tabs for different views
        tabs = QTabWidget()
        
        # Tab 1: Flux Map
        map_widget = QWidget()
        map_layout = QVBoxLayout()
        
        # Controls for map
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Display:"))
        self.param_combo = QComboBox()
        param_items = [f"{self.species_label} Flux", "Detection Count"]
        if self.contam_flux_map is not None:
            param_items.append("Contam Flux")
        if self.contam_detection_map is not None:
            param_items.append("Contam Detection Count")
        self.param_combo.addItems(param_items)
        self.param_combo.currentIndexChanged.connect(self._update_plot)
        ctrl_layout.addWidget(self.param_combo)
        
        ctrl_layout.addWidget(QLabel("  Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["viridis", "plasma", "inferno", "magma", "coolwarm", "RdYlGn"])
        self.cmap_combo.currentIndexChanged.connect(self._update_plot)
        ctrl_layout.addWidget(self.cmap_combo)
        
        ctrl_layout.addWidget(QLabel("  Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Linear", "Log", "Percentile"])
        self.scale_combo.currentIndexChanged.connect(self._update_plot)
        ctrl_layout.addWidget(self.scale_combo)
        
        # Toggle button for pixel selection mode
        self.pixel_select_btn = QPushButton("✓ Pixel Select ON")
        self.pixel_select_btn.setCheckable(True)
        self.pixel_select_btn.setChecked(True)
        self.pixel_select_btn.setStyleSheet("background-color: #90EE90; font-weight: bold;")
        self.pixel_select_btn.clicked.connect(self._toggle_pixel_selection)
        ctrl_layout.addWidget(self.pixel_select_btn)
        ctrl_layout.addWidget(QLabel("  (Turn OFF to zoom/pan freely)"))
        
        ctrl_layout.addStretch()
        map_layout.addLayout(ctrl_layout)
        
        # Plot area for map
        self.fig = Figure(figsize=(12, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        map_layout.addWidget(toolbar)
        map_layout.addWidget(self.canvas)
        map_widget.setLayout(map_layout)
        tabs.addTab(map_widget, "Flux Map")
        
        layout.addWidget(tabs)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export as FITS")
        export_btn.clicked.connect(self._export_to_fits)
        btn_layout.addWidget(export_btn)
        
        stats_btn = QPushButton("Show Statistics")
        stats_btn.clicked.connect(self._show_statistics)
        btn_layout.addWidget(stats_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        
        self._update_plot()
    
    def _toggle_pixel_selection(self):
        """Toggle pixel selection mode on/off.

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
        """Handle click on flux map to inspect pixel spectrum.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Click event.

        Returns:
            None
        """
        # Only process clicks if pixel selection is enabled
        if not self.pixel_selection_enabled:
            return
        
        if event.inaxes is None or self.cube_data is None or self.wavelength is None:
            return
        
        # Get pixel coordinates from click
        x = int(np.round(event.xdata))
        y = int(np.round(event.ydata))
        
        # Validate coordinates
        if 0 <= x < self.flux_map.shape[1] and 0 <= y < self.flux_map.shape[0]:
            self._show_pixel_spectrum(x, y)
    
    def _show_pixel_spectrum(self, x, y):
        """Extract and display spectrum for a clicked pixel with fitted Gaussians.

        Args:
            x (int): Pixel x coordinate.
            y (int): Pixel y coordinate.

        Returns:
            None
        """
        try:
            from .flux_map_extractor import extract_flux_from_spectrum
            
            # Extract spectrum for this pixel
            spectrum = self.cube_data.get_pixel_spectrum(x, y)
            
            if spectrum is None or np.all(np.isnan(spectrum)):
                QMessageBox.warning(self, "No Data", f"Pixel ({x}, {y}) has no valid data")
                return
            
            # Re-extract flux for this pixel to get fit info
            if self.target_catalog is not None:
                result = extract_flux_from_spectrum(
                    self.wavelength, spectrum, self.target_catalog, self.contam_catalog,
                    wave_min=self.wave_range[0] if self.wave_range else None,
                    wave_max=self.wave_range[1] if self.wave_range else None,
                    continuum_params=self.continuum_params,
                    header=self.cube_data.header, wave_unit=self.cube_data.wave_unit
                )
                
                # Create inspection dialog
                dlg = QDialog(self)
                dlg.setWindowTitle(f"Pixel ({x}, {y}) Spectrum")
                dlg.resize(1000, 600)
                
                layout = QVBoxLayout()
                
                # Info
                flux_val = self.flux_map[y, x] if 0 <= x < self.flux_map.shape[1] and 0 <= y < self.flux_map.shape[0] else 0
                n_lines = self.detection_map[y, x] if 0 <= x < self.detection_map.shape[1] and 0 <= y < self.detection_map.shape[0] else 0
                info = QLabel(
                    f"<b>Pixel ({x}, {y})</b> | "
                    f"{self.species_label} Flux: {flux_val:.4e} {self.flux_unit_str} | "
                    f"Lines Detected: {n_lines}"
                )
                info.setStyleSheet("background-color: #f0f0f0; padding: 4px; border-radius: 3px;")
                layout.addWidget(info)
                
                # Plot spectrum with fit
                fig = Figure(figsize=(10, 5))
                ax = fig.add_subplot(111)
                
                # Create canvas FIRST so figure has a canvas for events
                canvas = FigureCanvasQTAgg(fig)
                
                # Plot spectrum
                ax.plot(self.wavelength, spectrum, 'k-', label='Spectrum', linewidth=1.2, alpha=0.8)

                # Plot the actual fitted continuum trend, not just its mean; a
                # flat mean line would mislead once drawn across the full spectrum.
                continuum_curve = result.get('continuum_curve')
                if continuum_curve is not None and len(continuum_curve) > 0:
                    ax.plot(continuum_curve['wave'], continuum_curve['continuum'], color='orange',
                            linestyle='--', linewidth=1.8, label='Continuum', alpha=0.8)
                
                # Grayed out if deselected via the species dialog before the map
                # was computed; this only displays that exclusion, not a control.
                detected = result.get('detected_lines')
                if detected is not None:
                    try:
                        import pandas as pd
                        if isinstance(detected, pd.DataFrame) and len(detected) > 0:
                            detected_target = detected[
                                (detected['disposition'] == 1) &
                                (detected['line'] >= self.wave_range[0]) &
                                (detected['line'] <= self.wave_range[1])
                            ]
                            for idx, line_row in detected_target.iterrows():
                                line_wave = float(line_row['line'])
                                is_excluded = any(abs(line_wave - ew) < 0.001 for ew in self.excluded_wavelengths)
                                color = 'gray' if is_excluded else 'red'
                                alpha = 0.3 if is_excluded else 0.5
                                ax.axvline(line_wave, color=color, alpha=alpha, linewidth=1.5,
                                           label=f'{self.species_label} Lines' if idx == detected_target.index[0] else '')
                    except Exception as e:
                        print(f"DEBUG: Error marking lines: {e}")
                        pass  # Silently skip line marking if it fails
                
                # Zoom to the fitted wavelength range so it fills the plot instead of being a
                # sliver inside the full cube bandpass, and rescale y to that window too
                if self.wave_range:
                    pad = 0.02 * (self.wave_range[1] - self.wave_range[0])
                    xlo, xhi = self.wave_range[0] - pad, self.wave_range[1] + pad
                    ax.set_xlim(xlo, xhi)
                    wave_arr = np.asarray(getattr(self.wavelength, 'value', self.wavelength))
                    visible = (wave_arr >= xlo) & (wave_arr <= xhi)
                    y_visible = np.asarray(spectrum)[visible]
                    y_visible = y_visible[np.isfinite(y_visible)]
                    if len(y_visible) > 0:
                        ylo, yhi = np.nanmin(y_visible), np.nanmax(y_visible)
                        ypad = 0.1 * (yhi - ylo) if yhi > ylo else 1.0
                        ax.set_ylim(ylo - ypad, yhi + ypad)

                ax.set_xlabel('Wavelength (μm)')
                ax.set_ylabel(f'Flux ({self.bunit_str})')
                ax.set_title(f'Pixel ({x}, {y}) Spectrum', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                canvas.draw()

                toolbar = NavigationToolbar2QT(canvas, dlg)
                plot_layout = QVBoxLayout()
                plot_layout.addWidget(toolbar)
                plot_layout.addWidget(canvas)
                plot_widget = QWidget()
                plot_widget.setLayout(plot_layout)
                layout.addWidget(plot_widget)

                # Close button
                close_btn = QPushButton("Close")
                close_btn.clicked.connect(dlg.accept)
                layout.addWidget(close_btn)
                
                dlg.setLayout(layout)
                dlg.exec_()
            else:
                QMessageBox.warning(self, "No Catalog", "Line catalogs not available for spectrum inspection")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to inspect pixel:\n{str(e)}")

    def _update_plot(self):
        """Update visualization.

        Args:
            None

        Returns:
            None
        """
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        param = self.param_combo.currentText()
        cmap = self.cmap_combo.currentText()
        scale = self.scale_combo.currentText()
        
        # Select data and units
        if param == f"{self.species_label} Flux":
            data = self.flux_map
            title = f"{self.species_label} Rovibrational Flux"
            cbar_label = f"Integrated {self.species_label} Flux ({self.flux_unit_str})"
        elif param == "Contam Flux":
            data = self.contam_flux_map
            title = "Contaminating-Line Flux"
            cbar_label = f"Integrated Contam Flux ({self.flux_unit_str})"
        elif param == "Contam Detection Count":
            data = self.contam_detection_map.astype(float)
            title = "Contaminating Lines Detected"
            cbar_label = "Number of Lines"
        else:
            data = self.detection_map.astype(float)
            title = f"{self.species_label} Lines Detected"
            cbar_label = "Number of Lines"

        # Handle NaN
        plot_data = np.nan_to_num(data, nan=0.0)
        
        # Scaling
        if scale == "Log":
            plot_data = np.abs(plot_data)
            plot_data[plot_data < 1e-10] = 1e-10
            plot_data = np.log10(plot_data)
        elif scale == "Percentile":
            valid = plot_data[plot_data != 0]
            if len(valid) > 0:
                vmin, vmax = np.percentile(valid, [2, 98])
                plot_data = np.clip(plot_data, vmin, vmax)
        
        im = ax.imshow(plot_data, origin='lower', cmap=cmap)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("X [pixels]")
        ax.set_ylabel("Y [pixels]")
        cbar = self.fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _export_to_fits(self):
        """Export results to FITS file.

        Args:
            None

        Returns:
            None
        """
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Flux Results",
            f"{self.species_label}_flux_map.fits",
            "FITS files (*.fits);;All files (*.*)"
        )
        
        if path:
            try:
                from astropy.io import fits
                
                # Flux HDU (Primary) - include original header if available
                hdu = fits.PrimaryHDU(data=self.flux_map.astype(np.float32))
                
                # Copy instrument/origin info from original header if available
                if self.header is not None:
                    # Copy essential metadata from original cube header
                    instrument_keywords = ['INSTRUME', 'TELESCOP', 'OBSERVER', 'OBJECT', 'DATE-OBS', 'EXPTIME']
                    for kw in instrument_keywords:
                        if kw in self.header:
                            hdu.header[kw] = self.header[kw]
                
                # Ensure INSTRUME is set - get from cube_data if not in header
                if 'INSTRUME' not in hdu.header and hasattr(self, 'cube_data') and self.cube_data is not None:
                    if hasattr(self.cube_data, 'instrument'):
                        hdu.header['INSTRUME'] = self.cube_data.instrument
                
                hdu.header['BUNIT'] = self.flux_unit_str
                hdu.header['SPECIES'] = (self.species_label, 'Target line species')
                hdu.header['DESC'] = f'{self.species_label} rovibrational flux: Integrated area under target line gaussians'
                if self.wave_range:
                    hdu.header['WAVEMIN'] = (self.wave_range[0], 'Min wavelength used (um)')
                    hdu.header['WAVEMAX'] = (self.wave_range[1], 'Max wavelength used (um)')
                
                # Remove only problematic WCS coordinate keywords (not structural FITS keywords)
                wcs_keywords = ['CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2', 'CRPIX1', 'CRPIX2',
                               'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2', 'CMATRIX',
                               'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
                for kw in wcs_keywords:
                    if kw in hdu.header:
                        del hdu.header[kw]
                
                hdul = fits.HDUList([hdu])
                
                # Detection HDU
                hdu = fits.ImageHDU(data=self.detection_map.astype(np.int32))
                hdu.header['EXTNAME'] = 'DETECTION'
                hdu.header['BUNIT'] = 'count'
                hdu.header['DESC'] = 'Number of target lines detected per pixel'
                hdul.append(hdu)

                # Contaminant flux HDU (only if this map has one -- older cached results won't)
                if self.contam_flux_map is not None:
                    hdu = fits.ImageHDU(data=self.contam_flux_map.astype(np.float32))
                    hdu.header['EXTNAME'] = 'CONTAM_FLUX'
                    hdu.header['BUNIT'] = self.flux_unit_str
                    hdu.header['DESC'] = 'Contaminating-line flux: integrated area under non-target line gaussians'
                    hdul.append(hdu)

                # Contaminant detection-count HDU (only if this map has one)
                if self.contam_detection_map is not None:
                    hdu = fits.ImageHDU(data=self.contam_detection_map.astype(np.int32))
                    hdu.header['EXTNAME'] = 'CONTAM_DETECTION'
                    hdu.header['BUNIT'] = 'count'
                    hdu.header['DESC'] = 'Number of contaminating lines detected per pixel'
                    hdul.append(hdu)

                # Write with warnings suppressed for missing instrument keywords
                hdul.writeto(path, overwrite=True, output_verify='warn')
                
                QMessageBox.information(self, "Success", f"Saved to:\n{path}")
            except Exception as e:
                import traceback
                error_msg = f"Failed to export:\n{str(e)}\n\n{traceback.format_exc()}"
                QMessageBox.critical(self, "Error", error_msg)
    
    def _show_statistics(self):
        """Show detailed statistics.

        Args:
            None

        Returns:
            None
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Statistics")
        dlg.resize(500, 350)
        
        layout = QVBoxLayout()
        
        stats = f"{self.species_label.upper()} FLUX MAP STATISTICS\n" + "="*50 + "\n\n"

        # Flux statistics with correct units
        flux_valid = self.flux_map[~np.isnan(self.flux_map)]
        if len(flux_valid) > 0:
            stats += f"{self.species_label.upper()} FLUX ({self.flux_unit_str}):\n"
            stats += f"  Count: {len(flux_valid):,}\n"
            stats += f"  Sum:   {np.sum(flux_valid):.4e}\n"
            stats += f"  Mean:  {np.mean(flux_valid):.4e}\n"
            stats += f"  Std:   {np.std(flux_valid):.4e}\n"
            stats += f"  Min/Max: {np.min(flux_valid):.4e} / {np.max(flux_valid):.4e}\n\n"
        
        # Detection statistics (unitless count)
        detection_valid = self.detection_map[self.detection_map > 0]
        if len(detection_valid) > 0:
            stats += f"DETECTION COUNT (lines/pixel):\n"
            stats += f"  Pixels with lines: {len(detection_valid):,}\n"
            stats += f"  Sum:   {np.sum(detection_valid):.0f}\n"
            stats += f"  Mean:  {np.mean(detection_valid):.2f}\n"
            stats += f"  Min/Max: {np.min(detection_valid)} / {np.max(detection_valid)}\n\n"

        # Contaminant flux statistics (only if this map has one)
        if self.contam_flux_map is not None:
            contam_valid = self.contam_flux_map[~np.isnan(self.contam_flux_map)]
            if len(contam_valid) > 0:
                stats += f"CONTAM FLUX ({self.flux_unit_str}):\n"
                stats += f"  Count: {len(contam_valid):,}\n"
                stats += f"  Sum:   {np.sum(contam_valid):.4e}\n"
                stats += f"  Mean:  {np.mean(contam_valid):.4e}\n"
                stats += f"  Std:   {np.std(contam_valid):.4e}\n"
                stats += f"  Min/Max: {np.min(contam_valid):.4e} / {np.max(contam_valid):.4e}\n\n"

        # Contaminant detection-count statistics (only if this map has one)
        if self.contam_detection_map is not None:
            contam_detection_valid = self.contam_detection_map[self.contam_detection_map > 0]
            if len(contam_detection_valid) > 0:
                stats += f"CONTAM DETECTION COUNT (lines/pixel):\n"
                stats += f"  Pixels with contam lines: {len(contam_detection_valid):,}\n"
                stats += f"  Sum:   {np.sum(contam_detection_valid):.0f}\n"
                stats += f"  Mean:  {np.mean(contam_detection_valid):.2f}\n"
                stats += f"  Min/Max: {np.min(contam_detection_valid)} / {np.max(contam_detection_valid)}\n"

        label = QLabel(stats)
        label.setStyleSheet("font-family: monospace; white-space: pre;")
        label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        layout.addWidget(label)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        
        dlg.setLayout(layout)
        dlg.exec_()
