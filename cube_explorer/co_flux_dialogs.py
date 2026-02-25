"""
CO Rovibrational Flux Map Dialog and Worker Classes

Contains QThread workers and dialog classes for CO flux map extraction.
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


class COFluxMapWorker(QThread):
    """Background worker for CO flux extraction"""
    progress = pyqtSignal(int, int)  # (current, total)
    finished = pyqtSignal(dict)       # Results dict
    error = pyqtSignal(str)           # Error message
    
    def __init__(self, cube_data, CO_catalog, contam_catalog, n_jobs=4, subtract_continuum=False, continuum_poly_deg=3, wave_min=None, wave_max=None):
        super().__init__()
        self.cube_data = cube_data
        self.CO_catalog = CO_catalog
        self.contam_catalog = contam_catalog
        self.n_jobs = n_jobs
        self.subtract_continuum = subtract_continuum
        self.continuum_poly_deg = continuum_poly_deg
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.cancelled = False  # Flag to stop processing
    
    def run(self):
        """Execute extraction"""
        try:
            if self.cancelled:
                self.error.emit("Extraction cancelled by user")
                return
            
            from .co_flux_extractor import create_co_flux_map
            
            results = create_co_flux_map(
                cube_data=self.cube_data,
                CO_catalog=self.CO_catalog,
                contam_catalog=self.contam_catalog,
                n_jobs=self.n_jobs,
                progress_callback=self._emit_progress,
                cancel_flag=self,  # Pass self so extraction can check cancelled flag
                subtract_continuum=self.subtract_continuum,
                continuum_poly_deg=self.continuum_poly_deg,
                wave_min=self.wave_min,
                wave_max=self.wave_max
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
        self.progress.emit(current, total)
    
    def request_cancel(self):
        """Request cancellation"""
        self.cancelled = True


class COFluxMapResultsDialog(QDialog):
    """Display CO flux extraction results with interactive viewer"""
    
    def __init__(self, flux_map, error_map, detection_map, header=None, wave_range=None, 
                 wavelength=None, cube_data=None, CO_catalog=None, contam_catalog=None,
                 subtract_continuum=False, parent=None):
        super().__init__(parent)
        self.flux_map = flux_map
        self.error_map = error_map
        self.detection_map = detection_map
        self.header = header
        self.wave_range = wave_range
        self.wavelength = wavelength
        self.cube_data = cube_data
        self.CO_catalog = CO_catalog
        self.contam_catalog = contam_catalog
        self.subtract_continuum = subtract_continuum
        self.selected_pixel = None  # For pixel inspection mode
        
        # Extract units from FITS header using astropy units
        try:
            bunit_str = header.get('BUNIT', 'MJy/sr') if header else 'MJy/sr'
            # Try to parse as astropy unit, fallback to string if it fails
            try:
                self.flux_unit_base = u.Unit(bunit_str)
            except (ValueError, TypeError):
                # If it's not a valid astropy unit, use it as-is for display
                self.flux_unit_base = bunit_str
            
            # CO flux unit is base flux unit per wavelength
            if isinstance(self.flux_unit_base, u.UnitBase):
                self.flux_unit = self.flux_unit_base * u.um
                self.flux_unit_str = f"{self.flux_unit_base} um"
                self.bunit_str = f"{self.flux_unit_base}"
            else:
                self.flux_unit_str = f"{self.flux_unit_base} um"
                self.bunit_str = self.flux_unit_base
        except Exception:
            # Fallback if header parsing fails
            self.flux_unit_str = "MJy/sr um"
            self.bunit_str = "MJy/sr"
            self.flux_unit_base = u.MJy / u.sr
            self.flux_unit = self.flux_unit_base * u.um
        
        self.setWindowTitle("CO Flux Map Results")
        self.resize(1200, 700)
        self._setup_ui()
    
    def _setup_ui(self):
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
            f"Total CO flux: {sum_flux:.4e} | Mean: {mean_flux:.4e} | Max: {max_flux:.4e} | "
            f"Mean detections: {mean_detections:.1f} lines/pixel"
        )
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
        self.param_combo.addItems(["CO Flux", "Error", "Detection Count"])
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
        
        ctrl_layout.addWidget(QLabel("  ℹ️ Click a pixel to inspect its spectrum"))
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
    
    def _on_canvas_click(self, event):
        """Handle click on flux map to inspect pixel spectrum"""
        if event.inaxes is None or self.cube_data is None or self.wavelength is None:
            return
        
        # Get pixel coordinates from click
        x = int(np.round(event.xdata))
        y = int(np.round(event.ydata))
        
        # Validate coordinates
        if 0 <= x < self.flux_map.shape[1] and 0 <= y < self.flux_map.shape[0]:
            self._show_pixel_spectrum(x, y)
    
    def _show_pixel_spectrum(self, x, y):
        """Extract and display spectrum for a clicked pixel with fitted Gaussians"""
        try:
            from .co_flux_extractor import extract_co_flux_from_spectrum
            
            # Extract spectrum for this pixel
            spectrum = self.cube_data.get_pixel_spectrum(x, y)
            
            if spectrum is None or np.all(np.isnan(spectrum)):
                QMessageBox.warning(self, "No Data", f"Pixel ({x}, {y}) has no valid data")
                return
            
            # Re-extract CO flux for this pixel to get fit info
            if self.CO_catalog is not None and self.contam_catalog is not None:
                result = extract_co_flux_from_spectrum(
                    self.wavelength, spectrum, self.CO_catalog, self.contam_catalog,
                    wave_min=self.wave_range[0] if self.wave_range else None,
                    wave_max=self.wave_range[1] if self.wave_range else None,
                    subtract_continuum=self.subtract_continuum
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
                    f"CO Flux: {flux_val:.4e} {self.flux_unit_str} | "
                    f"Lines Detected: {n_lines}"
                )
                info.setStyleSheet("background-color: #f0f0f0; padding: 4px; border-radius: 3px;")
                layout.addWidget(info)
                
                # Plot spectrum with fit
                fig = Figure(figsize=(10, 5))
                ax = fig.add_subplot(111)
                
                # Plot spectrum
                ax.plot(self.wavelength, spectrum, 'k-', label='Spectrum', linewidth=1.2, alpha=0.8)
                
                # Plot continuum if subtracted
                if self.subtract_continuum and result.get('mean_continuum'):
                    ax.axhline(result['mean_continuum'], color='orange', linestyle='--', label='Continuum', alpha=0.7)
                
                # Mark detected CO lines with disposition=1
                detected = result.get('detected_lines')
                if detected is not None:
                    try:
                        import pandas as pd
                        if isinstance(detected, pd.DataFrame):
                            # DataFrame format - filter by disposition and wavelength range
                            detected_co = detected[
                                (detected['disposition'] == 1) & 
                                (detected['line'] >= self.wave_range[0]) & 
                                (detected['line'] <= self.wave_range[1])
                            ]
                            for idx, line_row in detected_co.iterrows():
                                line_wave = float(line_row['line'])
                                ax.axvline(line_wave, color='red', alpha=0.5, linewidth=1.5, label='CO Lines' if idx == detected_co.index[0] else '')
                    except Exception as e:
                        pass  # Silently skip line marking if it fails
                
                ax.set_xlabel('Wavelength (μm)')
                ax.set_ylabel(f'Flux ({self.bunit_str})')
                ax.set_title(f'Pixel ({x}, {y}) Spectrum', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                canvas = FigureCanvasQTAgg(fig)
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
        """Update visualization"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        param = self.param_combo.currentText()
        cmap = self.cmap_combo.currentText()
        scale = self.scale_combo.currentText()
        
        # Select data and units
        if param == "CO Flux":
            data = self.flux_map
            title = "CO Rovibrational Flux"
            cbar_label = f"Integrated CO Flux ({self.flux_unit_str})"
        elif param == "Error":
            data = self.error_map
            title = "Uncertainty (σ)"
            cbar_label = f"Noise Std ({self.bunit_str})"
        else:
            data = self.detection_map.astype(float)
            title = "CO Lines Detected"
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
        """Export results to FITS file"""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CO Flux Results",
            "CO_flux_map.fits",
            "FITS files (*.fits);;All files (*.*)"
        )
        
        if path:
            try:
                from astropy.io import fits
                
                hdul = fits.HDUList()
                
                # Flux HDU
                hdu = fits.PrimaryHDU(data=self.flux_map.astype(np.float32))
                hdu.header['EXTNAME'] = 'FLUX'
                hdu.header['BUNIT'] = self.flux_unit_str
                hdu.header['DESC'] = 'CO rovibrational flux: Integrated area under CO line gaussians'
                if self.wave_range:
                    hdu.header['WAVEMIN'] = (self.wave_range[0], 'Min wavelength used (um)')
                    hdu.header['WAVEMAX'] = (self.wave_range[1], 'Max wavelength used (um)')
                
                # Remove problematic WCS keywords if present
                wcs_keywords = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 
                               'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2', 'CRPIX1', 'CRPIX2',
                               'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2', 'CMATRIX',
                               'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
                for kw in wcs_keywords:
                    if kw in hdu.header:
                        del hdu.header[kw]
                
                hdul.append(hdu)
                
                # Error HDU
                hdu = fits.ImageHDU(data=self.error_map.astype(np.float32))
                hdu.header['EXTNAME'] = 'ERROR'
                hdu.header['BUNIT'] = self.bunit_str
                hdu.header['DESC'] = 'Noise standard deviation per pixel'
                hdul.append(hdu)
                
                # Detection HDU
                hdu = fits.ImageHDU(data=self.detection_map.astype(np.int32))
                hdu.header['EXTNAME'] = 'DETECTION'
                hdu.header['BUNIT'] = 'count'
                hdu.header['DESC'] = 'Number of CO lines detected per pixel'
                hdul.append(hdu)
                
                # Write with warnings suppressed for missing instrument keywords
                hdul.writeto(path, overwrite=True, output_verify='warn')
                
                QMessageBox.information(self, "Success", f"Saved to:\n{path}")
            except Exception as e:
                import traceback
                error_msg = f"Failed to export:\n{str(e)}\n\n{traceback.format_exc()}"
                QMessageBox.critical(self, "Error", error_msg)
    
    def _show_statistics(self):
        """Show detailed statistics"""
        dlg = QDialog(self)
        dlg.setWindowTitle("Statistics")
        dlg.resize(400, 300)
        
        layout = QVBoxLayout()
        
        stats = "CO FLUX MAP STATISTICS\n" + "="*40 + "\n\n"
        
        for name, data in [("Flux", self.flux_map), ("Error", self.error_map), ("Detection", self.detection_map)]:
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                stats += f"{name}:\n"
                stats += f"  Count: {len(valid):,}\n"
                stats += f"  Sum: {np.sum(valid):.4e}\n"
                stats += f"  Mean: {np.mean(valid):.4e}\n"
                stats += f"  Min/Max: {np.min(valid):.4e} / {np.max(valid):.4e}\n\n"
        
        label = QLabel(stats)
        label.setStyleSheet("font-family: monospace;")
        label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        layout.addWidget(label)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        
        dlg.setLayout(layout)
        dlg.exec_()
