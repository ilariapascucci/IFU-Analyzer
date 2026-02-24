"""
CO Rovibrational Flux Map Dialog and Worker Classes

Contains QThread workers and dialog classes for CO flux map extraction.
Simplified: no parameter dialog, automatic processing of entire spectrum.
"""

import numpy as np
import os
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QMessageBox, QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.cm as cm


class COFluxMapWorker(QThread):
    """Background worker for CO flux extraction"""
    progress = pyqtSignal(int, int)  # (current, total)
    finished = pyqtSignal(dict)       # Results dict
    error = pyqtSignal(str)           # Error message
    
    def __init__(self, cube_data, CO_catalog, contam_catalog, n_jobs=4, subtract_continuum=False, continuum_poly_deg=3):
        super().__init__()
        self.cube_data = cube_data
        self.CO_catalog = CO_catalog
        self.contam_catalog = contam_catalog
        self.n_jobs = n_jobs
        self.subtract_continuum = subtract_continuum
        self.continuum_poly_deg = continuum_poly_deg
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
                continuum_poly_deg=self.continuum_poly_deg
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
    
    def __init__(self, flux_map, error_map, detection_map, parent=None):
        super().__init__(parent)
        self.flux_map = flux_map
        self.error_map = error_map
        self.detection_map = detection_map
        
        self.setWindowTitle("CO Flux Map Results")
        self.resize(1200, 700)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # Info section
        n_valid = np.sum(~np.isnan(self.flux_map))
        n_total = self.flux_map.size
        mean_flux = np.nanmean(self.flux_map) if n_valid > 0 else 0
        max_flux = np.nanmax(self.flux_map) if n_valid > 0 else 0
        detect_with_lines = self.detection_map[self.detection_map > 0]
        mean_detections = np.mean(detect_with_lines) if len(detect_with_lines) > 0 else 0
        
        info_text = (
            f"Valid pixels: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%) | "
            f"Mean CO flux: {mean_flux:.4e} | Max flux: {max_flux:.4e} | "
            f"Mean detections: {mean_detections:.1f} lines/pixel"
        )
        info_label = QLabel(info_text)
        info_label.setStyleSheet("background-color: #f0f0f0; padding: 6px; border-radius: 3px;")
        layout.addWidget(info_label)
        
        # Controls
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
        
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)
        
        # Plot area
        self.fig = Figure(figsize=(12, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        
        self._update_plot()
        
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
    
    def _update_plot(self):
        """Update visualization"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        param = self.param_combo.currentText()
        cmap = self.cmap_combo.currentText()
        scale = self.scale_combo.currentText()
        
        # Select data
        if param == "CO Flux":
            data = self.flux_map
            title = "CO Rovibrational Flux"
        elif param == "Error":
            data = self.error_map
            title = "Uncertainty"
        else:
            data = self.detection_map.astype(float)
            title = "Detection Count"
        
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
        cbar.set_label("Value")
        
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
                hdu = fits.PrimaryHDU(data=self.flux_map)
                hdu.header['EXTNAME'] = 'FLUX'
                hdu.header['BUNIT'] = 'erg/(s cm2)'
                hdu.header['DESC'] = 'CO rovibrational flux'
                hdul.append(hdu)
                
                # Error HDU
                hdu = fits.ImageHDU(data=self.error_map)
                hdu.header['EXTNAME'] = 'ERROR'
                hdu.header['BUNIT'] = 'erg/(s cm2)'
                hdul.append(hdu)
                
                # Detection HDU
                hdu = fits.ImageHDU(data=self.detection_map)
                hdu.header['EXTNAME'] = 'DETECTION'
                hdu.header['BUNIT'] = 'count'
                hdul.append(hdu)
                
                hdul.writeto(path, overwrite=True)
                
                QMessageBox.information(self, "Success", f"Saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")
    
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
                stats += f"  Mean: {np.mean(valid):.4e}\n"
                stats += f"  Min/Max: {np.min(valid):.4e} / {np.max(valid):.4e}\n\n"
        
        label = QLabel(stats)
        label.setStyleSheet("font-family: monospace;")
        layout.addWidget(label)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        
        dlg.setLayout(layout)
        dlg.exec_()
