"""
Gaussian Fitting Dialog and Worker Classes

Contains QThread workers and dialog classes for Gaussian map fitting
to keep the main GUI file cleaner and more maintainable.
"""

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QFileDialog, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class GaussianMapWorker(QThread):
    """Background worker thread for Gaussian map fitting"""
    finished = pyqtSignal(dict)  # Emits results when done
    error = pyqtSignal(str)  # Emits error message if failed
    
    def __init__(self, cube, wavelength, lam_min, lam_max, n_jobs=8):
        super().__init__()
        self.cube = cube
        self.wavelength = wavelength
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.n_jobs = n_jobs
    
    def run(self):
        """Execute fitting in background thread"""
        try:
            from .gaussian_fitter import fit_gaussians_per_pixel
            results = fit_gaussians_per_pixel(
                cube=self.cube,
                wavelength=self.wavelength,
                lam_min=self.lam_min,
                lam_max=self.lam_max,
                n_jobs=self.n_jobs
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class GaussianMapResultsDialog(QDialog):
    """
    Dialog window for displaying Gaussian parameter maps.
    
    Shows 2D maps of fitted parameters (area, amplitude, center, etc.)
    with interactive parameter selection and visualization controls.
    """
    
    def __init__(self, parent, results, cube_data, gauss_left_x, gauss_right_x):
        super().__init__(parent)
        self.parent_widget = parent
        self.results = results
        self.cube_data = cube_data
        self.gauss_left_x = gauss_left_x
        self.gauss_right_x = gauss_right_x
        self._export_map_btn = None
        
        self.setWindowTitle("Gaussian Parameter Maps")
        self.resize(800, 700)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize dialog UI components"""
        layout = QVBoxLayout()
        
        # Info label
        n_success = np.sum(self.results['success'])
        n_total = self.results['success'].size
        info_label = QLabel(
            f"Successfully fit {n_success:,} / {n_total:,} pixels "
            f"({100*n_success/n_total:.1f}%)"
        )
        layout.addWidget(info_label)
        
        # Parameter selector
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Parameter:"))
        self.param_combo = QComboBox()
        self.param_combo.addItems([
            "Area (AUC)", "Amplitude", "Center", "Sigma", 
            "FWHM", "χ² (fit quality)", "Success Mask", "Residuals"
        ])
        param_layout.addWidget(self.param_combo)
        param_layout.addStretch()
        layout.addLayout(param_layout)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        
        # Connect combo box to update function
        self.param_combo.currentIndexChanged.connect(self._update_plot)
        
        # Initial plot
        self._update_plot()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
    
    def _update_plot(self):
        """Update plot when parameter selection changes"""
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
        
        # Select data based on parameter
        if param_name == "Area (AUC)":
            data = self.results['area']
            title = f"Integrated Area Under Gaussian (AUC) {wave_range}"
            cmap = 'viridis'
        elif param_name == "Amplitude":
            data = self.results['amplitude']
            title = f"Gaussian Peak Amplitude {wave_range}"
            cmap = 'plasma'
        elif param_name == "Center":
            data = self.results['center']
            title = f"Gaussian Center Position {wave_range}"
            cmap = 'coolwarm'
        elif param_name == "Sigma":
            data = self.results['sigma']
            title = f"Gaussian Width (σ) {wave_range}"
            cmap = 'cividis'
        elif param_name == "FWHM":
            data = self.results['fwhm']
            title = f"Full Width at Half Maximum {wave_range}"
            cmap = 'cividis'
        elif param_name == "χ² (fit quality)":
            data = self.results['r_squared']
            title = f"Reduced χ² (≈1 is good fit) {wave_range}"
            cmap = 'RdYlGn_r'
        elif param_name == "Success Mask":
            data = self.results['success'].astype(float)
            title = f"Successful Fits (1=success, 0=failed) {wave_range}"
            cmap = 'gray'
        elif param_name == "Residuals":
            self._handle_residuals_display(ax)
            return
        
        # Plot with NaN handling
        im = ax.imshow(data, origin='lower', cmap=cmap, interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")
        
        # Colorbar
        self.fig.colorbar(im, ax=ax, label=param_name)
        
        # Add statistics
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            stats_text = (
                f"Min: {np.nanmin(data):.4e}\n"
                f"Max: {np.nanmax(data):.4e}\n"
                f"Mean: {np.nanmean(data):.4e}\n"
                f"Median: {np.nanmedian(data):.4e}"
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.8), fontsize=9)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _handle_residuals_display(self, ax):
        """Handle residuals display with export button"""
        if 'residuals_cube' in self.results:
            residual_cube = self.results['residuals_cube']
            # Show middle wavelength slice
            mid_idx = residual_cube.shape[0] // 2
            data = residual_cube[mid_idx]
            
            # Format wavelength range
            lam_min = min(self.gauss_left_x, self.gauss_right_x)
            lam_max = max(self.gauss_left_x, self.gauss_right_x)
            wave_range = f"({lam_min:.4f} - {lam_max:.4f})"
            title = f"Normalized Residuals (Peak Wavelength) {wave_range}"
            cmap = 'seismic'
            
            # Plot
            im = ax.imshow(data, origin='lower', cmap=cmap, interpolation='nearest')
            ax.set_title(title)
            ax.set_xlabel("X Pixel")
            ax.set_ylabel("Y Pixel")
            self.fig.colorbar(im, ax=ax, label="Residuals")
            
            # Add statistics
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                stats_text = (
                    f"Min: {np.nanmin(data):.4e}\n"
                    f"Max: {np.nanmax(data):.4e}\n"
                    f"Mean: {np.nanmean(data):.4e}\n"
                    f"Median: {np.nanmedian(data):.4e}"
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.8), fontsize=9)
            
            # Add export button
            self._export_map_btn = QPushButton("Export Residuals Cube")
            self._export_map_btn.clicked.connect(self._export_residuals)
            # Insert before close button
            self.layout().insertWidget(self.layout().count() - 1, self._export_map_btn)
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
        """Export residuals cube to FITS file"""
        out_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Residuals FITS", 
            "residual_cube.fits", 
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
