"""
Pixel Diagnostics Tool for Gaussian Fitting

Provides interactive visualization of Gaussian fits for individual pixels,
allowing users to inspect fitting quality and understand the fitting process.
"""

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton, 
    QMessageBox, QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from .gaussian_fitter import gaussian, calculate_baseline


class PixelDiagnosticsDialog(QDialog):
    """
    Interactive dialog for inspecting Gaussian fits at individual pixels.
    
    Allows user to select a pixel (by row/column) and see:
    - Spectrum data points
    - Linear baseline
    - Fitted Gaussian model
    - Residuals
    - Quality metrics and detection status
    """
    
    def __init__(self, parent, cube, wavelength, lam_min, lam_max, fit_results, fac_above_stdres=3.0):
        """
        Args:
            parent: Parent widget
            cube: 3D data cube (n_wavelength, n_y, n_x)
            wavelength: 1D wavelength array
            lam_min, lam_max: Wavelength range used for fitting
            fit_results: Dictionary returned from fit_gaussians_per_pixel()
            fac_above_stdres: Detection threshold in units of noise (e.g., 3.0 for 3σ)
        """
        super().__init__(parent)
        self.cube = cube
        self.wavelength = wavelength
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.fit_results = fit_results
        self.fac_above_stdres = fac_above_stdres
        
        # Get wavelength mask
        self.wave_mask = (wavelength >= lam_min) & (wavelength <= lam_max)
        self.x_data = wavelength[self.wave_mask]
        
        n_wave, n_y, n_x = cube.shape
        self.n_y = n_y
        self.n_x = n_x
        
        self.setWindowTitle("Pixel Gaussian Fit Diagnostics")
        self.resize(1200, 700)
        
        self._setup_ui()
        
        # Display first pixel by default
        self._plot_pixel(0, 0)
    
    def _setup_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        
        # Pixel selection controls
        selection_layout = QHBoxLayout()
        
        selection_layout.addWidget(QLabel("Row (Y):"))
        self.row_spinbox = QSpinBox()
        self.row_spinbox.setMinimum(0)
        self.row_spinbox.setMaximum(self.n_y - 1)
        self.row_spinbox.setValue(0)
        self.row_spinbox.valueChanged.connect(self._on_pixel_changed)
        selection_layout.addWidget(self.row_spinbox)
        
        selection_layout.addWidget(QLabel("  Column (X):"))
        self.col_spinbox = QSpinBox()
        self.col_spinbox.setMinimum(0)
        self.col_spinbox.setMaximum(self.n_x - 1)
        self.col_spinbox.setValue(0)
        self.col_spinbox.valueChanged.connect(self._on_pixel_changed)
        selection_layout.addWidget(self.col_spinbox)
        
        selection_layout.addWidget(QLabel("  Jump to:"))
        self.jump_combo = QComboBox()
        self.jump_combo.addItem("Select pixel type...")
        self.jump_combo.addItem("Next Detection")
        self.jump_combo.addItem("Next Upper Limit")
        self.jump_combo.addItem("Next Failed Fit")
        self.jump_combo.currentIndexChanged.connect(self._on_jump_selected)
        selection_layout.addWidget(self.jump_combo)
        
        selection_layout.addStretch()
        layout.addLayout(selection_layout)
        
        # Info label
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        layout.addWidget(self.info_label)
        
        # Create matplotlib figure with subplots
        self.fig = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
    
    def _on_pixel_changed(self):
        """Called when row or column spinboxes change"""
        row = self.row_spinbox.value()
        col = self.col_spinbox.value()
        self._plot_pixel(row, col)
    
    def _on_jump_selected(self, index):
        """Jump to next pixel of selected type"""
        if index == 0:
            return  # "Select..." placeholder
        
        current_row = self.row_spinbox.value()
        current_col = self.col_spinbox.value()
        
        pixel_type = self.jump_combo.currentText()
        
        if pixel_type == "Next Detection":
            detection_map = self.fit_results['is_detection']
        elif pixel_type == "Next Upper Limit":
            detection_map = self.fit_results['is_upper_limit']
        elif pixel_type == "Next Failed Fit":
            detection_map = ~self.fit_results['success']
        else:
            return
        
        # Search starting from next pixel
        found = False
        for row in range(current_row, self.n_y):
            for col in range(current_col if row == current_row else 0, self.n_x):
                if row == current_row and col <= current_col:
                    continue
                if detection_map[row, col]:
                    self.row_spinbox.setValue(row)
                    self.col_spinbox.setValue(col)
                    found = True
                    break
            if found:
                break
        
        if not found:
            QMessageBox.information(self, "Not Found", 
                f"No more '{pixel_type}' pixels found after current position.")
        
        self.jump_combo.setCurrentIndex(0)  # Reset combo
    
    def _plot_pixel(self, row: int, col: int):
        """
        Plot the Gaussian fit for a specific pixel.
        
        Shows:
        - Spectrum data points
        - Baseline
        - Fitted Gaussian
        - Residuals
        - Quality metrics
        """
        # Extract spectrum for this pixel
        y_data = self.cube[self.wave_mask, row, col]
        
        # Get fit results
        amplitude = self.fit_results['amplitude'][row, col]
        center = self.fit_results['center'][row, col]
        sigma = self.fit_results['sigma'][row, col]
        area = self.fit_results['area'][row, col]
        r_squared = self.fit_results['r_squared'][row, col]
        success = self.fit_results['success'][row, col]
        is_detection = self.fit_results['is_detection'][row, col]
        is_upper_limit = self.fit_results['is_upper_limit'][row, col]
        
        # Calculate baseline from endpoints
        y1 = y_data[0]
        y2 = y_data[-1]
        baseline = calculate_baseline(self.x_data[0], y1, self.x_data[-1], y2, self.x_data)
        y_corrected = y_data - baseline
        
        # Generate fitted curve
        if success and not np.isnan(amplitude):
            y_fit = gaussian(self.x_data, amplitude, center, sigma)
            residuals = y_corrected - y_fit
        else:
            y_fit = np.full_like(self.x_data, np.nan)
            residuals = np.full_like(y_corrected, np.nan)
        
        # Clear figure
        self.fig.clear()
        
        # Top subplot: Data, baseline, and fit
        ax_data = self.fig.add_subplot(2, 1, 1)
        ax_data.plot(self.x_data, y_data, 'o-', label='Data', linewidth=1.5, markersize=4)
        ax_data.plot(self.x_data, baseline, '--', label='Baseline', linewidth=2, alpha=0.7)
        if success and not np.isnan(amplitude):
            ax_data.plot(self.x_data, baseline + y_fit, '-', label='Gaussian Fit', 
                        linewidth=2, color='red', alpha=0.8)
        ax_data.set_ylabel('Flux')
        ax_data.set_title(f'Pixel [{row}, {col}] - Spectral Fit')
        ax_data.legend(loc='best')
        ax_data.grid(True, alpha=0.3)
        
        # Bottom subplot: Baseline-corrected data and residuals
        ax_residuals = self.fig.add_subplot(2, 1, 2)
        ax_residuals.plot(self.x_data, y_corrected, 'o-', label='Baseline-corrected Data', 
                         linewidth=1.5, markersize=4)
        if success and not np.isnan(amplitude):
            ax_residuals.plot(self.x_data, y_fit, '-', label='Gaussian Model', 
                            linewidth=2, color='red', alpha=0.8)
            ax_residuals.plot(self.x_data, residuals, 's', label='Residuals', 
                            markersize=3, alpha=0.6)
            ax_residuals.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        ax_residuals.set_xlabel('Wavelength')
        ax_residuals.set_ylabel('Flux')
        ax_residuals.set_title('Baseline-Subtracted Fit and Residuals')
        ax_residuals.legend(loc='best')
        ax_residuals.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Compute local R² from baseline-corrected data for this pixel
        if success and np.any(np.isfinite(y_fit)):
            ss_res = np.nansum((y_corrected - y_fit) ** 2)
            ss_tot = np.nansum((y_corrected - np.nanmean(y_corrected)) ** 2)
            r_squared_local = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        else:
            r_squared_local = np.nan

        # Update info label with fit details
        info_text = self._format_fit_info(row, col, amplitude, center, sigma, 
                                         area, r_squared_local, success, is_detection, 
                                         is_upper_limit, y_data, y_corrected, 
                                         residuals)
        self.info_label.setText(info_text)
    
    def _format_fit_info(self, row, col, amplitude, center, sigma, area, r_squared, 
                        success, is_detection, is_upper_limit, y_data, y_corrected, 
                        residuals) -> str:
        """Format fit information as readable text"""
        
        # Determine fit status
        if not success and np.isnan(amplitude):
            status = "❌ FAILED FIT"
        elif is_detection:
            status = f"✅ DETECTION (>{self.fac_above_stdres}σ)"
        elif is_upper_limit:
            status = f"⚠️  UPPER LIMIT (noise level, <{self.fac_above_stdres}σ)"
        else:
            status = "⚠️  UNCERTAIN"
        
        # Build info string
        lines = [
            f"{'='*80}",
            f"Pixel Location: [{row:3d}, {col:3d}]  |  Status: {status}",
            f"{'='*80}",
            "",
            f"Fit Parameters:",
            f"  Center (λ):        {center:12.6f}  (wavelength units)",
            f"  Amplitude (A):     {amplitude:12.6e}  (peak height)",
            f"  Sigma (σ):         {sigma:12.6e}  (Gaussian width)",
            f"  FWHM:              {2.355*sigma:12.6e}  (full width at half max)",
            f"  Area (AUC):        {area:12.6e}  (integrated flux)",
            "",
            f"Fit Quality:",
            f"  R² (goodness):     {r_squared:12.6f}  (1.0 = perfect)",
            f"  Data points:       {len(y_data):6d}",
            f"  Data range:        [{np.nanmin(y_data):12.6e}, {np.nanmax(y_data):12.6e}]",
        ]
        
        if np.any(np.isfinite(residuals)):
            mean_residual = np.nanmean(residuals)
            std_residual = np.nanstd(residuals)
            lines.extend([
                f"  Residuals (mean):  {mean_residual:12.6e}  (should be ~0)",
                f"  Residuals (std):   {std_residual:12.6e}  (fit scatter)",
            ])
        
        lines.extend([
            f"  Baseline-corr:     [{np.nanmin(y_corrected):12.6e}, {np.nanmax(y_corrected):12.6e}]",
            f"{'='*80}",
        ])
        
        return "\n".join(lines)
