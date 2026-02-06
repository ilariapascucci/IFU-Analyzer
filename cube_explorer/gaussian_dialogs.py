"""
Gaussian Fitting Dialog and Worker Classes

Contains QThread workers and dialog classes for Gaussian map fitting
to keep the main GUI file cleaner and more maintainable.
"""

import numpy as np
import os
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
    
    def __init__(self, cube, wavelength, lam_min, lam_max, n_jobs=8, fac_above_stdres=3.0,
                 noise_min=None, noise_max=None, header=None, wave_unit=None):
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
    
    def run(self):
        """Execute fitting in background thread"""
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
                wave_unit=self.wave_unit
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
    
    def __init__(self, parent, results, cube_data, gauss_left_x, gauss_right_x, wavelength=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.results = results
        self.cube_data = cube_data
        self.gauss_left_x = gauss_left_x
        self.gauss_right_x = gauss_right_x
        self.wavelength = wavelength
        self._export_map_btn = None
        
        self.setWindowTitle("Gaussian Parameter Maps")
        self.resize(800, 700)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize dialog UI components"""
        layout = QVBoxLayout()
        
        # Info label with detection stats
        n_success = np.sum(self.results['success'])
        n_detections = np.sum(self.results.get('is_detection', np.zeros_like(self.results['success'])))
        n_upper_limits = np.sum(self.results.get('is_upper_limit', np.zeros_like(self.results['success'])))
        n_total = self.results['success'].size
        info_label = QLabel(
            f"Successfully fit {n_success:,} / {n_total:,} pixels "
            f"({100*n_success/n_total:.1f}%)\n"
            f"Detections: {n_detections:,} | Upper limits: {n_upper_limits:,}"
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
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(toolbar)
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
        export_btn = QPushButton("Export Current Parameter")
        export_btn.clicked.connect(self._export_current_parameter)
        button_layout.addWidget(export_btn)
        
        # Pixel diagnostics button
        diag_btn = QPushButton("Pixel by Pixel")
        diag_btn.clicked.connect(self._open_pixel_diagnostics)
        button_layout.addWidget(diag_btn)
        
        button_layout.addStretch()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
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
            default_cmap = 'viridis'
        elif param_name == "Amplitude":
            data = self.results['amplitude']
            title = f"Gaussian Peak Amplitude {wave_range}"
            default_cmap = 'plasma'
        elif param_name == "Center":
            data = self.results['center']
            title = f"Gaussian Center Position {wave_range}"
            default_cmap = 'twilight'  # Better for showing small variations
        elif param_name == "Sigma":
            data = self.results['sigma']
            title = f"Gaussian Width (σ) {wave_range}"
            default_cmap = 'cividis'
        elif param_name == "FWHM":
            data = self.results['fwhm']
            title = f"Full Width at Half Maximum {wave_range}"
            default_cmap = 'cividis'
        elif param_name == "χ² (fit quality)":
            data = self.results['r_squared']
            title = f"Reduced χ² (≈1 is good fit) {wave_range}"
            default_cmap = 'RdYlGn_r'
        elif param_name == "Success Mask":
            data = self.results['success'].astype(float)
            title = f"Successful Fits (1=success, 0=failed) {wave_range}"
            default_cmap = 'gray'
        elif param_name == "Residuals":
            self._handle_residuals_display(ax)
            return
        
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
        """Export the currently displayed parameter as a 3D FITS map"""
        param_name = self.param_combo.currentText()
        
        # Generate base filename from original cube filename
        if hasattr(self.cube_data, 'filename') and self.cube_data.filename:
            base_name = os.path.splitext(os.path.basename(self.cube_data.filename))[0]
        else:
            base_name = "gaussian"
        
        # Map parameter names to data and default filenames
        param_map = {
            "Area (AUC)": (self.results['area'], f"{base_name}_gaussian_area_map.fits"),
            "Amplitude": (self.results['amplitude'], f"{base_name}_gaussian_amplitude_map.fits"),
            "Center": (self.results['center'], f"{base_name}_gaussian_center_map.fits"),
            "Sigma": (self.results['sigma'], f"{base_name}_gaussian_sigma_map.fits"),
            "FWHM": (self.results['fwhm'], f"{base_name}_gaussian_fwhm_map.fits"),
            "χ² (fit quality)": (self.results['r_squared'], f"{base_name}_gaussian_chi2_map.fits"),
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
        
        try:
            from astropy.io import fits
            
            # Reshape 2D data to 3D: (n_y, n_x) -> (1, n_y, n_x)
            data_3d = data[np.newaxis, :, :]
            
            # Start with a copy of the original header (like residuals export does)
            if self.cube_data.header is not None:
                out_header = self.cube_data.header.copy()
            else:
                out_header = fits.Header()
            
            # Get instrument info
            instr = out_header.get('INSTRUME', '').strip().upper() or self.cube_data.instrument or 'UNKNOWN'
            teles = out_header.get('TELESCOP', '').strip().upper() or 'UNKNOWN'
            
            # Update header with detected values
            if instr != 'UNKNOWN':
                out_header['INSTRUME'] = instr
            if teles != 'UNKNOWN':
                out_header['TELESCOP'] = teles
            
            # Add metadata about the Gaussian fit
            lam_min = min(self.gauss_left_x, self.gauss_right_x)
            lam_max = max(self.gauss_left_x, self.gauss_right_x)
            
            # Determine unit for BUNIT based on parameter type and whether conversion was applied
            if param_name == "Area (AUC)":
                # Check if unit conversion was applied (header has CDELT1/CDELT2)
                from .gaussian_fitter import get_pixel_solid_angle
                solid_angle = get_pixel_solid_angle(self.cube_data.header)
                if solid_angle is not None:
                    bunit = "erg/s/cm2"
                    out_header['COMMENT'] = 'Area converted to erg/s/cm² using pixel solid angle from CDELT1/CDELT2'
                else:
                    bunit = "MJy/sr*um"
                    out_header['COMMENT'] = 'Area in native units (MJy/sr·μm) - no pixel scale available for conversion'
            else:
                bunit = param_name
            
            out_header['BUNIT'] = bunit
            out_header['OBJECT'] = (out_header.get('OBJECT', 'Unknown') + f' ({param_name})')
            out_header['PARAM'] = param_name
            out_header['LAMMIN'] = (lam_min, "Minimum wavelength of fit")
            out_header['LAMMAX'] = (lam_max, "Maximum wavelength of fit")
            out_header['NSUCC'] = (int(np.sum(self.results['success'])), "Number of successful fits")
            out_header['NTOTAL'] = (int(self.results['success'].size), "Total number of pixels")
            out_header['COMMENT'] = f'Gaussian fit parameter: {param_name}'
            
            # Update NAXIS for 3D map (1, n_y, n_x)
            out_header['NAXIS'] = 3
            out_header['NAXIS1'] = int(data_3d.shape[2])
            out_header['NAXIS2'] = int(data_3d.shape[1])
            out_header['NAXIS3'] = int(data_3d.shape[0])
            
            # Create HDU and write
            hdu = fits.PrimaryHDU(data=data_3d, header=out_header)
            hdul = fits.HDUList([hdu])
            hdul.writeto(out_path, overwrite=True)
            
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
    
    
    def _open_pixel_diagnostics(self):
        """Open interactive pixel diagnostics viewer"""
        if self.wavelength is None:
            QMessageBox.warning(self, "No Wavelength Data", 
                               "Wavelength array not available for diagnostics")
            return
        
        from .pixel_diagnostics import PixelDiagnosticsDialog
        
        fac_above_stdres = self.results.get('fac_above_stdres', 3.0)
        
        diag_dlg = PixelDiagnosticsDialog(
            self,
            self.cube_data.cube,
            self.wavelength,
            self.gauss_left_x,
            self.gauss_right_x,
            self.results,
            fac_above_stdres=fac_above_stdres
        )
        diag_dlg.exec_()

