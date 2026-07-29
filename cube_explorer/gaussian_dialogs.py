"""
Gaussian Fitting Dialog and Worker Classes

Contains QThread workers and dialog classes for Gaussian map fitting
to keep the main GUI file cleaner and more maintainable.
"""

import numpy as np
import os
import threading
from .unit_converter import gaussian_map_unit_labels
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem
)
from PyQt5.QtGui import QColor, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


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

        Does not block. Call wait() on this QThread afterward if you need the caller
        to pause until the thread has actually exited.
        """
        self._cancel_event.set()

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
                wave_unit=self.wave_unit,
                flux_unit=self.flux_unit,
                lambda_0=self.lambda_0,
                progress_callback=self._emit_progress,
                cancel_event=self._cancel_event,
            )
            # results is None when cancel() was called before completion. Nothing is
            # listening by then (the window that owned this worker is closing), so
            # there is nothing useful to emit -- just let the thread end.
            if results is not None:
                self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


    def _emit_progress(self, current, total):
        """Emit progress signal"""
        self.progress.emit(current, total)


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
        self._current_unit_label = None  # Store the actual displayed unit label
        
        # Extract units from data
        self.wave_unit = self._get_wavelength_unit()
        self.flux_unit = self._get_flux_unit()
        
        self.setWindowTitle("Gaussian Parameter Maps")
        self.resize(800, 700)
        
        self._setup_ui()
    
    def _get_wavelength_unit(self):
        """Extract wavelength unit from cube_data or results"""
        # Prioritize cube_data since it has the actual unit used for fitting
        if hasattr(self.cube_data, 'wave_unit') and self.cube_data.wave_unit:
            return self.cube_data.wave_unit
        # Fall back to results if available
        if 'wavelength_unit' in self.results and self.results['wavelength_unit'] and self.results['wavelength_unit'] != 'unknown':
            return self.results['wavelength_unit']
        return 'um'  # Default fallback
    
    def _get_flux_unit(self):
        """Extract flux unit from cube_data"""
        if hasattr(self.cube_data, 'flux_unit') and self.cube_data.flux_unit:
            return self.cube_data.flux_unit
        return 'native'  # Default fallback
    
    def _get_parameter_units(self):
        """
        Determine unit strings for all parameters based on instrument/data.
        
        Returns dict with parameter -> (unit_label, colormap) pairs
        """
        # Ask unit_converter what the maps are in rather than pattern-matching BUNIT here.
        # It probes with the same conversions the fitter used, so the label cannot claim a
        # unit the numbers are not in -- which is how these maps ended up labelled
        # "[ΜM]" (from 'µm'.upper()) and "erg/(s*cm2*sr)" (an amplitude that is really
        # per-wavelength, with the steradian already divided out).
        header = getattr(self.cube_data, 'header', None)
        lam_mid = (min(self.gauss_left_x, self.gauss_right_x)
                   + max(self.gauss_left_x, self.gauss_right_x)) / 2.0
        labels = gaussian_map_unit_labels(
            self.flux_unit, self.wave_unit, header=header, wavelength=lam_mid)
        wave_display = labels['wave']
        amp_unit = labels['amplitude']
        area_unit = labels['area']

        return {
            'Area (AUC)': (f'Area [{area_unit}]', 'viridis'),
            'Amplitude': (f'Amplitude [{amp_unit}]', 'plasma'),
            'Center': (f'Center [{wave_display}]', 'twilight'),
            'Sigma': (f'Sigma [{wave_display}]', 'cividis'),
            'FWHM': (f'FWHM [{wave_display}]', 'cividis'),
            'chi2 (fit quality)': ('chi^2 [dimensionless]', 'RdYlGn_r'),
            'Success Mask': ('Success [1=yes, 0=no]', 'gray'),
        }
    
    def _setup_ui(self):
        """Initialize dialog UI components"""
        layout = QVBoxLayout()
        
        # Info label with detection stats
        n_success = np.sum(self.results['success'])
        n_detections = np.sum(self.results.get('is_detection', np.zeros_like(self.results['success'])))
        n_upper_limits = np.sum(self.results.get('is_upper_limit', np.zeros_like(self.results['success'])))
        n_total = self.results['success'].size
        global_fit = self.results.get('global_fit') or {}
        if global_fit:
            center_src = global_fit.get('center_source', 'total-spectrum')
            global_text = (
                f"Global 1D fit: A={global_fit.get('amplitude', float('nan')):.4e}, "
                f"mu={global_fit.get('center', float('nan')):.4f}, "
                f"sigma={global_fit.get('sigma', float('nan')):.4f} "
                f"(center from {center_src})"
            )
        else:
            global_text = "Global 1D fit: unavailable"

        info_label = QLabel(
            f"Successfully fit {n_success:,} / {n_total:,} pixels "
            f"({100*n_success/n_total:.1f}%)\n"
            f"Detections: {n_detections:,} | Upper limits: {n_upper_limits:,}\n"
            f"{global_text}"
        )
        layout.addWidget(info_label)
        
        # Parameter selector
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Parameter:"))
        self.param_combo = QComboBox()
        self.param_combo.addItems([
            "Area (AUC)", "Amplitude", "Center", "Sigma", 
            "FWHM", "chi2 (fit quality)", "Success Mask", "Residuals"
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
        export_btn = QPushButton("Save as FITS File")
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
        
        # Get dynamic units for all parameters
        param_units = self._get_parameter_units()
        
        # Select data based on parameter
        if param_name == "Area (AUC)":
            data = self.results['area']
            title = f"Integrated Area Under Gaussian (AUC) {wave_range}"
            unit_label, default_cmap = param_units['Area (AUC)']
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
        
        # Add statistics
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            stats_text = (
                f"Min: {np.nanmin(data):.4e}\n"
                f"Max: {np.nanmax(data):.4e}\n"
                f"Mean: {np.nanmean(data):.4e}\n"
                f"Median: {np.nanmedian(data):.4e}\n"
                f"Sum: {np.nansum(data):.4e}"
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
        
        # Handle Residuals separately (3D cube, already in correct shape)
        if param_name == "Residuals":
            if 'residuals_cube' not in self.results or self.results['residuals_cube'] is None:
                QMessageBox.warning(self, "No Data", "Residuals not available for export")
                return
            
            default_filename = f"{base_name}_gaussian_residuals_cube.fits"
            out_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Residuals Cube",
                default_filename,
                "FITS files (*.fits)"
            )
            
            if not out_path:
                return
            
            try:
                from astropy.io import fits
                residuals_data = self.results['residuals_cube']
                
                if self.cube_data.header is not None:
                    out_header = self.cube_data.header.copy()
                else:
                    out_header = fits.Header()
                
                # Update header
                instr = out_header.get('INSTRUME', '').strip().upper() or self.cube_data.instrument or 'UNKNOWN'
                teles = out_header.get('TELESCOP', '').strip().upper() or 'UNKNOWN'
                if instr != 'UNKNOWN':
                    out_header['INSTRUME'] = instr
                if teles != 'UNKNOWN':
                    out_header['TELESCOP'] = teles
                
                out_header['BUNIT'] = 'Normalized residuals'
                out_header['OBJECT'] = (out_header.get('OBJECT', 'Unknown') + ' (Residuals)')
                out_header['COMMENT'] = 'Gaussian fit residuals: (data - fit) / noise'
                
                lam_min = min(self.gauss_left_x, self.gauss_right_x)
                lam_max = max(self.gauss_left_x, self.gauss_right_x)
                out_header['LAMMIN'] = (lam_min, "Minimum wavelength of fit")
                out_header['LAMMAX'] = (lam_max, "Maximum wavelength of fit")
                
                # Update NAXIS
                out_header['NAXIS'] = 2
                out_header['NAXIS1'] = int(residuals_data.shape[2])
                out_header['NAXIS2'] = int(residuals_data.shape[1])
                
                hdu = fits.PrimaryHDU(data=residuals_data, header=out_header)
                hdul = fits.HDUList([hdu])
                hdul.writeto(out_path, overwrite=True)
                
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
            return
        
        # Map parameter names to data and default filenames (2D maps)
        param_map = {
            "Area (AUC)": (self.results['area'], f"{base_name}_gaussian_area_map.fits"),
            "Amplitude": (self.results['amplitude'], f"{base_name}_gaussian_amplitude_map.fits"),
            "Center": (self.results['center'], f"{base_name}_gaussian_center_map.fits"),
            "Sigma": (self.results['sigma'], f"{base_name}_gaussian_sigma_map.fits"),
            "FWHM": (self.results['fwhm'], f"{base_name}_gaussian_fwhm_map.fits"),
            "chi2 (fit quality)": (self.results['r_squared'], f"{base_name}_gaussian_chi2_map.fits"),
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
            
            # Use the EXACT unit label that's displayed on the colorbar
            # This ensures consistency between visualization and FITS export
            if self._current_unit_label:
                # Extract unit from "Parameter [unit]" format
                if '[' in self._current_unit_label and ']' in self._current_unit_label:
                    bunit = self._current_unit_label[self._current_unit_label.index('[')+1:self._current_unit_label.index(']')]
                else:
                    bunit = self._current_unit_label
                out_header['COMMENT'] = f'{param_name}: {bunit}'
            else:
                # Fallback if no unit label stored
                bunit = param_name
                out_header['COMMENT'] = f'Gaussian fit parameter: {param_name}'
            
            out_header['BUNIT'] = bunit
            out_header['OBJECT'] = (out_header.get('OBJECT', 'Unknown') + f' ({param_name})')
            out_header['PARAM'] = param_name
            out_header['LAMMIN'] = (lam_min, "Minimum wavelength of fit")
            out_header['LAMMAX'] = (lam_max, "Maximum wavelength of fit")
            out_header['NSUCC'] = (int(np.sum(self.results['success'])), "Number of successful fits")
            out_header['NTOTAL'] = (int(self.results['success'].size), "Total number of pixels")
            out_header['COMMENT'] = f'Gaussian fit parameter: {param_name}'
            
            out_header['NAXIS'] = 2
            out_header['NAXIS1'] = int(data.shape[1])
            out_header['NAXIS2'] = int(data.shape[0])
            for key in list(out_header.keys()):
                if any(key.startswith(prefix) for prefix in ["NAXIS3", "WCSAXES", "CRPIX3", "CRVAL3", "CDELT3", "CTYPE3", "CUNIT3", "PC1_3", "PC2_3", "PC3_3", "PC3_1", "PC3_2"]):
                    del out_header[key]
            
            # Create HDU and write
            hdu = fits.PrimaryHDU(data=data, header=out_header)
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
        """Export results to CSV file"""
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

