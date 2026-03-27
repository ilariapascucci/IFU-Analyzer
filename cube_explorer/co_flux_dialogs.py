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
    
    def __init__(self, cube_data, CO_catalog, contam_catalog, n_jobs=4, subtract_continuum=False, continuum_poly_deg=3, wave_min=None, wave_max=None, excluded_wavelengths=None):
        super().__init__()
        self.cube_data = cube_data
        self.CO_catalog = CO_catalog
        self.contam_catalog = contam_catalog
        self.n_jobs = n_jobs
        self.subtract_continuum = subtract_continuum
        self.continuum_poly_deg = continuum_poly_deg
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.excluded_wavelengths = excluded_wavelengths
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
                wave_max=self.wave_max,
                excluded_wavelengths=self.excluded_wavelengths
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
    
    def __init__(self, flux_map, detection_map, header=None, wave_range=None, 
                 wavelength=None, cube_data=None, CO_catalog=None, contam_catalog=None,
                 subtract_continuum=False, parent=None, excluded_wavelengths=None, flux_unit=None):
        super().__init__(parent)
        self.flux_map = flux_map
        self.detection_map = detection_map
        self.header = header
        self.wave_range = wave_range
        self.wavelength = wavelength
        self.cube_data = cube_data
        self.CO_catalog = CO_catalog
        self.contam_catalog = contam_catalog
        self.subtract_continuum = subtract_continuum
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
            self.flux_unit_str = self.flux_unit_output if self.flux_unit_output else "MJy/sr um"
            self.bunit_str = self.flux_unit_output if self.flux_unit_output else "MJy/sr"
            if self.flux_unit_output == 'erg/s/cm2':
                self.flux_unit_base = u.erg / (u.s * u.cm**2)
                self.flux_unit = self.flux_unit_base
            else:
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
        self.param_combo.addItems(["CO Flux", "Detection Count"])
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
        """Toggle pixel selection mode on/off"""
        self.pixel_selection_enabled = self.pixel_select_btn.isChecked()
        if self.pixel_selection_enabled:
            self.pixel_select_btn.setText("✓ Pixel Select ON")
            self.pixel_select_btn.setStyleSheet("background-color: #90EE90; font-weight: bold;")
        else:
            self.pixel_select_btn.setText("✗ Pixel Select OFF")
            self.pixel_select_btn.setStyleSheet("background-color: #FFB6C6; font-weight: bold;")
    
    def _on_canvas_click(self, event):
        """Handle click on flux map to inspect pixel spectrum"""
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
        """Extract and display spectrum for a clicked pixel with fitted Gaussians"""
        try:
            from .co_flux_extractor import extract_co_flux_from_spectrum
            
            # Extract spectrum for this pixel
            spectrum = self.cube_data.get_pixel_spectrum(x, y)
            
            if spectrum is None or np.all(np.isnan(spectrum)):
                QMessageBox.warning(self, "No Data", f"Pixel ({x}, {y}) has no valid data")
                return
            
            # Re-extract CO flux for this pixel to get fit info
            if self.CO_catalog is not None:
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
                
                # Create canvas FIRST so figure has a canvas for events
                canvas = FigureCanvasQTAgg(fig)
                
                # Plot spectrum
                ax.plot(self.wavelength, spectrum, 'k-', label='Spectrum', linewidth=1.2, alpha=0.8)
                
                # Plot continuum if subtracted
                if self.subtract_continuum and result.get('mean_continuum'):
                    ax.axhline(result['mean_continuum'], color='orange', linestyle='--', label='Continuum', alpha=0.7)
                
                # Initialize line tracking dicts
                self.detected_lines_dict = {}  # Maps wavelength to line info
                self.line_artists = {}  # Maps wavelength to matplotlib line artist
                
                # Mark detected CO lines with disposition=1
                detected = result.get('detected_lines')
                if detected is not None:
                    try:
                        import pandas as pd
                        if isinstance(detected, pd.DataFrame):
                            # DEBUG: Check what columns and rows we have
                            print(f"DEBUG: detected_lines shape={detected.shape}, columns={detected.columns.tolist()}")
                            
                            # Filter by disposition and wavelength range
                            if len(detected) > 0:
                                detected_co = detected[
                                    (detected['disposition'] == 1) & 
                                    (detected['line'] >= self.wave_range[0]) & 
                                    (detected['line'] <= self.wave_range[1])
                                ]
                                print(f"DEBUG: After filtering: {len(detected_co)} lines out of {len(detected)}")
                                
                                for idx, line_row in detected_co.iterrows():
                                    line_wave = float(line_row['line'])
                                    self.detected_lines_dict[line_wave] = line_row
                                    
                                    # Determine color based on exclusion status
                                    is_excluded = any(abs(line_wave - ew) < 0.001 for ew in self.excluded_wavelengths)
                                    color = 'gray' if is_excluded else 'red'
                                    alpha = 0.3 if is_excluded else 0.5
                                    
                                    # Create line with picker enabled
                                    line_artist = ax.axvline(line_wave, color=color, alpha=alpha, linewidth=1.5, 
                                                            label='CO Lines' if idx == detected_co.index[0] else '', picker=5)
                                    print(f"DEBUG: Created line at {line_wave:.6f} μm, artist={line_artist}, picker enabled")
                                    self.line_artists[line_wave] = line_artist
                    except Exception as e:
                        import traceback
                        # DEBUG: Print error
                        print(f"DEBUG: Error marking lines: {e}\n{traceback.format_exc()}")
                        pass  # Silently skip line marking if it fails
                
                ax.set_xlabel('Wavelength (μm)')
                ax.set_ylabel(f'Flux ({self.bunit_str})')
                ax.set_title(f'Pixel ({x}, {y}) Spectrum', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Draw the figure NOW so all lines exist before connecting pick event
                canvas.draw()
                
                # Canvas already created above for pick event support
                toolbar = NavigationToolbar2QT(canvas, dlg)
                plot_layout = QVBoxLayout()
                plot_layout.addWidget(toolbar)
                plot_layout.addWidget(canvas)
                plot_widget = QWidget()
                plot_widget.setLayout(plot_layout)
                layout.addWidget(plot_widget)
                
                # IMPORTANT: Set up click event handler AFTER drawing so all lines exist
                def on_click(event):
                    if event.inaxes != ax:
                        return
                    
                    click_wavelength = event.xdata
                    print(f"DEBUG: Click at wavelength {click_wavelength:.6f} μm")
                    
                    # Find nearest line within tolerance
                    tolerance = 0.01  # μm
                    best_match = None
                    best_distance = tolerance
                    
                    for line_wave in self.line_artists.keys():
                        distance = abs(line_wave - click_wavelength)
                        if distance < best_distance:
                            best_distance = distance
                            best_match = line_wave
                    
                    if best_match is not None:
                        print(f"DEBUG: Found matching line at {best_match:.6f} μm (distance: {best_distance:.6f} μm)")
                        self._toggle_line_exclusion(best_match, fig, canvas)
                    else:
                        print(f"DEBUG: No line found within {tolerance} μm tolerance")
                
                print(f"DEBUG: Connecting click_event handler. {len(self.line_artists)} lines available for clicking")
                canvas.mpl_connect('button_press_event', on_click)
                
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
    
    def _toggle_line_exclusion(self, line_wave, fig, canvas):
        """Toggle exclusion status of a CO line and update visualization"""
        # Toggle the line in excluded set
        if any(abs(line_wave - ew) < 0.001 for ew in self.excluded_wavelengths):
            # Line is currently excluded, remove it
            self.excluded_wavelengths = {ew for ew in self.excluded_wavelengths if abs(ew - line_wave) >= 0.001}
            print(f"  Re-enabled CO line at {line_wave:.6f} μm")
        else:
            # Line is not excluded, add it
            self.excluded_wavelengths.add(line_wave)
            print(f"  Excluded CO line at {line_wave:.6f} μm")
        
        # Update line color and alpha
        if line_wave in self.line_artists:
            is_excluded = any(abs(line_wave - ew) < 0.001 for ew in self.excluded_wavelengths)
            line = self.line_artists[line_wave]
            line.set_color('gray' if is_excluded else 'red')
            line.set_alpha(0.3 if is_excluded else 0.5)
        
        # Redraw
        canvas.draw_idle()
    
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
                hdu.header['DESC'] = 'CO rovibrational flux: Integrated area under CO line gaussians'
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
        dlg.resize(500, 350)
        
        layout = QVBoxLayout()
        
        stats = "CO FLUX MAP STATISTICS\n" + "="*50 + "\n\n"
        
        # Flux statistics with correct units
        flux_valid = self.flux_map[~np.isnan(self.flux_map)]
        if len(flux_valid) > 0:
            stats += f"FLUX ({self.flux_unit_str}):\n"
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
            stats += f"  Min/Max: {np.min(detection_valid)} / {np.max(detection_valid)}\n"
        
        label = QLabel(stats)
        label.setStyleSheet("font-family: monospace; white-space: pre;")
        label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        layout.addWidget(label)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        
        dlg.setLayout(layout)
        dlg.exec_()
