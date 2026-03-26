import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMessageBox, QProgressDialog, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from .co_flux_dialogs import COFluxMapWorker, COFluxMapResultsDialog
from .co_flux_extractor import fit_continuum_median_peaks

class GuiCOFluxMixin:
    """
    Mixin class handling all CO Flux Map extraction logic for the CubeExplorer.
    """

    def _extract_co_flux_map(self):
        """
        Extract CO rovibrational flux from entire cube.
        Includes validation checks and preview before processing.
        """
        if not hasattr(self, 'cube_data') or self.cube_data is None or self.cube_data.cube is None:
            QMessageBox.warning(self, "No Data", "Load a FITS cube first")
            return
        
        if not hasattr(self, 'CO_lines') or self.CO_lines is None:
            QMessageBox.warning(self, "Missing Catalog", "CO line catalog not loaded")
            return
        
        if not hasattr(self, 'contam_lines'):
            self.contam_lines = None
        
        reply = QMessageBox.question(
            self,
            "Automatic Continuum Required",
            "CO Flux extraction requires automatic continuum subtraction.\n\n"
            "Did you already run:\n"
            "  Continuum Tools > Automatic Continuum > Continuum Line?\n\n"
            "If NOT, please do that first before proceeding.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        
        try:
            wave_min = float(self.min_wave_input.text())
            wave_max = float(self.max_wave_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Wavelength", "Enter valid min/max wavelength values.")
            return
        
        if wave_min >= wave_max:
            QMessageBox.warning(self, "Invalid Range", "Min wavelength must be less than Max.")
            return
        
        if hasattr(self, 'co_flux_result_cache') and hasattr(self, 'co_flux_params_cache'):
            params_match = (
                self.co_flux_params_cache.get('wave_min') == wave_min and
                self.co_flux_params_cache.get('wave_max') == wave_max and
                self.co_flux_params_cache.get('subtract_continuum') == (hasattr(self, 'auto_continuum_subtracted') and self.auto_continuum_subtracted)
            )
            hitran_match = (
                hasattr(self, 'co_flux_hitran_cache') and
                len(self.co_flux_hitran_cache) == len(self.hitran_lines)
            )
            
            if params_match and hitran_match:
                reply = QMessageBox.question(
                    self, "Cached Results Available",
                    "Results from the last extraction with these parameters are cached.\n\n"
                    "Click 'Yes' to reopen the cached results (instant)\n"
                    "Click 'No' to re-extract with current data",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    results_dialog = COFluxMapResultsDialog(
                        self.co_flux_result_cache['flux_map'],
                        self.co_flux_result_cache['detection_map'],
                        header=self.co_flux_result_cache.get('header'),
                        wave_range=self.co_flux_result_cache.get('wave_range'),
                        wavelength=getattr(self, 'co_flux_wavelength', None),
                        cube_data=self.cube_data,
                        CO_catalog=getattr(self, 'co_flux_catalog', None),
                        contam_catalog=getattr(self, 'co_flux_contam_catalog', None),
                        subtract_continuum=getattr(self, 'co_flux_subtract_continuum', False),
                        parent=self,
                        excluded_wavelengths=getattr(self, 'co_flux_excluded_wavelengths', set()),
                        flux_unit=self.co_flux_result_cache.get('flux_unit')
                    )
                    results_dialog.exec_()
                    return
        
        if not hasattr(self, 'hitran_lines') or not self.hitran_lines:
            QMessageBox.warning(self, "No HITRAN Data", "Click the HITRAN button first to fetch HITRAN lines for the wavelength range.")
            return
        
        hitran_in_range = [line for line in self.hitran_lines if wave_min <= float(line.get('wavelength', 0)) <= wave_max]
        
        if not hitran_in_range:
            QMessageBox.warning(self, "No HITRAN Lines", f"No HITRAN lines found in wavelength range {wave_min:.4f} - {wave_max:.4f} μm")
            return
        
        filtered_lines = self._filter_co_flux_lines(hitran_in_range)
        if not filtered_lines:
            QMessageBox.warning(self, "No Lines", "No valid lines after filtering.")
            return
        
        cube_shape = self.cube_data.cube.shape
        center_y, center_x = cube_shape[1] // 2, cube_shape[2] // 2
        
        spectrum_flux = self.cube_data.get_pixel_spectrum(center_x, center_y)
        spectrum_wave = self.cube_data.wavelength.value if hasattr(self.cube_data.wavelength, 'unit') else self.cube_data.wavelength
        
        confirmation = self._show_co_flux_confirmation(filtered_lines, spectrum_wave, spectrum_flux, center_x, center_y)
        if confirmation != QMessageBox.Yes:
            return
        
        try: n_jobs = min(os.cpu_count() or 8, 8)
        except: n_jobs = 8
        
        if hasattr(self, 'hitran_lines') and self.hitran_lines:
            hitran_df = pd.DataFrame(self.hitran_lines)
            if 'wavelength' in hitran_df.columns and 'line_wave_um' not in hitran_df.columns:
                hitran_df = hitran_df.rename(columns={'wavelength': 'line_wave_um'})
            # CRITICAL: Add species column if missing - all HITRAN lines are CO (12C16O unless otherwise specified)
            if 'species' not in hitran_df.columns:
                hitran_df['species'] = '12C16O'  # Mark all lines as 12C16O (main isotopologue)
            use_catalog = hitran_df
        else:
            use_catalog = self.CO_lines
        
        self.co_flux_wavelength = self.cube_data.wavelength
        self.co_flux_catalog = use_catalog
        # Only use contaminating lines if they exist and have data
        self.co_flux_contam_catalog = self.contam_lines if (hasattr(self, 'contam_lines') and self.contam_lines is not None and len(self.contam_lines) > 0) else None
        self.co_flux_wave_min = wave_min
        self.co_flux_wave_max = wave_max
        self.co_flux_subtract_continuum = True 
        
        self.co_flux_worker = COFluxMapWorker(
            self.cube_data, use_catalog, self.co_flux_contam_catalog, n_jobs=n_jobs,
            subtract_continuum=True, continuum_poly_deg=3,
            wave_min=wave_min, wave_max=wave_max,
            excluded_wavelengths=getattr(self, 'co_flux_excluded_wavelengths', set())
        )
        
        self.co_flux_progress = QProgressDialog("Extracting CO flux from all pixels...", "Cancel", 0, 0, self)
        self.co_flux_progress.setWindowModality(Qt.WindowModal)
        self.co_flux_progress.canceled.connect(self._cancel_co_flux_extraction)
        
        self.co_flux_worker.progress.connect(self._update_co_flux_progress)
        self.co_flux_worker.finished.connect(self._on_co_flux_map_complete)
        self.co_flux_worker.error.connect(self._on_co_flux_map_error)
        self.co_flux_worker.start()
    
    def _filter_co_flux_lines(self, lines):
        filtered = []
        distance_threshold = 0.002
        sorted_lines = sorted(lines, key=lambda x: float(x.get('wavelength', 0)))
        skip_indices = set()
        
        for i, line in enumerate(sorted_lines):
            if i in skip_indices: continue
            
            wave = float(line.get('wavelength', 0))
            peak = line.get('a_coeff') or line.get('intensity')
            if not peak or peak == 0: continue
            
            neighbors = []
            for j in range(i + 1, len(sorted_lines)):
                neighbor_wave = float(sorted_lines[j].get('wavelength', 0))
                if abs(neighbor_wave - wave) > distance_threshold: break
                neighbor_peak = sorted_lines[j].get('a_coeff') or sorted_lines[j].get('intensity')
                if neighbor_peak: neighbors.append((j, neighbor_peak))
            
            is_highest = True
            if neighbors:
                max_neighbor_peak = max(n[1] for n in neighbors)
                if peak < max_neighbor_peak:
                    is_highest = False
                    skip_indices.update(n[0] for n in neighbors)
                else:
                    skip_indices.update(n[0] for n in neighbors)
            
            if is_highest: filtered.append(line)
        return filtered
    
    def _show_co_flux_confirmation(self, lines, spectrum_wave, spectrum_flux, pixel_x, pixel_y):
        use_continuum_subtraction = hasattr(self, 'auto_continuum_subtracted') and self.auto_continuum_subtracted
        preview_flux = spectrum_flux.copy()
        
        if use_continuum_subtraction:
            try:
                continuum = fit_continuum_median_peaks(spectrum_wave, preview_flux)
                if continuum is not None and not np.all(np.isnan(continuum)):
                    preview_flux = preview_flux - continuum
            except: pass
        
        dlg = QDialog(self)
        dlg.setWindowTitle("CO Flux Map Preview")
        dlg.setGeometry(100, 100, 1000, 600)
        layout = QVBoxLayout()
        
        title = QLabel(f"<b>CO Flux Map Preview</b><br>Pixel ({pixel_x}, {pixel_y}) | {len(lines)} filtered lines")
        layout.addWidget(title)
        
        fig = Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        spectrum_label = 'Continuum-Subtracted Spectrum' if use_continuum_subtraction else 'Spectrum'
        ax.plot(spectrum_wave, preview_flux, 'k-', linewidth=1.2, label=spectrum_label, alpha=0.8)
        
        wave_min, wave_max = float(self.min_wave_input.text()), float(self.max_wave_input.text())
        for line in lines:
            wave = float(line.get('wavelength', 0))
            if wave_min <= wave <= wave_max:
                ax.axvline(wave, color='red', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Flux')
        ax.set_title(f'CO Line Positions ({len(lines)} lines)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        canvas = FigureCanvas(fig)
        plot_layout = QVBoxLayout()
        toolbar = NavigationToolbar(canvas, dlg)
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(canvas)
        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)
        layout.addWidget(plot_widget)
        
        info_text = QLabel("<b>Filtered Lines Summary:</b><br>Proceed with extraction?")
        layout.addWidget(info_text)
        
        btn_layout = QHBoxLayout()
        yes_btn, no_btn = QPushButton("YES - Extract Map"), QPushButton("NO - Cancel")
        yes_btn.clicked.connect(dlg.accept)
        no_btn.clicked.connect(dlg.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(yes_btn)
        btn_layout.addWidget(no_btn)
        layout.addLayout(btn_layout)
        dlg.setLayout(layout)
        
        return QMessageBox.Yes if dlg.exec_() == QDialog.Accepted else QMessageBox.No
    
    def _update_co_flux_progress(self, current, total):
        self.co_flux_progress.setMaximum(total)
        self.co_flux_progress.setValue(current)
        self.co_flux_progress.setLabelText(f"Extracting CO flux: {current}/{total} pixels")
    
    def _on_co_flux_map_complete(self, result):
        self.co_flux_progress.close()
        self.co_flux_result_cache = result
        self.co_flux_hitran_cache = self.hitran_lines.copy() if hasattr(self, 'hitran_lines') else []
        self.co_flux_params_cache = {
            'wave_min': float(self.min_wave_input.text()),
            'wave_max': float(self.max_wave_input.text()),
            'subtract_continuum': hasattr(self, 'auto_continuum_subtracted') and self.auto_continuum_subtracted,
        }
        
        results_dialog = COFluxMapResultsDialog(
            result['flux_map'], result['detection_map'], header=result.get('header'),
            wave_range=result.get('wave_range'), wavelength=getattr(self, 'co_flux_wavelength', None),
            cube_data=self.cube_data, CO_catalog=getattr(self, 'co_flux_catalog', None),
            contam_catalog=getattr(self, 'co_flux_contam_catalog', None),
            subtract_continuum=getattr(self, 'co_flux_subtract_continuum', False),
            parent=self, excluded_wavelengths=getattr(self, 'co_flux_excluded_wavelengths', set()),
            flux_unit=result.get('flux_unit')
        )
        results_dialog.exec_()
    
    def _on_co_flux_map_error(self, error_msg):
        self.co_flux_progress.close()
        if "cancelled" not in error_msg.lower(): QMessageBox.critical(self, "Extraction Error", error_msg)
        else: QMessageBox.information(self, "Extraction Cancelled", "CO flux extraction was cancelled")
    
    def _cancel_co_flux_extraction(self):
        if hasattr(self, 'co_flux_worker') and self.co_flux_worker:
            self.co_flux_worker.request_cancel()
            self.co_flux_progress.setLabelText("Cancelling extraction...")
    
    def _show_co_flux_map_results(self, result):
        pass