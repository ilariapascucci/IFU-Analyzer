import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMessageBox, QProgressDialog, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from .flux_map_dialogs import FluxMapWorker, FluxMapResultsDialog
from .flux_map_extractor import fit_auto_continuum
from .continuum_dialog import ContinuumParametersDialog
from .species_dialog import SpeciesDialog
from .unit_converter import extract_units

class GuiFluxMapMixin:
    """
    Mixin class handling all Flux Map extraction logic for the CubeExplorer.
    """

    def _extract_flux_map(self):
        """
        Extract rovibrational flux from entire cube.
        Includes validation checks and preview before processing.
        """
        if not hasattr(self, 'cube_data') or self.cube_data is None or self.cube_data.cube is None:
            QMessageBox.warning(self, "No Data", "Load a FITS cube first")
            return
        
        if not hasattr(self, 'contam_lines'):
            self.contam_lines = None
        
        reply = QMessageBox.question(
            self,
            "Continuum Subtraction",
            "Flux extraction automatically fits and subtracts a continuum for every "
            "pixel, using default parameters that work well for typical spectra.\n\n"
            "Use custom continuum parameters instead?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.No
        )
        if reply == QMessageBox.Cancel:
            return
        if reply == QMessageBox.Yes:
            params_dialog = ContinuumParametersDialog(self)
            if params_dialog.exec_() != QDialog.Accepted:
                return
            self.flux_map_continuum_params = params_dialog.get_parameters()
        else:
            self.flux_map_continuum_params = None

        try:
            wave_min = float(self.min_wave_input.text())
            wave_max = float(self.max_wave_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Wavelength", "Enter valid min/max wavelength values.")
            return
        
        if wave_min >= wave_max:
            QMessageBox.warning(self, "Invalid Range", "Min wavelength must be less than Max.")
            return
        
        if hasattr(self, 'flux_map_result_cache') and hasattr(self, 'flux_map_params_cache'):
            params_match = (
                self.flux_map_params_cache.get('wave_min') == wave_min and
                self.flux_map_params_cache.get('wave_max') == wave_max and
                self.flux_map_params_cache.get('continuum_params') == self.flux_map_continuum_params
            )
            selection_match = (
                hasattr(self, 'flux_map_hitran_cache') and
                len(self.flux_map_hitran_cache) == len(getattr(self, 'plotted_lines', None) or [])
            )
            
            if params_match and selection_match:
                reply = QMessageBox.question(
                    self, "Cached Results Available",
                    "Results from the last extraction with these parameters are cached.\n\n"
                    "Click 'Yes' to reopen the cached results (instant)\n"
                    "Click 'No' to re-extract with current data",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    results_dialog = FluxMapResultsDialog(
                        self.flux_map_result_cache['flux_map'],
                        self.flux_map_result_cache['detection_map'],
                        header=self.flux_map_result_cache.get('header'),
                        wave_range=self.flux_map_result_cache.get('wave_range'),
                        wavelength=getattr(self, 'flux_map_wavelength', None),
                        cube_data=self.cube_data,
                        target_catalog=getattr(self, 'flux_map_catalog', None),
                        contam_catalog=getattr(self, 'flux_map_contam_catalog', None),
                        parent=self,
                        excluded_wavelengths=getattr(self, 'flux_map_excluded_wavelengths', set()),
                        flux_unit=self.flux_map_result_cache.get('flux_unit'),
                        continuum_params=self.flux_map_continuum_params,
                        contam_flux_map=self.flux_map_result_cache.get('contam_flux_map'),
                        contam_detection_map=self.flux_map_result_cache.get('contam_detection_map')
                    )
                    results_dialog.exec_()
                    return
        
        # Source of truth is self.plotted_lines -- exactly what the Lines dialog (species_dialog)
        # currently has checked, the same list the main spectrum plot and Multi-Gaussian already
        # use. Previously this read self.hitran_lines (the *entire* fetched/loaded catalog, e.g.
        # all 183 lines from a wide HITRAN query) minus flux_map_excluded_wavelengths (only ever
        # populated by individually clicking a line marker on the plot, never by the Lines
        # dialog's checkboxes) -- so deselecting lines in the Lines dialog had no effect here at
        # all, and a separate, unrelated proximity-clustering pass (_filter_co_flux_lines) is
        # what actually decided the count shown in the preview. Both are gone now: whatever's
        # checked in the Lines dialog is what gets fit, full stop, mirroring Multi-Gaussian.
        if not getattr(self, 'plotted_lines', None):
            QMessageBox.warning(self, "No Lines Selected", "Fetch/load lines and select them in the Lines dialog first.")
            return

        detection_rows = [r for r in self.plotted_lines if r.get('source') != 'Contaminant']
        contam_rows = [r for r in self.plotted_lines if r.get('source') == 'Contaminant']

        def _in_range(row):
            wave = SpeciesDialog.get_row_wavelength(row)
            return wave is not None and wave_min <= float(wave) <= wave_max

        filtered_lines = [r for r in detection_rows if _in_range(r)]
        if not filtered_lines:
            QMessageBox.warning(self, "No Lines Selected",
                                 f"No selected (checked) lines fall within {wave_min:.6f} - {wave_max:.6f} μm. "
                                 "Select lines in the Lines dialog first.")
            return
        
        cube_shape = self.cube_data.cube.shape
        spectrum_wave, _ = extract_units(self.cube_data.wavelength)

        # Preview the brightest pixel within the fit range rather than the cube's geometric
        # center, which may have little or no target-line signal
        wave_mask = (spectrum_wave >= wave_min) & (spectrum_wave <= wave_max)
        preview_y, preview_x = cube_shape[1] // 2, cube_shape[2] // 2
        if np.any(wave_mask):
            with np.errstate(all='ignore'):
                pixel_peak = np.nanmax(self.cube_data.cube[wave_mask, :, :], axis=0)
            if not np.all(np.isnan(pixel_peak)):
                preview_y, preview_x = np.unravel_index(np.nanargmax(pixel_peak), pixel_peak.shape)

        spectrum_flux = self.cube_data.get_pixel_spectrum(preview_x, preview_y)

        confirmation = self._show_flux_map_confirmation(filtered_lines, spectrum_wave, spectrum_flux, preview_x, preview_y)
        if confirmation != QMessageBox.Yes:
            return
        
        # Capped at 4, not higher: measured directly on this workload (many small,
        # independent per-pixel curve_fit calls) -- above 4 concurrent threads, throughput
        # gets *worse*, not better (BLAS-internal threading inside each curve_fit call
        # contends with the outer ThreadPoolExecutor threads for the same cores).
        try: n_jobs = min(os.cpu_count() or 4, 4)
        except Exception: n_jobs = 4
        
        # Same filtered_lines/contam_rows computed above (and already shown in the preview) --
        # converted to the 'line_wave_um'/'species' DataFrame shape find_lines()/fit_glines()
        # expect. No species override here: unlike the old hitran_df path, this never force-
        # labels every row 'CO' -- Flux Map fits whatever's actually selected, any species.
        def _rows_to_catalog(rows):
            if not rows:
                return pd.DataFrame(columns=['line_wave_um', 'species'])
            df = pd.DataFrame(rows)
            wave_col = 'lambda_micron' if 'lambda_micron' in df.columns else 'wavelength'
            return df.rename(columns={wave_col: 'line_wave_um'})

        use_catalog = _rows_to_catalog(filtered_lines)
        self.flux_map_wavelength = self.cube_data.wavelength
        self.flux_map_catalog = use_catalog
        contam_catalog_df = _rows_to_catalog(contam_rows)
        self.flux_map_contam_catalog = contam_catalog_df if len(contam_catalog_df) > 0 else None
        self.flux_map_wave_min = wave_min
        self.flux_map_wave_max = wave_max

        self.flux_map_worker = FluxMapWorker(
            self.cube_data, use_catalog, self.flux_map_contam_catalog, n_jobs=n_jobs,
            wave_min=wave_min, wave_max=wave_max,
            excluded_wavelengths=getattr(self, 'flux_map_excluded_wavelengths', set()),
            continuum_params=self.flux_map_continuum_params
        )
        
        self.flux_map_progress = QProgressDialog("Extracting flux from all pixels...", "Cancel", 0, 0, self)
        self.flux_map_progress.setWindowModality(Qt.WindowModal)
        self.flux_map_progress.canceled.connect(self._cancel_flux_map_extraction)
        
        self.flux_map_worker.progress.connect(self._update_flux_map_progress)
        self.flux_map_worker.finished.connect(self._on_flux_map_complete)
        self.flux_map_worker.error.connect(self._on_flux_map_error)
        self.flux_map_worker.start()
    
    def _show_flux_map_confirmation(self, lines, spectrum_wave, spectrum_flux, pixel_x, pixel_y):
        # Flux Map always continuum-subtracts every pixel (with default or custom
        # parameters, chosen in the popup before this preview), so the preview should too
        preview_flux = spectrum_flux.copy()
        continuum = None
        try:
            continuum = fit_auto_continuum(spectrum_wave, preview_flux, **(self.flux_map_continuum_params or {}))
            if continuum is not None and not np.all(np.isnan(continuum)):
                preview_flux = preview_flux - continuum
        except Exception: pass

        # Same species-label logic as FluxMapResultsDialog -- 'H2' if every selected line is
        # H2, 'H2+CO' if mixed, 'Target' as a generic fallback.
        species_vals = sorted({str(l.get('species', '')).strip() for l in lines if str(l.get('species', '')).strip()})
        species_label = ('+'.join(species_vals) if 0 < len(species_vals) <= 3
                          else (f'{len(species_vals)} species' if species_vals else 'Target'))

        dlg = QDialog(self)
        dlg.setWindowTitle(f"{species_label} Flux Map Preview")
        dlg.setGeometry(100, 100, 1000, 600)
        layout = QVBoxLayout()

        title = QLabel(f"<b>{species_label} Flux Map Preview</b><br>Brightest pixel in range: ({pixel_x}, {pixel_y}) | {len(lines)} filtered lines")
        layout.addWidget(title)

        fig = Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        # Raw spectrum + the fitted continuum drawn on top (same style as the pixel-inspect
        # dialog) -- lets you judge whether the continuum fit itself looks right before
        # committing to the full extraction, which the continuum-subtracted view alone can't show.
        ax.plot(spectrum_wave, spectrum_flux, 'k-', linewidth=1.2, label='Spectrum', alpha=0.8)
        if continuum is not None and not np.all(np.isnan(continuum)):
            ax.plot(spectrum_wave, continuum, color='orange', linestyle='--', linewidth=1.8,
                    label='Continuum', alpha=0.8)

        wave_min, wave_max = float(self.min_wave_input.text()), float(self.max_wave_input.text())
        for line in lines:
            # get_row_wavelength(), not a hardcoded 'wavelength' key: `lines` here is drawn
            # from self.plotted_lines, whose rows can use either 'wavelength' or 'lambda_micron'
            wave = SpeciesDialog.get_row_wavelength(line)
            if wave is not None and wave_min <= float(wave) <= wave_max:
                ax.axvline(float(wave), color='red', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Flux')
        ax.set_title(f'{species_label} Line Positions ({len(lines)} lines)')
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
        
        info_text = QLabel(
            "<b>Lines shown:</b> exactly what's currently checked in the Lines dialog, "
            "within this wavelength range -- open the Lines dialog to change the selection.<br>"
            "Proceed with extraction?"
        )
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
    
    def _update_flux_map_progress(self, current, total):
        self.flux_map_progress.setMaximum(total)
        self.flux_map_progress.setValue(current)
        self.flux_map_progress.setLabelText(f"Extracting flux: {current}/{total} pixels")
    
    def _on_flux_map_complete(self, result):
        self.flux_map_progress.close()
        self.flux_map_result_cache = result
        # Snapshot of what was actually selected (self.plotted_lines), not the full fetched
        # catalog -- matches what the cache-validity check below now compares against.
        self.flux_map_hitran_cache = list(getattr(self, 'plotted_lines', None) or [])
        self.flux_map_params_cache = {
            'wave_min': float(self.min_wave_input.text()),
            'wave_max': float(self.max_wave_input.text()),
            'continuum_params': self.flux_map_continuum_params,
        }

        results_dialog = FluxMapResultsDialog(
            result['flux_map'], result['detection_map'], header=result.get('header'),
            wave_range=result.get('wave_range'), wavelength=getattr(self, 'flux_map_wavelength', None),
            cube_data=self.cube_data, target_catalog=getattr(self, 'flux_map_catalog', None),
            contam_catalog=getattr(self, 'flux_map_contam_catalog', None),
            parent=self, excluded_wavelengths=getattr(self, 'flux_map_excluded_wavelengths', set()),
            flux_unit=result.get('flux_unit'),
            continuum_params=self.flux_map_continuum_params,
            contam_flux_map=result.get('contam_flux_map'),
            contam_detection_map=result.get('contam_detection_map')
        )
        results_dialog.exec_()
    
    def _on_flux_map_error(self, error_msg):
        self.flux_map_progress.close()
        if "cancelled" not in error_msg.lower(): QMessageBox.critical(self, "Extraction Error", error_msg)
        else: QMessageBox.information(self, "Extraction Cancelled", "Flux extraction was cancelled")
    
    def _cancel_flux_map_extraction(self):
        if hasattr(self, 'flux_map_worker') and self.flux_map_worker:
            self.flux_map_worker.request_cancel()
            self.flux_map_progress.setLabelText("Cancelling extraction...")