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
from .utils import take_resolution_notes

class GuiFluxMapMixin:
    """
    Mixin class handling all Flux Map extraction logic for the CubeExplorer.
    """

    def _extract_flux_map(self):
        """Extract rovibrational flux from entire cube. Includes validation
        checks and preview before processing.

        Args:
            None

        Returns:
            None
        """
        if not hasattr(self, 'cube_data') or self.cube_data is None or self.cube_data.cube is None:
            QMessageBox.warning(self, "No Data", "Load a FITS cube first")
            return
        
        if not hasattr(self, 'contam_lines'):
            self.contam_lines = None

        # Check the range and the lines before the first prompt.
        try:
            wave_min = float(self.min_wave_input.text())
            wave_max = float(self.max_wave_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Wavelength", "Enter valid min/max wavelength values.")
            return

        if wave_min >= wave_max:
            QMessageBox.warning(self, "Invalid Range", "Min wavelength must be less than Max.")
            return

        # Source of truth is self.plotted_lines: exactly what's checked in the
        # Show Lines dialog, mirroring Multi-Gaussian (previously ignored checkbox state).
        if not getattr(self, 'plotted_lines', None):
            QMessageBox.warning(self, "No Lines Selected", "Fetch or load lines, then click Show Lines to choose them.")
            return

        detection_rows = [r for r in self.plotted_lines if r.get('source') != 'Contaminant']
        contam_rows = [r for r in self.plotted_lines if r.get('source') == 'Contaminant']

        def _in_range(row):
            """Check whether a line row's wavelength falls in [wave_min, wave_max].

            Args:
                row (dict): A line row, as used throughout the Show Lines dialog.

            Returns:
                in_range (bool): True if the row's wavelength is in range.
            """
            wave = SpeciesDialog.get_row_wavelength(row)
            return wave is not None and wave_min <= float(wave) <= wave_max

        filtered_lines = [r for r in detection_rows if _in_range(r)]
        if not filtered_lines:
            QMessageBox.warning(self, "No Lines Selected",
                                 f"No selected (checked) lines fall within {wave_min:.6f} - {wave_max:.6f} μm. "
                                 "Click Show Lines to choose lines in this range.")
            return

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

        cube_shape = self.cube_data.cube.shape
        spectrum_wave, _ = extract_units(self.cube_data.wavelength)

        # Preview the brightest pixel within the fit range rather than the cube's
        # geometric center, which can have little or no target-line signal.
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
        
        # Capped at 4: measured on this workload, more threads make it worse
        # (BLAS-internal threading contends with the outer ThreadPoolExecutor).
        try: n_jobs = min(os.cpu_count() or 4, 4)
        except Exception: n_jobs = 4
        
        # Converts filtered_lines/contam_rows to the DataFrame shape find_lines()/
        # fit_glines() expect; no species override, unlike the old hitran_df path.
        def _rows_to_catalog(rows):
            """Convert line rows to the 'line_wave_um'/'species' DataFrame
            shape find_lines()/fit_glines() expect.

            Args:
                rows (list): Line rows to convert.

            Returns:
                df (pandas.DataFrame): Rows with the wavelength column
                    renamed to 'line_wave_um'.
            """
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

        take_resolution_notes()  # drop the warnings of an earlier run
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
        """Preview the brightest in-range pixel's continuum-subtracted
        spectrum and line positions, and confirm before running the map.

        Args:
            lines (list): Filtered target-line rows to mark on the preview.
            spectrum_wave (numpy.ndarray): Wavelength array for the preview pixel.
            spectrum_flux (numpy.ndarray): Flux array for the preview pixel.
            pixel_x (int): Preview pixel's x coordinate.
            pixel_y (int): Preview pixel's y coordinate.

        Returns:
            reply: QMessageBox.Yes if the user chose to proceed, else another value.
        """
        # Flux Map always continuum-subtracts every pixel, so the preview does too.
        preview_flux = spectrum_flux.copy()
        continuum = None
        try:
            continuum = fit_auto_continuum(spectrum_wave, preview_flux, **(self.flux_map_continuum_params or {}))
            if continuum is not None and not np.all(np.isnan(continuum)):
                preview_flux = preview_flux - continuum
        except Exception: pass

        # Same species-label logic as FluxMapResultsDialog._species_label:
        # first non-empty 'species' value seen, 'Target' as the fallback.
        species_label = next(
            (s for l in lines if (s := str(l.get('species', '')).strip())), 'Target'
        )

        dlg = QDialog(self)
        dlg.setWindowTitle(f"{species_label} Flux Map Preview")
        dlg.setGeometry(100, 100, 1000, 600)
        layout = QVBoxLayout()

        title = QLabel(f"<b>{species_label} Flux Map Preview</b><br>Brightest pixel in range: ({pixel_x}, {pixel_y}) | {len(lines)} filtered lines")
        layout.addWidget(title)

        fig = Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        # Raw spectrum + fitted continuum (pixel-inspect style) lets you judge
        # the continuum fit before committing to the full extraction.
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
            "<b>Lines shown:</b> the lines checked in Show Lines, "
            f"within {wave_min:.6f} - {wave_max:.6f} μm (the λ min and λ max boxes). "
            "Click Show Lines to change the selection.<br>"
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
        """Update the flux map progress dialog.

        Args:
            current (int): Pixels extracted so far.
            total (int): Total pixels to extract.

        Returns:
            None
        """
        self.flux_map_progress.setMaximum(total)
        self.flux_map_progress.setValue(current)
        self.flux_map_progress.setLabelText(f"Extracting flux: {current}/{total} pixels")
    
    def _on_flux_map_complete(self, result):
        """Cache the completed flux map and show its results dialog.

        Args:
            result (dict): Flux map results from FluxMapWorker.

        Returns:
            None
        """
        self.flux_map_progress.close()
        resolution_note = take_resolution_notes()
        if resolution_note:
            QMessageBox.warning(self, "Spectral Resolution Held Flat", resolution_note)
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
        """Show a failed (or cancelled) flux map extraction's message.

        Args:
            error_msg (str): Error message from FluxMapWorker.

        Returns:
            None
        """
        self.flux_map_progress.close()
        if "cancelled" not in error_msg.lower(): QMessageBox.critical(self, "Extraction Error", error_msg)
        else: QMessageBox.information(self, "Extraction Cancelled", "Flux extraction was cancelled")

    def _cancel_flux_map_extraction(self):
        """Request cancellation of the running flux map extraction.

        Args:
            None

        Returns:
            None
        """
        if hasattr(self, 'flux_map_worker') and self.flux_map_worker:
            self.flux_map_worker.request_cancel()
            self.flux_map_progress.setLabelText("Cancelling extraction...")