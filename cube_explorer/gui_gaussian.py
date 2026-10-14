import numpy as np
from PyQt5.QtWidgets import QMessageBox, QProgressDialog, QDialog, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
from astropy.units import Quantity

from .gaussian_fitter import fit_gaussian_to_line, GaussianFitResult
from .gaussian_dialogs import GaussianMapWorker, GaussianMapResultsDialog, MultiGaussianFitWorker
from .species_dialog import SpeciesDialog
from .unit_converter import clean_flux_unit, clean_wave_unit
from .utils import take_resolution_notes
from .flux_map_extractor import (
    fit_auto_continuum, fit_hitran_lines_collective, DISPLAY_CROP_SIGMA_MULTIPLIER,
    AREA_ERROR_SANITY_RATIO, SIGMA_BOUND_LOWER_MULTIPLIER, SIGMA_BOUND_UPPER_MULTIPLIER, gaussian,
)

# Same floor fit_gaussians_per_pixel uses for the per-pixel map: below this many finite
# points, a 3-parameter Gaussian fit has 0-1 degrees of freedom, too little to trust.
MIN_FINITE_POINTS = 5

# Matches compute_line_free_noise_std's own floor (gaussian_fitter.py): below this
# many finite points, the noise estimate silently comes back None.
MIN_NOISE_POINTS = 4


class GuiGaussianMixin:
    """
    Mixin class handling all Gaussian Fitting logic (Single, Multi, and Map) for the CubeExplorer.
    """

    def _activate_gaussian_left(self):
        """Toggle boundary-picking mode to the Gaussian left edge.

        Args:
            None

        Returns:
            None
        """
        if self.gauss_left_btn.isChecked():
            self.gauss_mode = 'left'
            self.gauss_peak_btn.setChecked(False)
            self.gauss_right_btn.setChecked(False)
            self.gauss_noise_left_btn.setChecked(False)
            self.gauss_noise_right_btn.setChecked(False)
            self.mark_contam_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _activate_gaussian_peak(self):
        """Toggle boundary-picking mode to the Gaussian peak center.

        Args:
            None

        Returns:
            None
        """
        if self.gauss_peak_btn.isChecked():
            self.gauss_mode = 'peak'
            self.gauss_left_btn.setChecked(False)
            self.gauss_right_btn.setChecked(False)
            self.gauss_noise_left_btn.setChecked(False)
            self.gauss_noise_right_btn.setChecked(False)
            self.mark_contam_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _activate_gaussian_right(self):
        """Toggle boundary-picking mode to the Gaussian right edge.

        Args:
            None

        Returns:
            None
        """
        if self.gauss_right_btn.isChecked():
            self.gauss_mode = 'right'
            self.gauss_left_btn.setChecked(False)
            self.gauss_peak_btn.setChecked(False)
            self.gauss_noise_left_btn.setChecked(False)
            self.gauss_noise_right_btn.setChecked(False)
            self.mark_contam_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _activate_gaussian_noise_left(self):
        """Toggle boundary-picking mode to the left noise-region edge.

        Args:
            None

        Returns:
            None
        """
        if self.gauss_noise_left_btn.isChecked():
            self.gauss_mode = 'noise_left'
            self.gauss_left_btn.setChecked(False)
            self.gauss_peak_btn.setChecked(False)
            self.gauss_right_btn.setChecked(False)
            self.gauss_noise_right_btn.setChecked(False)
            self.mark_contam_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _activate_gaussian_noise_right(self):
        """Toggle boundary-picking mode to the right noise-region edge.

        Args:
            None

        Returns:
            None
        """
        if self.gauss_noise_right_btn.isChecked():
            self.gauss_mode = 'noise_right'
            self.gauss_left_btn.setChecked(False)
            self.gauss_peak_btn.setChecked(False)
            self.gauss_right_btn.setChecked(False)
            self.gauss_noise_left_btn.setChecked(False)
            self.mark_contam_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _mark_gaussian_boundary(self, x_pos):
        """Place the marker for whichever Gaussian boundary is active.

        Args:
            x_pos (float): Wavelength position clicked on the spectrum.

        Returns:
            None
        """
        ax = self.spec_plotter.ax
        
        if self.gauss_mode == 'left':
            if hasattr(self, 'gauss_left_marker') and self.gauss_left_marker:
                try: self.gauss_left_marker.remove()
                except Exception: pass
            self.gauss_left_marker = ax.axvline(x_pos, color='green', linestyle='--', linewidth=1.5, label='Left', alpha=0.7)
            self.gauss_left_x = x_pos
            self.gauss_left_btn.setChecked(False)
            self.gauss_mode = None
            
        elif self.gauss_mode == 'peak':
            if hasattr(self, 'gauss_peak_marker') and self.gauss_peak_marker:
                try: self.gauss_peak_marker.remove()
                except Exception: pass
            self.gauss_peak_marker = ax.axvline(x_pos, color='red', linestyle=':', linewidth=1.0, label='Peak', alpha=0.8)
            self.gauss_peak_x = x_pos
            self.gauss_peak_btn.setChecked(False)
            self.gauss_mode = None
            
        elif self.gauss_mode == 'right':
            if hasattr(self, 'gauss_right_marker') and self.gauss_right_marker:
                try: self.gauss_right_marker.remove()
                except Exception: pass
            self.gauss_right_marker = ax.axvline(x_pos, color='purple', linestyle='--', linewidth=1.5, label='Right', alpha=0.7)
            self.gauss_right_x = x_pos
            self.gauss_right_btn.setChecked(False)
            self.gauss_mode = None

        elif self.gauss_mode == 'noise_left':
            if hasattr(self, 'gauss_noise_left_marker') and self.gauss_noise_left_marker:
                try: self.gauss_noise_left_marker.remove()
                except Exception: pass
            self.gauss_noise_left_marker = ax.axvline(x_pos, color='orange', linestyle='--', linewidth=1.3, label='Noise L', alpha=0.7)
            self.gauss_noise_left_x = x_pos
            self.gauss_noise_left_btn.setChecked(False)
            self.gauss_mode = None

        elif self.gauss_mode == 'noise_right':
            if hasattr(self, 'gauss_noise_right_marker') and self.gauss_noise_right_marker:
                try: self.gauss_noise_right_marker.remove()
                except Exception: pass
            self.gauss_noise_right_marker = ax.axvline(x_pos, color='sienna', linestyle='--', linewidth=1.3, label='Noise R', alpha=0.7)
            self.gauss_noise_right_x = x_pos
            self.gauss_noise_right_btn.setChecked(False)
            self.gauss_mode = None
        
        self.spec_canvas.draw()

    def _swap_gaussian_boundaries_for_tab(self, old_tab, new_tab):
        """Save the outgoing tab's Gaussian boundary state, then load the
        incoming tab's, keeping gauss_left_x/peak_x/etc. as a live view of
        "the active tab's boundaries".

        Args:
            old_tab (str): Tab being switched away from.
            new_tab (str): Tab being switched to.

        Returns:
            None
        """
        if not hasattr(self, 'gauss_boundaries'):
            self.gauss_boundaries = {"full": {}, "left": {}, "right": {}}

        boundary_names = ['gauss_left_x', 'gauss_peak_x', 'gauss_right_x', 'gauss_noise_left_x', 'gauss_noise_right_x']
        marker_names = ['gauss_left_marker', 'gauss_peak_marker', 'gauss_right_marker', 'gauss_noise_left_marker', 'gauss_noise_right_marker']

        saved = {name: getattr(self, name, None) for name in boundary_names + marker_names}
        self.gauss_boundaries[old_tab] = saved
        for name in marker_names:
            marker = saved.get(name)
            if marker is not None:
                try: marker.set_visible(False)
                except Exception: pass

        restored = self.gauss_boundaries.get(new_tab, {})
        for name in boundary_names + marker_names:
            setattr(self, name, restored.get(name))
        for name in marker_names:
            marker = getattr(self, name, None)
            if marker is not None:
                try: marker.set_visible(True)
                except Exception: pass

    def _find_nearest_line_in_range(self, lam_min, lam_max):
        """Find the loaded line closest to the middle of a wavelength range.

        Args:
            lam_min (float): Lower bound of the range.
            lam_max (float): Upper bound of the range.

        Returns:
            wavelength (float): Nearest line's wavelength, or None if no
                loaded line falls within the range.
        """
        if not hasattr(self, 'hitran_lines') or not self.hitran_lines:
            return None

        mid = (lam_min + lam_max) / 2.0
        line_wavelengths = []
        for row in self.hitran_lines:
            try:
                if 'lambda_micron' in row:
                    lam = float(row['lambda_micron'])
                elif 'wavelength' in row:
                    lam = float(row['wavelength'])
                else: continue
                if lam_min <= lam <= lam_max:
                    line_wavelengths.append(lam)
            except (ValueError, KeyError):
                continue

        if not line_wavelengths: return None
        line_wavelengths = np.array(line_wavelengths)
        closest_idx = np.argmin(np.abs(line_wavelengths - mid))
        return float(line_wavelengths[closest_idx])

    def _fit_gaussian(self):
        """Fit a single Gaussian to the currently selected line boundaries.

        Args:
            None

        Returns:
            None
        """
        if self.gauss_left_x is None or self.gauss_right_x is None:
            QMessageBox.warning(self, "Missing Boundaries", "Please select LEFT, PEAK, and RIGHT boundaries in order.")
            return
        
        if self.gauss_peak_x is None:
            QMessageBox.warning(self, "Peak Selection Required", "You must select a peak to fit.")
            return
        
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None or not hasattr(self, 'cube_data') or self.cube_data is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            return

        # Read the active tab's own (possibly shifted) spectrum, not the
        # pristine cube-wide one, so lam_min/lam_max match what was clicked.
        spectrum_x = self._get_tab_wavelength(self.active_spectrum_tab)
        spectrum_y = self._get_active_tab_spectrum()

        if spectrum_x is None or spectrum_y is None or len(spectrum_x) != len(spectrum_y):
            QMessageBox.warning(self, "Data Error", "Spectrum data missing or mismatched.")
            return

        lam_min, lam_max = min(self.gauss_left_x, self.gauss_right_x), max(self.gauss_left_x, self.gauss_right_x)
        spectrum_x_vals = spectrum_x.value if isinstance(spectrum_x, Quantity) else spectrum_x
        spectrum_y_vals = spectrum_y.value if isinstance(spectrum_y, Quantity) else spectrum_y
        fit_mask = (spectrum_x_vals >= lam_min) & (spectrum_x_vals <= lam_max)
        n_finite_in_fit = np.count_nonzero(fit_mask & np.isfinite(spectrum_y_vals))

        if n_finite_in_fit < MIN_FINITE_POINTS:
            QMessageBox.warning(
                self, "Not Enough Data",
                f"Only {n_finite_in_fit} finite point(s) between Left and Right "
                f"(need at least {MIN_FINITE_POINTS}). Widen the boundaries."
            )
            return
        
        if (self.gauss_noise_left_x is None) ^ (self.gauss_noise_right_x is None):
            self.gauss_noise_left_x, self.gauss_noise_right_x = None, None

        noise_min, noise_max = None, None
        if self.gauss_noise_left_x is not None and self.gauss_noise_right_x is not None:
            noise_min, noise_max = min(self.gauss_noise_left_x, self.gauss_noise_right_x), max(self.gauss_noise_left_x, self.gauss_noise_right_x)
            n_finite_in_noise = np.count_nonzero(
                (spectrum_x_vals >= noise_min) & (spectrum_x_vals <= noise_max) & np.isfinite(spectrum_y_vals))
            if n_finite_in_noise < MIN_NOISE_POINTS:
                QMessageBox.warning(
                    self, "Noise Region Too Narrow",
                    f"Only {n_finite_in_noise} finite point(s) in the noise region "
                    f"(need at least {MIN_NOISE_POINTS}). Widen Noise L/Noise R, or "
                    f"clear them to fit without a noise region."
                )
                return

        self.lambda_0 = self._find_nearest_line_in_range(lam_min, lam_max) if hasattr(self, 'hitran_lines') else None

        try:
            result = fit_gaussian_to_line(
                self.gauss_left_x, self.gauss_right_x, spectrum_x, spectrum_y,
                peak_x=self.gauss_peak_x, noise_min=noise_min, noise_max=noise_max,
                header=self.cube_data.header, wave_unit=self.cube_data.wave_unit
            )
            if not result.success:
                QMessageBox.warning(self, "Fit Failed", result.message)
                return
            
            self._display_gaussian_results(result)
            self.spec_plotter.plot_gaussian_fit(result)
            self.spec_canvas.draw_idle()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gaussian fitting error:\n{str(e)}")

    def _display_gaussian_results(self, result: GaussianFitResult):
        """Show a single Gaussian fit's inputs and fitted results in a dialog.

        Args:
            result (GaussianFitResult): Completed fit result to display.

        Returns:
            None
        """
        wave_unit_raw = getattr(self.cube_data, 'wave_unit', 'spectral units')
        flux_unit_raw = getattr(self.cube_data, 'flux_unit', 'flux units')
        wave_unit = clean_wave_unit(wave_unit_raw)
        flux_unit = clean_flux_unit(flux_unit_raw)
        
        axis_name = "Velocity" if "km/s" in wave_unit or "m/s" in wave_unit else ("Wavelength" if wave_unit in ["Å", "µm", "nm"] else "Spectral axis")
        
        # The fitter knows whether the CGS conversion actually ran, so take the
        # unit from the result rather than deriving it from the cube's units.
        area_unit = result.area_unit or f"{flux_unit}·{wave_unit}"

        slope_unit = f"{flux_unit}/{wave_unit}" if wave_unit not in flux_unit else flux_unit

        # Echo back the boundaries the user clicked so the fitted values can be
        # compared against what was actually asked for.
        def _fmt_input(val):
            """Format a boundary value with its wave unit, or 'not set'.

            Args:
                val (float): Boundary value, or None.

            Returns:
                text (str): Formatted value string.
            """
            return f"{val:.6f} {wave_unit}" if val is not None else "not set"

        left_x = getattr(self, 'gauss_left_x', None)
        peak_x = getattr(self, 'gauss_peak_x', None)
        right_x = getattr(self, 'gauss_right_x', None)
        noise_left_x = getattr(self, 'gauss_noise_left_x', None)
        noise_right_x = getattr(self, 'gauss_noise_right_x', None)

        # Noise is always optional and never feeds the fit itself -- only the
        # normalized-residual display below -- so just report the point count.
        n_points = len(result.x_data)
        if noise_left_x is not None and noise_right_x is not None:
            noise_text = (f"{_fmt_input(min(noise_left_x, noise_right_x))} - "
                          f"{_fmt_input(max(noise_left_x, noise_right_x))} "
                          f"({n_points} points in fit window)")
        else:
            noise_text = f"not set ({n_points} points in fit window)"

        user_input_text = f"""USER INPUT:
  Left boundary:         {_fmt_input(left_x)}
  Peak:                  {_fmt_input(peak_x)}
  Right boundary:        {_fmt_input(right_x)}
  Noise region:          {noise_text}"""

        results_text = f"""GAUSSIAN FIT RESULTS
{'='*60}
{user_input_text}

FITTED PARAMETERS:
  Center ({axis_name}):  {result.center:.6f} {wave_unit}
  Amplitude:             {result.amplitude:.4e} {flux_unit}
  Sigma (σ):             {result.sigma:.6g} {wave_unit}
  FWHM:                  {result.fwhm:.6g} {wave_unit}

INTEGRATED QUANTITIES:
  Gaussian Area:         {result.area:.4e} {area_unit}
  Equivalent Width:      {result.area_native/np.mean(result.baseline):.4e} {wave_unit}

BASELINE:
  Slope:                 {result.baseline_slope:.4e} {slope_unit}
  Left endpoint:         {result.baseline_intercept:.4e} {flux_unit}

FIT QUALITY:
  R² (coefficient):      {result.r_squared:.4f}
  Data points:           {len(result.x_data)}

STATUS: {result.message}
{'='*60}"""
        
        dlg = QDialog(self)
        dlg.setWindowTitle("Gaussian Fit Results")
        # Tall enough for the whole report without scrolling: 28 lines of Courier
        # plus the Close button and layout margins.
        dlg.setMinimumSize(600, 520)
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setPlainText(results_text)
        text_edit.setReadOnly(True)
        text_edit.setFontFamily("Courier")
        layout.addWidget(text_edit)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.setLayout(layout)
        dlg.exec_()

    def clear_gaussian_fit(self):
        """Clear all Gaussian fit/boundary markers and reset fit state.

        Args:
            None

        Returns:
            None
        """
        try:
            if hasattr(self.spec_plotter, 'gauss_fit_artists'):
                for artist in self.spec_plotter.gauss_fit_artists:
                    try: artist.remove()
                    except Exception: pass
                self.spec_plotter.gauss_fit_artists = []
            
            for attr in ['gauss_left_marker', 'gauss_peak_marker', 'gauss_right_marker', 'gauss_noise_left_marker', 'gauss_noise_right_marker']:
                if hasattr(self, attr):
                    marker = getattr(self, attr)
                    if marker is not None:
                        try: marker.remove()
                        except Exception: pass
                    setattr(self, attr, None)
            
            for marker in getattr(self, 'gauss_markers', []):
                try: marker.remove()
                except Exception: pass
            self.gauss_markers = []
            
            self.gauss_left_x, self.gauss_peak_x, self.gauss_right_x = None, None, None
            self.gauss_noise_left_x, self.gauss_noise_right_x = None, None
            
            for btn in [self.gauss_left_btn, self.gauss_peak_btn, self.gauss_right_btn, self.gauss_noise_left_btn, self.gauss_noise_right_btn]:
                btn.setChecked(False)
            
            if hasattr(self, 'multi_gauss_artists'):
                for artist in self.multi_gauss_artists:
                    try: artist.remove()
                    except Exception: pass
                self.multi_gauss_artists = []
            
            self.multi_gauss_results = None
            self.spec_plotter.ax.legend(loc='upper right', fontsize=8)
            self.spec_canvas.draw()
            
        except Exception as e:
            print(f"[DEBUG] Error clearing Gaussian fit: {e}")

    def _active_contam_lines(self):
        """Contaminant lines currently checked/plotted, not the full loaded
        catalog -- unchecking one in the Lines dialog, or clicking its marker
        off the spectrum, must exclude it from the fit, not just the picture.

        Args: None

        Returns:
            contam_lines (pd.DataFrame | None): self.contam_lines filtered to
                rows whose wavelength is in self.plotted_lines, or None.
        """
        contam_lines = getattr(self, 'contam_lines', None)
        if contam_lines is None or len(contam_lines) == 0:
            return None
        active_wl = [
            SpeciesDialog.get_row_wavelength(r)
            for r in self.plotted_lines if r.get('source') == 'Contaminant'
        ]
        if not active_wl:
            return None
        mask = contam_lines['line_wave_um'].apply(
            lambda wl: any(abs(wl - w) < 1e-6 for w in active_wl)
        )
        active = contam_lines[mask].reset_index(drop=True)
        return active if len(active) else None

    def _fit_multi_gaussian(self):
        """Fit every selected HITRAN detection line simultaneously.

        Args:
            None

        Returns:
            None
        """
        if not hasattr(self, 'plotted_lines') or not self.plotted_lines:
            QMessageBox.warning(self, "No Lines Selected", "Please first select HITRAN lines to fit.")
            return

        # plotted_lines mixes detection lines with checked contaminants (one shared
        # list). _active_contam_lines() filters contam_lines to just the checked ones.
        detection_lines = [r for r in self.plotted_lines if r.get('source') != 'Contaminant']
        if not detection_lines:
            QMessageBox.warning(self, "No Detection Lines Selected", "Please select at least one detection line (not just contaminants) to fit.")
            return

        if not hasattr(self, 'cube_data') or self.cube_data is None or not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            return

        # Read the ACTIVE tab's own (possibly shifted) wavelength/flux - same
        # reasoning as _fit_gaussian: fit against whatever's actually shown.
        spectrum_x = self._get_tab_wavelength(self.active_spectrum_tab)
        spectrum_y = self._get_active_tab_spectrum()

        if spectrum_x is None or spectrum_y is None or len(spectrum_x) != len(spectrum_y):
            QMessageBox.warning(self, "Data Error", "Data mismatch.")
            return

        contam_lines = self._active_contam_lines()

        # Cache key: lines/contaminants/tab being fit. If unchanged since the
        # last run, offer cached results instead of re-running the slow fit.
        cache_key = (
            tuple(sorted(round(float(r.get('wavelength', 0)), 6) for r in detection_lines)),
            tuple(sorted(round(float(w), 6) for w in contam_lines['line_wave_um']))
                if contam_lines is not None and len(contam_lines) else (),
            self.active_spectrum_tab,
        )
        if hasattr(self, 'multi_gauss_result_cache') and getattr(self, 'multi_gauss_cache_key', None) == cache_key:
            reply = QMessageBox.question(
                self, "Cached Results Available",
                "Results from the last fit with these same lines are cached.\n\n"
                "Click 'Yes' to reopen the cached results (instant)\n"
                "Click 'No' to re-fit with current data",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                fit_results, contam_results = self.multi_gauss_result_cache
                self.multi_gauss_results = fit_results
                self.multi_gauss_contam_results = contam_results
                self._plot_multi_gaussian_on_spectrum(fit_results, contam_results)
                self._show_multi_gaussian_results(fit_results, contam_results, None)
                return

        # continuum_fitted is always a dict, never None; check this tab's own
        # entry (not the dict) so an unfit tab is correctly detected.
        if not hasattr(self, 'continuum_fitted') or self.continuum_fitted.get(self.active_spectrum_tab) is None:
            try:
                continuum_y = fit_auto_continuum(spectrum_x, spectrum_y)
            except Exception as e:
                QMessageBox.warning(self, "Continuum Error", f"Could not compute continuum automatically:\n{str(e)}")
                return
        else:
            continuum_y = self.continuum_fitted[self.active_spectrum_tab]

        self._multi_gauss_cache_key_pending = cache_key

        self.multi_gauss_worker = MultiGaussianFitWorker(
            spectrum_x, spectrum_y, detection_lines, continuum_y,
            header=self.cube_data.header, wave_unit=self.cube_data.wave_unit,
            contam_lines=contam_lines,
        )
        # No cancel button: fit_glines has no cancellation checkpoint, and
        # QThread.terminate() is unsafe here. min=max=0 shows a busy indicator.
        self.multi_gauss_progress = QProgressDialog("Fitting HITRAN lines...", None, 0, 0, self)
        self.multi_gauss_progress.setWindowModality(Qt.WindowModal)
        self.multi_gauss_progress.setMinimumDuration(0)

        self.multi_gauss_worker.progress.connect(self._update_multi_gaussian_progress)
        self.multi_gauss_worker.finished.connect(self._on_multi_gaussian_fit_complete)
        self.multi_gauss_worker.error.connect(self._on_multi_gaussian_fit_error)
        self.multi_gauss_worker.start()

    def _update_multi_gaussian_progress(self, current, total):
        """Update the multi-Gaussian fit progress dialog.

        Args:
            current (int): Chunks fit so far.
            total (int): Total chunks to fit.

        Returns:
            None
        """
        if hasattr(self, 'multi_gauss_progress'):
            self.multi_gauss_progress.setMaximum(total)
            self.multi_gauss_progress.setValue(current)
            self.multi_gauss_progress.setLabelText(f"Fitting HITRAN lines... chunk {current}/{total}")

    def _on_multi_gaussian_fit_error(self, message):
        """Show a failed multi-Gaussian fit's error message.

        Args:
            message (str): Error message from MultiGaussianFitWorker.

        Returns:
            None
        """
        if hasattr(self, 'multi_gauss_progress'):
            self.multi_gauss_progress.close()
        QMessageBox.critical(self, "Fit Error", f"Multi-Gaussian fit failed:\n{message}")

    def _on_multi_gaussian_fit_complete(self, fit_pairs, contam_pairs):
        """Split fit results into successes/failures and show them.

        Args:
            fit_pairs (list): (line, result) pairs for requested detection lines.
            contam_pairs (list): (line, result) pairs for contaminant lines.

        Returns:
            None
        """
        if hasattr(self, 'multi_gauss_progress'):
            self.multi_gauss_progress.close()

        fit_results = [{'line': line, 'result': result} for line, result in fit_pairs if result.success]
        # Every requested line unaccounted for above -- shown in the results dialog below,
        # not just silently absent from the table, per-line reason and all.
        failed_results = [{'line': line, 'result': result} for line, result in fit_pairs if not result.success]
        contam_results = [{'line': line, 'result': result} for line, result in contam_pairs]

        if not fit_results:
            detail = "\n".join(
                f"  - {r['line'].get('species', 'line')}: {r['result'].message}" for r in failed_results
            )
            QMessageBox.warning(self, "Fit Failed",
                                 "Could not fit any HITRAN lines." + (f"\n\n{detail}" if detail else ""))
            return

        self.multi_gauss_results = fit_results
        self.multi_gauss_contam_results = contam_results
        self.multi_gauss_result_cache = (fit_results, contam_results)
        self.multi_gauss_cache_key = getattr(self, '_multi_gauss_cache_key_pending', None)
        self._plot_multi_gaussian_on_spectrum(fit_results, contam_results)
        self._show_multi_gaussian_results(fit_results, contam_results, None, failed_results=failed_results)

    def _plot_multi_gaussian_on_spectrum(self, fit_results, contam_results=None):
        """Draw every fitted line's baseline/Gaussian curve on the spectrum.

        Args:
            fit_results (list): {'line', 'result'} dicts for detection lines.
            contam_results (list): {'line', 'result'} dicts for contaminants.

        Returns:
            None
        """
        ax = self.spec_plotter.ax
        if not hasattr(self, 'multi_gauss_artists'): self.multi_gauss_artists = []
        for artist in self.multi_gauss_artists:
            try: artist.remove()
            except Exception: pass
        self.multi_gauss_artists = []
        
        # Use whichever line the active tab shows; global_line is stale
        # whenever a non-Full tab is active.
        norm_factor = 1.0
        active_line = self._get_active_tab_line() if hasattr(self, '_get_active_tab_line') else getattr(self.spec_plotter, 'global_line', None)
        if active_line is not None:
            active_y = np.asarray(active_line.get_ydata(), dtype=float)
            active_y = active_y[np.isfinite(active_y)]
            if len(active_y):
                active_median = np.nanmedian(active_y)
                if 0.5 < active_median < 2.0:
                    fit_median = np.nanmedian(np.concatenate([fit['result'].y_data for fit in fit_results]))
                    norm_factor = fit_median if fit_median > 0 else 1.0
        
        # Own Gaussian only (not the chunk sum, so neighbors don't bleed in), on
        # a fine grid, cropped to DISPLAY_CROP_SIGMA_MULTIPLIER.
        def _own_curve(result):
            """Build one line's own baseline/fit curve on a fine x-grid.

            Args:
                result (GaussianFitResult): The line's fit result.

            Returns:
                curve (tuple): (x_fine, base_fine, fit_fine) arrays.
            """
            x_data = result.x_data
            model = result.y_fit_own if result.y_fit_own is not None else result.y_fit
            y_abs = (np.asarray(model) + result.baseline) / norm_factor
            own_gaussian = gaussian(x_data, result.amplitude, result.center, result.sigma)
            base_abs = y_abs - own_gaussian / norm_factor
            half_width = DISPLAY_CROP_SIGMA_MULTIPLIER * abs(result.sigma)
            lo = max(x_data.min(), result.center - half_width)
            hi = min(x_data.max(), result.center + half_width)
            if hi <= lo:
                return x_data, base_abs, y_abs
            x_fine = np.linspace(lo, hi, 150)
            base_fine = np.interp(x_fine, x_data, base_abs)
            fit_fine = base_fine + gaussian(x_fine, result.amplitude, result.center, result.sigma) / norm_factor
            return x_fine, base_fine, fit_fine

        # Continuum plus every Gaussian curve_fit solved for at once, not just this
        # line's own. Lines sharing an extent give the same curve -- draw it once.
        def _chunk_curve(result):
            """Build the joint chunk curve (all lines the fit solved together).

            Args:
                result (GaussianFitResult): Any line's fit result sharing the chunk.

            Returns:
                curve (tuple): (x_data, base_abs, fit_abs) arrays.
            """
            x_data = result.x_data
            fit_abs = (np.asarray(result.y_fit) + result.baseline) / norm_factor
            base_abs = np.asarray(result.baseline) / norm_factor
            return x_data, base_abs, fit_abs

        # Always the real joint chunk curve here (never the per-line crop) -- lines
        # sharing an extent are deduped so the same curve isn't drawn/stacked twice.
        seen_extents = set()
        idx = 0
        for fit_info in fit_results:
            result = fit_info['result']
            extent_key = (round(float(result.x_data[0]), 6), round(float(result.x_data[-1]), 6))
            if extent_key in seen_extents:
                continue
            seen_extents.add(extent_key)
            x_local, base_local, fit_local = _chunk_curve(result)
            baseline_line, = ax.plot(x_local, base_local, 'k--', linewidth=2, alpha=0.7,
                                      label='Baseline' if idx == 0 else '', zorder=10)
            gauss_line, = ax.plot(x_local, fit_local, 'r-', linewidth=3, alpha=0.9,
                                   label='Gaussian Fit' if idx == 0 else '', zorder=11)
            fill = ax.fill_between(x_local, base_local, fit_local, color='red', alpha=0.2, zorder=9)
            self.multi_gauss_artists.extend([baseline_line, gauss_line, fill])
            idx += 1

        # Contaminants get their own color: drawing them in red alongside real
        # hits would look like an extra, unreported detection.
        seen_species = set()
        for contam_info in (contam_results or []):
            result = contam_info['result']
            species = contam_info['line'].get('species', 'Contaminant')
            label = f"Contaminant ({species})" if species not in seen_species else ''
            seen_species.add(species)
            x_local, base_local, fit_local = _own_curve(result)
            # Outline only, no fill, so it doesn't compete with the red fill;
            # grey mirrors the marker this contaminant used before fitting.
            contam_line, = ax.plot(x_local, fit_local, color='dimgray', linewidth=2, alpha=1.0, label=label, zorder=12)
            self.multi_gauss_artists.append(contam_line)

        ax.legend(loc='upper right', fontsize=8)
        self.spec_canvas.draw_idle()

    def _show_multi_gaussian_results(self, fit_results, contam_results=None, velocity_shift_kms=None,
                                      failed_results=None):
        """Build and show the multi-Gaussian fit results text dialog.

        Args:
            fit_results (list): {'line', 'result'} dicts for detection lines.
            contam_results (list): {'line', 'result'} dicts for contaminants.
            velocity_shift_kms (float): Applied velocity shift, or None.
            failed_results (list): {'line', 'result'} dicts for lines that
                could not be fit.

        Returns:
            None
        """
        wave_unit = clean_wave_unit(getattr(self.cube_data, 'wave_unit', 'µm'))
        flux_unit = clean_flux_unit(getattr(self.cube_data, 'flux_unit', 'MJy/sr'))
        fallback_area_unit = f"{flux_unit}·{wave_unit}"
        failed_results = failed_results or []

        header_line = f"Successfully fitted {len(fit_results)} lines"
        if failed_results:
            header_line += f" ({len(failed_results)} requested line(s) could not be fit -- see below)"
        results_text = f"MULTI-GAUSSIAN FIT RESULTS\n{'='*70}\n{header_line}\n"
        if velocity_shift_kms is not None:
            results_text += (f"Applied a {velocity_shift_kms:+.2f} km/s wavelength shift to every line's "
                              f"fixed center (estimated from these lines' own peaks) -- "
                              f"'Center (λ)' below reflects this, 'Line λ (catalog)' does not.\n")

        if failed_results:
            results_text += f"\nSKIPPED / FAILED LINES:\n{'='*70}\n"
            for idx, fail_info in enumerate(failed_results, 1):
                result, line = fail_info['result'], fail_info['line']
                wl = line.get('lambda_micron', line.get('wavelength'))
                wl_text = f"{wl:.6f} {wave_unit}" if wl is not None else "N/A"
                results_text += f"{idx}. {line.get('species', 'Unknown')} @ {wl_text}: {result.message}\n"

        results_text += f"\nFITTED LINES (target and contaminant, by wavelength):\n{'='*70}\n"

        def _unreliable_reason(result):
            """
            Which known mechanism makes this line's area_error unreliable, if any.

            Args:
                result (GaussianFitResult): One line's fit result

            Returns:
                reason (str): 'bound' if fitted sigma is pinned at its curve_fit
                    bound (an optimizer-boundary artifact, not real flux ambiguity),
                    'blend' if area_error exceeds AREA_ERROR_SANITY_RATIO times the
                    area for any other reason (usually a genuine close blend), or
                    None if the line is reliable
            """
            if (result.area_error is None
                    or abs(result.area_error) <= AREA_ERROR_SANITY_RATIO * abs(result.area)):
                return None
            if result.catalog_sigma:
                lower = SIGMA_BOUND_LOWER_MULTIPLIER * result.catalog_sigma
                upper = SIGMA_BOUND_UPPER_MULTIPLIER * result.catalog_sigma
                if abs(result.sigma - lower) < 0.01 * lower or abs(result.sigma - upper) < 0.01 * upper:
                    return 'bound'
            return 'blend'

        # Interleaved by wavelength so the listing matches spectrum order;
        # Line N / Contam N number within their own kind, not globally.
        merged_entries = (
            [('line', idx, fit_info) for idx, fit_info in enumerate(fit_results, 1)]
            + [('contam', idx, contam_info) for idx, contam_info in enumerate(contam_results or [], 1)]
        )
        merged_entries.sort(key=lambda e: e[2]['result'].center)

        for kind, idx, info in merged_entries:
            result, line = info['result'], info['line']
            area_err_text = f" ± {result.area_error:.4e}" if result.area_error is not None else ""
            reason = _unreliable_reason(result)
            if reason == 'bound':
                reliability_tag = "  [FLAG: sigma pinned at fit bound]"
            elif reason == 'blend':
                reliability_tag = "  [FLAG: error exceeds area]"
            else:
                reliability_tag = ""

            if kind == 'contam':
                results_text += (
                    f"Contam {idx}: {line.get('species', 'Contaminant')} (not one of the "
                    f"{len(fit_results)} target lines; shown in grey on the plot)\n"
                    f"  Center (λ):            {result.center:.6f} {wave_unit}\n"
                    f"  Amplitude:             {result.amplitude:.4e} {flux_unit}\n"
                    f"  Integrated Area:       {result.area:.4e}{area_err_text} "
                    f"{result.area_unit or fallback_area_unit}{reliability_tag}\n"
                    f"  {'─'*60}\n"
                )
                continue

            # Catalog identification -- present for HITRAN-sourced lines, absent for
            # custom .lst catalogs, so each is only shown when the line dict has it.
            catalog_wave = line.get('lambda_micron', line.get('wavelength'))
            catalog_wave_text = f"{catalog_wave:.6f} {wave_unit}" if catalog_wave is not None else "N/A"
            transition_text = line.get('transition') or "N/A"
            a_coeff = line.get('a_coeff')
            a_coeff_text = f"{a_coeff:.4e} s⁻¹" if a_coeff is not None else "N/A"
            e_upper = line.get('e_upper')
            e_upper_text = f"{e_upper:.2f} K" if e_upper is not None else "N/A"

            results_text += f"""Line {idx}: {line.get('species', 'Unknown')}
  Line λ (catalog):      {catalog_wave_text}
  Transition:            {transition_text}
  A coeff:               {a_coeff_text}
  E_upper:               {e_upper_text}
  Center (λ):            {result.center:.6f} {wave_unit}
  Amplitude:             {result.amplitude:.4e} {flux_unit}
  Sigma (σ):             {result.sigma:.6g} {wave_unit}
  FWHM:                  {result.fwhm:.6g} {wave_unit}
  Integrated Area:       {result.area:.4e}{area_err_text} {result.area_unit or fallback_area_unit}{reliability_tag}
  R²:                    {result.r_squared:.4f}\n  {'─'*60}\n"""

        def _sum_area(results_list):
            """Sum area across results only if they share one area_unit;
            refuses to sum mismatched units rather than guessing.

            Args:
                results_list (list): {'line', 'result'} dicts to sum.

            Returns:
                result (tuple): (total, unit_text) if all results share a
                    unit; (None, note) explaining why not, otherwise.
            """
            if not results_list:
                return 0.0, fallback_area_unit
            units = {r['result'].area_unit or fallback_area_unit for r in results_list}
            if len(units) == 1:
                return sum(r['result'].area for r in results_list), units.pop()
            return None, f"cannot sum -- mixed units across lines ({', '.join(sorted(units))})"

        total_flux, total_flux_unit = _sum_area(fit_results)
        r2_values = [fit_info['result'].r_squared for fit_info in fit_results]
        mean_r2 = np.mean(r2_values) if r2_values else float('nan')
        n_good = sum(1 for r2 in r2_values if r2 >= 0.5)

        # Quadrature sum over reliable lines' area_error only; an unreliable
        # line's area still counts in total_flux, just not its error.
        reliable_fits = [fit_info for fit_info in fit_results if _unreliable_reason(fit_info['result']) is None]
        area_errors = [fit_info['result'].area_error for fit_info in reliable_fits]
        total_flux_error = (float(np.sqrt(sum(e ** 2 for e in area_errors)))
                             if total_flux is not None and area_errors and all(e is not None for e in area_errors)
                             else None)

        results_text += f"\nOVERALL:\n  Total lines fitted:    {len(fit_results)}\n"
        if total_flux is not None:
            err_text = f" ± {total_flux_error:.4e}" if total_flux_error is not None else ""
            results_text += f"  Total Integrated Flux: {total_flux:.4e}{err_text} {total_flux_unit}\n"
        else:
            results_text += f"  Total Integrated Flux: {total_flux_unit}\n"
        results_text += f"  Mean R²:               {mean_r2:.4f}\n"
        results_text += f"  Lines with R² >= 0.5:  {n_good}/{len(fit_results)}\n"

        if contam_results:
            total_contam_flux, total_contam_flux_unit = _sum_area(contam_results)
            if total_contam_flux is not None:
                results_text += f"  Total Contaminant Flux: {total_contam_flux:.4e} {total_contam_flux_unit}\n"
            else:
                results_text += f"  Total Contaminant Flux: {total_contam_flux_unit}\n"

        poor_fits = [fit_info for fit_info in fit_results if fit_info['result'].r_squared < 0.5]
        contam_loaded = getattr(self, 'contam_lines', None) is not None and len(self.contam_lines) > 0
        if poor_fits and not contam_loaded:
            results_text += (
                f"\nNOTE: {len(poor_fits)} line(s) have R² < 0.5. If any of these overlap a "
                f"real feature from another species (not in this line list), it has no "
                f"template of its own and can distort nearby lines' amplitudes. Load one via "
                f"the \"Contam. Lines\" button, then run Multi-Gaussian Fit again.\n"
            )

        dlg = QDialog(self)
        dlg.setWindowTitle("Multi-Gaussian Fit Results")
        # Extra headroom for the 4 added rows per line (catalog wavelength,
        # transition, A coeff, E_upper), to avoid scrolling for 5-10 lines.
        dlg.setMinimumSize(750, 750)
        layout = QVBoxLayout()
        text_display = QTextEdit()
        text_display.setPlainText(results_text)
        text_display.setReadOnly(True)
        # Match the single-line Gaussian Fit Results dialog's font exactly.
        text_display.setFontFamily("Courier")
        layout.addWidget(text_display)
        
        btn_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        dlg.setLayout(layout)
        dlg.exec_()

    def _fit_gaussian_map(self):
        """Fit a Gaussian at every pixel of the cube (or a region on Left/Right).

        Args:
            None

        Returns:
            None
        """
        if self.gauss_left_x is None or self.gauss_right_x is None:
            QMessageBox.warning(self, "Missing Boundaries", "Please select both LEFT and RIGHT boundaries first.")
            return

        if not hasattr(self, 'cube_data') or self.cube_data is None:
            return

        # A cancelled worker keeps running briefly while in-flight rows finish;
        # starting a second fit on top would run two thread pools at once.
        prev_worker = getattr(self, 'gauss_worker', None)
        if prev_worker is not None and prev_worker.isRunning():
            QMessageBox.warning(
                self, "Fit Still Stopping",
                "The previous Gaussian map fit is still shutting down after being "
                "cancelled. Wait a few seconds and try again."
            )
            return
        
        # The fit always covers the whole cube; Left/Right's displayed map is
        # masked to that side's region afterward, so a region is required.
        active_tab = self.active_spectrum_tab
        tab_label = active_tab.capitalize()
        region_mask = None
        if active_tab in ("left", "right"):
            sel = self.last_selection_left if active_tab == "left" else self.last_selection_right
            if not sel or sel.get("shape") == "*":
                QMessageBox.warning(
                    self, "Region Selection Required",
                    f"A Gaussian map needs multiple pixels - select a region "
                    f"(not a single point) on the {tab_label} side first."
                )
                return
            region_mask = sel.get("mask")

        cube = self.cube_data.cube
        wavelength = self._get_tab_wavelength(active_tab)

        if cube is None or wavelength is None: return

        lam_min, lam_max = min(self.gauss_left_x, self.gauss_right_x), max(self.gauss_left_x, self.gauss_right_x)

        if self.gauss_noise_left_x is None or self.gauss_noise_right_x is None:
            QMessageBox.warning(self, "Noise Region Required", "You MUST define a noise region for map fitting.")
            return

        noise_min, noise_max = min(self.gauss_noise_left_x, self.gauss_noise_right_x), max(self.gauss_noise_left_x, self.gauss_noise_right_x)
        wavelength_vals = wavelength.value if isinstance(wavelength, Quantity) else wavelength
        n_wave_in_noise = np.count_nonzero((wavelength_vals >= noise_min) & (wavelength_vals <= noise_max))
        if n_wave_in_noise < MIN_NOISE_POINTS:
            QMessageBox.warning(
                self, "Noise Region Too Narrow",
                f"Only {n_wave_in_noise} wavelength sample(s) in the noise region "
                f"(need at least {MIN_NOISE_POINTS}). Widen Noise L/Noise R."
            )
            return

        # gauss_cache_tab must also match: two tabs can share bounds but need
        # different (differently masked) result maps.
        if (getattr(self, 'gauss_cache', None) is not None and
            getattr(self, 'gauss_cache_tab', None) == active_tab and
            self.gauss_cache_lam_min == lam_min and self.gauss_cache_lam_max == lam_max and
            self.gauss_cache_noise_min == noise_min and self.gauss_cache_noise_max == noise_max):
            QMessageBox.information(self, "Cached Results", "Using previously fitted Gaussian map.")
            self._show_gaussian_map_results(self.gauss_cache, tab=active_tab)
            return
        
        if QMessageBox.question(self, "Fit Map", "Continue with map fitting?", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes: return

        # Real 0-100 bar (worker reports per row) with autoClose/autoReset off,
        # so it survives past 100% until post-fit unit conversion finishes.
        n_pixels = cube.shape[1] * cube.shape[2]
        self.gauss_progress = QProgressDialog(
            f"Fitting Gaussians to {n_pixels:,} pixels...", "Cancel", 0, 100, self)
        self.gauss_progress.setWindowTitle("Gaussian Map Fitting")
        self.gauss_progress.setWindowModality(Qt.NonModal)
        self.gauss_progress.setAutoClose(False)
        self.gauss_progress.setAutoReset(False)
        self.gauss_progress.setValue(0)
        self.gauss_progress.canceled.connect(self._on_gaussian_map_cancel_clicked)
        self.gauss_progress.show()

        self.lambda_0 = self._find_nearest_line_in_range(lam_min, lam_max) if hasattr(self, 'hitran_lines') else None

        # Capture which tab/mask this run belongs to now, not read it back
        # later, since the user could switch tabs before the QThread finishes.
        self._gauss_map_fit_tab = active_tab
        self._gauss_map_fit_mask = region_mask

        take_resolution_notes()  # drop the warnings of an earlier run
        self.gauss_worker = GaussianMapWorker(
            cube=cube, wavelength=wavelength, lam_min=lam_min, lam_max=lam_max, n_jobs=8,
            fac_above_stdres=3.0, noise_min=noise_min, noise_max=noise_max,
            header=self.cube_data.header, wave_unit=self.cube_data.wave_unit,
            flux_unit=self.cube_data.flux_unit, lambda_0=self.lambda_0
        )
        self.gauss_worker.finished.connect(self._on_gaussian_map_complete)
        self.gauss_worker.error.connect(self._on_gaussian_map_error)
        self.gauss_worker.progress.connect(self._on_gaussian_map_progress)
        self.gauss_worker.start()

    def _on_gaussian_map_cancel_clicked(self):
        """Stop a running Gaussian map fit. Qt already hides the dialog as
        part of the button click, so this just tells the worker to stop.

        Args:
            None

        Returns:
            None
        """
        worker = getattr(self, 'gauss_worker', None)
        if worker is not None and worker.isRunning():
            worker.cancel()

    def _on_gaussian_map_progress(self, current, total):
        """Mirror the worker's per-row progress into the progress dialog.

        Args:
            current (int): Pixels fit so far.
            total (int): Total pixels to fit.

        Returns:
            None
        """
        dlg = getattr(self, 'gauss_progress', None)
        if dlg is None or total <= 0 or dlg.wasCanceled():
            return
        pct = int(100 * current / total)
        dlg.setLabelText(
            f"Fitting Gaussians...\n{current:,} / {total:,} pixels ({pct}%)")
        dlg.setValue(pct)

    def _mask_gaussian_map_results_for_tab(self, results, tab, mask):
        """Copy results and set every spatial parameter map to NaN (False for
        boolean maps) outside the region mask; Full passes through unmasked.
        Only changes what's cached/displayed, not what was actually fit.

        Args:
            results (dict): Per-pixel Gaussian map results to mask.
            tab (str): Tab the results belong to, 'full', 'left', or 'right'.
            mask (numpy.ndarray): Boolean region mask, or None.

        Returns:
            results (dict): The masked results, or the input unchanged for
                Full or when no mask applies.
        """
        if tab not in ("left", "right") or mask is None or results is None:
            return results
        mask = np.asarray(mask, dtype=bool)
        masked = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[-2:] == mask.shape:
                arr = value.copy()
                if np.issubdtype(arr.dtype, np.bool_):
                    arr[..., ~mask] = False
                else:
                    arr = arr.astype(float, copy=False)
                    arr[..., ~mask] = np.nan
                masked[key] = arr
            else:
                masked[key] = value
        return masked

    def _on_gaussian_map_complete(self, results):
        """Cache the completed Gaussian map fit and show its results.

        Args:
            results (dict): Per-pixel Gaussian map results from the worker.

        Returns:
            None
        """
        self.gauss_progress.close()
        resolution_note = take_resolution_notes()
        if resolution_note:
            QMessageBox.warning(self, "Spectral Resolution Held Flat", resolution_note)
        tab = getattr(self, '_gauss_map_fit_tab', self.active_spectrum_tab)
        mask = getattr(self, '_gauss_map_fit_mask', None)
        masked_results = self._mask_gaussian_map_results_for_tab(results, tab, mask)
        self.gauss_cache = masked_results
        self.gauss_cache_tab = tab
        self.gauss_cache_lam_min = min(self.gauss_left_x, self.gauss_right_x)
        self.gauss_cache_lam_max = max(self.gauss_left_x, self.gauss_right_x)
        self.gauss_cache_noise_min = min(self.gauss_noise_left_x, self.gauss_noise_right_x)
        self.gauss_cache_noise_max = max(self.gauss_noise_left_x, self.gauss_noise_right_x)
        self._show_gaussian_map_results(masked_results, tab=tab)

    def _on_gaussian_map_error(self, error_msg):
        """Show a failed Gaussian map fit's error message.

        Args:
            error_msg (str): Error message from the map worker.

        Returns:
            None
        """
        self.gauss_progress.close()
        QMessageBox.critical(self, "Error", f"Gaussian map fitting error:\n{error_msg}")

    def _show_gaussian_map_results(self, results: dict, tab=None):
        """Open the Gaussian map results dialog for a tab's completed fit.

        Args:
            results (dict): Per-pixel Gaussian map results to display.
            tab (str): Tab the results belong to; defaults to the active tab.

        Returns:
            None
        """
        # tab defaults to the active one; callers with results tied to a
        # specific tab pass it explicitly so the axis matches the results.
        if tab is None:
            tab = self.active_spectrum_tab
        wavelength = self._get_tab_wavelength(tab)
        # Noise bounds come from the cache, not the live markers, which can be
        # cleared/moved after a fit; the view must show the actual threshold used.
        dlg = GaussianMapResultsDialog(
            self, results, self.cube_data, self.gauss_left_x, self.gauss_right_x,
            wavelength=wavelength,
            noise_min=getattr(self, 'gauss_cache_noise_min', None),
            noise_max=getattr(self, 'gauss_cache_noise_max', None),
            gauss_peak_x=getattr(self, 'gauss_peak_x', None),
        )
        dlg.exec_()