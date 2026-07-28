import numpy as np
from PyQt5.QtWidgets import QMessageBox, QProgressDialog, QDialog, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from astropy.units import Quantity

from .gaussian_fitter import fit_gaussian_to_line, GaussianFitResult, auto_detect_boundaries
from .gaussian_dialogs import GaussianMapWorker, GaussianMapResultsDialog
from .species_dialog import SpeciesDialog
from .unit_converter import clean_flux_unit, clean_wave_unit
from .co_flux_extractor import fit_continuum_median_peaks

# Below this many samples in the fit window, a 1D fit will not run without a noise
# region. The noise value itself does not enter the fit; the gate exists because a
# window this narrow is too poorly sampled to trust on its own.
NOISE_REQUIRED_BELOW_N_POINTS = 15


def _extract_wavelength_value(wavelength_quantity_or_array, index=None):
    if isinstance(wavelength_quantity_or_array, Quantity):
        if index is not None:
            return wavelength_quantity_or_array[index].value
        else:
            return wavelength_quantity_or_array.value
    else:
        if index is not None:
            return wavelength_quantity_or_array[index]
        else:
            return wavelength_quantity_or_array


class GuiGaussianMixin:
    """
    Mixin class handling all Gaussian Fitting logic (Single, Multi, and Map) for the CubeExplorer.
    """

    def _activate_gaussian_left(self):
        if self.gauss_left_btn.isChecked():
            self.gauss_mode = 'left'
            self.gauss_peak_btn.setChecked(False)
            self.gauss_right_btn.setChecked(False)
            self.gauss_noise_left_btn.setChecked(False)
            self.gauss_noise_right_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _activate_gaussian_peak(self):
        if self.gauss_peak_btn.isChecked():
            self.gauss_mode = 'peak'
            self.gauss_left_btn.setChecked(False)
            self.gauss_right_btn.setChecked(False)
            self.gauss_noise_left_btn.setChecked(False)
            self.gauss_noise_right_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _activate_gaussian_right(self):
        if self.gauss_right_btn.isChecked():
            self.gauss_mode = 'right'
            self.gauss_left_btn.setChecked(False)
            self.gauss_peak_btn.setChecked(False)
            self.gauss_noise_left_btn.setChecked(False)
            self.gauss_noise_right_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _activate_gaussian_noise_left(self):
        if self.gauss_noise_left_btn.isChecked():
            self.gauss_mode = 'noise_left'
            self.gauss_left_btn.setChecked(False)
            self.gauss_peak_btn.setChecked(False)
            self.gauss_right_btn.setChecked(False)
            self.gauss_noise_right_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _activate_gaussian_noise_right(self):
        if self.gauss_noise_right_btn.isChecked():
            self.gauss_mode = 'noise_right'
            self.gauss_left_btn.setChecked(False)
            self.gauss_peak_btn.setChecked(False)
            self.gauss_right_btn.setChecked(False)
            self.gauss_noise_left_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _mark_gaussian_boundary(self, x_pos):
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

    def _find_nearest_line_in_range(self, lam_min, lam_max):
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
        if self.gauss_left_x is None or self.gauss_right_x is None:
            QMessageBox.warning(self, "Missing Boundaries", "Please select LEFT, PEAK, and RIGHT boundaries in order.")
            return
        
        if self.gauss_peak_x is None:
            QMessageBox.warning(self, "Peak Selection Required", "You must select a peak to fit.")
            return
        
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None or not hasattr(self, 'cube_data') or self.cube_data is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            return
        
        spectrum_x = _extract_wavelength_value(self.spec_plotter.wavelength)
        spectrum_y = self.cube_data.spectrum
        
        if spectrum_x is None or spectrum_y is None or len(spectrum_x) != len(spectrum_y):
            QMessageBox.warning(self, "Data Error", "Spectrum data missing or mismatched.")
            return
        
        lam_min, lam_max = min(self.gauss_left_x, self.gauss_right_x), max(self.gauss_left_x, self.gauss_right_x)
        spectrum_x_vals = spectrum_x.value if isinstance(spectrum_x, Quantity) else spectrum_x
        n_points_in_fit = np.count_nonzero((spectrum_x_vals >= lam_min) & (spectrum_x_vals <= lam_max))
        
        if n_points_in_fit < NOISE_REQUIRED_BELOW_N_POINTS and (self.gauss_noise_left_x is None or self.gauss_noise_right_x is None):
            QMessageBox.warning(
                self, "Noise Region Required",
                f"Narrow range ({n_points_in_fit} points, under "
                f"{NOISE_REQUIRED_BELOW_N_POINTS}). Define a noise region first, "
                f"or widen the fit window."
            )
            return
        
        if (self.gauss_noise_left_x is None) ^ (self.gauss_noise_right_x is None):
            self.gauss_noise_left_x, self.gauss_noise_right_x = None, None

        noise_min, noise_max = None, None
        if self.gauss_noise_left_x is not None and self.gauss_noise_right_x is not None:
            noise_min, noise_max = min(self.gauss_noise_left_x, self.gauss_noise_right_x), max(self.gauss_noise_left_x, self.gauss_noise_right_x)

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
            return f"{val:.6f} {wave_unit}" if val is not None else "not set"

        left_x = getattr(self, 'gauss_left_x', None)
        peak_x = getattr(self, 'gauss_peak_x', None)
        right_x = getattr(self, 'gauss_right_x', None)
        noise_left_x = getattr(self, 'gauss_noise_left_x', None)
        noise_right_x = getattr(self, 'gauss_noise_right_x', None)

        # A noise region is only demanded for narrow windows (<15 points), and even
        # then it does not feed into the fitted values. Say which case applies so an
        # unset noise region doesn't read as something the user forgot.
        n_points = len(result.x_data)
        requirement = "required" if n_points < NOISE_REQUIRED_BELOW_N_POINTS else "not required"
        if noise_left_x is not None and noise_right_x is not None:
            noise_text = (f"{_fmt_input(min(noise_left_x, noise_right_x))} - "
                          f"{_fmt_input(max(noise_left_x, noise_right_x))} "
                          f"({requirement} at {n_points} points)")
        else:
            noise_text = f"not set ({requirement} at {n_points} points)"

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

    def _fit_multi_gaussian(self):
        if not hasattr(self, 'plotted_lines') or not self.plotted_lines:
            QMessageBox.warning(self, "No Lines Selected", "Please first select HITRAN lines to fit.")
            return
        
        if not hasattr(self, 'cube_data') or self.cube_data is None or not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            return
        
        spectrum_x = _extract_wavelength_value(self.spec_plotter.wavelength)
        spectrum_y = self.cube_data.spectrum
        
        if spectrum_x is None or spectrum_y is None or len(spectrum_x) != len(spectrum_y):
            QMessageBox.warning(self, "Data Error", "Data mismatch.")
            return
        
        # Auto-compute continuum with default values if not already fitted
        if not hasattr(self, 'continuum_fitted') or self.continuum_fitted is None:
            try:
                continuum_y = fit_continuum_median_peaks(spectrum_x, spectrum_y)
            except Exception as e:
                QMessageBox.warning(self, "Continuum Error", f"Could not compute continuum automatically:\n{str(e)}")
                return
        else:
            continuum_y = self.continuum_fitted
        
        fit_results = []
        for line in self.plotted_lines:
            try:
                line_center = line.get('lambda_micron') or line.get('wavelength')
                if line_center is None: continue
                line_center = float(line_center)
                
                x1, x2 = auto_detect_boundaries(line_center, spectrum_x, spectrum_y, continuum_y=continuum_y, threshold_percent=15.0)
                region_mask = (spectrum_x >= x1) & (spectrum_x <= x2)
                
                result = fit_gaussian_to_line(
                    x1, x2, spectrum_x[region_mask], spectrum_y[region_mask] - continuum_y[region_mask],
                    peak_x=line_center, noise_min=None, noise_max=None,
                    header=self.cube_data.header, wave_unit=self.cube_data.wave_unit
                )
                
                if result.success:
                    result.baseline = continuum_y[region_mask].copy()
                    fit_results.append({'line': line, 'result': result})
            except Exception:
                continue
        
        if not fit_results:
            QMessageBox.warning(self, "Fit Failed", "Could not fit any HITRAN lines.")
            return
        
        self.multi_gauss_results = fit_results
        self._plot_multi_gaussian_on_spectrum(fit_results)
        self._show_multi_gaussian_results(fit_results)

    def _plot_multi_gaussian_on_spectrum(self, fit_results):
        ax = self.spec_plotter.ax
        if not hasattr(self, 'multi_gauss_artists'): self.multi_gauss_artists = []
        for artist in self.multi_gauss_artists:
            try: artist.remove()
            except Exception: pass
        self.multi_gauss_artists = []
        
        norm_factor = 1.0
        if hasattr(self.spec_plotter, 'global_line') and self.spec_plotter.global_line is not None:
            global_y = self.spec_plotter.global_line.get_ydata()
            global_median = np.nanmedian(global_y[np.isfinite(global_y)])
            if 0.5 < global_median < 2.0:
                fit_median = np.nanmedian(np.concatenate([fit['result'].y_data for fit in fit_results]))
                norm_factor = fit_median if fit_median > 0 else 1.0
        
        for idx, fit_info in enumerate(fit_results):
            result = fit_info['result']
            baseline_norm = result.baseline / norm_factor
            y_fit_norm = result.y_fit / norm_factor
            
            if idx == 0:
                baseline_line, = ax.plot(result.x_data, baseline_norm, 'k--', linewidth=2, alpha=0.7, label='Baseline', zorder=10)
                self.multi_gauss_artists.append(baseline_line)
            
            gauss_line, = ax.plot(result.x_data, baseline_norm + y_fit_norm, 'r-', linewidth=3, alpha=0.9, label='Gaussian Fit' if idx == 0 else '', zorder=11)
            fill = ax.fill_between(result.x_data, baseline_norm, baseline_norm + y_fit_norm, color='red', alpha=0.2, zorder=9)
            self.multi_gauss_artists.extend([gauss_line, fill])
        
        ax.legend(loc='upper right', fontsize=8)
        self.spec_canvas.draw_idle()

    def _show_multi_gaussian_results(self, fit_results):
        wave_unit = clean_wave_unit(getattr(self.cube_data, 'wave_unit', 'µm'))
        flux_unit = clean_flux_unit(getattr(self.cube_data, 'flux_unit', 'MJy/sr'))
        fallback_area_unit = f"{flux_unit}·{wave_unit}"

        results_text = f"MULTI-GAUSSIAN FIT RESULTS\n{'='*70}\nSuccessfully fitted {len(fit_results)} lines\n\nFITTED LINES:\n{'='*70}\n"
        
        for idx, fit_info in enumerate(fit_results, 1):
            result, line = fit_info['result'], fit_info['line']
            results_text += f"""Line {idx}: {line.get('species', 'Unknown')}
  Center (λ):            {result.center:.6f} {wave_unit}
  Amplitude:             {result.amplitude:.4e} {flux_unit}
  Sigma (σ):             {result.sigma:.6g} {wave_unit}
  FWHM:                  {result.fwhm:.6g} {wave_unit}
  Integrated Area:       {result.area:.4e} {result.area_unit or fallback_area_unit}
  R²:                    {result.r_squared:.4f}\n  {'─'*60}\n"""
        
        results_text += f"\nOVERALL:\n  Total lines fitted:    {len(fit_results)}\n"
        
        dlg = QDialog(self)
        dlg.setWindowTitle("Multi-Gaussian Fit Results")
        dlg.setMinimumSize(750, 600)
        layout = QVBoxLayout()
        text_display = QTextEdit()
        text_display.setPlainText(results_text)
        text_display.setReadOnly(True)
        text_display.setFont(QFont('Courier', 9))
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
        if self.gauss_left_x is None or self.gauss_right_x is None:
            QMessageBox.warning(self, "Missing Boundaries", "Please select both LEFT and RIGHT boundaries first.")
            return
        
        if not hasattr(self, 'cube_data') or self.cube_data is None:
            return
        
        cube = self.cube_data.cube
        wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
        
        if cube is None or wavelength is None: return
        
        lam_min, lam_max = min(self.gauss_left_x, self.gauss_right_x), max(self.gauss_left_x, self.gauss_right_x)
        # if np.count_nonzero((wavelength >= lam_min) & (wavelength <= lam_max)) < 50:
        #     if QMessageBox.warning(self, "Narrow Fit Region", "Region is very narrow (<50 points). Continue?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
        #         return
        
        if self.gauss_noise_left_x is None or self.gauss_noise_right_x is None:
            QMessageBox.warning(self, "Noise Region Required", "You MUST define a noise region for map fitting.")
            return

        noise_min, noise_max = min(self.gauss_noise_left_x, self.gauss_noise_right_x), max(self.gauss_noise_left_x, self.gauss_noise_right_x)
        
        if (getattr(self, 'gauss_cache', None) is not None and 
            self.gauss_cache_lam_min == lam_min and self.gauss_cache_lam_max == lam_max and
            self.gauss_cache_noise_min == noise_min and self.gauss_cache_noise_max == noise_max):
            QMessageBox.information(self, "Cached Results", "Using previously fitted Gaussian map.")
            self._on_gaussian_map_complete(self.gauss_cache)
            return
        
        if QMessageBox.question(self, "Fit Map", "Continue with map fitting?", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes: return

        self.gauss_progress = QProgressDialog("Fitting Gaussians...", "Hide", 0, 0, self)
        self.gauss_progress.setWindowModality(Qt.NonModal)
        self.gauss_progress.show()

        self.lambda_0 = self._find_nearest_line_in_range(lam_min, lam_max) if hasattr(self, 'hitran_lines') else None
        
        self.gauss_worker = GaussianMapWorker(
            cube=cube, wavelength=wavelength, lam_min=lam_min, lam_max=lam_max, n_jobs=8,
            fac_above_stdres=3.0, noise_min=noise_min, noise_max=noise_max,
            header=self.cube_data.header, wave_unit=self.cube_data.wave_unit,
            flux_unit=self.cube_data.flux_unit, lambda_0=self.lambda_0
        )
        self.gauss_worker.finished.connect(self._on_gaussian_map_complete)
        self.gauss_worker.error.connect(self._on_gaussian_map_error)
        self.gauss_worker.start()
    
    def _on_gaussian_map_complete(self, results):
        self.gauss_progress.close()
        self.gauss_cache = results
        self.gauss_cache_lam_min = min(self.gauss_left_x, self.gauss_right_x)
        self.gauss_cache_lam_max = max(self.gauss_left_x, self.gauss_right_x)
        self.gauss_cache_noise_min = min(self.gauss_noise_left_x, self.gauss_noise_right_x)
        self.gauss_cache_noise_max = max(self.gauss_noise_left_x, self.gauss_noise_right_x)
        self._show_gaussian_map_results(results)
    
    def _on_gaussian_map_error(self, error_msg):
        self.gauss_progress.close()
        QMessageBox.critical(self, "Error", f"Gaussian map fitting error:\n{error_msg}")
    
    def _show_gaussian_map_results(self, results: dict):
        dlg = GaussianMapResultsDialog(self, results, self.cube_data, self.gauss_left_x, self.gauss_right_x, wavelength=self.spec_plotter.wavelength)
        dlg.exec_()