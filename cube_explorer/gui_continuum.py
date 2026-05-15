import numpy as np
import traceback
from PyQt5.QtWidgets import QMessageBox, QApplication, QFileDialog
from scipy.ndimage import median_filter
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

from .continuum_dialog import ContinuumParametersDialog
from .custom_continuum import CustomContinuumSelector
from .unit_converter import clean_flux_unit, clean_wave_unit

def _extract_wavelength_value(wavelength_quantity_or_array, index=None):
    # Standard helper function used across the app
    from astropy.units import Quantity
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

class GuiContinuumMixin:
    """
    Mixin class handling all Continuum Subtraction logic for the CubeExplorer.
    Assumes it is mixed into a class that has self.spec_plotter, self.cube_data, etc.
    """

    def _on_continuum_line_clicked(self, checked):
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            self.continuum_line_btn.setChecked(False)
            return

        cube = getattr(self, 'cube_data', None) or getattr(self, 'cubedata', None)

        if cube is None or cube.spectrum is None:
            QMessageBox.warning(self, "No Spectrum", "No spectrum data available.")
            self.continuum_line_btn.setChecked(False)
            return
        
        if checked:
            if self.subtract_continuum_btn.isChecked():
                self._restore_original_spectrum()
                self.subtract_continuum_btn.setChecked(False)

            dialog = ContinuumParametersDialog(self)
            if dialog.exec_() == dialog.Accepted:
                params = dialog.get_parameters()
                self._fit_and_display_continuum(params)
            else:
                self.continuum_line_btn.setChecked(False)
        else:
            if hasattr(self, 'continuum_line') and self.continuum_line is not None:
                try:
                    self.continuum_line.remove()
                except Exception:
                    pass
                self.continuum_line = None

            if hasattr(self, 'spec_plotter') and self.spec_plotter is not None:
                self.spec_plotter.ax.legend(loc='upper right', fontsize=8)
                self.spec_canvas.draw()
            
            # Additional cleanup if needed
            if hasattr(self, 'continuum_subtracted') and self.continuum_subtracted:
                self._restore_original_spectrum()
                self.subtract_continuum_btn.setEnabled(False)
                self.subtract_continuum_btn.setChecked(False)

    def _fit_and_display_continuum(self, params=None):
        if params is None:
            params = {'window_size': 51, 'sigma_threshold': 2.0, 'peak_width': 1, 
                      'peak_distance': 3, 'peak_exclusion': 3, 'iterations': 2, 'smoothing_factor': 5}
        try:            
            wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
            cube = getattr(self, 'cube_data', None) or getattr(self, 'cubedata', None)
            if cube is None: return
            if self.original_spectrum_backup is not None:
                flux = self.original_spectrum_backup.copy()
            else:
                flux = cube.spectrum
            
            if wavelength is None or flux is None: return
            
            if self.original_spectrum_backup is None:
                self.original_spectrum_backup = cube.spectrum.copy()
            
            mask = np.isfinite(wavelength) & np.isfinite(flux)
            wavelength_clean = wavelength[mask]
            flux_clean = flux[mask]
            
            if len(wavelength_clean) < 10: return
            
            window_size = params['window_size']
            flux_smoothed = median_filter(flux_clean, size=window_size, mode='reflect')
            peaks, properties = find_peaks(
                flux_clean, prominence=np.std(flux_clean) * params['sigma_threshold'],
                width=params['peak_width'], distance=params['peak_distance']
            )

            exclude_mask = np.zeros(len(flux_clean), dtype=bool)
            for peak_idx in peaks:
                exclusion = params['peak_exclusion']
                left_edge = max(0, peak_idx - exclusion)
                right_edge = min(len(flux_clean), peak_idx + exclusion + 1)
                exclude_mask[left_edge:right_edge] = True
            
            flux_for_continuum = flux_clean.copy()
            good_indices = np.where(~exclude_mask)[0]
            if len(good_indices) < 10:
                exclude_mask = np.zeros(len(flux_clean), dtype=bool)
            
            flux_good = flux_clean.copy()
            flux_good[exclude_mask] = np.nan
            
            if np.any(exclude_mask):
                flux_good[exclude_mask] = np.interp(
                    wavelength_clean[exclude_mask], wavelength_clean[~exclude_mask], flux_clean[~exclude_mask]
                )
            
            continuum = median_filter(flux_good, size=window_size, mode='reflect')
            
            for iteration in range(params['iterations']):
                residuals = flux_clean - continuum
                sigma = np.std(residuals[~exclude_mask])
                outliers = (residuals > 2.0 * sigma) & (~exclude_mask)
                if np.sum(outliers) == 0: break
                exclude_mask |= outliers
                flux_good = flux_clean.copy()
                flux_good[exclude_mask] = np.interp(
                    wavelength_clean[exclude_mask], wavelength_clean[~exclude_mask], flux_clean[~exclude_mask]
                )
                continuum = median_filter(flux_good, size=window_size, mode='reflect')
            
            try:
                spline = UnivariateSpline(
                    wavelength_clean[~exclude_mask], continuum[~exclude_mask], 
                    s=len(wavelength_clean) / params['smoothing_factor'], k=3
                )
                continuum_smooth = spline(wavelength_clean)
            except Exception:
                continuum_smooth = continuum
            
            continuum_values = np.full_like(wavelength, np.nan)
            continuum_values[mask] = continuum_smooth
            self.continuum_fitted = continuum_values
            
            ax = self.spec_plotter.ax
            self.continuum_line = ax.plot(wavelength, continuum_values, color='orange',
                                        linewidth=2, linestyle='-', label='Continuum', zorder=10)[0]
            ax.legend(loc='upper right', fontsize=8)
            self.spec_canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Fitting Error", f"Error fitting continuum:\n{str(e)}")
            self.continuum_line_btn.setChecked(False)
            traceback.print_exc()

        if hasattr(self, 'subtract_continuum_btn'):
            self.subtract_continuum_btn.setEnabled(True)

    def _activate_continuum_subtraction(self):
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            self.subtract_continuum_btn.setChecked(False)
            return

        if self.subtract_continuum_btn.isChecked():
            if not hasattr(self, 'continuum_fitted') or self.continuum_fitted is None:
                QMessageBox.warning(self, "No Continuum", "Please create the continuum line first.")
                self.subtract_continuum_btn.setChecked(False)
                return

            if self.continuum_subtracted and hasattr(self, 'global_continuum_applied') and self.global_continuum_applied:
                self._restore_original_spectrum()
                self.subtract_continuum_btn.setChecked(False)
                return

            if self.custom_subtract_btn.isChecked() or (hasattr(self, 'custom_continuum_applied') and self.custom_continuum_applied):
                reply = QMessageBox.question(self, "Custom Active", "This will clear custom progress. Continue?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.subtract_continuum_btn.setChecked(False)
                    return
                if hasattr(self, 'custom_continuum_applied') and self.custom_continuum_applied:
                    self._restore_from_custom_subtract()
                self.custom_subtract_btn.setChecked(False)

            self._perform_global_continuum_subtraction()
            self.subtract_continuum_btn.setChecked(True)
        else:
            self._restore_original_spectrum()

    def _perform_global_continuum_subtraction(self):
        try:
            if self.original_spectrum_backup is None:
                self.original_spectrum_backup = self.cube_data.spectrum.copy()

            wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
            flux = self.original_spectrum_backup.copy()
            continuum = self.continuum_fitted

            if flux.shape != continuum.shape:
                QMessageBox.warning(
                    self, "Shape Mismatch",
                    f"Spectrum length {flux.shape} does not match the fitted "
                    f"continuum length {continuum.shape}.\n\n"
                    "Please refit the continuum and try again."
                )
                return

            flux_subtracted = flux - continuum
            self.cube_data.spectrum = flux_subtracted

            ax = self.spec_plotter.ax
            ax.clear()
            ax.plot(wavelength, flux_subtracted, 'b-', label='Continuum-Subtracted Spectrum',
                    linewidth=1, zorder=5)
            ax.axhline(0, color='lightgrey', linestyle=':', linewidth=1.5, alpha=0.8,
                    label='Zero Level', zorder=3)

            ax.relim()
            ax.autoscale_view()

            y_min, y_max = ax.get_ylim()
            x_min, x_max = ax.get_xlim()
            ax.set_ylim(y_min - (y_max - y_min) * 0.10, y_max + (y_max - y_min) * 0.10)
            ax.set_xlim(x_min - (x_max - x_min) * 0.01, x_max + (x_max - x_min) * 0.01)

            ax.set_xlabel(self.spec_plotter.ax.get_xlabel())
            ax.set_ylabel(self.spec_plotter.ax.get_ylabel())
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

            self.spec_canvas.draw()
            QApplication.processEvents()

            self.continuum_subtracted = True
            self.global_continuum_applied = True
            self.auto_continuum_subtracted = True

        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(
                self, "Subtraction Failed",
                f"Could not subtract the continuum:\n{e}"
            )
            self.subtract_continuum_btn.setChecked(False)

    def _activate_custom_continuum_subtraction(self):
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            self.custom_subtract_btn.setChecked(False)
            return

        if self.custom_subtract_btn.isChecked():
            if self.continuum_subtracted and hasattr(self, 'custom_continuum_applied') and self.custom_continuum_applied:
                self._restore_from_custom_subtract()
                self.custom_subtract_btn.setChecked(False)
                return

            if self.continuum_subtracted and self.original_spectrum_backup is not None:
                self.cube_data.spectrum = self.original_spectrum_backup.copy()
                self.continuum_subtracted = False

            if self.subtract_continuum_btn.isChecked():
                self.subtract_continuum_btn.setChecked(False)

            if self.original_spectrum_backup is None:
                self.original_spectrum_backup = self.cube_data.spectrum.copy()

            base_flux = (self.original_spectrum_backup.copy()
                         if self.original_spectrum_backup is not None
                         else self.cube_data.spectrum)

            if self.custom_continuum_selector is None:
                self.custom_continuum_selector = CustomContinuumSelector(
                    ax=self.spec_plotter.ax, canvas=self.spec_canvas,
                    wavelength=_extract_wavelength_value(self.spec_plotter.wavelength),
                    flux=base_flux, parent_widget=self
                )
            else:
                self.custom_continuum_selector.flux = base_flux
                self.custom_continuum_selector.wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
                self.custom_continuum_selector.ax = self.spec_plotter.ax

            self.custom_continuum_selector.activate()
            self.custom_continuum_active = True
            self.custom_delete_btn.setChecked(False)
            self.custom_continuum_selector.set_delete_mode(False)

        else:
            if self.continuum_subtracted and hasattr(self, 'custom_continuum_applied') and self.custom_continuum_applied:
                self._restore_from_custom_subtract()
            else:
                if self.custom_continuum_selector is not None:
                    self.custom_continuum_selector.deactivate()
                self.custom_continuum_active = False

    def _restore_from_custom_subtract(self):
        if self.original_spectrum_backup is None: return
        self.cube_data.spectrum = self.original_spectrum_backup.copy()
        self.spec_plotter.plot_spectrum(self.cube_data.spectrum)

        if self.continuum_fitted is not None and self.continuum_line_btn.isChecked():
            wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
            ax = self.spec_plotter.ax
            self.continuum_line = ax.plot(wavelength, self.continuum_fitted, color='orange',
                                          linewidth=2, linestyle='-', label='Continuum', zorder=10)[0]

        self.spec_plotter.ax.legend(loc='upper right', fontsize=8)
        self.spec_canvas.draw()
        self.continuum_subtracted = False
        self.custom_continuum_applied = False

    def _perform_custom_subtraction(self, continuum):
        try:
            if self.original_spectrum_backup is None:
                self.original_spectrum_backup = self.cube_data.spectrum.copy()

            subtracted = self.original_spectrum_backup - continuum
            self.cube_data.spectrum = subtracted
            self.continuum_fitted = continuum
            wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)

            ax = self.spec_plotter.ax
            ax.clear()
            ax.plot(wavelength, subtracted, color='blue', linewidth=1.0,
                    label='Custom Continuum-Subtracted Spectrum', zorder=5)
            ax.axhline(0, color='lightgrey', linestyle=':', linewidth=1.5, alpha=0.8,
                    label='Zero Level', zorder=3)

            ax.relim()
            ax.autoscale_view()

            y_min, y_max = ax.get_ylim()
            x_min, x_max = ax.get_xlim()
            ax.set_ylim(y_min - (y_max - y_min) * 0.10, y_max + (y_max - y_min) * 0.10)
            ax.set_xlim(x_min - (x_max - x_min) * 0.01, x_max + (x_max - x_min) * 0.01)

            ax.set_xlabel(self.spec_plotter.ax.get_xlabel())
            ax.set_ylabel(self.spec_plotter.ax.get_ylabel())
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.legend(loc='upper right', fontsize=8)
            self.spec_canvas.draw()

            self.continuum_subtracted = True
            self.custom_continuum_applied = True
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(
                self, "Subtraction Failed",
                f"Could not subtract the custom continuum:\n{e}"
            )

    def _restore_original_spectrum(self):
        if self.original_spectrum_backup is None: return
        self.cube_data.spectrum = self.original_spectrum_backup.copy()
        self.spec_plotter.plot_spectrum(self.cube_data.spectrum)

        if self.continuum_fitted is not None and self.continuum_line_btn.isChecked():
            wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
            ax = self.spec_plotter.ax
            self.continuum_line = ax.plot(wavelength, self.continuum_fitted, color='orange',
                                          linewidth=2, linestyle='-', label='Continuum', zorder=10)[0]

        if self.continuum_line is not None and not self.continuum_line_btn.isChecked():
            try: self.continuum_line.remove()
            except Exception: pass
            self.continuum_line = None

        self.spec_plotter.ax.legend(loc='upper right', fontsize=8)
        self.spec_canvas.draw()
        self.continuum_subtracted = False
        self.global_continuum_applied = False

    def on_auto_continuum_toggle(self):
        if self.auto_continuum_toggle.isChecked():
            if self.custom_continuum_toggle.isChecked():
                if self.has_custom_progress():
                    reply = QMessageBox.question(self, "Switch Mode?", "Switching to Automatic Continuum will clear Custom progress. Continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply == QMessageBox.No:
                        self.auto_continuum_toggle.setChecked(False)
                        return
                    else: self.clear_custom_progress()
                self.custom_continuum_toggle.setChecked(False)
            self.show_automatic_buttons()
        else:
            if self.has_automatic_progress():
                reply = QMessageBox.question(self, "Turn Off?", "Turning off Automatic Continuum will clear progress. Continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.auto_continuum_toggle.setChecked(True)
                    return
                else: self.clear_automatic_progress()
            self.hide_all_action_buttons()

    def on_custom_continuum_toggle(self):
        if self.custom_continuum_toggle.isChecked():
            if self.auto_continuum_toggle.isChecked():
                if self.has_automatic_progress():
                    reply = QMessageBox.question(self, "Switch Mode?", "Switching to Custom Continuum will clear Automatic progress. Continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply == QMessageBox.No:
                        self.custom_continuum_toggle.setChecked(False)
                        return
                    else: self.clear_automatic_progress()
                self.auto_continuum_toggle.setChecked(False)
            self.show_custom_buttons()
        else:
            if self.has_custom_progress():
                reply = QMessageBox.question(self, "Turn Off?", "Turning off Custom Continuum will clear progress. Continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.custom_continuum_toggle.setChecked(True)
                    return
                else: self.clear_custom_progress()
            self.hide_all_action_buttons()

    def show_automatic_buttons(self):
        self.hide_all_action_buttons()
        self.continuum_line_btn.show()
        self.subtract_continuum_btn.show()
        self.custom_ok_btn.hide()

    def show_custom_buttons(self):
        self.hide_all_action_buttons()
        self.custom_subtract_btn.show()
        self.custom_delete_btn.show()
        self.custom_save_pts_btn.show()
        self.custom_load_pts_btn.show()
        self.custom_ok_btn.show()

    def hide_all_action_buttons(self):
        self.continuum_line_btn.hide()
        self.subtract_continuum_btn.hide()
        self.custom_subtract_btn.hide()
        self.custom_delete_btn.hide()
        self.custom_save_pts_btn.hide()
        self.custom_load_pts_btn.hide()
        self.custom_ok_btn.hide()

    def has_automatic_progress(self):
        return (self.continuum_line_btn.isChecked() or self.subtract_continuum_btn.isChecked() or 
               (hasattr(self, 'continuum_fitted') and self.continuum_fitted is not None) or 
               (hasattr(self, 'continuum_subtracted') and self.continuum_subtracted))

    def has_custom_progress(self):
        return (self.custom_subtract_btn.isChecked() or 
               (hasattr(self, 'custom_continuum_active') and self.custom_continuum_active) or 
               (hasattr(self, 'custom_continuum_applied') and self.custom_continuum_applied))

    def clear_automatic_progress(self):
        try:
            self.continuum_line_btn.setChecked(False)
            self.subtract_continuum_btn.setChecked(False)
            if hasattr(self, 'continuum_line') and self.continuum_line is not None:
                try: self.continuum_line.remove()
                except: pass
                self.continuum_line = None
            if hasattr(self, 'continuum_subtracted') and self.continuum_subtracted:
                self._restore_original_spectrum()
            self.continuum_fitted = None
            self.continuum_subtracted = False
            if hasattr(self, 'global_continuum_applied'):
                self.global_continuum_applied = False
        except: pass

    def clear_custom_progress(self):
        try:
            self.custom_subtract_btn.setChecked(False)
            self.custom_delete_btn.setChecked(False)
            if hasattr(self, 'custom_continuum_selector') and self.custom_continuum_selector is not None:
                self.custom_continuum_selector.clear_all()
                self.custom_continuum_selector.deactivate()
            if hasattr(self, 'custom_continuum_applied') and self.custom_continuum_applied:
                self._restore_from_custom_subtract()
            self.custom_continuum_active = False
            self.custom_continuum_applied = False
        except: pass

    def _on_custom_delete_toggled(self, checked):
        if hasattr(self, 'custom_continuum_selector') and self.custom_continuum_selector is not None:
            self.custom_continuum_selector.set_delete_mode(checked)
    
    def _finalize_custom_continuum(self):
        """Keyboard ENTER alternative to clicking the OK button.

        custom_continuum.py's CustomContinuumSelector calls this when
        the user presses ENTER on the canvas. We just delegate to the
        existing OK-button handler so both paths behave identically.
        """
        self.on_custom_ok_clicked()

    def on_custom_ok_clicked(self):
        if not self.custom_continuum_active or self.custom_continuum_selector is None:
            return

        num_points = len(getattr(self.custom_continuum_selector, 'continuum_points', []))
        if num_points < 3:
            QMessageBox.warning(self, "Too Few Points",
                                "Place at least 3 continuum points before confirming.")
            return

        try:
            continuum = self.custom_continuum_selector.fit_continuum()
            if continuum is None:
                return

            wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
            ax = self.spec_plotter.ax

            # ── Remove any stale preview line from a previous OK click ──
            if hasattr(self, '_custom_preview_line') and self._custom_preview_line is not None:
                try:
                    self._custom_preview_line.remove()
                except Exception:
                    pass
                self._custom_preview_line = None

            # ── Draw the fitted continuum over the ORIGINAL spectrum ────
            self._custom_preview_line = ax.plot(
                wavelength, continuum,
                color='orange', linewidth=2, linestyle='-',
                label='Custom Continuum (preview)', zorder=10
            )[0]
            ax.legend(loc='upper right', fontsize=8)
            self.spec_canvas.draw()
            QApplication.processEvents()

            # ── Ask for confirmation ────────────────────────────────────
            reply = QMessageBox.question(
                self, "Confirm Subtraction",
                "The fitted continuum is shown in orange.\n\nSubtract it from the spectrum?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )

            # Always remove the preview line before doing anything next
            try:
                self._custom_preview_line.remove()
            except Exception:
                pass
            self._custom_preview_line = None

            if reply == QMessageBox.Yes:
                self._perform_custom_subtraction(continuum)
                self.custom_continuum_selector.deactivate()
                self.custom_continuum_active = False
            else:
                # User wants to keep editing — redraw clean spectrum with points still visible
                self.spec_canvas.draw()

        except Exception:
            traceback.print_exc()

    
    def _save_custom_points(self):
        if not hasattr(self, 'custom_continuum_active') or not self.custom_continuum_active or self.custom_continuum_selector is None:
            QMessageBox.warning(self, "Not Active", "Start a custom continuum session first by clicking 'Custom Subtract'.")
            return

        points = getattr(self.custom_continuum_selector, 'continuum_points', [])
        if len(points) == 0:
            QMessageBox.warning(self, "No Points", "There are no points on the graph to save.")
            return

        fname, _ = QFileDialog.getSaveFileName(self, "Save Continuum Points", filter="Text files (*.txt);;CSV files (*.csv)")
        if not fname:
            return

        try:
            with open(fname, 'w') as f:
                f.write("# Wavelength, Flux\n")
                for x, y in points:
                    f.write(f"{x:.6e}, {y:.6e}\n")
            QMessageBox.information(self, "Saved", f"Points successfully saved to:\n{fname}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save points:\n{str(e)}")

    def _load_custom_points(self):
        if not hasattr(self, 'custom_continuum_active') or not self.custom_continuum_active or self.custom_continuum_selector is None:
            QMessageBox.warning(self, "Not Active", "Please click 'Custom Subtract' to start a session before loading points.")
            return

        fname, _ = QFileDialog.getOpenFileName(self, "Load Continuum Points", filter="Text files (*.txt *.csv)")
        if not fname:
            return

        try:
            points = []
            with open(fname, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        points.append((x, y))

            if not points:
                QMessageBox.warning(self, "No Valid Points", "No valid coordinates found in the file.")
                return

            # Send the points to the selector to draw them
            self.custom_continuum_selector.load_points(points)
            QMessageBox.information(self, "Loaded", f"Successfully loaded {len(points)} points.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load points:\n{str(e)}")