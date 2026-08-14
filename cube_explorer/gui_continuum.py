import numpy as np
import traceback
from PyQt5.QtWidgets import QMessageBox, QApplication, QFileDialog

from .continuum_dialog import ContinuumParametersDialog
from .custom_continuum import CustomContinuumSelector
from .unit_converter import clean_flux_unit, clean_wave_unit
from .co_flux_extractor import fit_auto_continuum

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

    Continuum fitting/subtraction is independent per Full/Left/Right spectrum
    tab: self.continuum_fitted, self.continuum_subtracted, and
    self.continuum_mode_applied (all dicts keyed "full"/"left"/"right",
    initialized in gui.py's _init_state) are only ever read/written for
    whichever tab an operation targets, never a single shared state. Actually
    drawing the result is delegated to _refresh_active_spectrum_tab
    (GuiSpectrumMixin), which applies each tab's own continuum subtraction as
    one more per-tab transform - the same pipeline normalization/shifting
    already go through - instead of this module hand-drawing directly onto
    the shared spec_plotter axes.
    """

    def _on_continuum_line_clicked(self, checked):
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            self.continuum_line_btn.setChecked(False)
            return

        tab = self.active_spectrum_tab
        if self._get_tab_raw_flux(tab) is None:
            QMessageBox.warning(self, "No Spectrum", "No spectrum data available for this tab.")
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
            if self.continuum_subtracted.get(tab, False):
                self._restore_original_spectrum()
                self.subtract_continuum_btn.setEnabled(False)
                self.subtract_continuum_btn.setChecked(False)
            else:
                self._refresh_active_spectrum_tab()

    def _fit_and_display_continuum(self, params=None):
        if params is None:
            params = {'window_size': 51, 'sigma_threshold': 2.0, 'peak_width': 1,
                      'peak_distance': 3, 'peak_exclusion': 3, 'iterations': 2, 'smoothing_factor': 5}

        tab = self.active_spectrum_tab
        try:
            wavelength = self._get_tab_wavelength(tab)
            flux = self._get_tab_raw_flux(tab)

            if wavelength is None or flux is None:
                return

            # Same algorithm CO Flux Map/the API use (fit_auto_continuum) -- one
            # shared, GUI-independent implementation instead of two copies that could
            # silently drift apart.
            continuum_values = fit_auto_continuum(wavelength, flux, **params)
            self.continuum_fitted[tab] = continuum_values
            self.continuum_mode_applied[tab] = 'auto'
            self._refresh_active_spectrum_tab()

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

        tab = self.active_spectrum_tab

        if self.subtract_continuum_btn.isChecked():
            if self.continuum_fitted.get(tab) is None:
                QMessageBox.warning(self, "No Continuum", "Please create the continuum line first.")
                self.subtract_continuum_btn.setChecked(False)
                return

            if self.continuum_subtracted.get(tab, False) and self.continuum_mode_applied.get(tab) == 'auto':
                self._restore_original_spectrum()
                self.subtract_continuum_btn.setChecked(False)
                return

            if self.custom_subtract_btn.isChecked() or self.continuum_mode_applied.get(tab) == 'custom':
                reply = QMessageBox.question(self, "Custom Active", "This will clear custom progress. Continue?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.subtract_continuum_btn.setChecked(False)
                    return
                if self.continuum_mode_applied.get(tab) == 'custom':
                    self._restore_from_custom_subtract()
                self.custom_subtract_btn.setChecked(False)

            self._perform_global_continuum_subtraction()
            self.subtract_continuum_btn.setChecked(True)
        else:
            self._restore_original_spectrum()

    def _perform_global_continuum_subtraction(self):
        tab = self.active_spectrum_tab
        try:
            flux = self._get_tab_raw_flux(tab)
            continuum = self.continuum_fitted.get(tab)

            if flux is None or continuum is None:
                return

            if flux.shape != continuum.shape:
                QMessageBox.warning(
                    self, "Shape Mismatch",
                    f"Spectrum length {flux.shape} does not match the fitted "
                    f"continuum length {continuum.shape}.\n\n"
                    "Please refit the continuum and try again."
                )
                return

            self.continuum_subtracted[tab] = True
            self.continuum_mode_applied[tab] = 'auto'
            self._refresh_active_spectrum_tab()

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

        tab = self.active_spectrum_tab

        if self.custom_subtract_btn.isChecked():
            if self.continuum_subtracted.get(tab, False) and self.continuum_mode_applied.get(tab) == 'custom':
                self._restore_from_custom_subtract()
                self.custom_subtract_btn.setChecked(False)
                return

            if self.continuum_subtracted.get(tab, False):
                # An auto subtraction is active for this tab - drop it before starting a custom session.
                self.continuum_subtracted[tab] = False
                self.continuum_mode_applied[tab] = None

            if self.subtract_continuum_btn.isChecked():
                self.subtract_continuum_btn.setChecked(False)

            base_flux = self._get_tab_raw_flux(tab)
            base_wavelength = self._get_tab_wavelength(tab)
            if base_flux is None or base_wavelength is None:
                QMessageBox.warning(self, "No Spectrum", "No spectrum data available for this tab.")
                self.custom_subtract_btn.setChecked(False)
                return

            # Remember which tab this custom session belongs to - it stays
            # tied to that tab even if the user switches tabs mid-selection.
            self._custom_continuum_tab = tab

            if self.custom_continuum_selector is None:
                self.custom_continuum_selector = CustomContinuumSelector(
                    ax=self.spec_plotter.ax, canvas=self.spec_canvas,
                    wavelength=base_wavelength, flux=base_flux, parent_widget=self
                )
            else:
                self.custom_continuum_selector.flux = base_flux
                self.custom_continuum_selector.wavelength = base_wavelength
                self.custom_continuum_selector.ax = self.spec_plotter.ax

            self.custom_continuum_selector.activate()
            self.custom_continuum_active = True
            self.custom_delete_btn.setChecked(False)
            self.custom_continuum_selector.set_delete_mode(False)

        else:
            if self.continuum_subtracted.get(tab, False) and self.continuum_mode_applied.get(tab) == 'custom':
                self._restore_from_custom_subtract()
            else:
                if self.custom_continuum_selector is not None:
                    self.custom_continuum_selector.deactivate()
                self.custom_continuum_active = False

    def _restore_from_custom_subtract(self):
        tab = self._custom_continuum_tab if self._custom_continuum_tab is not None else self.active_spectrum_tab
        self.continuum_subtracted[tab] = False
        self.continuum_mode_applied[tab] = None
        self._refresh_active_spectrum_tab()

    def _perform_custom_subtraction(self, continuum):
        tab = self._custom_continuum_tab if self._custom_continuum_tab is not None else self.active_spectrum_tab
        try:
            self.continuum_fitted[tab] = continuum
            self.continuum_subtracted[tab] = True
            self.continuum_mode_applied[tab] = 'custom'
            self._refresh_active_spectrum_tab()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(
                self, "Subtraction Failed",
                f"Could not subtract the custom continuum:\n{e}"
            )

    def _restore_original_spectrum(self):
        tab = self.active_spectrum_tab
        self.continuum_subtracted[tab] = False
        self.continuum_mode_applied[tab] = None
        self._refresh_active_spectrum_tab()

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
        tab = self.active_spectrum_tab
        return (self.continuum_line_btn.isChecked() or self.subtract_continuum_btn.isChecked() or
               self.continuum_fitted.get(tab) is not None or
               self.continuum_subtracted.get(tab, False))

    def has_custom_progress(self):
        return (self.custom_subtract_btn.isChecked() or
               (hasattr(self, 'custom_continuum_active') and self.custom_continuum_active) or
               self.continuum_mode_applied.get(self.active_spectrum_tab) == 'custom')

    def clear_automatic_progress(self):
        try:
            tab = self.active_spectrum_tab
            self.continuum_line_btn.setChecked(False)
            self.subtract_continuum_btn.setChecked(False)
            if self.continuum_subtracted.get(tab, False):
                self._restore_original_spectrum()
            self.continuum_fitted[tab] = None
            self.continuum_subtracted[tab] = False
            self.continuum_mode_applied[tab] = None
            self._refresh_active_spectrum_tab()
        except Exception: pass

    def clear_custom_progress(self):
        try:
            self.custom_subtract_btn.setChecked(False)
            self.custom_delete_btn.setChecked(False)
            if hasattr(self, 'custom_continuum_selector') and self.custom_continuum_selector is not None:
                self.custom_continuum_selector.clear_all()
                self.custom_continuum_selector.deactivate()
            tab = self._custom_continuum_tab if self._custom_continuum_tab is not None else self.active_spectrum_tab
            if self.continuum_mode_applied.get(tab) == 'custom':
                self._restore_from_custom_subtract()
            self.custom_continuum_active = False
        except Exception: pass

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

            tab = self._custom_continuum_tab if self._custom_continuum_tab is not None else self.active_spectrum_tab
            wavelength = self._get_tab_wavelength(tab)
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
                # User wants to keep editing - redraw the clean tab with points still visible
                self._refresh_active_spectrum_tab()

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
