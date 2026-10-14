import os
import numpy as np
import traceback
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QApplication, QFileDialog, QProgressDialog

from .continuum_dialog import ContinuumParametersDialog, ContinuumCubeWorker
from .custom_continuum import CustomContinuumSelector
from .unit_converter import clean_flux_unit, clean_wave_unit
from .flux_map_extractor import fit_auto_continuum

def _extract_wavelength_value(wavelength_quantity_or_array, index=None):
    """Extract a numeric wavelength value from a Quantity or plain array.

    Args:
        wavelength_quantity_or_array: Wavelength Quantity, scalar, or array.
        index (int): Optional index to read; None reads the whole value.

    Returns:
        value: Numeric value(s) with units stripped if Quantity.
    """
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
        """Handle the Continuum Line toggle: fit and show, or clear it.

        Args:
            checked (bool): New checked state of the toggle.

        Returns:
            None
        """
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
        """Fit an automatic continuum for the active tab and display it.

        Args:
            params (dict): fit_auto_continuum() keyword arguments, or None
                for the default parameters.

        Returns:
            None
        """
        if params is None:
            params = {'window_size': 51, 'sigma_threshold': 2.0, 'peak_width': 1,
                      'peak_distance': 3, 'peak_exclusion': 3, 'iterations': 2, 'smoothing_factor': 5}

        tab = self.active_spectrum_tab
        try:
            wavelength = self._get_tab_wavelength(tab)
            flux = self._get_tab_raw_flux(tab)

            if wavelength is None or flux is None:
                return

            # Same algorithm Flux Map/the API use (fit_auto_continuum), so
            # there's one shared implementation instead of copies that drift apart.
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
        """Handle the Subtract Continuum toggle for the active tab.

        Args:
            None

        Returns:
            None
        """
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
        """Subtract the active tab's fitted continuum from its spectrum.

        Args:
            None

        Returns:
            None
        """
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
        """Handle the Custom Subtract toggle for the active tab.

        Args:
            None

        Returns:
            None
        """
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
        """Undo a custom continuum subtraction for its owning tab.

        Args:
            None

        Returns:
            None
        """
        tab = self._custom_continuum_tab if self._custom_continuum_tab is not None else self.active_spectrum_tab
        self.continuum_subtracted[tab] = False
        self.continuum_mode_applied[tab] = None
        self._refresh_active_spectrum_tab()

    def _perform_custom_subtraction(self, continuum):
        """Apply a custom-fitted continuum subtraction to its owning tab.

        Args:
            continuum (numpy.ndarray): Custom-fitted continuum to subtract.

        Returns:
            None
        """
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
        """Undo any continuum subtraction for the active tab.

        Args:
            None

        Returns:
            None
        """
        tab = self.active_spectrum_tab
        self.continuum_subtracted[tab] = False
        self.continuum_mode_applied[tab] = None
        self._refresh_active_spectrum_tab()

    def _save_continuum_cubes(self):
        """Fit the continuum at every pixel and save the continuum cube and the
        continuum-subtracted cube. Asks for the method (automatic or custom) and one base file name.

        Args:
            None

        Returns:
            None
        """
        if getattr(self, 'cube_data', None) is None or self.cube_data.cube is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            return
        worker = getattr(self, 'continuum_cube_worker', None)
        if worker is not None and worker.isRunning():
            QMessageBox.information(self, "Already Running", "A continuum cube save is already running.")
            return

        choice = QMessageBox(self)
        choice.setWindowTitle("Save Continuum Cubes")
        choice.setText("Which continuum should be fitted at every pixel?")
        choice.setInformativeText(
            "Automatic: the automatic fit, with parameters you set next.\n"
            "Custom: the wavelengths you clicked in the Custom Continuum tool. Each pixel gets "
            "a spline through its own flux at those wavelengths.")
        automatic_btn = choice.addButton("Automatic", QMessageBox.AcceptRole)
        custom_btn = choice.addButton("Custom", QMessageBox.AcceptRole)
        choice.addButton(QMessageBox.Cancel)
        choice.exec_()
        clicked = choice.clickedButton()

        params = None
        anchor_wavelengths = None
        if clicked is automatic_btn:
            dialog = ContinuumParametersDialog(self)
            if dialog.exec_() != dialog.Accepted:
                return
            params = dialog.get_parameters()
        elif clicked is custom_btn:
            anchor_wavelengths = self._get_custom_anchor_wavelengths()
            if anchor_wavelengths is None:
                return
        else:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Continuum Cubes (base name)", "", "FITS files (*.fits)")
        if not path:
            return
        base = path[:-5] if path.lower().endswith('.fits') else path
        continuum_path = base + "_continuum.fits"
        contsub_path = base + "_contsub.fits"
        existing = [p for p in (continuum_path, contsub_path) if os.path.exists(p)]
        if existing:
            reply = QMessageBox.question(
                self, "Overwrite?",
                "These files already exist:\n\n" + "\n".join(existing) + "\n\nOverwrite them?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        n_pixels = self.cube_data.cube.shape[1] * self.cube_data.cube.shape[2]
        self.continuum_cube_progress = QProgressDialog(
            f"Fitting the continuum at {n_pixels:,} pixels...", "Cancel", 0, 100, self)
        self.continuum_cube_progress.setWindowTitle("Save Continuum Cubes")
        self.continuum_cube_progress.setWindowModality(Qt.NonModal)
        self.continuum_cube_progress.setAutoClose(False)
        self.continuum_cube_progress.setAutoReset(False)
        self.continuum_cube_progress.setValue(0)
        self.continuum_cube_progress.canceled.connect(self._on_continuum_cubes_cancel_clicked)
        self.continuum_cube_progress.show()

        self._continuum_cube_paths = (continuum_path, contsub_path)
        self.continuum_cube_worker = ContinuumCubeWorker(
            self.cube_data, continuum_path, contsub_path, continuum_params=params,
            anchor_wavelengths=anchor_wavelengths, n_jobs=8)
        self.continuum_cube_worker.finished.connect(self._on_continuum_cubes_complete)
        self.continuum_cube_worker.error.connect(self._on_continuum_cubes_error)
        self.continuum_cube_worker.progress.connect(self._on_continuum_cubes_progress)
        self.continuum_cube_worker.start()

    def _get_custom_anchor_wavelengths(self):
        """Get the anchor wavelengths for the custom cube fit, from the points on
        the spectrum or from a saved points file. Only the wavelengths are used.

        Args:
            None

        Returns:
            wavelengths (numpy.ndarray): Anchor wavelengths in the cube's own wavelength array,
                or None if the user cancelled or the anchors are not usable.
        """
        selector = getattr(self, 'custom_continuum_selector', None)
        points = list(getattr(selector, 'continuum_points', None) or [])
        x_values = None
        if len(points) >= 3:
            reply = QMessageBox.question(
                self, "Custom Points",
                f"Use the {len(points)} custom continuum points currently on the spectrum?\n\n"
                "Click 'No' to load a saved points file instead.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
            if reply == QMessageBox.Cancel:
                return None
            if reply == QMessageBox.Yes:
                x_values = [x for x, _ in points]

        if x_values is None:
            fname, _ = QFileDialog.getOpenFileName(self, "Load Continuum Points", filter="Text files (*.txt *.csv)")
            if not fname:
                return None
            try:
                x_values = []
                with open(fname, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            x_values.append(float(line.split(',')[0]))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read points:\n{str(e)}")
                return None

        # Points are on the active tab's axis, which can be shifted. Map them to slices, then
        # to the cube's own wavelength array, so the export finds the same slices again.
        tab = self._custom_continuum_tab if getattr(self, '_custom_continuum_tab', None) is not None \
            else self.active_spectrum_tab
        tab_wavelength = self._get_tab_wavelength(tab)
        half_step = 0.5 * np.median(np.abs(np.diff(tab_wavelength)))
        x_values = np.asarray(x_values, dtype=float)
        if np.any((x_values < tab_wavelength.min() - half_step) | (x_values > tab_wavelength.max() + half_step)):
            QMessageBox.warning(
                self, "Points Outside Spectrum",
                "Some points are outside the spectrum shown in the active tab "
                f"({tab_wavelength.min():.5g} to {tab_wavelength.max():.5g}). "
                "Check the file, or the tab's wavelength shift.")
            return None
        slices = np.abs(tab_wavelength[None, :] - x_values[:, None]).argmin(axis=1)
        if len(np.unique(slices)) < 3:
            QMessageBox.warning(self, "Too Few Points", "Custom anchors need at least 3 different wavelengths.")
            return None
        wavelength = self.cube_data.wavelength
        return np.asarray(wavelength.value if hasattr(wavelength, 'value') else wavelength, dtype=float)[slices]

    def _on_continuum_cubes_cancel_clicked(self):
        """Stop a running continuum cube save. Qt already hides the dialog as
        part of the button click, so this only tells the worker to stop.

        Args:
            None

        Returns:
            None
        """
        worker = getattr(self, 'continuum_cube_worker', None)
        if worker is not None and worker.isRunning():
            worker.cancel()

    def _on_continuum_cubes_progress(self, current, total):
        """Mirror the worker's per-row progress into the progress dialog.

        Args:
            current (int): Pixels fit so far.
            total (int): Total pixels to fit.

        Returns:
            None
        """
        dlg = getattr(self, 'continuum_cube_progress', None)
        if dlg is None or total <= 0 or dlg.wasCanceled():
            return
        pct = int(100 * current / total)
        dlg.setLabelText(f"Fitting the continuum...\n{current:,} / {total:,} pixels ({pct}%)")
        dlg.setValue(pct)

    def _on_continuum_cubes_complete(self, n_failed):
        """Close the progress dialog and tell the user where the cubes were saved.

        Args:
            n_failed (int): Number of pixels that have data but got no continuum.

        Returns:
            None
        """
        self.continuum_cube_progress.close()
        continuum_path, contsub_path = self._continuum_cube_paths
        message = f"Continuum cube:\n{continuum_path}\n\nContinuum-subtracted cube:\n{contsub_path}"
        note = ("A pixel needs at least 10 valid slices for the automatic fit, or at least 3 valid anchors "
                "for the custom fit. A pixel with only a few more valid slices than that, usually at the "
                "edge of the field of view, still gets a continuum, but it can be unreliable.")
        if n_failed:
            QMessageBox.warning(
                self, "Saved With Skipped Pixels",
                f"{message}\n\n{n_failed:,} pixels have data but got no continuum. "
                f"They are NaN in both cubes.\n\n{note}")
        else:
            QMessageBox.information(self, "Continuum Cubes Saved", f"{message}\n\n{note}")

    def _on_continuum_cubes_error(self, message):
        """Close the progress dialog and show why the save failed.

        Args:
            message (str): Error message from the worker.

        Returns:
            None
        """
        self.continuum_cube_progress.close()
        QMessageBox.critical(self, "Save Failed", f"Could not save the continuum cubes:\n{message}")

    def on_auto_continuum_toggle(self):
        """Handle the Automatic Continuum group toggle.

        Args:
            None

        Returns:
            None
        """
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
        """Handle the Custom Continuum group toggle.

        Args:
            None

        Returns:
            None
        """
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
        """Show the automatic-continuum action buttons, hiding the rest.

        Args:
            None

        Returns:
            None
        """
        self.hide_all_action_buttons()
        self.continuum_line_btn.show()
        self.subtract_continuum_btn.show()
        self.custom_ok_btn.hide()

    def show_custom_buttons(self):
        """Show the custom-continuum action buttons, hiding the rest.

        Args:
            None

        Returns:
            None
        """
        self.hide_all_action_buttons()
        self.custom_subtract_btn.show()
        self.custom_delete_btn.show()
        self.custom_save_pts_btn.show()
        self.custom_load_pts_btn.show()
        self.custom_ok_btn.show()

    def hide_all_action_buttons(self):
        """Hide every continuum action button.

        Args:
            None

        Returns:
            None
        """
        self.continuum_line_btn.hide()
        self.subtract_continuum_btn.hide()
        self.custom_subtract_btn.hide()
        self.custom_delete_btn.hide()
        self.custom_save_pts_btn.hide()
        self.custom_load_pts_btn.hide()
        self.custom_ok_btn.hide()

    def has_automatic_progress(self):
        """Check whether the active tab has automatic-continuum progress.

        Args:
            None

        Returns:
            has_progress (bool): True if any automatic-continuum state exists.
        """
        tab = self.active_spectrum_tab
        return (self.continuum_line_btn.isChecked() or self.subtract_continuum_btn.isChecked() or
               self.continuum_fitted.get(tab) is not None or
               self.continuum_subtracted.get(tab, False))

    def has_custom_progress(self):
        """Check whether the active tab has custom-continuum progress.

        Args:
            None

        Returns:
            has_progress (bool): True if any custom-continuum state exists.
        """
        return (self.custom_subtract_btn.isChecked() or
               (hasattr(self, 'custom_continuum_active') and self.custom_continuum_active) or
               self.continuum_mode_applied.get(self.active_spectrum_tab) == 'custom')

    def clear_automatic_progress(self):
        """Reset the active tab's automatic-continuum state.

        Args:
            None

        Returns:
            None
        """
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
        """Reset the custom-continuum session's state.

        Args:
            None

        Returns:
            None
        """
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
        """Handle the Delete Point toggle for custom continuum selection.

        Args:
            checked (bool): New checked state of the toggle.

        Returns:
            None
        """
        if hasattr(self, 'custom_continuum_selector') and self.custom_continuum_selector is not None:
            self.custom_continuum_selector.set_delete_mode(checked)

    def _finalize_custom_continuum(self):
        """Keyboard ENTER alternative to clicking the OK button; delegates
        to the same handler so both paths behave identically.

        Args:
            None

        Returns:
            None
        """
        self.on_custom_ok_clicked()

    def on_custom_ok_clicked(self):
        """Preview the custom-fitted continuum and confirm before subtracting.

        Args:
            None

        Returns:
            None
        """
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

            # --- Remove any stale preview line from a previous OK click ---
            if hasattr(self, '_custom_preview_line') and self._custom_preview_line is not None:
                try:
                    self._custom_preview_line.remove()
                except Exception:
                    pass
                self._custom_preview_line = None

            # --- Draw the fitted continuum over the original spectrum ---
            self._custom_preview_line = ax.plot(
                wavelength, continuum,
                color='orange', linewidth=2, linestyle='-',
                label='Custom Continuum (preview)', zorder=10
            )[0]
            ax.legend(loc='upper right', fontsize=8)
            self.spec_canvas.draw()
            QApplication.processEvents()

            # --- Ask for confirmation ---
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
        """Save the current custom-continuum points to a text/CSV file.

        Args:
            None

        Returns:
            None
        """
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
        """Load custom-continuum points from a text/CSV file.

        Args:
            None

        Returns:
            None
        """
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
