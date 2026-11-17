import os
import numpy as np
import traceback
from matplotlib.patches import Rectangle
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import (QMessageBox, QApplication, QFileDialog, QProgressDialog, QPushButton, QButtonGroup,
                             QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLayout)

from .continuum_dialog import ContinuumParametersDialog, ContinuumCubeWorker
from .custom_continuum import CustomContinuumSelector
from .fit_file import check_same_flux_unit
from .flux_map_extractor import (fit_auto_continuum, NONE_LINES, UNKNOWN_LINES, automatic_lines, custom_lines,
                                 mask_file_lines, merge_mask_ranges, read_mask_file)
from .spectrum_file import normalize_wave_unit
from .utils import take_resolution_notes


class ContinuumPopup(QWidget):
    """
    A small tool window that holds the continuum buttons for one fitting mode.
    It does not block the main window, so the user can click the spectrum while it is open.
    """

    def __init__(self, title, parent=None):
        """Build an empty pop-up that closes with its window button.

        Args:
            title (str): Text shown in the title bar of the pop-up.
            parent (QWidget): Main window that the pop-up stays on top of.

        Returns:
            None
        """
        # A plain QWidget (not a QDialog), so Enter does not press a default button.
        super().__init__(parent, Qt.Tool | Qt.WindowCloseButtonHint)
        self.setWindowTitle(title)
        self._layout = QVBoxLayout(self)
        # The window follows its content when a row is shown or hidden.
        self._layout.setSizeConstraint(QLayout.SetFixedSize)
        self._placed = False

    def add_text(self, text):
        """Add a line of wrapped text to the pop-up.

        Args:
            text (str): Text of the label.

        Returns:
            label (QLabel): The label that was added.
        """
        label = QLabel(text)
        label.setWordWrap(True)
        # A wide window can give a wrapped label too little height, and the text is cut off.
        # A fixed width and a minimum height for that width prevent this.
        label.setFixedWidth(420)
        label.setMinimumHeight(label.heightForWidth(420))
        self._layout.addWidget(label)
        return label

    def add_row(self, widgets):
        """Add a horizontal row of widgets to the pop-up.

        Args:
            widgets (list): Widgets to place in the row, left to right.

        Returns:
            row (QWidget): The row. Show or hide it to show or hide all its widgets.
        """
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            row_layout.addWidget(widget)
        row_layout.addStretch()
        self._layout.addWidget(row)
        return row

    def show_below(self, anchor):
        """Show the pop-up and raise it. The first time, put it under the anchor widget.

        Args:
            anchor (QWidget): Widget that opens the pop-up.

        Returns:
            None
        """
        if not self._placed:
            self.move(anchor.mapToGlobal(QPoint(0, anchor.height())))
            self._placed = True
        self.show()
        self.raise_()
        self.activateWindow()


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
        """Handle the Fit Continuum toggle: fit and show, or clear it.

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
            # A mask that changes after the fit would not match the fitted continuum.
            self._end_mask_mode()
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
            params = self._continuum_params_with_masks(params, tab)
            continuum_values = fit_auto_continuum(wavelength, flux, **params)
            self.continuum_fitted[tab] = continuum_values
            self.continuum_mode_applied[tab] = 'auto'
            self._record_continuum(tab, continuum_values, automatic_lines(params), params)
            self._refresh_active_spectrum_tab()

        except Exception as e:
            QMessageBox.critical(self, "Fitting Error", f"Error fitting continuum:\n{str(e)}")
            self.continuum_line_btn.setChecked(False)
            traceback.print_exc()

        if hasattr(self, 'subtract_continuum_btn'):
            self.subtract_continuum_btn.setEnabled(True)

    def _record_continuum(self, tab, curve, lines, params=None):
        """Keep the header lines that say how a tab's continuum was made, with the curve they belong to.
        A saved file writes these lines, so the file tells which continuum it used.

        Args:
            tab (str): Spectrum tab, 'full', 'left' or 'right'
            curve (numpy.ndarray): The fitted continuum that the lines describe
            lines (list): The lines of text, without "#"
            params (dict): The fit_auto_continuum settings of an automatic fit, so that the lines can be
                written again after a Shift. None for a custom continuum.

        Returns:
            None
        """
        if not hasattr(self, '_continuum_records'):
            self._continuum_records = {}
        self._continuum_records[tab] = {'curve': curve, 'lines': list(lines),
                                        'params': dict(params) if params is not None else None}

    def _continuum_params_with_masks(self, params, tab):
        """Add the masked ranges of a tab to the settings of the automatic continuum.

        Args:
            params (dict): fit_auto_continuum() keyword arguments, or None for the defaults
            tab (str): Spectrum tab whose masks are added, 'full', 'left' or 'right'

        Returns:
            merged (dict): A copy of params with the key mask_ranges added. None if params is None and the tab has no masks.
        """
        merged = dict(params) if params else {}
        masks = self.continuum_masks.get(tab)
        if masks:
            merged['mask_ranges'] = list(masks)
        return merged or None

    def _reset_continuum_masks(self):
        """Remove every masked range of every tab, the waiting first click and the gray ranges on the plot.
        The Mask Spectrum button goes back to off.

        Args:
            None

        Returns:
            None
        """
        self._remove_mask_artists()
        self.continuum_masks = {"full": [], "left": [], "right": []}
        self._mask_pending = None
        button = getattr(self, 'mask_spectrum_btn', None)
        if button is not None:
            button.setChecked(False)

    def _remove_mask_artists(self):
        """Take the gray ranges and the line of a waiting first click off the spectrum plot.

        Args:
            None

        Returns:
            None
        """
        artists = list(getattr(self, '_mask_span_artists', []))
        artists.append(getattr(self, '_mask_pending_line', None))
        for artist in artists:
            if artist is not None and artist.axes is not None:
                artist.remove()
        self._mask_span_artists = []
        self._mask_pending_line = None

    def _draw_mask_spans(self):
        """Draw the masked ranges of the active tab as gray bands on the spectrum plot.
        A refresh of the active tab calls this, so the bands follow the tab that is showing.

        Args:
            None

        Returns:
            None
        """
        for artist in getattr(self, '_mask_span_artists', []):
            if artist.axes is not None:
                artist.remove()
        self._mask_span_artists = []
        ax = self.spec_plotter.ax
        for low, high in self.continuum_masks.get(self.active_spectrum_tab, []):
            # Only the first band has a label, so the legend has one entry. The label puts it in the Continuum group.
            band = Rectangle((low, 0), high - low, 1, transform=ax.get_xaxis_transform(),
                             facecolor='gray', edgecolor='none', alpha=0.25, zorder=1,
                             label='Continuum Mask' if not self._mask_span_artists else '')
            ax.add_patch(band)
            self._mask_span_artists.append(band)

    def _cancel_pending_mask(self):
        """Forget a first click that waits for its second click, and remove its line from the plot.

        Args:
            None

        Returns:
            None
        """
        self._mask_pending = None
        line = getattr(self, '_mask_pending_line', None)
        if line is not None and line.axes is not None:
            line.remove()
        self._mask_pending_line = None

    def _end_mask_mode(self):
        """Switch Mask Spectrum off. A first click that waits for its second click is forgotten.

        Args:
            None

        Returns:
            None
        """
        had_pending = self._mask_pending is not None
        self.mask_spectrum_btn.setChecked(False)
        self._cancel_pending_mask()
        if had_pending:
            self.spec_canvas.draw_idle()

    def _on_mask_spectrum_clicked(self, checked):
        """Handle the Mask Spectrum toggle. While it is on, two clicks on the spectrum mask the range between them.
        A click that lands inside a masked range removes that range.

        Args:
            checked (bool): New checked state of the toggle.

        Returns:
            None
        """
        if not checked:
            self._cancel_pending_mask()
            self.spec_canvas.draw_idle()
            return
        if getattr(self, 'spec_plotter', None) is None or self._get_tab_raw_flux(self.active_spectrum_tab) is None:
            QMessageBox.warning(self, "No Spectrum", "No spectrum data available for this tab.")
            self.mask_spectrum_btn.setChecked(False)
            return
        # Every click mode uses the spectrum clicks, so only one of them can be on.
        for button in (self.mark_contam_btn, self.gauss_left_btn, self.gauss_peak_btn, self.gauss_right_btn,
                       self.gauss_noise_left_btn, self.gauss_noise_right_btn):
            button.setChecked(False)
        self.gauss_mode = None

    def _handle_mask_click(self, wavelength):
        """Use one click of Mask Spectrum. The first click of a pair marks one end of a range.
        The second click closes the range. A click inside a masked range removes it when no first click waits.

        Args:
            wavelength (float): Wavelength that was clicked, on the axis of the active tab.

        Returns:
            None
        """
        tab = self.active_spectrum_tab
        if self._mask_pending is not None and self._mask_pending[0] != tab:
            self._cancel_pending_mask()

        if self._mask_pending is None:
            # A fitted continuum would not match the new masks, so it goes before any change.
            if not self._drop_automatic_continuum_for_masks():
                return
            inside = [m for m in self.continuum_masks[tab] if m[0] <= wavelength <= m[1]]
            if inside:
                self.continuum_masks[tab] = [m for m in self.continuum_masks[tab] if m not in inside]
                self._refresh_active_spectrum_tab()
                return
            self._mask_pending = (tab, float(wavelength))
            self._mask_pending_line = self.spec_plotter.ax.axvline(
                wavelength, color='gray', linestyle='--', linewidth=1.5, zorder=3)
            self.spec_canvas.draw_idle()
            return

        first = self._mask_pending[1]
        self._cancel_pending_mask()
        if first != wavelength:
            self.continuum_masks[tab] = merge_mask_ranges(self.continuum_masks[tab] + [(first, float(wavelength))])
        self._refresh_active_spectrum_tab()

    def _on_clear_masks_clicked(self):
        """Handle Clear Masks: remove every masked range of the active tab.

        Args:
            None

        Returns:
            None
        """
        tab = self.active_spectrum_tab
        if not self.continuum_masks[tab]:
            self._cancel_pending_mask()
            self._refresh_active_spectrum_tab()
            return
        if not self._drop_automatic_continuum_for_masks():
            return
        self._cancel_pending_mask()
        self.continuum_masks[tab] = []
        self._refresh_active_spectrum_tab()

    def _save_masks(self, _checked=False):
        """Save the masked ranges of the active tab to a tab-separated text file. The file also gives
        the wavelength unit and the Doppler shift factor, so that Load Masks can put the ranges on the right axis.

        Args:
            _checked (bool): Checked state that the button click passes. Not used.

        Returns:
            None
        """
        tab = self.active_spectrum_tab
        ranges = self.continuum_masks[tab]
        if not ranges:
            QMessageBox.warning(self, "No Masks", "There are no masked ranges on this spectrum to save.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Masks", f"masks_{tab}.txt", "Text files (*.txt)")
        if not fname:
            return
        if not os.path.splitext(fname)[1]:
            fname += ".txt"
        try:
            lines = mask_file_lines(ranges, getattr(self.cube_data, 'wave_unit', None),
                                    self.spectrum_shift_factor.get(tab, 1.0))
            with open(fname, 'w') as f:
                f.write("\n".join(lines) + "\n")
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Error", f"Failed to save the masks:\n{error}")
            return
        QMessageBox.information(self, "Saved", f"Masks saved to:\n{fname}")

    def _masks_on_current_axis(self, ranges, file_unit, file_shift, tab):
        """Move the ranges of a masks file onto the wavelength axis that a tab has now.
        The unit must be the same kind of unit. The Doppler shift can be different.

        Args:
            ranges (list): (low, high) tuples from the file
            file_unit (str): Wavelength unit of the file, or None if the file does not give it
            file_shift (float): Doppler shift factor of the file, or None if the file does not give it
            tab (str): Spectrum tab that gets the ranges, 'full', 'left' or 'right'

        Returns:
            result (tuple): (ranges, notes). ranges is the list of (low, high) tuples on the axis of the tab.
                notes is a list of str that the user must know.

        Raises:
            ValueError: if the unit of the file is not a unit of length that this program uses,
                or if it is not the unit of the spectrum
        """
        unit = getattr(self.cube_data, 'wave_unit', None)
        notes = []
        factor = 1.0
        if file_unit is None:
            notes.append(f"The file does not give its wavelength unit. The ranges are used as {unit}.")
        else:
            file_name, file_factor = normalize_wave_unit(file_unit)
            name, factor_now = normalize_wave_unit(unit)
            if file_name != name:
                raise ValueError(f"The masks are in {file_unit}, but the spectrum is in {unit}.")
            factor = file_factor / factor_now
        if file_shift is not None:
            factor *= self.spectrum_shift_factor.get(tab, 1.0) / file_shift
        moved = [(low * factor, high * factor) for low, high in ranges]

        wavelength = np.asarray(self._get_tab_wavelength(tab), dtype=float)
        lowest, highest = np.nanmin(wavelength), np.nanmax(wavelength)
        outside = [pair for pair in moved if pair[1] < lowest or pair[0] > highest]
        if outside:
            notes.append(f"{len(outside)} of {len(moved)} range(s) are outside the wavelength range of this "
                         f"spectrum ({lowest:.4g} to {highest:.4g} {unit}). They have no effect.")
        return moved, notes

    def _load_masks(self, _checked=False):
        """Load masked ranges from a file that Save Masks wrote. The ranges replace the masks of the active tab.
        A fitted automatic continuum is removed first, as it is for any change of the masks.

        Args:
            _checked (bool): Checked state that the button click passes. Not used.

        Returns:
            None
        """
        tab = self.active_spectrum_tab
        if getattr(self, 'spec_plotter', None) is None or self._get_tab_raw_flux(tab) is None:
            QMessageBox.warning(self, "No Spectrum", "No spectrum data available for this tab.")
            return
        fname, _ = QFileDialog.getOpenFileName(self, "Load Masks", filter="Text files (*.txt *.csv)")
        if not fname:
            return
        try:
            ranges, file_unit, file_shift = read_mask_file(fname)
            ranges, notes = self._masks_on_current_axis(ranges, file_unit, file_shift, tab)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Error", f"Failed to load the masks:\n{error}")
            return

        if not self._drop_automatic_continuum_for_masks():
            return
        self._cancel_pending_mask()
        self.continuum_masks[tab] = merge_mask_ranges(ranges)
        self._refresh_active_spectrum_tab()
        if notes:
            QMessageBox.information(self, "Masks Loaded", "\n".join(notes))

    def _drop_automatic_continuum_for_masks(self):
        """Remove the fitted automatic continuum of the active tab before its masks change.
        A continuum that is subtracted needs the user to agree first.

        Args:
            None

        Returns:
            agreed (bool): True if the masks can change. False if the user said no.
        """
        tab = self.active_spectrum_tab
        if self.continuum_fitted.get(tab) is None or self.continuum_mode_applied.get(tab) == 'custom':
            return True
        if self.continuum_subtracted.get(tab, False):
            reply = QMessageBox.question(
                self, "Change the Masks?",
                "The continuum is subtracted from this spectrum. A change of the masks removes the fitted "
                "continuum and the subtraction. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return False
        self._drop_automatic_continuum()
        return True

    def _drop_automatic_continuum(self):
        """Remove the fitted continuum of the active tab and undo its subtraction.
        The masks stay.

        Args:
            None

        Returns:
            None
        """
        tab = self.active_spectrum_tab
        self.continuum_line_btn.setChecked(False)
        self.subtract_continuum_btn.setChecked(False)
        if self.continuum_subtracted.get(tab, False):
            self._restore_original_spectrum()
        self.continuum_fitted[tab] = None
        self.continuum_subtracted[tab] = False
        self.continuum_mode_applied[tab] = None
        self._refresh_active_spectrum_tab()

    def _scale_continuum_masks(self, tab, factor):
        """Move the masked ranges of a tab with a Doppler shift, so that they stay on the same part of the spectrum.
        The header lines of the fitted continuum are written again with the new ranges.

        Args:
            tab (str): Spectrum tab that was shifted, 'full', 'left' or 'right'
            factor (float): The wavelength axis of the tab was multiplied by this number

        Returns:
            None
        """
        self._cancel_pending_mask()
        self.continuum_masks[tab] = [(low * factor, high * factor) for low, high in self.continuum_masks[tab]]
        record = getattr(self, '_continuum_records', {}).get(tab)
        if record is not None and record.get('params') is not None:
            record['params'] = self._continuum_params_with_masks(
                {k: v for k, v in record['params'].items() if k != 'mask_ranges'}, tab)
            record['lines'] = automatic_lines(record['params'])

    def _continuum_lines(self, tab):
        """Give the header lines for the continuum that a tab has now. The lines are used only if the
        record belongs to the stored curve. Without that, the lines say that the settings are unknown.

        Args:
            tab (str): Spectrum tab, 'full', 'left' or 'right'

        Returns:
            lines (list): The lines of text, without "#". "Continuum: none" if the tab has no continuum.
        """
        curve = self.continuum_fitted.get(tab)
        if curve is None:
            return list(NONE_LINES)
        record = getattr(self, '_continuum_records', {}).get(tab)
        if record is not None and record['curve'] is curve:
            return list(record['lines'])
        return list(UNKNOWN_LINES)

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
                QMessageBox.warning(self, "No Continuum", "Click Fit Continuum first.")
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

    def _activate_custom_continuum_subtraction(self, _checked=False, for_cubes=False):
        """Handle the Select Points toggle: start or end the point-placing session.
        Ending it removes the points and drops a custom continuum that is not subtracted.

        Args:
            _checked (bool): Checked state that the button click passes. Not used.
            for_cubes (bool): True when Save Continuum Cubes starts the session, so the
                points are for the cubes and Fit Continuum asks for the file name.

        Returns:
            None
        """
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            self.custom_subtract_btn.setChecked(False)
            return

        tab = self.active_spectrum_tab

        if self.custom_subtract_btn.isChecked():
            # A new session starts from nothing: drop the last custom continuum, subtracted or not.
            if self.continuum_mode_applied.get(tab) == 'custom':
                self._discard_custom_continuum(tab)

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

            self.custom_continuum_selector.activate(for_cubes)
            self.custom_continuum_active = True
            self._custom_fit_points = None
            self.custom_delete_btn.setChecked(False)
            self.custom_continuum_selector.set_delete_mode(False)

        else:
            if self.custom_continuum_selector is not None:
                self.custom_continuum_selector.clear_all()
                self.custom_continuum_selector.deactivate()
            self.custom_delete_btn.setChecked(False)
            self.custom_continuum_active = False
            session_tab = self._custom_continuum_tab if self._custom_continuum_tab is not None else tab
            if self.continuum_mode_applied.get(session_tab) == 'custom':
                self._discard_custom_continuum(session_tab)

    def _discard_custom_continuum(self, tab):
        """Remove a tab's custom continuum: the curve, the subtraction and the stored fit.

        Args:
            tab (str): 'full', 'left' or 'right'.

        Returns:
            None
        """
        self.continuum_fitted[tab] = None
        self.continuum_subtracted[tab] = False
        self.continuum_mode_applied[tab] = None
        self.custom_contsub_btn.setChecked(False)
        self._refresh_active_spectrum_tab()

    def _on_custom_contsub_clicked(self, checked):
        """Handle the Subtract Continuum toggle of the custom method. Pressing it subtracts
        the fitted curve and ends the point session. Pressing it again undoes the subtraction.

        Args:
            checked (bool): New checked state of the toggle.

        Returns:
            None
        """
        tab = self._custom_continuum_tab if self._custom_continuum_tab is not None else self.active_spectrum_tab
        if not checked:
            if self.continuum_subtracted.get(tab, False) and self.continuum_mode_applied.get(tab) == 'custom':
                self.continuum_subtracted[tab] = False
                self._refresh_active_spectrum_tab()
            return

        selector = self.custom_continuum_selector
        in_session = selector is not None and selector.is_active and self.custom_continuum_active
        if in_session and selector.for_cubes:
            QMessageBox.warning(self, "Points for the Cubes",
                                "These points are for the continuum cubes. Use Save Cubes in the Pixel-by-Pixel tools.")
            self.custom_contsub_btn.setChecked(False)
            return
        has_fit = self.continuum_mode_applied.get(tab) == 'custom' and self.continuum_fitted.get(tab) is not None
        if not has_fit:
            QMessageBox.warning(self, "No Continuum",
                                "Click Select Points, place at least 3 points, and click Fit Continuum first.")
            self.custom_contsub_btn.setChecked(False)
            return
        if in_session and list(selector.continuum_points) != getattr(self, '_custom_fit_points', None):
            QMessageBox.warning(self, "Points Changed",
                                "The points changed after the last fit. Click Fit Continuum again.")
            self.custom_contsub_btn.setChecked(False)
            return

        self._perform_custom_subtraction(self.continuum_fitted[tab])
        if in_session:
            selector.clear_all()
            selector.deactivate()
            self.custom_continuum_active = False
            self.custom_delete_btn.setChecked(False)
            self.custom_subtract_btn.setChecked(False)

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

    def _build_continuum_popups(self):
        """Build the two continuum pop-ups and move the continuum buttons into them.
        The Spectrum pop-up holds the buttons that fit the integrated spectrum.
        The Pixel-by-Pixel pop-up holds the buttons that save the continuum cubes.

        Args:
            None

        Returns:
            None
        """
        self.spectrum_continuum_popup = ContinuumPopup("Continuum: Integrated Spectrum", self)
        self.spectrum_continuum_popup.add_text(
            "Fit the continuum of the integrated spectrum and subtract it from the spectrum.")
        self.spectrum_continuum_popup.add_row([self.auto_continuum_toggle, self.custom_continuum_toggle])
        self.spectrum_continuum_popup.add_row([
            self.mask_spectrum_btn, self.clear_masks_btn, self.save_masks_btn, self.load_masks_btn,
            self.continuum_line_btn, self.subtract_continuum_btn, self.custom_subtract_btn,
            self.custom_delete_btn, self.custom_save_pts_btn, self.custom_load_pts_btn, self.custom_ok_btn,
            self.custom_contsub_btn])

        self.pixel_continuum_popup = ContinuumPopup("Continuum: Pixel-by-Pixel", self)
        self.pixel_continuum_popup.add_text(
            "Fit the continuum at every pixel. Then save the continuum cube and the "
            "continuum-subtracted cube.")
        self.pixel_auto_view_btn = QPushButton("Automatic")
        self.pixel_custom_view_btn = QPushButton("Custom")
        for btn in (self.pixel_auto_view_btn, self.pixel_custom_view_btn):
            btn.setCheckable(True)
        # These two buttons only choose which row is shown. They do not start a fit.
        view_group = QButtonGroup(self.pixel_continuum_popup)
        view_group.addButton(self.pixel_auto_view_btn)
        view_group.addButton(self.pixel_custom_view_btn)
        self.pixel_continuum_popup.add_row([self.pixel_auto_view_btn, self.pixel_custom_view_btn])

        self.pixel_auto_save_btn = QPushButton("Set Parameters and Save")
        self.pixel_auto_save_btn.setToolTip(
            "Set the automatic fit parameters, choose the file name, and save the continuum cubes")
        self.pixel_auto_save_btn.clicked.connect(self._save_continuum_cubes_automatic)
        self.pixel_auto_row = self.pixel_continuum_popup.add_row([self.pixel_auto_save_btn])

        self.pixel_points_btn = QPushButton("Select Points")
        self.pixel_points_btn.setToolTip(
            "Click continuum points on the spectrum. Each pixel gets a spline through its own flux at those wavelengths.")
        self.pixel_points_btn.clicked.connect(self._start_custom_points_for_cubes)
        self.pixel_delete_btn = QPushButton("Delete Point")
        self.pixel_delete_btn.setCheckable(True)
        self.pixel_delete_btn.setToolTip(self.custom_delete_btn.toolTip())
        # Delete Point keeps one state: this button follows the one in the Spectrum pop-up.
        self.pixel_delete_btn.clicked.connect(self.custom_delete_btn.click)
        self.custom_delete_btn.toggled.connect(self.pixel_delete_btn.setChecked)
        self.pixel_save_pts_btn = QPushButton("Save Points")
        self.pixel_save_pts_btn.setToolTip(self.custom_save_pts_btn.toolTip())
        self.pixel_save_pts_btn.clicked.connect(self._save_custom_points)
        self.pixel_load_pts_btn = QPushButton("Load Points")
        self.pixel_load_pts_btn.setToolTip(self.custom_load_pts_btn.toolTip())
        self.pixel_load_pts_btn.clicked.connect(lambda _checked=False: self._load_custom_points(for_cubes=True))
        self.pixel_cancel_btn = QPushButton("Cancel")
        self.pixel_cancel_btn.setToolTip("Remove the points and end the point selection")
        self.pixel_cancel_btn.clicked.connect(self._cancel_cube_points)
        self.pixel_save_cubes_btn = QPushButton("Save Cubes")
        self.pixel_save_cubes_btn.setToolTip("Choose the file name and save the continuum cubes")
        self.pixel_save_cubes_btn.clicked.connect(self._on_pixel_save_cubes_clicked)
        self.pixel_custom_row = self.pixel_continuum_popup.add_row([
            self.pixel_points_btn, self.pixel_delete_btn, self.pixel_save_pts_btn,
            self.pixel_load_pts_btn, self.pixel_cancel_btn, self.pixel_save_cubes_btn])

        # A click mode of the spectrum ends Mask Spectrum, because all of them use the clicks on the spectrum.
        for button in (self.mark_contam_btn, self.gauss_left_btn, self.gauss_peak_btn, self.gauss_right_btn,
                       self.gauss_noise_left_btn, self.gauss_noise_right_btn):
            button.toggled.connect(lambda checked: checked and self._end_mask_mode())

        self.pixel_auto_row.hide()
        self.pixel_custom_row.hide()
        self.pixel_auto_view_btn.toggled.connect(self.pixel_auto_row.setVisible)
        self.pixel_custom_view_btn.toggled.connect(self.pixel_custom_row.setVisible)

    def _open_spectrum_continuum_popup(self):
        """Open the Spectrum continuum pop-up. A Custom session for the cubes would
        change what the Fit Continuum button does, so it must end first.

        Args:
            None

        Returns:
            None
        """
        selector = self.custom_continuum_selector
        if selector is not None and selector.is_active and selector.for_cubes:
            reply = QMessageBox.question(
                self, "Cancel Pixel-by-Pixel Points?",
                "The points for the continuum cubes are still open. Opening the Integrated Spectrum tools will clear them. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
            self.clear_custom_progress()
            self.custom_continuum_toggle.setChecked(False)
            self.hide_all_action_buttons()
        self.spectrum_continuum_popup.show_below(self.continuum_spectrum_btn)

    def _open_pixel_continuum_popup(self):
        """Open the Pixel-by-Pixel continuum pop-up.

        Args:
            None

        Returns:
            None
        """
        self.pixel_continuum_popup.show_below(self.continuum_pixel_btn)

    def _continuum_cubes_ready(self):
        """Check that a cube is loaded and that no continuum cube save is running.

        Args:
            None

        Returns:
            ready (bool): True if a continuum cube save can start. A message tells the user why if not.
        """
        if getattr(self, 'cube_data', None) is None or self.cube_data.cube is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            return False
        worker = getattr(self, 'continuum_cube_worker', None)
        if worker is not None and worker.isRunning():
            QMessageBox.information(self, "Already Running", "A continuum cube save is already running.")
            return False
        return True

    def _save_continuum_cubes_automatic(self):
        """Ask for the automatic fit parameters, then save the continuum cubes.

        Args:
            None

        Returns:
            None
        """
        if not self._continuum_cubes_ready():
            return
        dialog = ContinuumParametersDialog(self)
        if dialog.exec_() != dialog.Accepted:
            return
        # The cubes use the masked ranges of the Full spectrum, on the wavelength axis of the cube.
        self._start_continuum_cube_save(self._continuum_params_with_masks(dialog.get_parameters(), "full"), None)

    def _on_pixel_save_cubes_clicked(self):
        """Handle Save Cubes in the Pixel-by-Pixel pop-up for the Custom fit.
        The points must be open first, because Save Cubes only saves the cubes for those points.

        Args:
            None

        Returns:
            None
        """
        selector = self.custom_continuum_selector
        if selector is None or not selector.is_active or not selector.for_cubes:
            QMessageBox.information(
                self, "Select Points First",
                "Click Select Points and place at least 3 continuum points on the spectrum. Then click Save Cubes.")
            return
        self.on_custom_ok_clicked()

    def _start_custom_points_for_cubes(self):
        """Open the Custom Continuum tool so the user can place the points for the cubes.
        Points already in a running session stay. The save starts when the user presses Save Cubes.

        Args:
            None

        Returns:
            None
        """
        if not self._continuum_cubes_ready():
            return

        # Turn the Custom Continuum tool on first. This can ask to clear Automatic progress.
        if not self.custom_continuum_toggle.isChecked():
            self.custom_continuum_toggle.setChecked(True)
            self.on_custom_continuum_toggle()
            if not self.custom_continuum_toggle.isChecked():
                return

        selector = self.custom_continuum_selector
        if selector is not None and selector.is_active and self.custom_continuum_active:
            selector.for_cubes = True
            QMessageBox.information(
                self, "Custom Continuum for the Cubes",
                "Your points stay. Add or remove points, then press Save Cubes to choose the file name.")
            return

        # A custom subtraction that is already applied would only be undone by the button.
        tab = self.active_spectrum_tab
        if self.continuum_subtracted.get(tab, False) and self.continuum_mode_applied.get(tab) == 'custom':
            self._restore_from_custom_subtract()
        self.custom_subtract_btn.setChecked(True)
        self._activate_custom_continuum_subtraction(for_cubes=True)

    def _finish_custom_points_for_cubes(self):
        """Use the points on the spectrum for the cubes, after the user pressed Save Cubes.
        If the user cancels the file name, the points stay and the user can keep editing.

        Args:
            None

        Returns:
            None
        """
        anchor_wavelengths = self._get_custom_anchor_wavelengths()
        if anchor_wavelengths is None:
            return
        if self._start_continuum_cube_save(None, anchor_wavelengths):
            self.custom_continuum_selector.deactivate()
            self.custom_continuum_active = False
            self.custom_subtract_btn.setChecked(False)

    def _start_continuum_cube_save(self, params, anchor_wavelengths):
        """Ask for the base file name, then start the background save of the two cubes.

        Args:
            params (dict): fit_auto_continuum() keyword arguments for the automatic fit, or None
            anchor_wavelengths (numpy.ndarray): Anchor wavelengths for the custom fit, or None

        Returns:
            started (bool): True if the save started, False if the user cancelled.
        """
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Continuum Cubes (base name)", "", "FITS files (*.fits)")
        if not path:
            return False
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
                return False

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
        take_resolution_notes()  # drop the warnings of an earlier run
        self.continuum_cube_worker = ContinuumCubeWorker(
            self.cube_data, continuum_path, contsub_path, continuum_params=params,
            anchor_wavelengths=anchor_wavelengths, n_jobs=8)
        self.continuum_cube_worker.finished.connect(self._on_continuum_cubes_complete)
        self.continuum_cube_worker.error.connect(self._on_continuum_cubes_error)
        self.continuum_cube_worker.progress.connect(self._on_continuum_cubes_progress)
        self.continuum_cube_worker.start()
        return True

    def _get_custom_anchor_wavelengths(self):
        """Get the anchor wavelengths for the custom cube fit, from the points on
        the spectrum. Only the wavelengths are used.

        Args:
            None

        Returns:
            wavelengths (numpy.ndarray): Anchor wavelengths in the cube's own wavelength array,
                or None if the anchors are not usable.
        """
        x_values = [x for x, _ in self.custom_continuum_selector.continuum_points]

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
                "Check the points, or the tab's wavelength shift.")
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
        resolution_note = take_resolution_notes()
        if resolution_note:
            QMessageBox.warning(self, "Spectral Resolution Held Flat", resolution_note)
        continuum_path, contsub_path = self._continuum_cube_paths
        message = f"Continuum cube:\n{continuum_path}\n\nContinuum-subtracted cube:\n{contsub_path}"
        # Skipped-pixel warning and 10-slice / 3-anchor note: off for now.
        # note = ("A pixel needs at least 10 valid slices for the automatic fit, or at least 3 valid anchors "
        #         "for the custom fit. A pixel with only a few more valid slices than that, usually at the "
        #         "edge of the field of view, still gets a continuum, but it can be unreliable.")
        # if n_failed:
        #     QMessageBox.warning(
        #         self, "Saved With Skipped Pixels",
        #         f"{message}\n\n{n_failed:,} pixels have data but got no continuum. "
        #         f"They are NaN in both cubes.\n\n{note}")
        # else:
        #     QMessageBox.information(self, "Continuum Cubes Saved", f"{message}\n\n{note}")
        QMessageBox.information(self, "Continuum Cubes Saved", message)

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
        self.mask_spectrum_btn.show()
        self.clear_masks_btn.show()
        self.save_masks_btn.show()
        self.load_masks_btn.show()
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
        self.custom_contsub_btn.show()

    def hide_all_action_buttons(self):
        """Hide every continuum action button.

        Args:
            None

        Returns:
            None
        """
        self.mask_spectrum_btn.hide()
        self.clear_masks_btn.hide()
        self.save_masks_btn.hide()
        self.load_masks_btn.hide()
        self.continuum_line_btn.hide()
        self.subtract_continuum_btn.hide()
        self.custom_subtract_btn.hide()
        self.custom_delete_btn.hide()
        self.custom_save_pts_btn.hide()
        self.custom_load_pts_btn.hide()
        self.custom_ok_btn.hide()
        self.custom_contsub_btn.hide()

    def has_automatic_progress(self):
        """Check whether the active tab has automatic-continuum progress.

        Args:
            None

        Returns:
            has_progress (bool): True if any automatic-continuum state exists.
        """
        tab = self.active_spectrum_tab
        return (self.continuum_line_btn.isChecked() or self.subtract_continuum_btn.isChecked() or
               self.mask_spectrum_btn.isChecked() or bool(self.continuum_masks.get(tab)) or
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
        """Reset the active tab's automatic-continuum state, with its masked ranges.

        Args:
            None

        Returns:
            None
        """
        try:
            self._end_mask_mode()
            self.continuum_masks[self.active_spectrum_tab] = []
            self._drop_automatic_continuum()
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
                self._discard_custom_continuum(tab)
            self.custom_contsub_btn.setChecked(False)
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

    def on_custom_ok_clicked(self):
        """Fit the custom continuum through the points and show it in orange.
        The points stay, so the user can edit them and fit again. The Subtract Continuum
        button subtracts the curve. When the points are for Save Continuum Cubes, this
        asks for the file name and saves the cubes instead.

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
                                "Place at least 3 continuum points before fitting.")
            return

        if self.custom_continuum_selector.for_cubes:
            self._finish_custom_points_for_cubes()
            return

        try:
            continuum = self.custom_continuum_selector.fit_continuum()
            if continuum is None:
                return

            tab = self._custom_continuum_tab if self._custom_continuum_tab is not None else self.active_spectrum_tab
            self.continuum_fitted[tab] = continuum
            self.continuum_mode_applied[tab] = 'custom'
            self._record_continuum(tab, continuum, custom_lines(self.custom_continuum_selector.continuum_points))
            # Subtract Continuum compares the points with these, so it never uses an old fit.
            self._custom_fit_points = list(self.custom_continuum_selector.continuum_points)
            self._refresh_active_spectrum_tab()

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Custom Continuum Error", f"Could not fit the custom continuum:\n{e}")


    def _save_custom_points(self):
        """Save the current custom-continuum points to a text/CSV file.

        Args:
            None

        Returns:
            None
        """
        if not hasattr(self, 'custom_continuum_active') or not self.custom_continuum_active or self.custom_continuum_selector is None:
            QMessageBox.warning(self, "Not Active", "Start a custom continuum session first. Click 'Select Points' or 'Load Points'.")
            return

        points = getattr(self.custom_continuum_selector, 'continuum_points', [])
        if len(points) == 0:
            QMessageBox.warning(self, "No Points", "There are no points on the graph to save.")
            return

        fname, _ = QFileDialog.getSaveFileName(self, "Save Continuum Points", filter="Text files (*.txt)")
        if not fname:
            return
        if not os.path.splitext(fname)[1]:
            fname += ".txt"

        try:
            with open(fname, 'w') as f:
                f.write("# Wavelength, Flux\n")
                f.write(f"# Flux unit: {getattr(self.cube_data, 'spectrum_flux_unit', '') or 'Unknown'}\n")
                for x, y in points:
                    f.write(f"{x:.6e}, {y:.6e}\n")
            QMessageBox.information(self, "Saved", f"Points successfully saved to:\n{fname}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save points:\n{str(e)}")

    def _custom_points_open(self):
        """Check whether a custom-continuum point selection is open.

        Args:
            None

        Returns:
            is_open (bool): True if the selector exists and is active.
        """
        selector = self.custom_continuum_selector
        return bool(self.custom_continuum_active) and selector is not None and selector.is_active

    def _load_custom_points(self, _checked=False, for_cubes=False):
        """Load custom-continuum points from a text/CSV file. The points replace the points on the plot.
        If no point selection is open, this opens it first, as the Select Points button does.

        Args:
            _checked (bool): Checked state that the button click passes. Not used.
            for_cubes (bool): True when the points are for Save Continuum Cubes. It matters only
                when the selection is not open yet.

        Returns:
            None
        """
        fname, _ = QFileDialog.getOpenFileName(self, "Load Continuum Points", filter="Text files (*.txt *.csv)")
        if not fname:
            return

        try:
            points = []
            file_unit = None
            with open(fname, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.lower().startswith('# flux unit:'):
                        file_unit = line.split(':', 1)[1].strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        points.append((x, y))

            if not points:
                QMessageBox.warning(self, "No Valid Points", "No valid coordinates found in the file.")
                return

            # The flux of the points is in the unit of the spectrum that they were clicked on.
            spectrum_unit = getattr(self.cube_data, 'spectrum_flux_unit', '')
            check_same_flux_unit(file_unit, spectrum_unit, what="points file")
            note = ""
            if file_unit is None:
                note = f"\nThe file does not give its flux unit. The points are used as {spectrum_unit or 'the unit of the spectrum'}."

            # The file is good, so the selection can open now. A message tells the user if it cannot.
            if not self._custom_points_open():
                if for_cubes:
                    self._start_custom_points_for_cubes()
                else:
                    self.custom_subtract_btn.setChecked(True)
                    self._activate_custom_continuum_subtraction()
                if not self._custom_points_open():
                    return

            # Send the points to the selector to draw them
            self.custom_continuum_selector.load_points(points)
            if note:
                QMessageBox.information(self, "Points Loaded", note.strip())

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load points:\n{str(e)}")

    def _cancel_cube_points(self):
        """Cancel the points for the continuum cubes. This removes the points and ends the point selection.
        It does nothing if the open points are not for the cubes.

        Args:
            None

        Returns:
            None
        """
        selector = self.custom_continuum_selector
        if selector is None or not selector.is_active or not selector.for_cubes:
            return
        self.clear_custom_progress()
