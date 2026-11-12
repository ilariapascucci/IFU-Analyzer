"""
Load a spectrum from a text file, and the spectrum-only mode of the main window.
A spectrum file has no cube. The slices, the regions and the per-pixel tools do not apply to it.
"""

import os

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QLabel, QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from .data_loader import CubeData
from .fit_file import fit_type_of_comments
from .gui_fit_file import FIT_BUTTON_TEXT
from .spectrum_file import default_settings, read_spectrum_table
from .spectrum_load_dialog import SpectrumLoadDialog
from .unit_converter import extract_value_from_quantity
from .visualization import SpectrumPlotter

# Buttons that need a cube. Each is turned off in spectrum-only mode.
_CUBE_ONLY_BUTTONS = ("gauss_map_btn", "flux_map_btn", "continuum_pixel_btn")
_CUBE_ONLY_TIP = "A spectrum file has no cube, so this tool is not available."


class GuiSpectrumFileMixin:
    """
    Mixin class for the Load Spectrum button. It loads a text file with a wavelength column,
    a flux column and an optional flux error column. The window then works on that one spectrum.
    """

    def _init_spectrum_file_controls(self):
        """Create the Load Spectrum button and the Error bars check box.

        Args:
            None

        Returns:
            None
        """
        self.load_spectrum_btn = QPushButton("Load Spectrum")
        self.load_spectrum_btn.setToolTip(
            "Load a spectrum from a text file (txt or csv) with wavelength, flux, and optional flux error. "
            "The continuum and Gaussian tools then work on it, with no FITS cube.")
        self.load_spectrum_btn.clicked.connect(self.load_spectrum)

        self.error_bars_checkbox = QCheckBox("Error bars")
        self.error_bars_checkbox.setChecked(True)
        self.error_bars_checkbox.setToolTip("Show the flux error of each point as a dot with a line above and below it.")
        self.error_bars_checkbox.toggled.connect(lambda _checked: self._refresh_active_spectrum_tab())
        self.error_bars_checkbox.setVisible(False)

        self._spectrum_only = False
        self._error_bars = None
        self._cube_only_tips = {}

    def load_spectrum(self):
        """Ask for a text file, read it, and show its spectrum. A file that Save Spectrum wrote
        loads with no question. For any other file, a dialog asks for the columns and the units.
        A fit file loads as a spectrum only. The message tells the user which button loads the fit.

        Args:
            None

        Returns:
            None
        """
        fname, _ = QFileDialog.getOpenFileName(
            self, "Load Spectrum", filter="Spectrum files (*.txt *.csv *.tsv *.dat);;All files (*)")
        if not fname:
            return

        opened = self._open_spectrum_file(fname, "Load Spectrum")
        if opened is None:
            return
        table, notes = opened
        fit_type = fit_type_of_comments(table.comments)
        if fit_type:
            notes.append(f"This file also holds a fit. Only the spectrum is loaded. "
                         f"To bring the fit back, use {FIT_BUTTON_TEXT[fit_type]}.")
        if notes:
            QMessageBox.information(self, "Load Spectrum", "\n".join(notes))

    def _open_spectrum_file(self, fname, title):
        """Read a text file and show its spectrum. Each problem is shown in a message box.
        A file that Save Spectrum wrote loads with no question. For any other file, a dialog asks.

        Args:
            fname (str): Path of the text file
            title (str): Title of the dialog and of the message boxes

        Returns:
            opened (tuple): (table, notes). table is the SpectrumTable that was read. notes is a list of
                str that tell what was changed in the data. Returns None if nothing was loaded.
        """
        try:
            table = read_spectrum_table(fname)
            settings = default_settings(table)
            if not settings.pop("complete"):
                dialog = SpectrumLoadDialog(table, self)
                if dialog.exec_() != QDialog.Accepted:
                    return None
                settings = dialog.settings()
            # Check the file on a spare object first, so a bad file leaves the window as it is.
            CubeData().load_from_table(table, settings)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, title, f"The spectrum could not be loaded:\n{error}")
            return None

        try:
            self._clear_all_displays()
            self.cube_data.free_memory()
            notes = self.cube_data.load_from_table(table, settings)
            self._reset_session_state()
            self._show_loaded_spectrum()
        except Exception as error:
            QMessageBox.critical(self, title, f"Error while showing the spectrum:\n{error}")
            return None

        return table, list(notes or [])

    def _reset_fit_state(self):
        """Clear the markers, fits, caches and plotted lines of the previous spectrum. They belong
        to the old wavelengths, so they cannot stay.

        Args:
            None

        Returns:
            None
        """
        for name in ("gauss_left_x", "gauss_peak_x", "gauss_right_x", "gauss_noise_left_x", "gauss_noise_right_x",
                     "gauss_mode", "gauss_left_marker", "gauss_peak_marker", "gauss_right_marker",
                     "gauss_noise_left_marker", "gauss_noise_right_marker", "gauss_cache", "gauss_cache_tab",
                     "gauss_cache_lam_min", "gauss_cache_lam_max", "gauss_cache_noise_min", "gauss_cache_noise_max"):
            setattr(self, name, None)
        self.gauss_boundaries = {"full": {}, "left": {}, "right": {}}
        self.single_fit_by_tab = {"full": None, "left": None, "right": None}
        self.multi_fit_by_tab = {"full": None, "left": None, "right": None}
        self.multi_gauss_cache = {}
        self.plotted_artists = []
        self.plotted_lines = []
        for button in (self.gauss_left_btn, self.gauss_peak_btn, self.gauss_right_btn,
                       self.gauss_noise_left_btn, self.gauss_noise_right_btn):
            button.setChecked(False)
        self._shared_y_applied = None

    def _show_loaded_spectrum(self):
        """Draw the spectrum that is now in cube_data, and switch the window to spectrum-only mode.

        Args:
            None

        Returns:
            None
        """
        cube_data = self.cube_data
        self.original_spectrum = np.array(cube_data.spectrum, dtype=float)
        self.base_wavelength = np.array(extract_value_from_quantity(cube_data.wavelength), dtype=float)

        name = os.path.basename(cube_data.filename)
        instrument = f"{cube_data.instrument}, " if cube_data.instrument else ""
        self.instrument_label.setText(f"Spectrum file: {instrument}{name}")

        self.spec_fig.clf()
        self.spec_plotter = SpectrumPlotter(self.spec_fig, cube_data.wavelength, cube_data.wave_unit)
        self.spec_plotter.cube_data = cube_data
        self.spec_toolbar.custom_home_action = self._spectrum_home_reset
        self.spec_plotter.plot_spectrum(cube_data.spectrum)
        self._reset_slice_indicators()
        self._set_spectrum_only_mode(True)
        self.spec_tab_bar.setCurrentIndex(0)
        self._spec_tab_line_left = None
        self._spec_tab_line_right = None
        self._continuum_line_artist = None
        self._continuum_zero_line = None
        self._refresh_active_spectrum_tab()
        self._fill_wave_range_inputs()

    def _set_spectrum_only_mode(self, on):
        """Turn spectrum-only mode on or off. In this mode the slice panels, the region tabs and the
        tools that need a cube are hidden or off. Off puts every one of them back.

        Args:
            on (bool): True for a loaded spectrum file, False for a FITS cube

        Returns:
            None
        """
        self._spectrum_only = bool(on)
        self.main_splitter.widget(0).setVisible(not on)
        for widget in (self.set_left_btn.parentWidget(), self.set_right_btn.parentWidget(),
                       self.spec_tab_bar.parentWidget(), self.overplot_left_checkbox, self.overplot_right_checkbox,
                       self.self_normalize_left_checkbox, self.self_normalize_right_checkbox):
            widget.setVisible(not on)
        for name in _CUBE_ONLY_BUTTONS:
            button = getattr(self, name)
            if on:
                self._cube_only_tips.setdefault(name, button.toolTip())
                button.setToolTip(_CUBE_ONLY_TIP)
            elif name in self._cube_only_tips:
                button.setToolTip(self._cube_only_tips[name])
            button.setEnabled(not on)
        # The Header button stays on. For a spectrum file it shows the comment lines of the file.
        if on:
            self._cube_only_tips.setdefault("header_btn", self.header_btn.toolTip())
            self.header_btn.setToolTip("Show the comment lines (the lines that start with #) of the spectrum file")
        elif "header_btn" in self._cube_only_tips:
            self.header_btn.setToolTip(self._cube_only_tips["header_btn"])
        if not on:
            self.error_bars_checkbox.setVisible(False)
            self._error_bars = None  # The figure is rebuilt by the cube load, so the bars are gone already.
            QTimer.singleShot(0, self._on_splitter_resized)

    def _show_spectrum_file_header(self):
        """Show the comment lines of the loaded spectrum file in a window. They are the lines
        that start with "#", as they are written in the file.

        Args:
            None

        Returns:
            None
        """
        comments = self.cube_data.header_comments
        name = os.path.basename(self.cube_data.filename)
        if not comments:
            QMessageBox.information(self, "Spectrum File Header",
                                    f"{name} has no comment lines. A comment line starts with #.")
            return

        self.header_window = QWidget()
        self.header_window.setWindowTitle(f"Spectrum File Header - {name}")
        self.header_window.setMinimumSize(700, 400)
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"<b>{name}</b>"))
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setLineWrapMode(QTextEdit.NoWrap)
        text_edit.setPlainText("\n".join(comments))
        layout.addWidget(text_edit)
        self.header_window.setLayout(layout)
        self.header_window.show()

    def _draw_error_bars(self):
        """Draw the flux error of the loaded spectrum as dots with vertical lines. The dots sit on the
        plotted spectrum, so they follow the Doppler shift, the continuum and the normalization.
        The Error bars check box turns them off.

        Args:
            None

        Returns:
            None
        """
        if self._error_bars is not None:
            try:
                self._error_bars.remove()
            except Exception:
                pass
            self._error_bars = None
        error = getattr(self.cube_data, 'spectrum_error', None)
        has_error = self._spectrum_only and error is not None
        self.error_bars_checkbox.setVisible(has_error)
        if not has_error or not self.error_bars_checkbox.isChecked():
            return

        x = self._get_tab_wavelength("full")
        y = self._apply_continuum("full", self.original_spectrum)
        divisor = self._normalization_divisor("full", None)
        self._error_bars = self.spec_plotter.ax.errorbar(
            x, np.asarray(y, dtype=float) / divisor, yerr=np.asarray(error, dtype=float) / divisor,
            fmt='o', markersize=2, color='blue', ecolor='steelblue', elinewidth=0.8, capsize=0,
            alpha=0.7, zorder=2)
