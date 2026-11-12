"""
Load Fit and Load Fits: bring a saved Gaussian fit back on the spectrum, with no new fit.
"""

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QPushButton

from .fit_file import MARKER_NAMES, read_fit_file, rebuild_fit

# The marker of each name, as the Gaussian mixin keeps it: (attribute of the value, mode of _mark_gaussian_boundary).
_MARKER_MODES = {
    "fit_left": ("gauss_left_x", "left"), "peak": ("gauss_peak_x", "peak"), "fit_right": ("gauss_right_x", "right"),
    "noise_left": ("gauss_noise_left_x", "noise_left"), "noise_right": ("gauss_noise_right_x", "noise_right"),
}


# The button that loads each type of fit file, as the messages name it.
FIT_BUTTON_TEXT = {"single": "Load Fit, in the Gaussian Fit menu", "multi": "Load Fits, in the Lines & Fit menu"}


class GuiFitFileMixin:
    """
    Mixin class for the Load Fit button (one line) and the Load Fits button (multi-Gaussian). They read a fit
    file that Save Fit wrote, and show the fit on the active spectrum as if it had just been fitted.
    """

    def _init_fit_file_controls(self):
        """Create the Load Fit and Load Fits buttons.

        Args:
            None

        Returns:
            None
        """
        self.load_fit_btn = QPushButton("Load Fit")
        self.load_fit_btn.setToolTip(
            "Load a single-line fit that Save Fit wrote. It shows the fit and the Fit and Noise markers "
            "on this spectrum, with no new fit. The spectrum must have the same wavelengths as the fit. "
            "If no spectrum is open, the spectrum in the fit file is loaded first.")
        self.load_fit_btn.clicked.connect(self._load_single_fit)

        self.load_fits_btn = QPushButton("Load Fits")
        self.load_fits_btn.setToolTip(
            "Load a multi-Gaussian fit that Save Fit wrote. It shows the fits and the results on this "
            "spectrum, with no new fit. The spectrum must have the same wavelengths as the fit. "
            "If no spectrum is open, the spectrum in the fit file is loaded first.")
        self.load_fits_btn.clicked.connect(self._load_multi_fits)

    def _read_fit_for_active_tab(self, title, expected_type):
        """Ask for a fit file, read it, and build the fit again on the active spectrum. If no spectrum
        is open, the spectrum of the file is loaded first. Each problem is shown in a message box,
        and then the function returns None.

        Args:
            title (str): Title of the file dialog and of the message boxes
            expected_type (str): 'single' or 'multi'. A file of the other type is refused.

        Returns:
            loaded (tuple): (fit_file, entries, wavelength, spectrum, tab), or None if nothing was loaded
        """
        path, _ = QFileDialog.getOpenFileName(self, title, filter="Text files (*.txt);;All files (*)")
        if not path:
            return None
        try:
            fit_file = read_fit_file(path)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, title, f"The fit could not be loaded:\n{error}")
            return None
        if fit_file.fit_type != expected_type:
            QMessageBox.critical(self, title, f"This file is a {fit_file.fit_type}-line fit file. "
                                              f"Use {FIT_BUTTON_TEXT[fit_file.fit_type]}.")
            return None

        notes = []
        from_file = self.base_wavelength is None
        if from_file:
            opened = self._open_spectrum_file(path, title)
            if opened is None:
                return None
            notes.extend(opened[1])

        tab = self.active_spectrum_tab
        wavelength = self._get_tab_wavelength(tab)
        spectrum = self._get_active_tab_spectrum()
        if wavelength is None or spectrum is None:
            QMessageBox.warning(self, title, "There is no spectrum on this tab to put the fit on.")
            return None
        try:
            rebuilt = rebuild_fit(fit_file, wavelength, spectrum)
        except ValueError as error:
            QMessageBox.critical(self, title, f"The fit does not go on this spectrum:\n{error}")
            return None

        notes.extend(rebuilt["notes"])
        if fit_file.tab and fit_file.tab != tab and not from_file:
            notes.append(f"The fit was made on the {fit_file.tab} spectrum. It is loaded on the {tab} spectrum.")
        if notes:
            QMessageBox.warning(self, title, "\n".join(notes))
        return fit_file, rebuilt["entries"], wavelength, spectrum, tab

    def _set_fit_markers(self, markers):
        """Put the Fit and Noise markers of a loaded fit on the spectrum. The old markers are removed first.

        Args:
            markers (dict): Marker values with keys from MARKER_NAMES. A key that is missing gives no marker.

        Returns:
            None
        """
        for name in ("gauss_left_marker", "gauss_peak_marker", "gauss_right_marker",
                     "gauss_noise_left_marker", "gauss_noise_right_marker"):
            marker = getattr(self, name, None)
            if marker is not None:
                try:
                    marker.remove()
                except Exception:
                    pass
            setattr(self, name, None)
        for value_attr, _mode in _MARKER_MODES.values():
            setattr(self, value_attr, None)
        for name in MARKER_NAMES:
            if markers.get(name) is not None:
                self.gauss_mode = _MARKER_MODES[name][1]
                self._mark_gaussian_boundary(markers[name])
        self.gauss_mode = None

    def _load_single_fit(self):
        """Load a single-line fit file onto the active spectrum, and show its results.

        Args:
            None

        Returns:
            None
        """
        loaded = self._read_fit_for_active_tab("Load Fit", "single")
        if loaded is None:
            return
        fit_file, entries, wavelength, spectrum, tab = loaded
        result = entries[0]["result"]
        self._set_fit_markers(fit_file.markers)
        self.gauss_fit_input = (wavelength, spectrum)
        self.single_fit_by_tab[tab] = result
        self._refresh_active_spectrum_tab()
        self._display_gaussian_results(result, fit_file.baseline_source or "loaded from the fit file",
                                       fit_file.continuum_lines)

    def _load_multi_fits(self):
        """Load a multi-Gaussian fit file onto the active spectrum, and show its results.

        Args:
            None

        Returns:
            None
        """
        loaded = self._read_fit_for_active_tab("Load Fits", "multi")
        if loaded is None:
            return
        fit_file, entries, wavelength, spectrum, tab = loaded
        fit_results = [{"line": e["line"], "result": e["result"]} for e in entries if e["kind"] == "line"]
        contam_results = [{"line": e["line"], "result": e["result"]} for e in entries if e["kind"] == "contaminant"]
        failed_results = [{"line": e["line"], "result": e["result"]} for e in entries if e["kind"] == "skipped"]

        self.multi_gauss_contam_results = contam_results
        self.multi_gauss_fit_input = (wavelength, spectrum)
        self.multi_fit_by_tab[tab] = {"fit_results": fit_results, "contam_results": contam_results}
        # The cache belongs to a fit that ran here. A loaded fit has no cache key, so the old entry goes.
        self.multi_gauss_cache.pop(tab, None)
        self._refresh_active_spectrum_tab()
        self._show_multi_gaussian_results(fit_results, contam_results, None, failed_results=failed_results,
                                          continuum_lines=fit_file.continuum_lines)
