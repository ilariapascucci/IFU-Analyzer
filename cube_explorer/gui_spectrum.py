"""
Spectrum tab bar (Full / Left / Right) for the CubeExplorer main window.
"""

import numpy as np
from PyQt5.QtWidgets import QTabBar, QLineEdit


class GuiSpectrumMixin:
    """
    Mixin class handling the Full/Left/Right spectrum tab bar: which spectrum
    is the "active" one shown as the main trace on the spectrum plot, and
    inline renaming of tab labels.
    """

    def _init_spectrum_tabs(self):
        """Create the Full/Left/Right spectrum tab bar (labels are user-editable)."""
        self.spec_tab_bar = QTabBar()
        self.spec_tab_bar.setExpanding(False)
        self.spec_tab_bar.addTab("Full")
        self.spec_tab_bar.addTab("Left")
        self.spec_tab_bar.addTab("Right")
        self.spec_tab_bar.setCurrentIndex(0)
        self.spec_tab_bar.currentChanged.connect(self._on_spectrum_tab_changed)
        self.spec_tab_bar.tabBarDoubleClicked.connect(self._rename_spectrum_tab)

        # Index -> key mapping. Fixed order, tabs are never inserted/reordered,
        # so this stays valid even after the user renames a tab's label.
        self._spectrum_tab_keys = ["full", "left", "right"]
        self._spectrum_tab_rename_edit = None  # temporary QLineEdit while renaming

    def _on_spectrum_tab_changed(self, index):
        """Handle the user switching the active spectrum tab."""
        if 0 <= index < len(self._spectrum_tab_keys):
            self.active_spectrum_tab = self._spectrum_tab_keys[index]
            self._refresh_active_spectrum_tab()

    def _get_spectrum_for_tab(self, key):
        """Return the flux array that should be shown for the given tab key.

        "full" is always the original spectrum. "left"/"right" mirror that
        side's current selection (point, region, or - with no selection -
        falls back to the original spectrum, same as at launch).
        """
        if key == "full":
            return self.original_spectrum

        sel = self.last_selection_left if key == "left" else self.last_selection_right
        if not sel:
            return self.original_spectrum

        if sel.get("shape") == "*":
            x, y = sel["coords"]
            return self.cube_data.get_pixel_spectrum(x, y)

        mask = sel.get("mask")
        if mask is None:
            return self.original_spectrum
        return self.cube_data.compute_region_spectrum(mask)

    def _refresh_active_spectrum_tab(self):
        """Update the plot's main spectrum line to match the active tab.

        Updates spec_plotter.global_line in place instead of calling
        spec_plotter.plot_spectrum() (which does a full axis clear and would
        wipe the region-overlay lines and slice-indicator vlines - that method
        is meant for "a new cube was loaded," not a lightweight tab swap).
        """
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            return
        if not hasattr(self, 'active_spectrum_tab'):
            return
        if getattr(self, 'original_spectrum', None) is None:
            return

        spectrum = self._get_spectrum_for_tab(self.active_spectrum_tab)
        if spectrum is None:
            return

        line = getattr(self.spec_plotter, 'global_line', None)
        if line is None:
            return

        y = np.asarray(spectrum).astype(np.float32)
        x = line.get_xdata()
        if len(x) != len(y):
            # Shape mismatch (e.g. stale mask relative to the current cube) -
            # skip rather than plot mismatched axes.
            return
        line.set_ydata(y)

        ax = self.spec_plotter.ax
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        try:
            self.spec_canvas.draw()
        except Exception:
            pass

    def _rename_spectrum_tab(self, index):
        """Start inline editing of a tab's label (triggered by double-click)."""
        if self._spectrum_tab_rename_edit is not None:
            self._spectrum_tab_rename_edit.deleteLater()
            self._spectrum_tab_rename_edit = None

        rect = self.spec_tab_bar.tabRect(index)
        edit = QLineEdit(self.spec_tab_bar)
        edit.setText(self.spec_tab_bar.tabText(index))
        edit.selectAll()
        edit.setGeometry(rect)

        def commit():
            new_text = edit.text().strip()
            if new_text:
                self.spec_tab_bar.setTabText(index, new_text)
            edit.deleteLater()
            if self._spectrum_tab_rename_edit is edit:
                self._spectrum_tab_rename_edit = None

        edit.editingFinished.connect(commit)
        edit.show()
        edit.setFocus()
        self._spectrum_tab_rename_edit = edit
