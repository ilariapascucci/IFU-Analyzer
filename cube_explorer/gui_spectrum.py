"""
Spectrum tab bar (Full / Left / Right) for the CubeExplorer main window.
"""

import numpy as np
from PyQt5.QtWidgets import QTabBar


class GuiSpectrumMixin:
    """
    Mixin class handling the Full/Left/Right spectrum tab bar. Each tab is
    fully isolated - Full always shows the original spectrum alone, Left/Right
    show only their own side's selection (nothing, if that side has none).
    """

    def _init_spectrum_tabs(self):
        """Create the Full/Left/Right spectrum tab bar."""
        self.spec_tab_bar = QTabBar()
        self.spec_tab_bar.setExpanding(False)
        self.spec_tab_bar.addTab("Full")
        self.spec_tab_bar.addTab("Left")
        self.spec_tab_bar.addTab("Right")
        self.spec_tab_bar.setCurrentIndex(0)
        self.spec_tab_bar.currentChanged.connect(self._on_spectrum_tab_changed)

        # Index -> key mapping. Fixed order, tabs are never inserted/reordered.
        self._spectrum_tab_keys = ["full", "left", "right"]
        # Dedicated Left/Right trace lines, created lazily in _ensure_tab_lines
        # (the "Full" trace reuses spec_plotter's own global_line).
        self._spec_tab_line_left = None
        self._spec_tab_line_right = None
        # Whether the user has explicitly picked a wavelength via "Set
        # Left/Right Slice" yet - the slice-indicator line for a side stays
        # hidden until then, rather than appearing at the default first-slice
        # position immediately on cube load.
        self._slice_indicator_set = {"left": False, "right": False}

    def _mark_slice_indicator_set(self, target):
        """Record that the user explicitly picked a wavelength for this side
        (called from _on_spec_click after update_slice_indicator) and refresh
        visibility so it now shows up on the appropriate tab(s)."""
        self._slice_indicator_set[target] = True
        self._refresh_active_spectrum_tab()

    def _reset_slice_indicators(self):
        """Reset indicator visibility state - call whenever a cube (re)loads."""
        self._slice_indicator_set = {"left": False, "right": False}

    def _on_spectrum_tab_changed(self, index):
        """Handle the user switching the active spectrum tab."""
        if 0 <= index < len(self._spectrum_tab_keys):
            self.active_spectrum_tab = self._spectrum_tab_keys[index]
            self._refresh_active_spectrum_tab()

    def _get_selection_spectrum(self, sel):
        """Compute the flux array for one stored selection dict (point or region).

        Region (non-point) spectra are scaled by pixel count to stay on a
        comparable absolute scale to the summed original spectrum - matching
        what the region-overlay line always showed before it was replaced by
        these tabs.
        """
        if not sel:
            return None
        if sel.get("shape") == "*":
            x, y = sel["coords"]
            return self.cube_data.get_pixel_spectrum(x, y)

        mask = sel.get("mask")
        if mask is None:
            return None
        mean_spectrum = self.cube_data.compute_region_spectrum(mask)
        if mean_spectrum is None:
            return None
        n_pixels = int(np.count_nonzero(mask))
        return mean_spectrum * float(n_pixels) if n_pixels else mean_spectrum

    def _get_active_tab_spectrum(self):
        """Return the flux array currently represented by the active tab, or
        None if that tab has nothing to show (Left/Right with no selection)."""
        tab = getattr(self, 'active_spectrum_tab', 'full')
        if tab == 'full':
            return getattr(self, 'original_spectrum', None)
        sel = self.last_selection_left if tab == 'left' else self.last_selection_right
        return self._get_selection_spectrum(sel)

    def _ensure_tab_lines(self):
        """Lazily (re)create the dedicated Left/Right trace lines, if missing
        or if the spectrum plotter's axes was replaced (e.g. a new cube was
        loaded, which builds a brand-new SpectrumPlotter/axes)."""
        # Darker shades of the slice-indicator colors (cyan/magenta), so the
        # region trace and its matching indicator line are visibly related
        # but still distinguishable when both are showing on the same tab.
        ax = self.spec_plotter.ax
        if self._spec_tab_line_left is None or self._spec_tab_line_left.axes is not ax:
            self._spec_tab_line_left, = ax.step([], [], where='mid', color='darkcyan', label='Left', zorder=2)
            self._spec_tab_line_left.set_visible(False)
        if self._spec_tab_line_right is None or self._spec_tab_line_right.axes is not ax:
            self._spec_tab_line_right, = ax.step([], [], where='mid', color='darkmagenta', label='Right', zorder=2)
            self._spec_tab_line_right.set_visible(False)

    def _refresh_active_spectrum_tab(self):
        """Show only the active tab's own trace; hide everything else.

        Full always shows only the original spectrum (spec_plotter.global_line).
        Left/Right show only their own dedicated line, and only if that side
        currently has a selection - otherwise the tab is empty (no fallback to
        Full). Slice-indicator lines only show once explicitly set, and only
        on the tab(s) they're relevant to (Left's line on Full+Left, Right's
        on Full+Right) - the OTHER side's indicator has no bearing on what
        you're looking at, so it stays hidden.
        """
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            return
        if not hasattr(self, 'active_spectrum_tab'):
            return

        global_line = getattr(self.spec_plotter, 'global_line', None)
        if global_line is None:
            return

        self._ensure_tab_lines()
        tab = self.active_spectrum_tab
        x = global_line.get_xdata()

        # Full is always just the original spectrum - never repurposed.
        global_line.set_visible(tab == "full")

        for side, line in (("left", self._spec_tab_line_left), ("right", self._spec_tab_line_right)):
            sel = self.last_selection_left if side == "left" else self.last_selection_right
            if tab == side and sel:
                y = self._get_selection_spectrum(sel)
                if y is not None and len(y) == len(x):
                    line.set_data(x, np.asarray(y).astype(np.float32))
                    line.set_visible(True)
                    continue
            line.set_visible(False)

        indicator_set = getattr(self, '_slice_indicator_set', {"left": False, "right": False})
        vline_left = getattr(self.spec_plotter, 'vline_left', None)
        vline_right = getattr(self.spec_plotter, 'vline_right', None)
        if vline_left is not None:
            vline_left.set_visible(indicator_set.get("left", False) and tab in ("full", "left"))
        if vline_right is not None:
            vline_right.set_visible(indicator_set.get("right", False) and tab in ("full", "right"))

        self._apply_shared_y_scale()

        # Rebuild the legend from only the currently-visible lines, so hidden
        # traces/indicators don't leave a stale entry behind.
        ax = self.spec_plotter.ax
        visible = [
            line for line in (global_line, self._spec_tab_line_left, self._spec_tab_line_right, vline_left, vline_right)
            if line is not None and line.get_visible()
        ]
        try:
            ax.legend(handles=visible, loc='upper right', fontsize=8)
        except Exception:
            pass

        try:
            self.spec_canvas.draw()
        except Exception:
            pass

    def _apply_shared_y_scale(self):
        """Set y-limits from the ORIGINAL spectrum's range, applied on every
        tab - not just whichever line happens to be visible.

        A region's flux can never exceed the full spectrum's (it's a subset
        of the same cube), so a single fixed scale lets you see at a glance
        how a region's peak compares to the total, instead of each tab
        auto-scaling to only its own data and hiding that relationship.
        """
        original = getattr(self, 'original_spectrum', None)
        if original is None:
            return
        y = np.asarray(original, dtype=float)
        finite = y[np.isfinite(y)]
        if finite.size == 0:
            return

        ymin, ymax = float(finite.min()), float(finite.max())
        pad = (ymax - ymin) * 0.05 if ymax > ymin else (abs(ymin) * 0.05 or 1.0)
        ymin, ymax = ymin - pad, ymax + pad

        ax = self.spec_plotter.ax
        ax.set_ylim(ymin, ymax)

        for vline in (self.spec_plotter.vline_left, self.spec_plotter.vline_right):
            if vline is not None:
                vline.set_ydata([ymin, ymax])
