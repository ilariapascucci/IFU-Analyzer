"""
Spectrum tab bar (Full / Left / Right) for the CubeExplorer main window.
"""

import numpy as np
from matplotlib.text import Text
from PyQt5.QtWidgets import QTabBar, QCheckBox, QPushButton, QMenu

from .visualization import HIDDEN_LEGEND_GROUPS, LEGEND_GROUPS, draw_spectrum_legend


class GuiSpectrumMixin:
    """
    Mixin class handling the Full/Left/Right spectrum tab bar. Each tab is
    fully isolated - Full always shows the original spectrum alone, Left/Right
    show only their own side's selection (nothing, if that side has none).
    """

    def _init_spectrum_tabs(self):
        """Create the Full/Left/Right spectrum tab bar.

        Args:
            None

        Returns:
            None
        """
        self.spec_tab_bar = QTabBar()
        self.spec_tab_bar.setExpanding(False)
        self.spec_tab_bar.addTab("Full")
        self.spec_tab_bar.addTab("Left Region")
        self.spec_tab_bar.addTab("Right Region")
        self.spec_tab_bar.setCurrentIndex(0)
        self.spec_tab_bar.currentChanged.connect(self._on_spectrum_tab_changed)

        # Index -> key mapping. Fixed order, tabs are never inserted/reordered.
        self._spectrum_tab_keys = ["full", "left", "right"]
        # Dedicated Left/Right trace lines, created lazily in _ensure_tab_lines
        # (the "Full" trace reuses spec_plotter's own global_line).
        self._spec_tab_line_left = None
        self._spec_tab_line_right = None
        # Whether the user has picked a wavelength via "Set Left/Right Slice"
        # yet; the indicator line stays hidden until then, not at the default slice.
        self._slice_indicator_set = {"left": False, "right": False}

        # Normalize (global, divides everything by Full's own peak) and
        # overplot (Full-tab-only: overlay Left/Right's trace onto Full).
        self.spectrum_normalized = False
        self.overplot_left = False
        self.overplot_right = False

        self.normalize_checkbox = QCheckBox("Normalize")
        self.normalize_checkbox.setToolTip("Divide every spectrum by the peak of the full spectrum.")
        self.normalize_checkbox.toggled.connect(self._on_normalize_toggled)

        self.overplot_left_checkbox = QCheckBox("Show Left")
        self.overplot_left_checkbox.setToolTip(
            "Overlay the left region's spectrum on the Full tab. Select a region on the left image first.")
        self.overplot_left_checkbox.setEnabled(False)
        self.overplot_left_checkbox.toggled.connect(lambda checked: self._on_overplot_toggled("left", checked))

        self.overplot_right_checkbox = QCheckBox("Show Right")
        self.overplot_right_checkbox.setToolTip(
            "Overlay the right region's spectrum on the Full tab. Select a region on the right image first.")
        self.overplot_right_checkbox.setEnabled(False)
        self.overplot_right_checkbox.toggled.connect(lambda checked: self._on_overplot_toggled("right", checked))

        # Self-normalize overrides Normalize's "divide by Full's peak" per
        # side, dividing by that side's own peak; meaningful only while Normalize is on.
        self.self_normalize_left = False
        self.self_normalize_right = False

        self.self_normalize_left_checkbox = QCheckBox("Self-Normalize Left")
        self.self_normalize_left_checkbox.setToolTip(
            "Divide the left region's spectrum by its own peak. Select a region on the left image first.")
        self.self_normalize_left_checkbox.setEnabled(False)
        self.self_normalize_left_checkbox.toggled.connect(lambda checked: self._on_self_normalize_toggled("left", checked))

        self.self_normalize_right_checkbox = QCheckBox("Self-Normalize Right")
        self.self_normalize_right_checkbox.setToolTip(
            "Divide the right region's spectrum by its own peak. Select a region on the right image first.")
        self.self_normalize_right_checkbox.setEnabled(False)
        self.self_normalize_right_checkbox.toggled.connect(lambda checked: self._on_self_normalize_toggled("right", checked))

        # Per-tab Doppler shift: base_wavelength is the permanent reference;
        # spectrum_shift_factor/preview are each tab's own applied/pending shift.
        self.base_wavelength = None
        self.spectrum_shift_factor = {"full": 1.0, "left": 1.0, "right": 1.0}
        self.spectrum_shift_preview = {"full": None, "left": None, "right": None}
        self._shift_preview_line = None

        # Continuum-fit/preview artists, shared across tabs like
        # _shift_preview_line; per-tab dicts live in gui.py's _init_state.
        self._continuum_line_artist = None
        self._continuum_zero_line = None

    def _on_normalize_toggled(self, checked):
        """Handle the Normalize checkbox - a global toggle, not per-tab.

        Args:
            checked (bool): New checked state of the Normalize checkbox.

        Returns:
            None
        """
        self.spectrum_normalized = bool(checked)
        self._refresh_active_spectrum_tab()

    def _on_overplot_toggled(self, side, checked):
        """Handle the Show Left/Show Right checkboxes (Full-tab-only overlay).

        Args:
            side (str): 'left' or 'right'.
            checked (bool): New checked state of the checkbox.

        Returns:
            None
        """
        if side == "left":
            self.overplot_left = bool(checked)
        else:
            self.overplot_right = bool(checked)
        self._refresh_active_spectrum_tab()

    def _on_self_normalize_toggled(self, side, checked):
        """Handle the Self-Normalize Left/Right checkboxes.

        Args:
            side (str): 'left' or 'right'.
            checked (bool): New checked state of the checkbox.

        Returns:
            None
        """
        if side == "left":
            self.self_normalize_left = bool(checked)
        else:
            self.self_normalize_right = bool(checked)
        self._refresh_active_spectrum_tab()

    def _full_spectrum_peak(self):
        """Return the Full spectrum's own peak value, or None if unavailable.

        Args:
            None

        Returns:
            peak (float): The Full spectrum's maximum finite value, or None.
        """
        original = getattr(self, 'original_spectrum', None)
        if original is None:
            return None
        y = np.asarray(original, dtype=float)
        finite = y[np.isfinite(y)]
        if finite.size == 0:
            return None
        return float(finite.max())

    def _normalization_divisor(self, tab, y):
        """Find the number that the plot divides a tab's trace by. This is the
        one rule for the plotted spectrum and for the fit curves drawn on it.

        Args:
            tab (str): 'full', 'left' or 'right'.
            y (numpy.ndarray): The trace before normalization. Only Self-Normalize reads it.

        Returns:
            divisor (float): Full's peak, the trace's own peak, or 1.0 if no normalization.
        """
        # A non-positive peak means degenerate data; dividing by it would
        # flip every sign, so fall back to raw values instead.
        self_normalize = {"left": self.self_normalize_left, "right": self.self_normalize_right}.get(tab, False)
        if self_normalize and y is not None:
            arr = np.asarray(y, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0 or finite.max() <= 0:
                return 1.0
            return float(finite.max())
        if not self.spectrum_normalized:
            return 1.0
        peak = self._full_spectrum_peak()
        return peak if peak and peak > 0 else 1.0

    def _maybe_normalize(self, y):
        """Divide by Full's peak if Normalize is on; otherwise return as-is.
        The one shared rule: applied to Full it's self-normalization,
        applied to Left/Right it's "relative to Full".

        Args:
            y (numpy.ndarray): Flux array to normalize.

        Returns:
            y (numpy.ndarray): Normalized (or unchanged) flux array.
        """
        divisor = self._normalization_divisor("full", y)
        return y if divisor == 1.0 else np.asarray(y, dtype=float) / divisor

    def _maybe_normalize_side(self, side, y):
        """Normalize Left/Right's data, honoring that side's Self-Normalize
        override if set; else falls back to _maybe_normalize (relative to
        Full's peak, or unchanged if Normalize is off).

        Args:
            side (str): 'left' or 'right'.
            y (numpy.ndarray): Flux array to normalize.

        Returns:
            y (numpy.ndarray): Normalized (or unchanged) flux array.
        """
        divisor = self._normalization_divisor(side, y)
        return y if divisor == 1.0 else np.asarray(y, dtype=float) / divisor

    def _overlay_converter(self):
        """Build a function that converts raw-flux curves to what the active tab's plot
        shows. The steps are the plot's own: subtract the continuum if subtraction is on,
        then normalize. Build it once for each plot, because it can read the whole spectrum.

        Args:
            None

        Returns:
            convert (callable): Function (x, y) that returns y in the units of the plot.
        """
        tab = self.active_spectrum_tab
        grid = self._get_tab_wavelength(tab)
        continuum = self.continuum_fitted.get(tab) if self.continuum_subtracted.get(tab, False) else None
        if continuum is not None and len(continuum) != len(grid):
            continuum = None
        # Self-Normalize divides by the peak of the whole trace, so only it needs the trace.
        self_normalize = {"left": self.self_normalize_left, "right": self.self_normalize_right}.get(tab, False)
        trace = self._apply_continuum(tab, self._get_tab_raw_flux(tab)) if self_normalize else None
        divisor = self._normalization_divisor(tab, trace)

        def convert(x, y):
            """Convert one raw-flux curve to the units of the plotted spectrum.

            Args:
                x (numpy.ndarray): Wavelengths of the curve, on the tab's wavelength axis.
                y (numpy.ndarray): Raw flux values of the curve.

            Returns:
                y (numpy.ndarray): The curve in the units of the plotted spectrum.
            """
            y = np.asarray(y, dtype=float)
            if continuum is not None:
                y = y - np.interp(x, grid, continuum)
            return y / divisor
        return convert

    def _mark_slice_indicator_set(self, target):
        """Record that the user explicitly picked a wavelength for this side
        and refresh visibility so it now shows up on the appropriate tab(s).

        Args:
            target (str): Panel whose indicator to mark, 'left' or 'right'.

        Returns:
            None
        """
        self._slice_indicator_set[target] = True
        self._refresh_active_spectrum_tab()

    def _reset_slice_indicators(self):
        """Reset indicator visibility state - call whenever a cube (re)loads.

        Args:
            None

        Returns:
            None
        """
        self._slice_indicator_set = {"left": False, "right": False}

    def _on_spectrum_tab_changed(self, index):
        """Handle the user switching the active spectrum tab.

        Args:
            index (int): New tab index (0=Full, 1=Left, 2=Right).

        Returns:
            None
        """
        if 0 <= index < len(self._spectrum_tab_keys):
            new_tab = self._spectrum_tab_keys[index]
            old_tab = getattr(self, 'active_spectrum_tab', 'full')
            # Swap Gaussian boundary state between tabs; guarded by hasattr
            # since GuiGaussianMixin owns this, not GuiSpectrumMixin.
            if new_tab != old_tab and hasattr(self, '_swap_gaussian_boundaries_for_tab'):
                self._swap_gaussian_boundaries_for_tab(old_tab, new_tab)
            self.active_spectrum_tab = new_tab
            self._refresh_active_spectrum_tab()

    def _get_selection_spectrum(self, sel):
        """Compute the flux array for one stored selection dict (point or
        region). Region spectra are scaled by pixel count to stay on a
        comparable absolute scale to the summed original spectrum.

        Args:
            sel (dict): Stored selection dict, or None/empty.

        Returns:
            spectrum (numpy.ndarray): Flux array for the selection, or None.
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

    def _get_tab_raw_flux(self, tab):
        """Return `tab`'s own raw flux array (before continuum subtraction or
        normalization). Full is always original_spectrum; Left/Right are
        derived fresh from their own stored selection every call.

        Args:
            tab (str): Tab to read, 'full', 'left', or 'right'.

        Returns:
            flux (numpy.ndarray): Raw flux array, or None if that tab has
                nothing to show (Left/Right with no selection).
        """
        if tab == 'full':
            return getattr(self, 'original_spectrum', None)
        sel = self.last_selection_left if tab == 'left' else self.last_selection_right
        return self._get_selection_spectrum(sel)

    def _apply_continuum(self, tab, y):
        """Subtract `tab`'s own fitted continuum from y if subtraction is
        active for that tab, else return y unchanged. Keeps Full/Left/Right's
        continuum subtractions independent.

        Args:
            tab (str): Tab to read continuum state for.
            y (numpy.ndarray): Flux array to subtract the continuum from.

        Returns:
            y (numpy.ndarray): Continuum-subtracted (or unchanged) flux array.
        """
        if y is None:
            return y
        if not self.continuum_subtracted.get(tab, False):
            return y
        continuum = self.continuum_fitted.get(tab)
        if continuum is None or len(continuum) != len(y):
            return y
        return np.asarray(y, dtype=float) - np.asarray(continuum, dtype=float)

    def _get_active_tab_spectrum(self):
        """Return the raw flux array for the active tab. Callers (Gaussian
        fitting) intentionally get pre-continuum-subtraction flux; that
        subtraction only affects what's drawn on the plot.

        Args:
            None

        Returns:
            flux (numpy.ndarray): Raw flux array, or None if the active tab
                has nothing to show (Left/Right with no selection).
        """
        return self._get_tab_raw_flux(getattr(self, 'active_spectrum_tab', 'full'))

    def _get_active_tab_line(self):
        """Return the Line2D representing the active tab's own trace
        (global_line for Full, the dedicated Left/Right line otherwise),
        for code that needs to inspect what's actually plotted.

        Args:
            None

        Returns:
            line (matplotlib.lines.Line2D): The active tab's trace, or None.
        """
        tab = getattr(self, 'active_spectrum_tab', 'full')
        if tab == 'full':
            return getattr(self.spec_plotter, 'global_line', None)
        return self._spec_tab_line_left if tab == 'left' else self._spec_tab_line_right

    def _get_tab_wavelength(self, tab):
        """Return this tab's own (possibly independently-shifted) wavelength
        array: base_wavelength * spectrum_shift_factor[tab]. Full's factor
        stays mirrored into cube_data.wavelength (see gui.py) for unmigrated features.

        Args:
            tab (str): Tab to read, 'full', 'left', or 'right'.

        Returns:
            wavelength (numpy.ndarray): This tab's wavelength array, or None
                if no cube is loaded.
        """
        if self.base_wavelength is None:
            return None
        factor = self.spectrum_shift_factor.get(tab, 1.0)
        return np.asarray(self.base_wavelength, dtype=float) * factor

    def _ensure_tab_lines(self):
        """Lazily (re)create the dedicated Left/Right trace lines, if missing
        or if the spectrum plotter's axes was replaced (e.g. a new cube was
        loaded, which builds a brand-new SpectrumPlotter/axes).

        Args:
            None

        Returns:
            None
        """
        # Darker shades of the slice-indicator colors (cyan/magenta) keep the
        # region trace and its indicator line related but distinguishable.
        ax = self.spec_plotter.ax
        if self._spec_tab_line_left is None or self._spec_tab_line_left.axes is not ax:
            self._spec_tab_line_left, = ax.step([], [], where='mid', color='darkcyan', label='Left', zorder=2)
            self._spec_tab_line_left.set_visible(False)
        if self._spec_tab_line_right is None or self._spec_tab_line_right.axes is not ax:
            self._spec_tab_line_right, = ax.step([], [], where='mid', color='darkmagenta', label='Right', zorder=2)
            self._spec_tab_line_right.set_visible(False)

    def _ensure_shift_preview_line(self):
        """Lazily (re)create the single shared shift-preview line (orange).
        Reused across tabs since only the active tab's preview shows at once;
        a pure preview cue that never touches the real traces' colors/labels.

        Args:
            None

        Returns:
            None
        """
        ax = self.spec_plotter.ax
        if self._shift_preview_line is None or self._shift_preview_line.axes is not ax:
            self._shift_preview_line, = ax.plot([], [], color='orange', linewidth=1.5, alpha=0.8, zorder=4)
            self._shift_preview_line.set_visible(False)

    def _ensure_continuum_artists(self):
        """Lazily (re)create the shared continuum-fit and zero-level lines.
        One of each is enough since only the active tab's continuum shows
        at once - same reuse pattern as _shift_preview_line.

        Args:
            None

        Returns:
            None
        """
        ax = self.spec_plotter.ax
        if self._continuum_line_artist is None or self._continuum_line_artist.axes is not ax:
            self._continuum_line_artist, = ax.plot(
                [], [], color='orange', linewidth=2, linestyle='-', label='Continuum', zorder=10
            )
            self._continuum_line_artist.set_visible(False)
        if self._continuum_zero_line is None or self._continuum_zero_line.axes is not ax:
            self._continuum_zero_line = ax.axhline(
                0, color='lightgrey', linestyle=':', linewidth=1.5, alpha=0.8, label='Zero Level', zorder=3
            )
            self._continuum_zero_line.set_visible(False)

    def _refresh_active_spectrum_tab(self):
        """Show the active tab's own trace, any Full-tab overplots, and the
        active tab's own pending shift preview. Each line's x-data comes
        from _get_tab_wavelength(that tab), since each tab shifts independently.

        Args:
            None

        Returns:
            None
        """
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            return
        if not hasattr(self, 'active_spectrum_tab'):
            return

        global_line = getattr(self.spec_plotter, 'global_line', None)
        if global_line is None:
            return

        self._ensure_tab_lines()
        self._ensure_shift_preview_line()
        self._ensure_continuum_artists()
        tab = self.active_spectrum_tab

        # Fallback x-array for the rare case base_wavelength isn't captured
        # yet - keeps this method from breaking outright.
        fallback_x = global_line.get_xdata()

        def tab_x(t):
            """Return a tab's wavelength array, or the fallback if unset.

            Args:
                t (str): Tab to read, 'full', 'left', or 'right'.

            Returns:
                x (numpy.ndarray): The tab's wavelength array, or fallback_x.
            """
            x = self._get_tab_wavelength(t)
            return x if x is not None else fallback_x

        x_full = tab_x("full")

        # Full is always just the original spectrum - never repurposed. Its
        # continuum subtraction (if active) applies before normalization.
        if tab == "full":
            y = self._apply_continuum("full", getattr(self, 'original_spectrum', None))
            y = self._maybe_normalize(y)
            if y is not None and len(y) == len(x_full):
                global_line.set_data(x_full, np.asarray(y, dtype=np.float32))
        full_shifted = self.spectrum_shift_factor.get("full", 1.0) != 1.0
        global_line.set_label("Shifted Spectrum" if full_shifted else "Total Spectrum")
        global_line.set_visible(tab == "full")

        overplot_flags = {"left": self.overplot_left, "right": self.overplot_right}
        overplot_checkboxes = {"left": self.overplot_left_checkbox, "right": self.overplot_right_checkbox}
        self_normalize_checkboxes = {"left": self.self_normalize_left_checkbox, "right": self.self_normalize_right_checkbox}
        for side, line in (("left", self._spec_tab_line_left), ("right", self._spec_tab_line_right)):
            sel = self.last_selection_left if side == "left" else self.last_selection_right
            overplot_checkboxes[side].setEnabled(tab == "full" and bool(sel))
            # Self-Normalize is a standalone toggle - only needs a selection
            # to act on, independent of whether Normalize (global) is on.
            self_normalize_checkboxes[side].setEnabled(bool(sel))
            # Each tab's OWN shift, independent of the others.
            side_shifted = self.spectrum_shift_factor.get(side, 1.0) != 1.0
            line.set_label(f"{side.capitalize()}{' (Shifted)' if side_shifted else ''}")

            show = (tab == side and sel) or (tab == "full" and overplot_flags[side] and sel)
            if show:
                x_side = tab_x(side)
                y = self._apply_continuum(side, self._get_selection_spectrum(sel))
                y = self._maybe_normalize_side(side, y)
                if y is not None and len(y) == len(x_side):
                    line.set_data(x_side, np.asarray(y).astype(np.float32))
                    line.set_visible(True)
                    continue
            line.set_visible(False)

        # Continuum-fit preview: only the ACTIVE tab's own fitted continuum ever shows. The
        # automatic one shows while its "Fit Continuum" toggle is on; a custom one, once fitted.
        continuum_fit = self.continuum_fitted.get(tab)
        line_btn = getattr(self, 'continuum_line_btn', None)
        # Once subtracted, the continuum is the zero line, so its own curve is hidden.
        continuum_shown = (((line_btn is not None and line_btn.isChecked())
                            or self.continuum_mode_applied.get(tab) == 'custom')
                           and not self.continuum_subtracted.get(tab, False))
        if continuum_shown and continuum_fit is not None:
            x_t = tab_x(tab)
            if len(continuum_fit) == len(x_t):
                self._continuum_line_artist.set_data(x_t, np.asarray(continuum_fit, dtype=float))
                self._continuum_line_artist.set_visible(True)
            else:
                self._continuum_line_artist.set_visible(False)
        else:
            self._continuum_line_artist.set_visible(False)

        # Zero-level reference line: shown when the active tab has a
        # continuum subtracted, since subtracted values sit near zero.
        self._continuum_zero_line.set_visible(bool(self.continuum_subtracted.get(tab, False)))

        # Shift preview: only the ACTIVE tab's own pending preview ever
        # shows, overplotted (in orange) against that tab's own trace/color.
        preview = self.spectrum_shift_preview.get(tab)
        if preview and self.base_wavelength is not None:
            preview_x = np.asarray(self.base_wavelength, dtype=float) * preview["factor"]
            if tab == "full":
                preview_y = self._apply_continuum("full", getattr(self, 'original_spectrum', None))
                preview_y = self._maybe_normalize(preview_y)
            else:
                sel = self.last_selection_left if tab == "left" else self.last_selection_right
                preview_y = self._apply_continuum(tab, self._get_selection_spectrum(sel))
                preview_y = self._maybe_normalize_side(tab, preview_y)
            if preview_y is not None and len(preview_y) == len(preview_x):
                self._shift_preview_line.set_data(preview_x, np.asarray(preview_y, dtype=np.float32))
                self._shift_preview_line.set_label(f"Preview (v={preview['velocity']:.1f} km/s)")
                self._shift_preview_line.set_visible(True)
            else:
                self._shift_preview_line.set_visible(False)
        else:
            self._shift_preview_line.set_visible(False)

        self._sync_shift_buttons()

        indicator_set = getattr(self, '_slice_indicator_set', {"left": False, "right": False})
        vline_left = getattr(self.spec_plotter, 'vline_left', None)
        vline_right = getattr(self.spec_plotter, 'vline_right', None)
        if vline_left is not None:
            vline_left.set_visible(indicator_set.get("left", False) and tab in ("full", "left"))
        if vline_right is not None:
            vline_right.set_visible(indicator_set.get("right", False) and tab in ("full", "right"))

        self._apply_shared_y_scale()

        # Gaussian fits belong to their tab: draw the active tab's, remove the others.
        if hasattr(self, '_redraw_fit_overlays'):
            self._redraw_fit_overlays()
        self._rebuild_spectrum_legend()

        try:
            self.spec_canvas.draw()
        except Exception:
            pass

    def _rebuild_spectrum_legend(self):
        """Draw the legend from the artists that are visible now, without the hidden groups.
        A hidden trace or marker leaves no stale entry behind.

        Args:
            None

        Returns:
            None
        """
        plotter = self.spec_plotter
        if plotter is None or getattr(plotter, 'global_line', None) is None:
            return
        # The spectrum is first in the legend. Some of the other artists are made later,
        # so a missing one is skipped.
        candidates = [
            plotter.global_line, getattr(self, '_spec_tab_line_left', None),
            getattr(self, '_spec_tab_line_right', None), plotter.vline_left, plotter.vline_right]
        candidates += [getattr(self, name, None) for name in (
            '_shift_preview_line', '_continuum_line_artist', '_continuum_zero_line',
            'gauss_left_marker', 'gauss_peak_marker', 'gauss_right_marker',
            'gauss_noise_left_marker', 'gauss_noise_right_marker')]
        # A removed artist has no axes, so a stale marker gets no entry.
        handles = [a for a in candidates if a is not None and a.axes is not None]
        if hasattr(self, '_fit_overlay_handles'):
            handles += self._fit_overlay_handles()
        # The box with Center, FWHM and R² follows the "Gaussian fits" group.
        show_box = "Gaussian fits" not in HIDDEN_LEGEND_GROUPS
        for artist in getattr(plotter, 'gauss_fit_artists', []):
            if isinstance(artist, Text):
                artist.set_visible(show_box)
        try:
            draw_spectrum_legend(plotter.ax, handles)
        except Exception:
            pass

    def _init_legend_button(self):
        """Create the Legend button. Its menu has one checkbox for each group of legend entries.

        Args:
            None

        Returns:
            None
        """
        self.legend_btn = QPushButton("Legend")
        self.legend_btn.setToolTip("Choose which entries the spectrum legend shows")
        menu = QMenu(self.legend_btn)
        for group in LEGEND_GROUPS:
            action = menu.addAction(group)
            action.setCheckable(True)
            action.setChecked(group not in HIDDEN_LEGEND_GROUPS)
            action.toggled.connect(lambda checked, name=group: self._on_legend_group_toggled(name, checked))
        self.legend_btn.setMenu(menu)

    def _on_legend_group_toggled(self, group, checked):
        """Show or hide one group of legend entries, then draw the legend again.

        Args:
            group (str): Name of the group, a key of LEGEND_GROUPS.
            checked (bool): True to show the group, False to hide it.

        Returns:
            None
        """
        if checked:
            HIDDEN_LEGEND_GROUPS.discard(group)
        else:
            HIDDEN_LEGEND_GROUPS.add(group)
        self._rebuild_spectrum_legend()
        self.spec_canvas.draw_idle()

    def _sync_shift_buttons(self):
        """Set the two shift buttons from the active tab's state. A preview or an applied
        shift on one tab has no bearing on the others.

        Args:
            None

        Returns:
            None
        """
        tab = self.active_spectrum_tab
        has_preview = bool(self.spectrum_shift_preview.get(tab))
        has_shift = self.spectrum_shift_factor.get(tab, 1.0) != 1.0
        if hasattr(self, 'shift_btn'):
            self.shift_btn.setChecked(has_preview)
        if hasattr(self, 'apply_shift_btn'):
            self.apply_shift_btn.setChecked(has_shift)
            self.apply_shift_btn.setEnabled(has_preview or has_shift)

    def _fit_xlim_to(self, x_array):
        """Set the spectrum plot's x-limits to span x_array with ~5% padding,
        matching matplotlib's own autoscale. Used after applying/reverting a
        shift, so the axis fits the new data instead of the old view window.

        Args:
            x_array (numpy.ndarray): Wavelength array to fit the x-axis to.

        Returns:
            None
        """
        if x_array is None or len(x_array) == 0:
            return
        data_min, data_max = float(np.min(x_array)), float(np.max(x_array))
        span = data_max - data_min
        pad = span * 0.05 if span > 0 else 1.0
        self.spec_plotter.ax.set_xlim(data_min - pad, data_max + pad)

    def _apply_shared_y_scale(self):
        """Set y-limits from the original spectrum's range on every tab, so a
        region's peak is always seen relative to the full spectrum's. Falls
        back to fitting the union of visible traces when that guarantee fails.

        Args:
            None

        Returns:
            None
        """
        self_normalized_visible = (
            (self.self_normalize_left
             and self._spec_tab_line_left is not None and self._spec_tab_line_left.get_visible())
            or (self.self_normalize_right
                and self._spec_tab_line_right is not None and self._spec_tab_line_right.get_visible())
        )
        continuum_active_on_shown = self.continuum_subtracted.get(
            getattr(self, 'active_spectrum_tab', 'full'), False
        )

        # Self-normalized or continuum-subtracted traces break the "region
        # <= full" guarantee, so fit to what's actually visible instead.
        if self_normalized_visible or continuum_active_on_shown:
            lines = (self.spec_plotter.global_line, self._spec_tab_line_left, self._spec_tab_line_right,
                     self._shift_preview_line)
            visible_ys = [
                np.asarray(line.get_ydata(), dtype=float)
                for line in lines
                if line is not None and line.get_visible() and len(line.get_ydata())
            ]
            if not visible_ys:
                return
            y = np.concatenate(visible_ys)
        else:
            original = getattr(self, 'original_spectrum', None)
            if original is None:
                return
            y = self._maybe_normalize(original)
            y = np.asarray(y, dtype=float)

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
