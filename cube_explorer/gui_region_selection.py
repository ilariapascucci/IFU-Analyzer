"""
RegionSelectionMixin - the shape-selection / region-stats / edit-shape
controller layer, shared by the 3D cube GUI (``CubeExplorer``) and the 2D
dual viewer (``Ifu_analyzer_2d_window``).

The low-level drawing engine (``RegionSelector`` / ``InteractivePatch`` in
``region_selector.py``) was always shared; this mixin removes the second copy
of the *wiring* around it. Host classes must provide the widgets/attributes
the 3D and 2D windows already have in common:

    region_selector_left / region_selector_right   (RegionSelector)
    slice1_plotter / slice2_plotter                (SlicePlotter, has .ax)
    slice1_canvas  / slice2_canvas                 (FigureCanvas)
    shape_buttons["left"|"right"][symbol]          (dict of QPushButton)
    edit_shape_btn_l / edit_shape_btn_r            (checkable QPushButton)
    stats1_label / stats2_label                    (QLabel)
    region_state                                   ({"left":..,"right":..})
    active_shape / active_target
    last_selection_left / last_selection_right

and override the hooks below where their behaviour differs (``_region_image``
is the only one with no usable default).
"""

import numpy as np


class RegionSelectionMixin:

    # ------------------------------------------------------------------ hooks
    def _region_image(self, target):
        """2D array shown in panel ``target`` ('left'/'right'), or None.

        Args:
            target (str): Panel to read from, 'left' or 'right'.

        Returns:
            image (numpy.ndarray): The displayed image. Hosts must override
                this; the base implementation always raises.
        """
        raise NotImplementedError

    def _region_flux_unit(self, target):
        """Cleaned flux-unit string for panel ``target`` ('' when unknown).

        Args:
            target (str): Panel to read from, 'left' or 'right'.

        Returns:
            unit (str): Flux-unit label; '' by default.
        """
        return ""

    def _region_pixel_coords(self, target, x, y):
        """(ra_deg, dec_deg) for pixel (x, y) in panel ``target``.

        Args:
            target (str): Panel the pixel belongs to, 'left' or 'right'.
            x (float): Pixel x coordinate.
            y (float): Pixel y coordinate.

        Returns:
            coords (tuple): (ra_deg, dec_deg); (None, None) by default.
        """
        return None, None

    def _region_pixel_extra_lines(self, target, x, y):
        """Extra stat lines for a pixel click (3D adds wavelength/velocity;
        2D adds nothing).

        Args:
            target (str): Panel the pixel belongs to, 'left' or 'right'.
            x (float): Pixel x coordinate.
            y (float): Pixel y coordinate.

        Returns:
            lines (list): Extra text lines to show; [] by default.
        """
        return []

    def _region_stats_placeholder(self, target):
        """Text a stats label reverts to when its selection is cleared.

        Args:
            target (str): Panel to build the placeholder for, 'left' or 'right'.

        Returns:
            text (str): Placeholder label text.
        """
        return "Intensity:\n-"

    def _region_slice_stat_keep(self, line):
        """True for existing stats-label lines to preserve above region stats.
        Default keeps nothing - a region selection replaces the panel stats.

        Args:
            line (str): One line of the current stats label text.

        Returns:
            keep (bool): Whether to preserve this line; False by default.
        """
        return False

    def _region_on_change(self, target, kind):
        """Post-hook after a selection changes. 3D refreshes the active
        spectrum tab; the 2D viewer redraws the panel.

        Args:
            target (str): Panel the change happened on, 'left' or 'right'.
            kind (str): Kind of change: 'select', 'pixel', or 'clear'.

        Returns:
            None
        """
        pass

    # -------------------------------------------------------------- state machine
    def _set_region_state(self, side, new_state):
        """Single source of truth for shape-button enablement, the Edit Shape
        button state, and InteractivePatch handler wiring, so they never
        drift out of sync.

        Args:
            side (str): Panel to update, 'left' or 'right'.
            new_state (str): New state: 'idle', 'drawing', 'placed', or 'editing'.

        Returns:
            None
        """
        selector = self.region_selector_left if side == "left" else self.region_selector_right
        edit_btn = self.edit_shape_btn_l if side == "left" else self.edit_shape_btn_r
        old_state = self.region_state.get(side, "idle")

        if old_state == "editing" and new_state != "editing" and selector.interactive_patch is not None:
            selector.interactive_patch.disconnect_handlers()

        self.region_state[side] = new_state

        if new_state == "idle":
            for btn in self.shape_buttons[side].values():
                btn.setEnabled(True)
                btn.setChecked(False)
            edit_btn.setEnabled(False)
            edit_btn.setChecked(False)
        elif new_state == "drawing":
            active_shape = self.active_shape if self.active_target == side else None
            for shape, btn in self.shape_buttons[side].items():
                btn.setEnabled(shape == active_shape)
            edit_btn.setEnabled(False)
            edit_btn.setChecked(False)
        elif new_state == "placed":
            for btn in self.shape_buttons[side].values():
                btn.setEnabled(False)
                btn.setChecked(False)
            sel = self.last_selection_left if side == "left" else self.last_selection_right
            edit_btn.setEnabled(bool(sel) and sel.get("shape") != "*")
            edit_btn.setChecked(False)
        elif new_state == "editing":
            for btn in self.shape_buttons[side].values():
                btn.setEnabled(False)
            edit_btn.setEnabled(True)
            edit_btn.setChecked(True)
            if selector.interactive_patch is not None:
                selector.interactive_patch.connect()

    def _toggle_edit_shape(self, side):
        """Handle the Edit Shape button click for one side.

        Args:
            side (str): Panel whose Edit Shape button was clicked, 'left'
                or 'right'.

        Returns:
            None
        """
        state = self.region_state.get(side, "idle")
        if state == "placed":
            self._set_region_state(side, "editing")
        elif state == "editing":
            self._set_region_state(side, "placed")
        else:
            edit_btn = self.edit_shape_btn_l if side == "left" else self.edit_shape_btn_r
            edit_btn.setChecked(False)

    def _set_shape(self, shape, target):
        """Handle shape button activation.

        Args:
            shape (str): Shape symbol that was clicked.
            target (str): Panel the shape applies to, 'left' or 'right'.

        Returns:
            None
        """
        if self.active_shape == shape and self.active_target == target:
            self.active_shape = None
            self.active_target = None
        else:
            self.active_shape = shape
            self.active_target = target

        for side in ["left", "right"]:
            for btn_shape, btn in self.shape_buttons[side].items():
                btn.setChecked(
                    side == target and btn_shape == shape and self.active_shape is not None
                )

        selector = self.region_selector_left if target == "left" else self.region_selector_right
        if self.active_shape:
            selector.activate_selector(shape)
            self._set_region_state(target, "drawing")
        else:
            selector.activate_selector(None)
            self._set_region_state(target, "idle")

    # ---------------------------------------------------------- region completed
    def _handle_region_select(self, mask, patch, coords, target):
        """Handle region selection completion (also fires on every live edit tick).

        Args:
            mask (numpy.ndarray): Boolean mask of the selected pixels.
            patch (matplotlib.patches.Patch): Patch outlining the selection.
            coords (tuple): (xmin, xmax, ymin, ymax) region bounds.
            target (str): Panel the selection is on, 'left' or 'right'.

        Returns:
            None
        """
        xmin, xmax, ymin, ymax = coords

        image = self._region_image(target)
        if image is None:
            return

        selector = self.region_selector_left if target == "left" else self.region_selector_right

        # This callback also fires on every mouse-move tick of a live drag/resize
        # while "editing"; don't tear down handles or touch region_state mid-drag.
        is_live_edit_tick = self.region_state.get(target) == "editing"

        if mask.shape == image.shape:
            full_mask = mask
        else:
            full_mask = np.zeros_like(image, dtype=bool)
            h_mask, w_mask = mask.shape
            y0, x0 = int(ymin), int(xmin)
            y1 = min(int(y0 + h_mask), image.shape[0])
            x1 = min(int(x0 + w_mask), image.shape[1])
            full_mask[y0:y1, x0:x1] = mask[:(y1 - y0), :(x1 - x0)]

        # During a live edit-tick, selector.active_shape is stale (reset to None
        # when the draw completed); reuse the existing selection's own shape instead.
        if is_live_edit_tick:
            existing_sel = self.last_selection_left if target == "left" else self.last_selection_right
            shape = existing_sel["shape"] if existing_sel else selector.active_shape
        else:
            shape = selector.active_shape

        selection = {
            "shape": shape,
            "coords": coords,
            "mask": full_mask,
            "label": f"{target.capitalize()} Selection",
            "patch": patch,
        }
        if target == "left":
            self.last_selection_left = selection
        else:
            self.last_selection_right = selection

        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter

        if hasattr(plotter.ax, "pixel_marker"):
            for artist in list(plotter.ax.pixel_marker):
                try:
                    artist.remove()
                except Exception:
                    pass
            plotter.ax.pixel_marker = []

        if hasattr(plotter.ax, "area_artists"):
            for artist in list(plotter.ax.area_artists):
                try:
                    artist.remove()
                except Exception:
                    pass
        plotter.ax.area_artists = []

        # Remove lingering handle scatter collections - but not during a live
        # edit-tick, where the only handle collection IS the one being dragged.
        if not is_live_edit_tick:
            for collection in list(plotter.ax.collections):
                try:
                    if getattr(collection, "_is_handle_collection", False):
                        collection.remove()
                except Exception:
                    pass

        plotter.ax.add_patch(patch)
        plotter.ax.area_artists.append(patch)

        # Draw now, while the mpl selector is still connected, so it recaptures
        # its blit background with this patch; otherwise a stale background shows.
        canvas = self.slice1_canvas if target == "left" else self.slice2_canvas
        canvas.draw()

        if selector.selector is not None:
            try:
                if hasattr(selector.selector, "set_active"):
                    selector.selector.set_active(False)
                if hasattr(selector.selector, "set_visible"):
                    selector.selector.set_visible(False)
            except Exception:
                pass

        if not is_live_edit_tick:
            self.active_shape = None
            self.active_target = None
            for side in ["left", "right"]:
                for btn in self.shape_buttons[side].values():
                    btn.setChecked(False)
            selector.activate_selector(None)
            self._set_region_state(target, "placed")

        selected_area = image[full_mask]
        if selected_area.size > 0:
            mn = np.nanmin(selected_area)
            mx = np.nanmax(selected_area)
            mean = np.nanmean(selected_area)
            total_sum = np.nansum(selected_area)
            std = np.nanstd(selected_area)

            stats_label = self.stats1_label if target == "left" else self.stats2_label
            current_text = stats_label.text().split("\n")
            slice_stats = [line for line in current_text if self._region_slice_stat_keep(line)]

            unit = self._region_flux_unit(target)
            # Unit on its own line so the narrow stats panel doesn't clip it.
            region_stats = ["Intensity (Area Selected)" if unit else "Intensity (Area Selected):"]
            if unit:
                region_stats.append(f"[{unit}]:")
            region_stats += [
                f"min={mn:.3e}",
                f"max={mx:.3e}",
                f"mean={mean:.3e}",
                f"sum={total_sum:.3e}",
                f"std={std:.3e}",
            ]
            combined_stats = slice_stats + [""] + region_stats if slice_stats else region_stats
            stats_label.setText("\n".join(combined_stats))

        canvas.draw()
        self._region_on_change(target, "select")

    # ---------------------------------------------------------- copy / recreate
    def copy_selection(self, source, target):
        """Copy the region selection from one panel to the other.

        Args:
            source (str): Panel to copy from, 'left' or 'right'.
            target (str): Panel to copy to, 'left' or 'right'.

        Returns:
            None
        """
        source_selection = (
            self.last_selection_right if source == "right" else self.last_selection_left
        )
        if not source_selection:
            return

        selector = self.region_selector_left if target == "left" else self.region_selector_right

        if source_selection["shape"] == "*":
            x, y = source_selection["coords"]
            self._handle_pixel_click(x, y, target)
            return

        if source_selection["shape"] == "✏":
            patch = source_selection.get("patch")
            if patch is not None and hasattr(patch, "lasso_verts"):
                selector.active_shape = "✏"
                selector.on_lasso_select(patch.lasso_verts)
                return
            # No stored vertices - fall through to a bounding-box rectangle.

        selector.active_shape = source_selection["shape"]
        try:
            coords = tuple(int(c) for c in source_selection["coords"])
            selector.create_region(coords)
        except Exception:
            pass

    def _recreate_region(self, xmin, xmax, ymin, ymax, mask, target):
        """Recreate a placed selection's visual patch (e.g. after a slice
        change), via the same RegionSelector pipeline used for an organic
        draw, so the shape type stays correct and lasso is handled.

        Args:
            xmin (int): Region's minimum x coordinate.
            xmax (int): Region's maximum x coordinate.
            ymin (int): Region's minimum y coordinate.
            ymax (int): Region's maximum y coordinate.
            mask (numpy.ndarray): Boolean mask of the selected pixels.
            target (str): Panel to recreate on, 'left' or 'right'.

        Returns:
            None
        """
        sel = self.last_selection_left if target == "left" else self.last_selection_right
        if not sel:
            return
        shape = sel.get("shape")
        selector = self.region_selector_left if target == "left" else self.region_selector_right

        if selector.interactive_patch is not None:
            selector.interactive_patch.disconnect()
            selector.interactive_patch = None

        selector.active_shape = shape
        if shape == "✏":
            verts = getattr(sel.get("patch"), "lasso_verts", None)
            if verts is not None:
                selector.on_lasso_select(verts)
            else:
                selector.active_shape = "▭"
                selector.create_region((xmin, xmax, ymin, ymax))
        else:
            selector.create_region((xmin, xmax, ymin, ymax))
        selector.active_shape = None

    # ---------------------------------------------------------------- clear
    def clear_region(self, target):
        """Clear the current region selection and its overlays for a panel.

        Args:
            target (str): Panel to clear, 'left' or 'right'.

        Returns:
            None
        """
        state = self.region_state.get(target, "idle")
        selector = self.region_selector_left if target == "left" else self.region_selector_right

        if state == "drawing":
            # Abort the in-progress draw - nothing has been placed yet.
            self.active_shape = None
            self.active_target = None
            selector.activate_selector(None)
            for btn in self.shape_buttons[target].values():
                btn.setChecked(False)
            self._set_region_state(target, "idle")
            return

        if target == "left":
            sel = self.last_selection_left
            plotter = self.slice1_plotter
            canvas = self.slice1_canvas
            stats_label = self.stats1_label
        else:
            sel = self.last_selection_right
            plotter = self.slice2_plotter
            canvas = self.slice2_canvas
            stats_label = self.stats2_label

        if selector.interactive_patch is not None:
            selector.interactive_patch.disconnect()
            selector.interactive_patch = None

        if not sel:
            self._set_region_state(target, "idle")
            return

        if hasattr(plotter.ax, "pixel_marker"):
            for marker in list(plotter.ax.pixel_marker):
                try:
                    marker.remove()
                except Exception:
                    pass
            plotter.ax.pixel_marker = []

        if hasattr(plotter.ax, "area_artists"):
            for artist in list(plotter.ax.area_artists):
                try:
                    artist.remove()
                except Exception:
                    pass
            plotter.ax.area_artists = []

        for collection in list(plotter.ax.collections):
            try:
                if getattr(collection, "_is_handle_collection", False):
                    collection.remove()
            except Exception:
                pass

        if target == "left":
            self.last_selection_left = None
        else:
            self.last_selection_right = None
        canvas.draw()
        stats_label.setText(self._region_stats_placeholder(target))

        self._set_region_state(target, "idle")
        self._region_on_change(target, "clear")

    # ---------------------------------------------------------------- pixel
    def _handle_pixel_click(self, x, y, target):
        """Handle a single-pixel ('*') selection.

        Args:
            x (int): Pixel x coordinate.
            y (int): Pixel y coordinate.
            target (str): Panel the click is on, 'left' or 'right'.

        Returns:
            None
        """
        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
        stats_label = self.stats1_label if target == "left" else self.stats2_label
        selector = self.region_selector_left if target == "left" else self.region_selector_right

        try:
            image = self._region_image(target)
            if image is None:
                return
            if not (0 <= y < image.shape[0] and 0 <= x < image.shape[1]):
                return

            intensity = float(image[y, x])
            label = f"{target.capitalize()} Selection"
            sel = {"shape": "*", "coords": (x, y), "mask": None, "label": label}
            if target == "left":
                self.last_selection_left = sel
            else:
                self.last_selection_right = sel

            unit = self._region_flux_unit(target)
            # Unit on its own line so the narrow stats column doesn't clip it.
            stats_text = [f"Pixel: (x={x}, y={y})", f"Intensity={intensity:.3e}"]
            if unit:
                stats_text.append(f"[{unit}]")
            ra, dec = self._region_pixel_coords(target, x, y)
            if ra is not None and dec is not None:
                stats_text += [f"RA={ra:.6f}°", f"DEC={dec:.6f}°"]
            stats_text += self._region_pixel_extra_lines(target, x, y)
            stats_label.setText("\n".join(stats_text))

            if selector.interactive_patch is not None:
                selector.interactive_patch.disconnect()
                selector.interactive_patch = None

            if hasattr(plotter.ax, "area_artists"):
                for artist in list(plotter.ax.area_artists):
                    try:
                        artist.remove()
                    except Exception:
                        pass
                plotter.ax.area_artists = []

            for collection in list(plotter.ax.collections):
                try:
                    if getattr(collection, "_is_handle_collection", False):
                        collection.remove()
                except Exception:
                    pass

            if hasattr(plotter.ax, "pixel_marker"):
                for marker in list(plotter.ax.pixel_marker):
                    try:
                        marker.remove()
                    except Exception:
                        pass
                plotter.ax.pixel_marker = []

            plotter.update_pixel_marker(x, y)

            canvas = self.slice1_canvas if target == "left" else self.slice2_canvas
            canvas.draw()

            self.active_shape = None
            self.active_target = None
            selector.activate_selector(None)
            self._set_region_state(target, "placed")
            self._region_on_change(target, "pixel")

        except Exception as e:
            print(f"Error handling pixel click: {e}")
