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
    shape_buttons["left"|"right"][shape name]      (dict of QPushButton)
    edit_shape_btn_l / edit_shape_btn_r            (checkable QPushButton)
    stats1_label / stats2_label                    (QLabel)
    region_state                                   ({"left":..,"right":..})
    active_shape / active_target
    last_selection_left / last_selection_right

and override the hooks below where their behaviour differs (``_region_image``
is the only one with no usable default).
"""

import numpy as np
from PyQt5.QtWidgets import QMessageBox

from .scale_dialog import scaling_stats_text
from .visualization import format_sky_position


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

    def _region_wcs(self, target):
        """Sky WCS of panel ``target`` ('left'/'right'), or None.

        Args:
            target (str): Panel to read from, 'left' or 'right'.

        Returns:
            wcs (astropy.wcs.WCS): 2D WCS of the panel; None by default.
        """
        return None

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
            text (str): Placeholder label text. By default, the whole-image stats.
        """
        return self._whole_image_stats_text(target)

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

    # ------------------------------------------------------------- shared utility
    def _ask_target_panel(self, title, question):
        """Ask which panel, left or right, an action should use. Used by
        the 2D window's contour and superimpose prompts.

        Args:
            title (str): Dialog window title.
            question (str): Prompt text shown to the user.

        Returns:
            target (str): 'left' or 'right', or None if cancelled.
        """
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(question)
        btn_left = msg.addButton("Left", QMessageBox.ActionRole)
        btn_right = msg.addButton("Right", QMessageBox.ActionRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.exec_()
        if msg.clickedButton() == btn_left:
            return "left"
        if msg.clickedButton() == btn_right:
            return "right"
        return None

    # ------------------------------------------------- panel readouts and masks
    def _whole_image_stats_text(self, target, image=None):
        """Whole-image stats for a panel: the data's own min/max/mean/sum/std,
        then the color range the plot uses. The unit is on its own line so the
        narrow stats column doesn't clip it.

        Args:
            target (str): Panel to describe, 'left' or 'right'.
            image (numpy.ndarray): Image to describe. None reads the panel's current image.

        Returns:
            text (str): Stats label text, or the blank placeholder if no image is loaded.
        """
        if image is None:
            image = self._region_image(target)
        if image is None:
            return "Intensity:\n-"
        scale_settings = self.slice1_scale if target == 'left' else self.slice2_scale
        mn, mx, mean = np.nanmin(image), np.nanmax(image), np.nanmean(image)
        total_sum, std = np.nansum(image), np.nanstd(image)
        unit = self._region_flux_unit(target)
        unit_line = f"[{unit}]:\n" if unit else ""
        return (
            f"Intensity (data)\n{unit_line}"
            f"min={mn:.3e}\n"
            f"max={mx:.3e}\n"
            f"mean={mean:.3e}\n"
            f"sum={total_sum:.3e}\n"
            f"std={std:.3e}\n"
            f"\n"
            + scaling_stats_text(image, scale_settings)
        )

    def _on_slice_hover(self, event, target):
        """Live cursor readout (pixel, RA, Dec and value) as the mouse moves over a panel.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Motion event.
            target (str): Panel hovered over, 'left' or 'right'.

        Returns:
            None
        """
        hover_label = self.hover1_label if target == "left" else self.hover2_label
        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
        if event.inaxes != plotter.ax or event.xdata is None or event.ydata is None:
            hover_label.setText("Cursor Info:\n-")
            return
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        image = self._region_image(target)
        if image is None or not (0 <= iy < image.shape[0] and 0 <= ix < image.shape[1]):
            hover_label.setText("Cursor Info:\n-")
            return
        unit = self._region_flux_unit(target)
        unit_str = f" [{unit}]" if unit else ""
        sky = format_sky_position(self._region_wcs(target), event.xdata, event.ydata)
        hover_label.setText(
            f"Cursor Info:\nPixel: {ix}, {iy}\n{sky}Val: {float(image[iy, ix]):.4e}{unit_str}")

    def _selection_to_full_mask(self, target, sel):
        """Build a full-image mask from a selection dict.

        Args:
            target (str): Panel the selection belongs to, 'left' or 'right'.
            sel (dict): Selection dict from last_selection_left/right.

        Returns:
            mask (numpy.ndarray): Boolean mask, same shape as the panel's image.
        """
        if sel.get("shape") == "POINT":
            # Point selections store coords, not a mask -- build a single-pixel one.
            x, y = sel["coords"]
            x, y = int(round(x)), int(round(y))
            image = self._region_image(target)
            mask = np.zeros(image.shape, dtype=bool)
            mask[y, x] = True
            return mask
        return sel["mask"]

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
            edit_btn.setEnabled(bool(sel) and sel.get("shape") != "POINT")
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
            shape (str): Shape name that was clicked.
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
            combined_stats = region_stats + [""] + slice_stats if slice_stats else region_stats
            stats_label.setText("\n".join(combined_stats))

        canvas.draw()
        self._region_on_change(target, "select")

    # ---------------------------------------------------------- copy / recreate
    def copy_selection(self, source, target):
        """Copy a region selection onto the other panel, WCS-aware: corners
        reproject through both panels' WCS (rectangles become polygons
        under rotation), else falls back to a plain pixel-index copy.

        Args:
            source (str): Panel to copy from, 'left' or 'right'.
            target (str): Panel to copy to, 'left' or 'right'.

        Returns:
            None
        """
        src_sel = self.last_selection_right if source == "right" else self.last_selection_left
        if not src_sel:
            return

        src_wcs = self._region_wcs(source)
        tgt_wcs = self._region_wcs(target)
        tgt_img = self._region_image(target)
        selector = self.region_selector_left if target == "left" else self.region_selector_right
        shape = src_sel["shape"]

        wcs_ok = (
            tgt_img is not None
            and src_wcs is not None and getattr(src_wcs, "has_celestial", False)
            and tgt_wcs is not None and getattr(tgt_wcs, "has_celestial", False)
            # One WCS object means one pixel grid, so a pixel copy is exact.
            and src_wcs is not tgt_wcs
        )

        def _to_target_px(xs, ys):
            """Reproject source-panel pixel coordinates onto the target panel.

            Args:
                xs (numpy.ndarray): Source-panel x pixel coordinates.
                ys (numpy.ndarray): Source-panel y pixel coordinates.

            Returns:
                coords (tuple): (tx, ty) target-panel pixel coordinates.
            """
            sw = src_wcs.celestial if src_wcs.pixel_n_dim > 2 else src_wcs
            tw = tgt_wcs.celestial if tgt_wcs.pixel_n_dim > 2 else tgt_wcs
            sky = sw.pixel_to_world(np.asarray(xs, float), np.asarray(ys, float))
            tx, ty = tw.world_to_pixel(sky)
            return np.asarray(tx, float), np.asarray(ty, float)

        # ---- pixel ("POINT") ----
        if shape == "POINT":
            x, y = src_sel["coords"]
            if wcs_ok:
                try:
                    tx, ty = _to_target_px([x], [y])
                    if np.isfinite(tx[0]) and np.isfinite(ty[0]):
                        x, y = float(tx[0]), float(ty[0])
                except Exception:
                    pass
            self._handle_pixel_click(int(round(x)), int(round(y)), target)
            return

        # ---- lasso (freehand shape) ----
        if shape == "POLYGON":
            patch = src_sel.get("patch")
            if patch is None or not hasattr(patch, "lasso_verts"):
                return
            verts = np.asarray(patch.lasso_verts, dtype=float)
            if wcs_ok:
                try:
                    tx, ty = _to_target_px(verts[:, 0], verts[:, 1])
                    if np.all(np.isfinite(tx)) and np.all(np.isfinite(ty)):
                        ny, nx = tgt_img.shape
                        verts = np.column_stack(
                            [np.clip(tx, 0, nx - 1), np.clip(ty, 0, ny - 1)])
                except Exception:
                    pass
            selector.active_shape = "POLYGON"
            try:
                selector.on_lasso_select(verts)
            except Exception:
                pass
            return

        # ---- rectangle / circle / ellipse ----
        xmin, xmax, ymin, ymax = (float(c) for c in src_sel["coords"])
        if wcs_ok:
            try:
                cx = [xmin, xmax, xmax, xmin]
                cy = [ymin, ymin, ymax, ymax]
                tx, ty = _to_target_px(cx, cy)
                if not (np.all(np.isfinite(tx)) and np.all(np.isfinite(ty))):
                    raise ValueError("non-finite reprojected corners")
                ny, nx = tgt_img.shape
                axis_aligned = (
                    abs(tx[0] - tx[3]) < 0.5 and abs(tx[1] - tx[2]) < 0.5 and
                    abs(ty[0] - ty[1]) < 0.5 and abs(ty[2] - ty[3]) < 0.5
                )
                if shape == "RECTANGLE" and not axis_aligned:
                    verts = np.column_stack(
                        [np.clip(tx, 0, nx - 1), np.clip(ty, 0, ny - 1)])
                    selector.active_shape = "POLYGON"
                    try:
                        selector.on_lasso_select(verts)
                    except Exception:
                        pass
                    return
                # round (not truncate) so a copy between identical/near-identical
                # grids is exact instead of drifting a pixel on the TAN round-trip
                xmin = max(0, min(nx - 1, int(round(float(np.min(tx))))))
                xmax = max(0, min(nx - 1, int(round(float(np.max(tx))))))
                ymin = max(0, min(ny - 1, int(round(float(np.min(ty))))))
                ymax = max(0, min(ny - 1, int(round(float(np.max(ty))))))
                if xmax - xmin < 1 or ymax - ymin < 1:
                    QMessageBox.information(
                        self, "Copy Region",
                        "The source selection does not overlap the other "
                        "panel's field of view.")
                    return
            except Exception:
                xmin, xmax, ymin, ymax = (float(c) for c in src_sel["coords"])

        selector.active_shape = shape
        try:
            selector.create_region((int(round(xmin)), int(round(xmax)),
                                    int(round(ymin)), int(round(ymax))))
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
        if shape == "POLYGON":
            verts = getattr(sel.get("patch"), "lasso_verts", None)
            if verts is not None:
                selector.on_lasso_select(verts)
            else:
                selector.active_shape = "RECTANGLE"
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
        """Handle a single-pixel ('POINT') selection.

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
            sel = {"shape": "POINT", "coords": (x, y), "mask": None, "label": label}
            if target == "left":
                self.last_selection_left = sel
            else:
                self.last_selection_right = sel

            current_text = stats_label.text().split("\n")
            slice_stats = [line for line in current_text if self._region_slice_stat_keep(line)]

            unit = self._region_flux_unit(target)
            # Unit on its own line so the narrow stats column doesn't clip it.
            pixel_stats = [f"Pixel: (x={x}, y={y})", f"Intensity={intensity:.3e}"]
            if unit:
                pixel_stats.append(f"[{unit}]")
            ra, dec = self._region_pixel_coords(target, x, y)
            if ra is not None and dec is not None:
                pixel_stats += [f"RA={ra:.6f}°", f"DEC={dec:.6f}°"]
            pixel_stats += self._region_pixel_extra_lines(target, x, y)
            combined_stats = pixel_stats + [""] + slice_stats if slice_stats else pixel_stats
            stats_label.setText("\n".join(combined_stats))

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
