"""
Small Qt mixins for methods that two windows or two dialogs had copied.

DualPanelViewMixin serves the 3D and 2D dual-panel windows. MapResultsDialogMixin
serves the Flux Map and Gaussian Map results dialogs. ProgressWorker and
CancellableWorker serve the background threads of the map dialogs.
"""

from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import QPushButton, QMenu, QCheckBox, QFrame, QDialog, QMessageBox

from .colormap_dialog import ColorMapDialog
from .icon_loader import set_button_icon
from .scale_dialog import ScaleDialog


class ProgressWorker(QThread):
    """Background thread that reports progress through a `progress` signal.
    The subclass declares `progress = pyqtSignal(int, int)` and calls _emit_progress from its run().
    """

    def _emit_progress(self, current, total):
        """Relay a progress update through the `progress` signal.

        Args:
            current (int): Units done so far.
            total (int): Total units.

        Returns:
            None
        """
        self.progress.emit(current, total)


class CancellableWorker(ProgressWorker):
    """Progress worker whose run() checks the `cancelled` flag at the end of each row."""

    def request_cancel(self):
        """Ask the run to stop at the next row.

        Args:
            None

        Returns:
            None
        """
        self.cancelled = True


def make_panel_button(icon, tooltip, callback, checkable=False, styled=True):
    """Build one square icon button of a panel sidebar.

    Args:
        icon (str): Name of the icon file, without the extension.
        tooltip (str): Tooltip text.
        callback (callable): Click handler, or None for a button that does nothing. A handler that
            takes an argument gets the checked state of the click.
        checkable (bool): True if the button stays pressed when clicked.
        styled (bool): True to give the button the large-font style that the other square buttons have.

    Returns:
        btn (QPushButton): The configured button.
    """
    btn = QPushButton()
    set_button_icon(btn, icon)
    if checkable:
        btn.setCheckable(True)
    btn.setFixedSize(35, 35)
    if styled:
        btn.setStyleSheet("QPushButton { font-size: 18pt; }")
    btn.setToolTip(tooltip)
    if callback is not None:
        btn.clicked.connect(callback)
    return btn


def make_vertical_separator():
    """Make a thin vertical line that splits the button groups of a top bar.

    Args:
        None

    Returns:
        line (QFrame): The vertical separator line. The caller puts it in a layout.
    """
    line = QFrame()
    line.setFixedWidth(1)
    line.setStyleSheet("background-color: palette(mid);")
    return line


def make_legend_checkbox(canvas):
    """Make a "Legend" check box that shows or hides the legends of a plot.
    The legend starts visible. The window keeps the plot, so the box needs no other state.

    Args:
        canvas (FigureCanvasQTAgg): Canvas of the plot. The box reads its figure at each toggle.

    Returns:
        box (QCheckBox): The check box, checked. The caller puts it in a layout.
    """
    box = QCheckBox("Legend")
    box.setChecked(True)
    box.setToolTip("Show or hide the legend")

    def _toggle(checked):
        for ax in canvas.figure.axes:
            legend = ax.get_legend()
            if legend is not None:
                legend.set_visible(checked)
        canvas.draw_idle()

    box.toggled.connect(_toggle)
    return box


def make_selectable(label):
    """Let the user select the text of a label with the mouse or the keyboard.
    A double-click then selects a word, and Ctrl+C copies the selection.

    Args:
        label (QLabel): Label to change.

    Returns:
        label (QLabel): The same label, so a call can be used inside an expression.
    """
    label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
    return label


class DualPanelViewMixin:
    """Sidebar-button helpers for a window with a left and a right panel.

    The host class must define slice1_scale, slice2_scale, slice1_cmap, slice2_cmap,
    scale_btn_left, scale_btn_right, copy_selection, _region_save_wcs_text,
    _region_import_wcs_text, _toggle_grid and the three hooks _scale_image, _redraw_panel
    and _set_panel_cmap. It can set _panel_noun to the word its tooltips use for one
    panel, and _no_image_message to the warning that the Scale button shows with no image.
    """

    _panel_noun = "slice"
    _no_image_message = ("No Image Loaded", "Load an image into this panel first.")
    # Words of the sidebar tooltips. A host sets "shapes" (a tooltip for each shape name) and the
    # texts "colormap", "clear" and "save". Each text has {side} for 'left' or 'right'.
    _panel_words = None

    def _init_panel_buttons(self):
        """Build the sidebar buttons that both dual-panel windows have: the four shape buttons, colormap,
        scale, copy, edit shape, clear, save, region and grid, for the left and the right panel.
        The words of the tooltips come from _panel_words. Save, region and grid stay off until an image loads.

        Args:
            None

        Returns:
            None
        """
        words = self._panel_words
        shapes = ["POINT", "RECTANGLE", "ELLIPSE", "POLYGON"]
        self.shape_buttons = {
            side: {shape: make_panel_button(
                       shape.lower(), words["shapes"][shape],
                       lambda _checked=False, s=shape, t=side: self._set_shape(s, target=t), checkable=True)
                   for shape in shapes}
            for side in ("left", "right")
        }

        self.color_btn_left = make_panel_button(
            "colormap", words["colormap"].format(side="left"),
            lambda _checked=False: self._open_colormap_dialog('left'), styled=False)
        self.color_btn_right = make_panel_button(
            "colormap", words["colormap"].format(side="right"),
            lambda _checked=False: self._open_colormap_dialog('right'), styled=False)

        self.scale_btn_left = self._make_scale_button('left')
        self.scale_btn_right = self._make_scale_button('right')

        self.copy_menu_btn_l = self._make_copy_button('left')
        self.copy_menu_btn_r = self._make_copy_button('right')

        # Edit Shape turns on drag and resize handles. It works only on a placed shape.
        self.edit_shape_btn_l = make_panel_button(
            "edit_shape", "Toggle Edit Shape (drag/resize)",
            lambda _checked=False: self._toggle_edit_shape("left"), checkable=True)
        self.edit_shape_btn_r = make_panel_button(
            "edit_shape", "Toggle Edit Shape (drag/resize)",
            lambda _checked=False: self._toggle_edit_shape("right"), checkable=True)
        self.edit_shape_btn_l.setEnabled(False)
        self.edit_shape_btn_r.setEnabled(False)

        self.erase_btn_l = make_panel_button(
            "clear", words["clear"].format(side="left"), lambda _checked=False: self.clear_region("left"))
        self.erase_btn_r = make_panel_button(
            "clear", words["clear"].format(side="right"), lambda _checked=False: self.clear_region("right"))

        # Save writes the region only if one is drawn, else the whole image.
        self.save_slice_btn_l = make_panel_button(
            "save", words["save"].format(side="left"), lambda _checked=False: self._region_save_slice("left"))
        self.save_slice_btn_r = make_panel_button(
            "save", words["save"].format(side="right"), lambda _checked=False: self._region_save_slice("right"))

        self.region_menu_btn_l = self._make_region_button("left")
        self.region_menu_btn_r = self._make_region_button("right")

        self.grid_btn_l = self._make_grid_button("left")
        self.grid_btn_r = self._make_grid_button("right")

        for btn in (self.save_slice_btn_l, self.save_slice_btn_r,
                    self.region_menu_btn_l, self.region_menu_btn_r,
                    self.grid_btn_l, self.grid_btn_r):
            btn.setEnabled(False)

    def _scale_image(self, target):
        """Give the image that the scale dialog works on for one panel.

        Args:
            target (str): Panel to read from, 'left' or 'right'.

        Returns:
            image (numpy.ndarray): 2D image shown in the panel, or None if the panel has none.

        Raises:
            NotImplementedError: if the host does not override this hook.
        """
        raise NotImplementedError("The host must give the image of a panel in _scale_image.")

    def _redraw_panel(self, target):
        """Draw one panel again, after its scale or colormap changed. It does nothing if the panel has no image.

        Args:
            target (str): Panel to redraw, 'left' or 'right'.

        Returns:
            None

        Raises:
            NotImplementedError: if the host does not override this hook.
        """
        raise NotImplementedError("The host must redraw a panel in _redraw_panel.")

    def _set_panel_cmap(self, target, cmap):
        """Store the colormap of one panel and show it.

        Args:
            target (str): Panel to recolor, 'left' or 'right'.
            cmap (str): Matplotlib colormap name.

        Returns:
            None

        Raises:
            NotImplementedError: if the host does not override this hook.
        """
        raise NotImplementedError("The host must apply a colormap in _set_panel_cmap.")

    def _open_colormap_dialog(self, target):
        """Open the colormap picker and apply the chosen map to a panel.

        Args:
            target (str): Panel to recolor, 'left' or 'right'.

        Returns:
            None
        """
        dialog = ColorMapDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self._set_panel_cmap(target, dialog.selected_cmap)

    def copy_colormap(self, source, target):
        """Copy the colormap of one panel onto the other panel.

        Args:
            source (str): Panel to copy the colormap from, 'left' or 'right'.
            target (str): Panel to copy the colormap to, 'left' or 'right'.

        Returns:
            None
        """
        src_cmap = getattr(self, 'slice1_cmap' if source == 'left' else 'slice2_cmap', None)
        if not src_cmap:
            return
        self._set_panel_cmap(target, src_cmap)

    def _open_scale_dialog(self, target):
        """Open the scale, gamma, and color range dialog for a panel.

        Args:
            target (str): Panel to configure, 'left' or 'right'.

        Returns:
            None
        """
        image = self._scale_image(target)
        if image is None:
            QMessageBox.warning(self, *self._no_image_message)
            return

        current = self.slice1_scale if target == 'left' else self.slice2_scale
        dialog = ScaleDialog(current, image, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return

        self._store_scale_settings(target, dialog.result_settings)
        self._redraw_panel(target)

    def copy_scale(self, source, target):
        """Copy the scale type, gamma, and color range of one panel onto the other panel.

        Args:
            source (str): Panel to copy from, 'left' or 'right'.
            target (str): Panel to copy to, 'left' or 'right'.

        Returns:
            None
        """
        settings = self.slice1_scale if source == 'left' else self.slice2_scale
        self._store_scale_settings(target, settings.copy())
        self._redraw_panel(target)

    def _store_scale_settings(self, target, settings):
        """Save a panel's scale settings and refresh its Scale button tooltip.

        Args:
            target (str): Panel to update, 'left' or 'right'.
            settings (ScaleSettings): The new settings.

        Returns:
            None
        """
        if target == 'left':
            self.slice1_scale = settings
        else:
            self.slice2_scale = settings
        btn = self.scale_btn_left if target == 'left' else self.scale_btn_right
        btn.setToolTip(f"Intensity scaling for {target} slice "
                       f"({settings.scale_label()}, {settings.range_label()})")

    def _make_scale_button(self, target):
        """Build a square button that opens the scale/range dialog.

        Args:
            target (str): Panel the button applies to ('left' or 'right').

        Returns:
            btn (QPushButton): The configured button.
        """
        btn = QPushButton()
        set_button_icon(btn, "scale")
        btn.setFixedSize(35, 35)
        btn.setStyleSheet("QPushButton { font-size: 18pt; }")
        btn.clicked.connect(lambda: self._open_scale_dialog(target))
        btn.setToolTip(f"Intensity scaling for {target} slice (Linear, Auto range)")
        return btn

    def _make_copy_button(self, target):
        """Build a square button that opens a menu to copy selection,
        colormap, or scale/range from the other panel.

        Args:
            target (str): Panel the button applies to ('left' or 'right').

        Returns:
            btn (QPushButton): The configured button.
        """
        other = "right" if target == "left" else "left"
        btn = QPushButton()
        set_button_icon(btn, "copy")
        btn.setFixedSize(35, 35)
        btn.setStyleSheet("QPushButton { font-size: 18pt; }")
        menu = QMenu(btn)
        menu.addAction("Copy Selection",
                       lambda: self.copy_selection(source=other, target=target))
        menu.addAction("Copy Colormap", lambda: self.copy_colormap(other, target))
        menu.addAction("Copy Scale/Range", lambda: self.copy_scale(other, target))
        btn.clicked.connect(lambda: menu.exec_(btn.mapToGlobal(btn.rect().bottomLeft())))
        btn.setToolTip(f"Copy selection, colormap, or scale/range from the {other} panel")
        return btn

    def _make_region_button(self, target):
        """Build a square button that opens a menu to save or import the region of one panel.

        Args:
            target (str): Panel the button applies to ('left' or 'right').

        Returns:
            btn (QPushButton): The configured button.
        """
        btn = QPushButton()
        set_button_icon(btn, "region")
        btn.setFixedSize(35, 35)
        btn.setStyleSheet("QPushButton { font-size: 18pt; }")
        menu = QMenu(btn)
        menu.addAction("Save Region...", lambda: self._region_save_wcs_text(target))
        menu.addAction("Import Region...", lambda: self._region_import_wcs_text(target))
        btn.clicked.connect(lambda: menu.exec_(btn.mapToGlobal(btn.rect().bottomLeft())))
        btn.setToolTip(
            f"Save the {target} region as sky coordinates, "
            f"or import a saved region onto the {target} {self._panel_noun}")
        return btn

    def _make_grid_button(self, target):
        """Build a square toggle button that shows the RA/Dec grid on one panel.

        Args:
            target (str): Panel the button applies to ('left' or 'right').

        Returns:
            btn (QPushButton): The configured button, not checked.
        """
        btn = QPushButton()
        set_button_icon(btn, "grid")
        btn.setCheckable(True)
        btn.setFixedSize(35, 35)
        btn.setStyleSheet("QPushButton { font-size: 18pt; }")
        btn.setToolTip(f"Show the RA/Dec grid on the {target} {self._panel_noun}")
        btn.toggled.connect(lambda checked: self._toggle_grid(target, checked))
        return btn


class MapResultsDialogMixin:
    """Pixel-select toggle, scale dialog and scale reset shared by the map results dialogs.

    The host class must define scale_settings and _update_plot, plus the hook _scale_dialog_inputs
    for the scale dialog, and pixel_select_btn and pixel_selection_enabled for the pixel-select toggle.
    """

    _no_scale_data_message = ("No Data", "This parameter has no map to scale.")

    def _scale_settings_changed(self):
        """Run after the user chose new scale settings and before the map is drawn again.
        A host that keeps the settings somewhere else overrides this. It does nothing by default.

        Args:
            None

        Returns:
            None
        """

    def _scale_dialog_inputs(self):
        """Give the data and the percentile mask for the scale dialog, for the map on screen.

        Args:
            None

        Returns:
            inputs (tuple): (data, mask). data is a 2D float array. mask is a boolean array or None.
                The result is None if the map on screen has no data.

        Raises:
            NotImplementedError: if the host does not override this hook.
        """
        raise NotImplementedError("The host must give the map data in _scale_dialog_inputs.")

    def _open_scale_dialog(self):
        """Open the shared scale/range picker for the map on screen and redraw.

        Args:
            None

        Returns:
            None
        """
        inputs = self._scale_dialog_inputs()
        if inputs is None:
            QMessageBox.warning(self, *self._no_scale_data_message)
            return
        data, mask = inputs
        dialog = ScaleDialog(self.scale_settings, data, mask=mask, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        self.scale_settings = dialog.result_settings
        self._scale_settings_changed()
        self._update_plot()

    def _toggle_pixel_selection(self):
        """Toggle click-a-pixel mode so the matplotlib toolbar can be used instead.

        Args:
            None

        Returns:
            None
        """
        self.pixel_selection_enabled = self.pixel_select_btn.isChecked()
        if self.pixel_selection_enabled:
            self.pixel_select_btn.setText("Pixel Select: ON")
            self.pixel_select_btn.setStyleSheet("background-color: #90EE90; font-weight: bold;")
        else:
            self.pixel_select_btn.setText("Pixel Select: OFF")
            self.pixel_select_btn.setStyleSheet("background-color: #FFB6C6; font-weight: bold;")

    def _on_param_changed(self):
        """Reset the scale to Linear with an Auto range, then redraw. A scale
        chosen for one map does not suit another.

        Args:
            None

        Returns:
            None
        """
        self.scale_settings.reset()
        self._update_plot()
