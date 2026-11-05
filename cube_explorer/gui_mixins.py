"""
Small Qt mixins for methods that two windows or two dialogs had copied.

DualPanelViewMixin serves the 3D and 2D dual-panel windows. MapResultsDialogMixin
serves the Flux Map and Gaussian Map results dialogs.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QMenu, QCheckBox

from .icon_loader import set_button_icon


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
    """Scale-button and copy-button helpers for a window with a left and a right panel.

    The host class must define slice1_scale, slice2_scale, scale_btn_left,
    scale_btn_right, _open_scale_dialog, copy_selection, copy_colormap and copy_scale.
    """

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


class MapResultsDialogMixin:
    """Pixel-select toggle and scale reset shared by the map results dialogs.

    The host class must define pixel_select_btn, pixel_selection_enabled,
    scale_settings and _update_plot.
    """

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
