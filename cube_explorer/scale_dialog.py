"""
Scaling
Intensity scaling shared by every image and map viewer: the scale settings
(Linear/Log/Power with gamma, absolute or percentile color range), the matplotlib
norm built from them, and the Scale dialog (slider plus typed values) that edits them.
"""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
from matplotlib.colors import LogNorm, PowerNorm, Normalize
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QMessageBox, QWidget
)
from PyQt5.QtCore import QPointF, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QPainter, QPen, QColor

SCALE_TYPES = ["Linear", "Log", "Power"]
RANGE_ABSOLUTE = "absolute"
RANGE_PERCENTILE = "percentile"
DEFAULT_GAMMA = 0.5
DEFAULT_PERCENTILES = (2.0, 98.0)


@dataclass
class ScaleSettings:
    """One viewer panel's scale type, gamma, and color range.

    Args:
        scale (str): 'Linear', 'Log', or 'Power'.
        gamma (float): Exponent for the Power scale. Must be above 0.
        range_mode (str): 'absolute' (vmin/vmax) or 'percentile' (p_lo/p_hi).
        vmin (float): Absolute lower bound, or None for the image minimum.
        vmax (float): Absolute upper bound, or None for the image maximum.
        p_lo (float): Lower percentile, from 0 to 100.
        p_hi (float): Upper percentile, from 0 to 100.
    """

    scale: str = "Linear"
    gamma: float = DEFAULT_GAMMA
    range_mode: str = RANGE_ABSOLUTE
    vmin: float = None
    vmax: float = None
    p_lo: float = DEFAULT_PERCENTILES[0]
    p_hi: float = DEFAULT_PERCENTILES[1]

    def copy(self):
        """Return an independent copy of these settings.

        Args:
            None

        Returns:
            settings (ScaleSettings): A new object with the same values.
        """
        return dataclasses.replace(self)

    def clear_absolute_range(self):
        """Reset vmin and vmax to Auto. Scale, gamma, and percentiles stay.

        Args:
            None

        Returns:
            None
        """
        self.vmin = None
        self.vmax = None

    def scale_label(self):
        """Return the scale name for a label, with gamma for Power.

        Args:
            None

        Returns:
            label (str): The scale name. Power also shows the gamma value.
        """
        if self.scale == "Power":
            return f"Power (γ={self.gamma:g})"
        return self.scale

    def range_label(self):
        """Return a short text for the color range, for tooltips.

        Args:
            None

        Returns:
            label (str): 'Auto', 'vmin to vmax', or 'lo-hi%'.
        """
        if self.range_mode == RANGE_PERCENTILE:
            return f"{self.p_lo:g}-{self.p_hi:g}%"
        if self.vmin is None and self.vmax is None:
            return "Auto"
        low = "Auto" if self.vmin is None else f"{self.vmin:.3e}"
        high = "Auto" if self.vmax is None else f"{self.vmax:.3e}"
        return f"{low} to {high}"


def finite_pixels(image, mask=None):
    """Return the finite pixel values of an image, as a 1D array.

    Args:
        image (numpy.ndarray): 2D array of pixel values.
        mask (numpy.ndarray): Boolean array of the same shape. Only True
            pixels count. None means all pixels count.

    Returns:
        values (numpy.ndarray): 1D array of the finite pixels.
    """
    image = np.asarray(image, dtype=float)
    keep = np.isfinite(image)
    if mask is not None:
        keep = keep & mask
    return image[keep]


def resolve_range(image, settings, mask=None):
    """Return the (vmin, vmax) the color scale uses for this image.

    Absolute mode uses the typed bounds, or the image minimum or maximum for
    a blank bound. Percentile mode uses the pixels in mask (all finite if none).

    Args:
        image (numpy.ndarray): 2D array of pixel values.
        settings (ScaleSettings): The panel's scale settings.
        mask (numpy.ndarray): Boolean array. Only True pixels set the
            percentiles. None means all finite pixels.

    Returns:
        range (tuple): (vmin, vmax) as floats. (0.0, 1.0) if no finite pixel.
    """
    finite = finite_pixels(image)
    if finite.size == 0:
        return 0.0, 1.0
    if settings.range_mode == RANGE_PERCENTILE:
        population = finite_pixels(image, mask)
        if population.size == 0:
            population = finite
        lo, hi = np.percentile(population, [settings.p_lo, settings.p_hi])
        return float(lo), float(hi)
    vmin = float(finite.min()) if settings.vmin is None else settings.vmin
    vmax = float(finite.max()) if settings.vmax is None else settings.vmax
    return vmin, vmax


def build_norm(image, settings, mask=None):
    """Build the matplotlib norm for an image, and the array to draw with it.

    Log shifts the image, vmin, and vmax by one constant if any value is at
    or below zero. The array to draw is then the shifted image.

    Args:
        image (numpy.ndarray): 2D array of pixel values.
        settings (ScaleSettings): The panel's scale settings.
        mask (numpy.ndarray): Boolean array for the percentile population,
            or None for all finite pixels.

    Returns:
        result (tuple): (norm, shown). norm is the matplotlib Normalize
            object. shown is the array to pass to imshow.
    """
    vmin, vmax = resolve_range(image, settings, mask)
    shown = image
    if settings.scale == "Log":
        finite = finite_pixels(image)
        img_min = float(finite.min()) if finite.size else 0.0
        if img_min <= 0:
            span = max(abs(img_min), abs(float(finite.max()))) if finite.size else 0.0
            # The floor must stay far below the data, so it shrinks for tiny values.
            eps = 1e-10 * span if 0 < span < 1 else 1e-10
            shift = -img_min + eps
            shown = image + shift
            vmin = vmin + shift
            vmax = vmax + shift
        return LogNorm(vmin=vmin, vmax=vmax), shown
    if settings.scale == "Power":
        return PowerNorm(gamma=settings.gamma, vmin=vmin, vmax=vmax), shown
    return Normalize(vmin=vmin, vmax=vmax), shown


def scaling_stats_text(image, settings, mask=None):
    """Return the three stats lines that describe the active color scale.

    Args:
        image (numpy.ndarray): 2D array of pixel values.
        settings (ScaleSettings): The panel's scale settings.
        mask (numpy.ndarray): Boolean array for the percentile population,
            or None for all finite pixels.

    Returns:
        text (str): The 'Scaling:', 'vmin=', and 'vmax=' lines.
    """
    vmin, vmax = resolve_range(image, settings, mask)
    low_tag = high_tag = ""
    if settings.range_mode == RANGE_PERCENTILE:
        low_tag, high_tag = f" ({settings.p_lo:g}%)", f" ({settings.p_hi:g}%)"
    return (
        f"Scaling: {settings.scale_label()}\n"
        f"vmin={vmin:.3e}{low_tag}\n"
        f"vmax={vmax:.3e}{high_tag}\n"
    )


class RangeSliderWidget(QWidget):
    """Horizontal dual-handle slider for picking vmin/vmax. Handle positions
    use the same value -> [0, 1] transform the active scale applies to
    colors, so equal drag distance means an equal step in color space.
    """

    rangeChanged = pyqtSignal(float, float)
    dragFinished = pyqtSignal(float, float)
    HANDLE_RADIUS = 7
    TRACK_MARGIN = 12

    def __init__(self, data_min, data_max, scale_type, gamma=1.0, vmin=None, vmax=None, parent=None):
        """Build the slider over [data_min, data_max], with handles at vmin/vmax.

        Args:
            data_min (float): Lower bound of the selectable range (the
                slice's own auto minimum).
            data_max (float): Upper bound of the selectable range (the
                slice's own auto maximum).
            scale_type (str): Initial layout transform ('Linear', 'Log',
                or 'Power').
            gamma (float): Exponent for the Power layout. Must be above 0.
            vmin (float): Initial lower handle value, or None for data_min.
            vmax (float): Initial upper handle value, or None for data_max.
            parent (QWidget): Parent widget, or None.

        Returns:
            None
        """
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setMinimumWidth(220)
        self._data_min = float(data_min)
        self._data_max = float(data_max) if data_max > data_min else float(data_min) + 1.0
        self._vmin = float(vmin) if vmin is not None else self._data_min
        self._vmax = float(vmax) if vmax is not None else self._data_max
        self._scale_type = scale_type
        self._gamma = float(gamma)
        self._dragging = None  # 'min', 'max', or None
        self._step = None  # Drag snap size. None means no snapping.
        self.setMouseTracking(True)

    def set_domain(self, data_min, data_max, scale_type, gamma, vmin, vmax, step=None):
        """Change the selectable range, the layout transform, and both handles.

        Args:
            data_min (float): New lower bound of the selectable range.
            data_max (float): New upper bound of the selectable range.
            scale_type (str): New layout transform ('Linear', 'Log', or 'Power').
            gamma (float): Exponent for the Power layout. Must be above 0.
            vmin (float): New lower handle value.
            vmax (float): New upper handle value.
            step (float): Snap size of a drag, which also keeps the handles one
                step apart. None means no snapping.

        Returns:
            None
        """
        self._data_min = float(data_min)
        self._data_max = float(data_max) if data_max > data_min else float(data_min) + 1.0
        self._scale_type = scale_type
        self._gamma = float(gamma)
        self._vmin, self._vmax = float(vmin), float(vmax)
        self._step = step
        self.update()

    def set_scale(self, scale_type, gamma):
        """Switch the layout transform and re-lay-out the handles (values
        unchanged -- only their on-screen position moves).

        Args:
            scale_type (str): New layout transform.
            gamma (float): Exponent for the Power layout. Must be above 0.

        Returns:
            None
        """
        self._scale_type = scale_type
        self._gamma = float(gamma)
        self.update()

    def set_range(self, vmin, vmax):
        """Move both handles to new values (e.g. from a typed Min/Max edit).

        Args:
            vmin (float): New lower handle value.
            vmax (float): New upper handle value.

        Returns:
            None
        """
        self._vmin, self._vmax = float(vmin), float(vmax)
        self.update()

    def value_range(self):
        """Current (vmin, vmax) handle values.

        Args:
            None

        Returns:
            result (tuple): (vmin, vmax) as floats.
        """
        return self._vmin, self._vmax

    def _layout_bounds(self):
        """Effective (lo, hi) for the track layout -- floors lo to a small
        positive value under Log, which can't lay out non-positive values,
        mirroring plot_slice's own non-positive handling for that mode.

        Args:
            None

        Returns:
            result (tuple): (lo, hi) floats, lo < hi.
        """
        lo, hi = self._data_min, self._data_max
        if self._scale_type == "Log" and lo <= 0:
            lo = hi * 1e-6 if hi > 0 else 1e-300
        return lo, hi

    def _value_to_frac(self, v):
        """Map a data value to its [0, 1] track position under the active scale.

        Args:
            v (float): Data value.

        Returns:
            frac (float): Position along the track, in [0, 1].
        """
        lo, hi = self._layout_bounds()
        if hi <= lo:
            return 0.0
        v = max(min(v, hi), lo)
        if self._scale_type == "Log":
            return (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        ratio = max(0.0, min(1.0, (v - lo) / (hi - lo)))
        if self._scale_type == "Power":
            return ratio ** self._gamma
        return ratio  # Linear

    def _frac_to_value(self, f):
        """Inverse of _value_to_frac -- map a [0, 1] track position back to
        a data value.

        Args:
            f (float): Position along the track, in [0, 1].

        Returns:
            v (float): Data value.
        """
        f = max(0.0, min(1.0, f))
        lo, hi = self._layout_bounds()
        if self._scale_type == "Log":
            return 10 ** (np.log10(lo) + f * (np.log10(hi) - np.log10(lo)))
        if self._scale_type == "Power":
            return lo + (f ** (1.0 / self._gamma)) * (hi - lo)
        return lo + f * (hi - lo)  # Linear

    def _track_x_bounds(self):
        """Pixel x-range of the draggable track (inset from the widget edges).

        Args:
            None

        Returns:
            result (tuple): (x0, x1) pixel bounds.
        """
        return self.TRACK_MARGIN, self.width() - self.TRACK_MARGIN

    def _handle_x(self, value):
        """Pixel x-position of a handle for a given data value.

        Args:
            value (float): Data value (the handle's current vmin or vmax).

        Returns:
            x (float): Pixel x-position.
        """
        x0, x1 = self._track_x_bounds()
        return x0 + self._value_to_frac(value) * (x1 - x0)

    def paintEvent(self, event):
        """Draw the track, the selected span, and both handles.

        Args:
            event (QPaintEvent): Unused; required by the Qt paint signature.

        Returns:
            None
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        x0, x1 = self._track_x_bounds()
        y = self.height() // 2

        painter.setPen(QPen(QColor(160, 160, 160), 2))
        painter.drawLine(x0, y, x1, y)

        xmin, xmax = self._handle_x(self._vmin), self._handle_x(self._vmax)
        painter.setPen(QPen(QColor(70, 130, 220), 3))
        painter.drawLine(int(xmin), y, int(xmax), y)

        painter.setBrush(QColor(70, 130, 220))
        painter.setPen(QPen(QColor(30, 60, 120), 1))
        for x in (xmin, xmax):
            painter.drawEllipse(QPointF(x, y), self.HANDLE_RADIUS, self.HANDLE_RADIUS)

    def mousePressEvent(self, event):
        """Grab whichever handle is closer to the click position.

        Args:
            event (QMouseEvent): The press event.

        Returns:
            None
        """
        x = event.pos().x()
        xmin, xmax = self._handle_x(self._vmin), self._handle_x(self._vmax)
        self._dragging = 'min' if abs(x - xmin) <= abs(x - xmax) else 'max'
        self._drag_to(x)

    def mouseMoveEvent(self, event):
        """Move the grabbed handle to follow the cursor.

        Args:
            event (QMouseEvent): The move event.

        Returns:
            None
        """
        if self._dragging:
            self._drag_to(event.pos().x())

    def mouseReleaseEvent(self, event):
        """Release the grabbed handle.

        Args:
            event (QMouseEvent): Unused; required by the Qt event signature.

        Returns:
            None
        """
        was_dragging = self._dragging is not None
        self._dragging = None
        if was_dragging:
            self.dragFinished.emit(self._vmin, self._vmax)

    def _drag_to(self, x):
        """Move the currently-grabbed handle to pixel position x, clamped so
        the two handles can't cross, then emit the new range.

        Args:
            x (int): Pixel x-position of the cursor.

        Returns:
            None
        """
        x0, x1 = self._track_x_bounds()
        frac = 0.0 if x1 <= x0 else (x - x0) / (x1 - x0)
        value = self._frac_to_value(frac)
        gap = 0.0
        if self._step:
            value = round(value / self._step) * self._step
            gap = self._step
        if self._dragging == 'min':
            self._vmin = min(value, self._vmax - gap)
        else:
            self._vmax = max(value, self._vmin + gap)
        self.update()
        self.rangeChanged.emit(self._vmin, self._vmax)


class ScaleDialog(QDialog):
    """Dialog to choose intensity scaling, gamma, and an absolute or percentile color range."""

    def __init__(self, settings, image, mask=None, parent=None):
        """Build the scale type, gamma, and color range picker.

        Args:
            settings (ScaleSettings): The panel's current settings. The dialog
                never changes this object.
            image (numpy.ndarray): The 2D image the settings apply to. It sets
                the Auto bounds and the percentile values.
            mask (numpy.ndarray): Boolean array of the pixels that set the
                percentiles, or None for all finite pixels.
            parent (QWidget): Parent widget, or None for a top-level dialog.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle("Scale and Color Range")
        self.setMinimumWidth(360)
        self.image = image
        self.mask = mask
        self.start_settings = settings.copy()
        self.result_settings = settings.copy()

        finite = finite_pixels(image)
        self.auto_vmin = float(finite.min()) if finite.size else 0.0
        self.auto_vmax = float(finite.max()) if finite.size else 1.0

        layout = QVBoxLayout()
        info = QLabel("Choose the intensity scaling and the color range. The range is "
                      "either absolute (drag the slider or type vmin/vmax, blank = Auto) "
                      "or relative (drag the slider from 0 to 100 or type the lower and "
                      "upper percentile of this image).")
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(SCALE_TYPES)
        self.scale_combo.setCurrentText(settings.scale)
        self.scale_combo.currentTextChanged.connect(self._on_scale_changed)
        form.addRow("Scaling:", self.scale_combo)

        self.gamma_label = QLabel("Gamma:")
        self.gamma_edit = QLineEdit(f"{settings.gamma:g}")
        gamma_validator = QDoubleValidator()
        gamma_validator.setNotation(QDoubleValidator.StandardNotation)
        self.gamma_edit.setValidator(gamma_validator)
        self.gamma_edit.setPlaceholderText("e.g. sqrt: gamma=0.5, square: gamma=2")
        self.gamma_edit.editingFinished.connect(self._on_scale_changed)
        form.addRow(self.gamma_label, self.gamma_edit)

        self.range_combo = QComboBox()
        self.range_combo.addItems(["Absolute (vmin / vmax)", "Percentile"])
        self.range_combo.setCurrentIndex(1 if settings.range_mode == RANGE_PERCENTILE else 0)
        self.range_combo.currentIndexChanged.connect(self._on_range_mode_changed)
        form.addRow("Color range:", self.range_combo)

        pct_validator = QDoubleValidator(0.0, 100.0, 4)
        pct_validator.setNotation(QDoubleValidator.StandardNotation)
        self.p_lo_label = QLabel("Lower percentile (%):")
        self.p_lo_edit = QLineEdit(f"{settings.p_lo:g}")
        self.p_lo_edit.setValidator(pct_validator)
        self.p_lo_edit.editingFinished.connect(self._refresh_percentile_range)
        form.addRow(self.p_lo_label, self.p_lo_edit)
        self.p_hi_label = QLabel("Upper percentile (%):")
        self.p_hi_edit = QLineEdit(f"{settings.p_hi:g}")
        self.p_hi_edit.setValidator(pct_validator)
        self.p_hi_edit.editingFinished.connect(self._refresh_percentile_range)
        form.addRow(self.p_hi_label, self.p_hi_edit)
        layout.addLayout(form)

        self.slider = RangeSliderWidget(self.auto_vmin, self.auto_vmax, settings.scale,
                                        gamma=settings.gamma,
                                        vmin=settings.vmin, vmax=settings.vmax)
        self.slider.rangeChanged.connect(self._on_slider_moved)
        self.slider.dragFinished.connect(self._on_slider_released)
        layout.addWidget(self.slider)

        form2 = QFormLayout()
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.ScientificNotation)

        self.min_edit = QLineEdit()
        self.min_edit.setValidator(validator)
        self.min_edit.setPlaceholderText(f"Auto ({self.auto_vmin:.3e})")
        if settings.vmin is not None:
            self.min_edit.setText(f"{settings.vmin:.6e}")
        self.min_edit.editingFinished.connect(self._sync_slider_from_fields)
        form2.addRow("vmin:", self.min_edit)

        self.max_edit = QLineEdit()
        self.max_edit.setValidator(validator)
        self.max_edit.setPlaceholderText(f"Auto ({self.auto_vmax:.3e})")
        if settings.vmax is not None:
            self.max_edit.setText(f"{settings.vmax:.6e}")
        self.max_edit.editingFinished.connect(self._sync_slider_from_fields)
        form2.addRow("vmax:", self.max_edit)

        layout.addLayout(form2)

        self.reset_btn = QPushButton("Reset to Auto")
        self.reset_btn.clicked.connect(self._reset_to_auto)
        layout.addWidget(self.reset_btn)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self._update_visibility()
        self._configure_slider_for_mode()

    def _is_percentile_mode(self):
        """Tell whether the Percentile color range is selected.

        Args:
            None

        Returns:
            is_percentile (bool): True for Percentile, False for Absolute.
        """
        return self.range_combo.currentIndex() == 1

    def _update_visibility(self):
        """Show the gamma row only for Power and the percentile rows only
        for Percentile. In Percentile mode, vmin and vmax are read-only.

        Args:
            None

        Returns:
            None
        """
        is_power = self.scale_combo.currentText() == "Power"
        self.gamma_label.setVisible(is_power)
        self.gamma_edit.setVisible(is_power)
        is_pct = self._is_percentile_mode()
        for widget in (self.p_lo_label, self.p_lo_edit, self.p_hi_label, self.p_hi_edit):
            widget.setVisible(is_pct)
        self.min_edit.setReadOnly(is_pct)
        self.max_edit.setReadOnly(is_pct)
        self.reset_btn.setText("Reset to Auto (Absolute)" if is_pct else "Reset to Auto")

    def _configure_slider_for_mode(self):
        """Set up the slider for the selected color range. Absolute uses the
        image values. Percentile uses a 0 to 100 track, with 0.1 steps.

        Args:
            None

        Returns:
            None
        """
        if self._is_percentile_mode():
            pair = self._percentiles_from_fields() or DEFAULT_PERCENTILES
            self.slider.set_domain(0.0, 100.0, "Linear", 1.0, pair[0], pair[1], step=0.1)
            self.slider.setToolTip("Drag a handle to set the lower and upper percentile (0 to 100).")
            self._refresh_percentile_range()
        else:
            self.slider.set_domain(self.auto_vmin, self.auto_vmax, self.scale_combo.currentText(),
                                   self._current_gamma(), self.auto_vmin, self.auto_vmax)
            self.slider.setToolTip("Drag a handle to set vmin and vmax.")
            self._sync_slider_from_fields()

    def _current_gamma(self):
        """Read the gamma box. Return the stored gamma if the box is not valid.

        Args:
            None

        Returns:
            gamma (float): The gamma value in the box, or the last valid one.
        """
        try:
            gamma = float(self.gamma_edit.text())
        except ValueError:
            return self.result_settings.gamma
        return gamma if gamma > 0 else self.result_settings.gamma

    def _on_scale_changed(self, *_):
        """Re-lay-out the slider's handles under the new scale, and push its
        current values into the Min/Max fields live -- a drag is not the
        only trigger for an update.

        Args:
            *_ (tuple): Unused signal arguments.

        Returns:
            None
        """
        self._update_visibility()
        if self._is_percentile_mode():
            return  # The percentile track is always linear.
        self.slider.set_scale(self.scale_combo.currentText(), self._current_gamma())
        self._on_slider_changed(*self.slider.value_range())

    def _on_range_mode_changed(self, _index):
        """Switch between Absolute and Percentile. Leaving Percentile keeps
        the percentile values in the vmin and vmax boxes as absolute numbers.

        Args:
            _index (int): Unused index of the new combo entry.

        Returns:
            None
        """
        self._update_visibility()
        self._configure_slider_for_mode()

    def _percentiles_from_fields(self):
        """Read the two percentile boxes.

        Args:
            None

        Returns:
            result (tuple): (p_lo, p_hi) as floats, or None if a box is not a
                valid number or the pair is not 0 <= lo < hi <= 100.
        """
        try:
            p_lo = float(self.p_lo_edit.text())
            p_hi = float(self.p_hi_edit.text())
        except ValueError:
            return None
        if not 0.0 <= p_lo < p_hi <= 100.0:
            return None
        return p_lo, p_hi

    def _refresh_percentile_range(self, move_slider=True):
        """Write the percentile values of this image into the vmin and vmax
        boxes. Move the slider handles to the percentiles. Do nothing for an
        invalid pair.

        Args:
            move_slider (bool): False when the slider moved first, so its
                handles already show the percentiles.

        Returns:
            None
        """
        pair = self._percentiles_from_fields()
        if pair is None or not self._is_percentile_mode():
            return
        trial = ScaleSettings(range_mode=RANGE_PERCENTILE, p_lo=pair[0], p_hi=pair[1])
        vmin, vmax = resolve_range(self.image, trial, self.mask)
        if move_slider:
            self.slider.set_range(pair[0], pair[1])
        self._on_slider_changed(vmin, vmax)

    def _on_slider_moved(self, low, high):
        """Show a slider drag in the boxes. Absolute mode writes vmin and vmax.
        Percentile mode writes the two percentile boxes only.

        Args:
            low (float): Lower handle value, in image units or percent.
            high (float): Upper handle value, in image units or percent.

        Returns:
            None
        """
        if self._is_percentile_mode():
            self.p_lo_edit.setText(f"{low:.1f}")
            self.p_hi_edit.setText(f"{high:.1f}")
        else:
            self._on_slider_changed(low, high)

    def _on_slider_released(self, _low, _high):
        """Update the vmin and vmax boxes after a percentile drag. This waits
        for the release because each update reads every pixel of the image.

        Args:
            _low (float): Unused lower handle value.
            _high (float): Unused upper handle value.

        Returns:
            None
        """
        if self._is_percentile_mode():
            self._refresh_percentile_range(move_slider=False)

    def _on_slider_changed(self, vmin, vmax):
        """Write the vmin and vmax values into the Min/Max text fields.

        Args:
            vmin (float): New lower handle value.
            vmax (float): New upper handle value.

        Returns:
            None
        """
        self.min_edit.setText(f"{vmin:.6e}")
        self.max_edit.setText(f"{vmax:.6e}")

    def _sync_slider_from_fields(self):
        """Move the slider's handles to match a typed Min/Max edit. A blank
        field falls back to this slice's own auto bound for the slider's
        position (the field itself stays blank, still meaning Auto).

        Args:
            None

        Returns:
            None
        """
        if self._is_percentile_mode():
            return  # The slider shows percentiles here, not the vmin and vmax values.
        min_text = self.min_edit.text().strip()
        max_text = self.max_edit.text().strip()
        try:
            vmin = float(min_text) if min_text else self.auto_vmin
            vmax = float(max_text) if max_text else self.auto_vmax
        except ValueError:
            return
        if vmin >= vmax:
            return
        self.slider.set_range(vmin, vmax)

    def _reset_to_auto(self):
        """Switch to the Absolute range, clear both Min/Max fields, and
        reset the slider, reverting to automatic scaling.

        Args:
            None

        Returns:
            None
        """
        self.range_combo.setCurrentIndex(0)
        self.min_edit.clear()
        self.max_edit.clear()
        self.slider.set_range(self.auto_vmin, self.auto_vmax)

    def accept(self):
        """Validate every box, then store the new settings in result_settings.
        Show a warning and keep the dialog open if a value is not valid.

        Args:
            None

        Returns:
            None
        """
        scale = self.scale_combo.currentText()
        try:
            gamma = float(self.gamma_edit.text())
        except ValueError:
            gamma = None
        if scale == "Power" and (gamma is None or gamma <= 0):
            QMessageBox.warning(self, "Invalid Gamma",
                                "Gamma must be a number above 0 (sqrt: 0.5, square: 2).")
            return
        new = self.start_settings.copy()
        new.scale = scale
        if gamma is not None and gamma > 0:
            new.gamma = gamma

        if self._is_percentile_mode():
            pair = self._percentiles_from_fields()
            if pair is None:
                QMessageBox.warning(self, "Invalid Percentile",
                                    "Percentiles must be numbers with 0 <= lower < upper <= 100.")
                return
            new.range_mode = RANGE_PERCENTILE
            new.p_lo, new.p_hi = pair
        else:
            min_text = self.min_edit.text().strip()
            max_text = self.max_edit.text().strip()
            try:
                vmin = float(min_text) if min_text else None
                vmax = float(max_text) if max_text else None
                if not all(v is None or math.isfinite(v) for v in (vmin, vmax)):
                    raise ValueError("vmin/vmax must be finite")
            except ValueError:
                QMessageBox.warning(self, "Invalid Value",
                                    "vmin/vmax must be finite numbers (e.g. 1.5e-16), or left blank for Auto.")
                return
            if vmin is not None and vmax is not None and vmin >= vmax:
                QMessageBox.warning(self, "Invalid Range", "vmin must be less than vmax.")
                return
            # A value outside this image moves to the nearest end of the image range.
            if vmin is not None:
                vmin = min(max(vmin, self.auto_vmin), self.auto_vmax)
            if vmax is not None:
                vmax = min(max(vmax, self.auto_vmin), self.auto_vmax)
            new.range_mode = RANGE_ABSOLUTE
            new.vmin, new.vmax = vmin, vmax

        self.result_settings = new
        super().accept()
