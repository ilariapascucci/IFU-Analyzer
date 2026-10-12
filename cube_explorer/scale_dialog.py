"""
Scale Selection Dialog
Provides a GUI dialog for choosing a slice's intensity scaling (Linear/Log/
Sqrt/Power) and an optional manual vmin/vmax color-scale range, via a
draggable dual-handle slider and/or typed Min/Max values.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QMessageBox, QWidget
)
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QPainter, QPen, QColor


class RangeSliderWidget(QWidget):
    """Horizontal dual-handle slider for picking vmin/vmax. Handle positions
    use the same value -> [0, 1] transform the active scale applies to
    colors, so equal drag distance means an equal step in color space.
    """

    rangeChanged = pyqtSignal(float, float)
    HANDLE_RADIUS = 7
    TRACK_MARGIN = 12

    def __init__(self, data_min, data_max, scale_type, vmin=None, vmax=None, parent=None):
        """Build the slider over [data_min, data_max], with handles at vmin/vmax.

        Args:
            data_min (float): Lower bound of the selectable range (the
                slice's own auto minimum).
            data_max (float): Upper bound of the selectable range (the
                slice's own auto maximum).
            scale_type (str): Initial layout transform ('Linear', 'Log',
                'Sqrt', or 'Power (gamma=2)').
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
        self._dragging = None  # 'min', 'max', or None
        self.setMouseTracking(True)

    def set_scale_type(self, scale_type):
        """Switch the layout transform and re-lay-out the handles (values
        unchanged -- only their on-screen position moves).

        Args:
            scale_type (str): New layout transform.

        Returns:
            None
        """
        self._scale_type = scale_type
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
        if self._scale_type == "Sqrt":
            return ratio ** 0.5
        if self._scale_type.startswith("Power"):
            return ratio ** 2
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
        if self._scale_type == "Sqrt":
            return lo + (f ** 2) * (hi - lo)
        if self._scale_type.startswith("Power"):
            return lo + (f ** 0.5) * (hi - lo)
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
        self._dragging = None

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
        if self._dragging == 'min':
            self._vmin = min(value, self._vmax)
        else:
            self._vmax = max(value, self._vmin)
        self.update()
        self.rangeChanged.emit(self._vmin, self._vmax)


class ScaleDialog(QDialog):
    """Dialog to choose intensity scaling and an optional manual vmin/vmax range."""

    SCALE_TYPES = ["Linear", "Log", "Sqrt", "Power (γ=2)"]

    def __init__(self, current_scale, auto_vmin, auto_vmax, vmin=None, vmax=None, parent=None):
        """Build the scale-type + vmin/vmax picker.

        Args:
            current_scale (str): Currently active scale type, preselected.
            auto_vmin (float): This slice's own minimum, shown as a
                placeholder/default when no manual vmin is set.
            auto_vmax (float): This slice's own maximum, shown as a
                placeholder/default when no manual vmax is set.
            vmin (float): Current manual vmin override, or None for auto.
            vmax (float): Current manual vmax override, or None for auto.
            parent (QWidget): Parent widget, or None for a top-level dialog.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle("Scale and Color Range")
        self.setMinimumWidth(340)
        self.auto_vmin = auto_vmin
        self.auto_vmax = auto_vmax
        self.result_scale = current_scale
        self.result_vmin = None
        self.result_vmax = None

        layout = QVBoxLayout()
        info = QLabel("Choose the intensity scaling and, optionally, a fixed "
                      "color-range (drag the slider or type vmin/vmax). Leave "
                      "vmin/vmax blank for Auto.")
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(self.SCALE_TYPES)
        if current_scale in self.SCALE_TYPES:
            self.scale_combo.setCurrentText(current_scale)
        self.scale_combo.currentTextChanged.connect(self._on_scale_changed)
        form.addRow("Scaling:", self.scale_combo)
        layout.addLayout(form)

        slider_vmin = vmin if vmin is not None else auto_vmin
        slider_vmax = vmax if vmax is not None else auto_vmax
        self.slider = RangeSliderWidget(auto_vmin, auto_vmax, self.scale_combo.currentText(),
                                        vmin=slider_vmin, vmax=slider_vmax)
        self.slider.rangeChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        form2 = QFormLayout()
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.ScientificNotation)

        self.min_edit = QLineEdit()
        self.min_edit.setValidator(validator)
        self.min_edit.setPlaceholderText(f"Auto ({auto_vmin:.3e})")
        if vmin is not None:
            self.min_edit.setText(f"{vmin:.6e}")
        self.min_edit.editingFinished.connect(self._sync_slider_from_fields)
        form2.addRow("vmin:", self.min_edit)

        self.max_edit = QLineEdit()
        self.max_edit.setValidator(validator)
        self.max_edit.setPlaceholderText(f"Auto ({auto_vmax:.3e})")
        if vmax is not None:
            self.max_edit.setText(f"{vmax:.6e}")
        self.max_edit.editingFinished.connect(self._sync_slider_from_fields)
        form2.addRow("vmax:", self.max_edit)

        layout.addLayout(form2)

        reset_btn = QPushButton("Reset to Auto")
        reset_btn.clicked.connect(self._reset_to_auto)
        layout.addWidget(reset_btn)

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

    def _on_scale_changed(self, scale_type):
        """Re-lay-out the slider's handles under the new scale, and push its
        current values into the Min/Max fields live -- a drag is not the
        only trigger for an update.

        Args:
            scale_type (str): Newly selected scale type.

        Returns:
            None
        """
        self.slider.set_scale_type(scale_type)
        self._on_slider_changed(*self.slider.value_range())

    def _on_slider_changed(self, vmin, vmax):
        """Mirror a slider drag into the Min/Max text fields.

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
        """Clear both Min/Max fields and reset the slider, reverting to
        automatic scaling.

        Args:
            None

        Returns:
            None
        """
        self.min_edit.clear()
        self.max_edit.clear()
        self.slider.set_range(self.auto_vmin, self.auto_vmax)

    def accept(self):
        """Validate and parse Min/Max (blank = auto / None) before closing.

        Args:
            None

        Returns:
            None
        """
        self.result_scale = self.scale_combo.currentText()

        min_text = self.min_edit.text().strip()
        max_text = self.max_edit.text().strip()
        try:
            self.result_vmin = float(min_text) if min_text else None
            self.result_vmax = float(max_text) if max_text else None
        except ValueError:
            QMessageBox.warning(self, "Invalid Value",
                                "vmin/vmax must be numbers (e.g. 1.5e-16), or left blank for Auto.")
            return

        if (self.result_vmin is not None and self.result_vmax is not None
                and self.result_vmin >= self.result_vmax):
            QMessageBox.warning(self, "Invalid Range", "vmin must be less than vmax.")
            return

        super().accept()
