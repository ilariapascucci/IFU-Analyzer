"""
Load the app's SVG icons and color them to match the current Qt theme.
"""

import os

from PyQt5.QtCore import Qt, QRectF, QSize
from PyQt5.QtGui import QIcon, QPainter, QPalette, QPixmap
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import QApplication

ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")

# Selection shape symbol to icon name. The symbol is still the shape id in the selection code.
SHAPE_ICON_NAMES = {"*": "point", "▭": "rectangle", "○": "circle", "⬭": "ellipse", "✏": "freehand"}


def _tinted_pixmap(renderer, size, color):
    """Render an SVG at the given size and fill its shapes with one color.

    Args:
        renderer (QSvgRenderer): Loaded SVG renderer.
        size (QSize): Icon size in device-independent pixels.
        color (QColor): Fill color for the icon.

    Returns:
        pixmap (QPixmap): Transparent pixmap that holds the colored icon.
    """
    ratio = QApplication.primaryScreen().devicePixelRatio()
    pixmap = QPixmap(int(size.width() * ratio), int(size.height() * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)
    area = QRectF(0, 0, size.width(), size.height())
    painter = QPainter(pixmap)
    renderer.render(painter, area)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(area, color)
    painter.end()
    return pixmap


def load_icon(name, size=22):
    """Load an SVG icon and color it for the current theme.

    Args:
        name (str): Icon file name without the .svg extension.
        size (int): Icon width and height in pixels.

    Returns:
        icon (QIcon): Icon with a normal look and a disabled look.

    Raises:
        FileNotFoundError: if the icon file does not exist.
        ValueError: if the file is not a valid SVG.
    """
    path = os.path.join(ICON_DIR, f"{name}.svg")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Icon file not found: {path}")
    renderer = QSvgRenderer(path)
    if not renderer.isValid():
        raise ValueError(f"Icon file is not a valid SVG: {path}")
    palette = QApplication.palette()
    qsize = QSize(size, size)
    icon = QIcon()
    icon.addPixmap(
        _tinted_pixmap(renderer, qsize, palette.color(QPalette.Active, QPalette.ButtonText)), QIcon.Normal)
    icon.addPixmap(
        _tinted_pixmap(renderer, qsize, palette.color(QPalette.Disabled, QPalette.ButtonText)), QIcon.Disabled)
    return icon


def set_button_icon(button, name, size=22):
    """Show an SVG icon on a button and clear its text.

    Args:
        button (QPushButton): Button to update.
        name (str): Icon file name without the .svg extension.
        size (int): Icon width and height in pixels.

    Returns:
        None
    """
    button.setText("")
    button.setIcon(load_icon(name, size))
    button.setIconSize(QSize(size, size))
