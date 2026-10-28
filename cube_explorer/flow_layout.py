"""
Wrapping layout for rows of controls that must fit any window width
"""

from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtWidgets import QHBoxLayout, QLayout, QWidget


class FlowLayout(QLayout):
    """Lay out widgets left to right and wrap to a new line when the row is full.

    This is the Qt equivalent of CSS flex-wrap. The minimum width is the
    widest single widget, so a window that holds the layout can get narrow.

    Args:
        parent (QWidget): Widget that owns the layout, or None.
        h_spacing (int): Gap in pixels between widgets on one line.
        v_spacing (int): Gap in pixels between lines.
    """

    def __init__(self, parent=None, h_spacing=6, v_spacing=4):
        """Create an empty wrapping layout.

        Args:
            parent (QWidget): Widget that owns the layout, or None.
            h_spacing (int): Gap in pixels between widgets on one line.
            v_spacing (int): Gap in pixels between lines.

        Returns:
            None
        """
        super().__init__(parent)
        self._items = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        """Add one layout item at the end of the layout.

        Args:
            item (QLayoutItem): Item that Qt hands to the layout.

        Returns:
            None
        """
        self._items.append(item)

    def count(self):
        """Count the items in the layout.

        Args:
            None

        Returns:
            count (int): Number of items.
        """
        return len(self._items)

    def itemAt(self, index):
        """Get the item at a position.

        Args:
            index (int): Position of the item.

        Returns:
            item (QLayoutItem): The item, or None when the index is out of range.
        """
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        """Remove the item at a position and return it.

        Args:
            index (int): Position of the item.

        Returns:
            item (QLayoutItem): The removed item, or None when the index is out of range.
        """
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        """Tell Qt that the layout does not ask for extra space.

        Args:
            None

        Returns:
            directions (Qt.Orientations): No expanding direction.
        """
        return Qt.Orientations()

    def hasHeightForWidth(self):
        """Tell Qt that the height depends on the width.

        Args:
            None

        Returns:
            has_height_for_width (bool): Always True.
        """
        return True

    def heightForWidth(self, width):
        """Calculate the height the layout needs at a given width.

        Args:
            width (int): Available width in pixels.

        Returns:
            height (int): Height in pixels.
        """
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        """Place every widget inside the given rectangle.

        Args:
            rect (QRect): Area that the layout fills.

        Returns:
            None
        """
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        """Give the preferred size, which equals the minimum size.

        Args:
            None

        Returns:
            size (QSize): Preferred size.
        """
        return self.minimumSize()

    def minimumSize(self):
        """Give the smallest size: the largest single widget plus the margins.

        Args:
            None

        Returns:
            size (QSize): Minimum size.
        """
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _do_layout(self, rect, test_only):
        """Place the widgets line by line, or only measure the height.

        Args:
            rect (QRect): Area that the layout fills.
            test_only (bool): True to measure the height without moving widgets.

        Returns:
            height (int): Height in pixels that the lines need.
        """
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x = area.x()
        y = area.y()
        line_height = 0
        for item in self._items:
            # Hidden widgets report empty and take no space.
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            if x > area.x() and x + hint.width() > area.right() + 1:
                x = area.x()
                y += line_height + self._v_spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._h_spacing
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


def row_of(widgets):
    """Pack widgets into one horizontal unit that a FlowLayout never splits.

    Args:
        widgets (list): Widgets to place side by side.

    Returns:
        unit (QWidget): Widget that holds the row.
    """
    unit = QWidget()
    row = QHBoxLayout(unit)
    row.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        row.addWidget(widget)
    return unit
