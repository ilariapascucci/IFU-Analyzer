"""
Size helpers that keep windows and button columns inside any screen
"""

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QApplication, QWidget


def fit_to_screen(window, width, height, fraction=0.9):
    """Resize a window to the wanted size, but never larger than part of the screen.

    Args:
        window (QWidget): Window or dialog to resize.
        width (int): Wanted width in pixels.
        height (int): Wanted height in pixels.
        fraction (float): Largest share of the usable screen that the window may take.

    Returns:
        None
    """
    area = QApplication.primaryScreen().availableGeometry()
    window.resize(min(width, int(area.width() * fraction)),
                  min(height, int(area.height() * fraction)))


def screen_limited_width(width, fraction=0.9):
    """Reduce a width so it fits inside part of the usable screen width.

    Args:
        width (int): Wanted width in pixels.
        fraction (float): Largest share of the usable screen width.

    Returns:
        width (int): The wanted width, or the limit if the wanted width is larger.
    """
    area = QApplication.primaryScreen().availableGeometry()
    return min(width, int(area.width() * fraction))


def fit_dialog_to_table(dialog, table, max_width=1600):
    """Widen a dialog so that every column of its table shows in full.
    The width is the natural width of the columns, limited to part of the screen.
    The height of the dialog does not change.

    Args:
        dialog (QWidget): Dialog that holds the table.
        table (QTableWidget): Table that sets the width.
        max_width (int): Largest width in pixels, before the screen limit.

    Returns:
        None
    """
    # The last column is not stretched while we measure, so it shows its own width.
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    natural_width = header.length()
    header.setStretchLastSection(True)
    frame_overhead = (table.verticalHeader().width() + 2 * table.frameWidth()
                      + table.style().pixelMetric(table.style().PM_ScrollBarExtent) + 40)
    dialog.resize(min(max(dialog.width(), natural_width + frame_overhead),
                      screen_limited_width(max_width)), dialog.height())


class ResponsiveButtonColumn(QWidget):
    """Stack buttons in one column, or in two columns when the height is short.

    A column of ten square buttons sets a tall minimum height for a window.
    This widget lets the window get shorter, because it switches to two
    columns when the column would not fit.

    Args:
        buttons (list): Buttons to show, from top to bottom.
        parent (QWidget): Parent widget, or None.
    """

    _GAP = 6

    def __init__(self, buttons, parent=None):
        """Place the buttons inside the widget.

        Args:
            buttons (list): Buttons to show, from top to bottom.
            parent (QWidget): Parent widget, or None.

        Returns:
            None
        """
        super().__init__(parent)
        self._buttons = list(buttons)
        for button in self._buttons:
            button.setParent(self)
            button.show()
        self._columns = 1
        self._place(1)

    def _cell_size(self):
        """Find the size of the largest button.

        Args:
            None

        Returns:
            size (QSize): Width and height of one grid cell.
        """
        hints = [button.sizeHint() for button in self._buttons]
        return QSize(max(h.width() for h in hints), max(h.height() for h in hints))

    def _size_for(self, columns):
        """Find the size the buttons need in a given number of columns.

        Args:
            columns (int): Number of columns.

        Returns:
            size (QSize): Width and height that the buttons need.
        """
        rows = -(-len(self._buttons) // columns)
        cell = self._cell_size()
        return QSize(columns * cell.width() + (columns - 1) * self._GAP,
                     rows * cell.height() + (rows - 1) * self._GAP)

    def _place(self, columns):
        """Move every button to its cell, filling each column from the top.

        Args:
            columns (int): Number of columns.

        Returns:
            None
        """
        rows = -(-len(self._buttons) // columns)
        cell = self._cell_size()
        for index, button in enumerate(self._buttons):
            column, row = divmod(index, rows)
            button.move(column * (cell.width() + self._GAP), row * (cell.height() + self._GAP))
        self._columns = columns

    def sizeHint(self):
        """Give the preferred size: the width of two columns, the height of one.

        Args:
            None

        Returns:
            size (QSize): Preferred size.
        """
        return QSize(self._size_for(2).width(), self._size_for(1).height())

    def minimumSizeHint(self):
        """Give the smallest size: two columns.

        Args:
            None

        Returns:
            size (QSize): Minimum size.
        """
        return self._size_for(2)

    def single_column_height(self):
        """Give the height at which the buttons switch to one column.

        Args:
            None

        Returns:
            height (int): Smallest height that holds all the buttons in one column.
        """
        return self._size_for(1).height()

    def resizeEvent(self, event):
        """Use one column when the height allows it, and two columns when it does not.

        Args:
            event (QResizeEvent): Resize event that Qt sends to the widget.

        Returns:
            None
        """
        super().resizeEvent(event)
        columns = 1 if self.height() >= self._size_for(1).height() else 2
        if columns != self._columns:
            self._place(columns)
