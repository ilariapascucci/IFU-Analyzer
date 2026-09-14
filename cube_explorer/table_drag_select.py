"""Shared drag-to-check behavior for a QTableWidget's checkbox column."""

from PyQt5 import sip
from PyQt5.QtCore import QObject, QEvent, Qt


class CheckboxColumnDragSelect(QObject):
    """Event filter that lets a user click-and-drag down a table column to
    set many rows' check state at once, instead of clicking each row alone.
    """

    def __init__(self, table, columns=0):
        """Install this filter on a table's viewport.

        Args:
            table (QTableWidget): The table to watch. Each column in
                `columns` must hold checkable items.
            columns (int | set[int]): The checkbox column(s) to drag across,
                one at a time -- a drag started in one column stays locked
                to it even if the mouse drifts into another. Default 0.

        Returns:
            None: Installs an event filter on table.viewport() as a side effect.
        """
        super().__init__(table)
        self.table = table
        self.columns = {columns} if isinstance(columns, int) else set(columns)
        self._dragging = False
        self._drag_state = Qt.Checked
        self._drag_column = None
        self._last_row = None
        table.viewport().installEventFilter(self)

    def _checkable_item(self, row, column):
        """Return row's item in this column, or None if it can't be dragged.

        Args:
            row (int): Table row index.
            column (int): Table column index.

        Returns:
            item (QTableWidgetItem | None): The item, if checkable and enabled.
        """
        item = self.table.item(row, column)
        if item is None:
            return None
        flags = item.flags()
        if not (flags & Qt.ItemIsUserCheckable) or not (flags & Qt.ItemIsEnabled):
            return None
        return item

    def eventFilter(self, obj, event):
        """Qt event filter entry point; see the class docstring for the behavior.

        Args:
            obj (QObject): The watched object (the table's viewport).
            event (QEvent): The event to inspect.

        Returns:
            handled (bool): True to stop this event from propagating further.
        """
        # A mouse event can stay queued past its window's close -- by the time
        # Qt delivers it, the table's C++ object may already be gone.
        if sip.isdeleted(self.table):
            return False
        if obj is not self.table.viewport():
            return False

        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            row = self.table.rowAt(event.pos().y())
            col = self.table.columnAt(event.pos().x())
            if row < 0 or col not in self.columns:
                return False
            item = self._checkable_item(row, col)
            if item is None:
                return False
            new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
            item.setCheckState(new_state)
            self._dragging = True
            self._drag_state = new_state
            self._drag_column = col
            self._last_row = row
            return True

        if event.type() == QEvent.MouseMove and self._dragging:
            row = self.table.rowAt(event.pos().y())
            if row < 0 or row == self._last_row:
                return True
            lo, hi = sorted((self._last_row, row))
            for r in range(lo, hi + 1):
                item = self._checkable_item(r, self._drag_column)
                if item:
                    item.setCheckState(self._drag_state)
            self._last_row = row
            return True

        if event.type() == QEvent.MouseButtonRelease and self._dragging:
            self._dragging = False
            return True

        return False


class SortableCheckboxTableMixin:
    """Mixin adding click-to-sort headers and checkbox-column drag-select to
    any QTableWidget-owning class. Same host-hook pattern as RegionSelectionMixin
    (gui_region_selection.py): call _init_sortable_checkbox_table() once, then
    override _permute_extra_row_state() if the host has its own row-indexed state.
    """

    def _init_sortable_checkbox_table(self, table, sortable_columns, sort_key_fn, drag_column=0):
        """Wire click-to-sort headers and checkbox drag-select onto `table`.

        Args:
            table (QTableWidget): The table to make sortable and drag-selectable.
            sortable_columns (set[int]): Column indices that sort on header click.
            sort_key_fn (callable): (row, column) -> a sort key (float or str).
            drag_column (int | set[int]): Checkbox column(s) to drag-select
                across. Default 0.

        Returns:
            None: Wires signals and stores state on self as a side effect.
        """
        self._sort_table = table
        self._sort_key_fn = sort_key_fn
        self._SORTABLE_COLUMNS = set(sortable_columns)
        self._sort_column = None
        self._sort_order = Qt.AscendingOrder

        header = table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.sectionClicked.connect(self._on_sortable_header_clicked)
        for col in self._SORTABLE_COLUMNS:
            header_item = table.horizontalHeaderItem(col)
            if header_item:
                header_item.setToolTip("Click to sort")

        self._checkbox_drag_select = CheckboxColumnDragSelect(table, columns=drag_column)

    def _reset_sort_state(self):
        """Clear the remembered sort column/direction and the header's arrow.

        Args: None

        Returns:
            None: Resets self._sort_column/_sort_order and the header indicator.
        """
        self._sort_column = None
        self._sort_order = Qt.AscendingOrder
        self._sort_table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)

    def _on_sortable_header_clicked(self, column):
        """Click-to-sort: the first click on a column sorts ascending, a
        second click reverses it.

        Args:
            column (int): Table column index the user clicked.

        Returns:
            None: Reorders self._sort_table's rows in place.
        """
        if column not in self._SORTABLE_COLUMNS:
            return
        table = self._sort_table
        if table.rowCount() == 0:
            return

        if self._sort_column == column and self._sort_order == Qt.AscendingOrder:
            self._sort_order = Qt.DescendingOrder
        else:
            self._sort_order = Qt.AscendingOrder
        self._sort_column = column

        n = table.rowCount()
        order = sorted(
            range(n),
            key=lambda r: self._sort_key_fn(r, column),
            reverse=(self._sort_order == Qt.DescendingOrder),
        )
        self._reorder_sortable_table_rows(order)
        table.horizontalHeader().setSortIndicator(column, self._sort_order)

    def _reorder_sortable_table_rows(self, order):
        """Move self._sort_table's rows into `order`, a plain take-and-reinsert
        of items, then call _permute_extra_row_state() for the host's own state.

        Args:
            order (list[int]): New row i holds old row order[i].

        Returns:
            None: Reorders the table and calls _permute_extra_row_state().
        """
        table = self._sort_table
        n = table.rowCount()
        if list(order) == list(range(n)):
            return

        table.blockSignals(True)
        table.setUpdatesEnabled(False)
        try:
            n_cols = table.columnCount()
            old_items = {c: [table.takeItem(r, c) for r in range(n)] for c in range(n_cols)}
            for new_row, old_row in enumerate(order):
                for c in range(n_cols):
                    table.setItem(new_row, c, old_items[c][old_row])
            row_map = {old_row: new_row for new_row, old_row in enumerate(order)}
            self._permute_extra_row_state(order, row_map)
        finally:
            table.setUpdatesEnabled(True)
            table.blockSignals(False)

    def _permute_extra_row_state(self, order, row_map):
        """Hook: keep the host's own row-indexed state in sync with a reorder.
        Default no-op -- override where the host has parallel lists or dicts
        keyed by row (e.g. self.rows, or a plot-data dict keyed by row).

        Args:
            order (list[int]): New row i holds old row order[i].
            row_map (dict[int, int]): Old row index -> new row index.

        Returns:
            None: Subclasses update their own attributes in place.
        """
        pass
