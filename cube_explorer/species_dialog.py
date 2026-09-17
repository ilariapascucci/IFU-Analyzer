from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QLabel
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from .table_drag_select import SortableCheckboxTableMixin
from .line_catalog import dedupe_colocated_lines

class SpeciesDialog(QDialog, SortableCheckboxTableMixin):
    """
    Dialog that shows a table of filtered spectral lines with
    checkboxes so the user can choose which lines to plot.

    rows: list of dicts (each row has 'lambda_micron', 'species', 'transition', ...)
    plotted_rows: optional list of dicts currently plotted in the main GUI
    """

    def __init__(self, rows, parent=None, plotted_rows=None):
        """Build the filtered-lines table dialog and its checkbox state.

        Args:
            rows (list): Line rows to display, each a dict with
                'lambda_micron', 'species', 'transition', etc.
            parent (QWidget): Parent widget, or None for a top-level dialog.
            plotted_rows (list): Rows currently plotted in the main GUI, used
                to pre-check their checkboxes; empty/None checks all rows.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle("Filtered Spectral Lines")
        self.setMinimumSize(700, 400)

        self.rows = rows  # the catalog rows shown in the dialog
        # plotted_rows is a list of dicts stored in the main GUI; can be empty
        self.plotted_rows = plotted_rows or []

        # Keyed on full row identity (see _row_key), not just wavelength: two
        # HITRAN entries can share one, so wavelength-only matching double-selected.
        self.plotted_keys = {self._row_key(r) for r in self.plotted_rows}

        # Default-deselect the weaker of a near-duplicate wavelength pair (e.g.
        # two HD lines at 5.027692 um). Only affects the first-time select-all.
        _kept, _dropped = dedupe_colocated_lines(
            rows, 1e-5,
            line_of=lambda r: {"wavelength": self.get_row_wavelength(r), "a_coeff": r.get("a_coeff") or 0.0},
        )
        for weak in _dropped:
            print(f"[species dialog] default-deselected {self.get_row_wavelength(weak):.6f} um "
                  f"(A={weak.get('a_coeff') or 0:.3e}) -- co-located duplicate")
        self._dedup_dropped_keys = {self._row_key(r) for r in _dropped}

        layout = QVBoxLayout(self)

        # Title: count of matching lines
        title_text = f"{len(rows)} matching lines found"
        layout.addWidget(QLabel(title_text))

        # ---- Table ----
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([" ", "λ (µm)", "Species", "Transition", "A-coeff (s⁻¹)", "E_upper (K)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setRowCount(len(rows))
        layout.addWidget(self.table)

        # Click-to-sort on Wavelength/Species/A-coeff/E_upper, not Transition
        # (sorts badly as plain text); also wires drag-select on column 0.
        self._init_sortable_checkbox_table(self.table, {1, 2, 4, 5}, self._sort_key)

        # If we have plotted_rows, use them to determine checkboxes
        # Otherwise (first time), check all boxes
        if plotted_rows:
            default_checked = None  # Will check against plotted_keys
        else:
            default_checked = True  # First time: select all
        self.populate_table(rows, default_checked=default_checked)

        # Buttons for selecting/unselecting all rows and filtering by source
        btn_layout2 = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_rows)
        self.unselect_all_btn = QPushButton("Unselect All")
        self.unselect_all_btn.clicked.connect(self.unselect_all_rows)
        
        self.detection_only_btn = QPushButton("Detection Only")
        self.detection_only_btn.setToolTip("Check only the lines you're trying to detect (HITRAN/custom-imported), uncheck contaminants")
        self.detection_only_btn.clicked.connect(self.select_detection_only)

        self.contaminating_only_btn = QPushButton("Contaminating Only")
        self.contaminating_only_btn.setToolTip("Check only contaminating lines, uncheck the detection lines")
        self.contaminating_only_btn.clicked.connect(self.select_contaminating_only)

        btn_layout2.addWidget(self.select_all_btn)
        btn_layout2.addWidget(self.unselect_all_btn)
        btn_layout2.addWidget(self.detection_only_btn)
        btn_layout2.addWidget(self.contaminating_only_btn)
        layout.addLayout(btn_layout2)

        # ---- OK / Cancel Buttons ----
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

    @staticmethod
    def _row_key(row):
        """Identify a row precisely enough to distinguish two HITRAN entries
        sharing a wavelength but differing in A coefficient; every column
        shown in the table (plus isotopologue) goes into the key.

        Args:
            row (dict): A line row, as used throughout this dialog

        Returns:
            key (tuple): (wavelength, species, transition, A coeff, E_upper,
                isotopologue, isotopologue_name), numeric fields rounded for stable
                comparison -- None for any field that's missing/unparseable, so
                still-comparable rows don't all collide on None
        """
        def _num(value, ndigits):
            """Round a value to ndigits, or None if it can't be parsed as a float.

            Args:
                value: Value to round.
                ndigits (int): Number of decimal digits to round to.

            Returns:
                rounded (float): The rounded value, or None if unparseable.
            """
            try:
                return round(float(value), ndigits)
            except (TypeError, ValueError):
                return None

        wave = row.get('lambda_micron') if 'lambda_micron' in row else row.get('wavelength')
        return (
            _num(wave, 6), row.get('species'), row.get('transition'),
            _num(row.get('a_coeff'), 12), _num(row.get('e_upper'), 6),
            row.get('isotopologue'), row.get('isotopologue_name'),
        )

    def populate_table(self, rows, default_checked=True):
        """Populate the table; each row gets a checkbox.

        Args:
            rows (list): Line rows to display.
            default_checked (bool): True checks all rows, False unchecks
                all rows, None checks only rows in self.plotted_keys.

        Returns:
            None
        """
        for i, row in enumerate(rows):
            # Decide default state:
            if default_checked is True:
                checked = self._row_key(row) not in self._dedup_dropped_keys
            elif default_checked is False:
                checked = False
            else:  # default_checked is None -> reflect plotted_rows membership
                checked = self._row_key(row) in self.plotted_keys

            # Tint rows by source (HITRAN/contaminant/custom) to match the
            # same colors used for these lines on the spectrum plot.
            tint = self._row_tint(row)

            # A real checkable item, not a QCheckBox widget -- a widget does not
            # survive a header-click row reorder (see gui_rotational_diagram.py).
            select_item = QTableWidgetItem()
            select_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            select_item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            if tint is not None:
                select_item.setBackground(tint)
            self.table.setItem(i, 0, select_item)

            # Wavelength
            lam = row.get('lambda_micron') if 'lambda_micron' in row else row.get('wavelength')
            item = QTableWidgetItem(f"{float(lam):.6f}")
            if tint is not None:
                item.setBackground(tint)
            self.table.setItem(i, 1, item)

            # Species (append isotopologue formula/number when available, e.g. "CO (13C16O)")
            species_text = row.get("species", "")
            iso_name = row.get("isotopologue_name")
            iso_num = row.get("isotopologue")
            if iso_name:
                species_text = f"{species_text} ({iso_name})"
            elif iso_num:
                species_text = f"{species_text} (iso {iso_num})"
            item = QTableWidgetItem(species_text)
            if tint is not None:
                item.setBackground(tint)
            self.table.setItem(i, 2, item)

            # Transition
            item = QTableWidgetItem(row.get("transition") or "N/A")
            if tint is not None:
                item.setBackground(tint)
            self.table.setItem(i, 3, item)

            # A-coeff -- "N/A" (not blank/0) when missing, since 0 is itself a
            # meaningful value and shouldn't be confused with "not provided".
            a_coeff = row.get("a_coeff")
            if a_coeff is not None:
                try:
                    item = QTableWidgetItem(f"{float(a_coeff):.2e}")
                except Exception:
                    item = QTableWidgetItem(str(a_coeff))
            else:
                item = QTableWidgetItem("N/A")
            if tint is not None:
                item.setBackground(tint)
            self.table.setItem(i, 4, item)

            # E_upper -- same "N/A when missing" treatment as A-coeff above.
            e_upper = row.get("e_upper")
            if e_upper is not None:
                try:
                    item = QTableWidgetItem(f"{float(e_upper):.2f}")
                except Exception:
                    item = QTableWidgetItem(str(e_upper))
            else:
                item = QTableWidgetItem("N/A")
            if tint is not None:
                item.setBackground(tint)
            self.table.setItem(i, 5, item)

        # Narrow first column
        self.table.setColumnWidth(0, 30)

    @staticmethod
    def _row_tint(row):
        """Faint background tint by source, or None for an untinted (Custom) row.

        Args:
            row (dict): A line row, as used throughout this dialog.

        Returns:
            color (QColor): Faint tint color, or None for an untinted row.
        """
        source = row.get('source')
        if source == 'HITRAN':
            return QColor(255, 200, 200, 80)  # Faint red, matches the darkred/orange marker
        if source == 'Contaminant':
            return QColor(190, 190, 190, 80)  # Faint grey, matches the black/dimgray marker
        return None

    def get_selected_rows(self):
        """Return the rows currently checked in the table.

        Args:
            None

        Returns:
            selected (list): Checked line rows, in table order.
        """
        selected = []
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item and item.checkState() == Qt.Checked:
                selected.append(self.rows[i])
        print("DEBUG selected rows:", [r.get('lambda_micron') or r.get('wavelength') for r in selected])
        return selected

    @staticmethod
    def get_row_wavelength(row):
        """Return a row's wavelength, from whichever key it's stored under.

        Args:
            row (dict): A line row, as used throughout this dialog.

        Returns:
            wavelength (float): The row's wavelength.
        """
        return row.get("lambda_micron") or row.get("wavelength")

    def select_all_rows(self):
        """Check every row's checkbox.

        Args:
            None

        Returns:
            None
        """
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item:
                item.setCheckState(Qt.Checked)

    def unselect_all_rows(self):
        """Uncheck every row's checkbox.

        Args:
            None

        Returns:
            None
        """
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item:
                item.setCheckState(Qt.Unchecked)

    def select_detection_only(self):
        """Check only the lines being detected (HITRAN or custom-imported), uncheck contaminants.

        Args:
            None

        Returns:
            None
        """
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item:
                item.setCheckState(Qt.Unchecked if self.rows[i].get('source') == 'Contaminant' else Qt.Checked)

    def select_contaminating_only(self):
        """Check only contaminating lines, uncheck the detection lines.

        Args:
            None

        Returns:
            None
        """
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item:
                item.setCheckState(Qt.Checked if self.rows[i].get('source') == 'Contaminant' else Qt.Unchecked)

    def _sort_key(self, row, column):
        """Sort key for one cell: numeric for Wavelength/A-coeff/E_upper,
        lowercased text for Species. A missing numeric value ranks lowest.

        Args:
            row (int): Table row index.
            column (int): Table column index. Must be in self._SORTABLE_COLUMNS.

        Returns:
            key (float | str): Sort key for this cell.
        """
        item = self.table.item(row, column)
        text = item.text().strip() if item else ""
        if column == 2:
            return text.lower()
        try:
            return float(text)
        except ValueError:
            return float('-inf')

    def _permute_extra_row_state(self, order, row_map):
        """Keep self.rows (row -> catalog dict) in step with a table reorder.

        Args:
            order (list[int]): New row i holds old row order[i].
            row_map (dict[int, int]): Old row index -> new row index. Unused
                here -- self.rows only needs the same permutation as the table.

        Returns:
            None: Reassigns self.rows in place.
        """
        self.rows = [self.rows[old_row] for old_row in order]
