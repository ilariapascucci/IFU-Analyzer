from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QLabel, QWidget, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

class SpeciesDialog(QDialog):
    """
    Dialog that shows a table of filtered spectral lines with
    checkboxes so the user can choose which lines to plot.

    rows: list of dicts (each row has 'lambda_micron', 'species', 'transition', ...)
    plotted_rows: optional list of dicts currently plotted in the main GUI
    """

    def __init__(self, rows, parent=None, plotted_rows=None):
        super().__init__(parent)
        self.setWindowTitle("Filtered Spectral Lines")
        self.setMinimumSize(700, 400)

        self.rows = rows  # the catalog rows shown in the dialog
        # plotted_rows is a list of dicts stored in the main GUI (may be empty)
        self.plotted_rows = plotted_rows or []

        # Keyed on the full row identity, not just wavelength -- two HITRAN entries can
        # share a wavelength (e.g. two HD transitions at 5.027692 um with the same
        # transition label and E_upper but a different A coefficient; see _row_key).
        # Matching by wavelength alone collapsed them into one set entry, so deselecting
        # one still left the wavelength "present" via its still-checked twin, and both
        # silently re-checked themselves the next time this dialog opened.
        self.plotted_keys = {self._row_key(r) for r in self.plotted_rows}

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
        """
        Identify a row precisely enough to distinguish two HITRAN entries that share a
        wavelength (e.g. two HD transitions at 5.027692 um, same transition label and
        E_upper, different A coefficient) -- wavelength alone can't tell them apart, so
        every column shown in the table (plus isotopologue, not shown as its own column
        but folded into the Species text) goes into the key.

        Args:
            row (dict): A line row, as used throughout this dialog

        Returns:
            key (tuple): (wavelength, species, transition, A coeff, E_upper,
                isotopologue, isotopologue_name), numeric fields rounded for stable
                comparison -- None for any field that's missing/unparseable, so
                still-comparable rows don't all collide on None
        """
        def _num(value, ndigits):
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
        """
        Populate the table. Each row gets a checkbox.
        default_checked:
          - True => check all rows
          - False => uncheck all rows
          - None => check only those present in self.plotted_keys
        """
        for i, row in enumerate(rows):
            checkbox_widget = QWidget()
            checkbox = QCheckBox()
            # Decide default state:
            if default_checked is True:
                checkbox.setChecked(True)
            elif default_checked is False:
                checkbox.setChecked(False)
            else:  # default_checked is None -> reflect plotted_rows membership
                checkbox.setChecked(self._row_key(row) in self.plotted_keys)

            layout = QHBoxLayout()
            layout.addWidget(checkbox)
            layout.setAlignment(Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            checkbox_widget.setLayout(layout)

            # Tint rows by source so HITRAN / contaminant / custom-imported lines
            # read apart at a glance, echoing the same colors used on the spectrum
            # itself (darkred/orange for HITRAN, black/dimgray for contaminants).
            tint = self._row_tint(row)
            if tint is not None:
                checkbox_widget.setStyleSheet(
                    f"background-color: rgba({tint.red()}, {tint.green()}, {tint.blue()}, {tint.alpha()});"
                )

            self.table.setCellWidget(i, 0, checkbox_widget)

            # Wavelength
            lam = row.get('lambda_micron') if 'lambda_micron' in row else row.get('wavelength')
            item = QTableWidgetItem(f"{float(lam):.6f}")
            if tint is not None:
                item.setBackground(tint)
            self.table.setItem(i, 1, item)

            # Species (append isotopologue formula/number when available, e.g. "CO (¹²C¹⁶O)")
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
        """Faint background tint by source, or None for an untinted (Custom) row."""
        source = row.get('source')
        if source == 'HITRAN':
            return QColor(255, 200, 200, 80)  # Faint red, matches the darkred/orange marker
        if source == 'Contaminant':
            return QColor(190, 190, 190, 80)  # Faint grey, matches the black/dimgray marker
        return None

    def get_selected_rows(self):
        selected = []
        for i in range(self.table.rowCount()):
            checkbox_widget = self.table.cellWidget(i, 0)
            if checkbox_widget is None:
                continue
            checkbox = checkbox_widget.layout().itemAt(0).widget()  # get QCheckBox
            if checkbox.isChecked():
                selected.append(self.rows[i])
        print("DEBUG selected rows:", [r.get('lambda_micron') or r.get('wavelength') for r in selected])
        return selected
    
    @staticmethod
    def get_row_wavelength(row):
        return row.get("lambda_micron") or row.get("wavelength")

    def select_all_rows(self):
        for i in range(self.table.rowCount()):
            widget = self.table.cellWidget(i, 0)
            if widget is None:
                continue
            cb = widget.layout().itemAt(0).widget()
            cb.setChecked(True)

    def unselect_all_rows(self):
        for i in range(self.table.rowCount()):
            widget = self.table.cellWidget(i, 0)
            if widget is None:
                continue
            cb = widget.layout().itemAt(0).widget()
            cb.setChecked(False)

    def select_detection_only(self):
        """Check only the lines being detected (HITRAN or custom-imported), uncheck contaminants."""
        for i in range(self.table.rowCount()):
            widget = self.table.cellWidget(i, 0)
            if widget is None:
                continue
            cb = widget.layout().itemAt(0).widget()
            cb.setChecked(self.rows[i].get('source') != 'Contaminant')

    def select_contaminating_only(self):
        """Check only contaminating lines, uncheck the detection lines."""
        for i in range(self.table.rowCount()):
            widget = self.table.cellWidget(i, 0)
            if widget is None:
                continue
            cb = widget.layout().itemAt(0).widget()
            cb.setChecked(self.rows[i].get('source') == 'Contaminant')
