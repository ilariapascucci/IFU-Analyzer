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

        # Build a set of wavelengths for quick membership checks
        # Use the same logic as get_row_wavelength to extract wavelengths
        self.plotted_waves = set()
        for r in self.plotted_rows:
            try:
                wave = float(r.get('lambda_micron') or r.get('wavelength') or 0.0)
                if wave > 0:
                    self.plotted_waves.add(wave)
            except Exception:
                pass
        
        print(f"[DEBUG] SpeciesDialog: {len(self.plotted_rows)} plotted rows, {len(self.plotted_waves)} wavelengths")

        layout = QVBoxLayout(self)

        # Title with counts
        lst_count = sum(1 for r in rows if r.get('source') != 'HITRAN')
        hitran_count = sum(1 for r in rows if r.get('source') == 'HITRAN')
        if hitran_count > 0:
            title_text = f"{len(rows)} matching lines found ({lst_count} LST, {hitran_count} HITRAN)"
        else:
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
            default_checked = None  # Will check against plotted_waves
        else:
            default_checked = True  # First time: select all
        self.populate_table(rows, default_checked=default_checked)

        # Buttons for selecting/unselecting all rows and filtering by source
        btn_layout2 = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_rows)
        self.unselect_all_btn = QPushButton("Unselect All")
        self.unselect_all_btn.clicked.connect(self.unselect_all_rows)
        
        self.hitran_only_btn = QPushButton("HITRAN Only")
        self.hitran_only_btn.clicked.connect(self.select_hitran_only)
        self.lst_only_btn = QPushButton("LST Only")
        self.lst_only_btn.clicked.connect(self.select_lst_only)

        btn_layout2.addWidget(self.select_all_btn)
        btn_layout2.addWidget(self.unselect_all_btn)
        btn_layout2.addWidget(self.hitran_only_btn)
        btn_layout2.addWidget(self.lst_only_btn)
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

    def populate_table(self, rows, default_checked=True):
        """
        Populate the table. Each row gets a checkbox.
        default_checked:
          - True => check all rows
          - False => uncheck all rows
          - None => check only those present in self.plotted_waves
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
                wave = float(row.get('lambda_micron') or row.get('wavelength') or 0.0)
                # Check with small tolerance for floating point comparison
                is_plotted = any(abs(wave - pw) < 1e-9 for pw in self.plotted_waves)
                checkbox.setChecked(is_plotted)

            layout = QHBoxLayout()
            layout.addWidget(checkbox)
            layout.setAlignment(Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            checkbox_widget.setLayout(layout)
            
            # Apply background color for HITRAN rows
            if row.get('source') == 'HITRAN':
                checkbox_widget.setStyleSheet("background-color: rgba(255, 200, 200, 80);")
            
            self.table.setCellWidget(i, 0, checkbox_widget)

            # Check if this is a HITRAN line for background coloring
            is_hitran = row.get('source') == 'HITRAN'
            hitran_color = QColor(255, 200, 200, 80)  # Faint red background
            
            # Wavelength
            lam = row.get('lambda_micron') if 'lambda_micron' in row else row.get('wavelength')
            item = QTableWidgetItem(f"{float(lam):.6f}")
            if is_hitran:
                item.setBackground(hitran_color)
            self.table.setItem(i, 1, item)
            
            # Species
            item = QTableWidgetItem(row.get("species", ""))
            if is_hitran:
                item.setBackground(hitran_color)
            self.table.setItem(i, 2, item)
            
            # Transition
            item = QTableWidgetItem(row.get("transition", ""))
            if is_hitran:
                item.setBackground(hitran_color)
            self.table.setItem(i, 3, item)
            
            # A-coeff
            a_coeff = row.get("a_coeff", "")
            if a_coeff != "" and a_coeff is not None:
                try:
                    item = QTableWidgetItem(f"{float(a_coeff):.2e}")
                except:
                    item = QTableWidgetItem(str(a_coeff))
            else:
                item = QTableWidgetItem("")
            if is_hitran:
                item.setBackground(hitran_color)
            self.table.setItem(i, 4, item)
            
            # E_upper
            e_upper = row.get("e_upper", "")
            if e_upper != "" and e_upper is not None:
                try:
                    item = QTableWidgetItem(f"{float(e_upper):.2f}")
                except:
                    item = QTableWidgetItem(str(e_upper))
            else:
                item = QTableWidgetItem("")
            if is_hitran:
                item.setBackground(hitran_color)
            self.table.setItem(i, 5, item)

        # Narrow first column
        self.table.setColumnWidth(0, 30)

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

    def select_hitran_only(self):
        """Uncheck all LST rows, keep HITRAN rows checked."""
        for i in range(self.table.rowCount()):
            widget = self.table.cellWidget(i, 0)
            if widget is None:
                continue
            cb = widget.layout().itemAt(0).widget()
            # Check if this row is HITRAN
            is_hitran = self.rows[i].get('source') == 'HITRAN'
            cb.setChecked(is_hitran)

    def select_lst_only(self):
        """Uncheck all HITRAN rows, keep LST rows checked."""
        for i in range(self.table.rowCount()):
            widget = self.table.cellWidget(i, 0)
            if widget is None:
                continue
            cb = widget.layout().itemAt(0).widget()
            # Check if this row is NOT HITRAN (i.e., LST)
            is_lst = self.rows[i].get('source') != 'HITRAN'
            cb.setChecked(is_lst)