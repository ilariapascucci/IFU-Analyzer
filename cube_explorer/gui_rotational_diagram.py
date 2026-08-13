"""
Rotational Diagram Analysis GUI for Molecular data (H2, CO) - SIMPLIFIED
Single unified interface for selecting molecules, lines, and entering flux data
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTableWidget, QTableWidgetItem, QSpinBox, 
    QDoubleSpinBox, QCheckBox, QMessageBox, QProgressDialog,
    QFrame, QComboBox, QDialog, QTextEdit, QFileDialog, QLineEdit, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QBrush
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .line_catalog import fetch_hitran_data
from .rotational_diagram_fitter import estimate_noise_local, gaussian_flux_upper_limit, get_spectral_resolution, get_gu


# Constants
AU = 1.496e13  # cm
PC = 3.086e18  # cm

# Isotopologue ranges per molecule (HITRAN numbering)
MOLECULE_ISOTOPOLOGUES = {
    'H2': {'labels': ['H₂', 'HD', 'D₂'], 'range': (1, 3)},
    'CO': {'labels': ['¹²C¹⁶O', '¹³C¹⁶O', '¹²C¹⁸O', '¹²C¹⁷O', '¹³C¹⁸O'], 'range': (1, 5)},
}

# Regex patterns for extracting vibrational transitions per molecule
MOLECULE_TRANSITION_PATTERNS = {
    'H2': r'v=\d+',  # Matches "v=0", "v=1", etc.
    'CO': r'v=(\d+)->(\d+)',  # Matches "v=1->0", "v=2->1", etc.
}


class HitranFetchThread(QThread):
    """Background thread for fetching HITRAN data"""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, wl_min, wl_max, molecule, isotopologue, v_upper=None, v_lower=None):
        super().__init__()
        self.wl_min = wl_min
        self.wl_max = wl_max
        self.molecule = molecule
        self.isotopologue = isotopologue
        self.v_upper = v_upper
        self.v_lower = v_lower
    
    def run(self):
        try:
            hitran_data, _ = fetch_hitran_data(
                self.wl_min, self.wl_max,
                molecule=self.molecule,
                isotopologue=self.isotopologue,
                v_upper=self.v_upper,
                v_lower=self.v_lower
            )
            self.finished.emit(hitran_data)
        except Exception as e:
            self.error.emit(str(e))


class RotationalDiagramWindow(QMainWindow):
    """Main window for rotational diagram analysis"""
    
    def __init__(self, parent_gui=None):  # parent_gui kept for backward compatibility, not used
        super().__init__()
        self.current_molecule = 'H2'  # Default molecule
        self.setWindowTitle(f"{self.current_molecule} Rotational Diagram Analyzer")
        self.setGeometry(100, 100, 1500, 800)
        
        self.hitran_lines = []
        self.fetch_thread = None
        self.updating_error_columns = False  # Flag to prevent recursive updates
        self.updating_flux_flag = False  # Same, for the Flux column's red/clear flag
        self.updating_manual_stats = False  # Same, for manually-entered A_ul/E_u/g_u
        self.updating_flux_error_flag = False  # Same, for the Flux Error column's red/clear flag
        
        # Track workflow: None=unset, 'csv'=import, 'hitran'=fetch
        self.workflow_mode = None
        
        # Main widget
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # HITRAN parameters section
        main_layout.addWidget(self.create_hitran_section())
        
        # Lines table
        main_layout.addWidget(QLabel("Select lines and enter flux data:"))
        
        # Error input mode toggle
        toggle_layout = QHBoxLayout()
        toggle_layout.addWidget(QLabel("Error input:"))
        self.error_mode = QComboBox()
        self.error_mode.addItems(["Values (Column 8)", "Percentages (Column 9)"])
        self.error_mode.currentTextChanged.connect(self.on_error_mode_changed)
        toggle_layout.addWidget(self.error_mode)
        
        # Select All / Deselect All buttons
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_lines)
        self.select_all_btn.setMaximumWidth(100)
        toggle_layout.addWidget(self.select_all_btn)
        
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all_lines)
        self.deselect_all_btn.setMaximumWidth(100)
        toggle_layout.addWidget(self.deselect_all_btn)
        
        toggle_layout.addStretch()
        main_layout.addLayout(toggle_layout)
        
        self.lines_table = QTableWidget()
        self.lines_table.setColumnCount(12)
        self.lines_table.setHorizontalHeaderLabels(
            ["✓", "λ (μm)", "Transition", "A_ul (s⁻¹)", "E_u (K)", "g_u", "Flux", "Flux Error",
             "Error %", "Upper Limit?", "Calculated?", "Source"]
        )
        # Every column sizes itself to its widest current cell and stays there --
        # no manual dragging, no guessed pixel widths to keep in sync as columns
        # get added/renamed. Re-fits automatically whenever cell content changes
        # (e.g. populate_lines_table, or Source getting filled in), not just once.
        self.lines_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # A little breathing room around cell/header text -- Qt includes QSS
        # padding in the size hint ResizeToContents measures, so this stays
        # correct on every re-fit rather than needing a one-off pixel bump.
        self.lines_table.setStyleSheet(
            "QTableWidget::item { padding: 2px 6px; } "
            "QHeaderView::section { padding: 2px 6px; }"
        )
        main_layout.addWidget(self.lines_table)
        
        # Connect table changes for auto-calculation
        self.lines_table.itemChanged.connect(self.on_table_item_changed)
        
        # Connect upper limit checkbox changes
        self.lines_table.cellClicked.connect(self.on_cell_clicked)
        
        # Bottom: Physical parameters + buttons
        bottom_layout = QHBoxLayout()
        
        bottom_layout.addWidget(QLabel("R (au):"))
        self.radius_unknown_check = QCheckBox("Unknown R")
        self.radius_unknown_check.setToolTip(
            "Leave R unspecified: Nu is computed as a total molecule count instead of "
            "a per-cm^2 column density (e.g. for comparing against PDRTPy)."
        )
        self.radius_unknown_check.stateChanged.connect(self.on_radius_unknown_toggled)
        bottom_layout.addWidget(self.radius_unknown_check)

        self.radius_input = QLineEdit()
        self.radius_input.setText("17")
        self.radius_input.setMaximumWidth(100)
        bottom_layout.addWidget(self.radius_input)

        bottom_layout.addWidget(QLabel("Distance (pc):"))
        self.distance_input = QLineEdit()
        self.distance_input.setText("71.5")
        self.distance_input.setMaximumWidth(100)
        bottom_layout.addWidget(self.distance_input)
        
        bottom_layout.addWidget(QLabel("Fit:"))
        self.fit_mode = QComboBox()
        self.fit_mode.addItems(["1T", "2T"])
        bottom_layout.addWidget(self.fit_mode)

        self.fit_opr_check = QCheckBox("Fit OPR")
        self.fit_opr_check.setToolTip(
            "Grid-search over OPR to minimize chi-squared, instead of "
            "assuming the fixed equilibrium value OPR=3. Only affects H2 S-branch lines."
        )
        self.fit_opr_check.stateChanged.connect(self.on_fit_opr_toggled)
        bottom_layout.addWidget(self.fit_opr_check)

        self.opr_advanced_btn = QPushButton("▶ Advanced")
        self.opr_advanced_btn.setCheckable(True)
        self.opr_advanced_btn.setMaximumWidth(90)
        self.opr_advanced_btn.setEnabled(False)
        self.opr_advanced_btn.clicked.connect(self.on_opr_advanced_toggled)
        bottom_layout.addWidget(self.opr_advanced_btn)

        # Advanced OPR grid controls - hidden until "Advanced" is clicked, expands to the
        # right in this same row. Only meaningful once "Fit OPR" is checked. Defaults
        # (0.1, 3.0, 400) match the notebook's own reference grid, not the fitter's
        # built-in default (0.1, 5.0, 200).
        self.opr_advanced_frame = QFrame()
        opr_advanced_layout = QHBoxLayout(self.opr_advanced_frame)
        opr_advanced_layout.setContentsMargins(10, 0, 0, 0)

        opr_advanced_layout.addWidget(QLabel("OPR min:"))
        self.opr_min_default_check = QCheckBox("Default")
        self.opr_min_default_check.setChecked(True)
        self.opr_min_default_check.stateChanged.connect(self.on_opr_min_default_toggled)
        opr_advanced_layout.addWidget(self.opr_min_default_check)
        self.opr_min_input = QLineEdit()
        self.opr_min_input.setText("0.1")
        self.opr_min_input.setMaximumWidth(60)
        self.opr_min_input.setEnabled(False)
        opr_advanced_layout.addWidget(self.opr_min_input)

        opr_advanced_layout.addWidget(QLabel("max:"))
        self.opr_max_default_check = QCheckBox("Default")
        self.opr_max_default_check.setChecked(True)
        self.opr_max_default_check.stateChanged.connect(self.on_opr_max_default_toggled)
        opr_advanced_layout.addWidget(self.opr_max_default_check)
        self.opr_max_input = QLineEdit()
        self.opr_max_input.setText("3.0")
        self.opr_max_input.setMaximumWidth(60)
        self.opr_max_input.setEnabled(False)
        opr_advanced_layout.addWidget(self.opr_max_input)

        opr_advanced_layout.addWidget(QLabel("steps:"))
        self.opr_steps_default_check = QCheckBox("Default")
        self.opr_steps_default_check.setChecked(True)
        self.opr_steps_default_check.stateChanged.connect(self.on_opr_steps_default_toggled)
        opr_advanced_layout.addWidget(self.opr_steps_default_check)
        self.opr_steps_input = QLineEdit()
        self.opr_steps_input.setText("400")
        self.opr_steps_input.setMaximumWidth(60)
        self.opr_steps_input.setEnabled(False)
        opr_advanced_layout.addWidget(self.opr_steps_input)

        self.opr_advanced_frame.setVisible(False)
        bottom_layout.addWidget(self.opr_advanced_frame)

        bottom_layout.addStretch()

        self.run_fit_btn = QPushButton("Run Fit")
        self.run_fit_btn.clicked.connect(self.run_rotational_fit)
        self.run_fit_btn.setEnabled(False)
        bottom_layout.addWidget(self.run_fit_btn)

        main_layout.addLayout(bottom_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def create_hitran_section(self):
        """Create HITRAN parameter input section"""
        frame = QFrame()
        layout = QVBoxLayout()
        
        # Molecule selector
        mol_layout = QHBoxLayout()
        mol_layout.addWidget(QLabel("Molecule:"))
        self.mol_combo = QComboBox()
        self.mol_combo.addItems(list(MOLECULE_ISOTOPOLOGUES.keys()))
        self.mol_combo.setCurrentText(self.current_molecule)
        self.mol_combo.currentTextChanged.connect(self.on_molecule_changed)
        mol_layout.addWidget(self.mol_combo)
        mol_layout.addStretch()
        layout.addLayout(mol_layout)
        
        # Input parameters
        input_layout = QHBoxLayout()
        
        input_layout.addWidget(QLabel("λ_min (μm):"))
        self.wl_min_input = QDoubleSpinBox()
        self.wl_min_input.setRange(0.1, 1000)
        self.wl_min_input.setValue(5.0)
        self.wl_min_input.setDecimals(6)
        input_layout.addWidget(self.wl_min_input)
        
        input_layout.addWidget(QLabel("λ_max (μm):"))
        self.wl_max_input = QDoubleSpinBox()
        self.wl_max_input.setRange(0.1, 1000)
        self.wl_max_input.setValue(20.0)
        self.wl_max_input.setDecimals(6)
        input_layout.addWidget(self.wl_max_input)
        
        input_layout.addWidget(QLabel("Isotopologue:"))
        self.isotopologue_spin = QSpinBox()
        self.update_isotopologue_range()
        input_layout.addWidget(self.isotopologue_spin)
        
        input_layout.addWidget(QLabel("v_upper:"))
        self.v_upper_spin = QSpinBox()
        self.v_upper_spin.setRange(0, 10)
        self.v_upper_spin.setValue(0)
        input_layout.addWidget(self.v_upper_spin)
        
        input_layout.addWidget(QLabel("v_lower:"))
        self.v_lower_spin = QSpinBox()
        self.v_lower_spin.setRange(0, 10)
        self.v_lower_spin.setValue(0)
        input_layout.addWidget(self.v_lower_spin)
        
        self.include_all_trans_check = QCheckBox("All transitions")
        self.include_all_trans_check.setChecked(True)
        self.include_all_trans_check.stateChanged.connect(self.on_include_all_transitions_toggled)
        input_layout.addWidget(self.include_all_trans_check)
        
        self.v_upper_spin.setEnabled(False)
        self.v_lower_spin.setEnabled(False)
        
        input_layout.addStretch()
        layout.addLayout(input_layout)
        
        # Fetch and Import buttons
        button_layout = QHBoxLayout()
        
        self.fetch_btn = QPushButton("Fetch from HITRAN")
        self.fetch_btn.clicked.connect(self.fetch_hitran_lines)
        button_layout.addWidget(self.fetch_btn)
        
        self.import_csv_btn = QPushButton("Import from CSV")
        self.import_csv_btn.clicked.connect(self.import_csv_file)
        button_layout.addWidget(self.import_csv_btn)
        
        self.calculate_fluxes_btn = QPushButton("Calculate Fluxes")
        self.calculate_fluxes_btn.clicked.connect(self.open_batch_flux_dialog)
        self.calculate_fluxes_btn.setVisible(False)  # Hidden until lines are loaded (HITRAN fetch or CSV import)
        button_layout.addWidget(self.calculate_fluxes_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        frame.setLayout(layout)
        return frame
    
    def on_molecule_changed(self, molecule):
        """Update UI when molecule selection changes"""
        self.current_molecule = molecule
        self.setWindowTitle(f"{molecule} Rotational Diagram Analyzer")
        self.update_isotopologue_range()
    
    def update_isotopologue_range(self):
        """Update isotopologue spinner range based on selected molecule"""
        mol_data = MOLECULE_ISOTOPOLOGUES.get(self.current_molecule)
        if mol_data:
            min_iso, max_iso = mol_data['range']
            self.isotopologue_spin.setRange(min_iso, max_iso)
            self.isotopologue_spin.setValue(min_iso)
    
    def on_include_all_transitions_toggled(self):
        checked = self.include_all_trans_check.isChecked()
        self.v_upper_spin.setEnabled(not checked)
        self.v_lower_spin.setEnabled(not checked)

    def on_radius_unknown_toggled(self):
        """Disable the R field when R is marked unknown (R=None is passed to the fitter)"""
        self.radius_input.setEnabled(not self.radius_unknown_check.isChecked())

    def on_fit_opr_toggled(self):
        """Enable the Advanced button only when Fit OPR is checked; collapse it otherwise"""
        checked = self.fit_opr_check.isChecked()
        self.opr_advanced_btn.setEnabled(checked)
        if not checked and self.opr_advanced_btn.isChecked():
            self.opr_advanced_btn.setChecked(False)
            self.on_opr_advanced_toggled()

    def on_opr_advanced_toggled(self):
        """Show/hide the OPR grid range/resolution controls"""
        expanded = self.opr_advanced_btn.isChecked()
        self.opr_advanced_frame.setVisible(expanded)
        self.opr_advanced_btn.setText(("▼" if expanded else "▶") + " Advanced")

    def on_opr_min_default_toggled(self):
        """Lock OPR min to its default value while 'Default' is checked"""
        checked = self.opr_min_default_check.isChecked()
        self.opr_min_input.setEnabled(not checked)
        if checked:
            self.opr_min_input.setText("0.1")

    def on_opr_max_default_toggled(self):
        """Lock OPR max to its default value while 'Default' is checked"""
        checked = self.opr_max_default_check.isChecked()
        self.opr_max_input.setEnabled(not checked)
        if checked:
            self.opr_max_input.setText("3.0")

    def on_opr_steps_default_toggled(self):
        """Lock OPR steps to its default value while 'Default' is checked"""
        checked = self.opr_steps_default_check.isChecked()
        self.opr_steps_input.setEnabled(not checked)
        if checked:
            self.opr_steps_input.setText("400")
    
    def on_error_mode_changed(self, mode):
        """Update table edit/read-only based on error input mode"""
        self.update_error_columns()
    
    def update_error_columns(self):
        """Update which columns are editable based on error mode"""
        is_value_mode = "Values" in self.error_mode.currentText()
        
        for row in range(self.lines_table.rowCount()):
            flux_error_item = self.lines_table.item(row, 7)
            error_pct_item = self.lines_table.item(row, 8)
            
            if flux_error_item and error_pct_item:
                if is_value_mode:
                    # Column 8 (Flux Error) is editable, Column 9 (Error %) is read-only
                    flux_error_item.setFlags(flux_error_item.flags() | Qt.ItemIsEditable)
                    error_pct_item.setFlags(error_pct_item.flags() & ~Qt.ItemIsEditable)
                else:
                    # Column 9 (Error %) is editable, Column 8 (Flux Error) is read-only
                    flux_error_item.setFlags(flux_error_item.flags() & ~Qt.ItemIsEditable)
                    error_pct_item.setFlags(error_pct_item.flags() | Qt.ItemIsEditable)
    
    def on_table_item_changed(self, item):
        """Auto-calculate flux error or percentage; sync manually-entered A_ul/E_u/g_u"""
        if self.lines_table.rowCount() == 0:
            return

        row = item.row()
        col = item.column()

        # A_ul / E_u / g_u are only editable on rows with no HITRAN match -- keep
        # self.hitran_lines and the row's fit checkbox in sync with what's typed.
        if col in (3, 4, 5):
            self.sync_manual_line_stats(row)
            return

        if col == 6:
            self.sync_flux_flag(row)
            self.sync_calculated_flag(row)

        if col == 7:
            self.sync_flux_error_flag(row)

        if self.updating_error_columns:
            return

        self.updating_error_columns = True

        try:
            # Columns 7 and 8 are flux error and error %
            if col not in (7, 8):
                return
            
            flux_error_item = self.lines_table.item(row, 7)
            error_pct_item = self.lines_table.item(row, 8)
            
            if not flux_error_item or not error_pct_item:
                return
            
            # Get flux value
            flux_item = self.lines_table.item(row, 6)
            if not flux_item:
                return
            
            try:
                flux = float(flux_item.text())
            except ValueError:
                return
            
            # Get current text from the item that changed
            current_text = item.text().strip() if item.text() else ""
            
            if col == 7:  # Flux Error column changed
                if current_text == "":
                    # User cleared flux error - also clear error %
                    error_pct_item.setText("")
                else:
                    # Calculate percentage
                    try:
                        flux_error = float(current_text)
                        pct = (flux_error / flux) * 100 if flux != 0 else 0
                        error_pct_item.setText(f"{pct:.1f}")
                    except ValueError:
                        pass
            
            elif col == 8:  # Error % column changed
                if current_text == "":
                    # User cleared error % - also clear flux error
                    flux_error_item.setText("")
                else:
                    # Calculate flux error
                    try:
                        pct = float(current_text)
                        flux_error = flux * (pct / 100.0)
                        flux_error_item.setText(f"{flux_error:.4e}")
                    except ValueError:
                        pass
        finally:
            self.updating_error_columns = False

    def sync_manual_line_stats(self, row):
        """For a row with no HITRAN match, write manually-typed A_ul/E_u/g_u back into
        self.hitran_lines and enable that row's fit checkbox only once all three are
        present and physically valid (A_ul > 0, g_u > 0). Never assumes a value for
        a field the user hasn't actually filled in.
        """
        if row >= len(self.hitran_lines):
            return

        line = self.hitran_lines[row]
        if line.get('hitran_matched', True):
            return  # A_ul/E_u/g_u are read-only HITRAN values here, nothing to sync

        if self.updating_manual_stats:
            return  # setBackground() below re-fires itemChanged -- don't recurse
        self.updating_manual_stats = True

        try:
            def _parse(col):
                cell = self.lines_table.item(row, col)
                text = cell.text().strip() if cell else ""
                if not text:
                    return None
                try:
                    return float(text)
                except ValueError:
                    return None

            a_val, e_val, g_val = _parse(3), _parse(4), _parse(5)
            complete = a_val is not None and e_val is not None and g_val is not None and a_val > 0 and g_val > 0

            checkbox = self.lines_table.cellWidget(row, 0)
            cells = [self.lines_table.item(row, c) for c in (3, 4, 5)]

            if complete:
                line['a_coeff'] = a_val
                line['e_upper'] = e_val
                line['Gp'] = g_val
                for cell in cells:
                    if cell:
                        cell.setBackground(QBrush())
                        cell.setToolTip("")
                if checkbox:
                    checkbox.setEnabled(True)
                    checkbox.setToolTip("")
            else:
                line.pop('a_coeff', None)
                line.pop('e_upper', None)
                line.pop('Gp', None)
                tip = "No HITRAN match -- enter A_ul, E_u, and g_u manually to include this line."
                for cell in cells:
                    if cell:
                        cell.setBackground(QColor(255, 214, 214))
                        cell.setToolTip(tip)
                if checkbox:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(False)
                    checkbox.blockSignals(False)
                    checkbox.setEnabled(False)
                    checkbox.setToolTip(tip)

            self.on_line_selected()
        finally:
            self.updating_manual_stats = False

    def sync_flux_flag(self, row):
        """Flag the Flux cell red while empty (flux is required for the fit); clear
        the flag as soon as a value is present, whether typed by hand or written by
        Calculate Fluxes (setText() there fires the same itemChanged signal)."""
        if self.updating_flux_flag:
            return
        self.updating_flux_flag = True
        try:
            item = self.lines_table.item(row, 6)
            if not item:
                return
            if item.text().strip():
                item.setBackground(QBrush())
                item.setToolTip("")
            else:
                item.setBackground(QColor(255, 214, 214))
                item.setToolTip("Flux is required for the fit -- enter it manually or use Calculate Fluxes.")
        finally:
            self.updating_flux_flag = False

    def sync_calculated_flag(self, row):
        """Editing the Flux cell means its value is no longer (only) what Calculate
        Fluxes computed -- reset Calculated?/Source to unset by default. This runs
        on every Flux edit, including the setText() calculate_fluxes_batch itself
        makes; that function then explicitly re-checks the box and re-fills Source
        right after, so a fresh auto-calculation still ends up marked calculated --
        only an actual manual edit is left unmarked."""
        calc_checkbox = self.lines_table.cellWidget(row, 10)
        if calc_checkbox:
            calc_checkbox.setChecked(False)
        source_item = self.lines_table.item(row, 11)
        if source_item:
            source_item.setText("")

    def sync_flux_error_flag(self, row):
        """Flag the Flux Error cell red when blank and the line isn't marked as an
        Upper Limit: rotational_diagram_fitter._build_dataset excludes any such
        detection from the fit entirely rather than assuming an error (no unweighted
        fallback), so this surfaces that before Run Fit instead of via the post-fit
        "Lines Excluded" dialog. Marking a line as Upper Limit clears the flag, since
        upper limits use a fixed 0 error by design and don't need one."""
        if self.updating_flux_error_flag:
            return
        self.updating_flux_error_flag = True
        try:
            item = self.lines_table.item(row, 7)
            if not item:
                return
            ul_checkbox = self.lines_table.cellWidget(row, 9)
            is_upper_limit = bool(ul_checkbox and ul_checkbox.isChecked())

            if item.text().strip() or is_upper_limit:
                item.setBackground(QBrush())
                item.setToolTip("")
            else:
                item.setBackground(QColor(255, 214, 214))
                item.setToolTip(
                    "No flux error entered -- this detection will be excluded from "
                    "the fit unless you enter one or mark it as an Upper Limit."
                )
        finally:
            self.updating_flux_error_flag = False

    def fetch_hitran_lines(self):
        """Fetch molecular lines from HITRAN"""
        wl_min = self.wl_min_input.value()
        wl_max = self.wl_max_input.value()
        isotopologue = self.isotopologue_spin.value()
        
        if wl_min >= wl_max:
            QMessageBox.warning(self, "Error", "Min wavelength must be less than max")
            return
        
        if self.include_all_trans_check.isChecked():
            v_upper = None
            v_lower = None
        else:
            v_upper = self.v_upper_spin.value()
            v_lower = self.v_lower_spin.value()
        
        self.fetch_btn.setEnabled(False)
        progress = QProgressDialog(f"Fetching {self.current_molecule} lines from HITRAN...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        self.fetch_thread = HitranFetchThread(wl_min, wl_max, self.current_molecule, isotopologue, v_upper, v_lower)
        self.fetch_thread.finished.connect(lambda data: self.on_hitran_fetched(data, progress))
        self.fetch_thread.error.connect(lambda err: self.on_hitran_error(err, progress))
        self.fetch_thread.start()
    
    def on_hitran_fetched(self, data, progress):
        progress.close()
        self.fetch_btn.setEnabled(True)
        
        if not data:
            QMessageBox.warning(self, "Error", f"No {self.current_molecule} lines found in this wavelength range")
            return
        
        # Set workflow mode to HITRAN
        self.workflow_mode = 'hitran'
        self.import_csv_btn.setEnabled(False)  # Disable CSV import
        self.calculate_fluxes_btn.setVisible(True)  # Show Calculate Fluxes button
        
        self.hitran_lines = data
        self.populate_lines_table()

    def on_hitran_error(self, error_msg, progress):
        progress.close()
        self.fetch_btn.setEnabled(True)
        QMessageBox.critical(self, "HITRAN Fetch Error", error_msg)
    
    def import_csv_file(self):
        """Import species, wavelength, and flux data from CSV, enriching with HITRAN data for A and E_u"""
        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import CSV File",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if not csv_path:
            return
        
        try:
            import csv
            
            # Read CSV file
            rows = []
            csv_data = []
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    QMessageBox.warning(self, "Error", "CSV file is empty")
                    return
                
                # Check for required columns (case-insensitive). Flux is optional —
                # a CSV with just species + wavelength is enough; missing fluxes get
                # filled in later via "Calculate Fluxes", same as the HITRAN workflow.
                available_cols = {col.strip().lower(): col.strip() for col in reader.fieldnames if col}
                required_cols = {'species', 'wavelength'}

                if not required_cols.issubset(available_cols.keys()):
                    missing = required_cols - available_cols.keys()
                    QMessageBox.warning(
                        self,
                        "Missing Columns",
                        f"CSV must have 'species' and 'wavelength' columns.\nMissing: {', '.join(missing)}"
                    )
                    return

                has_flux = 'flux' in available_cols
                flux_col = available_cols.get('flux')

                # Check for optional error column
                has_error = False
                error_col = None
                for col_lower, col_actual in available_cols.items():
                    if 'error' in col_lower and col_lower != 'wavelength':
                        has_error = True
                        error_col = col_actual
                        break

                # Map column names (case-insensitive)
                species_col = available_cols['species']
                wavelength_col = available_cols['wavelength']

                for row_dict in reader:
                    try:
                        # csv.DictReader fills columns missing from a short/ragged row
                        # with None (not '') -- guard every .get() the same way so a
                        # row that only has species+wavelength doesn't crash on
                        # None.strip().
                        species = (row_dict.get(species_col) or '').strip().upper()
                        wavelength = float((row_dict.get(wavelength_col) or '').strip())

                        row_info = {
                            'species': species,
                            'wavelength': wavelength,
                        }

                        # Flux is best-effort: a blank/missing value just leaves the
                        # row without one, to be filled in later via Calculate Fluxes
                        if has_flux and flux_col:
                            flux_str = (row_dict.get(flux_col) or '').strip()
                            if flux_str:
                                try:
                                    row_info['flux'] = float(flux_str)
                                except ValueError:
                                    pass

                        # Add error if available
                        if has_error and error_col:
                            error_str = (row_dict.get(error_col) or '').strip()
                            if error_str:
                                try:
                                    row_info['flux_error'] = float(error_str)
                                except ValueError:
                                    pass

                        csv_data.append(row_info)
                    except (ValueError, TypeError):
                        # Skip rows with invalid species/wavelength
                        continue
            
            if not csv_data:
                QMessageBox.warning(self, "Error", "No valid rows found in CSV")
                return
            
            # Now enrich CSV data with HITRAN values for A and E_u
            # Group CSV data by species to batch-fetch HITRAN data
            species_list = list(set(row['species'] for row in csv_data))
            
            # Get wavelength ranges to fetch
            all_wavelengths = [row['wavelength'] for row in csv_data]
            wl_min = min(all_wavelengths) - 0.1  # Buffer for rounding
            wl_max = max(all_wavelengths) + 0.1
            
            hitran_cache = {}  # Cache HITRAN data by species
            
            progress = QProgressDialog(
                "Enriching CSV data with HITRAN information...",
                None, 0, len(species_list), self
            )
            progress.setWindowModality(Qt.WindowModal)
            
            # Fetch HITRAN data for each species
            for idx, species in enumerate(species_list):
                progress.setValue(idx)
                try:
                    hitran_data, _ = fetch_hitran_data(
                        wl_min, wl_max,
                        molecule=species,
                        isotopologue=1
                    )
                    hitran_cache[species] = hitran_data
                except Exception as e:
                    print(f"Warning: Could not fetch HITRAN data for {species}: {e}")
                    hitran_cache[species] = []
            
            progress.close()
            
            # Match CSV rows to HITRAN data
            for csv_row in csv_data:
                species = csv_row['species']
                wavelength = csv_row['wavelength']
                hitran_lines = hitran_cache.get(species, [])
                
                # Find matching HITRAN line (within 0.0001 μm = 4 decimal places)
                best_match = None
                best_diff = float('inf')
                
                for hitran_line in hitran_lines:
                    diff = abs(hitran_line['wavelength'] - wavelength)
                    if diff < 0.0001 and diff < best_diff:  # 4 decimal place tolerance
                        best_match = hitran_line
                        best_diff = diff
                
                # Create data row with original CSV wavelength but HITRAN A and E_u
                if best_match:
                    row_data = {
                        'wavelength': wavelength,  # Keep original CSV value
                        'hitran_matched': True,
                        'a_coeff': best_match.get('a_coeff', 0.0),
                        'e_upper': best_match.get('e_upper', 0.0),
                        'Gp': best_match.get('Gp', 1.0),
                        'j_lower': best_match.get('j_lower'),
                        'branch': best_match.get('branch'),
                        'transition': best_match.get('transition', 'User Data'),
                        'species': species
                    }
                else:
                    # No matching HITRAN line -- A_ul/E_u/g_u are hard physics inputs
                    # (Nu = 4*pi*F*lam/(h*c*A_ul), E_u is the plot's x-axis), so we never
                    # assume a value for them. The row is flagged in the table and its
                    # checkbox stays disabled until the user types in all three by hand.
                    row_data = {
                        'wavelength': wavelength,
                        'hitran_matched': False,
                        'transition': 'NA',
                        'species': species
                    }
                    print(f"Warning: No HITRAN match for {species} at {wavelength:.6f} um "
                          f"-- A_ul/E_u/g_u left blank; this line is excluded from fitting "
                          f"until they're entered manually.")

                # Flux is optional in the CSV — only set it if the row actually had one
                if 'flux' in csv_row:
                    row_data['flux'] = csv_row['flux']

                # Add error if available
                if 'flux_error' in csv_row:
                    row_data['flux_error'] = csv_row['flux_error']

                rows.append(row_data)

            self.hitran_lines = rows
            self.populate_lines_table()

            # Pre-fill errors if they were provided in CSV
            self.populate_csv_errors()

            # Set workflow mode to CSV
            self.workflow_mode = 'csv'
            self.fetch_btn.setEnabled(False)  # Disable HITRAN fetch
            self.calculate_fluxes_btn.setVisible(True)  # Show Calculate Fluxes button (fills in any missing fluxes)

            # Count successful matches
            matched = sum(1 for r in rows if r.get('hitran_matched', True))
            with_flux = sum(1 for r in rows if 'flux' in r)
            flux_note = (
                "All rows already have a flux.\n" if with_flux == len(rows)
                else f"{len(rows) - with_flux}/{len(rows)} rows have no flux yet — "
                     f"select them and use Calculate Fluxes to fill them in.\n"
            )
            unmatched_note = (
                "" if matched == len(rows)
                else f"{len(rows) - matched} row(s) have no HITRAN match and are "
                     f"disabled (shown in red) until you enter A_ul, E_u, and g_u "
                     f"manually.\n"
            )
            QMessageBox.information(
                self,
                "Import Successful",
                f"Imported {len(rows)} rows from CSV\n"
                f"Matched {matched}/{len(rows)} lines to HITRAN data\n"
                f"{unmatched_note}"
                f"{flux_note}"
            )
        
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import CSV:\n{str(e)}")
    
    def populate_csv_errors(self):
        """Populate error column if flux_error was in CSV"""
        for row in range(self.lines_table.rowCount()):
            if row < len(self.hitran_lines):
                line_data = self.hitran_lines[row]
                if 'flux_error' in line_data:
                    # Determine which column to fill based on error mode
                    is_value_mode = "Values" in self.error_mode.currentText()
                    
                    if is_value_mode:
                        # Column 7: Flux Error (absolute value)
                        error_item = self.lines_table.item(row, 7)
                        if error_item:
                            error_item.setText(f"{line_data['flux_error']:.4e}")
                        
                        # Calculate percentage for column 8
                        flux_text = self.lines_table.item(row, 6).text()
                        if flux_text:
                            try:
                                flux_val = float(flux_text)
                                pct = (line_data['flux_error'] / flux_val) * 100 if flux_val != 0 else 0
                                pct_item = self.lines_table.item(row, 8)
                                if pct_item:
                                    pct_item.setText(f"{pct:.1f}")
                            except ValueError:
                                pass
    
    def open_batch_flux_dialog(self):
        """Open dialog for batch flux calculation from multiple FITS files"""
        # Get selected lines that don't already have a flux -- Calculate Fluxes fills
        # in blanks, it never overwrites a value the user typed in or imported from
        # CSV. To recompute a row, clear its Flux cell first, then select and run.
        selected_rows = []
        skipped_rows = []
        for row in range(self.lines_table.rowCount()):
            checkbox = self.lines_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                flux_item = self.lines_table.item(row, 6)
                if flux_item and flux_item.text().strip():
                    skipped_rows.append(row)
                else:
                    selected_rows.append(row)

        if not selected_rows:
            if skipped_rows:
                QMessageBox.information(
                    self, "Nothing To Calculate",
                    f"All {len(skipped_rows)} selected line(s) already have a flux value. "
                    "Clear a row's Flux cell first if you want to recompute it."
                )
            else:
                QMessageBox.warning(self, "No Lines Selected", "Please select lines to calculate fluxes for.")
            return

        # Open file dialog for FITS files
        fits_paths, _ = QFileDialog.getOpenFileNames(
            self,
            f"Select FITS Files (for {len(selected_rows)} lines)",
            "",
            "FITS Files (*.fits *.fit);;All Files (*.*)"
        )

        if not fits_paths:
            return

        # Calculate fluxes using FITS files
        self.calculate_fluxes_batch(selected_rows, fits_paths, skipped_rows)
    
    def calculate_fluxes_batch(self, selected_rows, fits_paths, skipped_rows=None):
        """Calculate fluxes for all selected lines using their matched FITS files.
        Lines sharing a covering FITS file are fit together, simultaneously, via
        compute_line_fluxes_collective -- see that function's docstring for why a
        joint linear-least-squares solve is preferred here over an independent
        per-line fit (mainly: it keeps a weak line's flux from silently absorbing a
        nearby unrelated feature)."""
        from .data_loader import CubeData
        from .rotational_diagram_fitter import compute_line_fluxes_collective

        progress = QProgressDialog(
            "Loading FITS files...",
            None, 0, len(fits_paths) + len(selected_rows), self
        )
        progress.setWindowModality(Qt.WindowModal)

        # --- Load each FITS file once and pre-sum spatially ---
        loaded_cubes = []
        summed_spectra = []
        for i, fits_path in enumerate(fits_paths):
            progress.setValue(i)
            try:
                cube = CubeData()
                cube.load_from_file(fits_path)
                if cube.cube.ndim == 3:
                    summed = np.nansum(cube.cube, axis=(1, 2))
                else:
                    summed = cube.cube
                loaded_cubes.append(cube)
                summed_spectra.append(summed)
            except Exception as e:
                print(f"Skipping {fits_path}: {e}")

        if not loaded_cubes:
            QMessageBox.warning(self, "Error", "Could not load any FITS files.")
            progress.close()
            return

        # --- Find each line's covering cube (first match wins, same as before) ---
        rows_by_cube_idx = {}
        for row in selected_rows:
            wl = self.hitran_lines[row]['wavelength']
            matched_idx = None
            for i, cube in enumerate(loaded_cubes):
                wav = cube.wavelength.value
                if wav[0] <= wl <= wav[-1]:
                    matched_idx = i
                    break

            if matched_idx is None:
                print(f"Warning: No FITS file covers wavelength {wl:.6f} um for line {row+1}")
                QMessageBox.warning(self, "No FITS Match", f"No FITS file covers {wl:.6f} μm")
                progress.close()
                return

            rows_by_cube_idx.setdefault(matched_idx, []).append(row)

        # --- Fit every line sharing a cube together, one group per FITS file ---
        for group_idx, (cube_idx, rows) in enumerate(rows_by_cube_idx.items()):
            progress.setValue(len(fits_paths) + group_idx)
            cube_data = loaded_cubes[cube_idx]
            spectrum = summed_spectra[cube_idx]

            try:
                wavelength = cube_data.wavelength.value
                header = cube_data.header
                instrument = cube_data.instrument

                # Ensure wavelength is increasing
                if wavelength[0] > wavelength[-1]:
                    wavelength = wavelength[::-1]
                    spectrum = spectrum[::-1]

                line_wavelengths = [self.hitran_lines[row]['wavelength'] for row in rows]
                group_results = compute_line_fluxes_collective(
                    wavelength, spectrum, line_wavelengths, header, instrument
                )

                for row, result in zip(rows, group_results):
                    wl = self.hitran_lines[row]['wavelength']

                    if not result["success"]:
                        print(f"Warning: {result['message']} for line {wl:.6f}")
                        continue

                    flux_integrated = result["flux_cgs"]
                    flux_error = result["flux_error_cgs"]
                    is_upper_limit = result["is_upper_limit"]
                    F_ul_cgs = result["F_ul_cgs"]

                    # Use upper limit value if detected as upper limit
                    flux_to_display = F_ul_cgs if is_upper_limit else flux_integrated

                    # Populate table
                    flux_item = self.lines_table.item(row, 6)
                    if flux_item:
                        flux_item.setText(f"{flux_to_display:.4e}")

                    error_item = self.lines_table.item(row, 7)
                    if error_item:
                        error_item.setText(f"{flux_error:.4e}")

                    pct_item = self.lines_table.item(row, 8)
                    if pct_item and flux_integrated > 0:
                        pct = (flux_error / flux_integrated) * 100
                        pct_item.setText(f"{pct:.1f}")

                    # Auto-check upper limit if detected
                    ul_checkbox = self.lines_table.cellWidget(row, 9)
                    if ul_checkbox and is_upper_limit:
                        ul_checkbox.blockSignals(True)
                        ul_checkbox.setChecked(True)
                        ul_checkbox.blockSignals(False)

                    # Mark this row as system-calculated and record its source --
                    # overrides the "manual" reset that flux_item.setText() above
                    # just triggered via sync_calculated_flag
                    calc_checkbox = self.lines_table.cellWidget(row, 10)
                    if calc_checkbox:
                        calc_checkbox.setChecked(True)
                    source_item = self.lines_table.item(row, 11)
                    if source_item:
                        filename = os.path.basename(cube_data.filename) if cube_data.filename else "?"
                        source_item.setText(f"{instrument} [{filename}]")

            except Exception as e:
                print(f"Error processing lines from cube {cube_idx}: {e}")
                QMessageBox.critical(self, "Calculation Error", f"Error processing lines from FITS file:\n{e}")

        progress.close()
        msg = "Flux calculation complete!"
        if skipped_rows:
            msg += f"\n\n{len(skipped_rows)} selected line(s) already had a flux and were left unchanged."
        QMessageBox.information(self, "Complete", msg)
    
    def populate_lines_table(self):
        """Populate the unified table with all columns"""
        self.lines_table.setRowCount(len(self.hitran_lines))
        
        for row, line in enumerate(self.hitran_lines):
            # A line with no HITRAN match has no real A_ul/E_u/g_u -- these are hard
            # physics inputs (Nu = 4*pi*F*lam/(h*c*A_ul), E_u is the plot's x-axis), so
            # we never assume a value for them. The row is flagged and excluded from
            # fitting until the user supplies all three by hand.
            matched = line.get('hitran_matched', True)
            unmatched_bg = QColor(255, 214, 214)
            unmatched_tip = (
                "No HITRAN match for this line -- enter A_ul, E_u, and g_u manually "
                "to include it in the fit."
            )

            # Checkbox
            checkbox = QCheckBox()
            checkbox.stateChanged.connect(self.on_line_selected)
            if not matched:
                checkbox.setEnabled(False)
                checkbox.setToolTip(unmatched_tip)
            self.lines_table.setCellWidget(row, 0, checkbox)

            # Wavelength (read-only)
            wl_item = QTableWidgetItem(f"{line['wavelength']:.6f}")
            wl_item.setFlags(wl_item.flags() & ~Qt.ItemIsEditable)
            self.lines_table.setItem(row, 1, wl_item)

            # Transition (read-only)
            trans = line.get('transition', 'N/A')
            trans_item = QTableWidgetItem(trans)
            trans_item.setFlags(trans_item.flags() & ~Qt.ItemIsEditable)
            if not matched:
                trans_item.setBackground(unmatched_bg)
                trans_item.setToolTip(unmatched_tip)
            self.lines_table.setItem(row, 2, trans_item)

            if matched:
                # A_ul (read-only)
                a_ul = line.get('a_coeff', 0)
                a_item = QTableWidgetItem(f"{a_ul:.4e}")
                a_item.setFlags(a_item.flags() & ~Qt.ItemIsEditable)

                # E_u (read-only)
                e_u = line.get('e_upper', 0)
                e_item = QTableWidgetItem(f"{e_u:.2f}")
                e_item.setFlags(e_item.flags() & ~Qt.ItemIsEditable)

                # g_u (read-only)
                g_u = line.get('Gp', 1)
                g_item = QTableWidgetItem(f"{g_u:.0f}")
                g_item.setFlags(g_item.flags() & ~Qt.ItemIsEditable)
            else:
                # A_ul / E_u / g_u (editable) -- left blank, not defaulted, and
                # flagged red until the user fills in a real value for all three
                a_item = QTableWidgetItem("")
                e_item = QTableWidgetItem("")
                g_item = QTableWidgetItem("")
                for it in (a_item, e_item, g_item):
                    it.setBackground(unmatched_bg)
                    it.setToolTip(unmatched_tip)

            self.lines_table.setItem(row, 3, a_item)
            self.lines_table.setItem(row, 4, e_item)
            self.lines_table.setItem(row, 5, g_item)
            
            # Flux (editable) -- required for the fit, so flagged red while empty.
            flux_val = line.get('flux', '')
            flux_item = QTableWidgetItem(f"{flux_val:.4e}" if flux_val else "")
            if not flux_val:
                flux_item.setBackground(unmatched_bg)
                flux_item.setToolTip("Flux is required for the fit -- enter it manually or use Calculate Fluxes.")
            self.lines_table.setItem(row, 6, flux_item)

            # Flux Error (editable) -- flagged red while empty and not an Upper
            # Limit; see sync_flux_error_flag for why. Flag is set below, once the
            # Upper Limit checkbox (needed to evaluate it) also exists.
            self.lines_table.setItem(row, 7, QTableWidgetItem(""))

            # Error % (editable)
            self.lines_table.setItem(row, 8, QTableWidgetItem(""))

            # Upper Limit? (checkbox - auto-suggested by "Calculate Fluxes" (HITRAN or
            # CSV) based on a 3-sigma noise threshold; always user-editable so the
            # final call is yours, not just the heuristic's)
            ul_checkbox = QCheckBox()
            ul_checkbox.setToolTip(
                "Auto-checked when a line's fit falls below the 3σ noise threshold "
                "(via Calculate Fluxes). You can check/uncheck this yourself either way."
            )
            ul_checkbox.stateChanged.connect(lambda _state, r=row: self.sync_flux_error_flag(r))
            self.lines_table.setCellWidget(row, 9, ul_checkbox)

            # Calculated? (read-only status badge, not user-editable -- set/cleared
            # by calculate_fluxes_batch and by editing the Flux cell by hand)
            calc_checkbox = QCheckBox()
            calc_checkbox.setEnabled(False)
            calc_checkbox.setToolTip(
                "Checked when this line's flux was computed by Calculate Fluxes, rather "
                "than typed in by hand or imported from CSV."
            )
            self.lines_table.setCellWidget(row, 10, calc_checkbox)

            # Source (read-only): instrument + FITS filename the flux was computed
            # from, filled in by calculate_fluxes_batch alongside the Calculated? flag
            source_item = QTableWidgetItem("")
            source_item.setFlags(source_item.flags() & ~Qt.ItemIsEditable)
            self.lines_table.setItem(row, 11, source_item)

            self.sync_flux_error_flag(row)
        
        # Set initial edit/read-only status based on current mode
        self.update_error_columns()
    
    def on_cell_clicked(self, row, col):
        """Handle cell clicks (for compatibility)"""
        pass
    
    def on_line_selected(self):
        """Enable Run Fit only once >=2 lines are checked -- a rotational-diagram
        line fit needs at least 2 points (rotational_diagram_fitter.py raises "Need
        at least 2 points" otherwise), so this surfaces that requirement up front
        instead of letting the user click Run Fit and get an error back."""
        checked_count = 0
        for row in range(self.lines_table.rowCount()):
            checkbox = self.lines_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                checked_count += 1
        self.run_fit_btn.setEnabled(checked_count >= 2)
        self.run_fit_btn.setToolTip(
            "" if checked_count >= 2 else "Select at least 2 lines -- a line fit needs 2+ points."
        )
    
    def select_all_lines(self):
        """Select all lines in the table (skips rows disabled for missing HITRAN data)"""
        for row in range(self.lines_table.rowCount()):
            checkbox = self.lines_table.cellWidget(row, 0)
            if checkbox and checkbox.isEnabled():
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)
        self.on_line_selected()
    
    def deselect_all_lines(self):
        """Deselect all lines in the table"""
        for row in range(self.lines_table.rowCount()):
            checkbox = self.lines_table.cellWidget(row, 0)
            if checkbox:
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(False)
        self.on_line_selected()
    
    def run_rotational_fit(self):
        """Run the fitting"""
        fluxes = {}
        flux_errors = {}
        upper_limits = {}  # Track which measurements are upper limits
        molecular_data = {
            'wavelengths': {},
            'Aul': {},
            'Eu': {},
            'gu': {}
        }
        
        j_counter = 1
        for row in range(self.lines_table.rowCount()):
            checkbox = self.lines_table.cellWidget(row, 0)
            if not checkbox or not checkbox.isChecked():
                continue

            line = self.hitran_lines[row]

            # Belt-and-suspenders: never fit a line with no real A_ul/E_u/g_u, even
            # if something else managed to check its (normally disabled) box.
            if not line.get('hitran_matched', True) and (
                'a_coeff' not in line or 'e_upper' not in line or 'Gp' not in line
            ):
                QMessageBox.warning(
                    self, "Error",
                    f"Row {row+1} has no HITRAN match and is missing A_ul/E_u/g_u. "
                    f"Enter them manually or deselect this line."
                )
                return
            
            # Get flux
            flux_text = self.lines_table.item(row, 6).text().strip()
            if not flux_text:
                QMessageBox.warning(self, "Error", f"Missing flux for row {row+1}")
                return
            
            try:
                flux = float(flux_text)
            except ValueError:
                QMessageBox.warning(self, "Error", f"Invalid flux at row {row+1}")
                return
            
            # Get error from either column (whichever is populated). If both are
            # blank, flux_error stays None -- rotational_diagram_fitter._build_dataset
            # then EXCLUDES this line from the fit entirely if it's not an upper limit
            # (no unweighted fallback; a missing error is never defaulted), reported
            # back afterward via results['dropped_J'] in the "Lines Excluded" dialog.
            # Upper limits are exempt since they use a fixed 0 error by design.
            error_text = self.lines_table.item(row, 7).text().strip()
            error_pct_text = self.lines_table.item(row, 8).text().strip()
            
            flux_error = None
            
            if error_text:
                try:
                    flux_error = float(error_text)
                except ValueError:
                    QMessageBox.warning(self, "Error", f"Invalid flux error at row {row+1}")
                    return
            elif error_pct_text:
                try:
                    pct = float(error_pct_text)
                    flux_error = flux * (pct / 100.0)
                except ValueError:
                    QMessageBox.warning(self, "Error", f"Invalid error % at row {row+1}")
                    return
            
            # Check if this is an upper limit
            ul_checkbox = self.lines_table.cellWidget(row, 9)
            is_upper_limit = ul_checkbox and ul_checkbox.isChecked()
            
            fluxes[j_counter] = flux
            if flux_error is not None:
                flux_errors[j_counter] = flux_error
            if is_upper_limit:
                upper_limits[j_counter] = True
            
            molecular_data['wavelengths'][j_counter] = line['wavelength']
            molecular_data['Aul'][j_counter] = line.get('a_coeff', 0)
            molecular_data['Eu'][j_counter] = line.get('e_upper', 0)

            # Bare rotational degeneracy (2*Ju+1) for genuine H2 S-branch lines, so
            # apply_OPR can supply the nuclear-spin weighting variably. Only valid for
            # H2's S-branch numbering (get_Ju(SJ) = SJ+2) - everything else (CO, other
            # branches, unmatched CSV rows with no HITRAN j_lower) keeps HITRAN's Gp,
            # which has no OPR-fittable ortho/para asymmetry to strip out in the first place.
            j_lower = line.get('j_lower')
            if self.current_molecule == 'H2' and line.get('branch') == 'S' and j_lower is not None:
                molecular_data['gu'][j_counter] = get_gu(j_lower)
            else:
                molecular_data['gu'][j_counter] = line.get('Gp', 1)
            
            j_counter += 1
        
        if not fluxes:
            QMessageBox.warning(self, "Error", "No lines selected")
            return
        
        # Check that R and distance are provided (R may be intentionally left unknown)
        if self.radius_unknown_check.isChecked():
            radius_au = None
        else:
            try:
                radius_au = float(self.radius_input.text().strip())
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid radius (R) - must be a number")
                return

            if radius_au <= 0:
                QMessageBox.warning(self, "Error", "Radius (R) must be greater than 0 AU")
                return

        try:
            distance_pc = float(self.distance_input.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid distance - must be a number")
            return

        if distance_pc <= 0:
            QMessageBox.warning(self, "Error", "Distance must be greater than 0 pc")
            return
        
        fit_2t = self.fit_mode.currentText() == "2T"
        fit_opr = self.fit_opr_check.isChecked()

        opr_grid = None
        if fit_opr:
            try:
                opr_min = float(self.opr_min_input.text().strip())
                opr_max = float(self.opr_max_input.text().strip())
                opr_steps = int(self.opr_steps_input.text().strip())
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid OPR grid (min/max/steps) - must be numbers")
                return

            if opr_min >= opr_max or opr_steps < 2:
                QMessageBox.warning(self, "Error", "OPR min must be less than max, and steps must be >= 2")
                return

            opr_grid = np.linspace(opr_min, opr_max, opr_steps)

        try:
            from .rotational_diagram_fitter import rotational_diagram

            # Pass raw AU and pc values (function converts to cm internally)
            results = rotational_diagram(
                fluxes,
                flux_errors,
                molecular_data=molecular_data,
                OPR=None,
                fit_opr=fit_opr,
                opr_grid=opr_grid,
                fit2T=fit_2t,
                R=radius_au,       # in AU
                dist=distance_pc,  # in pc
                upper_limits=upper_limits
            )
            
            dropped_J = results.get("dropped_J")
            if dropped_J:
                dropped_desc = "\n".join(
                    f"  • {molecular_data['wavelengths'][j]:.6f} μm"
                    for j in dropped_J
                )
                QMessageBox.warning(
                    self,
                    "Lines Excluded From Fit",
                    "The following detections were excluded from the fit because no "
                    "flux error was entered for them (upper limits don't need one, but "
                    "detections do - no default is assumed):\n\n" + dropped_desc
                )

            self.show_results(results)

        except Exception as e:
            QMessageBox.critical(self, "Fitting Error", f"Error during fitting:\n{str(e)}")
    
    def show_results(self, results):
        """Show results with plot"""
        results_text = "\n" + "="*50 + "\n"
        results_text += "ROTATIONAL DIAGRAM ANALYSIS RESULTS\n"
        results_text += "="*50 + "\n\n"

        if "OPR" in results:
            if "OPR_err" in results:
                results_text += f"Best-fit OPR = {results['OPR']:.3f}\n"
                results_text += (
                    f"OPR 1σ error: -{results['OPR_err_neg']:.4f} / "
                    f"+{results['OPR_err_pos']:.4f} (symm ≈ {results['OPR_err']:.4f})\n"
                )
                results_text += f"OPR = {results['OPR']:.3f} ± {results['OPR_err']:.3f} (fitted)\n\n"
            else:
                results_text += f"OPR = {results['OPR']:.2f} (fixed)\n\n"

        def _convergence_note(fit_dict):
            """Return a warning line if the censored PyMC fit shows convergence issues."""
            n_div = fit_dict.get("n_divergences")
            if n_div is None:
                return ""
            note = f"[PyMC: {n_div} divergence(s), max r_hat = {fit_dict.get('max_r_hat', float('nan')):.3f}]"
            if n_div > 0 or fit_dict.get("max_r_hat", 1.0) > 1.01:
                note += "  ⚠ possible non-convergence - treat this result with caution"
            return note + "\n"

        if "1T" in results:
            r1t = results["1T"]
            results_text += "--- 1T FIT ---\n"
            results_text += f"T = {r1t['T']:.1f} ± {r1t['dT']:.1f} K\n"
            results_text += f"Ncol = {r1t['Ncol']:.2e} ± {r1t['dNcol']:.2e} cm^-2\n"
            results_text += _convergence_note(r1t)
            results_text += "\n"

        if "2T" in results:
            r2t = results["2T"]
            results_text += "--- 2T FIT ---\n"
            results_text += f"T1 = {r2t['T1']:.1f} ± {r2t.get('dT1', float('nan')):.1f} K\n"
            results_text += f"T2 = {r2t['T2']:.1f} ± {r2t.get('dT2', float('nan')):.1f} K\n"
            results_text += f"N1 = {r2t['Ncol1']:.2e} ± {r2t.get('dNcol1', float('nan')):.2e} cm^-2\n"
            results_text += f"N2 = {r2t['Ncol2']:.2e} ± {r2t.get('dNcol2', float('nan')):.2e} cm^-2\n"
            results_text += _convergence_note(r2t)

        results_text += "="*50 + "\n"

        print(results_text)

        # Create results dialog with plot
        dialog = QDialog(self)
        dialog.setWindowTitle("Rotational Diagram Results")
        dialog.setGeometry(100, 100, 1200, 600)

        layout = QHBoxLayout(dialog)

        # Left side: text results
        text_widget = QTextEdit()
        text_widget.setText(results_text)
        text_widget.setReadOnly(True)
        layout.addWidget(text_widget, 1)

        # Right side: plot with toolbar
        if "_plot_data" in results:
            plot_data = results["_plot_data"]
            fig = Figure(figsize=(6, 5), dpi=100)
            ax = fig.add_subplot(111)

            E_arr     = plot_data["E_arr"]
            y_arr     = plot_data["y_arr"]
            y_err     = plot_data["y_err"]
            is_ul     = plot_data.get("is_ul_mask", np.zeros(len(E_arr), dtype=bool))
            popt1     = plot_data["popt1"]
            popt2     = plot_data["popt2"]

            # Convert ln -> log10
            y_arr_log10 = y_arr / np.log(10)
            # y_err is fractional (dNu/Nu), convert to log10 scale
            y_err_log10 = y_err / np.log(10)

            det = ~is_ul

            # Detections — black circles with error bars
            if np.any(det):
                ax.errorbar(E_arr[det], y_arr_log10[det], yerr=y_err_log10[det],
                            fmt='o', color='black', markersize=6, capsize=4, zorder=3,
                            label='Detection')

            # Upper limits — red downward arrows, no error bar
            if np.any(is_ul):
                ax.errorbar(E_arr[is_ul], y_arr_log10[is_ul], yerr=0.15,
                            fmt='v', color='red', markersize=8,
                            uplims=True, zorder=3, label='Upper limit')

            # Generate fit lines
            E_fit = np.linspace(np.min(E_arr), np.max(E_arr), 200)
            ax.set_xlim(np.min(E_arr) - 500, np.max(E_arr) + 500)

            is_2T = popt2 is not None

            # Show 1T line only if NOT doing 2T fit
            if popt1 is not None and not is_2T:
                from .rotational_diagram_fitter import model_1T
                y1T = model_1T(E_fit, *popt1)
                ax.plot(E_fit, y1T / np.log(10), '--', linewidth=2, label='1T')

            # 2T: show two component lines AND combined curve
            if popt2 is not None:
                from .rotational_diagram_fitter import model_2T
                a1_2, T1_2, a2_2, T2_2 = popt2

                y_comp1 = a1_2 - E_fit / T1_2
                ax.plot(E_fit, y_comp1 / np.log(10), ':', linewidth=2.5,
                        color='blue', label=f'Warm: T={T1_2:.0f} K')

                y_comp2 = a2_2 - E_fit / T2_2
                ax.plot(E_fit, y_comp2 / np.log(10), ':', linewidth=2.5,
                        color='orange', label=f'Hot: T={T2_2:.0f} K')

                y2T = model_2T(E_fit, *popt2)
                ax.plot(E_fit, y2T / np.log(10), '-', linewidth=3,
                        color='red', label='2-component model')

            ax.tick_params(axis='both', which='major', labelsize=11)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            ax.set_xlabel('E_u (K)', fontsize=12, fontweight='bold')
            ax.set_ylabel('log(N_u / g_u)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=10, loc='upper right')

            plot_widget = QWidget()
            plot_layout = QVBoxLayout(plot_widget)

            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, dialog)

            plot_layout.addWidget(toolbar)
            plot_layout.addWidget(canvas)
            layout.addWidget(plot_widget, 1)

        dialog.exec_()
