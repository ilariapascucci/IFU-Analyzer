"""
Rotational Diagram Analysis GUI for Molecular data (H2, CO) - SIMPLIFIED
Single unified interface for selecting molecules, lines, and entering flux data
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTableWidget, QTableWidgetItem, QSpinBox, 
    QDoubleSpinBox, QCheckBox, QMessageBox, QProgressDialog,
    QFrame, QComboBox, QDialog, QTextEdit, QFileDialog, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .line_catalog import fetch_hitran_data


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
    
    def __init__(self):
        super().__init__()
        self.current_molecule = 'H2'  # Default molecule
        self.setWindowTitle(f"{self.current_molecule} Rotational Diagram Analyzer")
        self.setGeometry(100, 100, 1500, 800)
        
        self.hitran_lines = []
        self.fetch_thread = None
        self.updating_error_columns = False  # Flag to prevent recursive updates
        
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
        toggle_layout.addStretch()
        main_layout.addLayout(toggle_layout)
        
        self.lines_table = QTableWidget()
        self.lines_table.setColumnCount(10)
        self.lines_table.setHorizontalHeaderLabels(
            ["✓", "λ (μm)", "Transition", "A_ul (s⁻¹)", "E_u (K)", "g_u", "Flux", "Flux Error", "Error %", "Upper Limit?"]
        )
        self.lines_table.setColumnWidth(0, 40)
        self.lines_table.setColumnWidth(1, 80)
        self.lines_table.setColumnWidth(2, 120)
        self.lines_table.setColumnWidth(3, 110)
        self.lines_table.setColumnWidth(4, 80)
        self.lines_table.setColumnWidth(5, 60)
        self.lines_table.setColumnWidth(9, 85)
        main_layout.addWidget(self.lines_table)
        
        # Connect table changes for auto-calculation
        self.lines_table.itemChanged.connect(self.on_table_item_changed)
        
        # Connect upper limit checkbox changes
        self.lines_table.cellClicked.connect(self.on_cell_clicked)
        
        # Bottom: Physical parameters + buttons
        bottom_layout = QHBoxLayout()
        
        bottom_layout.addWidget(QLabel("R (au):"))
        self.radius_input = QDoubleSpinBox()
        self.radius_input.setRange(0.01, 10000)
        self.radius_input.setValue(17)
        self.radius_input.setDecimals(4)
        bottom_layout.addWidget(self.radius_input)
        
        bottom_layout.addWidget(QLabel("Distance (pc):"))
        self.distance_input = QDoubleSpinBox()
        self.distance_input.setRange(0.1, 100000)
        self.distance_input.setValue(71.5)
        self.distance_input.setDecimals(4)
        bottom_layout.addWidget(self.distance_input)
        
        bottom_layout.addWidget(QLabel("Fit:"))
        self.fit_mode = QComboBox()
        self.fit_mode.addItems(["1T", "2T"])
        bottom_layout.addWidget(self.fit_mode)
        
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
        self.wl_min_input.setDecimals(4)
        input_layout.addWidget(self.wl_min_input)
        
        input_layout.addWidget(QLabel("λ_max (μm):"))
        self.wl_max_input = QDoubleSpinBox()
        self.wl_max_input.setRange(0.1, 1000)
        self.wl_max_input.setValue(20.0)
        self.wl_max_input.setDecimals(4)
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
        """Auto-calculate flux error or percentage when user enters value, clear the other when empty"""
        if self.updating_error_columns:
            return
        
        self.updating_error_columns = True
        
        try:
            if self.lines_table.rowCount() == 0:
                return
            
            row = item.row()
            col = item.column()
            
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
                
                # Check for required columns (case-insensitive)
                available_cols = {col.strip().lower(): col.strip() for col in reader.fieldnames if col}
                required_cols = {'species', 'wavelength', 'flux'}
                
                if not required_cols.issubset(available_cols.keys()):
                    missing = required_cols - available_cols.keys()
                    QMessageBox.warning(
                        self,
                        "Missing Columns",
                        f"CSV must have 'species', 'wavelength', and 'flux' columns.\nMissing: {', '.join(missing)}"
                    )
                    return
                
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
                flux_col = available_cols['flux']
                
                for row_dict in reader:
                    try:
                        species = row_dict.get(species_col, '').strip().upper()
                        wavelength = float(row_dict.get(wavelength_col, '').strip())
                        flux = float(row_dict.get(flux_col, '').strip())
                        
                        row_info = {
                            'species': species,
                            'wavelength': wavelength,
                            'flux': flux
                        }
                        
                        # Add error if available
                        if has_error and error_col:
                            try:
                                flux_error = float(row_dict.get(error_col, '').strip())
                                row_info['flux_error'] = flux_error
                            except (ValueError, AttributeError):
                                pass
                        
                        csv_data.append(row_info)
                    except (ValueError, TypeError):
                        # Skip rows with invalid data
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
                        'flux': csv_row['flux'],
                        'a_coeff': best_match.get('a_coeff', 0.0),
                        'e_upper': best_match.get('e_upper', 0.0),
                        'Gp': best_match.get('Gp', 1.0),
                        'transition': best_match.get('transition', 'User Data'),
                        'species': species
                    }
                else:
                    # No matching HITRAN line found, use defaults
                    row_data = {
                        'wavelength': wavelength,
                        'flux': csv_row['flux'],
                        'a_coeff': 0.0,  # Unknown
                        'e_upper': 0.0,  # Unknown
                        'Gp': 1.0,
                        'transition': 'User Data (no HITRAN match)',
                        'species': species
                    }
                    print(f"Warning: No HITRAN match for {species} at {wavelength:.4f} μm")
                
                # Add error if available
                if 'flux_error' in csv_row:
                    row_data['flux_error'] = csv_row['flux_error']
                
                rows.append(row_data)
            
            self.hitran_lines = rows
            self.populate_lines_table()
            
            # Pre-fill errors if they were provided in CSV
            self.populate_csv_errors()
            
            # Count successful matches
            matched = sum(1 for r in rows if r['a_coeff'] > 0)
            QMessageBox.information(
                self,
                "Import Successful",
                f"Imported {len(rows)} rows from CSV\n"
                f"Matched {matched}/{len(rows)} lines to HITRAN data"
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
    
    def populate_lines_table(self):
        """Populate the unified table with all columns"""
        self.lines_table.setRowCount(len(self.hitran_lines))
        
        for row, line in enumerate(self.hitran_lines):
            # Checkbox
            checkbox = QCheckBox()
            checkbox.stateChanged.connect(self.on_line_selected)
            self.lines_table.setCellWidget(row, 0, checkbox)
            
            # Wavelength (read-only)
            wl_item = QTableWidgetItem(f"{line['wavelength']:.6f}")
            wl_item.setFlags(wl_item.flags() & ~Qt.ItemIsEditable)
            self.lines_table.setItem(row, 1, wl_item)
            
            # Transition (read-only)
            trans = line.get('transition', 'N/A')
            trans_item = QTableWidgetItem(trans)
            trans_item.setFlags(trans_item.flags() & ~Qt.ItemIsEditable)
            self.lines_table.setItem(row, 2, trans_item)
            
            # A_ul (read-only)
            a_ul = line.get('a_coeff', 0)
            a_item = QTableWidgetItem(f"{a_ul:.4e}")
            a_item.setFlags(a_item.flags() & ~Qt.ItemIsEditable)
            self.lines_table.setItem(row, 3, a_item)
            
            # E_u (read-only)
            e_u = line.get('e_upper', 0)
            e_item = QTableWidgetItem(f"{e_u:.2f}")
            e_item.setFlags(e_item.flags() & ~Qt.ItemIsEditable)
            self.lines_table.setItem(row, 4, e_item)
            
            # g_u (read-only)
            g_u = line.get('Gp', 1)
            g_item = QTableWidgetItem(f"{g_u:.0f}")
            g_item.setFlags(g_item.flags() & ~Qt.ItemIsEditable)
            self.lines_table.setItem(row, 5, g_item)
            
            # Flux (editable)
            flux_val = line.get('flux', '')
            if flux_val:
                self.lines_table.setItem(row, 6, QTableWidgetItem(f"{flux_val:.4e}"))
            else:
                self.lines_table.setItem(row, 6, QTableWidgetItem(""))
            
            # Flux Error (editable)
            self.lines_table.setItem(row, 7, QTableWidgetItem(""))
            
            # Error % (editable)
            self.lines_table.setItem(row, 8, QTableWidgetItem(""))
            
            # Upper Limit? (editable checkbox)
            ul_checkbox = QCheckBox()
            # Connect checkbox state change to upper limit dialog
            ul_checkbox.stateChanged.connect(lambda state, r=row: self.on_upper_limit_checkbox_changed(r, state))
            self.lines_table.setCellWidget(row, 9, ul_checkbox)
        
        # Set initial edit/read-only status based on current mode
        self.update_error_columns()
    
    def on_line_selected(self):
        """Enable Run Fit button if lines selected"""
        has_selection = False
        for row in range(self.lines_table.rowCount()):
            checkbox = self.lines_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                has_selection = True
                break
        self.run_fit_btn.setEnabled(has_selection)
    
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
            
            # Get error from either column (whichever is populated)
            # Errors are OPTIONAL - if missing, flux_error will be None (unweighted fitting)
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
            molecular_data['gu'][j_counter] = line.get('Gp', 1)
            
            j_counter += 1
        
        if not fluxes:
            QMessageBox.warning(self, "Error", "No lines selected")
            return
        
        # Check that R and distance are provided
        radius_au = self.radius_input.value()
        distance_pc = self.distance_input.value()
        
        if radius_au <= 0:
            QMessageBox.warning(self, "Error", "Radius (R) must be greater than 0 AU")
            return
        
        if distance_pc <= 0:
            QMessageBox.warning(self, "Error", "Distance must be greater than 0 pc")
            return
        
        try:
            fit_2t = self.fit_mode.currentText() == "2T"
            
            from .rotational_diagram_fitter import rotational_diagram
            
            # Pass raw AU and pc values (function converts to cm internally)
            results = rotational_diagram(
                fluxes,
                flux_errors,
                molecular_data=molecular_data,
                OPR=None,
                fit2T=fit_2t,
                R=radius_au,       # in AU
                dist=distance_pc,  # in pc
                upper_limits=upper_limits
            )
            
            self.show_results(results)
            
        except Exception as e:
            QMessageBox.critical(self, "Fitting Error", f"Error during fitting:\n{str(e)}")
    
    def show_results(self, results):
        """Show results with plot"""
        results_text = "\n" + "="*50 + "\n"
        results_text += "ROTATIONAL DIAGRAM ANALYSIS RESULTS\n"
        results_text += "="*50 + "\n\n"
        
        if "1T" in results:
            r1t = results["1T"]
            results_text += "--- 1T FIT ---\n"
            results_text += f"T = {r1t['T']:.1f} ± {r1t['dT']:.1f} K\n"
            results_text += f"Ncol = {r1t['Ncol']:.2e} ± {r1t['dNcol']:.2e} cm^-2\n\n"
        
        if "2T" in results:
            r2t = results["2T"]
            results_text += "--- 2T FIT ---\n"
            results_text += f"T1 = {r2t['T1']:.1f} ± {r2t['dT1']:.1f} K\n"
            results_text += f"T2 = {r2t['T2']:.1f} ± {r2t['dT2']:.1f} K\n"
            results_text += f"N1 = {r2t['Ncol1']:.2e} ± {r2t['dNcol1']:.2e} cm^-2\n"
            results_text += f"N2 = {r2t['Ncol2']:.2e} ± {r2t['dN2']:.2e} cm^-2\n"
        
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
            
            E_arr = plot_data["E_arr"]
            y_arr = plot_data["y_arr"]
            y_err = plot_data["y_err"]
            popt1 = plot_data["popt1"]
            popt2 = plot_data["popt2"]
            
            # Convert to log10 scale
            y_arr_log10 = np.log10(np.exp(y_arr))
            y_err_log10 = np.log10(np.exp(y_err))
            
            # Plot data points with error bars
            ax.errorbar(E_arr, y_arr_log10, yerr=y_err_log10, fmt='o', color='black', 
                       markersize=6, capsize=4, zorder=3)
            
            # Generate fit lines
            E_fit = np.linspace(np.min(E_arr), np.max(E_arr), 200)
            
            # Determine if 2T was performed
            is_2T = popt2 is not None
            
            # Show 1T line only if NOT doing 2T fit
            if popt1 is not None and not is_2T:
                from .rotational_diagram_fitter import model_1T
                y1T = model_1T(E_fit, *popt1)
                ax.plot(E_fit, np.log10(np.exp(y1T)), '--', linewidth=2, label='1T')
            
            # 2T: show two component lines AND combined curve
            if popt2 is not None:
                from .rotational_diagram_fitter import model_2T
                a1_2, T1_2, a2_2, T2_2 = popt2
                
                # Component 1 (warm) - dotted
                y_comp1 = a1_2 - E_fit / T1_2
                ax.plot(E_fit, np.log10(np.exp(y_comp1)), ':', linewidth=2.5, color='blue', label=f'Warm: T={T1_2:.0f} K')
                
                # Component 2 (hot) - dotted
                y_comp2 = a2_2 - E_fit / T2_2
                ax.plot(E_fit, np.log10(np.exp(y_comp2)), ':', linewidth=2.5, color='orange', label=f'Hot: T={T2_2:.0f} K')
                
                # Combined 2T curve - solid
                y2T = model_2T(E_fit, *popt2)
                ax.plot(E_fit, np.log10(np.exp(y2T)), '-', linewidth=3, color='red', label='2-component model')
            
            # Increase tick sizes
            ax.tick_params(axis='both', which='major', labelsize=11)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            
            ax.set_xlabel('E_u (K)', fontsize=12, fontweight='bold')
            ax.set_ylabel('log(N_u / g_u)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=10, loc='upper right')
            
            # Plot widget container with toolbar
            plot_widget = QWidget()
            plot_layout = QVBoxLayout(plot_widget)
            
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, dialog)
            
            plot_layout.addWidget(toolbar)
            plot_layout.addWidget(canvas)
            layout.addWidget(plot_widget, 1)
        
        dialog.exec_()
    
    def on_upper_limit_checkbox_changed(self, row, state):
        """Handle upper limit checkbox state change"""
        ul_checkbox = self.lines_table.cellWidget(row, 9)
        if not ul_checkbox:
            return
        
        # Only process if checkbox just became checked
        if state != Qt.Checked:
            return
        
        # Clear error columns when upper limit is checked
        flux_error_item = self.lines_table.item(row, 7)
        error_pct_item = self.lines_table.item(row, 8)
        if flux_error_item:
            flux_error_item.setText("")
        if error_pct_item:
            error_pct_item.setText("")
        
        # User checked upper limit - prompt for FITS file
        dialog = UpperLimitFitsDialog(self, row, self.hitran_lines[row])
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            flux_ul = dialog.get_flux_ul()
            if flux_ul is not None:
                # Auto-populate the flux column
                flux_item = self.lines_table.item(row, 6)
                if flux_item:
                    flux_item.setText(f"{flux_ul:.4e}")
                # Also clear error columns again to be safe
                if flux_error_item:
                    flux_error_item.setText("")
                if error_pct_item:
                    error_pct_item.setText("")
        else:
            # User cancelled, uncheck the box
            ul_checkbox.blockSignals(True)
            ul_checkbox.setChecked(False)
            ul_checkbox.blockSignals(False)
    
    def on_cell_clicked(self, row, col):
        """Handle cell clicks (for compatibility)"""
        pass


class UpperLimitFitsDialog(QDialog):
    """Dialog for loading FITS and performing Gaussian fit for upper limits - reuses main window pattern"""
    
    def __init__(self, parent, row, line_data):
        super().__init__(parent)
        self.row = row
        self.line_data = line_data
        self.flux_ul = None
        self.sigma_noise = None  # Measured noise std dev
        self.fits_file = None
        self.spectrum = None
        self.wavelength = None
        self.gauss_mode = None  # 'left', 'right', 'noise_left', 'noise_right', or None
        self.fits_header = None  # Store header for unit conversion
        self.fits_instrument = None  # Store instrument for unit conversion
        
        # Boundary markers for line region
        self.gauss_left_x = None
        self.gauss_right_x = None
        self.gauss_left_marker = None
        self.gauss_right_marker = None
        
        # Noise region markers
        self.noise_left_x = None
        self.noise_right_x = None
        self.noise_left_marker = None
        self.noise_right_marker = None
        
        # Gaussian boundary markers
        self.gauss_left_x = None
        self.gauss_right_x = None
        self.gauss_left_marker = None
        self.gauss_right_marker = None
        
        self.setWindowTitle(f"Upper Limit: Fit Line {line_data['wavelength']:.6f} μm")
        self.setGeometry(100, 100, 1100, 750)
        
        layout = QVBoxLayout()
        
        # FITS file selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("FITS File:"))
        self.file_label = QLabel("No file selected")
        file_layout.addWidget(self.file_label)
        
        load_btn = QPushButton("Load FITS")
        load_btn.clicked.connect(self.load_fits_file)
        file_layout.addWidget(load_btn)
        file_layout.addStretch()
        layout.addLayout(file_layout)
        
        # Spectrum plot area
        self.fig = Figure(figsize=(11, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        layout.addWidget(self.canvas)
        
        # Add matplotlib toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        
        # Instructions
        instructions = QLabel(
            "1. Load FITS file\n"
            "2. Click 'Noise L', then click on line-free region to mark left edge\n"
            "3. Click 'Noise R', then click to mark right edge (noise std will be calculated)\n"
            "4. Click 'Left' button, then click spectrum to mark line left boundary\n"
            "5. Click 'Right' button, then click spectrum to mark line right boundary\n"
            "6. Click 'Fit Gaussian' to calculate 3σ upper limit"
        )
        layout.addWidget(instructions)
        
        # Interactive buttons
        button_layout = QHBoxLayout()
        
        # Noise selection buttons
        self.noise_left_btn = QPushButton("Noise L")
        self.noise_left_btn.setCheckable(True)
        self.noise_left_btn.clicked.connect(self.on_noise_left_clicked)
        self.noise_left_btn.setEnabled(False)
        button_layout.addWidget(self.noise_left_btn)
        
        self.noise_right_btn = QPushButton("Noise R")
        self.noise_right_btn.setCheckable(True)
        self.noise_right_btn.clicked.connect(self.on_noise_right_clicked)
        self.noise_right_btn.setEnabled(False)
        button_layout.addWidget(self.noise_right_btn)
        
        button_layout.addSpacing(10)
        
        # Line boundary buttons
        self.left_btn = QPushButton("Left")
        self.left_btn.setCheckable(True)
        self.left_btn.clicked.connect(self.on_left_clicked)
        self.left_btn.setEnabled(False)
        button_layout.addWidget(self.left_btn)
        
        self.right_btn = QPushButton("Right")
        self.right_btn.setCheckable(True)
        self.right_btn.clicked.connect(self.on_right_clicked)
        self.right_btn.setEnabled(False)
        button_layout.addWidget(self.right_btn)
        
        button_layout.addStretch()
        
        self.fit_btn = QPushButton("Fit Gaussian")
        self.fit_btn.clicked.connect(self.fit_gaussian)
        self.fit_btn.setEnabled(False)
        button_layout.addWidget(self.fit_btn)
        
        layout.addLayout(button_layout)
        
        # Result display (selectable text)
        self.result_display = QLineEdit()
        self.result_display.setReadOnly(True)
        layout.addWidget(self.result_display)
        
        # OK/Cancel buttons
        dialog_buttons = QHBoxLayout()
        
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setEnabled(False)
        dialog_buttons.addWidget(self.ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        dialog_buttons.addWidget(cancel_btn)
        
        layout.addLayout(dialog_buttons)
        
        self.setLayout(layout)
    
    def on_left_clicked(self):
        """Activate left boundary selection mode"""
        if self.left_btn.isChecked():
            self.gauss_mode = 'left'
            self.right_btn.setChecked(False)
            self.noise_left_btn.setChecked(False)
            self.noise_right_btn.setChecked(False)
        else:
            self.gauss_mode = None
    
    def on_right_clicked(self):
        """Activate right boundary selection mode"""
        if self.right_btn.isChecked():
            self.gauss_mode = 'right'
            self.left_btn.setChecked(False)
            self.noise_left_btn.setChecked(False)
            self.noise_right_btn.setChecked(False)
        else:
            self.gauss_mode = None
    
    def on_noise_left_clicked(self):
        """Activate noise left boundary selection mode"""
        if self.noise_left_btn.isChecked():
            self.gauss_mode = 'noise_left'
            self.left_btn.setChecked(False)
            self.right_btn.setChecked(False)
            self.noise_right_btn.setChecked(False)
        else:
            self.gauss_mode = None
    
    def on_noise_right_clicked(self):
        """Activate noise right boundary selection mode"""
        if self.noise_right_btn.isChecked():
            self.gauss_mode = 'noise_right'
            self.left_btn.setChecked(False)
            self.right_btn.setChecked(False)
            self.noise_left_btn.setChecked(False)
        else:
            self.gauss_mode = None
    
    def on_canvas_click(self, event):
        """Handle canvas clicks to set boundaries and noise region"""
        if event.inaxes != self.ax or event.xdata is None:
            return
        
        if self.gauss_mode == 'noise_left':
            # Remove old marker
            if self.noise_left_marker:
                try:
                    self.noise_left_marker.remove()
                except:
                    pass
            
            # Add new marker
            self.noise_left_marker = self.ax.axvline(event.xdata, color='orange', linestyle='--', linewidth=1.3, label='Noise L', alpha=0.7)
            self.noise_left_x = event.xdata
            self.noise_left_btn.setChecked(False)
            self.gauss_mode = None
            self.canvas.draw()
            
            # If both noise boundaries are set, calculate sigma_noise
            if self.noise_left_x is not None and self.noise_right_x is not None:
                self._calculate_noise_std()
        
        elif self.gauss_mode == 'noise_right':
            # Remove old marker
            if self.noise_right_marker:
                try:
                    self.noise_right_marker.remove()
                except:
                    pass
            
            # Add new marker
            self.noise_right_marker = self.ax.axvline(event.xdata, color='orange', linestyle='--', linewidth=1.3, label='Noise R', alpha=0.7)
            self.noise_right_x = event.xdata
            self.noise_right_btn.setChecked(False)
            self.gauss_mode = None
            self.canvas.draw()
            
            # If both noise boundaries are set, calculate sigma_noise
            if self.noise_left_x is not None and self.noise_right_x is not None:
                self._calculate_noise_std()
        
        elif self.gauss_mode == 'left':
            # Remove old marker
            if self.gauss_left_marker:
                try:
                    self.gauss_left_marker.remove()
                except:
                    pass
            
            # Add new marker
            self.gauss_left_marker = self.ax.axvline(event.xdata, color='green', linestyle='--', linewidth=1.5, label='Left', alpha=0.7)
            self.gauss_left_x = event.xdata
            self.left_btn.setChecked(False)
            self.gauss_mode = None
            self.canvas.draw()
        
        elif self.gauss_mode == 'right':
            # Remove old marker
            if self.gauss_right_marker:
                try:
                    self.gauss_right_marker.remove()
                except:
                    pass
            
            # Add new marker
            self.gauss_right_marker = self.ax.axvline(event.xdata, color='purple', linestyle='--', linewidth=1.5, label='Right', alpha=0.7)
            self.gauss_right_x = event.xdata
            self.right_btn.setChecked(False)
            self.gauss_mode = None
            self.canvas.draw()
        
        # Enable fit button if all required selections are made
        if (self.gauss_left_x is not None and self.gauss_right_x is not None and 
            self.noise_left_x is not None and self.noise_right_x is not None and
            self.sigma_noise is not None):
            self.fit_btn.setEnabled(True)
    
    def load_fits_file(self):
        """Load FITS file and display spectrum (using main window's CubeData loader for consistency)"""
        fits_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select FITS File",
            "",
            "FITS Files (*.fits *.fit);;All Files (*.*)"
        )
        
        if not fits_path:
            return
        
        try:
            # Import CubeData from data_loader to ensure same wavelength extraction as main window
            from .data_loader import CubeData
            
            print(f"[DEBUG] Loading FITS: {fits_path}")
            
            # Use CubeData to load FITS (handles MUSE, ALMA, MIRI, NIRSpec properly)
            cube_data = CubeData()
            cube_data.load_from_file(fits_path)
            
            print(f"[DEBUG] CubeData loaded: cube shape={cube_data.cube.shape}, wavelength shape={cube_data.wavelength.shape}")
            
            # Extract center pixel spectrum from cube
            if cube_data.cube.ndim == 3:
                # Shape is (wavelength, y, x) - take center pixel
                ny, nx = cube_data.cube.shape[1], cube_data.cube.shape[2]
                center_y, center_x = ny // 2, nx // 2
                spectrum = cube_data.cube[:, center_y, center_x]
                print(f"[DEBUG] Extracted center pixel at ({center_y}, {center_x}), spectrum shape={spectrum.shape}")
            elif cube_data.cube.ndim == 1:
                spectrum = cube_data.cube
                print(f"[DEBUG] 1D spectrum: shape={spectrum.shape}")
            else:
                print(f"[DEBUG] ERROR: Unsupported cube shape: {cube_data.cube.shape}")
                QMessageBox.warning(self, "Error", f"Unsupported cube shape: {cube_data.cube.shape}")
                return
            
            # Get wavelength with units and convert to plain array
            wavelength_quantity = cube_data.wavelength
            if wavelength_quantity is not None:
                wavelength = wavelength_quantity.value  # Extract numeric values
                print(f"[DEBUG] Extracted wavelength values, shape={wavelength.shape}")
                
                # Safety checks
                if np.any(np.isnan(wavelength)):
                    print(f"[DEBUG] WARNING: wavelength contains NaN values!")
                    nan_count = np.sum(np.isnan(wavelength))
                    print(f"[DEBUG] NaN count: {nan_count}/{len(wavelength)}")
                
                # Ensure wavelength is increasing
                if wavelength[0] > wavelength[-1]:
                    print(f"[DEBUG] Wavelength is decreasing, reversing...")
                    wavelength = wavelength[::-1]
                    spectrum = spectrum[::-1]
                
                # Sanity check on wavelength range
                print(f"[DEBUG] Wavelength range: {wavelength[0]:.6f} to {wavelength[-1]:.6f}")
                if wavelength[-1] - wavelength[0] < 0.001:
                    print(f"[DEBUG] WARNING: wavelength span is very small ({wavelength[-1] - wavelength[0]:.6e})")
                    
            else:
                print(f"[DEBUG] ERROR: wavelength_quantity is None")
                QMessageBox.warning(self, "Error", "Could not extract wavelength from FITS")
                return
            
            print(f"[DEBUG] spectrum stats: min={np.nanmin(spectrum):.4e}, max={np.nanmax(spectrum):.4e}, mean={np.nanmean(spectrum):.4e}")
            print(f"[DEBUG] spectrum has {np.sum(np.isnan(spectrum))} NaN values out of {len(spectrum)}")
            print(f"[DEBUG] spectrum has {np.sum(np.isinf(spectrum))} inf values")
            print(f"[DEBUG] wavelength stats: min={np.min(wavelength):.6f}, max={np.max(wavelength):.6f}")
            
            # Check if spectrum is all zeros
            if np.allclose(spectrum, 0):
                print(f"[DEBUG] WARNING: spectrum is all zeros!")
            
            # Check if spectrum has any valid data
            valid_mask = ~np.isnan(spectrum) & ~np.isinf(spectrum)
            if not np.any(valid_mask):
                print(f"[DEBUG] ERROR: No valid data in spectrum!")
                QMessageBox.warning(self, "Error", "No valid data in spectrum (all NaN or inf)")
                return
            
            valid_count = np.sum(valid_mask)
            print(f"[DEBUG] Valid spectrum points: {valid_count}/{len(spectrum)}")
            
            # Store spectrum and wavelength
            self.spectrum = spectrum
            self.wavelength = wavelength
            self.fits_file = fits_path
            self.fits_header = cube_data.header  # Store header for unit conversion
            self.fits_instrument = cube_data.instrument  # Store instrument type
            self.file_label.setText(os.path.basename(fits_path))
            
            # Enable boundary and noise buttons
            self.left_btn.setEnabled(True)
            self.right_btn.setEnabled(True)
            self.noise_left_btn.setEnabled(True)
            self.noise_right_btn.setEnabled(True)
            
            print(f"[DEBUG] About to call plot_spectrum()")
            # Plot spectrum
            self.plot_spectrum()
            print(f"[DEBUG] plot_spectrum() completed")
        
        except Exception as e:
            import traceback
            print(f"[DEBUG] Exception in load_fits_file: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load FITS:\n{str(e)}")
    
    def plot_spectrum(self):
        """Plot the loaded spectrum (using visualization.py approach for robustness)"""
        if self.spectrum is None or self.wavelength is None:
            print("[DEBUG] plot_spectrum: spectrum or wavelength is None")
            return
        
        print(f"[DEBUG] plot_spectrum: spectrum shape={self.spectrum.shape}, wavelength shape={self.wavelength.shape}")
        print(f"[DEBUG] spectrum range: [{np.nanmin(self.spectrum):.4e}, {np.nanmax(self.spectrum):.4e}]")
        print(f"[DEBUG] wavelength range: [{np.nanmin(self.wavelength):.6f}, {np.nanmax(self.wavelength):.6f}]")
        
        self.ax.clear()
        
        # Reset axis limits (like visualization.py does)
        self.ax.set_xlim(auto=True)
        self.ax.set_ylim(auto=True)
        
        try:
            # Use raw spectrum data directly
            x = self.wavelength
            y = np.asarray(self.spectrum).astype(np.float32)
            
            # Plot as step (like main window)
            self.ax.step(x, y, where='mid', color='blue', linewidth=1.2, label='Spectrum')
            
            # Highlight peak (catalog line)
            peak_wl = self.line_data['wavelength']
            self.ax.axvline(peak_wl, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Catalog line ({peak_wl:.6f} μm)')
            
            self.ax.set_xlabel('Wavelength (μm)', fontsize=11)
            self.ax.set_ylabel('Flux', fontsize=11)
            self.ax.set_title('Click Left and Right buttons, then click spectrum to mark boundaries')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            
            print(f"[DEBUG] Plot created successfully")
            
        except Exception as e:
            print(f"[DEBUG] Error in plot_spectrum: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: empty plot
            self.ax.plot([], [])
        
        # Tight layout and draw
        self.fig.tight_layout()
        print(f"[DEBUG] Calling canvas.draw()")
        self.canvas.draw()
        print(f"[DEBUG] Canvas drawn")
    
    def _calculate_noise_std(self):
        """Calculate noise standard deviation (in flux units) from selected noise region"""
        if self.spectrum is None or self.wavelength is None:
            return
        
        # Ensure left < right
        left = min(self.noise_left_x, self.noise_right_x)
        right = max(self.noise_left_x, self.noise_right_x)
        
        # Extract noise region
        mask = (self.wavelength >= left) & (self.wavelength <= right)
        noise_spectrum = self.spectrum[mask]
        
        if len(noise_spectrum) < 2:
            QMessageBox.warning(self, "Error", "Noise region too small")
            return
        
        # Calculate std (RMS noise level in flux units, e.g., MJy/sr)
        self.sigma_noise = np.nanstd(noise_spectrum)
        print(f"[DEBUG] Noise region: σ_noise = {self.sigma_noise:.6e} (flux units)")
        
        # Update result display
        self.result_display.setText(f"σ_noise = {self.sigma_noise:.6e} (measured)")
    
    def _get_spectral_resolution_sigma(self, wavelength_um):
        """
        Get spectral resolution as Gaussian sigma (in µm) based on FITS instrument.
        Returns sigma_ref = (λ / R) / 2.355, where R is spectral resolving power.
        """
        if self.fits_header is None:
            return None
        

        grating = str(self.fits_header.get('GRATING', '')).strip().upper()
        instrume = str(self.fits_header.get('INSTRUME', '')).strip().upper()

        # NIRSpec spectral resolutions (R values depend on grating)
        if 'NIRSPEC' in instrume:
            if grating.startswith('G140H'):
                R = 2700
            elif grating.startswith('G235H'):
                R = 2700
            elif grating.startswith('G395H'):
                R = 2700
            elif grating.startswith('G140M'):
                R = 1000
            elif grating.startswith('G235M'):
                R = 1000
            elif grating.startswith('G395M'):
                R = 1000
            else:
                print(f"[DEBUG] Unknown NIRSpec grating: {grating}")
                return None

            # R = λ / Δλ, so Δλ = λ / R
            delta_lambda = wavelength_um / R
            # Convert FWHM to sigma: sigma = FWHM / 2.355
            sigma_ref = delta_lambda / 2.355
            return sigma_ref

        print(f"[DEBUG] Could not match instrument: INSTRUME={instrume}, GRATING={grating}")
        return None

    def fit_gaussian(self):
        """
        Fit Gaussian to spectrum and calculate upper limit flux using noise region std.
        This is the old behavior: fits a Gaussian (even to noise), uses fitted sigma for upper limit.
        """
        if self.spectrum is None or self.wavelength is None:
            QMessageBox.warning(self, "Error", "No spectrum loaded")
            return
        if self.sigma_noise is None:
            QMessageBox.warning(self, "Error", "Noise region not selected or calculated")
            return
        left = self.gauss_left_x
        right = self.gauss_right_x
        peak = self.line_data['wavelength']
        if not (left < peak < right):
            QMessageBox.warning(self, "Error", f"Peak ({peak:.6f}) must be between left ({left:.6f}) and right ({right:.6f})")
            return
        try:
            # Extract spectrum region
            mask = (self.wavelength >= left) & (self.wavelength <= right)
            x_fit = self.wavelength[mask]
            y_fit = self.spectrum[mask]
            if len(x_fit) < 5:
                QMessageBox.warning(self, "Error", "Region too small for fitting")
                return
            from .gaussian_fitter import calculate_baseline, estimate_gaussian_params, gaussian
            from scipy.optimize import curve_fit
            y1 = y_fit[0]
            y2 = y_fit[-1]
            baseline = calculate_baseline(x_fit[0], y1, x_fit[-1], y2, x_fit, y_fit)
            y_corrected = y_fit - baseline
            amp_guess, center_guess, sigma_guess = estimate_gaussian_params(x_fit, y_corrected)
            bounds = (
                [0, x_fit[0], 0.001 * (x_fit[-1] - x_fit[0])],
                [10 * np.max(y_corrected), x_fit[-1], 2 * (x_fit[-1] - x_fit[0])]
            )
            amp_guess = np.clip(amp_guess, bounds[0][0], bounds[1][0])
            center_guess = np.clip(center_guess, bounds[0][1], bounds[1][1])
            sigma_guess = np.clip(sigma_guess, bounds[0][2], bounds[1][2])
            popt, pcov = curve_fit(
                gaussian, x_fit, y_corrected,
                p0=[amp_guess, center_guess, sigma_guess],
                bounds=bounds,
                maxfev=10000
            )
            amplitude_fit, center_fit, sigma_fit = popt
            # --- USE INSTRUMENT SIGMA FOR UPPER LIMIT ---
            sigma_instr = self._get_spectral_resolution_sigma(peak)

            if sigma_instr is None:
                # fallback to fitted sigma if instrument unavailable
                print("[DEBUG] Using fitted sigma (no instrument sigma available)")
                sigma_use = sigma_fit
            else:
                sigma_use = sigma_instr

            amplitude_3sigma = 3.0 * self.sigma_noise
            flux_ul_native = amplitude_3sigma * np.sqrt(2 * np.pi) * sigma_use

            flux_ul_cgs = None
            
            if self.fits_header is not None:
                from .gaussian_fitter import convert_area_to_cgs
                try:
                    flux_ul_cgs = convert_area_to_cgs(
                        amplitude_mjy_sr=amplitude_3sigma,
                        sigma_val=sigma_use,
                        center_val=peak,
                        header=self.fits_header,
                        wave_unit='µm',
                        flux_unit_input=self.fits_header.get('BUNIT', None)
                    )
                except Exception as e:
                    print(f"[DEBUG] Could not convert to CGS: {e}")
                    flux_ul_cgs = None
            self.flux_ul = flux_ul_cgs if flux_ul_cgs is not None else flux_ul_native
            result_parts = [
                f"Amplitude (fitted): {amplitude_fit:.4e}",
                f"σ (fitted): {sigma_use:.6f} μm",
                f"σ_noise (measured): {self.sigma_noise:.6e} MJy/sr"
            ]
            if flux_ul_cgs is not None:
                result_parts.append(f"3σ UL (CGS): {flux_ul_cgs:.4e} erg/s/cm²")
            result_text = " | ".join(result_parts)
            self.result_display.setText(result_text)
            self.result_display.setStyleSheet("background-color: #90EE90; font-weight: bold;")
            self.ok_btn.setEnabled(True)
            from scipy.interpolate import interp1d
            self.ax.clear()
            try:
                if len(self.wavelength) > 10:
                    f = interp1d(self.wavelength, self.spectrum, kind='cubic', bounds_error=False, fill_value='extrapolate')
                    wl_smooth = np.linspace(self.wavelength[0], self.wavelength[-1], len(self.wavelength) * 3)
                    spectrum_smooth = f(wl_smooth)
                    self.ax.plot(wl_smooth, spectrum_smooth, 'b-', linewidth=1.2, label='Spectrum', alpha=0.7)
            except:
                self.ax.plot(self.wavelength, self.spectrum, 'b-', linewidth=1, alpha=0.7)
        except Exception as e:
            QMessageBox.critical(self, "Fit Error", f"Failed to fit Gaussian:\n{str(e)}")

    
    def get_flux_ul(self):
        """Return calculated upper limit flux"""
        return self.flux_ul
