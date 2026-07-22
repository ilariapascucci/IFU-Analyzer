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
    QFrame, QComboBox, QDialog, QTextEdit, QFileDialog, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .line_catalog import fetch_hitran_data
from .rotational_diagram_fitter import estimate_noise_local, gaussian_flux_upper_limit, get_spectral_resolution


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
        
        self.calculate_fluxes_btn = QPushButton("Calculate Fluxes")
        self.calculate_fluxes_btn.clicked.connect(self.open_batch_flux_dialog)
        self.calculate_fluxes_btn.setVisible(False)  # Hidden until HITRAN lines fetched
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
        
        # Set workflow mode to HITRAN
        self.workflow_mode = 'hitran'
        self.import_csv_btn.setEnabled(False)  # Disable CSV import
        self.calculate_fluxes_btn.setVisible(True)  # Show Calculate Fluxes button
        
        self.hitran_lines = data
        self.populate_lines_table()
        
        # Disable upper limit checkboxes for HITRAN (we auto-calculate them)
        for row in range(self.lines_table.rowCount()):
            ul_checkbox = self.lines_table.cellWidget(row, 9)
            if ul_checkbox:
                ul_checkbox.setEnabled(False)
    
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
            
            # Set workflow mode to CSV
            self.workflow_mode = 'csv'
            self.fetch_btn.setEnabled(False)  # Disable HITRAN fetch
            self.calculate_fluxes_btn.setVisible(False)  # Hide Calculate Fluxes button
            
            # Disable upper limit checkboxes for CSV (user shouldn't calculate fluxes)
            for row in range(self.lines_table.rowCount()):
                ul_checkbox = self.lines_table.cellWidget(row, 9)
                if ul_checkbox:
                    ul_checkbox.setEnabled(False)
            
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
    
    def open_batch_flux_dialog(self):
        """Open dialog for batch flux calculation from multiple FITS files"""
        # Get selected lines
        selected_lines = []
        selected_rows = []
        for row in range(self.lines_table.rowCount()):
            checkbox = self.lines_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                selected_lines.append(self.hitran_lines[row])
                selected_rows.append(row)
        
        if not selected_lines:
            QMessageBox.warning(self, "No Lines Selected", "Please select lines to calculate fluxes for.")
            return
        
        # Open file dialog for FITS files
        fits_paths, _ = QFileDialog.getOpenFileNames(
            self,
            f"Select FITS Files (for {len(selected_lines)} lines)",
            "",
            "FITS Files (*.fits *.fit);;All Files (*.*)"
        )
        
        if not fits_paths:
            return
        
        # Calculate fluxes using FITS files
        self.calculate_fluxes_batch(selected_rows, fits_paths)
    
    def calculate_fluxes_batch(self, selected_rows, fits_paths):
        """Calculate fluxes for all selected lines using their matched FITS files"""
        from .data_loader import CubeData
        from .rotational_diagram_fitter import compute_line_flux

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

        # --- Process each line ---
        for idx, row in enumerate(selected_rows):
            progress.setValue(len(fits_paths) + idx)
            line = self.hitran_lines[row]
            wl = line['wavelength']

            # Find matching cube by wavelength range (no re-loading)
            cube_data = None
            spectrum = None
            for cube, summed in zip(loaded_cubes, summed_spectra):
                wav = cube.wavelength.value
                if wav[0] <= wl <= wav[-1]:
                    cube_data = cube
                    spectrum = summed
                    break

            if cube_data is None:
                print(f"Warning: No FITS file covers wavelength {wl:.4f} μm for line {row+1}")
                QMessageBox.warning(self, "No FITS Match", f"No FITS file covers {wl:.4f} μm")
                return

            try:
                wavelength = cube_data.wavelength.value
                header = cube_data.header
                instrument = cube_data.instrument

                # Ensure wavelength is increasing
                if wavelength[0] > wavelength[-1]:
                    wavelength = wavelength[::-1]
                    spectrum = spectrum[::-1]

                # Fit the line and compute flux, error, and upper-limit status (CGS)
                result = compute_line_flux(wavelength, spectrum, wl, header, instrument)

                if not result["success"]:
                    print(f"Warning: {result['message']} for line {wl:.4f}")
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

            except Exception as e:
                print(f"Error processing row {row}: {e}")
                QMessageBox.critical(self, "Calculation Error", f"Error processing line {row+1}:\n{e}")

        progress.close()
        QMessageBox.information(self, "Complete", "Flux calculation complete!")
    
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
            
            # Upper Limit? (checkbox - auto-filled by batch calculation, disabled for HITRAN)
            ul_checkbox = QCheckBox()
            self.lines_table.setCellWidget(row, 9, ul_checkbox)
        
        # Set initial edit/read-only status based on current mode
        self.update_error_columns()
    
    def on_cell_clicked(self, row, col):
        """Handle cell clicks (for compatibility)"""
        pass
    
    def on_line_selected(self):
        """Enable Run Fit button if lines selected"""
        has_selection = False
        for row in range(self.lines_table.rowCount()):
            checkbox = self.lines_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                has_selection = True
                break
        self.run_fit_btn.setEnabled(has_selection)
    
    def select_all_lines(self):
        """Select all lines in the table"""
        for row in range(self.lines_table.rowCount()):
            checkbox = self.lines_table.cellWidget(row, 0)
            if checkbox:
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
            results_text += f"T1 = {r2t['T1']:.1f} ± {r2t.get('dT1', float('nan')):.1f} K\n"
            results_text += f"T2 = {r2t['T2']:.1f} ± {r2t.get('dT2', float('nan')):.1f} K\n"
            results_text += f"N1 = {r2t['Ncol1']:.2e} ± {r2t.get('dNcol1', float('nan')):.2e} cm^-2\n"
            results_text += f"N2 = {r2t['Ncol2']:.2e} ± {r2t.get('dNcol2', float('nan')):.2e} cm^-2\n"

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
            y_arr_log10 = np.log10(np.exp(y_arr))
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
                ax.plot(E_fit, np.log10(np.exp(y1T)), '--', linewidth=2, label='1T')

            # 2T: show two component lines AND combined curve
            if popt2 is not None:
                from .rotational_diagram_fitter import model_2T
                a1_2, T1_2, a2_2, T2_2 = popt2

                y_comp1 = a1_2 - E_fit / T1_2
                ax.plot(E_fit, np.log10(np.exp(y_comp1)), ':', linewidth=2.5,
                        color='blue', label=f'Warm: T={T1_2:.0f} K')

                y_comp2 = a2_2 - E_fit / T2_2
                ax.plot(E_fit, np.log10(np.exp(y_comp2)), ':', linewidth=2.5,
                        color='orange', label=f'Hot: T={T2_2:.0f} K')

                y2T = model_2T(E_fit, *popt2)
                ax.plot(E_fit, np.log10(np.exp(y2T)), '-', linewidth=3,
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
    



class UpperLimitFitsDialog(QDialog):
    """Dialog for computing 3-sigma upper limit flux from noise RMS and instrumental line width."""

    def __init__(self, parent, row, line_data):
        super().__init__(parent)
        self.row = row
        self.line_data = line_data
        self.flux_ul = None
        self.sigma_noise = None
        self.fits_file = None
        self.spectrum = None
        self.wavelength = None
        self.fits_header = None
        self.fits_instrument = None
        self.cube_data = None  # Store cube data for noise extraction

        self.setWindowTitle(f"Upper Limit: {line_data['wavelength']:.6f} μm")
        self.setGeometry(100, 100, 600, 200)

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

        # Instructions
        instructions = QLabel(
            f"Line: {line_data['wavelength']:.6f} μm\n"
            "Workflow:\n"
            "1. Load FITS file\n"
            "2. Click 'Calculate Upper Limit' → noise estimated from spatially-summed cube, 13 points each side of line"
        )
        layout.addWidget(instructions)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.fit_btn = QPushButton("Calculate Upper Limit")
        self.fit_btn.clicked.connect(self.fit_gaussian)
        self.fit_btn.setEnabled(False)
        button_layout.addWidget(self.fit_btn)

        layout.addLayout(button_layout)

        # Result display
        self.result_display = QLineEdit()
        self.result_display.setReadOnly(True)
        layout.addWidget(self.result_display)

        # OK / Cancel
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

    
    def load_fits_file(self):
        """Load FITS file and extract spectrum from center pixel (using CubeData loader for consistency)"""
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
            
            cube_data = CubeData()
            cube_data.load_from_file(fits_path)
            
            # Extract center pixel spectrum from cube (for wavelength reference)
            if cube_data.cube.ndim == 3:
                ny, nx = cube_data.cube.shape[1], cube_data.cube.shape[2]
                center_y, center_x = ny // 2, nx // 2
                spectrum = cube_data.cube[:, center_y, center_x]
            elif cube_data.cube.ndim == 1:
                spectrum = cube_data.cube
            else:
                QMessageBox.warning(self, "Error", f"Unsupported cube shape: {cube_data.cube.shape}")
                return
            
            # Get wavelength with units and convert to plain array
            wavelength_quantity = cube_data.wavelength
            if wavelength_quantity is not None:
                wavelength = wavelength_quantity.value

                # Ensure wavelength is increasing
                if wavelength[0] > wavelength[-1]:
                    wavelength = wavelength[::-1]
                    spectrum = spectrum[::-1]
                    
            else:
                QMessageBox.warning(self, "Error", "Could not extract wavelength from FITS")
                return
            
            # Check if spectrum has any valid data
            valid_mask = ~np.isnan(spectrum) & ~np.isinf(spectrum)
            if not np.any(valid_mask):
                QMessageBox.warning(self, "Error", "No valid data in spectrum (all NaN or inf)")
                return
            
            # Store spectrum and wavelength
            self.spectrum = spectrum
            self.wavelength = wavelength
            self.fits_file = fits_path
            self.fits_header = cube_data.header
            self.fits_instrument = cube_data.instrument
            self.cube_data = cube_data
            self.file_label.setText(os.path.basename(fits_path))
            self.fit_btn.setEnabled(True)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load FITS:\n{str(e)}")
    
    
    def fit_gaussian(self):
        """
        Calculate 3-sigma upper limit flux from noise RMS and instrumental line width.

        Uses boss's method: sum all spaxels spatially, then estimate noise from
        13 points on each side of line center with linear continuum subtraction.

        Formula:  F_ul = 3 * sigma_noise * delta_lambda
        where delta_lambda = lambda / R  (instrumental spectral resolution)
        """
        if self.cube_data is None or self.wavelength is None:
            QMessageBox.warning(self, "Error", "No FITS file loaded")
            return

        peak = self.line_data['wavelength']

        # --- Sum cube spatially (same as batch calculation) ---
        if self.cube_data.cube.ndim == 3:
            spectrum_summed = np.nansum(self.cube_data.cube, axis=(1, 2))
        else:
            spectrum_summed = self.cube_data.cube

        # --- Calculate noise on summed spectrum ---
        sigma_noise = estimate_noise_local(self.wavelength, spectrum_summed, peak, n_pixels=13)
        if sigma_noise is None:
            QMessageBox.warning(self, "Error", "Not enough pixels around line to estimate noise")
            return
        self.sigma_noise = sigma_noise

        # --- Instrumental delta_lambda ---
        try:
            R = get_spectral_resolution(peak, instrument=self.fits_instrument, header=self.fits_header)
            delta_lambda = peak / R
        except Exception as e:
            QMessageBox.warning(
                self, "Error",
                f"Could not determine spectral resolution from FITS header.\n"
                f"Check that INSTRUME and GRATING keywords are present.\n{e}"
            )
            return

        # --- Upper limit flux: F_ul = 3 * sigma_noise * delta_lambda ---
        flux_ul_native = 3.0 * sigma_noise * delta_lambda

        flux_ul_cgs = None
        if self.fits_header is not None:
            try:
                from .rotational_diagram_fitter import convert_flux_upper_limit_to_cgs
                flux_ul_cgs = convert_flux_upper_limit_to_cgs(
                    flux_ul_native, peak, self.fits_header, flux_unit='MJy/sr'
                )
            except Exception as e:
                print(f"Could not convert to CGS: {e}")

        self.flux_ul = flux_ul_cgs if flux_ul_cgs is not None else flux_ul_native

        result_parts = [
            f"σ_noise (summed cube, 13+13 pts): {self.sigma_noise:.6e} MJy/sr",
            f"Δλ: {delta_lambda:.6f} μm  (λ/R at {peak:.4f} μm)",
            f"3σ UL (native): {flux_ul_native:.4e} MJy/sr·μm",
        ]
        if flux_ul_cgs is not None:
            result_parts.append(f"3σ UL (CGS): {flux_ul_cgs:.4e} erg/s/cm²")

        self.result_display.setText(" | ".join(result_parts))
        self.result_display.setStyleSheet("background-color: #90EE90; font-weight: bold;")
        self.ok_btn.setEnabled(True)

    
    def get_flux_ul(self):
        """Return calculated upper limit flux"""
        return self.flux_ul