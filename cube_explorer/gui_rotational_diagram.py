"""
Rotational Diagram Analysis GUI for Molecular data (H2, CO) - SIMPLIFIED
Single unified interface for selecting molecules, lines, and entering flux data
"""

import os
import csv
import numpy as np

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

from .data_loader import CubeData
from .line_catalog import fetch_hitran_data, load_contaminating_lines_catalog
from .rotational_diagram_fitter import rotational_diagram, get_gu, compute_line_fluxes_collective, model_1T, model_2T
from .co_flux_extractor import _line_sigma


# Constants
AU = 1.496e13  # cm
PC = 3.086e18  # cm

# Isotopologue ranges per molecule (HITRAN numbering)
MOLECULE_ISOTOPOLOGUES = {
    'H2': {'labels': ['H₂', 'HD', 'D₂'], 'range': (1, 3)},
    'CO': {'labels': ['¹²C¹⁶O', '¹³C¹⁶O', '¹²C¹⁸O', '¹²C¹⁷O', '¹³C¹⁸O'], 'range': (1, 5)},
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


class RotationalFitWorker(QThread):
    """Background thread for the rotational-diagram fit itself."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str, int, int)  # (phase "1T"/"2T", current, total)

    def __init__(self, fluxes, flux_errors, molecular_data, fit_opr, opr_grid,
                 fit2T, R, dist, upper_limits, fit1T=None):
        super().__init__()
        self.fluxes = fluxes
        self.flux_errors = flux_errors
        self.molecular_data = molecular_data
        self.fit_opr = fit_opr
        self.opr_grid = opr_grid
        self.fit2T = fit2T
        # Default: report 1T only when 2T wasn't requested -- the "1T"/"2T" dropdown is
        # mutually exclusive, so asking for 2T means 1T's own result isn't wanted (its
        # cheap curve_fit seed is still used internally either way, just not reported).
        self.fit1T = (not fit2T) if fit1T is None else fit1T
        self.R = R
        self.dist = dist
        self.upper_limits = upper_limits

    def run(self):
        try:
            results = rotational_diagram(
                self.fluxes,
                self.flux_errors,
                molecular_data=self.molecular_data,
                OPR=None,
                fit_opr=self.fit_opr,
                opr_grid=self.opr_grid,
                fit1T=self.fit1T,
                fit2T=self.fit2T,
                R=self.R,
                dist=self.dist,
                upper_limits=self.upper_limits,
                progress_callback=self._emit_progress,
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

    def _emit_progress(self, phase, current, total):
        self.progress.emit(phase, current, total)


class RotationalDiagramWindow(QMainWindow):
    """Main window for rotational diagram analysis"""
    
    def __init__(self, parent_gui=None):  # parent_gui kept for backward compatibility, not used
        super().__init__()
        self.current_molecule = 'H2'  # Default molecule
        self.setWindowTitle(f"{self.current_molecule} Rotational Diagram Analyzer")
        self.setGeometry(100, 100, 1500, 800)
        
        self.hitran_lines = []
        self.fetch_thread = None

        # Flags to prevent recursive updates
        self.updating_error_columns = False
        self.updating_flux_flag = False
        self.updating_manual_stats = False
        self.updating_flux_error_flag = False

        # Loaded once and reused to calculate fluxes
        self.loaded_batch_fits_paths = []
        self.loaded_batch_cubes = []
        self.loaded_batch_spectra = []
        self.contam_lines = None
        # row -> {'count', 'closest_wl'} for real HITRAN water lines found near that
        # target line -- informational only (see toggle_hitran_water_lines), never
        # fed into contam_lines or the fit itself.
        self._water_warnings = {}

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
        self.lines_table.setColumnCount(16)
        self.lines_table.setHorizontalHeaderLabels(
            ["✓", "λ (μm)", "Transition", "A_ul (s⁻¹)", "E_u (K)", "g_u", "Flux", "Flux Error",
             "Error %", "Upper Limit?", "UL Flux (3σ)", "Calculated?", "Source", "Fit R²", "Fit",
             "H2O Nearby"]
        )
        # row -> plot data dict (x_data, y_data, baseline, y_fit, center, fwhm, r_squared,
        # wavelength) for the "Fit" column's link, populated by calculate_fluxes_batch.
        self._line_fit_plot_data = {}
        # Resize cells given content; padding
        self.lines_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.lines_table.setStyleSheet(
            "QTableWidget::item { padding: 2px 6px; } "
            "QHeaderView::section { padding: 2px 6px; }"
        )
        main_layout.addWidget(self.lines_table)
        
        # Connect table changes for auto-calculation
        self.lines_table.itemChanged.connect(self.on_table_item_changed)

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

        # Advanced OPR grid controls with script defaults
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
        self.calculate_fluxes_btn.setVisible(False)  # Hidden until lines are loaded
        button_layout.addWidget(self.calculate_fluxes_btn)

        self.load_fits_btn = QPushButton("Load FITS Files")
        self.load_fits_btn.setToolTip(
            "Select the source FITS cube(s) for Calculate Fluxes. Loaded once and reused "
            "on every click, instead of re-prompting each time."
        )
        self.load_fits_btn.clicked.connect(self.load_batch_fits_files)
        self.load_fits_btn.setVisible(False)  # Hidden until lines are loaded
        button_layout.addWidget(self.load_fits_btn)

        self.load_contam_btn = QPushButton("Load Contaminating Lines")
        self.load_contam_btn.setToolTip(
            "Optional: a catalog of nearby non-target lines, so Calculate Fluxes can keep "
            "them from leaking into a requested line's amplitude. Same catalog format as "
            "the main Gaussian-fit window's contaminating-lines loader."
        )
        self.load_contam_btn.clicked.connect(self.load_contaminating_lines_for_batch)
        self.load_contam_btn.setVisible(False)  # Hidden until lines are loaded
        button_layout.addWidget(self.load_contam_btn)

        self.water_lines_checkbox = QCheckBox("Flag Nearby HITRAN Water Lines")
        self.water_lines_checkbox.setToolTip(
            "H2O is the dominant contaminant for H2 rotational lines in this wavelength "
            "range. Checks each target line against real H2O lines from HITRAN (cached, "
            "same as H2/CO) and flags it in the 'H2O Nearby' column if any are close "
            "enough to matter. Informational only -- this does NOT correct for the "
            "contamination or change any computed flux. If a line is flagged and you "
            "need it corrected, provide a pre-decontaminated spectrum."
        )
        self.water_lines_checkbox.stateChanged.connect(self.toggle_hitran_water_lines)
        self.water_lines_checkbox.setVisible(False)  # Hidden until lines are loaded
        button_layout.addWidget(self.water_lines_checkbox)

        self.contam_status_label = QLabel("")  # Only ever shows text once something is actually loaded
        self.contam_status_label.setVisible(False)  # Hidden until lines are loaded
        button_layout.addWidget(self.contam_status_label)

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
        """Update which columns are editable based on error mode."""
        is_value_mode = "Values" in self.error_mode.currentText()

        for row in range(self.lines_table.rowCount()):
            ul_checkbox = self.lines_table.cellWidget(row, 9)
            if ul_checkbox and ul_checkbox.isChecked():
                continue

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

        if col == 10:
            self.sync_calculated_flag(row)

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
        """For a row with no HITRAN match, write manually-typed A_ul/E_u/g_u.
        Enable that row's fit checkbox only once all three are present and physically valid.
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
        """Flag the Flux cell red while empty; clear otherwise."""
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
        """Reset Calculated?/Source to unset by default when edited."""
        calc_checkbox = self.lines_table.cellWidget(row, 11)
        if calc_checkbox:
            calc_checkbox.setChecked(False)
        source_item = self.lines_table.item(row, 12)
        if source_item:
            source_item.setText("")

    def sync_flux_error_flag(self, row):
        """Flag the Flux Error cell red when blank and the line isn't marked as an Upper Limit."""
        if self.updating_flux_error_flag:
            return
        self.updating_flux_error_flag = True
        try:
            error_item = self.lines_table.item(row, 7)
            pct_item = self.lines_table.item(row, 8)
            if not error_item or not pct_item:
                return
            ul_checkbox = self.lines_table.cellWidget(row, 9)
            is_upper_limit = bool(ul_checkbox and ul_checkbox.isChecked())

            if is_upper_limit:
                for item in (error_item, pct_item):
                    text = item.text().strip()
                    if text:
                        item.setData(Qt.UserRole, text)
                    item.setText("")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setBackground(QBrush())
                    item.setToolTip(
                        "Not used for upper limits -- ignored in the fit regardless "
                        "of what's entered here."
                    )
                return

            # Not an upper limit: restore whatever was stashed (if anything).
            for item in (error_item, pct_item):
                stashed = item.data(Qt.UserRole)
                if stashed:
                    item.setText(stashed)
                    item.setData(Qt.UserRole, None)
                item.setToolTip("")

            if error_item.text().strip():
                error_item.setBackground(QBrush())
            else:
                error_item.setBackground(QColor(255, 214, 214))
                error_item.setToolTip(
                    "No flux error entered -- this detection will be excluded from "
                    "the fit unless you enter one or mark it as an Upper Limit."
                )

            # Let update_error_columns() decide which of Flux Error / Error % is
            # editable based on the current Values/Percentages mode.
            self.update_error_columns()
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
        
        # Set workflow mode to HITRAN. Fetch/Import stay both enabled
        self.workflow_mode = 'hitran'
        self.calculate_fluxes_btn.setVisible(True)
        self.load_fits_btn.setVisible(True)
        self.load_contam_btn.setVisible(True)
        self.water_lines_checkbox.setVisible(True)
        self.contam_status_label.setVisible(True)

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
            # Read CSV file
            rows = []
            csv_data = []
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    QMessageBox.warning(self, "Error", "CSV file is empty")
                    return
                
                # Check for required columns (case-insensitive). Flux is optional.
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
                        species = (row_dict.get(species_col) or '').strip().upper()
                        wavelength = float((row_dict.get(wavelength_col) or '').strip())

                        row_info = {
                            'species': species,
                            'wavelength': wavelength,
                        }

                        # Flux is best-effort
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
                    # No matching HITRAN line, so we never assume a value for them.
                    # The row is flagged in the table and disabled until values are input.
                    row_data = {
                        'wavelength': wavelength,
                        'hitran_matched': False,
                        'transition': 'NA',
                        'species': species
                    }
                    print(f"Warning: No HITRAN match for {species} at {wavelength:.6f} um "
                          f"-- A_ul/E_u/g_u left blank; this line is excluded from fitting "
                          f"until they're entered manually.")

                # Flux is optional in the CSV
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

            # Set workflow mode to CSV. Fetch/Import stay both enabled.
            self.workflow_mode = 'csv'
            self.calculate_fluxes_btn.setVisible(True)
            self.load_fits_btn.setVisible(True)
            self.load_contam_btn.setVisible(True)
            self.water_lines_checkbox.setVisible(True)
            self.contam_status_label.setVisible(True)

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
    
    def load_batch_fits_files(self):
        """Select and cache the FITS cube(s) Calculate Fluxes will use."""
        fits_paths, _ = QFileDialog.getOpenFileNames(
            self, "Load FITS Files For Flux Calculation", "",
            "FITS Files (*.fits *.fit);;All Files (*.*)"
        )
        if not fits_paths:
            return

        loaded_paths, loaded_cubes, loaded_spectra, failed = [], [], [], []
        for fits_path in fits_paths:
            try:
                cube = CubeData()
                cube.load_from_file(fits_path)
                summed = np.nansum(cube.cube, axis=(1, 2)) if cube.cube.ndim == 3 else cube.cube
                loaded_paths.append(fits_path)
                loaded_cubes.append(cube)
                loaded_spectra.append(summed)
            except Exception as e:
                failed.append((fits_path, str(e)))

        self.loaded_batch_fits_paths = loaded_paths
        self.loaded_batch_cubes = loaded_cubes
        self.loaded_batch_spectra = loaded_spectra

        # Show loaded files
        names = "\n".join(os.path.basename(p) for p in loaded_paths)
        self.load_fits_btn.setToolTip(names if names else "Select the source FITS cube(s) for Calculate Fluxes.")

        if failed:
            failed_msg = "\n".join(f"  {os.path.basename(p)}: {err}" for p, err in failed)
            QMessageBox.warning(
                self, "Some Files Failed To Load",
                f"{len(failed)} file(s) could not be loaded:\n\n{failed_msg}"
            )
        elif loaded_paths:
            QMessageBox.information(self, "FITS Files Loaded", f"Loaded {len(loaded_paths)} file(s):\n\n{names}")

    def load_contaminating_lines_for_batch(self):
        """Lines can span several different FITS files/ranges at once. Lines already filtered
        per-group to whatever falls in that group's own fit window."""

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Contaminating Lines CSV", "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if not file_path:
            return

        try:
            loaded = load_contaminating_lines_catalog(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"Failed to load contaminating lines:\n{e}")
            self.contam_lines = None
            self.contam_status_label.setText("")
            return

        if loaded is None or len(loaded) == 0:
            self.contam_lines = None
            self.contam_status_label.setText("")
            QMessageBox.warning(self, "Empty File", "The CSV file is empty or has no valid lines.")
            return

        self.contam_lines = loaded
        self.contam_status_label.setText(f"{len(loaded)} contaminating line(s) loaded")
        self.contam_status_label.setToolTip(os.path.basename(file_path))

    # Same convention fit_hitran_lines_collective's own sigma_multiplier default
    # uses to decide which lines share a fit chunk (1-sigma core + 3-sigma margin
    # each side, from Pascucci's own fit_Glines bounds -- see co_flux_extractor.py).
    WATER_LINE_SIGMA_WINDOW = 7.0

    def toggle_hitran_water_lines(self, state):
        """Flag (not correct for) nearby real water contamination -- per Pascucci's
        original ask: warn the user, don't silently try to fix it. Fetches real H2O
        lines from HITRAN (cached, same mechanism as H2/CO) and checks each target
        line for real water lines within WATER_LINE_SIGMA_WINDOW, populating the
        "H2O Nearby" column. Purely informational: nothing here is added to
        contam_lines or fit -- an earlier version tried folding detected water
        lines into the simultaneous multi-Gaussian fit the same way H2/CO
        contaminants are handled, but water is dense enough (~2300 real lines in a
        single ~1um MIRI sub-band) that even after this same proximity filter,
        target lines ended up with 5-30+ extra Gaussians to fit simultaneously --
        far more free parameters than MIRI's sparse per-line sampling (~6-8 points)
        can constrain, so every fit failed outright. Flagging what's nearby is
        useful and correct; trying to fit around it with this app's existing
        contamination mechanism is not, at least for now.
        """
        if state != Qt.Checked:
            self._water_warnings = {}
            self._update_water_warning_column()
            return

        if not self.loaded_batch_cubes:
            # No FITS loaded means no header, and _line_sigma requires a real header
            # to compute the instrumental resolution the proximity check needs.
            QMessageBox.warning(
                self, "No FITS Files Loaded",
                "Load FITS files first (Load FITS Files) -- checking for nearby water "
                "lines needs each file's real spectral resolution."
            )
            self.water_lines_checkbox.blockSignals(True)
            self.water_lines_checkbox.setChecked(False)
            self.water_lines_checkbox.blockSignals(False)
            return

        try:
            warnings = {}
            for cube in self.loaded_batch_cubes:
                wave = cube.wavelength.value
                wl_min, wl_max = float(np.min(wave)), float(np.max(wave))
                file_rows, _ = fetch_hitran_data(wl_min, wl_max, 'H2O', isotopologue=1)
                header = cube.header
                for row, line in enumerate(self.hitran_lines):
                    target = line['wavelength']
                    if not (wl_min <= target <= wl_max):
                        continue
                    try:
                        sigma = _line_sigma(target, header)
                    except ValueError:
                        continue
                    nearby = [r for r in file_rows
                              if abs(r['wavelength'] - target) <= self.WATER_LINE_SIGMA_WINDOW * sigma]
                    if nearby:
                        closest = min(nearby, key=lambda r: abs(r['wavelength'] - target))
                        warnings[row] = {'count': len(nearby), 'closest_wl': closest['wavelength']}
        except Exception as e:
            QMessageBox.critical(self, "HITRAN Fetch Failed", f"Could not fetch H2O lines from HITRAN:\n{e}")
            self.water_lines_checkbox.blockSignals(True)
            self.water_lines_checkbox.setChecked(False)
            self.water_lines_checkbox.blockSignals(False)
            return

        self._water_warnings = warnings
        self._update_water_warning_column()

        if not warnings:
            QMessageBox.information(
                self, "No Nearby Water Lines",
                f"No H2O lines within {self.WATER_LINE_SIGMA_WINDOW}σ of any loaded target line."
            )

    def _update_water_warning_column(self):
        """Reflect self._water_warnings (row -> {count, closest_wl}) into the "H2O
        Nearby" column -- informational only, doesn't touch contam_lines or the fit."""
        for row in range(self.lines_table.rowCount()):
            item = self.lines_table.item(row, 15)
            if item is None:
                continue
            info = self._water_warnings.get(row)
            if info:
                item.setText(f"⚠ {info['count']} (closest {info['closest_wl']:.4f} μm)")
                item.setBackground(QColor(255, 230, 170))
                item.setToolTip(
                    f"{info['count']} real H2O line(s) from HITRAN within "
                    f"{self.WATER_LINE_SIGMA_WINDOW}σ of this line's expected width. "
                    "This flux is not corrected for water contamination -- if you need "
                    "that, provide a pre-decontaminated spectrum."
                )
            else:
                item.setText("")
                item.setBackground(QBrush())
                item.setToolTip("")

    def open_batch_flux_dialog(self):
        """Open dialog for batch flux calculation from multiple FITS files"""
        selected_rows = []
        skipped_rows = []
        for row in range(self.lines_table.rowCount()):
            checkbox = self.lines_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                flux_item = self.lines_table.item(row, 6)
                calc_checkbox = self.lines_table.cellWidget(row, 11)
                was_calculated = bool(calc_checkbox and calc_checkbox.isChecked())
                if flux_item and flux_item.text().strip() and not was_calculated:
                    skipped_rows.append(row)
                else:
                    selected_rows.append(row)

        if not selected_rows:
            if skipped_rows:
                QMessageBox.information(
                    self, "Nothing To Calculate",
                    f"All {len(skipped_rows)} selected line(s) already have a manually-entered "
                    "flux value. Clear a row's Flux cell first if you want to recompute it."
                )
            else:
                QMessageBox.warning(self, "No Lines Selected", "Please select lines to calculate fluxes for.")
            return

        if not self.loaded_batch_cubes:
            QMessageBox.warning(
                self, "No FITS Files Loaded",
                "Click 'Load FITS Files' first to select the source spectra for these lines."
            )
            return

        # Calculate fluxes using the cached FITS files
        self.calculate_fluxes_batch(selected_rows, skipped_rows)

    def calculate_fluxes_batch(self, selected_rows, skipped_rows=None):
        """Lines sharing a covering FITS file are fit together,
        simultaneously, via compute_line_fluxes_collective."""

        loaded_cubes = self.loaded_batch_cubes
        summed_spectra = self.loaded_batch_spectra

        progress = QProgressDialog(
            "Calculating fluxes...",
            None, 0, len(selected_rows), self
        )
        progress.setWindowModality(Qt.WindowModal)

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
            progress.setValue(group_idx)
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
                    wavelength, spectrum, line_wavelengths, header,
                    contam_lines=self.contam_lines,
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

                    # Flux always gets the real fitted value now [UL Flux (col 10) carries the ceiling instead].
                    flux_item = self.lines_table.item(row, 6)
                    if flux_item:
                        flux_item.setText(f"{flux_integrated:.4e}")

                    # Error is always computed now (real curve_fit-covariance error,
                    # regardless of the upper-limit call) -- None only if it was genuinely
                    # uncomputable, not because this line happens to be a upper limit. It
                    # still gets written here either way; sync_flux_error_flag() below is
                    # what actually stashes-and-blanks it for a checked Upper Limit row (and
                    # would restore it if you later uncheck that box), so overriding the
                    # auto-classification always has a real error ready, not nothing.
                    error_item = self.lines_table.item(row, 7)
                    if error_item:
                        error_item.setText(f"{flux_error:.4e}" if flux_error is not None else "")

                    pct_item = self.lines_table.item(row, 8)
                    if pct_item:
                        if flux_error is not None and flux_integrated > 0:
                            pct = (flux_error / flux_integrated) * 100
                            pct_item.setText(f"{pct:.1f}")
                        else:
                            pct_item.setText("")

                    # UL Flux: always filled in, detection or not
                    ul_flux_item = self.lines_table.item(row, 10)
                    if ul_flux_item:
                        ul_flux_item.setText(f"{F_ul_cgs:.4e}")

                    # Auto-check upper limit if detected. Signals blocked so this doesn't
                    # trigger on_line_selected() mid-batch-populate; sync_flux_error_flag()
                    # is called explicitly right after so the stash-and-blank behavior still
                    # applies to freshly-calculated rows, not just later manual toggles.
                    ul_checkbox = self.lines_table.cellWidget(row, 9)
                    if ul_checkbox:
                        ul_checkbox.blockSignals(True)
                        ul_checkbox.setChecked(is_upper_limit)
                        ul_checkbox.blockSignals(False)
                        self.sync_flux_error_flag(row)

                    # Mark this row as system-calculated and record its source
                    calc_checkbox = self.lines_table.cellWidget(row, 11)
                    if calc_checkbox:
                        calc_checkbox.setChecked(True)
                    source_item = self.lines_table.item(row, 12)
                    if source_item:
                        filename = os.path.basename(cube_data.filename) if cube_data.filename else "?"
                        source_item.setText(f"{instrument} [{filename}]")

                    # Fit quality (R^2 of the Gaussian fit this flux came from)
                    r2_item = self.lines_table.item(row, 13)
                    if r2_item:
                        r_squared = result.get("r_squared")
                        if r_squared is None:
                            r2_item.setText("")
                            r2_item.setBackground(QBrush())
                        else:
                            r2_item.setText(f"{r_squared:.3f}")
                            # Same 0.5 threshold gaussian_fitter.py itself uses to call a fit
                            # "poor" vs "good" -- flag it here too rather than a silent number.
                            r2_item.setBackground(QColor(255, 214, 214) if r_squared < 0.5 else QBrush())

                    # Stash this line's fit plot data and turn the Fit cell into a real
                    # link -- only once real data exists for it, so it's never a dead
                    # link pointing at nothing.
                    if all(result.get(k) is not None for k in ("x_data", "y_data", "baseline", "y_fit")):
                        self._line_fit_plot_data[row] = {
                            "wavelength": wl,
                            "x_data": result["x_data"],
                            "y_data": result["y_data"],
                            "baseline": result["baseline"],
                            "y_fit": result["y_fit"],
                            "center": result.get("center"),
                            "fwhm": result.get("fwhm"),
                            "r_squared": result.get("r_squared"),
                            "amplitude": result.get("amplitude"),
                            "flux_cgs": result.get("flux_cgs"),
                            "flux_error_cgs": result.get("flux_error_cgs"),
                            "is_upper_limit": result.get("is_upper_limit"),
                            # Full spectrum this line's fit came from (already loaded in
                            # this window independently of the main window) -- lets the
                            # fit dialog show real spectral context around the line
                            # instead of being confined to the fit's own narrow window.
                            "full_wavelength": wavelength,
                            "full_spectrum": spectrum,
                            "bunit": header.get('BUNIT', '') if header is not None else '',
                        }
                        fit_label = self.lines_table.cellWidget(row, 14)
                        if isinstance(fit_label, QLabel):
                            fit_label.setText('<a href="#">View</a>')
                            fit_label.setTextFormat(Qt.RichText)
                            fit_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
                            fit_label.setOpenExternalLinks(False)
                            fit_label.setAlignment(Qt.AlignCenter)
                            try:
                                fit_label.linkActivated.disconnect()
                            except TypeError:
                                pass  # nothing was connected yet
                            fit_label.linkActivated.connect(lambda _url, r=row: self.show_line_fit(r))

            except Exception as e:
                print(f"Error processing lines from cube {cube_idx}: {e}")
                QMessageBox.critical(self, "Calculation Error", f"Error processing lines from FITS file:\n{e}")

        progress.close()
        msg = "Flux calculation complete!"
        if skipped_rows:
            msg += f"\n\n{len(skipped_rows)} selected line(s) already had a manually-entered flux and were left unchanged."
        QMessageBox.information(self, "Complete", msg)

    def show_line_fit(self, row):
        """Show the Gaussian fit for one line: spectrum, baseline, fit curve, and the
        shaded area under the curve -- same visual style as the main window's
        Multi-Gaussian fit display (visualization.py's plot_gaussian_fit). The
        background spectrum trace comes from the full wavelength/spectrum arrays this
        window already loaded independently (not the main window's, which may be
        showing an unrelated cube) -- gives real spectral context around the line
        instead of being confined to the fit's own narrow +/-3*sigma window.
        """
        data = self._line_fit_plot_data.get(row)
        if data is None:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Line Fit @ {data['wavelength']:.6f} μm")
        dlg.resize(800, 550)
        layout = QVBoxLayout()

        info_parts = [f"<b>λ = {data['wavelength']:.6f} μm</b>"]
        if data.get('flux_cgs') is not None:
            flux_str = f"Flux = {data['flux_cgs']:.3e}"
            if data.get('flux_error_cgs') is not None:
                flux_str += f" ± {data['flux_error_cgs']:.3e}"
            flux_str += " erg/s/cm²"
            if data.get('is_upper_limit'):
                flux_str += " (upper limit)"
            info_parts.append(flux_str)
        if data.get('amplitude') is not None:
            bunit = data.get('bunit')
            info_parts.append(f"Amplitude = {data['amplitude']:.3e}" + (f" {bunit}" if bunit else ""))
        if data.get('r_squared') is not None:
            info_parts.append(f"R² = {data['r_squared']:.3f}")
        if data.get('fwhm') is not None:
            info_parts.append(f"FWHM = {data['fwhm']:.4g} μm")
        info = QLabel(" | ".join(info_parts))
        info.setStyleSheet("background-color: #f0f0f0; padding: 4px; border-radius: 3px;")
        layout.addWidget(info)

        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)

        x = np.asarray(data['x_data'])
        baseline = np.asarray(data['baseline'])
        # y_fit comes from fit_hitran_lines_collective, which fits against the
        # continuum-SUBTRACTED spectrum (y_corrected = spectrum_y - continuum) -- so
        # it's on a small-residual scale, while `baseline` is the separately computed
        # absolute continuum. Adding baseline back puts it on the same absolute flux
        # scale as the spectrum trace below; without this the two end up on
        # unrelated scales entirely (see the earlier version of this dialog).
        y_fit = np.asarray(data['y_fit']) + baseline

        # Wider context: pad by a fixed number of the spectrum's own native sample
        # spacing rather than a hardcoded wavelength range, so this scales sensibly
        # across instruments (MIRI vs. NIRSpec have very different um/pixel scales).
        full_wave = data.get('full_wavelength')
        full_spec = data.get('full_spectrum')
        center = data.get('center') if data.get('center') is not None else float(np.mean(x))
        if full_wave is not None and full_spec is not None and len(full_wave) > 1:
            full_wave = np.asarray(full_wave)
            full_spec = np.asarray(full_spec)
            dwave = float(np.median(np.diff(full_wave)))
            fwhm = data.get('fwhm')
            pad = max(40 * abs(dwave), 8 * fwhm if fwhm else 0)
            xlo, xhi = center - pad, center + pad
            visible = (full_wave >= xlo) & (full_wave <= xhi)
            ax.step(full_wave[visible], full_spec[visible], where='mid', color='steelblue',
                    linewidth=1.2, alpha=0.8, label='Spectrum', zorder=5)
            ax.set_xlim(xlo, xhi)
        elif len(x) > 0 and x.max() > x.min():
            pad = 0.5 * (x.max() - x.min())
            ax.set_xlim(x.min() - pad, x.max() + pad)

        ax.plot(x, baseline, 'k--', linewidth=2, alpha=0.7, label='Baseline', zorder=9)
        ax.plot(x, y_fit, 'r-', linewidth=3, alpha=0.9, label='Gaussian Fit', zorder=10)
        ax.fill_between(x, baseline, y_fit, color='red', alpha=0.2, label='Integrated Area', zorder=7)

        # Disable matplotlib's default "+1.703e1"-style offset notation -- with a
        # narrow x-range far from zero it otherwise shows tick labels like "0.000"
        # that look like the wavelength axis starts at 0, when it's really an offset
        # from a corner annotation that's easy to miss.
        ax.ticklabel_format(axis='x', useOffset=False, style='plain')

        ax.set_xlabel('Wavelength (μm)')
        bunit = data.get('bunit')
        ax.set_ylabel(f'Flux ({bunit})' if bunit else 'Flux')
        ax.set_title(f"Line @ {data['wavelength']:.6f} μm", fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        canvas.draw()

        toolbar = NavigationToolbar(canvas, dlg)
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(canvas)
        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)
        layout.addWidget(plot_widget)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        dlg.setLayout(layout)
        dlg.exec_()

    def populate_lines_table(self):
        """Populate the unified table with all columns"""
        # Row indices are about to be reassigned to a new set of lines -- old plot
        # data / water warnings keyed by row would otherwise silently point at the
        # wrong line.
        self._line_fit_plot_data = {}
        self._water_warnings = {}
        self.water_lines_checkbox.blockSignals(True)
        self.water_lines_checkbox.setChecked(False)
        self.water_lines_checkbox.blockSignals(False)
        self.lines_table.setRowCount(len(self.hitran_lines))
        
        for row, line in enumerate(self.hitran_lines):
            # A line with no HITRAN match has no real A_ul/E_u/g_u: we never assume a value for them.
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
                # A_ul / E_u / g_u (editable)
                a_item = QTableWidgetItem("")
                e_item = QTableWidgetItem("")
                g_item = QTableWidgetItem("")
                for it in (a_item, e_item, g_item):
                    it.setBackground(unmatched_bg)
                    it.setToolTip(unmatched_tip)

            self.lines_table.setItem(row, 3, a_item)
            self.lines_table.setItem(row, 4, e_item)
            self.lines_table.setItem(row, 5, g_item)
            
            # Flux (editable)
            flux_val = line.get('flux', '')
            flux_item = QTableWidgetItem(f"{flux_val:.4e}" if flux_val else "")
            if not flux_val:
                flux_item.setBackground(unmatched_bg)
                flux_item.setToolTip("Flux is required for the fit -- enter it manually or use Calculate Fluxes.")
            self.lines_table.setItem(row, 6, flux_item)

            # Flux Error (editable)
            self.lines_table.setItem(row, 7, QTableWidgetItem(""))

            # Error % (editable)
            self.lines_table.setItem(row, 8, QTableWidgetItem(""))

            # Upper Limit? (checkbox autosuggested but always editable)
            ul_checkbox = QCheckBox()
            ul_checkbox.setToolTip(
                "Auto-checked when a line's fit falls below the 3σ noise threshold "
                "(via Calculate Fluxes). You can check/uncheck this yourself either way."
            )
            ul_checkbox.stateChanged.connect(lambda _state, r=row: self.sync_flux_error_flag(r))
            self.lines_table.setCellWidget(row, 9, ul_checkbox)

            # UL Flux (editable)
            self.lines_table.setItem(row, 10, QTableWidgetItem(""))

            # Calculated? (read-only)
            calc_checkbox = QCheckBox()
            calc_checkbox.setEnabled(False)
            calc_checkbox.setToolTip(
                "Checked when this line's flux was computed by Calculate Fluxes, rather "
                "than typed in by hand or imported from CSV."
            )
            self.lines_table.setCellWidget(row, 11, calc_checkbox)

            # Source (read-only)
            source_item = QTableWidgetItem("")
            source_item.setFlags(source_item.flags() & ~Qt.ItemIsEditable)
            self.lines_table.setItem(row, 12, source_item)

            # Fit R² (read-only; filled in by Calculate Fluxes, blank for manually-entered flux)
            r2_item = QTableWidgetItem("")
            r2_item.setFlags(r2_item.flags() & ~Qt.ItemIsEditable)
            self.lines_table.setItem(row, 13, r2_item)

            # Fit (link; becomes clickable once Calculate Fluxes populates plot data for this row)
            fit_label = QLabel("—")
            fit_label.setAlignment(Qt.AlignCenter)
            self.lines_table.setCellWidget(row, 14, fit_label)

            # H2O Nearby (read-only; filled in by the "Include HITRAN Water Lines" checkbox)
            water_item = QTableWidgetItem("")
            water_item.setFlags(water_item.flags() & ~Qt.ItemIsEditable)
            self.lines_table.setItem(row, 15, water_item)

            self.sync_flux_error_flag(row)
        
        # Set initial edit/read-only status based on current mode
        self.update_error_columns()
    
    def on_line_selected(self):
        """Enable Run Fit only once >=2 lines are checked"""
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

            # Never fit a line with no real A_ul/E_u/g_u
            if not line.get('hitran_matched', True) and (
                'a_coeff' not in line or 'e_upper' not in line or 'Gp' not in line
            ):
                QMessageBox.warning(
                    self, "Error",
                    f"Row {row+1} has no HITRAN match and is missing A_ul/E_u/g_u. "
                    f"Enter them manually or deselect this line."
                )
                return
            
            # Check if this is an upper limit FIRST.
            # Unchecked: use Flux, the real fitted (or manually-entered) value.
            ul_checkbox = self.lines_table.cellWidget(row, 9)
            is_upper_limit = bool(ul_checkbox and ul_checkbox.isChecked())

            flux_col = 6
            if is_upper_limit:
                ul_flux_text = self.lines_table.item(row, 10).text().strip()
                if ul_flux_text:
                    flux_col = 10

            flux_text = self.lines_table.item(row, flux_col).text().strip()
            if not flux_text:
                QMessageBox.warning(self, "Error", f"Missing flux for row {row+1}")
                return

            try:
                flux = float(flux_text)
            except ValueError:
                QMessageBox.warning(self, "Error", f"Invalid flux at row {row+1}")
                return

            # Get error from either column (whichever is populated).
            # If None, _build_dataset then EXCLUDES this line from the fit entirely if it's not an upper limit.
            error_item = self.lines_table.item(row, 7)
            pct_item = self.lines_table.item(row, 8)
            error_text = error_item.text().strip()
            error_pct_text = pct_item.text().strip()
            if is_upper_limit and not error_text and not error_pct_text:
                # sync_flux_error_flag() blanks these cells and stashes the previous
                # value in Qt.UserRole when Upper Limit is checked (the value is still
                # real, just hidden). Recover it here -- censored_1T/censored_2T use
                # each upper limit's own error as its censoring width (ul_sigma), so
                # this needs to actually reach the fit rather than being discarded.
                error_text = (error_item.data(Qt.UserRole) or "").strip()
                error_pct_text = (pct_item.data(Qt.UserRole) or "").strip()

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
            
            fluxes[j_counter] = flux
            if flux_error is not None:
                flux_errors[j_counter] = flux_error
            if is_upper_limit:
                upper_limits[j_counter] = True
            
            molecular_data['wavelengths'][j_counter] = line['wavelength']
            molecular_data['Aul'][j_counter] = line.get('a_coeff', 0)
            molecular_data['Eu'][j_counter] = line.get('e_upper', 0)

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

        # Runs in a background QThread rather than directly here
        self._fit_molecular_data = molecular_data

        self.rot_fit_progress = QProgressDialog(
            "Running rotational diagram fit...", None, 0, 100, self)
        self.rot_fit_progress.setWindowTitle("Rotational Diagram Fit")
        self.rot_fit_progress.setWindowModality(Qt.NonModal)
        self.rot_fit_progress.setAutoClose(False)
        self.rot_fit_progress.setAutoReset(False)
        self.rot_fit_progress.setValue(0)
        self.rot_fit_progress.show()

        self.run_fit_btn.setEnabled(False)

        self.rot_fit_worker = RotationalFitWorker(
            fluxes, flux_errors, molecular_data, fit_opr, opr_grid,
            fit_2t, radius_au, distance_pc, upper_limits,
        )
        self.rot_fit_worker.finished.connect(self._on_rotational_fit_complete)
        self.rot_fit_worker.error.connect(self._on_rotational_fit_error)
        self.rot_fit_worker.progress.connect(self._on_rotational_fit_progress)
        self.rot_fit_worker.start()

    def _on_rotational_fit_progress(self, phase, current, total):
        """Mirror the worker's nutpie sampling progress into the progress dialog."""
        dlg = getattr(self, 'rot_fit_progress', None)
        if dlg is None:
            return
        # In 2T mode, the "1T" phase is just 2T's own silent seed computation -- label it
        # as such instead of "Running 1T fit", which would misleadingly suggest a 1T
        # result the user didn't ask for and won't see reported. Same progress numbers
        # either way, just different label text.
        worker = getattr(self, 'rot_fit_worker', None)
        if phase == "1T" and worker is not None and getattr(worker, 'fit2T', False):
            label = "Preparing 2T fit"
        else:
            label = f"Running {phase} fit"

        if total <= 0:
            # censored_1T/2T send (0, 0) right before nutpie.compile_pymc_model(),
            # which has no sub-progress hook of its own -- this is usually the
            # dominant chunk of wall time (~20-50s), so show a busy/indeterminate
            # bar instead of leaving the dialog looking stalled on the fit's
            # initial static text for that whole stretch.
            dlg.setRange(0, 0)
            dlg.setLabelText(f"{label}: compiling model (nutpie)...")
            return

        dlg.setRange(0, 100)
        pct = int(100 * current / total)
        dlg.setLabelText(f"{label} (nutpie)...\n{current}/{total} ({pct}%)")
        dlg.setValue(pct)

    def _on_rotational_fit_complete(self, results):
        dlg = getattr(self, 'rot_fit_progress', None)
        if dlg is not None:
            dlg.close()
        self.on_line_selected()  # re-derive from current checkbox state (>=2), don't just force True

        molecular_data = self._fit_molecular_data
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

    def _on_rotational_fit_error(self, message):
        dlg = getattr(self, 'rot_fit_progress', None)
        if dlg is not None:
            dlg.close()
        self.on_line_selected()  # re-derive from current checkbox state (>=2), don't just force True
        QMessageBox.critical(self, "Fitting Error", f"Error during fitting:\n{message}")
    
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
            """Return the raw PyMC diagnostic numbers (divergences, max r_hat) for
            a censored fit"""
            n_div = fit_dict.get("n_divergences")
            if n_div is None:
                return ""
            return f"[PyMC: {n_div} divergence(s), max r_hat = {fit_dict.get('max_r_hat', float('nan')):.3f}]\n"

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

        # results_text (this exact string) is what the GUI text widget below shows
        console_text = (
            results_text
            .replace("±", "+/-")
            .replace("σ", "sigma")
            .replace("≈", "~=")
            .replace("⚠", "WARNING:")
        )
        print(console_text)

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
                y1T = model_1T(E_fit, *popt1)
                ax.plot(E_fit, y1T / np.log(10), '--', linewidth=2, label='1T')

            # 2T: show two component lines AND combined curve
            if popt2 is not None:
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
