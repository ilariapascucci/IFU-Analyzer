"""
Rotational Diagram Analysis GUI for Molecular data (H2, CO) - SIMPLIFIED
Single unified interface for selecting molecules, lines, and entering flux data
"""

import os
import csv
import time
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
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
from .table_drag_select import SortableCheckboxTableMixin
from .rotational_diagram_h2_fitter import (
    rotational_diagram as rotational_diagram_h2,
    get_gu, compute_line_fluxes_collective, model_1T, model_2T,
)
from .rotational_diagram_general_fitter import rotational_diagram as rotational_diagram_general
from .flux_map_extractor import DISPLAY_CROP_SIGMA_MULTIPLIER, gaussian
from .rotational_diagram_map import convolution_plan, _line_keys, convolve_line_maps
from .rotational_diagram_map_dialogs import (
    RotDiagMapWorker, RotDiagMapResultsDialog, LineFluxMapDialog,
    CalculateNativeMapsWorker, CombineMapsWorker,
)


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
        """Store the HITRAN query parameters for run() to use.

        Args:
            wl_min (float): Minimum wavelength of the query, in micron.
            wl_max (float): Maximum wavelength of the query, in micron.
            molecule (str): HITRAN molecule name to query.
            isotopologue (int): HITRAN isotopologue number, or None for all.
            v_upper (int): Upper vibrational level filter, or None.
            v_lower (int): Lower vibrational level filter, or None.

        Returns:
            None
        """
        super().__init__()
        self.wl_min = wl_min
        self.wl_max = wl_max
        self.molecule = molecule
        self.isotopologue = isotopologue
        self.v_upper = v_upper
        self.v_lower = v_lower

    def run(self):
        """Fetch HITRAN lines and emit `finished` or `error`.

        Args:
            None

        Returns:
            None
        """
        try:
            hitran_data, fetch_meta = fetch_hitran_data(
                self.wl_min, self.wl_max,
                molecule=self.molecule,
                isotopologue=self.isotopologue,
                v_upper=self.v_upper,
                v_lower=self.v_lower
            )
            # fetch_hitran_data reports failure via fetch_meta, not by raising --
            # route that to .error so a real network failure isn't shown as "no lines found".
            if fetch_meta.get('error'):
                self.error.emit(fetch_meta.get('message', 'HITRAN fetch failed'))
            else:
                self.finished.emit(hitran_data)
        except Exception as e:
            self.error.emit(str(e))


class RotationalFitWorker(QThread):
    """Background thread for the rotational-diagram fit itself."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str, int, int)  # (phase "1T"/"2T", current, total)

    def __init__(self, fluxes, flux_errors, molecular_data, fit_opr, opr_grid,
                 fit2T, R, dist, upper_limits, fit1T=None, molecule='H2', isotopologue=1):
        """Store fit inputs for run() to use.

        Args:
            fluxes (dict): Line fluxes keyed by line identifier.
            flux_errors (dict): Flux uncertainties keyed by line identifier.
            molecular_data (dict): Per-line molecular data (E_upper, g_up, A).
            fit_opr (bool): Whether to fit the ortho/para ratio (H2 only).
            opr_grid (numpy.ndarray): OPR grid to search when fit_opr is set.
            fit2T (bool): Whether to run the two-temperature fit.
            R (float): Source radius in cm, for flux-to-column conversion.
            dist (float): Source distance in cm.
            upper_limits (dict): Upper-limit flags keyed by line identifier.
            fit1T (bool): Whether to report the one-temperature fit.
            molecule (str): Molecule to fit ('H2' or a general-fitter name).
            isotopologue (int): HITRAN isotopologue number for the fit.

        Returns:
            None
        """
        super().__init__()
        self.fluxes = fluxes
        self.flux_errors = flux_errors
        self.molecular_data = molecular_data
        self.fit_opr = fit_opr
        self.opr_grid = opr_grid
        self.fit2T = fit2T
        # Default: report 1T only when 2T wasn't requested -- 1T's cheap
        # curve_fit seed still runs internally either way, just unreported.
        self.fit1T = (not fit2T) if fit1T is None else fit1T
        self.R = R
        self.dist = dist
        self.upper_limits = upper_limits
        self.molecule = molecule
        self.isotopologue = isotopologue

    def run(self):
        """Run the 1T/2T rotational-diagram fit and emit `finished` or `error`.

        Args:
            None

        Returns:
            None
        """
        try:
            # H2 gets its own hardcoded-partition-function/OPR-capable fitter; every
            # other molecule goes through the general, HITRAN-Q(T) one (no OPR param).
            if self.molecule == 'H2':
                results = rotational_diagram_h2(
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
            else:
                results = rotational_diagram_general(
                    self.fluxes,
                    self.flux_errors,
                    molecular_data=self.molecular_data,
                    fit1T=self.fit1T,
                    fit2T=self.fit2T,
                    R=self.R,
                    dist=self.dist,
                    upper_limits=self.upper_limits,
                    progress_callback=self._emit_progress,
                    molecule=self.molecule,
                    isotopologue=self.isotopologue,
                )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

    def _emit_progress(self, phase, current, total):
        """Relay a fit-progress update through the `progress` signal.

        Args:
            phase (str): Fit phase, '1T' or '2T'.
            current (int): Current step within the phase.
            total (int): Total steps in the phase.

        Returns:
            None
        """
        self.progress.emit(phase, current, total)


class BatchFluxWorker(QThread):
    """Background thread for calculate_fluxes_batch's fitting -- Qt widgets
    aren't thread-safe here, so this only computes; _apply_batch_flux_group_results
    applies everything to the table back on the main thread."""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # (FITS-file group index, total groups)

    def __init__(self, rows_by_cube_idx, loaded_cubes, summed_spectra, line_wavelengths, contam_lines):
        """Store batch flux-fit inputs for run() to use.

        Args:
            rows_by_cube_idx (dict): Table rows grouped by loaded-cube index.
            loaded_cubes (list): Loaded CubeData objects, one per FITS file.
            summed_spectra (list): Per-cube summed spectrum, aligned by index.
            line_wavelengths (dict): Wavelength keyed by table row.
            contam_lines (pandas.DataFrame): Contaminating lines to exclude,
                or None.

        Returns:
            None
        """
        super().__init__()
        self.rows_by_cube_idx = rows_by_cube_idx
        self.loaded_cubes = loaded_cubes
        self.summed_spectra = summed_spectra
        self.line_wavelengths = line_wavelengths  # row -> wavelength
        self.contam_lines = contam_lines

    def run(self):
        """Fit every row group's line fluxes and emit `finished` or `error`.

        Args:
            None

        Returns:
            None
        """
        print(f"[TIMING] BatchFluxWorker.run() started at {time.time():.2f}")
        try:
            groups = []
            total = len(self.rows_by_cube_idx)
            for group_idx, (cube_idx, rows) in enumerate(self.rows_by_cube_idx.items()):
                self.progress.emit(group_idx, total)
                cube_data = self.loaded_cubes[cube_idx]
                spectrum = self.summed_spectra[cube_idx]
                try:
                    wavelength = cube_data.wavelength.value
                    header = cube_data.header

                    if wavelength[0] > wavelength[-1]:
                        wavelength = wavelength[::-1]
                        spectrum = spectrum[::-1]

                    line_wls = [self.line_wavelengths[row] for row in rows]
                    t0 = time.time()
                    group_results = compute_line_fluxes_collective(
                        wavelength, spectrum, line_wls, header,
                        contam_lines=self.contam_lines,
                    )
                    print(f"[TIMING] fit for {len(rows)} line(s): {time.time()-t0:.2f}s")
                    groups.append({
                        "success": True, "rows": rows, "results": group_results,
                        "wavelength": wavelength, "spectrum": spectrum,
                        "instrument": cube_data.instrument, "filename": cube_data.filename,
                        "bunit": header.get('BUNIT', '') if header is not None else '',
                    })
                except Exception as e:
                    # One bad FITS file/group shouldn't abort the rest of the batch --
                    # same recovery as the old inline per-group try/except.
                    groups.append({"success": False, "cube_idx": cube_idx, "error": str(e)})
            self.progress.emit(total, total)
            print(f"[TIMING] BatchFluxWorker emitting finished at {time.time():.2f}")
            self.finished.emit(groups)
        except Exception as e:
            self.error.emit(str(e))


class RotationalDiagramWindow(QMainWindow, SortableCheckboxTableMixin):
    """Main window for rotational diagram analysis"""
    
    def __init__(self, parent_gui=None):  # parent_gui kept for backward compatibility, not used
        """Build the rotational-diagram window and initialize its state.

        Args:
            parent_gui: Unused; kept for backward compatibility.

        Returns:
            None
        """
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

        # Current lines_table sort state: which column header was last clicked, and
        # in which direction. None = unsorted (natural fetch/import order).
        self._sort_column = None
        self._sort_order = Qt.AscendingOrder

        # Loaded once and reused to calculate fluxes
        self.loaded_batch_fits_paths = []
        self.loaded_batch_cubes = []
        self.loaded_batch_spectra = []
        self.contam_lines = None

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
        self.lines_table.setColumnCount(17)
        self.lines_table.setHorizontalHeaderLabels(
            ["✓", "λ (μm)", "Transition", "A_ul (s⁻¹)", "E_u (K)", "g_u", "Flux", "Flux Error",
             "Error %", "Upper Limit?", "UL Flux (3σ)", "Calculated?", "Source", "Fit R²", "Fit",
             "Map", "Convolved Map"]
        )
        # row -> plot data dict (x_data, y_data, baseline, y_fit, center, fwhm, r_squared,
        # wavelength) for the "Fit" column's link, populated by calculate_fluxes_batch.
        self._line_fit_plot_data = {}
        # row -> {wavelength, flux_map, flux_error_map, header, molecule} for the
        # "Map" column's link, populated by _populate_map_column after Run Map or
        # Calculate Maps.
        self._line_map_plot_data = {}
        # Same shape as _line_map_plot_data, for the "Convolved Map" column --
        # populated by Convolve Maps or Run Map, multi-cube only.
        self._line_convolved_map_plot_data = {}
        # Resize cells given content; padding
        self.lines_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.lines_table.setStyleSheet(
            "QTableWidget::item { padding: 2px 6px; } "
            "QHeaderView::section { padding: 2px 6px; }"
        )
        main_layout.addWidget(self.lines_table)

        # Click-to-sort for wavelength/A_ul/E_u/g_u/Flux/Flux Error/Error %/UL Flux.
        # Also drag-checks columns 0 (select) and 9 (Upper Limit?) across rows.
        self._init_sortable_checkbox_table(
            self.lines_table, {1, 3, 4, 5, 6, 7, 8, 10}, self._lines_table_sort_key,
            drag_column={0, 9},
        )

        # Connect table changes for auto-calculation
        self.lines_table.itemChanged.connect(self.on_table_item_changed)
        # "Fit" column (14) opens the fit-view dialog on double-click, once
        # Calculate Fluxes has given that row real data to show.
        self.lines_table.itemDoubleClicked.connect(self.on_lines_table_item_double_clicked)

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
            "Grid-search over OPR to minimize chi-squared, instead of assuming the "
            "fixed equilibrium value OPR=3. H2 only -- disabled for other molecules, "
            "since OPR is a nuclear-spin-statistics effect specific to H2."
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

        self.run_map_btn = QPushButton("Run Map")
        self.run_map_btn.setToolTip(
            "Run the same fit at every pixel of the loaded cube(s). Detections "
            "only, 1T. Load cube(s) with the Load FITS Files button.")
        self.run_map_btn.clicked.connect(self.run_rotational_map)
        self.run_map_btn.setEnabled(False)
        bottom_layout.addWidget(self.run_map_btn)

        main_layout.addLayout(bottom_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def create_hitran_section(self):
        """Create HITRAN parameter input section.

        Args:
            None

        Returns:
            frame (QFrame): The assembled HITRAN parameter input panel.
        """
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

        self.save_fluxes_btn = QPushButton("Save Fluxes")
        self.save_fluxes_btn.setToolTip(
            "Save the checked lines' species, wavelength, flux, and flux error to a CSV, "
            "in the same format Import from CSV reads -- so re-selecting these same lines "
            "later doesn't need Calculate Fluxes run again."
        )
        self.save_fluxes_btn.clicked.connect(self.save_fluxes_to_csv)
        self.save_fluxes_btn.setVisible(False)  # Hidden until lines are loaded
        button_layout.addWidget(self.save_fluxes_btn)

        self.calculate_maps_btn = QPushButton("Calculate Maps")
        self.calculate_maps_btn.setToolTip(
            "Fit each checked line's own flux map (native resolution per cube) without "
            "combining them into a temperature fit yet. Populates the Map column so you "
            "can inspect each line before running the full map. Caches its result so a "
            "later Run Map reuses it instead of recomputing."
        )
        self.calculate_maps_btn.clicked.connect(self.calculate_rotational_maps)
        self.calculate_maps_btn.setVisible(False)  # Hidden until lines are loaded
        button_layout.addWidget(self.calculate_maps_btn)

        self.convolve_maps_btn = QPushButton("Convolve Maps")
        self.convolve_maps_btn.setToolTip(
            "Convolve/reproject every checked line's already-calculated map (Calculate "
            "Maps) onto a common grid at the worst (coarsest) resolution among them. "
            "Populates the Convolved Map column. Needs 2+ loaded cubes and Calculate "
            "Maps run first."
        )
        self.convolve_maps_btn.clicked.connect(self.convolve_rotational_maps)
        self.convolve_maps_btn.setVisible(False)  # Hidden until lines are loaded
        button_layout.addWidget(self.convolve_maps_btn)

        self.contam_status_label = QLabel("")  # Only ever shows text once something is actually loaded
        self.contam_status_label.setVisible(False)  # Hidden until lines are loaded
        button_layout.addWidget(self.contam_status_label)

        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        frame.setLayout(layout)
        return frame
    
    def on_molecule_changed(self, molecule):
        """Update UI when the Molecule dropdown itself changes.

        Args:
            molecule (str): Newly selected molecule name.

        Returns:
            None
        """
        self._apply_molecule_change(molecule)
        # A new molecule means different wavelengths/A_ul/E_u -- start clean
        # rather than leave the old molecule's stale table around.
        self._clear_loaded_lines()

    def _apply_molecule_change(self, molecule):
        """Sync window title, isotopologue range, and OPR availability to a
        molecule, without touching any lines already loaded -- the part of a
        molecule switch that's safe to reuse when the lines themselves are
        the reason for the switch (see import_csv_file's auto-sync).

        Args:
            molecule (str): Molecule name to apply.

        Returns:
            None
        """
        self.current_molecule = molecule
        self.setWindowTitle(f"{molecule} Rotational Diagram Analyzer")
        self.update_isotopologue_range()

        # OPR is H2-only physics; block the checkbox for anything else.
        is_h2 = (molecule == 'H2')
        if not is_h2 and self.fit_opr_check.isChecked():
            self.fit_opr_check.setChecked(False)
        self.fit_opr_check.setEnabled(is_h2)

    def _clear_loaded_lines(self):
        """Discard fetched/imported lines and drop the FITS cubes/contaminant
        catalog loaded for the previous molecule -- a molecule switch is a
        new workflow, not more files for the old one.

        Args:
            None

        Returns:
            None
        """
        self.hitran_lines = []
        self.workflow_mode = None
        self.populate_lines_table()
        self.on_line_selected()
        self.calculate_fluxes_btn.setVisible(False)
        self.load_fits_btn.setVisible(False)
        self.load_contam_btn.setVisible(False)
        self.save_fluxes_btn.setVisible(False)
        self.calculate_maps_btn.setVisible(False)
        self.convolve_maps_btn.setVisible(False)
        self.contam_status_label.setVisible(False)

        self.loaded_batch_fits_paths = []
        self.loaded_batch_cubes = []
        self.loaded_batch_spectra = []
        self.load_fits_btn.setToolTip("Select the source FITS cube(s) for Calculate Fluxes.")
        self.contam_lines = None
        self.contam_status_label.setText("")

    def _reset_batch_cubes(self):
        """Drop cubes loaded for Calculate Fluxes / Run Map: a new line list is a fresh slate.

        Args:
            None

        Returns:
            None
        """
        # Loading files again with the same lines still appends (load_batch_fits_files).
        if getattr(self, "loaded_batch_cubes", None):
            print(f"[rot diag] {len(self.loaded_batch_cubes)} loaded cube(s) cleared: line list changed")
        self.loaded_batch_fits_paths = []
        self.loaded_batch_cubes = []
        self.loaded_batch_spectra = []
        self.load_fits_btn.setToolTip("Select the source FITS cube(s) for Calculate Fluxes.")

    def update_isotopologue_range(self):
        """Update isotopologue spinner range based on selected molecule.

        Args:
            None

        Returns:
            None
        """
        mol_data = MOLECULE_ISOTOPOLOGUES.get(self.current_molecule)
        if mol_data:
            min_iso, max_iso = mol_data['range']
            self.isotopologue_spin.setRange(min_iso, max_iso)
            self.isotopologue_spin.setValue(min_iso)
    
    def on_include_all_transitions_toggled(self):
        """Enable/disable the v_upper/v_lower spinners with the checkbox.

        Args:
            None

        Returns:
            None
        """
        checked = self.include_all_trans_check.isChecked()
        self.v_upper_spin.setEnabled(not checked)
        self.v_lower_spin.setEnabled(not checked)

    def on_radius_unknown_toggled(self):
        """Disable the R field when R is marked unknown (R=None is passed to the fitter).

        Args:
            None

        Returns:
            None
        """
        self.radius_input.setEnabled(not self.radius_unknown_check.isChecked())

    def on_fit_opr_toggled(self):
        """Enable the Advanced button only when Fit OPR is checked; collapse it otherwise.

        Args:
            None

        Returns:
            None
        """
        checked = self.fit_opr_check.isChecked()
        self.opr_advanced_btn.setEnabled(checked)
        if not checked and self.opr_advanced_btn.isChecked():
            self.opr_advanced_btn.setChecked(False)
            self.on_opr_advanced_toggled()

    def on_opr_advanced_toggled(self):
        """Show/hide the OPR grid range/resolution controls.

        Args:
            None

        Returns:
            None
        """
        expanded = self.opr_advanced_btn.isChecked()
        self.opr_advanced_frame.setVisible(expanded)
        self.opr_advanced_btn.setText(("▼" if expanded else "▶") + " Advanced")

    def on_opr_min_default_toggled(self):
        """Lock OPR min to its default value while 'Default' is checked.

        Args:
            None

        Returns:
            None
        """
        checked = self.opr_min_default_check.isChecked()
        self.opr_min_input.setEnabled(not checked)
        if checked:
            self.opr_min_input.setText("0.1")

    def on_opr_max_default_toggled(self):
        """Lock OPR max to its default value while 'Default' is checked.

        Args:
            None

        Returns:
            None
        """
        checked = self.opr_max_default_check.isChecked()
        self.opr_max_input.setEnabled(not checked)
        if checked:
            self.opr_max_input.setText("3.0")

    def on_opr_steps_default_toggled(self):
        """Lock OPR steps to its default value while 'Default' is checked.

        Args:
            None

        Returns:
            None
        """
        checked = self.opr_steps_default_check.isChecked()
        self.opr_steps_input.setEnabled(not checked)
        if checked:
            self.opr_steps_input.setText("400")
    
    def on_error_mode_changed(self, mode):
        """Update table edit/read-only based on error input mode.

        Args:
            mode (str): New error-input mode label from the combo box.

        Returns:
            None
        """
        self.update_error_columns()
    
    def update_error_columns(self):
        """Update which columns are editable based on error mode, for every row.

        Args:
            None

        Returns:
            None
        """
        for row in range(self.lines_table.rowCount()):
            self._update_error_columns_for_row(row)

    def _update_error_columns_for_row(self, row):
        """Same as update_error_columns() but for one row -- avoids an O(n) sweep
        when only one row actually changed (see sync_flux_error_flag).

        Args:
            row (int): Table row index to update.

        Returns:
            None
        """
        ul_item = self.lines_table.item(row, 9)
        if ul_item and ul_item.checkState() == Qt.Checked:
            return

        flux_error_item = self.lines_table.item(row, 7)
        error_pct_item = self.lines_table.item(row, 8)
        if not (flux_error_item and error_pct_item):
            return

        is_value_mode = "Values" in self.error_mode.currentText()
        if is_value_mode:
            flux_error_item.setFlags(flux_error_item.flags() | Qt.ItemIsEditable)
            error_pct_item.setFlags(error_pct_item.flags() & ~Qt.ItemIsEditable)
        else:
            flux_error_item.setFlags(flux_error_item.flags() & ~Qt.ItemIsEditable)
            error_pct_item.setFlags(error_pct_item.flags() | Qt.ItemIsEditable)
    
    def on_table_item_changed(self, item):
        """Auto-calculate flux error or percentage; sync manually-entered A_ul/E_u/g_u.

        Args:
            item (QTableWidgetItem): The table cell that changed.

        Returns:
            None
        """
        if self.lines_table.rowCount() == 0:
            return

        row = item.row()
        col = item.column()

        # Columns 0 and 9 are checkable items, not QCheckBox widgets, so their
        # state change routes through this same itemChanged signal.
        if col == 0:
            self.on_line_selected()
            return

        if col == 9:
            self.sync_flux_error_flag(row)
            return

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
        Enable that row's fit checkbox only once all three are present and
        physically valid.

        Args:
            row (int): Table row index to sync.

        Returns:
            None
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
                """Parse a table cell in this row as a float, or None.

                Args:
                    col (int): Column index to read.

                Returns:
                    value (float): The parsed value, or None if blank/invalid.
                """
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

            select_item = self.lines_table.item(row, 0)
            cells = [self.lines_table.item(row, c) for c in (3, 4, 5)]

            if complete:
                line['a_coeff'] = a_val
                line['e_upper'] = e_val
                line['Gp'] = g_val
                for cell in cells:
                    if cell:
                        cell.setBackground(QBrush())
                        cell.setToolTip("")
                if select_item:
                    select_item.setFlags(select_item.flags() | Qt.ItemIsEnabled)
                    select_item.setToolTip("")
            else:
                line.pop('a_coeff', None)
                line.pop('e_upper', None)
                line.pop('Gp', None)
                tip = "No HITRAN match -- enter A_ul, E_u, and g_u manually to include this line."
                for cell in cells:
                    if cell:
                        cell.setBackground(QColor(255, 214, 214))
                        cell.setToolTip(tip)
                if select_item:
                    self.lines_table.blockSignals(True)
                    select_item.setCheckState(Qt.Unchecked)
                    self.lines_table.blockSignals(False)
                    select_item.setFlags(select_item.flags() & ~Qt.ItemIsEnabled)
                    select_item.setToolTip(tip)

            self.on_line_selected()
        finally:
            self.updating_manual_stats = False

    def sync_flux_flag(self, row):
        """Flag the Flux cell red while empty; clear otherwise.

        Args:
            row (int): Table row index to sync.

        Returns:
            None
        """
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
        """Reset Calculated?/Source to unset by default when edited.

        Args:
            row (int): Table row index to reset.

        Returns:
            None
        """
        calc_item = self.lines_table.item(row, 11)
        if calc_item:
            calc_item.setCheckState(Qt.Unchecked)
        source_item = self.lines_table.item(row, 12)
        if source_item:
            source_item.setText("")

    def sync_flux_error_flag(self, row):
        """Flag the Flux Error cell red when blank and the line isn't marked as an Upper Limit.

        Args:
            row (int): Table row index to sync.

        Returns:
            None
        """
        if self.updating_flux_error_flag:
            return
        self.updating_flux_error_flag = True
        try:
            error_item = self.lines_table.item(row, 7)
            pct_item = self.lines_table.item(row, 8)
            if not error_item or not pct_item:
                return
            ul_item = self.lines_table.item(row, 9)
            is_upper_limit = bool(ul_item and ul_item.checkState() == Qt.Checked)

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

            self._update_error_columns_for_row(row)
        finally:
            self.updating_flux_error_flag = False

    def fetch_hitran_lines(self):
        """Fetch molecular lines from HITRAN.

        Args:
            None

        Returns:
            None
        """
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
        """Populate the lines table with a completed HITRAN fetch's results.

        Args:
            data (list): Fetched HITRAN line rows.
            progress (QProgressDialog): Fetch progress dialog to close.

        Returns:
            None
        """
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
        self.save_fluxes_btn.setVisible(True)
        self.calculate_maps_btn.setVisible(True)
        self.convolve_maps_btn.setVisible(True)
        self.contam_status_label.setVisible(True)

        self._reset_batch_cubes()
        self.hitran_lines = data
        self.populate_lines_table()

    def on_hitran_error(self, error_msg, progress):
        """Show a failed HITRAN fetch's error message to the user.

        Args:
            error_msg (str): Error message from the fetch thread.
            progress (QProgressDialog): Fetch progress dialog to close.

        Returns:
            None
        """
        progress.close()
        self.fetch_btn.setEnabled(True)
        QMessageBox.critical(self, "HITRAN Fetch Error", error_msg)
    
    def import_csv_file(self):
        """Import species, wavelength, and flux data from CSV, enriching
        with HITRAN data for A and E_u.

        Args:
            None

        Returns:
            None
        """
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

                # Check for optional upper_limit column (written by Save Fluxes)
                upper_limit_col = available_cols.get('upper_limit')

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

                        # Upper-limit flag, if the CSV has one (Save Fluxes writes 1/0)
                        if upper_limit_col:
                            ul_str = (row_dict.get(upper_limit_col) or '').strip().lower()
                            row_info['upper_limit'] = ul_str in ('1', 'true', 'yes')

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

            # A CSV that's entirely one molecule -- switch the Molecule dropdown to
            # match instead of silently fitting it as whatever molecule was already
            # selected (H2's j_lower keying breaks on non-H2 P/R-branch data).
            if len(species_list) == 1 and species_list[0] != self.current_molecule \
                    and species_list[0] in MOLECULE_ISOTOPOLOGUES:
                new_molecule = species_list[0]
                self.mol_combo.blockSignals(True)
                self.mol_combo.setCurrentText(new_molecule)
                self.mol_combo.blockSignals(False)
                self._apply_molecule_change(new_molecule)

            # Get wavelength ranges to fetch
            all_wavelengths = [row['wavelength'] for row in csv_data]
            wl_min = min(all_wavelengths) - 0.1  # Buffer for rounding
            wl_max = max(all_wavelengths) + 0.1
            
            hitran_cache = {}  # Cache HITRAN data by species
            failed_species = {}  # species -> message, for genuine fetch failures (not "zero lines")

            progress = QProgressDialog(
                "Enriching CSV data with HITRAN information...",
                None, 0, len(species_list), self
            )
            progress.setWindowModality(Qt.WindowModal)

            # Fetch HITRAN data for each species
            for idx, species in enumerate(species_list):
                progress.setValue(idx)
                try:
                    hitran_data, fetch_meta = fetch_hitran_data(
                        wl_min, wl_max,
                        molecule=species,
                        isotopologue=1
                    )
                    hitran_cache[species] = hitran_data
                    if fetch_meta.get('error'):
                        failed_species[species] = fetch_meta.get('message', 'unknown error')
                except Exception as e:
                    print(f"Warning: Could not fetch HITRAN data for {species}: {e}")
                    hitran_cache[species] = []
                    failed_species[species] = str(e)
            
            progress.close()
            
            # Match CSV rows to HITRAN data
            for csv_row in csv_data:
                species = csv_row['species']
                wavelength = csv_row['wavelength']
                hitran_lines = hitran_cache.get(species, [])
                
                # Find matching HITRAN line (within 0.0001 um = 4 decimal places)
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
                    if species in failed_species:
                        print(f"Warning: Could not fetch HITRAN data for {species} at "
                              f"{wavelength:.6f} um ({failed_species[species]}) -- "
                              f"A_ul/E_u/g_u left blank.")
                    else:
                        print(f"Warning: No HITRAN match for {species} at {wavelength:.6f} um "
                              f"-- A_ul/E_u/g_u left blank; this line is excluded from fitting "
                              f"until they're entered manually.")

                # Flux is optional in the CSV
                if 'flux' in csv_row:
                    row_data['flux'] = csv_row['flux']

                # Add error if available
                if 'flux_error' in csv_row:
                    row_data['flux_error'] = csv_row['flux_error']

                # Carry the upper-limit flag, if the CSV had one
                if 'upper_limit' in csv_row:
                    row_data['upper_limit'] = csv_row['upper_limit']

                rows.append(row_data)

            self._reset_batch_cubes()
            self.hitran_lines = rows
            self.populate_lines_table()

            # Pre-fill errors if they were provided in CSV
            self.populate_csv_errors()

            # Set workflow mode to CSV. Fetch/Import stay both enabled.
            self.workflow_mode = 'csv'
            self.calculate_fluxes_btn.setVisible(True)
            self.load_fits_btn.setVisible(True)
            self.load_contam_btn.setVisible(True)
            self.save_fluxes_btn.setVisible(True)
            self.calculate_maps_btn.setVisible(True)
            self.convolve_maps_btn.setVisible(True)
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
            # Distinguish "genuinely not in HITRAN" from "couldn't reach HITRAN" --
            # the latter is a network failure, not a real zero-match result.
            fetch_fail_note = (
                "" if not failed_species
                else "\nCould not fetch HITRAN data for: " + ", ".join(
                    f"{sp} ({msg})" for sp, msg in sorted(failed_species.items())
                ) + "\nNo offline copy covers this range -- check your internet "
                    "connection, or pre-cache this range while online.\n"
            )
            QMessageBox.information(
                self,
                "Import Successful",
                f"Imported {len(rows)} rows from CSV\n"
                f"Matched {matched}/{len(rows)} lines to HITRAN data\n"
                f"{fetch_fail_note}"
                f"{unmatched_note}"
                f"{flux_note}"
            )
        
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import CSV:\n{str(e)}")
    
    def populate_csv_errors(self):
        """Populate error column if flux_error was in CSV, then re-check
        Upper Limit if the CSV had that flag -- reusing sync_flux_error_flag's
        own stash, so a re-imported upper limit keeps its real error too.

        Args:
            None

        Returns:
            None
        """
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

                # Must run after the error/pct text above is set, so the stash
                # below has the real value to hide, not a blank cell.
                if line_data.get('upper_limit'):
                    ul_item = self.lines_table.item(row, 9)
                    if ul_item:
                        self.lines_table.blockSignals(True)
                        ul_item.setCheckState(Qt.Checked)
                        self.lines_table.blockSignals(False)
                        self.sync_flux_error_flag(row)

    def load_batch_fits_files(self):
        """Select and cache the FITS cube(s) Calculate Fluxes will use. Adds
        to whatever's already loaded, so reopening this dialog for another
        folder doesn't lose earlier files.

        Args:
            None

        Returns:
            None
        """
        fits_paths, _ = QFileDialog.getOpenFileNames(
            self, "Load FITS Files For Flux Calculation", "",
            "FITS Files (*.fits *.fit);;All Files (*.*)"
        )
        if not fits_paths:
            return

        already_loaded = set(self.loaded_batch_fits_paths)
        new_paths = [p for p in fits_paths if p not in already_loaded]
        n_duplicates = len(fits_paths) - len(new_paths)

        failed = []
        for fits_path in new_paths:
            try:
                cube = CubeData()
                cube.load_from_file(fits_path)
                summed = np.nansum(cube.cube, axis=(1, 2)) if cube.cube.ndim == 3 else cube.cube
                self.loaded_batch_fits_paths.append(fits_path)
                self.loaded_batch_cubes.append(cube)
                self.loaded_batch_spectra.append(summed)
            except Exception as e:
                failed.append((fits_path, str(e)))

        # Tooltip lists everything loaded so far, not just this batch
        names = "\n".join(os.path.basename(p) for p in self.loaded_batch_fits_paths)
        self.load_fits_btn.setToolTip(names if names else "Select the source FITS cube(s) for Calculate Fluxes.")

        if failed:
            failed_msg = "\n".join(f"  {os.path.basename(p)}: {err}" for p, err in failed)
            QMessageBox.warning(
                self, "Some Files Failed To Load",
                f"{len(failed)} file(s) could not be loaded:\n\n{failed_msg}"
            )
        elif new_paths:
            added_names = "\n".join(os.path.basename(p) for p in new_paths)
            dup_note = f"\n\n({n_duplicates} already loaded, skipped)" if n_duplicates else ""
            QMessageBox.information(
                self, "FITS Files Loaded",
                f"Added {len(new_paths)} file(s):\n\n{added_names}{dup_note}\n\n"
                f"{len(self.loaded_batch_fits_paths)} total loaded."
            )
        elif n_duplicates:
            QMessageBox.information(self, "Already Loaded", "All selected file(s) were already loaded.")

    def load_contaminating_lines_for_batch(self):
        """Load a contaminating-lines CSV shared across FITS files/ranges;
        each fit group filters it down to its own fit window.

        Args:
            None

        Returns:
            None
        """

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

    def save_fluxes_to_csv(self):
        """Save the checked lines' species/wavelength/flux/flux error/upper-limit
        flag to a CSV that import_csv_file() reads back, so Calculate Fluxes
        need not rerun on them. Upper limits keep their real error too --
        it's read from its UserRole stash, not the blanked visible cell.

        Args:
            None

        Returns:
            None
        """
        rows_to_save = []
        for row in range(self.lines_table.rowCount()):
            select_item = self.lines_table.item(row, 0)
            if not (select_item and select_item.checkState() == Qt.Checked):
                continue
            flux_item = self.lines_table.item(row, 6)
            flux_text = flux_item.text().strip() if flux_item else ""
            if not flux_text:
                continue
            ul_item = self.lines_table.item(row, 9)
            is_upper_limit = bool(ul_item and ul_item.checkState() == Qt.Checked)

            # Upper limits stash their error in UserRole and blank the visible
            # cell (sync_flux_error_flag) -- read the stash, not the blank text.
            error_item = self.lines_table.item(row, 7)
            error_text = ""
            if error_item:
                error_text = (error_item.data(Qt.UserRole) if is_upper_limit
                              else error_item.text()) or ""
                error_text = error_text.strip()

            rows_to_save.append({
                "species": self.hitran_lines[row].get("species", "Unknown"),
                "wavelength": self.hitran_lines[row]["wavelength"],
                "flux": flux_text,
                "flux_error": error_text,
                "upper_limit": "1" if is_upper_limit else "0",
            })

        if not rows_to_save:
            QMessageBox.warning(
                self, "Nothing to Save",
                "No checked line has a flux value yet -- run Calculate Fluxes, "
                "or enter fluxes by hand, first."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Fluxes as CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return
        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"

        try:
            with open(file_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["species", "wavelength", "flux", "flux_error", "upper_limit"])
                writer.writeheader()
                writer.writerows(rows_to_save)
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save fluxes:\n{e}")
            return

        QMessageBox.information(
            self, "Fluxes Saved",
            f"Saved {len(rows_to_save)} line(s) to:\n{file_path}\n\n"
            f"Use Import from CSV to reload them without recalculating."
        )

    def open_batch_flux_dialog(self):
        """Open dialog for batch flux calculation from multiple FITS files.

        Args:
            None

        Returns:
            None
        """
        selected_rows = []
        skipped_rows = []
        for row in range(self.lines_table.rowCount()):
            select_item = self.lines_table.item(row, 0)
            if select_item and select_item.checkState() == Qt.Checked:
                flux_item = self.lines_table.item(row, 6)
                calc_item = self.lines_table.item(row, 11)
                was_calculated = bool(calc_item and calc_item.checkState() == Qt.Checked)
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
        """Lines sharing a FITS file are fit together via
        compute_line_fluxes_collective, in a background BatchFluxWorker so
        the GUI doesn't freeze; _on_batch_flux_complete applies results.

        Args:
            selected_rows (list[int]): Table rows to calculate fluxes for.
            skipped_rows (list[int]): Rows left unchanged (already had a
                manually-entered flux), reported in the completion message.

        Returns:
            None
        """

        loaded_cubes = self.loaded_batch_cubes

        # --- Find each line's covering cube (first match wins, same as before) ---
        # Cheap and Qt-free, so this stays on the main thread rather than the worker.
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
                return

            rows_by_cube_idx.setdefault(matched_idx, []).append(row)

        self._batch_flux_skipped_rows = skipped_rows

        self.batch_flux_progress = QProgressDialog("Calculating fluxes...", None, 0, 0, self)
        self.batch_flux_progress.setWindowTitle("Calculate Fluxes")
        self.batch_flux_progress.setWindowModality(Qt.WindowModal)
        self.batch_flux_progress.setAutoClose(False)
        self.batch_flux_progress.setAutoReset(False)
        self.batch_flux_progress.show()

        self.calculate_fluxes_btn.setEnabled(False)

        line_wavelength_lookup = {row: self.hitran_lines[row]['wavelength'] for row in selected_rows}
        self.batch_flux_worker = BatchFluxWorker(
            rows_by_cube_idx, loaded_cubes, self.loaded_batch_spectra,
            line_wavelength_lookup, self.contam_lines,
        )
        self.batch_flux_worker.finished.connect(self._on_batch_flux_complete)
        self.batch_flux_worker.error.connect(self._on_batch_flux_error)
        self.batch_flux_worker.progress.connect(self._on_batch_flux_progress)
        print(f"[TIMING] starting BatchFluxWorker at {time.time():.2f}")
        self.batch_flux_worker.start()

    def _on_batch_flux_progress(self, current, total):
        """Update the batch-flux progress dialog for the current FITS group.

        Args:
            current (int): Index of the FITS-file group in progress.
            total (int): Total number of FITS-file groups.

        Returns:
            None
        """
        dlg = getattr(self, 'batch_flux_progress', None)
        if dlg is None:
            return
        if total <= 0:
            dlg.setRange(0, 0)  # busy/indeterminate -- e.g. the common single-FITS-file case
            return
        dlg.setRange(0, total)
        dlg.setLabelText(f"Calculating fluxes... (FITS file group {min(current+1, total)}/{total})")
        dlg.setValue(current)

    def _on_batch_flux_complete(self, groups):
        """Apply every FITS-group's computed fluxes to the table.

        Args:
            groups (list[dict]): Per-FITS-group results from BatchFluxWorker.

        Returns:
            None
        """
        print(f"[TIMING] _on_batch_flux_complete entered at {time.time():.2f}")
        dlg = getattr(self, 'batch_flux_progress', None)
        if dlg is not None:
            dlg.close()
        self.calculate_fluxes_btn.setEnabled(True)

        t0 = time.time()
        self.lines_table.setUpdatesEnabled(False)
        try:
            for group in groups:
                if not group["success"]:
                    # One bad FITS file/group doesn't abort the rest of the batch.
                    print(f"Error processing lines from cube {group['cube_idx']}: {group['error']}")
                    QMessageBox.critical(self, "Calculation Error",
                                          f"Error processing lines from FITS file:\n{group['error']}")
                    continue
                t1 = time.time()
                self._apply_batch_flux_group_results(group)
                print(f"[TIMING] _apply_batch_flux_group_results for {len(group['rows'])} row(s): {time.time()-t1:.2f}s")
        finally:
            self.lines_table.setUpdatesEnabled(True)
        print(f"[TIMING] full table-update loop: {time.time()-t0:.2f}s")

        skipped_rows = getattr(self, '_batch_flux_skipped_rows', None)
        msg = "Flux calculation complete!"
        if skipped_rows:
            msg += f"\n\n{len(skipped_rows)} selected line(s) already had a manually-entered flux and were left unchanged."
        QMessageBox.information(self, "Complete", msg)

    def _on_batch_flux_error(self, message):
        """Show a failed batch flux calculation's error message.

        Args:
            message (str): Error message from BatchFluxWorker.

        Returns:
            None
        """
        dlg = getattr(self, 'batch_flux_progress', None)
        if dlg is not None:
            dlg.close()
        self.calculate_fluxes_btn.setEnabled(True)
        QMessageBox.critical(self, "Calculation Error", f"Unexpected error calculating fluxes:\n{message}")

    def _apply_batch_flux_group_results(self, group):
        """Apply one FITS-file group's computed fluxes to the table.
        Main-thread only (Qt widgets) -- called from
        _on_batch_flux_complete once BatchFluxWorker finishes.

        Args:
            group (dict): One FITS-group's fit results and metadata.

        Returns:
            None
        """
        rows = group["rows"]
        group_results = group["results"]
        wavelength = group["wavelength"]
        spectrum = group["spectrum"]
        instrument = group["instrument"]
        filename = os.path.basename(group["filename"]) if group["filename"] else "?"
        bunit = group["bunit"]

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

            # Error is always the real curve_fit value, written regardless of
            # upper-limit status; sync_flux_error_flag() stashes/blanks it below.
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

            # Auto-check upper limit if detected. Blocked signals skip on_line_selected();
            # sync_flux_error_flag() runs right after, so the stash-and-blank still applies.
            ul_item = self.lines_table.item(row, 9)
            if ul_item:
                self.lines_table.blockSignals(True)
                ul_item.setCheckState(Qt.Checked if is_upper_limit else Qt.Unchecked)
                self.lines_table.blockSignals(False)
                self.sync_flux_error_flag(row)

            # Mark this row as system-calculated and record its source
            calc_item = self.lines_table.item(row, 11)
            if calc_item:
                calc_item.setCheckState(Qt.Checked)
            source_item = self.lines_table.item(row, 12)
            if source_item:
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
            # link, only once real data exists so it's never a dead link.
            if all(result.get(k) is not None for k in ("x_data", "y_data", "baseline", "y_fit")):
                self._line_fit_plot_data[row] = {
                    "wavelength": wl,
                    "x_data": result["x_data"],
                    "y_data": result["y_data"],
                    "baseline": result["baseline"],
                    "y_fit": result["y_fit"],
                    "y_fit_own": result.get("y_fit_own"),
                    "center": result.get("center"),
                    "sigma": result.get("sigma"),
                    "fwhm": result.get("fwhm"),
                    "r_squared": result.get("r_squared"),
                    "amplitude": result.get("amplitude"),
                    "flux_cgs": result.get("flux_cgs"),
                    "flux_error_cgs": result.get("flux_error_cgs"),
                    "is_upper_limit": result.get("is_upper_limit"),
                    # Full spectrum this fit came from, so the fit dialog can
                    # show real context instead of just the fit's own narrow window.
                    "full_wavelength": wavelength,
                    "full_spectrum": spectrum,
                    "bunit": bunit,
                }
                fit_item = self.lines_table.item(row, 14)
                if fit_item:
                    fit_item.setText("View")
                    fit_item.setForeground(QColor(0, 0, 238))
                    font = fit_item.font()
                    font.setUnderline(True)
                    fit_item.setFont(font)
                    fit_item.setToolTip("Double-click to view this line's fit")

    def on_lines_table_item_double_clicked(self, item):
        """Open the fit/map/convolved-map dialog for a double-clicked cell.

        Args:
            item (QTableWidgetItem): The cell the user double-clicked.

        Returns:
            None: Opens show_line_fit's, show_line_map's, or
                show_line_convolved_map's dialog as a side effect, if applicable.
        """
        if item.column() == 14:
            self.show_line_fit(item.row())
        elif item.column() == 15:
            self.show_line_map(item.row())
        elif item.column() == 16:
            self.show_line_convolved_map(item.row())

    def show_line_fit(self, row):
        """Show the Gaussian fit for one line: spectrum, baseline, fit curve,
        and shaded area, matching the main window's Multi-Gaussian display.
        Uses this window's own loaded spectrum, not the main window's cube.

        Args:
            row (int): Table row whose fit to display.

        Returns:
            None
        """
        data = self._line_fit_plot_data.get(row)
        if data is None:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Line Fit @ {data['wavelength']:.6f} μm")
        dlg.resize(800, 550)
        layout = QVBoxLayout()

        info_parts = [f"<b>λ = {data['wavelength']:.6f} μm</b>"]
        flux_str = None
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

        chunkwide_checkbox = QCheckBox("Chunk-wide fit curve")
        chunkwide_checkbox.setChecked(True)
        chunkwide_checkbox.setToolTip(
            "Checked: draw the real joint fit curve, shared by every line in the same chunk. "
            "Unchecked: draw this line's own Gaussian alone, cropped to its own peak."
        )
        layout.addWidget(chunkwide_checkbox)

        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)

        x = np.asarray(data['x_data'])
        global_continuum = np.asarray(data['baseline'])

        def _own_curve():
            """
            This line's own Gaussian alone, cropped to DISPLAY_CROP_SIGMA_MULTIPLIER.

            Args:
                None

            Returns:
                curve (tuple): (x_curve, baseline_curve, y_fit_curve) arrays
            """
            # Falls back to the full-chunk y_fit for data stashed before y_fit_own existed.
            y_fit_own = data.get('y_fit_own')
            model = y_fit_own if y_fit_own is not None else data['y_fit']
            y_fit = np.asarray(model) + global_continuum
            own_gaussian = gaussian(x, data['amplitude'], data['center'], data['sigma'])
            baseline = y_fit - own_gaussian
            if len(x) < 2 or x.max() <= x.min():
                return x, baseline, y_fit
            crop_half_width = DISPLAY_CROP_SIGMA_MULTIPLIER * abs(data['sigma'])
            lo = max(x.min(), data['center'] - crop_half_width)
            hi = min(x.max(), data['center'] + crop_half_width)
            x_curve = np.linspace(lo, hi, 300)
            baseline_curve = np.interp(x_curve, x, baseline)
            own_gaussian_curve = gaussian(x_curve, data['amplitude'], data['center'], data['sigma'])
            return x_curve, baseline_curve, baseline_curve + own_gaussian_curve

        def _chunk_curve():
            """
            The real joint fit's own chunk: continuum plus every Gaussian curve_fit
            solved for at once, not just this line's own component.

            Args:
                None

            Returns:
                curve (tuple): (x_curve, baseline_curve, y_fit_curve) arrays
            """
            y_fit = np.asarray(data['y_fit']) + global_continuum
            return x, global_continuum, y_fit

        fit_artists = []

        def _draw_fit(chunk_wide):
            """Redraw the fit curve/shaded area, wide or line-only.

            Args:
                chunk_wide (bool): True to show the fit's full fit chunk;
                    False to show just the line region.

            Returns:
                None
            """
            for artist in fit_artists:
                try: artist.remove()
                except Exception: pass
            fit_artists.clear()
            x_curve, baseline_curve, y_fit_curve = _chunk_curve() if chunk_wide else _own_curve()
            b_line, = ax.plot(x_curve, baseline_curve, 'k--', linewidth=2, alpha=0.7,
                               label='Baseline', zorder=9)
            g_line, = ax.plot(x_curve, y_fit_curve, 'r-', linewidth=3, alpha=0.9,
                               label='Gaussian Fit', zorder=10)
            fill = ax.fill_between(x_curve, baseline_curve, y_fit_curve, color='red',
                                    alpha=0.2, label='Integrated Area', zorder=7)
            fit_artists.extend([b_line, g_line, fill])
            ax.legend(loc='upper right', fontsize=8)
            canvas.draw_idle()

        # Pad by the spectrum's own sample spacing, not a fixed wavelength
        # range, so this scales across instruments (MIRI vs. NIRSpec differ a lot).
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

        # Marks this row's own fitted center, since the chunk-wide curve can
        # show several lines' peaks at once (same as the main window's markers).
        ax.axvline(center, color='darkred', linestyle='--', linewidth=1.5, alpha=0.8, zorder=8)

        _draw_fit(chunkwide_checkbox.isChecked())
        chunkwide_checkbox.toggled.connect(_draw_fit)

        # Disable matplotlib's "+1.703e1"-style offset notation -- otherwise a
        # narrow x-range shows "0.000"-like ticks that look like they start at 0.
        ax.ticklabel_format(axis='x', useOffset=False, style='plain')

        ax.set_xlabel('Wavelength (μm)')
        bunit = data.get('bunit')
        ax.set_ylabel(f'Flux ({bunit})' if bunit else 'Flux')
        ax.set_title(f"Line @ {data['wavelength']:.6f} μm", fontweight='bold')
        ax.grid(True, alpha=0.3)
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

    def show_line_map(self, row):
        """Open the per-pixel native-resolution flux map for one line,
        populated by Calculate Maps or Run Map. The Run Map analogue of
        show_line_fit: a 2D image instead of a single spectrum, since Run
        Map fits every pixel rather than one summed region.

        Args:
            row (int): Table row whose map to display.

        Returns:
            None
        """
        data = self._line_map_plot_data.get(row)
        if data is None:
            return
        LineFluxMapDialog(
            data['wavelength'], data['flux_map'], data.get('header'),
            molecule=data.get('molecule', 'H2'), parent=self,
        ).exec_()

    def show_line_convolved_map(self, row):
        """Open the post-convolution flux map for one line, populated by
        Convolve Maps or Run Map (multi-cube only).

        Args:
            row (int): Table row whose convolved map to display.

        Returns:
            None
        """
        data = self._line_convolved_map_plot_data.get(row)
        if data is None:
            return
        LineFluxMapDialog(
            data['wavelength'], data['flux_map'], data.get('header'),
            molecule=data.get('molecule', 'H2'), parent=self,
        ).exec_()

    def _row_for_wavelength(self, wl):
        """Find the table row whose line matches a wavelength.

        Args:
            wl (float): Target line's wavelength (um).

        Returns:
            row (int or None): Matching row index, or None if no row matches.
                Matches by wavelength since kept_lines' dicts come straight
                from self.hitran_lines (via the checked-rows line_table),
                same convention _rot_map_match_lines_to_cubes already uses.
        """
        return next((r for r, hl in enumerate(self.hitran_lines)
                     if abs(hl.get("wavelength", float("nan")) - wl) < 1e-9), None)

    def _set_view_link(self, item, tooltip):
        """Style a table cell as a blue underlined "View" link.

        Args:
            item (QTableWidgetItem): Cell to style.
            tooltip (str): Tooltip text for the cell.

        Returns:
            None
        """
        item.setText("View")
        item.setForeground(QColor(0, 0, 238))
        font = item.font()
        font.setUnderline(True)
        item.setFont(font)
        item.setToolTip(tooltip)

    def _populate_map_column(self, result):
        """Turn each mapped line's "Map" cell into a "View" link, using the
        native (pre-convolution) map when available -- Calculate Maps'
        native_result, or a full Run Map result, both carry it; a plain
        single-cube result only has line_flux_maps, since there is only one
        resolution to begin with. Lines dropped by dedup, or not part of
        this result, are left as "—".

        Args:
            result (dict): Output of compute_native_line_maps,
                create_rotational_diagram_map, or create_rotational_diagram_map_multi.

        Returns:
            None
        """
        kept_lines = result.get("kept_lines") or []
        flux_maps = result.get("native_flux_maps") or result.get("line_flux_maps") or []
        flux_error_maps = result.get("native_flux_error_maps") or result.get("line_flux_error_maps") or []
        if not kept_lines or not flux_maps:
            return
        native_wcs = result.get("native_wcs")  # multi-cube: one per kept_lines entry

        for i, line in enumerate(kept_lines):
            wl = line.get("wavelength")
            row = self._row_for_wavelength(wl)
            if row is None:
                continue
            if native_wcs is not None:
                header = native_wcs[i].to_header() if native_wcs[i] is not None else None
            else:
                header = result.get("header")  # single-cube's own header
            self._line_map_plot_data[row] = {
                "wavelength": wl,
                "flux_map": flux_maps[i],
                "flux_error_map": flux_error_maps[i] if i < len(flux_error_maps) else None,
                "header": header,
                "molecule": result.get("molecule", "H2"),
            }
            map_item = self.lines_table.item(row, 15)
            if map_item:
                self._set_view_link(map_item, "Double-click to view this line's flux map")

    def _populate_convolved_map_column(self, result):
        """Turn each mapped line's "Convolved Map" cell into a "View" link,
        using result['kept_lines']/['line_flux_maps'] (post-convolution,
        common-grid arrays). Multi-cube only -- a single-cube result has no
        convolution step, so this is a no-op for one.

        Args:
            result (dict): Output of convolve_line_maps or
                create_rotational_diagram_map_multi.

        Returns:
            None
        """
        kept_lines = result.get("kept_lines") or []
        line_flux_maps = result.get("line_flux_maps") or []
        line_flux_error_maps = result.get("line_flux_error_maps") or []
        dst_wcs = result.get("dst_wcs")
        if not kept_lines or not line_flux_maps or dst_wcs is None:
            return
        source_header = dst_wcs.to_header()

        for i, line in enumerate(kept_lines):
            wl = line.get("wavelength")
            row = self._row_for_wavelength(wl)
            if row is None:
                continue
            self._line_convolved_map_plot_data[row] = {
                "wavelength": wl,
                "flux_map": line_flux_maps[i],
                "flux_error_map": line_flux_error_maps[i] if i < len(line_flux_error_maps) else None,
                "header": source_header,
                "molecule": result.get("molecule", "H2"),
            }
            convolved_item = self.lines_table.item(row, 16)
            if convolved_item:
                self._set_view_link(convolved_item, "Double-click to view this line's convolved flux map")

    def _make_checkable_item(self, checked=False, enabled=True, tooltip=""):
        """Build a checkbox-style cell as a real checkable item, not a QCheckBox
        widget -- a widget does not survive a header-click row reorder.

        Args:
            checked (bool): Initial check state.
            enabled (bool): Whether the user can toggle it.
            tooltip (str): Tooltip text. Empty string means no tooltip.

        Returns:
            item (QTableWidgetItem): The configured item, not yet inserted.
        """
        item = QTableWidgetItem()
        flags = Qt.ItemIsUserCheckable | Qt.ItemIsSelectable
        if enabled:
            flags |= Qt.ItemIsEnabled
        item.setFlags(flags)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        if tooltip:
            item.setToolTip(tooltip)
        return item

    def populate_lines_table(self):
        """Populate the unified table with all columns.

        Args:
            None

        Returns:
            None
        """
        # Row indices are about to be reassigned to a new set of lines -- old plot
        # data keyed by row would otherwise silently point at the wrong line.
        self._line_fit_plot_data = {}
        self._line_map_plot_data = {}
        self._line_convolved_map_plot_data = {}
        self._reset_sort_state()
        self.lines_table.setRowCount(len(self.hitran_lines))
        
        for row, line in enumerate(self.hitran_lines):
            # A line with no HITRAN match has no real A_ul/E_u/g_u: we never assume a value for them.
            matched = line.get('hitran_matched', True)
            unmatched_bg = QColor(255, 214, 214)
            unmatched_tip = (
                "No HITRAN match for this line -- enter A_ul, E_u, and g_u manually "
                "to include it in the fit."
            )

            # Row-select checkbox -- checked by default (a line with no HITRAN
            # match stays unchecked too, since it's disabled until completed).
            select_item = self._make_checkable_item(
                checked=matched, enabled=matched, tooltip=("" if matched else unmatched_tip)
            )
            self.lines_table.setItem(row, 0, select_item)

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

            # Upper Limit? (auto-suggested but always user-editable)
            ul_item = self._make_checkable_item(tooltip=(
                "Auto-checked when a line's fit falls below the 3σ noise threshold "
                "(via Calculate Fluxes). You can check/uncheck this yourself either way."
            ))
            self.lines_table.setItem(row, 9, ul_item)

            # UL Flux (editable)
            self.lines_table.setItem(row, 10, QTableWidgetItem(""))

            # Calculated? (read-only)
            calc_item = self._make_checkable_item(enabled=False, tooltip=(
                "Checked when this line's flux was computed by Calculate Fluxes, rather "
                "than typed in by hand or imported from CSV."
            ))
            self.lines_table.setItem(row, 11, calc_item)

            # Source (read-only)
            source_item = QTableWidgetItem("")
            source_item.setFlags(source_item.flags() & ~Qt.ItemIsEditable)
            self.lines_table.setItem(row, 12, source_item)

            # Fit R^2 (read-only; filled in by Calculate Fluxes, blank for manually-entered flux)
            r2_item = QTableWidgetItem("")
            r2_item.setFlags(r2_item.flags() & ~Qt.ItemIsEditable)
            self.lines_table.setItem(row, 13, r2_item)

            # Fit (double-click to view; becomes real text once Calculate Fluxes
            # populates plot data for this row -- see on_lines_table_item_double_clicked)
            fit_item = QTableWidgetItem("—")
            fit_item.setFlags(fit_item.flags() & ~Qt.ItemIsEditable)
            fit_item.setTextAlignment(Qt.AlignCenter)
            self.lines_table.setItem(row, 14, fit_item)

            # Map (double-click to view; becomes real text once Run Map or
            # Calculate Maps populates this line's flux map -- see _populate_map_column)
            map_item = QTableWidgetItem("—")
            map_item.setFlags(map_item.flags() & ~Qt.ItemIsEditable)
            map_item.setTextAlignment(Qt.AlignCenter)
            self.lines_table.setItem(row, 15, map_item)

            # Convolved Map (double-click to view; multi-cube only -- see
            # _populate_convolved_map_column)
            convolved_map_item = QTableWidgetItem("—")
            convolved_map_item.setFlags(convolved_map_item.flags() & ~Qt.ItemIsEditable)
            convolved_map_item.setTextAlignment(Qt.AlignCenter)
            self.lines_table.setItem(row, 16, convolved_map_item)

            self.sync_flux_error_flag(row)
        
        # Set initial edit/read-only status based on current mode
        self.update_error_columns()
        # Rows just defaulted to checked -- reflect that in the Run Fit button.
        self.on_line_selected()

    def _on_sortable_header_clicked(self, column):
        """Override the mixin's click-to-sort: block it while a background
        flux calculation is writing results into the table by row index,
        since reordering rows mid-calculation would land results on the wrong line.

        Args:
            column (int): Table column index the user clicked.

        Returns:
            None: Defers to SortableCheckboxTableMixin's sort, unless blocked.
        """
        if not self.calculate_fluxes_btn.isEnabled():
            return
        super()._on_sortable_header_clicked(column)

    def _lines_table_sort_key(self, row, column):
        """Numeric sort key for one cell. A missing or unparseable value (a
        blank A_ul/E_u/g_u, or a flux not yet calculated) ranks lowest. An
        incomplete row never breaks the sort.

        Args:
            row (int): Table row index.
            column (int): Table column index. Must be in self._SORTABLE_COLUMNS.

        Returns:
            key (float): Sort key for this cell.
        """
        if column == 1:
            try:
                return float(self.hitran_lines[row].get('wavelength', float('-inf')))
            except (TypeError, ValueError):
                return float('-inf')
        if column in (3, 4, 5):
            field = {3: 'a_coeff', 4: 'e_upper', 5: 'Gp'}[column]
            try:
                return float(self.hitran_lines[row].get(field, float('-inf')))
            except (TypeError, ValueError):
                return float('-inf')
        # Flux (6), Flux Error (7), Error % (8), and UL Flux (10) are live-edited --
        # the table cell holds the true value, not self.hitran_lines.
        item = self.lines_table.item(row, column)
        text = item.text().strip() if item else ""
        try:
            return float(text)
        except ValueError:
            return float('-inf')

    def _permute_extra_row_state(self, order, row_map):
        """Keep self.hitran_lines, _line_fit_plot_data, _line_map_plot_data, and
        _line_convolved_map_plot_data in step with a lines_table reorder (all
        four are keyed by row index).

        Args:
            order (list[int]): New row i holds old row order[i].
            row_map (dict[int, int]): Old row index -> new row index.

        Returns:
            None: Reassigns the four attributes in place.
        """
        self.hitran_lines = [self.hitran_lines[old_row] for old_row in order]
        self._line_fit_plot_data = {
            row_map[old_row]: v for old_row, v in self._line_fit_plot_data.items()
        }
        self._line_map_plot_data = {
            row_map[old_row]: v for old_row, v in self._line_map_plot_data.items()
        }
        self._line_convolved_map_plot_data = {
            row_map[old_row]: v for old_row, v in self._line_convolved_map_plot_data.items()
        }

    def on_line_selected(self):
        """Enable Run Fit only once >=2 lines are checked.

        Args:
            None

        Returns:
            None
        """
        checked_count = 0
        for row in range(self.lines_table.rowCount()):
            select_item = self.lines_table.item(row, 0)
            if select_item and select_item.checkState() == Qt.Checked:
                checked_count += 1
        self.run_fit_btn.setEnabled(checked_count >= 2)
        self.run_fit_btn.setToolTip(
            "" if checked_count >= 2 else "Select at least 2 lines -- a line fit needs 2+ points."
        )
        # The map needs at least 3 detections a pixel to constrain a 1T slope.
        self.run_map_btn.setEnabled(checked_count >= 3)
        self.run_map_btn.setToolTip(
            "" if checked_count >= 3 else "Select at least 3 lines for a per-pixel map."
        )
        # Calculate/Convolve Maps don't need a T-fit -- any checked line's own
        # map is meaningful on its own. "Need a loaded cube"/"need 2+ cubes"/
        # "run Calculate Maps first" are checked at click time instead, same
        # as Run Map's own "No Cube" check.
        self.calculate_maps_btn.setEnabled(checked_count >= 1)
        self.calculate_maps_btn.setToolTip(
            "" if checked_count >= 1 else "Select at least 1 line."
        )
        self.convolve_maps_btn.setEnabled(checked_count >= 1)
        self.convolve_maps_btn.setToolTip(
            "" if checked_count >= 1 else "Select at least 1 line."
        )

    def select_all_lines(self):
        """Select all lines in the table (skips rows disabled for missing HITRAN data).

        Args:
            None

        Returns:
            None
        """
        self.lines_table.blockSignals(True)
        for row in range(self.lines_table.rowCount()):
            select_item = self.lines_table.item(row, 0)
            if select_item and select_item.flags() & Qt.ItemIsEnabled:
                select_item.setCheckState(Qt.Checked)
        self.lines_table.blockSignals(False)
        self.on_line_selected()

    def deselect_all_lines(self):
        """Deselect all lines in the table.

        Args:
            None

        Returns:
            None
        """
        self.lines_table.blockSignals(True)
        for row in range(self.lines_table.rowCount()):
            select_item = self.lines_table.item(row, 0)
            if select_item:
                select_item.setCheckState(Qt.Unchecked)
        self.lines_table.blockSignals(False)
        self.on_line_selected()
    
    def run_rotational_fit(self):
        """Run the fitting.

        Args:
            None

        Returns:
            None
        """
        fluxes = {}
        flux_errors = {}
        upper_limits = {}  # Track which measurements are upper limits
        molecular_data = {
            'wavelengths': {},
            'Aul': {},
            'Eu': {},
            'gu': {}
        }
        
        rows_data = []  # (line, flux, flux_error, is_upper_limit), one per selected row
        for row in range(self.lines_table.rowCount()):
            select_item = self.lines_table.item(row, 0)
            if not select_item or select_item.checkState() != Qt.Checked:
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
            ul_item = self.lines_table.item(row, 9)
            is_upper_limit = bool(ul_item and ul_item.checkState() == Qt.Checked)

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
                # sync_flux_error_flag() blanks these cells and stashes the real
                # value in Qt.UserRole when Upper Limit is checked -- recover it here.
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
            
            rows_data.append((line, flux, flux_error, is_upper_limit))

        if not rows_data:
            QMessageBox.warning(self, "Error", "No lines selected")
            return

        # H2 keys by the line's own j_lower (matches the notebook), so
        # apply_OPR's ortho/para check is correct for any line selection.
        try:
            keys = _line_keys([r[0] for r in rows_data], self.current_molecule)
        except ValueError as exc:
            QMessageBox.warning(self, "Error", str(exc))
            return

        for key, (line, flux, flux_error, is_upper_limit) in zip(keys, rows_data):
            fluxes[key] = flux
            if flux_error is not None:
                flux_errors[key] = flux_error
            if is_upper_limit:
                upper_limits[key] = True

            molecular_data['wavelengths'][key] = line['wavelength']
            molecular_data['Aul'][key] = line.get('a_coeff', 0)
            molecular_data['Eu'][key] = line.get('e_upper', 0)

            j_lower = line.get('j_lower')
            if self.current_molecule == 'H2' and line.get('branch') == 'S' and j_lower is not None:
                molecular_data['gu'][key] = get_gu(j_lower)
            else:
                molecular_data['gu'][key] = line.get('Gp', 1)
        
        # Check that R and distance are provided (R can be left unknown on purpose)
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
        self.run_map_btn.setEnabled(False)

        self.rot_fit_worker = RotationalFitWorker(
            fluxes, flux_errors, molecular_data, fit_opr, opr_grid,
            fit_2t, radius_au, distance_pc, upper_limits,
            molecule=self.current_molecule, isotopologue=self.isotopologue_spin.value(),
        )
        self.rot_fit_worker.finished.connect(self._on_rotational_fit_complete)
        self.rot_fit_worker.error.connect(self._on_rotational_fit_error)
        self.rot_fit_worker.progress.connect(self._on_rotational_fit_progress)
        self.rot_fit_worker.start()

    def _on_rotational_fit_progress(self, phase, current, total):
        """Mirror the worker's nutpie sampling progress into the progress dialog.

        Args:
            phase (str): Fit phase, '1T' or '2T'.
            current (int): Current sampling step.
            total (int): Total sampling steps.

        Returns:
            None
        """
        dlg = getattr(self, 'rot_fit_progress', None)
        if dlg is None:
            return
        # In 2T mode, the "1T" phase is just 2T's own silent seed -- label it as
        # such, not "Running 1T fit", which would suggest an unreported 1T result.
        worker = getattr(self, 'rot_fit_worker', None)
        if phase == "1T" and worker is not None and getattr(worker, 'fit2T', False):
            label = "Preparing 2T fit"
        else:
            label = f"Running {phase} fit"

        if total <= 0:
            # upper_limit_1T/2T send (0, 0) before the ~20-50s nutpie compile
            # step, which has no sub-progress hook -- show a busy bar instead.
            dlg.setRange(0, 0)
            dlg.setLabelText(f"{label}: compiling model (nutpie)...")
            return

        dlg.setRange(0, 100)
        pct = int(100 * current / total)
        dlg.setLabelText(f"{label} (nutpie)...\n{current}/{total} ({pct}%)")
        dlg.setValue(pct)

    def _on_rotational_fit_complete(self, results):
        """Show the completed rotational-diagram fit's results.

        Args:
            results (dict): Fit results from RotationalFitWorker.

        Returns:
            None
        """
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
        """Show a failed rotational-diagram fit's error message.

        Args:
            message (str): Error message from RotationalFitWorker.

        Returns:
            None
        """
        dlg = getattr(self, 'rot_fit_progress', None)
        if dlg is not None:
            dlg.close()
        self.on_line_selected()  # re-derive from current checkbox state (>=2), don't just force True
        QMessageBox.critical(self, "Fitting Error", f"Error during fitting:\n{message}")

    def run_rotational_map(self):
        """Run the same fit at every pixel of the loaded cube(s), 1T,
        detections only. A single cube fits on its own native grid; two or
        more always convolve/reproject onto a common WCS, even if they
        already look aligned -- registration is verified by construction,
        never assumed from a tolerance check.

        Args:
            None

        Returns:
            None
        """
        cubes = list(self.loaded_batch_cubes)
        if not cubes:
            QMessageBox.warning(self, "No Cube",
                                "Load one or more FITS cubes first (Load FITS Files).")
            return

        line_table, discarded_ul, err = self._rot_map_gather_lines()
        if err:
            QMessageBox.warning(self, "Lines", err)
            return

        if discarded_ul:
            wavelengths = "\n".join(f"  {l['wavelength']:.4f} um" for l in discarded_ul)
            proceed = QMessageBox.question(
                self, "Upper Limits Not Supported",
                "Lines marked Upper Limit will not be considered for the "
                f"Rotational Diagram Map:\n{wavelengths}\n\n"
                f"Continue with the remaining {len(line_table)} line(s)?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if proceed != QMessageBox.Yes:
                return

        multi_mode = False
        cubes_and_lines = None
        if len(cubes) > 1:
            # Always convolve/reproject for >1 cube -- never assume two cubes
            # are already on one grid from a tolerance check (see
            # convolution_plan/create_rotational_diagram_map_multi).
            cubes_and_lines, match_err = self._rot_map_match_lines_to_cubes(cubes, line_table)
            if match_err:
                QMessageBox.warning(self, "Cubes", match_err)
                return
            multi_mode = True

        fit_kwargs = self._rot_map_gather_settings()
        if fit_kwargs is None:
            return

        cube_arg = cubes[0] if len(cubes) == 1 else cubes

        # Fast path: Calculate Maps + Convolve Maps already produced a
        # convolved result for these same cubes/lines -- skip straight to the
        # cheap final combination instead of the full slow worker. Only the
        # geometry (cubes/lines) needs to match; fit settings (distance,
        # min_lines, OPR, ...) can differ freely since this stage recomputes
        # them from the cached, already-convolved maps.
        if multi_mode:
            geometry_key = self._rot_map_geometry_key(cubes_and_lines)
            convolved_cache = getattr(self, "_rot_map_convolved_cache", None)
            if convolved_cache is not None and convolved_cache["key"] == geometry_key:
                self._run_map_fast_combine(cubes, cubes_and_lines, convolved_cache["result"], fit_kwargs)
                return

        # Reuse a cached result if the same cubes/lines/settings produced one.
        # Stashed since multi-mode nulls self._rot_map_line_table below.
        cache = getattr(self, "_rot_map_cache", None)
        cache_key = self._rot_map_cache_key(cubes, line_table, multi_mode, fit_kwargs)
        self._rot_map_pending_cache_key = cache_key
        if cache is not None and cache["key"] == cache_key:
            reply = QMessageBox.question(
                self, "Cached Results Available",
                "A map with these same cubes, lines, and settings is already "
                "cached.\n\n"
                "Click 'Yes' to reopen the cached results (instant)\n"
                "Click 'No' to recompute",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self._rot_map_multi_mode = cache["multi_mode"]
                self._rot_map_cube_arg = cache["cube_arg"]
                self._rot_map_line_table = cache["line_table"]
                self._rot_map_fit_kwargs = cache["fit_kwargs"]
                self._populate_map_column(cache["result"])
                self._populate_convolved_map_column(cache["result"])
                self._show_rot_map_results(cache["result"])
                return

        # Confirm before the run: a cheap, fit-free check, nothing that scales
        # with pixel count. The real region-fit cross-check runs later, in the worker.
        if multi_mode:
            try:
                plan = convolution_plan(cubes_and_lines)
            except Exception as exc:
                QMessageBox.critical(self, "Plan Failed", str(exc))
                return
            if not self._show_convolution_plan_confirm(plan):
                return
        else:
            # Only reachable with exactly one cube now -- >1 always takes the
            # multi_mode branch above.
            n_y, n_x = cubes[0].cube.shape[1], cubes[0].cube.shape[2]
            proceed = QMessageBox.question(
                self, "Run Map",
                f"Run the fit for every pixel of the {n_y}x{n_x} px cube, "
                f"{len(line_table)} line(s)?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if proceed != QMessageBox.Yes:
                return

        self._rot_map_multi_mode = multi_mode
        self._rot_map_cube_arg = None if multi_mode else cube_arg
        self._rot_map_line_table = None if multi_mode else line_table
        self._rot_map_fit_kwargs = fit_kwargs

        self.rot_map_progress = QProgressDialog(
            "Fitting rotational diagram per pixel...", "Cancel", 0, 0, self)
        self.rot_map_progress.setWindowTitle("Rotational Diagram Map")
        self.rot_map_progress.setWindowModality(Qt.WindowModal)
        self.rot_map_progress.setAutoClose(False)
        self.rot_map_progress.setAutoReset(False)
        self.rot_map_progress.canceled.connect(self._cancel_rot_map)
        self.rot_map_progress.show()

        self.run_fit_btn.setEnabled(False)
        self.run_map_btn.setEnabled(False)

        if multi_mode:
            self.rot_map_worker = RotDiagMapWorker(
                cubes_and_lines=cubes_and_lines, **fit_kwargs)
        else:
            self.rot_map_worker = RotDiagMapWorker(cube_arg, line_table, **fit_kwargs)
        self.rot_map_worker.progress.connect(self._on_rot_map_progress)
        self.rot_map_worker.finished.connect(self._on_rot_map_complete)
        self.rot_map_worker.error.connect(self._on_rot_map_error)
        self.rot_map_worker.start()

    def _run_map_fast_combine(self, cubes, cubes_and_lines, convolved_result, fit_kwargs):
        """Run Map's fast path: recombine an already-convolved result
        (Calculate Maps + Convolve Maps already ran for these cubes/lines)
        into a temperature map, skipping native extraction and convolution
        entirely -- just the cheap final per-pixel combination.

        Args:
            cubes (list): Currently loaded CubeData objects, for the final cache key
            cubes_and_lines (list): (CubeData, line_dict) pairs matching convolved_result
            convolved_result (dict): Output of convolve_line_maps
            fit_kwargs (dict): molecule/isotopologue/opr/fit_opr/opr_grid/dist_pc/r_au/min_lines

        Returns:
            None
        """
        line_table = [line for _, line in cubes_and_lines]

        self.rot_map_progress = QProgressDialog(
            "Fitting rotational diagram per pixel (reusing calculated maps)...",
            "Cancel", 0, 0, self)
        self.rot_map_progress.setWindowTitle("Rotational Diagram Map")
        self.rot_map_progress.setWindowModality(Qt.WindowModal)
        self.rot_map_progress.setAutoClose(False)
        self.rot_map_progress.setAutoReset(False)
        self.rot_map_progress.canceled.connect(self._cancel_rot_map)
        self.rot_map_progress.show()

        self.run_fit_btn.setEnabled(False)
        self.run_map_btn.setEnabled(False)

        self._rot_map_multi_mode = True
        self._rot_map_cube_arg = None
        self._rot_map_line_table = None
        self._rot_map_fit_kwargs = fit_kwargs
        self._rot_map_pending_cache_key = self._rot_map_cache_key(cubes, line_table, True, fit_kwargs)

        # Reuses _on_rot_map_complete/_on_rot_map_error as-is -- they only
        # care that the result dict matches create_rotational_diagram_map_multi's
        # shape, not how it was computed.
        self.rot_map_worker = CombineMapsWorker(convolved_result, cubes_and_lines, **fit_kwargs)
        self.rot_map_worker.progress.connect(self._on_rot_map_progress)
        self.rot_map_worker.finished.connect(self._on_rot_map_complete)
        self.rot_map_worker.error.connect(self._on_rot_map_error)
        self.rot_map_worker.start()

    def calculate_rotational_maps(self):
        """Compute each checked line's own flux map without combining them
        into a temperature fit yet -- lets the user inspect what Run Map
        would actually use before running it, and caches the result so Run
        Map can reuse it instead of recomputing.

        Single cube: there is no separable expensive/cheap split (the
        T-combination is already a cheap add-on to the same per-pixel fit),
        so this just runs create_rotational_diagram_map in full and caches
        the whole result -- the speedup Run Map gets afterward comes purely
        from finding a matching cache, exactly like it already does today.

        2+ cubes: runs only the native-resolution extraction
        (compute_native_line_maps), the genuinely expensive stage; Convolve
        Maps and Run Map build on this without repeating it.

        Args:
            None

        Returns:
            None
        """
        cubes = list(self.loaded_batch_cubes)
        if not cubes:
            QMessageBox.warning(self, "No Cube",
                                "Load one or more FITS cubes first (Load FITS Files).")
            return

        line_table, discarded_ul, err = self._rot_map_gather_lines(min_lines=1)
        if err:
            QMessageBox.warning(self, "Lines", err)
            return

        if discarded_ul:
            wavelengths = "\n".join(f"  {l['wavelength']:.4f} um" for l in discarded_ul)
            proceed = QMessageBox.question(
                self, "Upper Limits Not Supported",
                "Lines marked Upper Limit will not be considered for Calculate "
                f"Maps:\n{wavelengths}\n\nContinue with the remaining {len(line_table)} line(s)?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if proceed != QMessageBox.Yes:
                return

        if len(cubes) > 1:
            cubes_and_lines, match_err = self._rot_map_match_lines_to_cubes(cubes, line_table)
            if match_err:
                QMessageBox.warning(self, "Cubes", match_err)
                return
        else:
            fit_kwargs = self._rot_map_gather_settings()
            if fit_kwargs is None:
                return

        self.rot_map_progress = QProgressDialog(
            "Calculating flux maps per pixel...", "Cancel", 0, 0, self)
        self.rot_map_progress.setWindowTitle("Calculate Maps")
        self.rot_map_progress.setWindowModality(Qt.WindowModal)
        self.rot_map_progress.setAutoClose(False)
        self.rot_map_progress.setAutoReset(False)
        self.rot_map_progress.canceled.connect(self._cancel_rot_map)
        self.rot_map_progress.show()

        self.run_fit_btn.setEnabled(False)
        self.run_map_btn.setEnabled(False)
        self.calculate_maps_btn.setEnabled(False)
        self.convolve_maps_btn.setEnabled(False)

        if len(cubes) == 1:
            self._calc_maps_cube = cubes[0]
            self._calc_maps_line_table = line_table
            self._calc_maps_fit_kwargs = fit_kwargs
            self.rot_map_worker = RotDiagMapWorker(cubes[0], line_table, **fit_kwargs)
            self.rot_map_worker.progress.connect(self._on_rot_map_progress)
            self.rot_map_worker.finished.connect(self._on_calculate_maps_single_complete)
            self.rot_map_worker.error.connect(self._on_calculate_maps_error)
        else:
            self._calc_maps_cubes_and_lines = cubes_and_lines
            self.rot_map_worker = CalculateNativeMapsWorker(
                cubes_and_lines, contam_lines=self.contam_lines)
            self.rot_map_worker.progress.connect(self._on_rot_map_progress)
            self.rot_map_worker.finished.connect(self._on_calculate_maps_multi_complete)
            self.rot_map_worker.error.connect(self._on_calculate_maps_error)
        self.rot_map_worker.start()

    def _on_calculate_maps_single_complete(self, result):
        """Cache a single-cube Calculate Maps result the same shape Run
        Map's own cache expects, then populate the Map column -- no results
        dialog, unlike Run Map itself.

        Args:
            result: Computed create_rotational_diagram_map result.

        Returns:
            None
        """
        dlg = getattr(self, "rot_map_progress", None)
        if dlg is not None:
            dlg.close()
        self.on_line_selected()
        cache_key = self._rot_map_cache_key(
            [self._calc_maps_cube], self._calc_maps_line_table, False, self._calc_maps_fit_kwargs)
        self._rot_map_cache = {
            "key": cache_key, "result": result, "multi_mode": False,
            "cube_arg": self._calc_maps_cube, "line_table": self._calc_maps_line_table,
            "fit_kwargs": self._calc_maps_fit_kwargs,
        }
        self._populate_map_column(result)
        QMessageBox.information(
            self, "Calculate Maps",
            "Flux maps calculated. View them in the Map column, or click Run Map "
            "to fit the temperature (instant -- reuses this result).")

    def _on_calculate_maps_multi_complete(self, result):
        """Cache a multi-cube Calculate Maps (native-resolution) result and
        populate the Map column.

        Args:
            result: Computed compute_native_line_maps result.

        Returns:
            None
        """
        dlg = getattr(self, "rot_map_progress", None)
        if dlg is not None:
            dlg.close()
        self.on_line_selected()
        geometry_key = self._rot_map_geometry_key(self._calc_maps_cubes_and_lines)
        self._rot_map_native_cache = {"key": geometry_key, "result": result}
        self._populate_map_column(result)
        QMessageBox.information(
            self, "Calculate Maps",
            "Native-resolution flux maps calculated. View them in the Map column, "
            "or click Convolve Maps to put them on a common grid.")

    def _on_calculate_maps_error(self, message):
        """Close progress, re-enable the buttons, and report the error.

        Args:
            message (str): Error message from the Calculate Maps worker.

        Returns:
            None
        """
        dlg = getattr(self, "rot_map_progress", None)
        if dlg is not None:
            dlg.close()
        self.on_line_selected()
        if "cancelled" in message.lower():
            QMessageBox.information(self, "Cancelled", "Calculate Maps cancelled.")
        else:
            QMessageBox.critical(self, "Calculate Maps Error", f"Error during Calculate Maps:\n{message}")

    def convolve_rotational_maps(self):
        """Convolve/reproject every checked line's already-calculated native
        map onto a common grid at the worst (coarsest) resolution among
        them. Requires 2+ loaded cubes and a matching Calculate Maps result
        already cached. Runs synchronously -- vectorized convolution and
        reprojection, not a per-pixel Python loop, so no progress dialog is
        needed.

        Args:
            None

        Returns:
            None
        """
        cubes = list(self.loaded_batch_cubes)
        if len(cubes) < 2:
            QMessageBox.warning(
                self, "Convolve Maps",
                "Convolve Maps needs 2+ loaded cubes -- with only one, there is "
                "only one resolution and nothing to convolve.")
            return

        line_table, discarded_ul, err = self._rot_map_gather_lines(min_lines=1)
        if err:
            QMessageBox.warning(self, "Lines", err)
            return

        cubes_and_lines, match_err = self._rot_map_match_lines_to_cubes(cubes, line_table)
        if match_err:
            QMessageBox.warning(self, "Cubes", match_err)
            return

        geometry_key = self._rot_map_geometry_key(cubes_and_lines)
        native_cache = getattr(self, "_rot_map_native_cache", None)
        if native_cache is None or native_cache["key"] != geometry_key:
            QMessageBox.warning(
                self, "Convolve Maps",
                "No matching Calculate Maps result for these cubes/lines -- "
                "run Calculate Maps first.")
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            convolved = convolve_line_maps(native_cache["result"])
        except Exception as exc:
            QMessageBox.critical(self, "Convolve Maps Error", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._rot_map_convolved_cache = {"key": geometry_key, "result": convolved}
        self._populate_convolved_map_column(convolved)
        QMessageBox.information(
            self, "Convolve Maps",
            "Maps convolved onto the common grid. View them in the Convolved Map "
            "column, or click Run Map to fit the temperature.")

    def _rot_map_match_lines_to_cubes(self, cubes, line_table):
        """Match each checked line to the loaded cube whose native wavelength
        range covers it (closest to that cube's centre wavelength if more
        than one covers it).

        Args:
            cubes (list): Loaded CubeData objects.
            line_table (list): Checked line rows to match.

        Returns:
            result (tuple): (pairs, error) -- pairs is a list of (line, cube)
                matches; error names any line no loaded cube covers, or None.
        """
        pairs, unmatched = [], []
        for line in line_table:
            wl = line["wavelength"]
            candidates = []
            for cube in cubes:
                cwl = cube.wavelength
                cwl = cwl.value if hasattr(cwl, "value") else np.asarray(cwl)
                lo, hi = float(np.nanmin(cwl)), float(np.nanmax(cwl))
                if lo <= wl <= hi:
                    candidates.append((abs(wl - (lo + hi) / 2.0), cube))
            if not candidates:
                unmatched.append(wl)
                continue
            candidates.sort(key=lambda c: c[0])
            pairs.append((candidates[0][1], line))
        if unmatched:
            wl_text = ", ".join(f"{w:.4f}" for w in unmatched)
            return None, f"No loaded cube covers these line(s) (um): {wl_text}."
        for cube, line in pairs:
            print(f"[rot map] {line['wavelength']:.4f} um -> "
                  f"{getattr(cube, 'filename', 'cube')}")
        return pairs, None

    def _rot_map_gather_lines(self, min_lines=3):
        """Split checked lines into fittable ones and discarded upper limits,
        since the map doesn't support upper limits.

        Args:
            min_lines (int): Fewest non-upper-limit lines required. Run Map
                needs 3 (a 1T fit needs 3+ points); Calculate Maps/Convolve
                Maps only need 1, since there is no temperature fit yet.

        Returns:
            result (tuple): (line_table, discarded_ul, error) -- line_table
                holds checked, non-upper-limit lines; discarded_ul holds
                checked lines also marked Upper Limit; error is a message
                string, or None.
        """
        rows = []
        discarded_ul = []
        for row in range(self.lines_table.rowCount()):
            item = self.lines_table.item(row, 0)
            if not (item and item.checkState() == Qt.Checked):
                continue
            ul_item = self.lines_table.item(row, 9)
            if ul_item and ul_item.checkState() == Qt.Checked:
                discarded_ul.append(self.hitran_lines[row])
            else:
                rows.append(self.hitran_lines[row])
        if len(rows) < min_lines:
            return None, discarded_ul, f"Select at least {min_lines} non-upper-limit line(s)."
        missing = [r for r in rows if r.get("a_coeff") is None or r.get("e_upper") is None]
        if missing:
            return None, discarded_ul, ("Some checked lines have no A_ul/E_u (no HITRAN match). "
                          "Deselect them or use lines with catalog data.")
        return rows, discarded_ul, None

    def _rot_map_gather_settings(self):
        """Parse distance/radius/OPR-grid inputs into fit_kwargs, the same
        way Run Map always has. Shared by Run Map, Calculate Maps (single
        cube only needs this), and the fast Run Map combine path.

        Args:
            None

        Returns:
            fit_kwargs (dict or None): molecule/isotopologue/opr/fit_opr/
                opr_grid/dist_pc/r_au/min_lines, or None if a field was
                invalid -- a warning was already shown in that case.
        """
        try:
            dist_pc = float(self.distance_input.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Distance", "Enter a valid distance (pc).")
            return None
        if self.radius_unknown_check.isChecked():
            r_au = None
        else:
            try:
                r_au = float(self.radius_input.text().strip())
            except ValueError:
                QMessageBox.warning(self, "Radius", "Enter a radius (AU), or check Unknown R.")
                return None

        fit_opr = self.fit_opr_check.isChecked()
        opr_grid = None
        if fit_opr:
            try:
                opr_grid = np.linspace(
                    float(self.opr_min_input.text().strip()),
                    float(self.opr_max_input.text().strip()),
                    int(self.opr_steps_input.text().strip()))
            except ValueError:
                QMessageBox.warning(self, "OPR Grid", "Invalid OPR min/max/steps.")
                return None

        return dict(
            molecule=self.current_molecule,
            isotopologue=self.isotopologue_spin.value(),
            opr=None, fit_opr=fit_opr, opr_grid=opr_grid,
            dist_pc=dist_pc, r_au=r_au, min_lines=3,
        )

    def _rot_map_geometry_key(self, cubes_and_lines, dedup_tol_um=1e-5):
        """Cache key for compute_native_line_maps/convolve_line_maps output:
        only what determines the native/convolved maps themselves (which
        cube is paired with which line's wavelength), not fit settings like
        distance or min_lines -- those two stages never see them.

        Args:
            cubes_and_lines (list): (CubeData, line_dict) pairs
            dedup_tol_um (float): Co-located-line merge tolerance in microns

        Returns:
            key (tuple): Hashable, comparable snapshot
        """
        pairs = tuple(sorted(
            (getattr(c, "filename", None), round(l["wavelength"], 6)) for c, l in cubes_and_lines
        ))
        return (pairs, dedup_tol_um)

    def _show_convolution_plan_confirm(self, plan):
        """Show the convolve/reproject plan (no fitting); return True to run the map.

        Args:
            plan (dict): Convolution/reprojection plan from convolution_plan().

        Returns:
            proceed (bool): True if the user chose to run the map.
        """
        ny, nx = plan["dst_shape"]
        worst = plan["worst_pix_arcsec"]
        lines = "\n".join(
            f"  {c['wavelength']:>9.4f} um  {c['filename'] or 'cube':<28}  "
            f"native {c['native_pix_arcsec']:.4f}\"/px  "
            + ("sets the resolution" if c["sigma_pix"] <= 1e-6 else
               f"blur sigma {c['sigma_arcsec']:.4f}\" ({c['sigma_pix']:.2f} px)")
            for c in plan["per_cube"]
        )
        text = (
            f"Convolution / reprojection plan\n{'=' * 50}\n\n"
            f"Common grid: {ny} x {nx} px @ {worst:.4f}\"/px (coarsest input)\n\n"
            f"{lines}\n\n"
            "Every pixel outside a cube's own reprojected footprint is NaN "
            "for that line.\n\nRun the fit at every pixel of the common grid?"
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Run Map -- convolution plan")
        dlg.resize(640, 420)
        lay = QVBoxLayout(dlg)
        body = QLabel(text)
        body.setStyleSheet("font-family: monospace;")
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(body)
        btns = QHBoxLayout()
        run_btn = QPushButton("Run Map")
        cancel_btn = QPushButton("Cancel")
        run_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btns.addStretch()
        btns.addWidget(run_btn)
        btns.addWidget(cancel_btn)
        lay.addLayout(btns)
        return dlg.exec_() == QDialog.Accepted

    def _on_rot_map_progress(self, current, total):
        """Update the map progress dialog.

        Args:
            current (int): Rows completed so far.
            total (int): Total rows to process.

        Returns:
            None
        """
        dlg = getattr(self, "rot_map_progress", None)
        if dlg is None:
            return
        dlg.setRange(0, total)
        dlg.setValue(current)
        dlg.setLabelText(f"Fitting rotational diagram: {current}/{total} pixels")

    def _rot_map_cache_key(self, cubes, line_table, multi_mode, fit_kwargs):
        """Snapshot of everything that determines Run Map's output, so a
        repeat click with identical inputs can reuse a cached result instead
        of recomputing.

        Args:
            cubes (list): Currently loaded CubeData objects
            line_table (list): Line dicts Run Map would use
            multi_mode (bool): Whether cubes need convolution/reprojection
            fit_kwargs (dict): molecule/isotopologue/opr/fit_opr/opr_grid/
                dist_pc/r_au/min_lines, as built in run_rotational_map

        Returns:
            key (tuple): Hashable, comparable snapshot
        """
        filenames = tuple(getattr(c, "filename", None) for c in cubes)
        wavelengths = tuple(sorted(round(l["wavelength"], 6) for l in line_table))
        kwargs_items = []
        for k in sorted(fit_kwargs):
            v = fit_kwargs[k]
            if isinstance(v, np.ndarray):
                v = tuple(np.round(v, 6).tolist())
            kwargs_items.append((k, v))
        return (multi_mode, filenames, wavelengths, tuple(kwargs_items))

    def _show_rot_map_results(self, result):
        """Build and show the map results dialog for an already-computed result.

        Args:
            result: Computed rotational-diagram map result.

        Returns:
            None
        """
        RotDiagMapResultsDialog(
            result, self._rot_map_cube_arg, self._rot_map_line_table,
            self._rot_map_fit_kwargs, parent=self,
        ).exec_()

    def _on_rot_map_complete(self, result):
        """Close progress, re-enable the buttons, cache the result so closing
        the results window doesn't force a recompute, and show it.

        Args:
            result: Computed rotational-diagram map result.

        Returns:
            None
        """
        dlg = getattr(self, "rot_map_progress", None)
        if dlg is not None:
            dlg.close()
        self.on_line_selected()
        self._rot_map_cache = {
            "key": self._rot_map_pending_cache_key,
            "result": result,
            "multi_mode": self._rot_map_multi_mode,
            "cube_arg": self._rot_map_cube_arg,
            "line_table": self._rot_map_line_table,
            "fit_kwargs": self._rot_map_fit_kwargs,
        }
        self._populate_map_column(result)
        self._populate_convolved_map_column(result)
        self._show_rot_map_results(result)

    def _on_rot_map_error(self, message):
        """Close progress, re-enable the buttons, and report the error.

        Args:
            message (str): Error message from the map worker.

        Returns:
            None
        """
        dlg = getattr(self, "rot_map_progress", None)
        if dlg is not None:
            dlg.close()
        self.on_line_selected()
        if "cancelled" in message.lower():
            QMessageBox.information(self, "Cancelled", "Rotational diagram map cancelled.")
        else:
            QMessageBox.critical(self, "Map Error", f"Error during map:\n{message}")

    def _cancel_rot_map(self):
        """Ask the map worker to stop at the next row.

        Args:
            None

        Returns:
            None
        """
        worker = getattr(self, "rot_map_worker", None)
        if worker is not None:
            worker.request_cancel()
        dlg = getattr(self, "rot_map_progress", None)
        if dlg is not None:
            dlg.setLabelText("Cancelling...")

    def show_results(self, results):
        """Show results with plot.

        Args:
            results (dict): Fit results to display.

        Returns:
            None
        """
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
            an upper-limit-barrier fit.

            Args:
                fit_dict (dict): One fit's result dict.

            Returns:
                note (str): Formatted diagnostic line, or '' if not a PyMC fit.
            """
            n_div = fit_dict.get("n_divergences")
            if n_div is None:
                return ""
            return f"[PyMC: {n_div} divergence(s), max r_hat = {fit_dict.get('max_r_hat', float('nan')):.3f}]\n"

        if "1T" in results:
            r1t = results["1T"]
            results_text += "--- 1T FIT ---\n"
            results_text += f"T = {r1t['T']:.1f} ± {r1t['dT']:.1f} K\n"
            results_text += f"Ncol = {r1t['Ncol']:.2e} ± {r1t['dNcol']:.2e} cm⁻²\n"
            results_text += _convergence_note(r1t)
            results_text += "\n"

        if "2T" in results:
            r2t = results["2T"]
            results_text += "--- 2T FIT ---\n"
            results_text += f"T1 = {r2t['T1']:.1f} ± {r2t.get('dT1', float('nan')):.1f} K\n"
            results_text += f"T2 = {r2t['T2']:.1f} ± {r2t.get('dT2', float('nan')):.1f} K\n"
            results_text += f"N1 = {r2t['Ncol1']:.2e} ± {r2t.get('dNcol1', float('nan')):.2e} cm⁻²\n"
            results_text += f"N2 = {r2t['Ncol2']:.2e} ± {r2t.get('dNcol2', float('nan')):.2e} cm⁻²\n"
            results_text += _convergence_note(r2t)

        results_text += "="*50 + "\n"

        # results_text (this exact string) is what the GUI text widget below shows
        console_text = (
            results_text
            .replace("±", "+/-")
            .replace("σ", "sigma")
            .replace("≈", "~=")
            .replace("⚠", "WARNING:")
            .replace("⁻²", "-2")
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

            # Detections -- black circles with error bars
            if np.any(det):
                ax.errorbar(E_arr[det], y_arr_log10[det], yerr=y_err_log10[det],
                            fmt='o', color='black', markersize=6, capsize=4, zorder=3,
                            label='Detection')

            # Upper limits -- red downward arrows, no error bar
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
