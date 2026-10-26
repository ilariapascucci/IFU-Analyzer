"""
Rotational Diagram Analysis GUI for Molecular data (H2, CO) - SIMPLIFIED
Single unified interface for selecting molecules, lines, and entering flux data
"""

import os
import csv
import io
import time
import zipfile
import numpy as np
from astropy.stats import gaussian_sigma_to_fwhm

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QSpinBox,
    QDoubleSpinBox, QCheckBox, QMessageBox, QProgressDialog,
    QFrame, QComboBox, QDialog, QTextEdit, QFileDialog, QLineEdit, QHeaderView, QInputDialog,
    QStyledItemDelegate
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QBrush
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .gui import CollapsibleButtonGroup
from .data_loader import CubeData
from .line_catalog import fetch_hitran_data, load_contaminating_lines_catalog, match_nearest_hitran_line
from .table_drag_select import SortableCheckboxTableMixin
from . import region_io
from .region_selector import mask_from_shape_coords
from .rotational_diagram_h2_fitter import (
    rotational_diagram as rotational_diagram_h2,
    get_gu, compute_line_fluxes_collective, model_1T, model_2T,
)
from .rotational_diagram_general_fitter import rotational_diagram as rotational_diagram_general
from .flux_map_extractor import DISPLAY_CROP_SIGMA_MULTIPLIER, gaussian
from .rotational_diagram_map import (
    _line_keys, build_imported_map_payload, merge_native_results,
)
from .import_map_dialog import _peek_header, _has_real_error
from .rotational_diagram_map_dialogs import (
    RotDiagMapWorker, RotDiagMapResultsDialog, LineFluxMapDialog,
    CalculateNativeMapsWorker, CombineMapsWorker, SelectCubeRegionDialog,
    attach_rot_diagram_hover, save_fit_csv, input_lines,
)
from .fits_export import build_line_map_header, write_fits_product
from .scale_dialog import ScaleSettings, build_norm
from .visualization import draw_map
from .convolve_reproject import convolve_reproject_maps, convolution_plan
from .utils import psf_fwhm_arcsec, make_2d_wcs, get_pixel_size_arcsec
from .icon_loader import set_button_icon

# Every open window registers here, so Python keeps it alive while it is open.
# A closed window leaves, so its loaded cubes can be freed.
_OPEN_WINDOWS = set()


# Constants
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


class FetchHitranDialog(QDialog):
    """Dialog to configure one HITRAN fetch: molecule, wavelength range,
    isotopologue, and vibrational transition filtering."""

    def __init__(self, parent=None, initial_molecule='H2', initial_isotopologue=1):
        """Build the Fetch from HITRAN prompt.

        Args:
            parent (QWidget): Parent widget, or None for a top-level dialog.
            initial_molecule (str): Molecule pre-selected in the dropdown.
            initial_isotopologue (int): Isotopologue pre-selected, clamped to
                the chosen molecule's own valid range.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle("Fetch from HITRAN")

        layout = QVBoxLayout()

        mol_row = QHBoxLayout()
        mol_row.addWidget(QLabel("Molecule:"))
        self.mol_combo = QComboBox()
        self.mol_combo.addItems(list(MOLECULE_ISOTOPOLOGUES.keys()))
        self.mol_combo.setCurrentText(initial_molecule)
        self.mol_combo.currentTextChanged.connect(self._update_isotopologue_range)
        mol_row.addWidget(self.mol_combo)
        mol_row.addStretch()
        layout.addLayout(mol_row)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("λ_min (μm):"))
        self.wl_min_input = QDoubleSpinBox()
        self.wl_min_input.setRange(0.1, 1000)
        self.wl_min_input.setValue(5.0)
        self.wl_min_input.setDecimals(6)
        range_row.addWidget(self.wl_min_input)
        range_row.addWidget(QLabel("λ_max (μm):"))
        self.wl_max_input = QDoubleSpinBox()
        self.wl_max_input.setRange(0.1, 1000)
        self.wl_max_input.setValue(20.0)
        self.wl_max_input.setDecimals(6)
        range_row.addWidget(self.wl_max_input)
        layout.addLayout(range_row)

        iso_row = QHBoxLayout()
        iso_row.addWidget(QLabel("Isotopologue:"))
        self.isotopologue_spin = QSpinBox()
        iso_row.addWidget(self.isotopologue_spin)
        iso_row.addStretch()
        layout.addLayout(iso_row)
        self._update_isotopologue_range(initial_molecule)
        self.isotopologue_spin.setValue(initial_isotopologue)

        v_row = QHBoxLayout()
        v_row.addWidget(QLabel("v_upper:"))
        self.v_upper_spin = QSpinBox()
        self.v_upper_spin.setRange(0, 10)
        v_row.addWidget(self.v_upper_spin)
        v_row.addWidget(QLabel("v_lower:"))
        self.v_lower_spin = QSpinBox()
        self.v_lower_spin.setRange(0, 10)
        v_row.addWidget(self.v_lower_spin)
        self.include_all_trans_check = QCheckBox("All transitions")
        self.include_all_trans_check.setChecked(True)
        self.include_all_trans_check.stateChanged.connect(self._on_all_trans_toggled)
        v_row.addWidget(self.include_all_trans_check)
        layout.addLayout(v_row)
        self.v_upper_spin.setEnabled(False)
        self.v_lower_spin.setEnabled(False)

        btn_row = QHBoxLayout()
        fetch_btn = QPushButton("Fetch")
        cancel_btn = QPushButton("Cancel")
        fetch_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(fetch_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def _update_isotopologue_range(self, molecule):
        """Reset the isotopologue spinner to the chosen molecule's own range.

        Args:
            molecule (str): Newly selected molecule name.

        Returns:
            None
        """
        mol_data = MOLECULE_ISOTOPOLOGUES.get(molecule)
        if mol_data:
            min_iso, max_iso = mol_data['range']
            self.isotopologue_spin.setRange(min_iso, max_iso)
            self.isotopologue_spin.setValue(min_iso)

    def _on_all_trans_toggled(self):
        """Enable v_upper/v_lower only when not fetching all transitions.

        Args:
            None

        Returns:
            None
        """
        checked = self.include_all_trans_check.isChecked()
        self.v_upper_spin.setEnabled(not checked)
        self.v_lower_spin.setEnabled(not checked)

    def get_values(self):
        """Read the dialog's current fields.

        Args:
            None

        Returns:
            values (tuple): (molecule, wl_min, wl_max, isotopologue, v_upper,
                v_lower, include_all) -- v_upper/v_lower are None when
                include_all is True.
        """
        include_all = self.include_all_trans_check.isChecked()
        return (
            self.mol_combo.currentText(), self.wl_min_input.value(), self.wl_max_input.value(),
            self.isotopologue_spin.value(),
            None if include_all else self.v_upper_spin.value(),
            None if include_all else self.v_lower_spin.value(),
            include_all,
        )

    def accept(self):
        """Validate the wavelength range before closing.

        Args:
            None

        Returns:
            None
        """
        if self.wl_min_input.value() >= self.wl_max_input.value():
            QMessageBox.warning(self, "Fetch from HITRAN", "Min wavelength must be less than max.")
            return
        super().accept()


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
            R (float): Emitting radius in au, or None, for flux-to-column conversion.
            dist (float): Source distance in pc.
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
            summed_spectra (dict): Summed spectrum keyed by cube index, for
                the cubes in rows_by_cube_idx (already region-restricted).
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


class LoadedHeadersDialog(QDialog):
    """Show the FITS header of any HDU of any loaded cube."""

    def __init__(self, cubes, paths, parent=None):
        """Build the cube and HDU pickers and show the first header.

        Args:
            cubes (list): Loaded CubeData objects, each with an open `hdul`.
            paths (list): File path of each cube, in the same order.
            parent (QWidget): Owner window.

        Returns:
            None
        """
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("FITS Headers")
        self.setMinimumSize(700, 500)
        self.cubes = list(cubes)

        layout = QVBoxLayout(self)
        pickers = QHBoxLayout()
        pickers.addWidget(QLabel("Cube:"))
        self.cube_combo = QComboBox()
        self.cube_combo.addItems([os.path.basename(p) for p in paths])
        self.cube_combo.currentIndexChanged.connect(self._fill_hdu_combo)
        pickers.addWidget(self.cube_combo, 1)
        pickers.addWidget(QLabel("HDU:"))
        self.hdu_combo = QComboBox()
        self.hdu_combo.currentIndexChanged.connect(self._show_header)
        pickers.addWidget(self.hdu_combo, 1)
        layout.addLayout(pickers)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFontFamily("Courier New")
        layout.addWidget(self.text)
        self._fill_hdu_combo()

    def _fill_hdu_combo(self):
        """List the HDUs of the selected cube, then show the first header.

        Args:
            None

        Returns:
            None
        """
        self.hdu_combo.blockSignals(True)
        self.hdu_combo.clear()
        hdul = getattr(self.cubes[self.cube_combo.currentIndex()], "hdul", None)
        for i, hdu in enumerate(hdul or []):
            name = (getattr(hdu, "name", "") or "").strip()
            label = f"HDU {i}: {name}" if name else f"HDU {i}:"
            self.hdu_combo.addItem(f"{label} ({type(hdu).__name__})")
        self.hdu_combo.blockSignals(False)
        self._show_header()

    def _show_header(self):
        """Show the header of the selected cube and HDU.

        Args:
            None

        Returns:
            None
        """
        hdul = getattr(self.cubes[self.cube_combo.currentIndex()], "hdul", None)
        hdu_index = self.hdu_combo.currentIndex()
        if hdul is None or hdu_index < 0:
            self.text.setPlainText("No header available for this cube.")
            return
        header = hdul[hdu_index].header
        self.text.setPlainText("\n".join(
            f"{k:8} = {v!r} / {header.comments[k]}" if k in header.comments else f"{k:8} = {v!r}"
            for k, v in header.items()
        ))


class CellEditDelegate(QStyledItemDelegate):
    """Edit delegate that makes a table cell's editor cover exactly that cell.
    The editor has an opaque background, so the old value does not show through.
    """

    def createEditor(self, parent, option, index):
        """Create the line editor for a cell, with an opaque background.

        Args:
            parent (QWidget): Parent widget for the editor.
            option (QStyleOptionViewItem): Style options of the cell.
            index (QModelIndex): Index of the cell to edit.

        Returns:
            editor (QWidget): The editor widget.
        """
        editor = super().createEditor(parent, option, index)
        if isinstance(editor, QLineEdit):
            editor.setStyleSheet(
                "QLineEdit { padding: 0px 6px; margin: 0px; background: palette(base); "
                "border: 1px solid palette(highlight); }")
        return editor

    def updateEditorGeometry(self, editor, option, index):
        """Fit the editor to the cell rectangle, ignoring the cell padding.

        Args:
            editor (QWidget): The editor to place.
            option (QStyleOptionViewItem): Style options of the cell, with its rectangle.
            index (QModelIndex): Index of the cell being edited.

        Returns:
            None
        """
        editor.setGeometry(option.rect)


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
        _OPEN_WINDOWS.add(self)
        self.current_molecule = 'H2'  # Default molecule
        self.current_isotopologue = 1  # Set by Fetch from HITRAN; read by Run Fit/Run Map
        self.setWindowTitle("Rotational Diagram Analyzer")
        screen_width = QApplication.primaryScreen().availableGeometry().width()
        self.setGeometry(100, 100, min(1650, int(screen_width * 0.95)), 800)
        
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
        self._headers_dialog = None
        self.contam_lines = None
        # Sky-coordinate region imported via "Import Region", restricting
        # Calculate Fluxes/Calculate Maps/Run Map/Run Fit to one region --
        # None fits the whole cube, the previous, unrestricted behavior.
        self._rot_map_region_sky = None

        # Track workflow: None=unset, 'csv'=import, 'hitran'=fetch
        self.workflow_mode = None
        
        # Main widget
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Header row: same Header / New Window pair and info label as the main window's top bar
        header_row = QHBoxLayout()
        self.header_btn = QPushButton("Header")
        self.header_btn.setToolTip("Show the FITS header of the loaded cubes.")
        self.header_btn.clicked.connect(self.show_loaded_headers)
        header_row.addWidget(self.header_btn)

        self.new_window_btn = QPushButton()
        set_button_icon(self.new_window_btn, "new_window")
        self.new_window_btn.setToolTip(
            "Open another rotational diagram window with nothing loaded, for example "
            "to compare two sets of cubes or lines.")
        self.new_window_btn.clicked.connect(self._open_new_window)
        header_row.addWidget(self.new_window_btn)

        def separator():
            """Build a thin vertical line that splits the header row's groups.

            Args:
                None

            Returns:
                line (QFrame): The vertical separator line.
            """
            line = QFrame()
            line.setFixedWidth(1)
            line.setStyleSheet("background-color: palette(mid);")
            return line

        header_row.addWidget(separator())

        self.molecule_label = QLabel("Molecule: --")
        self.molecule_label.setStyleSheet("font-weight: bold;")
        header_row.addWidget(self.molecule_label)
        header_row.addWidget(separator())

        lines_label = QLabel("Lines:")
        lines_label.setStyleSheet("font-weight: bold;")
        header_row.addWidget(lines_label)

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_lines)
        self.select_all_btn.setMaximumWidth(100)
        header_row.addWidget(self.select_all_btn)

        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all_lines)
        self.deselect_all_btn.setMaximumWidth(100)
        header_row.addWidget(self.deselect_all_btn)
        header_row.addStretch()
        main_layout.addLayout(header_row)

        # HITRAN parameters section
        main_layout.addWidget(self.create_hitran_section())

        # Lines table

        self.lines_table = QTableWidget()
        self.lines_table.setColumnCount(17)
        self.lines_table.setHorizontalHeaderLabels(
            ["Use", "λ (μm)", "Transition", "A_ul (s⁻¹)", "E_u (K)", "g_u", "Flux", "Flux Error",
             "Error %", "Upper Limit?", "UL Flux (3σ)", "Calculated?", "Source", "Fit R²", "Fit",
             "Map", "Convolved Map"]
        )
        # Flux/Flux Error's own tooltip is set after _init_sortable_checkbox_table instead,
        # which would otherwise overwrite it (see below).
        self.lines_table.horizontalHeaderItem(15).setToolTip(
            "Always computed on the full cube, regardless of any active region.")
        self.lines_table.horizontalHeaderItem(16).setToolTip(
            "Always computed on the full cube, regardless of any active region.")
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
        self.lines_table.setItemDelegate(CellEditDelegate(self.lines_table))
        main_layout.addWidget(self.lines_table)

        # Click-to-sort for wavelength/A_ul/E_u/g_u/Flux/Flux Error/Error %/UL Flux/Fit R^2.
        # Also drag-checks columns 0 (select) and 9 (Upper Limit?) across rows.
        self._init_sortable_checkbox_table(
            self.lines_table, {1, 3, 4, 5, 6, 7, 8, 10, 13}, self._lines_table_sort_key,
            drag_column={0, 9},
        )
        # Set after _init_sortable_checkbox_table, which overwrites every
        # sortable column's header tooltip with "Click to sort".
        for col in (6, 7):
            self.lines_table.horizontalHeaderItem(col).setToolTip(
                "Click to sort. Reflects the active region, if any -- see the Region group.")

        # Connect table changes for auto-calculation
        self.lines_table.itemChanged.connect(self.on_table_item_changed)
        # "Fit" column (14) opens the fit-view dialog on double-click, once
        # Calculate Fluxes has given that row real data to show.
        self.lines_table.itemDoubleClicked.connect(self.on_lines_table_item_double_clicked)

        # Bottom: Physical parameters + buttons
        bottom_layout = QHBoxLayout()
        
        bottom_layout.addWidget(QLabel("Emitting Radius (au):"))
        self.radius_input = QLineEdit()
        self.radius_input.setPlaceholderText("optional")
        self.radius_input.setToolTip(
            "Radius of the emitting region, in au. With a radius, N_u is a column density "
            "(cm^-2). Without it, N_u is a total molecule count (for example, to compare "
            "against PDRTPy)."
        )
        self.radius_input.setMaximumWidth(100)
        bottom_layout.addWidget(self.radius_input)

        bottom_layout.addWidget(QLabel("Distance (pc):"))
        self.distance_input = QLineEdit()
        self.distance_input.setPlaceholderText("required")
        self.distance_input.setToolTip("Distance to the source, in pc.")
        self.distance_input.setMaximumWidth(100)
        bottom_layout.addWidget(self.distance_input)
        
        bottom_layout.addWidget(QLabel("Fit:"))
        self.fit_mode = QComboBox()
        self.fit_mode.addItems(["1T", "2T"])
        self.fit_mode.currentTextChanged.connect(self.on_line_selected)
        bottom_layout.addWidget(self.fit_mode)

        self.fit_opr_check = QCheckBox("Fit OPR")
        self.fit_opr_check.setToolTip(
            "Grid-search over OPR to minimize chi-squared, instead of assuming the "
            "fixed equilibrium value OPR=3. H2 only -- disabled for other molecules, "
            "since OPR is a nuclear-spin-statistics effect specific to H2."
        )
        self.fit_opr_check.stateChanged.connect(self.on_fit_opr_toggled)
        bottom_layout.addWidget(self.fit_opr_check)

        self.opr_advanced_btn = QPushButton("▸ Advanced")
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
        self.run_fit_btn.setToolTip(
            "Fit the rotational diagram to the checked lines. The Fit box sets 1T or 2T.")
        self.run_fit_btn.clicked.connect(self.run_rotational_fit)
        self.run_fit_btn.setEnabled(False)
        bottom_layout.addWidget(self.run_fit_btn)

        self.run_map_btn = QPushButton("Run Map")
        self.run_map_btn.setToolTip(
            "Run the same fit at every pixel of the common grid. It uses detections only. "
            "The Fit box sets 1T or 2T.")
        self.run_map_btn.clicked.connect(self.run_rotational_map)
        self.run_map_btn.setEnabled(False)
        bottom_layout.addWidget(self.run_map_btn)

        main_layout.addLayout(bottom_layout)

        # Each button's own description. on_line_selected adds the "why disabled" reason after it.
        self._base_tooltips = {
            btn: btn.toolTip() for btn in (
                self.run_fit_btn, self.run_map_btn, self.calculate_maps_btn,
                self.convolve_maps_btn, self.select_map_region_btn)
        }

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

        # Fetch and Import buttons, grouped into collapsible sections --
        # same CollapsibleButtonGroup the main window and 2D viewer use.
        self.fetch_btn = QPushButton("Fetch from HITRAN")
        self.fetch_btn.clicked.connect(self.fetch_hitran_lines)

        self.import_csv_btn = QPushButton("Import from CSV")
        self.import_csv_btn.clicked.connect(self.import_csv_file)

        self.import_map_btn = QPushButton("Load Maps")
        self.import_map_btn.setToolTip(
            "Attach one or more already-reduced flux maps (line-integrated, already "
            "convolved/resolution-matched elsewhere), alongside lines computed normally "
            "from loaded cubes. Feeds Convolve Maps/Run Map only -- not Run Fit or "
            "Calculate Fluxes."
        )
        self.import_map_btn.clicked.connect(self.import_map_line)

        self.load_fits_btn = QPushButton("Load Cubes")
        self.load_fits_btn.setToolTip(
            "Select the source FITS cube(s) for Calculate Fluxes. Loaded once and reused "
            "on every click, instead of re-prompting each time."
        )
        self.load_fits_btn.clicked.connect(self.load_batch_fits_files)
        self.load_fits_btn.setEnabled(False)  # Blocked until lines are loaded

        self.clear_fits_btn = QPushButton("Clear Cubes")
        self.clear_fits_btn.setToolTip(
            "Remove the loaded cubes, the region, and everything calculated from the cubes "
            "(fluxes, fits, cached maps). The line list and manually entered fluxes stay, "
            "so you can load new cubes for the same lines."
        )
        self.clear_fits_btn.clicked.connect(self.clear_loaded_cubes)
        self.clear_fits_btn.setEnabled(False)  # Blocked until cubes are loaded

        self.load_contam_btn = QPushButton("Load Contaminating Lines")
        self.load_contam_btn.setToolTip(
            "Optional: a catalog of nearby non-target lines, so Calculate Fluxes can keep "
            "them from leaking into a requested line's amplitude. Same catalog format as "
            "the main Gaussian-fit window's contaminating-lines loader."
        )
        self.load_contam_btn.clicked.connect(self.load_contaminating_lines_for_batch)
        self.load_contam_btn.setEnabled(False)  # Blocked until lines are loaded

        self.calculate_fluxes_btn = QPushButton("Calculate Fluxes")
        self.calculate_fluxes_btn.clicked.connect(self.open_batch_flux_dialog)
        self.calculate_fluxes_btn.setEnabled(False)  # Blocked until lines are loaded

        self.save_fluxes_btn = QPushButton("Save Fluxes")
        self.save_fluxes_btn.setToolTip(
            "Save the checked lines' species, wavelength, flux, and flux error to a CSV, "
            "in the same format Import from CSV reads -- so re-selecting these same lines "
            "later doesn't need Calculate Fluxes run again."
        )
        self.save_fluxes_btn.clicked.connect(self.save_fluxes_to_csv)
        self.save_fluxes_btn.setEnabled(False)  # Blocked until lines are loaded

        self.calculate_maps_btn = QPushButton("Calculate Maps")
        self.calculate_maps_btn.setToolTip(
            "Fit each checked line's own flux map (native resolution per cube) without "
            "combining them into a temperature fit yet. Populates the Map column so you "
            "can inspect each line before running the full map. Caches its result so a "
            "later Run Map reuses it instead of recomputing. Asks for the detection "
            "threshold (σ) when clicked."
        )
        self.calculate_maps_btn.clicked.connect(self.calculate_rotational_maps)
        self.calculate_maps_btn.setEnabled(False)  # Blocked until lines are loaded

        self.convolve_maps_btn = QPushButton("Convolve Maps")
        self.convolve_maps_btn.setToolTip(
            "Convolve/reproject every checked line's already-calculated map (Calculate "
            "Maps) onto a common grid at the worst (coarsest) resolution among them. "
            "Populates the Convolved Map column. Needs 2+ loaded cubes and Calculate "
            "Maps run first."
        )
        self.convolve_maps_btn.clicked.connect(self.convolve_rotational_maps)
        self.convolve_maps_btn.setEnabled(False)  # Blocked until lines are loaded

        self.save_maps_btn = QPushButton("Save Maps")
        self.save_maps_btn.setToolTip(
            "Save every line currently shown in the Map/Convolved Map columns to a "
            "zip file: Flux Maps/{fits,png}/ and Convolved Maps/{fits,png}/, one file "
            "per line in each. Saves whichever of the two has been computed -- run "
            "Calculate Maps (and optionally Convolve Maps) first."
        )
        self.save_maps_btn.clicked.connect(self.save_all_maps)
        self.save_maps_btn.setEnabled(False)  # Blocked until lines are loaded

        self.select_map_region_btn = QPushButton("Draw Region")
        self.select_map_region_btn.setToolTip(
            "Draw a region on a line's flux map. Calculate Fluxes and Run Fit then use only that region."
        )
        self.select_map_region_btn.clicked.connect(self.select_rot_map_region)
        self.select_map_region_btn.setEnabled(False)  # Blocked until a line/cube match exists

        self.import_map_region_btn = QPushButton("Import Region")
        self.import_map_region_btn.setToolTip(
            "Restrict Calculate Fluxes/Run Fit to one sky region, imported from a region "
            "saved elsewhere (Save Region, main window or 2D viewer). Applies per cube using "
            "that cube's own WCS, so the same region works across cubes at different pixel "
            "scales. Calculate Maps/Convolve Maps/Run Map always use the whole cube."
        )
        self.import_map_region_btn.clicked.connect(self.import_rot_map_region)
        self.import_map_region_btn.setEnabled(False)  # Blocked until lines are loaded

        self.save_map_region_btn = QPushButton("Save Region")
        self.save_map_region_btn.setToolTip(
            "Save the region as a WCS text file. Import Region here, in the main window, "
            "or in the 2D viewer can read it back."
        )
        self.save_map_region_btn.clicked.connect(self.save_rot_map_region)
        self.save_map_region_btn.setEnabled(False)  # Blocked until a region is set

        self.clear_map_region_btn = QPushButton("Delete Region")
        self.clear_map_region_btn.setToolTip(
            "Delete the region. Calculate Fluxes and Run Fit then use the whole cube."
        )
        self.clear_map_region_btn.clicked.connect(self.clear_rot_map_region)
        self.clear_map_region_btn.setEnabled(False)  # Blocked until a region is imported

        collapsible_layout = QHBoxLayout()
        collapsible_layout.setSpacing(0)
        collapsible_layout.setContentsMargins(0, 0, 0, 0)

        self.lines_group = CollapsibleButtonGroup("Lines")
        self.lines_group.add_button(self.fetch_btn)
        self.lines_group.add_button(self.import_csv_btn)
        collapsible_layout.addWidget(self.lines_group)

        self.source_group = CollapsibleButtonGroup("Source")
        self.source_group.add_button(self.load_fits_btn)
        self.source_group.add_button(self.clear_fits_btn)
        self.source_group.add_button(self.import_map_btn)
        self.source_group.add_button(self.load_contam_btn)
        collapsible_layout.addWidget(self.source_group)

        self.flux_group = CollapsibleButtonGroup("Flux")
        self.flux_group.add_button(self.calculate_fluxes_btn)
        self.flux_group.add_button(self.save_fluxes_btn)
        collapsible_layout.addWidget(self.flux_group)

        self.maps_group = CollapsibleButtonGroup("Maps")
        self.maps_group.add_button(self.calculate_maps_btn)
        self.maps_group.add_button(self.convolve_maps_btn)
        self.maps_group.add_button(self.save_maps_btn)
        collapsible_layout.addWidget(self.maps_group)

        self.region_group = CollapsibleButtonGroup("Region")
        self.region_group.add_button(self.select_map_region_btn)
        self.region_group.add_button(self.import_map_region_btn)
        self.region_group.add_button(self.save_map_region_btn)
        self.region_group.add_button(self.clear_map_region_btn)
        collapsible_layout.addWidget(self.region_group)

        collapsible_layout.addStretch()
        layout.addLayout(collapsible_layout)

        status_layout = QHBoxLayout()
        self.contam_status_label = QLabel("")  # Only ever shows text once something is actually loaded
        status_layout.addWidget(self.contam_status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        frame.setLayout(layout)
        return frame

    def closeEvent(self, event):
        """Drop this window's collapsible groups from the shared registry on
        close, same as the 2D viewer, so cross-window accordion logic never
        touches a deleted widget.

        Args:
            event (QCloseEvent): Close event Qt sends to the widget.

        Returns:
            None
        """
        for _g in (getattr(self, 'lines_group', None), getattr(self, 'source_group', None),
                   getattr(self, 'flux_group', None), getattr(self, 'maps_group', None),
                   getattr(self, 'region_group', None)):
            try:
                if _g in CollapsibleButtonGroup._all_groups:
                    CollapsibleButtonGroup._all_groups.remove(_g)
            except Exception:
                pass
        # A running worker needs this window and its cubes until it finishes.
        if not self._has_running_worker():
            self._free_loaded_cubes()
            _OPEN_WINDOWS.discard(self)
        event.accept()

    def _has_running_worker(self):
        """Report whether any background thread of this window is still running.

        Args:
            None

        Returns:
            running (bool): True if a QThread attribute of this window is running.
        """
        return any(isinstance(v, QThread) and v.isRunning() for v in vars(self).values())

    def _free_loaded_cubes(self):
        """Close the loaded cubes' FITS files and free their arrays.
        With a worker running, the cubes are only dropped, since freeing them would break it.

        Args:
            None

        Returns:
            None
        """
        cubes = self.loaded_batch_cubes
        self.loaded_batch_cubes = []
        self.loaded_batch_fits_paths = []
        if self._headers_dialog is not None:
            self._headers_dialog.close()
        if not self._has_running_worker():
            for cube in cubes:
                cube.free_memory()

    def show_loaded_headers(self):
        """Open the FITS header viewer for the loaded cubes.

        Args:
            None

        Returns:
            None
        """
        if not self.loaded_batch_cubes:
            QMessageBox.warning(self, "No cubes loaded", "Load a cube first (Source > Load Cubes).")
            return
        if self._headers_dialog is not None:
            self._headers_dialog.close()
        dlg = LoadedHeadersDialog(self.loaded_batch_cubes, self.loaded_batch_fits_paths, parent=self)
        dlg.destroyed.connect(lambda *_: setattr(self, "_headers_dialog", None))
        self._headers_dialog = dlg
        dlg.show()

    def _open_new_window(self):
        """Open another rotational diagram window, with nothing loaded.

        Args:
            None

        Returns:
            None
        """
        try:
            new_win = RotationalDiagramWindow()
            new_win.resize(self.size())
            new_win.move(self.x() + 30, self.y() + 30)
            new_win.show()
        except Exception as e:
            QMessageBox.warning(self, "New Window Error", f"Could not open a new window:\n{e}")

    def on_molecule_changed(self, molecule):
        """Update UI when the molecule changes (Fetch from HITRAN's own dialog).

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
        self.molecule_label.setText(f"Molecule: {molecule}")
        self.update_isotopologue_range()

        # OPR is H2-only physics; block the checkbox for anything else.
        is_h2 = (molecule == 'H2')
        if not is_h2 and self.fit_opr_check.isChecked():
            self.fit_opr_check.setChecked(False)
        self.fit_opr_check.setEnabled(is_h2)

    def _clear_loaded_lines(self):
        """Discard fetched/imported lines and the contaminant catalog loaded
        for the previous molecule. The loaded cubes stay. Fetch and Import ask
        whether to clear them once the new lines arrive.

        Args:
            None

        Returns:
            None
        """
        self.hitran_lines = []
        self.workflow_mode = None
        self.populate_lines_table()
        self.on_line_selected()
        self.calculate_fluxes_btn.setEnabled(False)
        self.load_fits_btn.setEnabled(False)
        self.load_contam_btn.setEnabled(False)
        self.save_fluxes_btn.setEnabled(False)
        self.calculate_maps_btn.setEnabled(False)
        self.convolve_maps_btn.setEnabled(False)
        self.save_maps_btn.setEnabled(False)
        self.select_map_region_btn.setEnabled(False)
        self.import_map_region_btn.setEnabled(False)
        self.save_map_region_btn.setEnabled(False)
        self.clear_map_region_btn.setEnabled(False)

        self.contam_lines = None
        self._rot_map_region_sky = None
        self.contam_status_label.setText("")

    def _count_manual_fluxes(self):
        """Count the fluxes in the table that were not calculated from the cubes.
        These are values typed in, imported from a CSV, or taken from a loaded map.

        Args:
            None

        Returns:
            count (int): Rows with a flux and "Calculated?" not checked.
        """
        count = 0
        for row in range(self.lines_table.rowCount()):
            flux_item = self.lines_table.item(row, 6)
            calc_item = self.lines_table.item(row, 11)
            calculated = calc_item is not None and calc_item.checkState() == Qt.Checked
            if flux_item and flux_item.text().strip() and not calculated:
                count += 1
        return count

    def _drop_loaded_cubes(self):
        """Remove the loaded cubes, the region, and everything calculated from the cubes.
        This covers the calculated fluxes, the Fit links, and the cached native and
        convolved maps. The line list and manually entered fluxes stay.

        Args:
            None

        Returns:
            None
        """
        self._free_loaded_cubes()
        self.load_fits_btn.setToolTip("Select the source FITS cube(s) for Calculate Fluxes.")
        self.clear_fits_btn.setEnabled(False)

        self.clear_rot_map_region()
        self._rot_map_native_cache = None
        for row in list(self._line_map_plot_data):
            del self._line_map_plot_data[row]
            stale_item = self.lines_table.item(row, 15)
            if stale_item:
                self._clear_view_link(stale_item)
        self._clear_all_convolved_maps()

    def clear_loaded_cubes(self):
        """Ask for confirmation, then remove the loaded cubes and all results from them.

        Args:
            None

        Returns:
            None
        """
        n_cubes = len(self.loaded_batch_cubes)
        if not n_cubes:
            return
        n_manual = self._count_manual_fluxes()
        reply = QMessageBox.question(
            self, "Clear Cubes",
            f"Clear the {n_cubes} loaded cube(s)?\n\n"
            "This also clears:\n"
            "  - the region\n"
            "  - the fluxes and fits calculated from the cubes\n"
            "  - the cached maps (Map and Convolved Map)\n\n"
            f"The line list stays. {n_manual} manually entered or imported flux(es) stay.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            print(f"[rot diag] {n_cubes} loaded cube(s) cleared")
            self._drop_loaded_cubes()

    def _ask_clear_cubes_for_new_lines(self):
        """Ask whether to clear the loaded cubes when a new line list replaces the old one.
        With no cubes loaded, this does nothing.

        Args:
            None

        Returns:
            None
        """
        n_cubes = len(self.loaded_batch_cubes)
        if not n_cubes:
            return
        reply = QMessageBox.question(
            self, "Clear Cubes",
            f"The line list is changing. Clear the {n_cubes} loaded cube(s)?\n\n"
            "Yes: clear the cubes, the region, and the cached maps.\n"
            "No: keep the cubes.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            print(f"[rot diag] {n_cubes} loaded cube(s) cleared: line list changed")
            self._drop_loaded_cubes()

    def _reveal_line_workflow_buttons(self):
        """Enable the line-workflow buttons once the table has at least one
        line -- shared by Fetch HITRAN, Import CSV, and Import Map. Draw/
        Delete Region are gated separately (on_line_selected/a region
        actually being active), not by this.

        Args:
            None

        Returns:
            None
        """
        self.calculate_fluxes_btn.setEnabled(True)
        self.load_fits_btn.setEnabled(True)
        self.load_contam_btn.setEnabled(True)
        self.save_fluxes_btn.setEnabled(True)
        self.calculate_maps_btn.setEnabled(True)
        self.convolve_maps_btn.setEnabled(True)
        self.save_maps_btn.setEnabled(True)
        self.import_map_region_btn.setEnabled(True)

    def update_isotopologue_range(self):
        """Reset the current isotopologue to the selected molecule's minimum.

        Args:
            None

        Returns:
            None
        """
        mol_data = MOLECULE_ISOTOPOLOGUES.get(self.current_molecule)
        if mol_data:
            min_iso, _max_iso = mol_data['range']
            self.current_isotopologue = min_iso
    
    def _read_radius_and_distance(self):
        """Read the emitting radius (optional) and distance (required) fields.
        A blank radius gives None. A blank distance is not valid.

        Args:
            None

        Returns:
            values (tuple): (radius_au, distance_pc). radius_au is a float or None,
                distance_pc is a float. None for the whole result if a field was
                invalid -- a warning was already shown in that case.
        """
        values = []
        for field, name, unit, optional in ((self.radius_input, "emitting radius", "au", True),
                                            (self.distance_input, "distance", "pc", False)):
            text = field.text().strip()
            if not text and optional:
                values.append(None)
                continue
            try:
                value = float(text)
            except ValueError:
                hint = " Or leave it blank." if optional else ""
                QMessageBox.warning(self, "Error", f"Enter a number for the {name}, in {unit}.{hint}")
                return None
            if value <= 0:
                QMessageBox.warning(self, "Error", f"The {name} must be greater than 0 {unit}.")
                return None
            values.append(value)

        return tuple(values)

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
        self.opr_advanced_btn.setText(("▾" if expanded else "▸") + " Advanced")

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
            self.on_line_selected()
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

        if col in (6, 7, 8, 10):
            self.on_line_selected()

        # An imported row has no scalar flux (col 6), so the flux-dependent
        # cross-calc below never applies -- update the payload directly.
        if col == 8 and row < len(self.hitran_lines):
            imported = self.hitran_lines[row].get('imported_map')
            if imported is not None:
                try:
                    imported['error_pct'] = float(item.text())
                except ValueError:
                    pass
                return

        if self.updating_error_columns:
            return

        self.updating_error_columns = True

        try:
            # Columns 6, 7 and 8 are flux, flux error and error %
            if col not in (6, 7, 8):
                return
            
            flux_error_item = self.lines_table.item(row, 7)
            error_pct_item = self.lines_table.item(row, 8)
            
            if not flux_error_item or not error_pct_item:
                return

            # Remember which error column was typed last, for the Flux branch below
            if col in (7, 8):
                self.lines_table.blockSignals(True)
                error_pct_item.setData(Qt.UserRole + 1, col == 8)
                self.lines_table.blockSignals(False)

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

            if col == 6:  # Flux changed: keep the error typed last, recompute the other
                ul_item = self.lines_table.item(row, 9)
                is_imported = row < len(self.hitran_lines) and bool(self.hitran_lines[row].get('imported_map'))
                if not (is_imported or (ul_item and ul_item.checkState() == Qt.Checked)):
                    self._recompute_other_error(row, flux)
                return

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
            if col in (7, 8):
                self._update_error_edit_mark(row)

    def _update_error_edit_mark(self, row):
        """Color the Flux Error cell of a calculated row when its value differs
        from the fitted one. The tooltip then shows the fitted value.

        Args:
            row (int): Table row index.

        Returns:
            None
        """
        error_item = self.lines_table.item(row, 7)
        calc_item = self.lines_table.item(row, 11)
        ul_item = self.lines_table.item(row, 9)
        if not (error_item and calc_item and calc_item.checkState() == Qt.Checked):
            return
        if ul_item and ul_item.checkState() == Qt.Checked:
            return
        fitted = error_item.data(Qt.UserRole + 2)
        if not fitted or not error_item.text().strip():
            return
        try:
            edited = float(error_item.text()) != float(fitted)
        except ValueError:
            return
        was_blocked = self.lines_table.blockSignals(True)
        if edited:
            error_item.setBackground(QColor(255, 244, 179))
            error_item.setToolTip(f"Edited by hand. The fit gave {fitted}.")
        else:
            error_item.setBackground(QBrush())
            error_item.setToolTip("")
        self.lines_table.blockSignals(was_blocked)

    def _recompute_other_error(self, row, flux):
        """Recompute the error column the user did not type last, from the one they did.
        Does nothing if the flux is zero or the typed error is blank or invalid.

        Args:
            row (int): Table row index.
            flux (float): The row's flux.

        Returns:
            None
        """
        if flux == 0:
            return
        flux_error_item = self.lines_table.item(row, 7)
        error_pct_item = self.lines_table.item(row, 8)
        if error_pct_item.data(Qt.UserRole + 1):
            source_text, other_item = error_pct_item.text().strip(), flux_error_item
        else:
            source_text, other_item = flux_error_item.text().strip(), error_pct_item
        try:
            source = float(source_text)
        except ValueError:
            return
        if other_item is flux_error_item:
            other_item.setText(f"{flux * (source / 100.0):.4e}")
        else:
            other_item.setText(f"{(source / flux) * 100:.1f}")

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
        # Tooltip, color, and flag changes also fire itemChanged. Left on, the Error %
        # cell's own handler would rebuild Flux Error from the rounded percent.
        was_blocked = self.lines_table.blockSignals(True)
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
            was_upper_limit = False
            for item in (error_item, pct_item):
                stashed = item.data(Qt.UserRole)
                if stashed:
                    item.setText(stashed)
                    item.setData(Qt.UserRole, None)
                    was_upper_limit = True
                item.setToolTip("")

            # Flux may have changed while the row was an upper limit
            if was_upper_limit:
                try:
                    self._recompute_other_error(row, float(self.lines_table.item(row, 6).text()))
                except ValueError:
                    pass

            # An imported row never feeds the point-fit pipeline this flag is
            # about -- a blank Flux Error there is normal, not a problem.
            is_imported = row < len(self.hitran_lines) and bool(self.hitran_lines[row].get('imported_map'))
            if error_item.text().strip() or is_imported:
                error_item.setBackground(QBrush())
            else:
                error_item.setBackground(QColor(255, 214, 214))
                error_item.setToolTip(
                    "No flux error entered -- this detection will be excluded from "
                    "the fit unless you enter one or mark it as an Upper Limit."
                )

            for item in (error_item, pct_item):
                item.setFlags(item.flags() | Qt.ItemIsEditable)
            self._update_error_edit_mark(row)
        finally:
            self.lines_table.blockSignals(was_blocked)
            self.updating_flux_error_flag = False

    def fetch_hitran_lines(self):
        """Fetch molecular lines from HITRAN, prompting for molecule, range,
        isotopologue, and transition filtering in a dedicated dialog.

        Args:
            None

        Returns:
            None
        """
        dlg = FetchHitranDialog(self, initial_molecule=self.current_molecule,
                                 initial_isotopologue=self.current_isotopologue)
        if dlg.exec_() != QDialog.Accepted:
            return
        molecule, wl_min, wl_max, isotopologue, v_upper, v_lower, _include_all = dlg.get_values()

        if molecule != self.current_molecule:
            self.on_molecule_changed(molecule)
        self.molecule_label.setText(f"Molecule: {molecule}")
        self.current_isotopologue = isotopologue

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
        self._reveal_line_workflow_buttons()

        self._ask_clear_cubes_for_new_lines()
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
            if len(species_list) == 1 and species_list[0] in MOLECULE_ISOTOPOLOGUES:
                new_molecule = species_list[0]
                if new_molecule != self.current_molecule:
                    self._apply_molecule_change(new_molecule)
                self.molecule_label.setText(f"Molecule: {new_molecule}")

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
                best_match = match_nearest_hitran_line(hitran_lines, wavelength)
                
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

            self._ask_clear_cubes_for_new_lines()
            self.hitran_lines = rows
            self.populate_lines_table()

            # Pre-fill errors if they were provided in CSV
            self.populate_csv_errors()

            # Set workflow mode to CSV. Fetch/Import stay both enabled.
            self.workflow_mode = 'csv'
            self._reveal_line_workflow_buttons()

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

    def import_map_line(self):
        """Attach one or more already-reduced flux maps to new line-table
        rows, asking for species/wavelength/error percent only when a
        file's header doesn't already carry them. Appends rather than replacing.

        Args:
            None

        Returns:
            None
        """
        fits_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Flux Map FITS", "", "FITS files (*.fits *.fz *.fit)")
        if not fits_paths:
            return

        imported, skipped = [], []
        for fits_path in fits_paths:
            basename = os.path.basename(fits_path)
            found = _peek_header(fits_path)

            species = str(found["MOLECULE"]).strip().upper() if found.get("MOLECULE") else None
            if species is None:
                text, ok = QInputDialog.getText(
                    self, "Import Map - Species", f"Species for {basename} (e.g. H2):")
                species = text.strip().upper() if ok else ""
                if not species:
                    skipped.append((basename, "no species given"))
                    continue

            wavelength = None
            if found.get("LINEWAVE") is not None:
                try:
                    wavelength = float(found["LINEWAVE"])
                except (TypeError, ValueError):
                    wavelength = None
            if wavelength is None:
                value, ok = QInputDialog.getDouble(
                    self, "Import Map - Wavelength", f"Wavelength (um) for {basename}:",
                    5.0, 0.1, 1000.0, 6)
                if not ok:
                    skipped.append((basename, "no wavelength given"))
                    continue
                wavelength = value

            has_full_identity = all(k in found for k in ("A_UL", "E_UPPER", "G_U", "TRANSITN"))
            header_identity = found if has_full_identity else None

            if _has_real_error(fits_path):
                error_pct = 0.0  # unused -- build_imported_map_payload prefers the real ERR extension
            else:
                value, ok = QInputDialog.getDouble(
                    self, "Import Map - Error %",
                    f"{basename} has no per-pixel error data.\n"
                    f"Flat error percent to use instead (flux_error = |flux| x pct/100):",
                    20.0, 0.0, 1000.0, 1)
                if not ok:
                    skipped.append((basename, "no error percent given"))
                    continue
                error_pct = value

            try:
                payload = build_imported_map_payload(fits_path, error_pct)
            except Exception as e:
                skipped.append((basename, str(e)))
                continue

            if header_identity is not None:
                # Already resolved at export time (Gaussian Map) -- re-querying
                # HITRAN would be redundant and could land on a different match.
                best_match = {
                    'a_coeff': header_identity['A_UL'], 'e_upper': header_identity['E_UPPER'],
                    'Gp': header_identity['G_U'], 'transition': header_identity['TRANSITN'],
                    'j_lower': header_identity.get('JLOWER'), 'branch': header_identity.get('BRANCH'),
                }
                fetch_meta = {}
            else:
                try:
                    hitran_lines, fetch_meta = fetch_hitran_data(
                        wavelength - 0.01, wavelength + 0.01, molecule=species, isotopologue=1)
                except Exception as e:
                    hitran_lines, fetch_meta = [], {"error": True, "message": str(e)}

                best_match = match_nearest_hitran_line(hitran_lines, wavelength)

            if best_match:
                line_data = {
                    'wavelength': wavelength, 'hitran_matched': True,
                    'a_coeff': best_match.get('a_coeff', 0.0), 'e_upper': best_match.get('e_upper', 0.0),
                    'Gp': best_match.get('Gp', 1.0), 'j_lower': best_match.get('j_lower'),
                    'branch': best_match.get('branch'),
                    'transition': best_match.get('transition', 'User Data'), 'species': species,
                }
            else:
                line_data = {
                    'wavelength': wavelength, 'hitran_matched': False,
                    'transition': 'NA', 'species': species,
                }
                if fetch_meta.get('error'):
                    print(f"Warning: Could not fetch HITRAN data for {species} at "
                          f"{wavelength:.6f} um ({fetch_meta.get('message', 'unknown error')}) -- "
                          f"A_ul/E_u/g_u left blank.")
                else:
                    print(f"Warning: No HITRAN match for {species} at {wavelength:.6f} um "
                          f"-- A_ul/E_u/g_u left blank; this line is excluded from fitting "
                          f"until they're entered manually.")
            line_data['imported_map'] = payload

            if not self.hitran_lines and species in MOLECULE_ISOTOPOLOGUES:
                if species != self.current_molecule:
                    self._apply_molecule_change(species)
                self.molecule_label.setText(f"Molecule: {species}")

            self._append_line_row(line_data)
            imported.append((basename, wavelength, best_match is not None))

        if imported:
            self._reveal_line_workflow_buttons()

        lines = [f"Imported {len(imported)}/{len(fits_paths)} map(s)."]
        for basename, wavelength, matched in imported:
            note = "matched to HITRAN" if matched else "no HITRAN match -- enter A_ul/E_u/g_u manually"
            lines.append(f"  {basename} @ {wavelength:.6f} um ({note})")
        for basename, reason in skipped:
            lines.append(f"  {basename}: skipped ({reason})")
        box = QMessageBox.information if imported else QMessageBox.warning
        box(self, "Import Map", "\n".join(lines))

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
                    # Column 7: Flux Error (absolute value); editing it fills column 8
                    error_item = self.lines_table.item(row, 7)
                    if error_item:
                        error_item.setText(f"{line_data['flux_error']:.4e}")

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
                self.loaded_batch_fits_paths.append(fits_path)
                self.loaded_batch_cubes.append(cube)
            except Exception as e:
                failed.append((fits_path, str(e)))

        # Tooltip lists everything loaded so far, not just this batch
        names = "\n".join(os.path.basename(p) for p in self.loaded_batch_fits_paths)
        self.clear_fits_btn.setEnabled(bool(self.loaded_batch_cubes))
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
        imported_rows = []
        for row in range(self.lines_table.rowCount()):
            select_item = self.lines_table.item(row, 0)
            if not (select_item and select_item.checkState() == Qt.Checked):
                continue
            # An imported-map row has no point flux to compute -- it would
            # otherwise hit "No FITS Match" below and abort the whole batch.
            if self.hitran_lines[row].get('imported_map'):
                imported_rows.append(row)
                continue
            flux_item = self.lines_table.item(row, 6)
            calc_item = self.lines_table.item(row, 11)
            was_calculated = bool(calc_item and calc_item.checkState() == Qt.Checked)
            if flux_item and flux_item.text().strip() and not was_calculated:
                skipped_rows.append(row)
            else:
                selected_rows.append(row)

        if not selected_rows:
            if skipped_rows or imported_rows:
                notes = []
                if skipped_rows:
                    notes.append(f"{len(skipped_rows)} already have a manually-entered flux value")
                if imported_rows:
                    notes.append(f"{len(imported_rows)} are imported maps, not a point flux")
                QMessageBox.information(
                    self, "Nothing To Calculate",
                    "Nothing to calculate: " + "; ".join(notes) + ".")
            else:
                QMessageBox.warning(self, "No Lines Selected", "Please select lines to calculate fluxes for.")
            return

        if not self.loaded_batch_cubes:
            QMessageBox.warning(
                self, "No FITS Files Loaded",
                "Click 'Load Cubes' first to select the source spectra for these lines."
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

        # Sum each involved cube's spectrum now, inside the current region (if
        # any) -- same shared region as Calculate Maps/Run Map/Run Fit, so it
        # has to be re-applied here rather than reused from a load-time cache.
        summed_spectra = {}
        try:
            for cube_idx in rows_by_cube_idx:
                cube = loaded_cubes[cube_idx]
                if cube.cube.ndim != 3:
                    # Already a flat spectrum, not a spatial cube -- no region to apply.
                    summed_spectra[cube_idx] = cube.cube
                    continue
                mask = self._rot_map_region_mask_for_cube(cube)
                if mask is None:
                    summed_spectra[cube_idx] = np.nansum(cube.cube, axis=(1, 2))
                else:
                    masked = np.where(mask[np.newaxis, :, :], cube.cube, np.nan)
                    summed_spectra[cube_idx] = np.nansum(masked, axis=(1, 2))
        except ValueError as exc:
            QMessageBox.warning(self, "Region", str(exc))
            return

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
            rows_by_cube_idx, loaded_cubes, summed_spectra,
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

            # Flux is read-only now. The fitted error is kept to show later edits.
            was_blocked = self.lines_table.blockSignals(True)
            if flux_item:
                flux_item.setFlags(flux_item.flags() & ~Qt.ItemIsEditable)
                flux_item.setBackground(QBrush())
                flux_item.setToolTip(
                    "Calculated by Calculate Fluxes, so it cannot be edited. Clear Cubes, "
                    "or a region change, clears the calculated fluxes.")
            if error_item:
                error_item.setData(Qt.UserRole + 2, f"{flux_error:.4e}" if flux_error is not None else "")
            self.lines_table.blockSignals(was_blocked)
            self._update_error_edit_mark(row)

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
            error_map=data.get('flux_error_map'), identity_header=data.get('identity_header'),
            src_filename=data.get('source_filename'), scale_store=data,
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
            error_map=data.get('flux_error_map'), identity_header=data.get('identity_header'),
            src_filename=data.get('source_filename'), scale_store=data,
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

    def _clear_view_link(self, item):
        """Undo _set_view_link: reset a cell back to its default "—" placeholder.

        Args:
            item (QTableWidgetItem): Cell to reset.

        Returns:
            None
        """
        item.setText("—")
        item.setData(Qt.ForegroundRole, None)
        item.setData(Qt.FontRole, None)
        item.setToolTip("")

    def _clear_all_convolved_maps(self):
        """Clear every Convolved Map cell and its cache: a fresh Calculate
        Maps run replaces the native maps Convolve Maps was built from, so
        the old convolved result would otherwise look current when it isn't.

        Args:
            None

        Returns:
            None
        """
        self._rot_map_convolved_cache = None
        for row in list(self._line_convolved_map_plot_data):
            del self._line_convolved_map_plot_data[row]
            stale_item = self.lines_table.item(row, 16)
            if stale_item:
                self._clear_view_link(stale_item)
        self.on_line_selected()  # re-sync Convolve Maps/Run Map against the new cache state

    def _populate_map_column(self, result):
        """Turn each mapped line's "Map" cell into a "View" link, using the
        native (pre-convolution) map when available -- Calculate Maps'
        native_result, or a full Run Map result, both carry it; a plain
        single-cube result only has line_flux_maps, since there is only one
        resolution to begin with. Any row left over from an earlier call
        (e.g. a line that was checked for a previous Calculate Maps run but
        isn't part of this one) is reset to "—" first, so a stale map never
        looks current. Lines dropped by dedup are left as "—" too.

        Args:
            result (dict): Output of compute_native_line_maps,
                create_rotational_diagram_map, or create_rotational_diagram_map_multi.

        Returns:
            None
        """
        kept_lines = result.get("kept_lines") or []
        flux_maps = result.get("native_flux_maps") or result.get("line_flux_maps") or []
        if not kept_lines or not flux_maps:
            return
        native_wcs = result.get("native_wcs")  # multi-cube: one per kept_lines entry
        native_headers = result.get("native_headers")  # multi-cube: real cube header per entry
        err_maps = result.get("native_flux_error_maps") or result.get("line_flux_error_maps") or []
        cube_filenames = result.get("cube_filenames")

        kept_rows = {self._row_for_wavelength(l.get("wavelength")) for l in kept_lines}
        for row in list(self._line_map_plot_data):
            if row not in kept_rows:
                del self._line_map_plot_data[row]
                stale_item = self.lines_table.item(row, 15)
                if stale_item:
                    self._clear_view_link(stale_item)

        for i, line in enumerate(kept_lines):
            wl = line.get("wavelength")
            row = self._row_for_wavelength(wl)
            if row is None:
                continue
            if native_wcs is not None:
                header = native_wcs[i].to_header() if native_wcs[i] is not None else None
                identity_header = native_headers[i] if native_headers else None
                source_filename = cube_filenames[i] if cube_filenames else None
            else:
                header = result.get("header")  # single-cube's own header, already real
                identity_header = header
                source_filename = result.get("cube_filename")
            self._line_map_plot_data[row] = {
                "wavelength": wl,
                "flux_map": flux_maps[i],
                "flux_error_map": err_maps[i] if i < len(err_maps) else None,
                "header": header,
                "identity_header": identity_header,
                "source_filename": source_filename,
                "molecule": result.get("molecule", "H2"),
            }
            map_item = self.lines_table.item(row, 15)
            if map_item:
                self._set_view_link(map_item, "Double-click to view this line's flux map")

    def _populate_convolved_map_column(self, result):
        """Turn each mapped line's "Convolved Map" cell into a "View" link,
        using result['kept_lines']/['line_flux_maps'] (post-convolution,
        common-grid arrays). Multi-cube only -- a single-cube result has no
        convolution step, so this is a no-op for one. Any row left over from
        an earlier call (e.g. a line convolved in a previous 7-line run that
        isn't part of a later 5-line one) is reset to "—" first -- its old
        convolved map is on a different grid/threshold now, so leaving the
        "View" link up would look current when it isn't.

        Args:
            result (dict): Output of convolve_reproject_maps or
                create_rotational_diagram_map_multi.

        Returns:
            None
        """
        kept_lines = result.get("kept_lines") or []
        line_flux_maps = result.get("line_flux_maps") or []
        dst_wcs = result.get("dst_wcs")
        if not kept_lines or not line_flux_maps or dst_wcs is None:
            return
        source_header = dst_wcs.to_header()
        err_maps = result.get("line_flux_error_maps") or []

        # No single cube owns the common grid -- use the coarsest-scale
        # cube's real header for INSTRUME/TARGNAME/etc, informational only.
        identity_header, identity_filename = None, None
        native_headers = result.get("native_headers")
        native_scales = result.get("native_pix_arcsec")
        cube_filenames = result.get("cube_filenames")
        if native_headers and native_scales:
            worst_i = int(np.argmax(native_scales))
            identity_header = native_headers[worst_i]
            identity_filename = cube_filenames[worst_i] if cube_filenames else None

        kept_rows = {self._row_for_wavelength(l.get("wavelength")) for l in kept_lines}
        for row in list(self._line_convolved_map_plot_data):
            if row not in kept_rows:
                del self._line_convolved_map_plot_data[row]
                stale_item = self.lines_table.item(row, 16)
                if stale_item:
                    self._clear_view_link(stale_item)

        for i, line in enumerate(kept_lines):
            wl = line.get("wavelength")
            row = self._row_for_wavelength(wl)
            if row is None:
                continue
            self._line_convolved_map_plot_data[row] = {
                "wavelength": wl,
                "flux_map": line_flux_maps[i],
                "flux_error_map": err_maps[i] if i < len(err_maps) else None,
                "header": source_header,
                "identity_header": identity_header,
                "source_filename": identity_filename,
                "molecule": result.get("molecule", "H2"),
            }
            convolved_item = self.lines_table.item(row, 16)
            if convolved_item:
                self._set_view_link(convolved_item, "Double-click to view this line's convolved flux map")

    def save_all_maps(self):
        """Zip every line shown in the Map/Convolved Map columns into
        Flux Maps/{fits,png}/ and Convolved Maps/{fits,png}/. Reads
        _line_map_plot_data/_line_convolved_map_plot_data directly, so an empty one is just skipped.

        Args:
            None

        Returns:
            None
        """
        if not self._line_map_plot_data and not self._line_convolved_map_plot_data:
            QMessageBox.warning(
                self, "Save Maps",
                "No maps to save yet -- run Calculate Maps (and optionally Convolve Maps) first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Maps", "rot_diag_maps.zip", "Zip files (*.zip)")
        if not path:
            return
        if not path.lower().endswith(".zip"):
            path += ".zip"
        try:
            n_flux, n_conv = self._write_maps_zip(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Maps failed", str(exc))
            return
        parts = []
        if n_flux:
            parts.append(f"{n_flux} flux map(s)")
        if n_conv:
            parts.append(f"{n_conv} convolved map(s)")
        QMessageBox.information(self, "Saved", f"Wrote {' and '.join(parts)} to:\n{path}")

    def _write_maps_zip(self, path):
        """Write both map dicts to one zip: <folder>/fits/<name>.fits and
        <folder>/png/<name>.png, one line per file.

        Args:
            path (str): Destination .zip path.

        Returns:
            counts (tuple): (n_flux, n_conv) lines actually written.
        """
        groups = [
            (self._line_map_plot_data, "Flux Maps", "flux"),
            (self._line_convolved_map_plot_data, "Convolved Maps", "convolved"),
        ]
        counts = []
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for data_dict, folder, tag in groups:
                entries = sorted(data_dict.values(), key=lambda d: d["wavelength"])
                for entry in entries:
                    stem = f"{entry.get('molecule', 'H2')}_{tag}_{entry['wavelength']:.4f}um"
                    zf.writestr(f"{folder}/fits/{stem}.fits", self._render_map_fits_bytes(entry))
                    zf.writestr(f"{folder}/png/{stem}.png", self._render_map_png_bytes(entry))
                counts.append(len(entries))
        return tuple(counts)

    def _render_map_fits_bytes(self, entry):
        """Build one line's flux-map FITS file in memory, same header
        convention as LineFluxMapDialog._export_fits.

        Args:
            entry (dict): One _line_map_plot_data/_line_convolved_map_plot_data value.

        Returns:
            data (bytes): The FITS file's raw bytes.
        """
        flux_map = entry["flux_map"]
        err_map = entry.get("flux_error_map")
        header, extensions = build_line_map_header(
            entry.get("header"), flux_map.shape, wavelength=entry["wavelength"],
            molecule=entry.get("molecule", "H2"), flux_map=flux_map, error_map=err_map,
            src_filename=entry.get("source_filename") or "UNKNOWN", fit_method="Multi-Gaussian fit",
            identity_header=entry.get("identity_header"), extra={"BUNIT": "erg/s/cm2"},
        )
        buf = io.BytesIO()
        write_fits_product(buf, np.asarray(flux_map, dtype=np.float32), header, extensions=extensions)
        return buf.getvalue()

    def _render_map_png_bytes(self, entry):
        """Render one line's flux map as a PNG with the colormap and scale last chosen
        in LineFluxMapDialog (inferno, Linear and Auto range if it was never opened).

        Args:
            entry (dict): One _line_map_plot_data/_line_convolved_map_plot_data value.

        Returns:
            data (bytes): The PNG file's raw bytes.
        """
        fig = Figure(figsize=(7, 5.5))
        ax = fig.add_subplot(111)
        norm, shown = build_norm(np.asarray(entry["flux_map"], dtype=float),
                                 entry.get("scale_settings") or ScaleSettings())
        draw_map(fig, ax, shown, norm, entry.get("cmap") or "inferno",
                 f"Flux @ {entry['wavelength']:.6f} um", "Flux (erg/s/cm2)")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        return buf.getvalue()

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
        """Populate the unified table with all columns, rebuilding every row.
        To add ONE row without touching the rest (Import Map), use
        _append_line_row instead.

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
            self._populate_lines_table_row(row, line)

        # Rows just defaulted to checked -- reflect that in the Run Fit button.
        self.on_line_selected()

    def _append_line_row(self, line):
        """Append one new line to the table without touching any other
        row's plot-data dicts or cached results -- see populate_lines_table.

        Args:
            line (dict): Line dict to append (e.g. from Import Map).

        Returns:
            row (int): The new row's index.
        """
        row = len(self.hitran_lines)
        self.hitran_lines.append(line)
        self.lines_table.setRowCount(row + 1)
        self._populate_lines_table_row(row, line)
        self.on_line_selected()
        return row

    def _populate_lines_table_row(self, row, line):
        """Build one line-table row's cells from its line dict. Shared by
        populate_lines_table's full rebuild and _append_line_row's
        single-row append.

        Args:
            row (int): Table row index, already sized by the caller.
            line (dict): Line dict to render.

        Returns:
            None
        """
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

        imported = line.get('imported_map')
        global_fit = imported.get('global_fit') if imported else None

        # Flux (editable) -- an imported row with a Gaussian Map global-fit
        # area (see build_imported_map_payload) shows that real point flux.
        if imported:
            flux_val = global_fit.get('area') if global_fit else None
        else:
            flux_val = line.get('flux', '')
        flux_item = QTableWidgetItem(f"{flux_val:.4e}" if flux_val else "")
        if not flux_val and not imported:
            flux_item.setBackground(unmatched_bg)
            flux_item.setToolTip("Flux is required for the fit -- enter it manually or use Calculate Fluxes.")
        self.lines_table.setItem(row, 6, flux_item)

        # Flux Error (editable) -- same source as Flux above.
        ferr_val = global_fit.get('area_error') if global_fit else None
        self.lines_table.setItem(row, 7, QTableWidgetItem(f"{ferr_val:.4e}" if ferr_val else ""))

        # Error % (editable) -- blank when a real ERR extension is already
        # in use, else pre-filled with the percent chosen on import.
        if imported and imported['error_pct'] is not None:
            err_pct_item = QTableWidgetItem(f"{imported['error_pct']:.1f}")
        elif imported:
            err_pct_item = QTableWidgetItem("")
            err_pct_item.setToolTip(
                "Using this file's own per-pixel error (ERR extension) -- "
                "type a value here to use a flat percent instead.")
        else:
            err_pct_item = QTableWidgetItem("")
        self.lines_table.setItem(row, 8, err_pct_item)

        # Upper Limit? -- a point-flux concept, meaningless for a per-pixel
        # map, so an imported row's checkbox is disabled rather than live.
        ul_item = self._make_checkable_item(
            enabled=not imported,
            tooltip=("Not applicable to an imported map." if imported else
                     "Auto-checked when a line's fit falls below the 3σ noise threshold "
                     "(via Calculate Fluxes). You can check/uncheck this yourself either way."))
        self.lines_table.setItem(row, 9, ul_item)

        # UL Flux (read-only) -- filled in by Calculate Fluxes
        ul_flux_item = QTableWidgetItem("")
        ul_flux_item.setFlags(ul_flux_item.flags() & ~Qt.ItemIsEditable)
        self.lines_table.setItem(row, 10, ul_flux_item)

        # Calculated? (read-only)
        calc_item = self._make_checkable_item(enabled=False, tooltip=(
            "Checked when this line's flux was computed by Calculate Fluxes, rather "
            "than typed in by hand or imported from CSV."
        ))
        self.lines_table.setItem(row, 11, calc_item)

        # Source (read-only) -- same "{instrument} [{filename}]" convention
        # Calculate Fluxes already uses, sourced from the import instead.
        source_item = QTableWidgetItem(
            f"{imported['instrument'] or 'UNKNOWN'} [{imported['filename']}]"
            if (imported and global_fit) else "")
        source_item.setFlags(source_item.flags() & ~Qt.ItemIsEditable)
        self.lines_table.setItem(row, 12, source_item)

        # Fit R^2 (read-only; filled in by Calculate Fluxes, or an imported
        # row's own Gaussian Map global fit, blank for manually-entered flux)
        r2_val = global_fit.get('r_squared') if global_fit else None
        r2_item = QTableWidgetItem(f"{r2_val:.3f}" if r2_val is not None else "")
        r2_item.setFlags(r2_item.flags() & ~Qt.ItemIsEditable)
        self.lines_table.setItem(row, 13, r2_item)

        # Fit (double-click to view; becomes real text once Calculate Fluxes,
        # or an imported row's own Gaussian Map global fit, has real data to show)
        fit_item = QTableWidgetItem("—")
        fit_item.setFlags(fit_item.flags() & ~Qt.ItemIsEditable)
        fit_item.setTextAlignment(Qt.AlignCenter)
        self.lines_table.setItem(row, 14, fit_item)
        has_full_curve = global_fit and global_fit.get('x_data') is not None and all(
            global_fit.get(k) is not None for k in ("amplitude", "center", "sigma", "baseline", "y_data"))
        if has_full_curve:
            g_amp, g_center, g_sigma = (
                global_fit["amplitude"], global_fit["center"], global_fit["sigma"])
            g_x, g_baseline = np.asarray(global_fit["x_data"]), np.asarray(global_fit["baseline"])
            g_y_fit = gaussian(g_x, g_amp, g_center, g_sigma) + g_baseline
            self._line_fit_plot_data[row] = {
                "wavelength": line["wavelength"], "x_data": g_x,
                "y_data": global_fit["y_data"], "baseline": g_baseline, "y_fit": g_y_fit,
                "y_fit_own": None, "center": g_center, "sigma": g_sigma,
                "fwhm": 2.355 * g_sigma if g_sigma is not None else None,
                "r_squared": global_fit.get("r_squared"),
                "amplitude": g_amp, "flux_cgs": global_fit.get("area"),
                "flux_error_cgs": global_fit.get("area_error"), "is_upper_limit": False,
                "full_wavelength": None, "full_spectrum": None, "bunit": None,
            }
            self._set_view_link(fit_item, "Double-click to view this imported line's fit")

        # Map (double-click to view). An imported row's map is already here --
        # populate it now, tinted to show it isn't a cube-fit result.
        map_item = QTableWidgetItem("—")
        map_item.setFlags(map_item.flags() & ~Qt.ItemIsEditable)
        map_item.setTextAlignment(Qt.AlignCenter)
        self.lines_table.setItem(row, 15, map_item)
        if imported:
            self._line_map_plot_data[row] = {
                "wavelength": line["wavelength"], "flux_map": imported["flux_map"],
                "flux_error_map": imported.get("flux_error_map"), "header": imported["header"],
                "identity_header": imported["header"], "source_filename": imported["filename"],
                "molecule": line.get("species", self.current_molecule),
            }
            self._set_view_link(map_item, f"Imported from {imported['filename']}")
            map_item.setBackground(QColor(214, 230, 255))

        # Convolved Map (double-click to view; multi-cube only -- see
        # _populate_convolved_map_column)
        convolved_map_item = QTableWidgetItem("—")
        convolved_map_item.setFlags(convolved_map_item.flags() & ~Qt.ItemIsEditable)
        convolved_map_item.setTextAlignment(Qt.AlignCenter)
        self.lines_table.setItem(row, 16, convolved_map_item)

        self.sync_flux_error_flag(row)

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

    def _checked_lines_ready_for_fit(self):
        """Check every checked line for the flux data Run Fit needs, mirroring
        run_rotational_fit's own per-row column choice so nothing it would
        silently drop looks ready here.

        Args:
            None

        Returns:
            result (tuple): (checked_count, all_ready) -- all_ready requires
                a real flux (UL Flux for a checked Upper Limit, else Flux)
                on every checked row, plus a Flux Error or Error % on every
                checked row that isn't an upper limit.
        """
        checked_count = 0
        all_ready = True
        for row in range(self.lines_table.rowCount()):
            select_item = self.lines_table.item(row, 0)
            if not (select_item and select_item.checkState() == Qt.Checked):
                continue
            checked_count += 1

            ul_item = self.lines_table.item(row, 9)
            is_upper_limit = bool(ul_item and ul_item.checkState() == Qt.Checked)

            flux_col = 6
            if is_upper_limit:
                ul_flux_item = self.lines_table.item(row, 10)
                if ul_flux_item and ul_flux_item.text().strip():
                    flux_col = 10
            flux_item = self.lines_table.item(row, flux_col)
            if not (flux_item and flux_item.text().strip()):
                all_ready = False
                continue

            if is_upper_limit:
                continue  # No error needed -- the fit uses censored regression for these.

            error_item = self.lines_table.item(row, 7)
            pct_item = self.lines_table.item(row, 8)
            has_error = bool(error_item and error_item.text().strip()) or \
                bool(pct_item and pct_item.text().strip())
            if not has_error:
                all_ready = False
        return checked_count, all_ready

    def on_line_selected(self):
        """Enable Run Fit (needs flux data too)/Calculate Maps on checked-line
        count; Convolve Maps/Run Map also require the previous stage's own
        cache, enforcing Calculate -> Convolve -> Run as the only path.

        Args:
            None

        Returns:
            None
        """
        checked_count, fit_data_ready = self._checked_lines_ready_for_fit()
        native_ready = getattr(self, "_rot_map_native_cache", None) is not None
        convolved_ready = getattr(self, "_rot_map_convolved_cache", None) is not None

        self.run_fit_btn.setEnabled(checked_count >= 2 and fit_data_ready)
        if checked_count < 2:
            self._set_gated_tooltip(self.run_fit_btn, "Select at least 2 lines -- a line fit needs 2+ points.")
        elif not fit_data_ready:
            self._set_gated_tooltip(
                self.run_fit_btn,
                "Every checked line needs a Flux (and Flux Error/Error %, unless Upper Limit).")
        else:
            self._set_gated_tooltip(self.run_fit_btn, "")
        # 1T needs 3 detections/pixel to constrain its slope, 2T needs 5 --
        # fewer total checked lines than that can never reach the threshold.
        fit2t = self.fit_mode.currentText() == "2T"
        min_lines_needed = 5 if fit2t else 3
        self.run_map_btn.setEnabled(checked_count >= min_lines_needed and convolved_ready)
        if checked_count < min_lines_needed:
            self._set_gated_tooltip(
                self.run_map_btn,
                f"Select at least {min_lines_needed} lines for a per-pixel {'2T' if fit2t else '1T'} map.")
        elif not convolved_ready:
            self._set_gated_tooltip(self.run_map_btn, "Run Calculate Maps, then Convolve Maps, first.")
        else:
            self._set_gated_tooltip(self.run_map_btn, "")
        # Calculate Maps doesn't need a T-fit -- "need a loaded cube" is
        # checked at click time instead, same as Run Map's own "No Cube" check.
        self.calculate_maps_btn.setEnabled(checked_count >= 1)
        self._set_gated_tooltip(
            self.calculate_maps_btn, "" if checked_count >= 1 else "Select at least 1 line.")
        # Needs 2+ lines (cube-fit or imported) -- with only one, there is
        # only one resolution and nothing to convolve.
        self.convolve_maps_btn.setEnabled(checked_count >= 2 and native_ready)
        if checked_count < 2:
            self._set_gated_tooltip(
                self.convolve_maps_btn, "Select at least 2 lines -- nothing to convolve with only one.")
        elif not native_ready:
            self._set_gated_tooltip(self.convolve_maps_btn, "Run Calculate Maps first.")
        else:
            self._set_gated_tooltip(self.convolve_maps_btn, "")
        # Draws on the cached Calculate Maps flux map of a cube-fit line.
        can_select_region = bool(self._rot_map_region_entries()[0])
        self.select_map_region_btn.setEnabled(can_select_region)
        self._set_gated_tooltip(
            self.select_map_region_btn,
            "" if can_select_region else "Run Calculate Maps on at least one cube-fit line first.")

    def _set_gated_tooltip(self, button, reason):
        """Show a button's own description, then the reason it is disabled.

        Args:
            button (QPushButton): Button to update.
            reason (str): Why the button is disabled, or an empty string if it is enabled.

        Returns:
            None
        """
        base = self._base_tooltips.get(button, "")
        button.setToolTip(f"{base}\n\n{reason}" if base and reason else base or reason)

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
        
        radius_and_distance = self._read_radius_and_distance()
        if radius_and_distance is None:
            return
        radius_au, distance_pc = radius_and_distance
        self._fit_inputs = radius_and_distance  # shown in the results window

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
            molecule=self.current_molecule, isotopologue=self.current_isotopologue,
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

    def _rot_map_worker_busy(self):
        """True if a previous Run Map/Calculate Maps/Convolve Maps worker is
        still running or still shutting down after being cancelled. Qt hides
        a cancelled QProgressDialog immediately (regardless of setAutoClose/
        setAutoReset), so without this check a second click could start a
        new worker while the first one is still alive in the background,
        both racing on the same self.rot_map_worker/self.rot_map_progress --
        the likely cause of a stray/duplicate "cancelled" message. Same
        pattern as gui_gaussian.py's Gaussian Map guard.

        Args:
            None

        Returns:
            bool: True if a worker from a previous click is still running.
        """
        worker = getattr(self, "rot_map_worker", None)
        return worker is not None and worker.isRunning()

    def run_rotational_map(self):
        """Run the same fit at every pixel of the loaded cube(s)/imported
        map(s), 1T, detections only. Always convolves/reprojects onto one
        common grid first, even a single cube's own lines (see Calculate Maps).

        Args:
            None

        Returns:
            None
        """
        if self._rot_map_worker_busy():
            QMessageBox.warning(
                self, "Fit Still Stopping",
                "The previous map operation is still shutting down after being "
                "cancelled. Wait a few seconds and try again.")
            return

        cubes = list(self.loaded_batch_cubes)

        line_table, discarded_ul, err = self._rot_map_gather_lines()
        if err:
            QMessageBox.warning(self, "Lines", err)
            return

        imported_rows = [l for l in line_table if l.get('imported_map')]
        cube_rows = [l for l in line_table if not l.get('imported_map')]

        if not cubes and not imported_rows:
            QMessageBox.warning(
                self, "No Cube",
                "Load one or more FITS cubes first (Load Cubes), or add "
                "a line via Load Maps.")
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

        cubes_and_lines = []
        if cube_rows:
            cubes_and_lines, match_err = self._rot_map_match_lines_to_cubes(cubes, cube_rows)
            if match_err:
                QMessageBox.warning(self, "Cubes", match_err)
                return

        # Run Map is only enabled once a convolved cache exists (on_line_selected),
        # so its threshold is already fixed -- asking again here would let a
        # different answer silently invalidate it and trigger a full native
        # refit, with no indication that's what just happened.
        convolved_cache = getattr(self, "_rot_map_convolved_cache", None)
        cached_n_sigma = convolved_cache["key"][2] if convolved_cache is not None else None
        fit_kwargs = self._rot_map_gather_settings(n_sigma=cached_n_sigma)
        if fit_kwargs is None:
            return

        # Fast path: skip straight to the cheap final combination if Calculate
        # Maps/Convolve Maps already ran for this same geometry/threshold.
        geometry_key = self._rot_map_geometry_key(
            cubes_and_lines, fit_kwargs["n_sigma"], imported_lines=imported_rows)
        if convolved_cache is not None and convolved_cache["key"] == geometry_key:
            self._run_map_fast_combine(cubes, cubes_and_lines, convolved_cache["result"],
                                       fit_kwargs, imported_lines=imported_rows)
            return

        # Reuse a cached result if the same cubes/lines/settings produced one.
        cache = getattr(self, "_rot_map_cache", None)
        cache_key = self._rot_map_cache_key(cubes, line_table, True, fit_kwargs)
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

        # Confirm before the run: cheap, fit-free, and always the real plan
        # even for one line -- the real region-fit cross-check runs in the worker.
        try:
            peek = self._rot_map_plan_peek(cubes_and_lines, imported_rows)
            plan = convolution_plan(peek)
        except Exception as exc:
            QMessageBox.critical(self, "Plan Failed", str(exc))
            return
        if not self._show_convolution_plan_confirm(plan):
            return

        self._rot_map_multi_mode = True
        self._rot_map_cube_arg = None
        self._rot_map_line_table = None
        self._rot_map_fit_kwargs = fit_kwargs

        self.rot_map_progress = QProgressDialog(
            "Fitting rotational diagram per pixel...", "Cancel", 0, 0, self)
        self.rot_map_progress.setWindowTitle("Rotational Diagram Map")
        self.rot_map_progress.setWindowModality(Qt.WindowModal)
        self.rot_map_progress.setAutoClose(False)
        self.rot_map_progress.setAutoReset(False)
        self.rot_map_progress.canceled.connect(self._cancel_rot_map)
        self.rot_map_progress.show()
        self._rot_map_progress_label = "Fitting rotational diagram"

        self.run_fit_btn.setEnabled(False)
        self.run_map_btn.setEnabled(False)

        self.rot_map_worker = RotDiagMapWorker(
            cubes_and_lines=cubes_and_lines, imported_lines=imported_rows, **fit_kwargs)
        self.rot_map_worker.progress.connect(self._on_rot_map_progress)
        self.rot_map_worker.finished.connect(self._on_rot_map_complete)
        self.rot_map_worker.error.connect(self._on_rot_map_error)
        self.rot_map_worker.start()

    def _rot_map_plan_peek(self, cubes_and_lines, imported_lines):
        """Cheap, fit-free native_result-shaped peek for a plan preview -- no
        flux arrays, just the header/WCS fields convolution_plan needs.

        Args:
            cubes_and_lines (list): (CubeData, line_dict) pairs, cube-backed only
            imported_lines (list): Line dicts with an 'imported_map' payload

        Returns:
            peek (dict): Just the keys convolution_plan reads
        """
        lines, wcses, shapes, scales, files, instruments = [], [], [], [], [], []
        for cube, line in cubes_and_lines:
            lines.append(line)
            wcses.append(make_2d_wcs(cube.header))
            shapes.append((cube.cube.shape[1], cube.cube.shape[2]))
            scales.append(get_pixel_size_arcsec(cube.header))
            files.append(getattr(cube, "filename", None))
            instruments.append(cube.instrument)
        for line in imported_lines:
            imp = line["imported_map"]
            lines.append(line)
            wcses.append(imp["wcs"])
            shapes.append(imp["shape"])
            scales.append(imp["pix_arcsec"])
            files.append(imp["filename"])
            instruments.append(imp["instrument"])
        return {
            "kept_lines": lines, "native_wcs": wcses, "native_shapes": shapes,
            "native_pix_arcsec": scales, "cube_filenames": files,
            "native_instrument": instruments, "dropped_lines": [],
        }

    def _run_map_fast_combine(self, cubes, cubes_and_lines, convolved_result, fit_kwargs,
                              imported_lines=None):
        """Run Map's fast path: recombine an already-convolved result into a
        temperature map, skipping native extraction and convolution entirely
        -- just the cheap final per-pixel combination.

        Args:
            cubes (list): Currently loaded CubeData objects, for the final cache key
            cubes_and_lines (list): (CubeData, line_dict) pairs matching convolved_result
            convolved_result (dict): Output of convolve_reproject_maps
            fit_kwargs (dict): molecule/isotopologue/opr/fit_opr/opr_grid/dist_pc/r_au/min_lines
            imported_lines (list): Line dicts with an 'imported_map' payload,
                or None/empty.

        Returns:
            None
        """
        line_table = [line for _, line in cubes_and_lines] + list(imported_lines or [])

        self.rot_map_progress = QProgressDialog(
            "Fitting rotational diagram per pixel (reusing calculated maps)...",
            "Cancel", 0, 0, self)
        self.rot_map_progress.setWindowTitle("Rotational Diagram Map")
        self.rot_map_progress.setWindowModality(Qt.WindowModal)
        self.rot_map_progress.setAutoClose(False)
        self.rot_map_progress.setAutoReset(False)
        self.rot_map_progress.canceled.connect(self._cancel_rot_map)
        self.rot_map_progress.show()
        self._rot_map_progress_label = "Fitting rotational diagram"

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
        self.rot_map_worker = CombineMapsWorker(convolved_result, cubes_and_lines,
                                                region_masks=None,  # Run Map always uses the whole cube.
                                                imported_lines=imported_lines, **fit_kwargs)
        self.rot_map_worker.progress.connect(self._on_rot_map_progress)
        self.rot_map_worker.finished.connect(self._on_rot_map_complete)
        self.rot_map_worker.error.connect(self._on_rot_map_error)
        self.rot_map_worker.start()

    def _reset_region_dependent_results(self):
        """Drop Calculate Fluxes' computed flux columns (and the Run Fit view
        they feed) after a region change -- manually entered fluxes are kept.
        Calculate Maps/Convolve Maps/Run Map never depend on a region.

        Args:
            None

        Returns:
            None
        """
        self.lines_table.blockSignals(True)
        try:
            for row in range(self.lines_table.rowCount()):
                calc_item = self.lines_table.item(row, 11)
                if calc_item is None or calc_item.checkState() != Qt.Checked:
                    continue  # manually-entered flux, not region-dependent
                flux_item = self.lines_table.item(row, 6)
                if flux_item:
                    flux_item.setText("")
                    flux_item.setFlags(flux_item.flags() | Qt.ItemIsEditable)
                    flux_item.setBackground(QColor(255, 214, 214))
                    flux_item.setToolTip("Flux is required for the fit -- enter it manually or use Calculate Fluxes.")
                error_item = self.lines_table.item(row, 7)
                if error_item:
                    error_item.setData(Qt.UserRole + 2, None)
                for col in (7, 8, 10, 12):
                    cell = self.lines_table.item(row, col)
                    if cell:
                        cell.setText("")
                        if col in (7, 8):
                            cell.setFlags(cell.flags() | Qt.ItemIsEditable)
                        cell.setBackground(QBrush())
                        cell.setToolTip("")
                r2_item = self.lines_table.item(row, 13)
                if r2_item:
                    r2_item.setText("")
                    r2_item.setBackground(QBrush())
                ul_item = self.lines_table.item(row, 9)
                if ul_item:
                    ul_item.setCheckState(Qt.Unchecked)
                calc_item.setCheckState(Qt.Unchecked)
                fit_item = self.lines_table.item(row, 14)
                if fit_item:
                    self._clear_view_link(fit_item)
                self._line_fit_plot_data.pop(row, None)
        finally:
            self.lines_table.blockSignals(False)

    def select_rot_map_region(self):
        """Draw a region on a line's Calculate Maps flux map, instead of
        importing one saved elsewhere -- a second way to set the same active
        region (see import_rot_map_region). Needs Calculate Maps to have run.

        Args:
            None

        Returns:
            None
        """
        entries, n_sigma = self._rot_map_region_entries()
        if not entries:
            QMessageBox.warning(
                self, "Draw Region",
                "Run Calculate Maps on at least one cube-fit line first -- the region "
                "is drawn on its flux map. Imported maps carry no fit, so they can't be used.")
            return

        dlg = SelectCubeRegionDialog(entries, n_sigma, parent=self)
        if dlg.exec_() != QDialog.Accepted or dlg.result_region is None:
            return

        self._rot_map_region_sky = dlg.result_region
        self.save_map_region_btn.setEnabled(True)
        self.clear_map_region_btn.setEnabled(True)
        self._reset_region_dependent_results()
        shape_name = region_io.SHAPE_NAMES.get(dlg.result_region["shape"], dlg.result_region["shape"])
        QMessageBox.information(
            self, "Region Selected",
            f"Region selected ({shape_name}). Calculate Fluxes and Run Fit will now only use "
            "pixels inside it, on every loaded cube. Any previously computed fluxes were "
            "cleared since they no longer reflect this region. Calculate Maps/Convolve Maps/"
            "Run Map are unaffected -- they always use the whole cube.")

    def _rot_map_region_entries(self):
        """Per-line data Draw Region draws on, taken from the cached
        Calculate Maps result. Imported-map lines are skipped: they have
        no fit, so no upper-limit data.

        Args:
            None

        Returns:
            result (tuple): (entries, n_sigma) -- entries is a list of dicts
                (see SelectCubeRegionDialog), sorted by wavelength; empty if
                Calculate Maps has not run or has no cube-fit line. n_sigma
                is the detection threshold that run used, or None.
        """
        native_cache = getattr(self, "_rot_map_native_cache", None)
        if native_cache is None:
            return [], None
        result = native_cache["result"]
        entries = []
        for i, line in enumerate(result["kept_lines"]):
            if result["native_status_maps"][i] is None:
                continue
            fname = result["cube_filenames"][i]
            wavelength = line["wavelength"]
            entries.append({
                "label": f"{wavelength:.4f} um",
                "file": os.path.basename(fname) if fname else "cube",
                "wavelength": wavelength,
                "header": result["native_headers"][i],
                "fit_flux": result["native_fit_flux_maps"][i],
                "status": result["native_status_maps"][i],
                "ceiling": result["native_ul_ceiling_maps"][i],
            })
        entries.sort(key=lambda e: e["wavelength"])
        return entries, native_cache["key"][2]

    def import_rot_map_region(self):
        """Import a region saved elsewhere (Save Region, main window or 2D
        viewer) to restrict Calculate Fluxes/Run Fit to one sky area --
        converts to each cube's own pixel mask at fit time via its own WCS.

        Args:
            None

        Returns:
            None
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Region as WCS Text", "", "Text files (*.txt *.csv)")
        if not path:
            return
        try:
            saved = region_io.read_region_wcs_txt(path)
        except Exception as e:
            QMessageBox.warning(self, "Import Failed", str(e))
            return

        self._rot_map_region_sky = saved
        self.save_map_region_btn.setEnabled(True)
        self.clear_map_region_btn.setEnabled(True)
        self._reset_region_dependent_results()
        QMessageBox.information(
            self, "Region Imported",
            f"Region imported ({region_io.SHAPE_NAMES.get(saved['shape'], saved['shape'])}). "
            "Calculate Fluxes and Run Fit will now only use pixels inside it, on every loaded "
            "cube. Any previously computed fluxes were cleared since they no longer reflect "
            "this region. Calculate Maps/Convolve Maps/Run Map are unaffected -- they always "
            "use the whole cube. Import a different region to replace it. Or click Delete Region "
            "to fit the whole cube again.")

    def clear_rot_map_region(self):
        """Drop the active region -- Calculate Fluxes/Run Fit go back to
        fitting the whole cube.

        Args:
            None

        Returns:
            None
        """
        self._rot_map_region_sky = None
        self.save_map_region_btn.setEnabled(False)
        self.clear_map_region_btn.setEnabled(False)
        self._reset_region_dependent_results()

    def save_rot_map_region(self):
        """Save the active region as a WCS text file. Import Region here, in
        the main window, or in the 2D viewer reads the same file back.

        Args:
            None

        Returns:
            None
        """
        region = self._rot_map_region_sky
        if region is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Region as WCS Text", "", "Text files (*.txt)")
        if not path:
            return
        try:
            region_io.write_region_wcs_txt(
                path, region["shape"], "Rotational Diagram window",
                region.get("pixel_points"), region["sky_points"],
                sky_center=region.get("sky_center"), sky_axis_x=region.get("sky_axis_x"),
                sky_axis_y=region.get("sky_axis_y"))
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", str(e))
            return
        QMessageBox.information(self, "Region Saved", f"Saved the region to:\n{path}")

    def _rot_map_region_mask_for_cube(self, cube):
        """Convert the active sky region (if any -- from Select or Import) to
        this cube's own pixel mask, via the same sky_to_region_shape
        conversion the main window's Import Region uses.

        Args:
            cube (CubeData): Cube to build the mask for.

        Returns:
            mask (numpy.ndarray or None): Boolean mask, shape (ny, nx), or
                None if no region is active.

        Raises:
            ValueError: the cube has no usable celestial WCS, or the region
                shares no sky overlap with this cube at all -- surfaced to
                the caller so it can be shown as a clear warning rather than
                silently fitting the whole cube.
        """
        saved = self._rot_map_region_sky
        if saved is None:
            return None
        wcs2d = make_2d_wcs(cube.header)
        if wcs2d is None or not getattr(wcs2d, "has_celestial", False):
            raise ValueError(
                f"{getattr(cube, 'filename', 'This cube')} has no usable celestial WCS, "
                "so the imported region can't be placed on it.")
        ny, nx = cube.cube.shape[1], cube.cube.shape[2]
        placed = region_io.sky_to_region_shape(
            saved["shape"], saved["sky_points"], saved["sky_center"],
            saved["sky_axis_x"], saved["sky_axis_y"], wcs2d, target_shape=(ny, nx))
        return mask_from_shape_coords(placed["shape"], placed.get("coords"), placed.get("verts"), (ny, nx))

    def calculate_rotational_maps(self):
        """Compute each checked line's own flux map at native resolution
        (always per-line, not per-cube -- lines sharing one cube can still
        need different PSF correction), caching it for Convolve Maps/Run Map.

        Args:
            None

        Returns:
            None
        """
        if self._rot_map_worker_busy():
            QMessageBox.warning(
                self, "Fit Still Stopping",
                "The previous map operation is still shutting down after being "
                "cancelled. Wait a few seconds and try again.")
            return

        cubes = list(self.loaded_batch_cubes)

        line_table, discarded_ul, err = self._rot_map_gather_lines(min_lines=1)
        if err:
            QMessageBox.warning(self, "Lines", err)
            return

        imported_rows = [l for l in line_table if l.get('imported_map')]
        cube_rows = [l for l in line_table if not l.get('imported_map')]

        if not cubes and not imported_rows:
            QMessageBox.warning(
                self, "No Cube",
                "Load one or more FITS cubes first (Load Cubes), or add "
                "a line via Load Maps.")
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

        cubes_and_lines = []
        if cube_rows:
            cubes_and_lines, match_err = self._rot_map_match_lines_to_cubes(cubes, cube_rows)
            if match_err:
                QMessageBox.warning(self, "Cubes", match_err)
                return
        n_sigma = self._rot_map_gather_n_sigma()
        if n_sigma is None:
            return

        # A pure-import checked set has nothing to fit -- merging is
        # synchronous, so finish right here instead of starting a worker.
        if not cube_rows:
            merged = merge_native_results(None, imported_rows)
            geometry_key = self._rot_map_geometry_key([], n_sigma, imported_lines=imported_rows)
            self._rot_map_native_cache = {"key": geometry_key, "result": merged}
            self._clear_all_convolved_maps()
            self._populate_map_column(merged)
            QMessageBox.information(
                self, "Calculate Maps",
                "Imported map(s) ready. View them in the Map column, or click "
                "Convolve Maps to put them on a common grid.")
            return

        threshold_note = (
            f"Detection threshold: {n_sigma:g}σ\n\n"
            f"A line fitted below {n_sigma:g}σ is an upper limit and stays NaN in the maps "
            "(Draw Region still shows its fitted value, marked).\n\n"
        )
        if imported_rows:
            threshold_note += "(Applies to the cube-fit lines only -- imported maps are used as-is.)\n\n"
        proceed = QMessageBox.question(
            self, "Calculate Maps", threshold_note + "Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if proceed != QMessageBox.Yes:
            return

        self.rot_map_progress = QProgressDialog(
            "Generating integrated flux maps per pixel...", "Cancel", 0, 0, self)
        self.rot_map_progress.setWindowTitle("Calculate Maps")
        self.rot_map_progress.setWindowModality(Qt.WindowModal)
        self.rot_map_progress.setAutoClose(False)
        self.rot_map_progress.setAutoReset(False)
        self.rot_map_progress.canceled.connect(self._cancel_rot_map)
        self.rot_map_progress.show()
        self._rot_map_progress_label = "Generating integrated flux maps"

        self.run_fit_btn.setEnabled(False)
        self.run_map_btn.setEnabled(False)
        self.calculate_maps_btn.setEnabled(False)
        self.convolve_maps_btn.setEnabled(False)

        self._calc_maps_cubes_and_lines = cubes_and_lines
        self._calc_maps_n_sigma = n_sigma
        self._calc_maps_imported_rows = imported_rows
        self.rot_map_worker = CalculateNativeMapsWorker(
            cubes_and_lines, contam_lines=self.contam_lines, n_sigma=n_sigma)
        self.rot_map_worker.progress.connect(self._on_rot_map_progress)
        self.rot_map_worker.finished.connect(self._on_calculate_maps_multi_complete)
        self.rot_map_worker.error.connect(self._on_calculate_maps_error)
        self.rot_map_worker.start()

    def _on_calculate_maps_multi_complete(self, result):
        """Merge in any imported rows, cache the combined native-resolution
        result, and populate the Map column.

        Args:
            result: Computed compute_native_line_maps result.

        Returns:
            None
        """
        dlg = getattr(self, "rot_map_progress", None)
        if dlg is not None:
            dlg.close()
        self.on_line_selected()
        imported_rows = getattr(self, "_calc_maps_imported_rows", [])
        merged = merge_native_results(result, imported_rows) if imported_rows else result
        geometry_key = self._rot_map_geometry_key(
            self._calc_maps_cubes_and_lines, self._calc_maps_n_sigma, imported_lines=imported_rows)
        self._rot_map_native_cache = {"key": geometry_key, "result": merged}
        self._clear_all_convolved_maps()
        self._populate_map_column(merged)
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
        """Convolve/reproject every checked line's native map (cube-fit or
        imported) onto a common grid. A cube-backed line needs a matching
        Calculate Maps cache; an imported line needs none, its map is final.

        Args:
            None

        Returns:
            None
        """
        if self._rot_map_worker_busy():
            QMessageBox.warning(
                self, "Fit Still Stopping",
                "The previous map operation is still shutting down after being "
                "cancelled. Wait a few seconds and try again.")
            return

        cubes = list(self.loaded_batch_cubes)

        line_table, discarded_ul, err = self._rot_map_gather_lines(min_lines=1)
        if err:
            QMessageBox.warning(self, "Lines", err)
            return

        imported_rows = [l for l in line_table if l.get('imported_map')]
        cube_rows = [l for l in line_table if not l.get('imported_map')]

        if len(cube_rows) + len(imported_rows) < 2:
            QMessageBox.warning(
                self, "Convolve Maps",
                "Convolve Maps needs 2+ lines (cube-fit or imported) -- "
                "with only one, there is only one resolution and nothing "
                "to convolve.")
            return

        cubes_and_lines = []
        if cube_rows:
            cubes_and_lines, match_err = self._rot_map_match_lines_to_cubes(cubes, cube_rows)
            if match_err:
                QMessageBox.warning(self, "Cubes", match_err)
                return

        native_cache = getattr(self, "_rot_map_native_cache", None)
        if cubes_and_lines:
            if native_cache is None:
                QMessageBox.warning(
                    self, "Convolve Maps",
                    "No Calculate Maps result found -- run Calculate Maps first.")
                return
            n_sigma = native_cache["key"][2]
            subset, missing = self._rot_map_subset_native_result(native_cache["result"], cubes_and_lines)
            if subset is None:
                wavelengths = "\n".join(f"  {wl:.4f} um" for wl in missing)
                QMessageBox.warning(
                    self, "Convolve Maps",
                    f"Calculate Maps hasn't been run for:\n{wavelengths}\n\n"
                    "Run Calculate Maps first (already-calculated lines don't need to be redone).")
                return
        else:
            subset = None
            # No cube-backed lines, so there's nothing to look up a cached
            # threshold from unless an earlier Calculate Maps run left one.
            n_sigma = native_cache["key"][2] if native_cache is not None else self._rot_map_gather_n_sigma()
            if n_sigma is None:
                return

        merged = merge_native_results(subset, imported_rows)

        try:
            plan = convolution_plan(merged)
        except Exception as exc:
            QMessageBox.critical(self, "Plan Failed", str(exc))
            return
        if not self._show_convolution_plan_confirm(
                plan, window_title="Convolve Maps -- convolution plan",
                question="Convolve/reproject every line onto this common grid?",
                accept_label="Convolve Maps"):
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            convolved = convolve_reproject_maps(merged)
        except Exception as exc:
            QMessageBox.critical(self, "Convolve Maps Error", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        geometry_key = self._rot_map_geometry_key(cubes_and_lines, n_sigma, imported_lines=imported_rows)
        self._rot_map_convolved_cache = {"key": geometry_key, "result": convolved}
        self.on_line_selected()  # unblock Run Map now that a convolved cache exists
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

    def _rot_map_gather_n_sigma(self):
        """Prompt for the detection threshold. Any number 0 or higher is
        allowed -- there is no upper limit.

        Args:
            None

        Returns:
            n_sigma (float or None): The threshold in noise sigmas, or None
                if the dialog was cancelled.
        """
        n_sigma, ok = QInputDialog.getDouble(
            self, "Detection Threshold", "Detection threshold (σ):",
            3.0, 0.0, 1.0e6, 1)
        if not ok:
            return None
        return n_sigma

    def _rot_map_gather_settings(self, n_sigma=None):
        """Parse distance/radius/OPR-grid/threshold inputs into fit_kwargs,
        the same way Run Map always has. Shared by Run Map, Calculate Maps
        (single cube only needs this), and the fast Run Map combine path.

        Args:
            n_sigma (float): Threshold to use as-is, skipping the prompt --
                Run Map passes its already-cached convolved result's own
                threshold here, since asking again could silently pick a
                different one and trigger a full native refit. None prompts
                as usual (Calculate Maps, which sets that threshold first).

        Returns:
            fit_kwargs (dict or None): molecule/isotopologue/opr/fit_opr/
                opr_grid/dist_pc/r_au/min_lines/n_sigma/fit2t, or None if a
                field was invalid -- a warning was already shown in that case.
        """
        if n_sigma is None:
            n_sigma = self._rot_map_gather_n_sigma()
            if n_sigma is None:
                return None
        radius_and_distance = self._read_radius_and_distance()
        if radius_and_distance is None:
            return None
        r_au, dist_pc = radius_and_distance

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

        fit2t = self.fit_mode.currentText() == "2T"
        # 2T has 4 free parameters vs 1T's 2, so it needs more detections per pixel.
        min_lines = 5 if fit2t else 3

        return dict(
            molecule=self.current_molecule,
            isotopologue=self.current_isotopologue,
            opr=None, fit_opr=fit_opr, opr_grid=opr_grid,
            dist_pc=dist_pc, r_au=r_au, min_lines=min_lines, n_sigma=n_sigma,
            fit2t=fit2t,
        )

    def _rot_map_geometry_key(self, cubes_and_lines, n_sigma, dedup_tol_um=1e-5, imported_lines=None):
        """Cache key for compute_native_line_maps/convolve_reproject_maps output
        -- only what determines the maps themselves (cube/imported-file paired
        with wavelength, detection threshold), not fit settings they never see.

        Args:
            cubes_and_lines (list): (CubeData, line_dict) pairs
            n_sigma (float): Detection threshold the maps were (or will be) made with
            dedup_tol_um (float): Co-located-line merge tolerance in microns
            imported_lines (list): Line dicts with an 'imported_map' payload, or None

        Returns:
            key (tuple): Hashable, comparable snapshot
        """
        pairs = [(getattr(c, "filename", None), round(l["wavelength"], 6)) for c, l in cubes_and_lines]
        pairs += [(l["imported_map"]["filename"], round(l["wavelength"], 6))
                  for l in (imported_lines or [])]
        return (tuple(sorted(pairs)), dedup_tol_um, n_sigma)

    def _rot_map_subset_native_result(self, native_result, cubes_and_lines, dedup_tol_um=1e-5):
        """Pick out just the requested (cube, line) pairs from an already-
        cached compute_native_line_maps result, so deselecting some checked
        lines lets Convolve Maps reuse the native maps already computed for
        the rest instead of forcing Calculate Maps to rerun for them --
        native maps only change with the cubes/lines/detection threshold,
        not with which subset happens to be checked right now.

        Args:
            native_result (dict): A cached compute_native_line_maps output.
            cubes_and_lines (list): (CubeData, line_dict) pairs currently checked.
            dedup_tol_um (float): Matching tolerance in microns.

        Returns:
            result (tuple): (subset, missing) -- subset is native_result
                filtered/reordered to cubes_and_lines (same shape
                compute_native_line_maps returns), or None if any requested
                line isn't in the cache; missing lists the wavelengths (um)
                not found, empty on success.
        """
        cached = list(zip(
            native_result["cube_filenames"],
            [l["wavelength"] for l in native_result["kept_lines"]],
        ))
        indices, missing = [], []
        for cube, line in cubes_and_lines:
            fname, wl = getattr(cube, "filename", None), line["wavelength"]
            idx = next((i for i, (cf, cwl) in enumerate(cached)
                       if cf == fname and abs(cwl - wl) <= dedup_tol_um), None)
            if idx is None:
                missing.append(wl)
            else:
                indices.append(idx)
        if missing:
            return None, missing
        subset = {
            key: [native_result[key][i] for i in indices]
            for key in ("kept_lines", "native_flux_maps", "native_flux_error_maps",
                       "native_wcs", "native_shapes", "native_pix_arcsec", "cube_filenames",
                       "native_instrument", "native_headers", "native_fit_flux_maps",
                       "native_status_maps", "native_ul_ceiling_maps")
        }
        subset["dropped_lines"] = native_result["dropped_lines"]
        return subset, []

    def _show_convolution_plan_confirm(self, plan, window_title="Run Map -- convolution plan",
                                       question="Run the fit at every pixel of the common grid?",
                                       accept_label="Run Map",
                                       grid_note="(reference line -- longest wavelength)"):
        """Show the convolve/reproject plan as a sortable table and let the
        user confirm or cancel before it runs. Shared by Run Map and Convolve
        Maps (wording differs, the plan table doesn't); sorts via its own
        plain closures so it can't collide with lines_table's sort state.

        Args:
            plan (dict): Convolution/reprojection plan from convolution_plan().
            window_title (str): Dialog title.
            question (str): Final line asking the user to confirm.
            accept_label (str): Text on the accept button.
            grid_note (str): Short parenthetical after the grid's pixel
                scale, describing why that line's resolution was chosen.

        Returns:
            proceed (bool): True if the user chose to proceed.
        """
        ny, nx = plan["dst_shape"]
        worst = plan["worst_pix_arcsec"]
        rows = list(plan["per_cube"])

        dlg = QDialog(self)
        dlg.setWindowTitle(window_title)
        dlg.resize(980, 480)
        lay = QVBoxLayout(dlg)

        header_label = QLabel(f"Common grid: {ny} x {nx} px @ {worst:.4f}\"/px {grid_note}")
        lay.addWidget(header_label)

        headers = ["Line (um)", "File", "Pixel Scale (\"/px)",
                   "Native PSF (FWHM, \")", "PSF Matching Kernel (FWHM, \")"]
        table = QTableWidget(len(rows), len(headers), dlg)
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # File names (JWST ones especially) can run long enough to blow the
        # dialog out to an unreasonable width -- cap this one column and
        # show the full name as a tooltip instead of forcing it to fit.
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        table.setColumnWidth(1, 260)
        table.horizontalHeader().setStretchLastSection(True)

        def _native_psf_arcsec(c):
            """This line's own native angular resolution (its instrument's PSF
            formula), before any convolution -- the reference row's value is the
            common resolution every other row is being PSF-matched to."""
            return psf_fwhm_arcsec(c["wavelength"], c["instrument"])

        def _blur_text(c):
            """
            FWHM of the Gaussian kernel applied to raise this line's own
            native PSF up to the reference line's native PSF (0 for the
            reference line itself -- it needs no PSF matching).

            Args:
                c (dict): One convolution_plan()['per_cube'] entry.

            Returns:
                text (str): FWHM of the PSF-matching kernel, in arcsec and
                    native pixels.
            """
            fwhm_arcsec = c["sigma_arcsec"] * gaussian_sigma_to_fwhm
            fwhm_pix = c["sigma_pix"] * gaussian_sigma_to_fwhm
            return f"{fwhm_arcsec:.4f}\" ({fwhm_pix:.2f} px)"

        def _fill():
            """
            Redraw every row of `rows` into `table`, in its current order.

            Args: None

            Returns:
                None
            """
            for r, c in enumerate(rows):
                fname = os.path.basename(c["filename"]) if c["filename"] else "cube"
                if c["sigma_pix"] <= 1e-6:
                    fname += "  (reference)"
                texts = (f"{c['wavelength']:.4f}", fname,
                        f"{c['native_pix_arcsec']:.4f}",
                        f"{_native_psf_arcsec(c):.4f}\"", _blur_text(c))
                for col, text in enumerate(texts):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter if col != 1 else Qt.AlignVCenter | Qt.AlignLeft)
                    if col == 1:
                        item.setToolTip(fname)
                    table.setItem(r, col, item)

        _fill()

        sort_key = {
            0: lambda c: c["wavelength"],
            1: lambda c: (os.path.basename(c["filename"]) if c["filename"] else "cube").lower(),
            2: lambda c: c["native_pix_arcsec"],
            3: lambda c: _native_psf_arcsec(c),
            4: lambda c: c["sigma_pix"],
        }
        sort_state = {"column": None, "order": Qt.AscendingOrder}

        def _on_header_clicked(column):
            """
            Sort `rows` by the clicked column, toggling direction on a repeat click.

            Args:
                column (int): Table column index the user clicked.

            Returns:
                None
            """
            if sort_state["column"] == column and sort_state["order"] == Qt.AscendingOrder:
                sort_state["order"] = Qt.DescendingOrder
            else:
                sort_state["order"] = Qt.AscendingOrder
            sort_state["column"] = column
            rows.sort(key=sort_key[column], reverse=(sort_state["order"] == Qt.DescendingOrder))
            _fill()
            table.horizontalHeader().setSortIndicator(column, sort_state["order"])

        header = table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.sectionClicked.connect(_on_header_clicked)
        lay.addWidget(table)

        # Size the dialog to fit every column's actual content. Measured with
        # stretch-last-section off so it reflects true natural widths, not
        # whatever the last column happens to be stretched to at its old size.
        table.horizontalHeader().setStretchLastSection(False)
        natural_width = table.horizontalHeader().length()
        table.horizontalHeader().setStretchLastSection(True)
        frame_overhead = (table.verticalHeader().width() + 2 * table.frameWidth()
                         + table.style().pixelMetric(table.style().PM_ScrollBarExtent) + 40)
        dlg.resize(min(max(dlg.width(), natural_width + frame_overhead), 1600), dlg.height())

        footer_label = QLabel(
            "Every pixel outside a cube's own reprojected footprint is NaN "
            f"for that line.\n\n{question}")
        footer_label.setWordWrap(True)
        lay.addWidget(footer_label)

        btns = QHBoxLayout()
        accept_btn = QPushButton(accept_label)
        cancel_btn = QPushButton("Cancel")
        accept_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btns.addStretch()
        btns.addWidget(accept_btn)
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
        label = getattr(self, "_rot_map_progress_label", "Fitting rotational diagram")
        dlg.setLabelText(f"{label}: {current}/{total} pixels")

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
        # (wavelength, imported-file-or-None) so re-importing a different
        # file at the same wavelength correctly invalidates the cache.
        wavelengths = tuple(sorted(
            (round(l["wavelength"], 6), (l.get("imported_map") or {}).get("filename"))
            for l in line_table
        ))
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
        results_text += input_lines(*self._fit_inputs)
        if "OPR" not in results:
            results_text += "\n"

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
        text_box = QWidget()
        text_layout = QVBoxLayout(text_box)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.addWidget(text_widget)
        layout.addWidget(text_box, 1)
        if "_plot_data" in results:
            save_btn = QPushButton("Save Fit")
            save_btn.setToolTip("Save these results and the plotted points to a CSV file.")
            save_btn.clicked.connect(lambda: save_fit_csv(
                dialog, results_text, results["_plot_data"],
                f"{self.current_molecule}_rotational_diagram_fit.csv"))
            text_layout.addWidget(save_btn)

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
                        color='blue', label=f'T1: {T1_2:.0f} K')

                y_comp2 = a2_2 - E_fit / T2_2
                ax.plot(E_fit, y_comp2 / np.log(10), ':', linewidth=2.5,
                        color='orange', label=f'T2: {T2_2:.0f} K')

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
            attach_rot_diagram_hover(canvas, ax, plot_data)
            toolbar = NavigationToolbar(canvas, dialog)

            plot_layout.addWidget(toolbar)
            plot_layout.addWidget(canvas)
            layout.addWidget(plot_widget, 1)

        dialog.exec_()
