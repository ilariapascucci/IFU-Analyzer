"""
Main GUI class for the Cube Explorer application
"""

import os
import re
import gc
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, Normalize

# Suppress FITS warning messages about header formatting
warnings.filterwarnings('ignore', category=Warning, module='astropy.wcs.wcs')
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QMessageBox, QTextEdit, QLineEdit, QInputDialog, QDialog, QToolTip, QComboBox,
    QListWidget, QListWidgetItem, QSplitter, QFrame, QProgressDialog, QDoubleSpinBox,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtWidgets import QApplication

from scipy.ndimage import median_filter
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks  # NEW: Import find_peaks

from .data_loader import CubeData
from .gui_continuum import GuiContinuumMixin
from .gui_gaussian import GuiGaussianMixin
from .gui_coflux import GuiCOFluxMixin
from .gui_spectrum import GuiSpectrumMixin
from .visualization import SpectrumPlotter, SlicePlotter
from .colormap_dialog import ColorMapDialog
from .region_selector import RegionSelector
from .continuum_dialog import ContinuumParametersDialog
from .custom_continuum import CustomContinuumSelector
from .line_catalog import get_available_molecules, normalize_species_input, hitran_gui_interface, load_CO_line_catalog, load_contaminating_lines_catalog, load_custom_line_catalog
from .species_dialog import SpeciesDialog
from .gaussian_fitter import fit_gaussian_to_line, GaussianFitResult, auto_detect_boundaries
from .gaussian_dialogs import GaussianMapWorker, GaussianMapResultsDialog
from .co_flux_dialogs import COFluxMapWorker, COFluxMapResultsDialog
from .unit_converter import convert_wavelength, are_units_compatible, clean_flux_unit, clean_wave_unit
from astropy.units import Quantity


def _extract_wavelength_value(wavelength_quantity_or_array, index=None):
    """
    Extract numeric value from wavelength Quantity or array.
    
    Args:
        wavelength_quantity_or_array: Wavelength (Quantity, scalar, or array)
        index: Optional index for array access
    
    Returns:
        Numeric value (units stripped if Quantity)
    """
    if isinstance(wavelength_quantity_or_array, Quantity):
        if index is not None:
            return wavelength_quantity_or_array[index].value
        else:
            return wavelength_quantity_or_array.value
    else:
        if index is not None:
            return wavelength_quantity_or_array[index]
        else:
            return wavelength_quantity_or_array
        
class CollapsibleButtonGroup(QWidget):
    """
    A collapsible button group that expands/collapses with animation.
    Uses snake_case for all method and variable names.
    Supports accordion behavior - only one group can be expanded at a time.
    """
    # Class-level list to track all instances for accordion behavior
    _all_groups = []
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.is_expanded = False
        self.title = title
        # Register this instance globally
        CollapsibleButtonGroup._all_groups.append(self)
        self.init_ui(title)
    
    def init_ui(self, title):
        """Initialize the UI components"""
        # Main horizontal layout
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)  # Tighter spacing
        
        # Toggle button (always visible)
        self.toggle_btn = QPushButton(f"▶ {title}")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setFixedWidth(150)  # Wider to fit "Continuum Tools"
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
                text-align: left;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:checked {
                background-color: #2d73c5;
            }
        """)
        self.toggle_btn.clicked.connect(self.toggle)
        self.main_layout.addWidget(self.toggle_btn)
        
        # Container for child buttons (collapsible)
        self.button_container = QFrame()
        self.button_layout = QHBoxLayout(self.button_container)
        self.button_layout.setContentsMargins(2, 0, 0, 0)
        self.button_layout.setSpacing(2)  # Buttons closer together
        
        # Initially hidden
        self.button_container.setMaximumWidth(0)
        self.button_container.setStyleSheet("QFrame { background-color: transparent; }")
        self.main_layout.addWidget(self.button_container)
        
        # Animation for smooth expand/collapse
        self.animation = QPropertyAnimation(self.button_container, b"maximumWidth")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
    
    def add_button(self, button):
        """Add a button to this collapsible group"""
        # Use default button styling (no custom stylesheet)
        self.button_layout.addWidget(button)
    
    def close_other_groups(self):
        """Close all other collapsible groups (accordion behavior)"""
        for group in CollapsibleButtonGroup._all_groups:
            if group is not self and group.is_expanded:
                group.close()
    
    def close(self):
        """Close/collapse this group"""
        if self.is_expanded:
            self.toggle_btn.setChecked(False)
            self.toggle()
    
    def toggle(self):
        """Toggle expand/collapse with animation"""
        if self.toggle_btn.isChecked():
            # close other groups when expanding (accordion behavior)
            self.close_other_groups()
            
            # Expand
            self.is_expanded = True
            self.toggle_btn.setText(self.toggle_btn.text().replace("▶", "▼"))
            
            # Calculate target width
            target_width = sum(
                self.button_layout.itemAt(i).widget().sizeHint().width() 
                for i in range(self.button_layout.count())
            )
            target_width += self.button_layout.spacing() * (self.button_layout.count() - 1)
            target_width += 4
            
            self.animation.setStartValue(0)
            self.animation.setEndValue(target_width)
            self.animation.start()
        else:
            # Collapse
            self.is_expanded = False
            self.toggle_btn.setText(self.toggle_btn.text().replace("▼", "▶"))
            
            self.animation.setStartValue(self.button_container.maximumWidth())
            self.animation.setEndValue(0)
            self.animation.start()

class CustomNavigationToolbar(NavigationToolbar):
    """
    Custom matplotlib toolbar with customizable button behavior.
    
    Allows overriding default button actions by setting custom callbacks.
    To customize button behavior, set callback functions like:
        toolbar.custom_home_action = my_custom_function
    """
    
    def __init__(self, canvas, parent=None, coordinates=True):
        super().__init__(canvas, parent, coordinates)
        # Initialize custom action callbacks
        self.custom_home_action = None
        self.custom_back_action = None
        self.custom_forward_action = None
        self.custom_pan_action = None
        self.custom_zoom_action = None
        self.custom_save_action = None
    
    def home(self, *args):
        """Override home button - calls custom action if set, otherwise default"""
        if self.custom_home_action is not None:
            self.custom_home_action(*args)
        else:
            super().home(*args)
    
    def back(self, *args):
        """Override back button - calls custom action if set, otherwise default"""
        if self.custom_back_action is not None:
            self.custom_back_action(*args)
        else:
            super().back(*args)
    
    def forward(self, *args):
        """Override forward button - calls custom action if set, otherwise default"""
        if self.custom_forward_action is not None:
            self.custom_forward_action(*args)
        else:
            super().forward(*args)
    
    def pan(self, *args):
        """Override pan button - calls custom action if set, otherwise default"""
        if self.custom_pan_action is not None:
            self.custom_pan_action(*args)
        else:
            super().pan(*args)
    
    def zoom(self, *args):
        """Override zoom button - calls custom action if set, otherwise default"""
        if self.custom_zoom_action is not None:
            self.custom_zoom_action(*args)
        else:
            super().zoom(*args)
    
    def save_figure(self, *args):
        """Override save button - calls custom action if set, otherwise default"""
        if self.custom_save_action is not None:
            self.custom_save_action(*args)
        else:
            super().save_figure(*args)


class WavelengthSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox whose up/down arrows (and Up/Down/PageUp/PageDown keys)
    step to the next/previous wavelength actually present in the cube,
    instead of a fixed increment - real wavelength grids aren't evenly
    spaced, so a fixed step would often skip past or land between samples.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wavelengths = None  # sorted ascending numpy array, or None

    def set_wavelength_grid(self, wavelengths):
        self._wavelengths = wavelengths

    def stepBy(self, steps):
        wl = self._wavelengths
        if wl is None or len(wl) == 0:
            super().stepBy(steps)
            return
        idx = int(np.abs(wl - self.value()).argmin())
        idx = max(0, min(len(wl) - 1, idx + steps))
        self.setValue(float(wl[idx]))


class CubeExplorer(QWidget, GuiContinuumMixin, GuiGaussianMixin, GuiCOFluxMixin, GuiSpectrumMixin):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cube Explorer")
        self.cube_data = CubeData()
        self._init_ui()
        # Load CO line catalogs on startup
        self._load_CO_catalogs()

    def closeEvent(self, event):
        # A Gaussian map fit runs in its own QThread and previously kept computing (and
        # printing to the console) after this window closed, since nothing ever told it
        # to stop. Request cancellation and give it a few seconds to actually exit
        # before letting the window go away.
        worker = getattr(self, 'gauss_worker', None)
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(5000)

        try:
            self.cube_data.free_memory()
        except AttributeError:
            pass
        plt.close('all')
        gc.collect()
        event.accept()

    def _load_CO_catalogs(self):
        """Load CO line catalogs at GUI startup."""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        CO_path = os.path.join(base_path, 'data', 'CO', 'CO_comprehensive_linelist_with_g.ecsv')
        
        self.CO_lines = load_CO_line_catalog(CO_path)
        # Contaminating lines are now optional - users can load them separately after HITRAN parameters are set
        self.contam_lines = None
    def _clear_all_displays(self):
        """Clear all plots, selections, and stats when loading a new cube."""
        print("=== CLEAR ALL DISPLAYS START ===")
        
        # Disconnect interactive patches before clearing axes
        if hasattr(self, 'region_selector_left') and self.region_selector_left.interactive_patch is not None:
            print("[DEBUG] Disconnecting left interactive patch")
            self.region_selector_left.interactive_patch.disconnect()
            self.region_selector_left.interactive_patch = None
        
        if hasattr(self, 'region_selector_right') and self.region_selector_right.interactive_patch is not None:
            print("[DEBUG] Disconnecting right interactive patch")
            self.region_selector_right.interactive_patch.disconnect()
            self.region_selector_right.interactive_patch = None

        # --- Slice 1 ---
        if hasattr(self, 'slice1_plotter'):
            print(f"[DEBUG] Clearing slice1_plotter, ax exists: {hasattr(self.slice1_plotter, 'ax')}")
            ax1 = self.slice1_plotter.ax
            if ax1:
                ax1.clear()
                # Remove colorbar safely
                if hasattr(self.slice1_plotter, 'cbar') and self.slice1_plotter.cbar is not None:
                    try:
                        self.slice1_plotter.cbar.remove()
                    except Exception as e:
                        print(f"[DEBUG] Error removing slice1 colorbar: {e}")
                    self.slice1_plotter.cbar = None
                # Reset axes explicitly
                ax1.set_xlabel("")
                ax1.set_ylabel("")
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                ax1.relim()
                ax1.autoscale()
                self.slice1_canvas.draw()
                print("[DEBUG] Slice1 canvas redrawn")

        # --- Slice 2 ---
        if hasattr(self, 'slice2_plotter'):
            print(f"[DEBUG] Clearing slice2_plotter, ax exists: {hasattr(self.slice2_plotter, 'ax')}")
            ax2 = self.slice2_plotter.ax
            if ax2:
                ax2.clear()
                if hasattr(self.slice2_plotter, 'cbar') and self.slice2_plotter.cbar is not None:
                    try:
                        self.slice2_plotter.cbar.remove()
                    except Exception as e:
                        print(f"[DEBUG] Error removing slice2 colorbar: {e}")
                    self.slice2_plotter.cbar = None
                ax2.set_xlabel("")
                ax2.set_ylabel("")
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])
                ax2.relim()
                ax2.autoscale()
                self.slice2_canvas.draw()
                print("[DEBUG] Slice2 canvas redrawn")

        # --- Spectrum plot ---
        if hasattr(self, 'spec_plotter') and hasattr(self.spec_plotter, 'ax'):
            print(f"[DEBUG] Clearing spec_plotter, ax exists: True")
            ax_spec = self.spec_plotter.ax
            ax_spec.clear()
            ax_spec.set_xlabel("")
            ax_spec.set_ylabel("")
            ax_spec.set_xlim(0, 1)
            ax_spec.set_ylim(0, 1)
            ax_spec.set_xticklabels([])
            ax_spec.set_yticklabels([])
            ax_spec.relim()
            ax_spec.autoscale()
            self.spec_canvas.draw()
            print("[DEBUG] Spectrum canvas redrawn")

        # --- Reset stats labels ---
        self.stats1_label.setText("Slice 1 Stats:\n-")
        self.stats2_label.setText("Slice 2 Stats:\n-")
        print("[DEBUG] Resetting stats labels")

        # --- Reset selections ---
        self.last_selection_left = None
        self.last_selection_right = None
        self.last_idx_left = 0
        self.last_idx_right = 0
        self.original_spectrum = None
        print("[DEBUG] Resetting selections")

        print("[DEBUG] Clearing continuum-related data...")

        if hasattr(self, 'custom_continuum_selector') and self.custom_continuum_selector is not None:
            self.custom_continuum_selector.clear_all()
            self.custom_continuum_selector.deactivate()
            self.custom_continuum_selector = None
            self.custom_continuum_active = False

        # Uncheck custom subtract button
        if hasattr(self, 'custom_subtract_btn'):
            self.custom_subtract_btn.setChecked(False)

        # Clear continuum line artist
        self.continuum_line = None

        # Clear fitted continuum data
        self.continuum_fitted = None

        # Clear backup spectrum
        self.original_spectrum_backup = None

        # Reset subtraction state
        self.continuum_subtracted = False
        self.auto_continuum_subtracted = False

        # Clear subtraction markers
        if hasattr(self, 'subtract_markers'):
            for marker in self.subtract_markers:
                try:
                    marker.remove()
                except Exception:
                    pass
            self.subtract_markers = []

        # Reset subtraction mode variables
        self.subtract_mode = None
        self.subtract_left_x = None
        self.subtract_right_x = None

        # Reset continuum toggles
        if hasattr(self, 'auto_continuum_toggle'):
            self.auto_continuum_toggle.setChecked(False)
        if hasattr(self, 'custom_continuum_toggle'):
            self.custom_continuum_toggle.setChecked(False)

        # Hide action buttons
        if hasattr(self, 'hide_all_action_buttons'):
            self.hide_all_action_buttons()

        print("[DEBUG] Continuum data cleared successfully")

        # --- Uncheck buttons ---
        for side in ["left", "right"]:
            for btn in self.shape_buttons.get(side, {}).values():
                btn.setChecked(False)
        self.set_left_btn.setChecked(False)
        self.set_right_btn.setChecked(False)

        # Reset region-selection state so shape/Edit-Shape buttons don't stay
        # disabled from a stale "drawing"/"placed"/"editing" state.
        if hasattr(self, 'region_state'):
            for side in ["left", "right"]:
                self._set_region_state(side, "idle")

        self.active_target = None
        
        # Reset per-tab Doppler shift state - new cube, new base wavelength.
        self.base_wavelength = None
        if hasattr(self, 'spectrum_shift_factor'):
            self.spectrum_shift_factor = {"full": 1.0, "left": 1.0, "right": 1.0}
        if hasattr(self, 'spectrum_shift_preview'):
            self.spectrum_shift_preview = {"full": None, "left": None, "right": None}
        if hasattr(self, '_shift_preview_line') and self._shift_preview_line is not None:
            try:
                self._shift_preview_line.remove()
            except Exception:
                pass
            self._shift_preview_line = None
        if hasattr(self, 'apply_shift_btn'):
            self.apply_shift_btn.setEnabled(False)
        if hasattr(self, 'cancel_shift_btn'):
            self.cancel_shift_btn.setEnabled(False)
        if hasattr(self, 'revert_shift_btn'):
            self.revert_shift_btn.setEnabled(False)
        
        print("[DEBUG] Unchecking all buttons")

        # --- Reset labels ---
        self.instrument_label.setText("Instrument: -")
        print("[DEBUG] Resetting instrument label")

        # --- Force garbage collection ---
        gc.collect()

        print("=== CLEAR ALL DISPLAYS END ===")

    def _init_ui(self):
        # File controls
        self._init_file_controls()

        # Initialize plots
        self._init_plots()

        # Spectrum tab bar (Full/Left/Right)
        self._init_spectrum_tabs()

        # Shape selection controls
        self._init_shape_controls()

        # Main layout setup
        self._setup_main_layout()

        # Initialize state variables
        self._init_state()

    def _init_file_controls(self):
        """Initialize file control buttons and labels"""
        self.open_btn = QPushButton("Open FITS")
        self.open_btn.clicked.connect(self.load_cube)
        self.open_btn.setToolTip("Browse and load a FITS cube file")

        self.header_btn = QPushButton("Header")
        self.header_btn.clicked.connect(self.show_header)
        self.header_btn.setToolTip("Display the FITS header information")

        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.clicked.connect(self.refresh_view)
        self.refresh_btn.setToolTip("Refresh all plots")

        # New Window button: spawn another CubeExplorer window
        self.new_window_btn = QPushButton("🗗")
        self.new_window_btn.setToolTip("Open a new Cube Explorer window")
        self.new_window_btn.clicked.connect(self._open_new_window)

        self.two_d_btn = QPushButton("2D")
        self.two_d_btn.setToolTip("Open a 2D FITS analyzer window")
        self.two_d_btn.clicked.connect(self.open_2d_window)

        self.rot_diag_btn = QPushButton("Rot Diag")
        self.rot_diag_btn.setToolTip("Open rotational diagram analyzer for H2")
        self.rot_diag_btn.clicked.connect(self.open_rotational_diagram)
        
        self.instrument_label = QLabel("Instrument: -")
        self.instrument_label.setStyleSheet("font-weight: bold;")

        self.controls_layout = QHBoxLayout()
        self.controls_layout.addWidget(self.open_btn)
        self.controls_layout.addWidget(self.header_btn)
        self.controls_layout.addWidget(self.refresh_btn)
        self.controls_layout.addWidget(self.new_window_btn)
        self.controls_layout.addWidget(self.two_d_btn)
        self.controls_layout.addWidget(self.instrument_label)
        self.controls_layout.addStretch()

    def _init_plots(self):
        """Initialize matplotlib figures and canvases"""
        # Left slice
        self.slice1_fig = Figure(figsize=(4, 4), dpi=100)
        self.slice1_canvas = FigureCanvas(self.slice1_fig)
        self.slice1_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.slice1_canvas.setMinimumHeight(200)
        self.slice1_plotter = SlicePlotter(self.slice1_fig)
        self.slice1_toolbar = CustomNavigationToolbar(self.slice1_canvas, self)
        self.slice1_canvas.mpl_connect("button_press_event", lambda e: self._on_slice_click(e, "left"))

        # Right slice
        self.slice2_fig = Figure(figsize=(4, 4), dpi=100)
        self.slice2_canvas = FigureCanvas(self.slice2_fig)
        self.slice2_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.slice2_canvas.setMinimumHeight(200)
        self.slice2_plotter = SlicePlotter(self.slice2_fig)
        self.slice2_toolbar = CustomNavigationToolbar(self.slice2_canvas, self)
        self.slice2_canvas.mpl_connect("button_press_event", lambda e: self._on_slice_click(e, "right"))

        # Spectrum
        self.spec_fig = Figure(figsize=(8, 3), dpi=100)
        self.spec_canvas = FigureCanvas(self.spec_fig)
        self.spec_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.spec_canvas.setMinimumHeight(220)
        self.spec_plotter = None  # Will be initialized after loading data
        for ax in self.spec_fig.axes:
            ax.set_clip_on(True)
        self.spec_toolbar = CustomNavigationToolbar(self.spec_canvas, self)
        self.spec_canvas.mpl_connect("button_press_event", self._on_spec_click)
        self.spec_canvas.mpl_connect("motion_notify_event", self._on_spec_hover)

        # Stats labels
        self.stats1_label = QLabel("Slice 1 Stats:\n-")
        self.stats1_label.setMinimumWidth(150)
        self.stats1_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        
        self.stats2_label = QLabel("Slice 2 Stats:\n-")
        self.stats2_label.setMinimumWidth(150)
        self.stats2_label.setStyleSheet("font-size: 10pt; font-weight: bold;")

        # Slice selection buttons
        self.set_left_btn = QPushButton("Set Left Slice")
        self.set_right_btn = QPushButton("Set Right Slice")

        # Direct wavelength entry for each slice - typing a value (or using
        # the spin arrows) jumps that slice to the closest available
        # wavelength, kept in sync with clicking on the spectrum/slice.
        self.left_wave_input = WavelengthSpinBox()
        self.left_wave_input.setFixedWidth(100)
        self.left_wave_input.setDecimals(6)
        self.left_wave_input.setKeyboardTracking(False)
        self.left_wave_input.setEnabled(False)
        self.left_wave_input.setToolTip(
            "Type a wavelength (or use the arrows) to jump the left slice "
            "to the closest available value"
        )

        self.right_wave_input = WavelengthSpinBox()
        self.right_wave_input.setFixedWidth(100)
        self.right_wave_input.setDecimals(6)
        self.right_wave_input.setKeyboardTracking(False)
        self.right_wave_input.setEnabled(False)
        self.right_wave_input.setToolTip(
            "Type a wavelength (or use the arrows) to jump the right slice "
            "to the closest available value"
        )

        self.shift_btn = QPushButton("Shift")
        self.apply_shift_btn = QPushButton("Apply Shift")
        self.cancel_shift_btn = QPushButton("Cancel Preview")
        self.revert_shift_btn = QPushButton("Revert Shift")


        # Gaussian fitting buttons
        self.gauss_left_btn = QPushButton("Left")
        self.gauss_peak_btn = QPushButton("Peak")
        self.gauss_right_btn = QPushButton("Right")
        self.gauss_noise_left_btn = QPushButton("Noise L")
        self.gauss_noise_right_btn = QPushButton("Noise R")
        self.gauss_fit_btn = QPushButton("Fit Gaussian")
        self.gauss_map_btn = QPushButton("Fit Gaussian Map")
        self.multi_gauss_fit_btn = QPushButton("Multi-Gauss")
        self.co_flux_map_btn = QPushButton("CO Flux Map")
        
        self.auto_continuum_toggle = QPushButton("Automatic Continuum")
        self.custom_continuum_toggle = QPushButton("Custom Continuum")

        self.auto_continuum_toggle.setCheckable(True)
        self.custom_continuum_toggle.setCheckable(True)

        # Action buttons (initially hidden)
        self.continuum_line_btn = QPushButton("Continuum Line")
        self.subtract_continuum_btn = QPushButton("Subtract Continuum")
        self.custom_subtract_btn = QPushButton("Custom Subtract")
        self.custom_delete_btn = QPushButton("Delete Point")
        self.custom_ok_btn = QPushButton("OK")

        self.custom_save_pts_btn = QPushButton("Save Points")
        self.custom_load_pts_btn = QPushButton("Load Points")

        self.custom_save_pts_btn.setToolTip("Save custom continuum points to a .txt file")
        self.custom_load_pts_btn.setToolTip("Load custom continuum points from a .txt file")

        self.save_results_btn = QPushButton("Save Results")

        # Make action buttons checkable
        self.continuum_line_btn.setCheckable(True)
        self.subtract_continuum_btn.setCheckable(True)
        self.custom_subtract_btn.setCheckable(True)
        self.custom_delete_btn.setCheckable(True)

        # Connect toggle buttons
        self.auto_continuum_toggle.clicked.connect(self.on_auto_continuum_toggle)
        self.custom_continuum_toggle.clicked.connect(self.on_custom_continuum_toggle)

        # Connect action buttons (keep existing connections)
        self.continuum_line_btn.clicked.connect(self._on_continuum_line_clicked)
        self.custom_subtract_btn.clicked.connect(self._activate_custom_continuum_subtraction)
        self.custom_delete_btn.clicked.connect(self._on_custom_delete_toggled)
        self.custom_ok_btn.clicked.connect(self.on_custom_ok_clicked)
        self.custom_save_pts_btn.clicked.connect(self._save_custom_points)
        self.custom_load_pts_btn.clicked.connect(self._load_custom_points)

        # Hide all action buttons initially
        self.continuum_line_btn.hide()
        self.subtract_continuum_btn.hide()
        self.custom_subtract_btn.hide()
        self.custom_ok_btn.hide()
        self.custom_save_pts_btn.hide()
        self.custom_load_pts_btn.hide()

        # Tooltips
        self.auto_continuum_toggle.setToolTip("Toggle automatic continuum fitting mode")
        self.custom_continuum_toggle.setToolTip("Toggle custom continuum point selection mode")
        self.continuum_line_btn.setToolTip("Fit and display continuum line")
        self.subtract_continuum_btn.setToolTip("Subtract the fitted continuum")
        self.custom_subtract_btn.setToolTip("Manually select continuum points and fit")
        self.custom_ok_btn.setToolTip("Calculate and apply custom continuum")
        self.custom_delete_btn.setToolTip("Toggle to delete points with left-click")

        self.subtract_continuum_btn.setCheckable(True)
        self.continuum_line_btn.setCheckable(True)
        self.custom_subtract_btn.setCheckable(True)      
        self.set_left_btn.setCheckable(True)
        self.set_right_btn.setCheckable(True)        
        self.gauss_left_btn.setCheckable(True)
        self.gauss_peak_btn.setCheckable(True)
        self.gauss_right_btn.setCheckable(True)
        self.gauss_noise_left_btn.setCheckable(True)
        self.gauss_noise_right_btn.setCheckable(True)
        
        self.set_left_btn.clicked.connect(lambda: self._select_target("left"))
        self.set_right_btn.clicked.connect(lambda: self._select_target("right"))
        self.left_wave_input.valueChanged.connect(lambda _v: self._on_wave_input_changed("left"))
        self.right_wave_input.valueChanged.connect(lambda _v: self._on_wave_input_changed("right"))
        self.shift_btn.clicked.connect(self.apply_doppler_shift)
        self.apply_shift_btn.clicked.connect(self._apply_shift)
        self.apply_shift_btn.setEnabled(False)
        self.apply_shift_btn.setToolTip("Apply the previewed shift to the active tab's spectrum")
        self.cancel_shift_btn.clicked.connect(self._cancel_shift_preview)
        self.cancel_shift_btn.setEnabled(False)
        self.cancel_shift_btn.setToolTip("Cancel the pending shift preview without applying it")
        self.revert_shift_btn.clicked.connect(self._revert_shift)
        self.revert_shift_btn.setEnabled(False)
        self.revert_shift_btn.setToolTip("Revert the active tab back to its original (unshifted) wavelength")
        self.gauss_left_btn.clicked.connect(self._activate_gaussian_left)
        self.gauss_peak_btn.clicked.connect(self._activate_gaussian_peak)
        self.gauss_right_btn.clicked.connect(self._activate_gaussian_right)
        self.gauss_noise_left_btn.clicked.connect(self._activate_gaussian_noise_left)
        self.gauss_noise_right_btn.clicked.connect(self._activate_gaussian_noise_right)
        self.gauss_fit_btn.clicked.connect(self._fit_gaussian)
        self.gauss_map_btn.clicked.connect(self._fit_gaussian_map)
        self.multi_gauss_fit_btn.clicked.connect(self._fit_multi_gaussian)
        self.co_flux_map_btn.clicked.connect(self._extract_co_flux_map)
        self.save_results_btn.clicked.connect(self._save_results_ascii)
        self.subtract_continuum_btn.clicked.connect(self._activate_continuum_subtraction)
        
        # Set tooltips for all buttons
        self.set_left_btn.setToolTip("Click on the left slice to set the wavelength from that pixel")
        self.set_right_btn.setToolTip("Click on the right slice to set the wavelength from that pixel")
        self.shift_btn.setToolTip("Preview a Doppler shift for the active tab's spectrum")
        self.gauss_left_btn.setToolTip("Click on spectrum to set left boundary of Gaussian fit")
        self.gauss_peak_btn.setToolTip("Click on spectrum to set peak center of Gaussian fit")
        self.gauss_right_btn.setToolTip("Click on spectrum to set right boundary of Gaussian fit")
        self.gauss_fit_btn.setToolTip("Fit a Gaussian to the selected line")
        self.save_results_btn.setToolTip(
            "Save wavelength, original spectrum, fitted continuum, "
            "and continuum-subtracted spectrum to a single ASCII file."
        )
        self.continuum_line_btn.setToolTip("Toggle continuum subtraction mode")
        self.subtract_continuum_btn.setToolTip("Subtract the fitted continuum from the spectrum")
        self.gauss_map_btn.setToolTip("Fit Gaussian to all pixels using Left/Right boundaries")
        self.multi_gauss_fit_btn.setToolTip("Fit Gaussians to all selected HITRAN lines using current continuum")
        self.gauss_noise_left_btn.setToolTip("Select left boundary of a line-free noise region")
        self.gauss_noise_right_btn.setToolTip("Select right boundary of a line-free noise region")
        self.custom_subtract_btn.setToolTip("Manually select continuum points")
        self.co_flux_map_btn.setToolTip("Fit multiple lines and and creat an integrated flux map")


        # Wavelength range inputs (for line search)
        self.min_wave_input = QDoubleSpinBox(self)
        self.min_wave_input.setFixedWidth(70)
        self.min_wave_input.setFixedHeight(28)
        self.min_wave_input.setRange(0.1, 1000)
        self.min_wave_input.setDecimals(6)
        self.min_wave_input.setSingleStep(0.1)
        self.min_wave_input.setToolTip("Min wavelength (λ)")
        self.max_wave_input = QDoubleSpinBox(self)
        self.max_wave_input.setFixedWidth(70)
        self.max_wave_input.setFixedHeight(28)
        self.max_wave_input.setRange(0.1, 1000)
        self.max_wave_input.setDecimals(6)
        self.max_wave_input.setSingleStep(0.1)
        self.max_wave_input.setToolTip("Max wavelength (λ)")

        # HITRAN collapsible group buttons
        self.hitran_btn = QPushButton("HITRAN")
        self.hitran_btn.setToolTip("Fetch HITRAN lines with transition level filtering")
        self.hitran_btn.clicked.connect(self.fetch_hitran_lines)

        self.load_custom_lines_btn = QPushButton("Import Lines")
        self.load_custom_lines_btn.setToolTip(
            "Load your own line list CSV (wavelength + species required) "
            "when HITRAN isn't reachable"
        )
        self.load_custom_lines_btn.clicked.connect(self._load_custom_lines_dialog)

        self.load_contam_btn = QPushButton("Contam. Lines")
        self.load_contam_btn.setToolTip("(Optional) Load CSV file with contaminating lines")
        self.load_contam_btn.clicked.connect(self._load_contaminating_lines_dialog)
        
        self.show_species_btn = QPushButton("Lines")
        self.show_species_btn.setToolTip("Show available HITRAN species")
        self.show_species_btn.clicked.connect(self.show_available_species)

    def _init_shape_controls(self):
        """Initialize shape selection buttons"""
        # Helper function to create shape buttons
        def make_shape_button(symbol, target, callback=None, checkable=True):
            btn = QPushButton(symbol)
            btn.setCheckable(checkable)
            btn.setFixedSize(40, 40)
            btn.setStyleSheet("QPushButton { font-size: 18pt; }")
            if callback is None:
                btn.clicked.connect(lambda _, s=symbol: self._set_shape(s, target=target))
            else:
                btn.clicked.connect(callback)
            return btn

        # Create left and right shape buttons
        shapes = ["*", "▭", "◯", "⬭", "✏"]
        shape_tooltips = {
            "*": "Point selection",
            "▭": "Rectangle selection",
            "◯": "Circle selection",
            "⬭": "Ellipse selection",
            "✏": "Freehand drawing"
        }
        self.shape_buttons = {
            "left": {shape: make_shape_button(shape, "left") for shape in shapes},
            "right": {shape: make_shape_button(shape, "right") for shape in shapes}
        }
        # Add tooltips to shape buttons
        for side in ["left", "right"]:
            for shape, tooltip in shape_tooltips.items():
                self.shape_buttons[side][shape].setToolTip(tooltip)

        # Colormap (color scale) buttons (non-shape, open dialog)
        self.color_btn_left = QPushButton("🌈")
        self.color_btn_left.setFixedSize(40, 40)
        self.color_btn_left.setToolTip("Change left slice color scale")
        self.color_btn_left.clicked.connect(lambda: self._open_colormap_dialog('left'))

        self.color_btn_right = QPushButton("🌈")
        self.color_btn_right.setFixedSize(36, 36)
        self.color_btn_right.setToolTip("Change right slice color scale")
        self.color_btn_right.clicked.connect(lambda: self._open_colormap_dialog('right'))

        # Copy colormap buttons
        self.copy_color_btn_l = QPushButton("⎘🌈")
        self.copy_color_btn_l.setFixedSize(40, 40)
        self.copy_color_btn_l.setToolTip("Copy right slice colormap to left slice")
        self.copy_color_btn_l.clicked.connect(lambda: self.copy_colormap('right','left'))

        self.copy_color_btn_r = QPushButton("⎘🌈")
        self.copy_color_btn_r.setFixedSize(40, 40)
        self.copy_color_btn_r.setToolTip("Copy left slice colormap to right slice")
        self.copy_color_btn_r.clicked.connect(lambda: self.copy_colormap('left','right'))

        # Scale selection dropdowns
        self.scale_combo_left = QComboBox()
        self.scale_combo_left.addItems(["Linear", "Log", "Sqrt", "Power (γ=2)"])
        self.scale_combo_left.setToolTip("Intensity scaling for left slice")
        self.scale_combo_left.currentTextChanged.connect(lambda: self._change_scale('left'))
        self.scale_combo_left.setMaximumWidth(120)

        self.scale_combo_right = QComboBox()
        self.scale_combo_right.addItems(["Linear", "Log", "Sqrt", "Power (γ=2)"])
        self.scale_combo_right.setToolTip("Intensity scaling for right slice")
        self.scale_combo_right.currentTextChanged.connect(lambda: self._change_scale('right'))
        self.scale_combo_right.setMaximumWidth(120)

        # Add copy buttons - each pulls the OTHER side's selection onto its own
        # side, matching the colormap-copy buttons' direction (copy_color_btn_l/r).
        self.copy_btn_l = make_shape_button("⎘", "left", checkable=False,
            callback=lambda: self.copy_selection(source="right", target="left"))
        self.copy_btn_l.setToolTip("Copy selection from right slice to left")
        self.copy_btn_r = make_shape_button("⎘", "right", checkable=False,
            callback=lambda: self.copy_selection(source="left", target="right"))
        self.copy_btn_r.setToolTip("Copy selection from left slice to right")

        # Edit Shape toggle buttons (enable drag/resize on a placed shape)
        self.edit_shape_btn_l = QPushButton("✎")
        self.edit_shape_btn_l.setCheckable(True)
        self.edit_shape_btn_l.setFixedSize(40, 40)
        self.edit_shape_btn_l.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.edit_shape_btn_l.setToolTip("Toggle Edit Shape (drag/resize)")
        self.edit_shape_btn_l.setEnabled(False)
        self.edit_shape_btn_l.clicked.connect(lambda: self._toggle_edit_shape("left"))

        self.edit_shape_btn_r = QPushButton("✎")
        self.edit_shape_btn_r.setCheckable(True)
        self.edit_shape_btn_r.setFixedSize(40, 40)
        self.edit_shape_btn_r.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.edit_shape_btn_r.setToolTip("Toggle Edit Shape (drag/resize)")
        self.edit_shape_btn_r.setEnabled(False)
        self.edit_shape_btn_r.clicked.connect(lambda: self._toggle_edit_shape("right"))

        # Erase (clear) selection buttons
        self.erase_btn_l = QPushButton("✖")
        self.erase_btn_l.setFixedSize(40, 40)
        self.erase_btn_l.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.erase_btn_l.setToolTip("Clear selection on left slice")
        self.erase_btn_l.clicked.connect(lambda: self.clear_region("left"))

        self.erase_btn_r = QPushButton("✖")
        self.erase_btn_r.setFixedSize(40, 40)
        self.erase_btn_r.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.erase_btn_r.setToolTip("Clear selection on right slice")
        self.erase_btn_r.clicked.connect(lambda: self.clear_region("right"))

    def _setup_main_layout(self):
        """Set up the main application layout"""
        main_layout = QVBoxLayout()

        # Add controls at top
        main_layout.addLayout(self.controls_layout)

        # Top area: left and right halves
        top_layout = QHBoxLayout()

        # LEFT HALF
        # Left stats column (will take 25% of left half)
        left_stats_layout = QVBoxLayout()
        left_stats_layout.addWidget(self.stats1_label)
        left_stats_layout.addStretch()

        # Left combined area (slice + sidebar) (will take 75% of left half)
        left_combined = QHBoxLayout()

        # Left main slice area (dominant part of combined)
        left_slice_layout = QVBoxLayout()
        left_slice_layout.addWidget(self.slice1_canvas)
        left_slice_layout.addWidget(self.slice1_toolbar)

        # Left sidebar (shape buttons / copy / erase)
        left_sidebar = QVBoxLayout()
        left_sidebar.addWidget(self.color_btn_left)
        left_sidebar.addWidget(QLabel("Scale:"))
        left_sidebar.addWidget(self.scale_combo_left)
        for shape in ["*", "▭", "◯", "⬭", "✏"]:
            left_sidebar.addWidget(self.shape_buttons["left"][shape])
        left_sidebar.addWidget(self.copy_btn_l)
        left_sidebar.addWidget(self.copy_color_btn_l)
        left_sidebar.addWidget(self.edit_shape_btn_l)
        left_sidebar.addWidget(self.erase_btn_l)
        left_sidebar.addStretch()

        left_combined.addLayout(left_slice_layout, 4)
        left_combined.addLayout(left_sidebar, 1)

        left_half = QHBoxLayout()
        left_half.addLayout(left_stats_layout, 1)
        left_half.addLayout(left_combined, 3)

        # RIGHT HALF (mirror)
        right_stats_layout = QVBoxLayout()
        right_stats_layout.addWidget(self.stats2_label)
        right_stats_layout.addStretch()

        right_combined = QHBoxLayout()

        right_sidebar = QVBoxLayout()
        right_sidebar.addWidget(self.color_btn_right)
        right_sidebar.addWidget(QLabel("Scale:"))
        right_sidebar.addWidget(self.scale_combo_right)
        for shape in ["*", "▭", "◯", "⬭", "✏"]:
            right_sidebar.addWidget(self.shape_buttons["right"][shape])
        right_sidebar.addWidget(self.copy_btn_r)
        right_sidebar.addWidget(self.copy_color_btn_r)
        right_sidebar.addWidget(self.edit_shape_btn_r)
        right_sidebar.addWidget(self.erase_btn_r)
        right_sidebar.addStretch()

        right_slice_layout = QVBoxLayout()
        right_slice_layout.addWidget(self.slice2_canvas)
        right_slice_layout.addWidget(self.slice2_toolbar)

        right_combined.addLayout(right_slice_layout, 4)
        right_combined.addLayout(right_sidebar, 1)

        right_half = QHBoxLayout()
        right_half.addLayout(right_stats_layout, 1)
        right_half.addLayout(right_combined, 3)

        top_layout.addLayout(left_half, 1)
        top_layout.addLayout(right_half, 1)

        main_layout.addLayout(top_layout, 1)

        # ===================================================================
        # COLLAPSIBLE BUTTON GROUPS (snake_case)
        # ===================================================================
        
        # Create collapsible button layout
        collapsible_layout = QHBoxLayout()
        collapsible_layout.setSpacing(0)
        collapsible_layout.setContentsMargins(0, 0, 0, 0)
        
        # Group 2: Gaussian Fit
        self.gaussian_group = CollapsibleButtonGroup("Gaussian Fit")
        self.gaussian_group.add_button(self.gauss_left_btn)
        self.gaussian_group.add_button(self.gauss_peak_btn)
        self.gaussian_group.add_button(self.gauss_right_btn)
        self.gaussian_group.add_button(self.gauss_noise_left_btn)
        self.gaussian_group.add_button(self.gauss_noise_right_btn)
        self.gaussian_group.add_button(self.gauss_fit_btn)
        self.gaussian_group.add_button(self.gauss_map_btn)
        collapsible_layout.addWidget(self.gaussian_group)
        
        # Group 3: Continuum Tools
        self.continuum_group = CollapsibleButtonGroup("Continuum Tools")
        self.continuum_group.add_button(self.auto_continuum_toggle)
        self.continuum_group.add_button(self.custom_continuum_toggle)
        collapsible_layout.addWidget(self.continuum_group)
        
        # Group 4: Rotational Diagram
        self.rotational_diagram_group = CollapsibleButtonGroup("Rotational Diagram")
        self.rotational_diagram_group.add_button(self.rot_diag_btn)
        collapsible_layout.addWidget(self.rotational_diagram_group)
        
        # Group 5: HITRAN (wavelength inputs + action buttons + CO flux map)
        self.hitran_group = CollapsibleButtonGroup("HITRAN")
        self.hitran_group.add_button(self.min_wave_input)
        self.hitran_group.add_button(self.max_wave_input)
        self.hitran_group.add_button(self.hitran_btn)
        self.hitran_group.add_button(self.load_custom_lines_btn)
        self.hitran_group.add_button(self.load_contam_btn)
        self.hitran_group.add_button(self.multi_gauss_fit_btn)
        self.hitran_group.add_button(self.show_species_btn)
        self.hitran_group.add_button(self.co_flux_map_btn)
        collapsible_layout.addWidget(self.hitran_group)
        
        collapsible_layout.addStretch()  # Push all groups to the left
        
        main_layout.addLayout(collapsible_layout)

        # ===================================================================

        # Spectrum control strip - three zones sharing one row: Slice
        # Selection (left), the spectrum tabs + view options (middle),
        # Shifting (right). All three operate through the spectrum itself
        # rather than the 2D slice images, so they live together here instead
        # of in the general collapsible-group toolbar above.
        spec_tab_layout = QHBoxLayout()
        spec_tab_layout.setContentsMargins(0, 0, 0, 0)

        spec_tab_layout.addWidget(self.set_left_btn)
        spec_tab_layout.addWidget(self.left_wave_input)
        spec_tab_layout.addWidget(self.set_right_btn)
        spec_tab_layout.addWidget(self.right_wave_input)

        spec_tab_layout.addStretch()

        spec_tab_layout.addWidget(QLabel("Spectrum:"))
        spec_tab_layout.addWidget(self.spec_tab_bar)
        spec_tab_layout.addWidget(self.normalize_checkbox)
        spec_tab_layout.addWidget(self.overplot_left_checkbox)
        spec_tab_layout.addWidget(self.overplot_right_checkbox)
        spec_tab_layout.addWidget(self.self_normalize_left_checkbox)
        spec_tab_layout.addWidget(self.self_normalize_right_checkbox)

        spec_tab_layout.addStretch()

        spec_tab_layout.addWidget(self.shift_btn)
        spec_tab_layout.addWidget(self.apply_shift_btn)
        spec_tab_layout.addWidget(self.cancel_shift_btn)
        spec_tab_layout.addWidget(self.revert_shift_btn)

        main_layout.addLayout(spec_tab_layout)

        # Add spectrum at bottom with expanded vertical space
        main_layout.addWidget(self.spec_canvas, 2)
        main_layout.addWidget(self.spec_toolbar)
        
        # Add save spectrum button separately
        self.continuum_action_layout = QHBoxLayout()
        self.continuum_action_layout.addStretch()
        self.continuum_action_layout.addWidget(self.continuum_line_btn)
        self.continuum_action_layout.addWidget(self.subtract_continuum_btn)
        self.continuum_action_layout.addWidget(self.custom_subtract_btn)
        self.continuum_action_layout.addWidget(self.custom_delete_btn)
        self.continuum_action_layout.addWidget(self.custom_save_pts_btn)
        self.continuum_action_layout.addWidget(self.custom_load_pts_btn)
        self.continuum_action_layout.addWidget(self.custom_ok_btn)
        self.continuum_action_layout.addWidget(self.save_results_btn)
        self.continuum_action_layout.addStretch()
        main_layout.addLayout(self.continuum_action_layout)

        self.setLayout(main_layout)

    def _cancel_custom_continuum(self):
        """Cancel custom continuum selection"""
        if self.custom_continuum_selector is not None:
            self.custom_continuum_selector.clear_all()
            self.custom_continuum_selector.deactivate()

        self.custom_subtract_btn.setChecked(False)
        self.custom_continuum_active = False

        QMessageBox.information(
            self,
            "Cancelled",
            "Custom continuum selection cancelled."
        )
        print("[DEBUG] Custom continuum cancelled")

    def _init_state(self):
        """Initialize state variables"""
        self.active_shape = None
        self.active_target = None
        self.last_idx_left = 0
        self.last_idx_right = 0
        self.last_selection_left = None
        self.last_selection_right = None
        # Region-selection interaction state per side: "idle" | "drawing" | "placed" | "editing"
        self.region_state = {"left": "idle", "right": "idle"}
        # Which spectrum tab is the active one shown as the main plot trace
        self.active_spectrum_tab = "full"
        self.hitran_lines = []  # Store HITRAN lines separately
        self.plotted_lines = []  # Store currently plotted spectral lines
        self.plotted_artists = []  # Store matplotlib artist objects for plotted lines (detection + contaminant)
        self.scale_left = 'Linear'
        self.scale_right = 'Linear'
        
        # Gaussian fitting state
        self.gauss_left_x = None
        self.gauss_peak_x = None
        self.gauss_right_x = None
        self.gauss_noise_left_x = None
        self.gauss_noise_right_x = None
        self.gauss_mode = None  # 'left', 'peak', 'right', 'noise_left', or 'noise_right'
        self.gauss_markers = []  # Store marker artists
        
        # Gaussian boundary marker lines (vertical lines)
        self.gauss_left_marker = None
        self.gauss_peak_marker = None
        self.gauss_right_marker = None
        self.gauss_noise_left_marker = None
        self.gauss_noise_right_marker = None
        
        # Gaussian fitting cache - avoid refitting if wavelength range hasn't changed
        self.gauss_cache = None  # Stores results from last fit
        self.gauss_cache_lam_min = None
        self.gauss_cache_lam_max = None
        self.gauss_cache_noise_min = None
        self.gauss_cache_noise_max = None
        self.gauss_cache_tab = None  # Which tab the cached fit/mask belongs to

        # Per-tab Gaussian boundary state, saved/restored by
        # _swap_gaussian_boundaries_for_tab whenever the active tab changes.
        self.gauss_boundaries = {"full": {}, "left": {}, "right": {}}
        
        # Continuum line state
        self.continuum_line = None  # Store continuum line artist
        self.continuum_fitted = None # Continuum subtraction state
        self.global_continuum_applied = False # Track if global continuum subtraction was applied
        self.custom_continuum_applied = False # Track if custom continuum was applied
        self.custom_continuum_selector = None
        self.custom_continuum_active = False
        self.subtract_mode = None  # 'left' or 'right'
        self.subtract_left_x = None
        self.subtract_right_x = None
        self.subtract_markers = []  # Store marker artists

        self.continuum_subtracted = False  # Track if continuum is currently subtracted
        self.original_spectrum_backup = None

        # Note: per-tab Doppler shift state (base_wavelength, spectrum_shift_factor,
        # spectrum_shift_preview) is initialized in _init_spectrum_tabs (GuiSpectrumMixin).

        # Keep references to spawned child windows to prevent garbage collection
        self.child_windows = []

        # Initialize region selectors
        self.region_selector_left = RegionSelector(
            self.slice1_plotter.ax,
            self.slice1_canvas,
            lambda mask, patch, coords: self._handle_region_select(mask, patch, coords, "left")
        )
        self.region_selector_right = RegionSelector(
            self.slice2_plotter.ax,
            self.slice2_canvas,
            lambda mask, patch, coords: self._handle_region_select(mask, patch, coords, "right")
        )

    def load_cube(self):
        """Open file dialog and load the selected FITS file"""
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open FITS", filter="FITS files (*.fits *.fz *.fit)")
        if not fname:
            return

        try:
            # Clear all previous plots and selections
            self._clear_all_displays()
            
            # Free any existing memory
            if hasattr(self, 'cube_data'):
                self.cube_data.free_memory()

            # Load the data
            success = self.cube_data.load_from_file(fname)
            if not success:
                QMessageBox.warning(self, "Load Error", "Failed to load the FITS file")
                return
                
            if self.cube_data.spectrum is None or len(self.cube_data.spectrum) == 0:
                QMessageBox.warning(self, "Data Error", "Failed to compute spectrum from the data")
                return

            # Copy spectrum
            self.original_spectrum = self.cube_data.spectrum.copy()

            # Permanent, never-mutated reference wavelength for the per-tab
            # shift system (each tab's displayed wavelength is derived from
            # this, never from a mutated-in-place shared array).
            self.base_wavelength = np.array(_extract_wavelength_value(self.cube_data.wavelength), dtype=float)

            # Update instrument label with filename
            filename = os.path.basename(self.cube_data.filename)
            self.instrument_label.setText(f"Instrument: {self.cube_data.instrument} [{filename}]")

            # Initialize the spectrum plotter with wavelength information
            self.spec_plotter = SpectrumPlotter(
                self.spec_fig,
                self.cube_data.wavelength,
                self.cube_data.wave_unit
            )
            self.spec_plotter.cube_data = self.cube_data  # Set cube_data for unit handling
            
            # Set custom home button action: zoom out AND clear Gaussian fit
            self.spec_toolbar.custom_home_action = self._spectrum_home_reset
            
            try:
                # Plot the spectrum
                self.spec_plotter.plot_spectrum(self.cube_data.spectrum)
                try:
                    self.spec_canvas.draw()
                except Exception:
                    pass
                # New cube: reset slice-indicator "explicitly set" state and
                # reset to the Full tab (a previously-active Left/Right tab
                # would otherwise be left pointing at the old cube's stale
                # selection). setCurrentIndex fires currentChanged, which
                # refreshes the display - unless it's already 0, in which case
                # plot_spectrum() above already shows the right thing.
                self._reset_slice_indicators()
                self.spec_tab_bar.setCurrentIndex(0)
            except Exception as plot_error:
                QMessageBox.warning(self, "Plot Error", f"Error plotting spectrum: {str(plot_error)}")
            
            try:
                # Display initial slices
                self._display_slice(0, "left")
                self._display_slice(0, "right")
            except Exception as slice_error:
                QMessageBox.warning(self, "Display Error", f"Error displaying slices: {str(slice_error)}")

            # Wire up left/right wavelength spin boxes to this cube's grid
            wavelength = _extract_wavelength_value(self.cube_data.wavelength)
            if wavelength is not None:
                wl = np.asarray(wavelength, dtype=float)
                for spin in (self.left_wave_input, self.right_wave_input):
                    spin.set_wavelength_grid(wl)
                    spin.setRange(float(np.min(wl)), float(np.max(wl)))
                    spin.setSuffix(f" {self.cube_data.wave_unit}")
                    spin.blockSignals(True)
                    spin.setValue(float(wl[0]))
                    spin.blockSignals(False)
                    spin.setEnabled(True)

        except MemoryError:
            QMessageBox.critical(self, "Memory Error", 
                "Not enough memory to load this file. Try closing other applications or using a smaller file.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
            print(f"Detailed error: {e}")  # For debugging

    def _open_new_window(self):
        """Open a new independent CubeExplorer window."""
        try:
            new_win = CubeExplorer()
            new_win.resize(self.width(), self.height())
            new_win.show()
            # Keep a reference so it isn't garbage collected
            self.child_windows.append(new_win)
        except Exception as e:
            QMessageBox.warning(self, "New Window Error", f"Could not open a new window:\n{e}")


    def show_header(self):
        """Display FITS headers with HDU selection dialog"""
        if not hasattr(self.cube_data, 'hdul') or self.cube_data.hdul is None:
            QMessageBox.warning(self, "No file loaded", "Please load a FITS file first.")
            return

        # Create header selection dialog
        hdul = self.cube_data.hdul
        hdu_names = []
        
        # Build list of HDU names
        for i, hdu in enumerate(hdul):
            hdu_name = getattr(hdu, 'name', f"HDU {i}")
            hdu_type = type(hdu).__name__
            if hdu_name.strip():
                hdu_names.append(f"HDU {i}: {hdu_name} ({hdu_type})")
            else:
                hdu_names.append(f"HDU {i}: ({hdu_type})")
        
        # Create selection dialog
        self.header_selection_dialog = QDialog(self)
        self.header_selection_dialog.setWindowTitle("Select FITS Header to View")
        self.header_selection_dialog.setMinimumSize(400, 300)
        
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Available HDUs. Click to select:"))
        
        # Create list widget for HDU selection
        list_widget = QListWidget()
        for i, name in enumerate(hdu_names):
            item = QListWidgetItem(name)
            list_widget.addItem(item)
        
        # Pre-select the first HDU
        list_widget.setCurrentRow(0)
        layout.addWidget(list_widget)
        
        # Add buttons
        button_layout = QHBoxLayout()
        view_btn = QPushButton("View Header")
        close_btn = QPushButton("Close")
        
        def view_selected_header():
            selected_idx = list_widget.currentRow()
            if selected_idx >= 0:
                self._display_header(selected_idx, hdul, hdu_names[selected_idx])
                
        view_btn.clicked.connect(view_selected_header)
        close_btn.clicked.connect(self.header_selection_dialog.close)
        
        button_layout.addWidget(view_btn)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        self.header_selection_dialog.setLayout(layout)
        self.header_selection_dialog.show()
    
    def _display_header(self, hdu_index, hdul, hdu_name):
        """Display the header for a specific HDU"""
        try:
            header = hdul[hdu_index].header
            header_text = "\n".join(
                f"{k:8} = {v!r} / {header.comments[k]}" if k in header.comments else f"{k:8} = {v!r}" 
                for k, v in header.items()
            )
            
            # Keep reference to prevent garbage collection
            self.header_window = QWidget()
            self.header_window.setWindowTitle(f"FITS Header - {hdu_name}")
            self.header_window.setMinimumSize(700, 500)
            
            layout = QVBoxLayout()
            layout.addWidget(QLabel(f"<b>{hdu_name}</b>"))
            
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(text_edit.font())  # Use monospace for better formatting
            text_edit.setText(header_text)
            layout.addWidget(text_edit)
            
            self.header_window.setLayout(layout)
            self.header_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display header: {str(e)}")

    def _display_slice(self, idx, target):
        """Display a wavelength/velocity slice with proper title and intensity units"""
        image = self.cube_data.get_slice(idx)
        
        # Select plotter and stats label
        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
        stats_label = self.stats1_label if target == "left" else self.stats2_label

        # Title with spectral value
        if hasattr(self.cube_data, 'wavelength') and self.cube_data.wavelength is not None:
            # Extract wavelength value, handling Quantity objects
            wave = _extract_wavelength_value(self.cube_data.wavelength, idx)
            if self.cube_data.wave_unit == "Å":
                title = f"{target.capitalize()} Slice (idx={idx}) @ {wave:.6f} Å"
            elif self.cube_data.wave_unit == "µm":
                title = f"{target.capitalize()} Slice (idx={idx}) @ {wave:.6f} µm"
            elif self.cube_data.wave_unit == "km/s":
                title = f"{target.capitalize()} Slice (idx={idx}) @ {wave:.1f} km/s"
            else:
                title = f"{target.capitalize()} Slice @ idx={idx}"
        else:
            title = f"{target.capitalize()} Slice @ idx={idx}"
        
        # Update plot using stored colormap preference per slice if available
        cmap = getattr(self, 'slice1_cmap' if target == 'left' else 'slice2_cmap', None)
        scale = self.scale_left if target == 'left' else self.scale_right
        # Get the flux unit for colorbar display
        flux_unit_raw = getattr(self.cube_data, "flux_unit", "")
        flux_unit_clean = clean_flux_unit(flux_unit_raw) if flux_unit_raw else ""
        plotter.plot_slice(image, title, cmap=cmap, scale=scale, flux_unit=flux_unit_clean)
        
        # Intensity stats with units
        mn, mx, mean = np.nanmin(image), np.nanmax(image), np.nanmean(image)
        unit_raw = getattr(self.cube_data, "flux_unit", "arbitrary units")  # fallback if flux_unit missing
        unit = clean_flux_unit(unit_raw)
        stats_label.setText(
            f"Intensity\n[{unit}]:\n"
            f"min={mn:.3e}\n"
            f"max={mx:.3e}\n"
            f"mean={mean:.3e}\n"
        )

        # Force redraw
        if target == "left":
            self.slice1_canvas.draw()
            self.last_idx_left = idx
        else:
            self.slice2_canvas.draw()
            self.last_idx_right = idx

    def _set_region_state(self, side, new_state):
        """Central state transition for region-selection interaction on one side.

        Single source of truth for shape-button enablement, the Edit Shape
        button's enabled/checked state, and InteractivePatch handler wiring -
        so these can never drift out of sync with each other (which is what
        previously let stale drag handlers survive an erase, and let editing
        be always-on with no explicit commit step).
        """
        selector = self.region_selector_left if side == "left" else self.region_selector_right
        edit_btn = self.edit_shape_btn_l if side == "left" else self.edit_shape_btn_r
        old_state = self.region_state.get(side, "idle")

        # Leaving "editing" always unwires the drag handlers first, regardless
        # of which state we're moving to (placed or idle).
        if old_state == "editing" and new_state != "editing" and selector.interactive_patch is not None:
            selector.interactive_patch.disconnect_handlers()

        self.region_state[side] = new_state

        if new_state == "idle":
            for btn in self.shape_buttons[side].values():
                btn.setEnabled(True)
                btn.setChecked(False)
            edit_btn.setEnabled(False)
            edit_btn.setChecked(False)
        elif new_state == "drawing":
            active_shape = self.active_shape if self.active_target == side else None
            for shape, btn in self.shape_buttons[side].items():
                btn.setEnabled(shape == active_shape)
            edit_btn.setEnabled(False)
            edit_btn.setChecked(False)
        elif new_state == "placed":
            for btn in self.shape_buttons[side].values():
                btn.setEnabled(False)
                btn.setChecked(False)
            sel = self.last_selection_left if side == "left" else self.last_selection_right
            edit_btn.setEnabled(bool(sel) and sel.get("shape") != "*")
            edit_btn.setChecked(False)
        elif new_state == "editing":
            for btn in self.shape_buttons[side].values():
                btn.setEnabled(False)
            edit_btn.setEnabled(True)
            edit_btn.setChecked(True)
            if selector.interactive_patch is not None:
                selector.interactive_patch.connect()

    def _toggle_edit_shape(self, side):
        """Handle the Edit Shape button click for one side."""
        state = self.region_state.get(side, "idle")
        if state == "placed":
            self._set_region_state(side, "editing")
        elif state == "editing":
            self._set_region_state(side, "placed")
        else:
            # Button should be disabled outside placed/editing; guard defensively.
            edit_btn = self.edit_shape_btn_l if side == "left" else self.edit_shape_btn_r
            edit_btn.setChecked(False)

    def _set_shape(self, shape, target):
        """Handle shape button activation"""
        # Update active shape and target
        if self.active_shape == shape and self.active_target == target:
            self.active_shape = None
            self.active_target = None
        else:
            self.active_shape = shape
            self.active_target = target

        # Update button states
        for side in ["left", "right"]:
            for btn_shape, btn in self.shape_buttons[side].items():
                btn.setChecked(
                    side == target and btn_shape == shape and self.active_shape is not None
                )

        # Activate appropriate selector
        selector = (
            self.region_selector_left if target == "left" else self.region_selector_right
        )
        if self.active_shape:
            selector.activate_selector(shape)
            self._set_region_state(target, "drawing")
        else:
            selector.activate_selector(None)
            self._set_region_state(target, "idle")

    def _handle_region_select(self, mask, patch, coords, target):
        """Handle region selection completion"""
        # Store selection (coords are xmin,xmax,ymin,ymax)
        xmin, xmax, ymin, ymax = coords

        # Determine which slice index and image to use
        idx = self.last_idx_left if target == "left" else self.last_idx_right
        image = self.cube_data.get_slice(idx)
        if image is None:
            return

        selector = self.region_selector_left if target == "left" else self.region_selector_right

        # This callback also fires on every mouse-move tick of a live
        # drag/resize while "editing" (not just once when a selection first
        # completes) - in that case we must NOT tear down the handles or
        # touch region_state, since the drag is still in progress.
        is_live_edit_tick = self.region_state.get(target) == "editing"

        # Build full-size mask matching the image shape
        if mask.shape == image.shape:
            full_mask = mask
        else:
            full_mask = np.zeros_like(image, dtype=bool)
            h_mask, w_mask = mask.shape
            y0 = ymin
            x0 = xmin
            y1 = y0 + h_mask
            x1 = x0 + w_mask

            y0, y1 = int(y0), int(y1)
            x0, x1 = int(x0), int(x1)
            
            # Clip to image bounds
            y1 = min(y1, image.shape[0])
            x1 = min(x1, image.shape[1])
            full_mask[y0:y1, x0:x1] = mask[:(y1-y0), :(x1-x0)]

        # Determine the shape identity. Normally read from the selector (not
        # self.active_shape) since copy_selection sets it there directly. But
        # during a live edit-tick, selector.active_shape is already stale -
        # activate_selector(None) at the end of the ORIGINAL completed draw
        # already reset it to None, so re-reading it here on every drag tick
        # would silently overwrite the selection's shape (e.g. a moved
        # freeform losing its "✏" identity and degrading to a rectangle).
        # A live edit is always modifying an already-placed selection, so its
        # shape can never change mid-edit - reuse the existing one.
        if is_live_edit_tick:
            existing_sel = self.last_selection_left if target == "left" else self.last_selection_right
            shape = existing_sel["shape"] if existing_sel else selector.active_shape
        else:
            shape = selector.active_shape

        # Save selection with full mask and label.
        selection = {
            "shape": shape,
            "coords": coords,
            "mask": full_mask,
            "label": f'{target.capitalize()} Selection',
            "patch": patch  # Store patch for lasso recreation
        }
        if target == "left":
            self.last_selection_left = selection
        else:
            self.last_selection_right = selection

        # Update plot: add patch and record it
        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
        # When creating a region, remove previous pixel marker(s) so selection is exclusive
        if hasattr(plotter.ax, 'pixel_marker'):
            for artist in getattr(plotter.ax, 'pixel_marker'):
                try:
                    artist.remove()
                except Exception:
                    pass
            plotter.ax.pixel_marker = []

        # Remove previous region patches so only last selection remains
        if hasattr(plotter.ax, 'area_artists'):
            for artist in getattr(plotter.ax, 'area_artists'):
                try:
                    artist.remove()
                except Exception:
                    pass
        plotter.ax.area_artists = []

        # Remove all handle scatter collections so they don't linger - but not
        # during a live edit-tick, where the only handle collection that
        # exists IS the one belonging to the shape currently being dragged.
        if not is_live_edit_tick:
            for collection in list(plotter.ax.collections):
                try:
                    if hasattr(collection, '_is_handle_collection') and collection._is_handle_collection:
                        collection.remove()
                        print(f"[DEBUG] Removed lingering handle collection")
                except Exception as e:
                    print(f"[DEBUG] Error removing lingering handle collection: {e}")

        # Add the new patch and record it
        plotter.ax.add_patch(patch)
        plotter.ax.area_artists.append(patch)
        
        # Hide the selector handles after selection is complete
        if selector.selector is not None:
            try:
                # Set selector to non-interactive to hide handles
                if hasattr(selector.selector, 'set_active'):
                    selector.selector.set_active(False)
                # Alternatively, make it invisible
                if hasattr(selector.selector, 'set_visible'):
                    selector.selector.set_visible(False)
            except Exception as e:
                print(f"[DEBUG] Could not hide selector handles: {e}")
        
        if target == "left":
            self.slice1_canvas.draw()
        else:
            self.slice2_canvas.draw()
        
        # Deselect the shape button and deactivate the selector
        self.active_shape = None
        self.active_target = None
        for side in ["left", "right"]:
            for btn in self.shape_buttons[side].values():
                btn.setChecked(False)
        selector.activate_selector(None)
        if not is_live_edit_tick:
            self._set_region_state(target, "placed")

        # Compute stats for the selected area and update label
        selected_area = image[full_mask]
        if selected_area.size > 0:
            mn = np.nanmin(selected_area)
            mx = np.nanmax(selected_area)
            mean = np.nanmean(selected_area)
            total_sum = np.nansum(selected_area) # Add sum
            std = np.nanstd(selected_area) # Add standard deviation
            
            stats_label = self.stats1_label if target == "left" else self.stats2_label
            current_text = stats_label.text().split("\n")
            
            # Keep the slice stats if they exist (they start with "Intensity:")
            slice_stats = [line for line in current_text if line.startswith("Intensity:")]
            
            # Get cleaned flux unit
            unit_raw = getattr(self.cube_data, "flux_unit", "")
            unit = clean_flux_unit(unit_raw) if unit_raw else ""

            # Create region stats with sum and std. The unit goes on its own
            # line (matching the slice-stats label above) so it is not clipped
            # by the narrow stats panel.
            region_stats = [
                "Intensity (Area Selected)" if unit else "Intensity (Area Selected):",
            ]
            if unit:
                region_stats.append(f"[{unit}]:")
            region_stats += [
                f"min={mn:.3e}",
                f"max={mx:.3e}",
                f"mean={mean:.3e}",
                f"sum={total_sum:.3e}",
                f"std={std:.3e}"
            ]
            
            # Combine stats (slice stats first, then region stats)
            combined_stats = slice_stats + [""] + region_stats if slice_stats else region_stats
            stats_label.setText("\n".join(combined_stats))

        # Diagnostic: print mask summary to help debug zero spectra
        try:
            n_pixels = int(np.count_nonzero(full_mask))
        except Exception:
            n_pixels = None
        print(f"REGION DEBUG [{target}] mask pixels={n_pixels}, mask_shape={full_mask.shape}")
        print(f"REGION DEBUG [left] type={type(self.last_selection_left)}, shape={getattr(self.last_selection_left, 'shape', None)}")

        # Note: the region spectrum is no longer plotted as an always-visible
        # overlay here - it's shown only when this side's spectrum tab is
        # active, via the _refresh_active_spectrum_tab() call below.

        # Ensure the slice canvas is redrawn to show handles (use draw() for live updates)
        try:
            if target == "left":
                self.slice1_canvas.draw()
            else:
                self.slice2_canvas.draw()
        except Exception as e:
            print(f"[DEBUG] Error updating slice canvas: {e}")

        # Keep the spectrum tab live if this side's tab is the active one
        if self.active_spectrum_tab == target:
            self._refresh_active_spectrum_tab()

    def copy_selection(self, source, target):
        """Copy region selection from one side to the other"""
        source_selection = (
            self.last_selection_right if source == "right" else self.last_selection_left
        )
        if not source_selection:
            return

        if source_selection["shape"] == "*":
            # Handle pixel selection
            x, y = source_selection["coords"]
            self._handle_pixel_click(x, y, target)
        elif source_selection["shape"] == "✏":
            # Handle lasso selection - need to recreate from vertices
            if "patch" in source_selection and hasattr(source_selection["patch"], "lasso_verts"):
                verts = source_selection["patch"].lasso_verts
                selector = (
                    self.region_selector_left if target == "left" else self.region_selector_right
                )
                selector.active_shape = "✏"
                # Call on_lasso_select directly with the stored vertices
                selector.on_lasso_select(verts)
            else:
                # Fallback: try to create from bounding box if vertices not available
                selector = (
                    self.region_selector_left if target == "left" else self.region_selector_right
                )
                selector.active_shape = source_selection["shape"]
                try:
                    # Convert coords to integers for safety
                    coords = tuple(int(c) for c in source_selection["coords"])
                    selector.create_region(coords)
                except Exception as e:
                    print(f"[DEBUG] Error copying lasso: {e}")
        else:
            # Handle regular region selection (rectangle, circle, ellipse)
            selector = (
                self.region_selector_left if target == "left" else self.region_selector_right
            )
            selector.active_shape = source_selection["shape"]
            # Convert coords to integers for safety
            coords = tuple(int(c) for c in source_selection["coords"])
            selector.create_region(coords)

    def clear_region(self, target):
        """Clear the current region selection and remove overlays for a side"""
        state = self.region_state.get(target, "idle")
        selector = self.region_selector_left if target == "left" else self.region_selector_right

        if state == "drawing":
            # Abort the in-progress draw - nothing has been placed yet.
            self.active_shape = None
            self.active_target = None
            selector.activate_selector(None)
            for btn in self.shape_buttons[target].values():
                btn.setChecked(False)
            self._set_region_state(target, "idle")
            return

        if target == "left":
            sel = self.last_selection_left
            plotter = self.slice1_plotter
        else:
            sel = self.last_selection_right
            plotter = self.slice2_plotter

        # Disconnect and drop the interactive patch so its drag handlers don't
        # survive the erase - previously they stayed wired to the canvas and
        # could resurrect/interfere with a later draw in the same spot.
        if selector.interactive_patch is not None:
            selector.interactive_patch.disconnect()
            selector.interactive_patch = None

        if not sel:
            self._set_region_state(target, "idle")
            return

        # Remove pixel marker (point/"*" selections don't use area_artists)
        if hasattr(plotter.ax, 'pixel_marker'):
            for marker in plotter.ax.pixel_marker:
                try:
                    marker.remove()
                except Exception:
                    pass
            plotter.ax.pixel_marker = []

        # Remove patches from the slice axes
        if hasattr(plotter.ax, 'area_artists'):
            for artist in plotter.ax.area_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            plotter.ax.area_artists = []

        # Remove all handle scatter collections associated with patches
        for collection in list(plotter.ax.collections):
            try:
                if hasattr(collection, '_is_handle_collection') and collection._is_handle_collection:
                    collection.remove()
                    print(f"[DEBUG] Removed handle scatter collection")
            except Exception as e:
                print(f"[DEBUG] Error removing handle collection: {e}")

        # Redraw slice and spectrum
        if target == "left":
            self.slice1_canvas.draw()
            self.last_selection_left = None
            self.stats1_label.setText("Slice 1 Stats:\n-")
        else:
            self.slice2_canvas.draw()
            self.last_selection_right = None
            self.stats2_label.setText("Slice 2 Stats:\n-")

        try:
            self.spec_canvas.draw()
        except Exception:
            pass

        self._set_region_state(target, "idle")

        # This side's tab (if active) now has no selection - _refresh_active_spectrum_tab
        # will correctly show it as empty rather than falling back to Full.
        if self.active_spectrum_tab == target:
            self._refresh_active_spectrum_tab()

    def _handle_pixel_click(self, x, y, target):
        """Handle pixel selection"""
        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
        stats_label = self.stats1_label if target == "left" else self.stats2_label
        idx = self.last_idx_left if target == "left" else self.last_idx_right
        
        try:
            # Get pixel intensity
            image = self.cube_data.get_slice(idx)
            if image is None:
                return
                
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                intensity = float(image[y, x])  # Ensure it's a regular float
                
                # Store selection for later use
                label = f'{target.capitalize()} Selection'
                if target == "left":
                    self.last_selection_left = {"shape": "*", "coords": (x, y), "mask": None, "label": label}
                else:
                    self.last_selection_right = {"shape": "*", "coords": (x, y), "mask": None, "label": label}

                # Get coordinates
                ra, dec = self.cube_data.get_coords(x, y)

                # Update stats label with appropriate formatting
                stats_text = [
                    f"Pixel: (x={x}, y={y})",
                    f"Intensity={intensity:.3e}"
                ]
                
                if ra is not None and dec is not None:
                    stats_text.extend([
                        f"RA={ra:.6f}°",
                        f"DEC={dec:.6f}°"
                    ])
                    
                if self.cube_data.wavelength is not None:
                    # Extract wavelength value, handling Quantity objects
                    wave = _extract_wavelength_value(self.cube_data.wavelength, idx)
                    if self.cube_data.wave_unit == "Å":
                        stats_text.append(f"λ={wave:.6f} Å")
                    elif self.cube_data.wave_unit == "µm":
                        stats_text.append(f"λ={wave:.6f} µm")
                    elif self.cube_data.wave_unit == "km/s":
                        stats_text.append(f"v={wave:.1f} km/s")
                    else:
                        stats_text.append(f"λ={wave:.6f}")

                stats_label.setText("\n".join(stats_text))

                # When clicking a pixel remove any existing region patches (exclusive selection)
                selector = self.region_selector_left if target == "left" else self.region_selector_right
                if selector.interactive_patch is not None:
                    selector.interactive_patch.disconnect()
                    selector.interactive_patch = None

                if hasattr(plotter.ax, 'area_artists'):
                    for artist in getattr(plotter.ax, 'area_artists'):
                        try:
                            artist.remove()
                        except Exception:
                            pass
                    plotter.ax.area_artists = []

                # Remove all handle scatter collections
                for collection in list(plotter.ax.collections):
                    try:
                        if hasattr(collection, '_is_handle_collection') and collection._is_handle_collection:
                            collection.remove()
                    except Exception:
                        pass

                # Clear existing pixel markers
                if hasattr(plotter.ax, 'pixel_marker'):
                    for marker in plotter.ax.pixel_marker:
                        marker.remove()
                    plotter.ax.pixel_marker = []

                # Update marker
                plotter.update_pixel_marker(x, y)
                
                # Force immediate redraw
                if target == "left":
                    self.slice1_canvas.draw()
                else:
                    self.slice2_canvas.draw()
                
                self.active_shape = None
                self.active_target = None
                self._set_region_state(target, "placed")

                # Keep the spectrum tab live if this side's tab is the active one
                if self.active_spectrum_tab == target:
                    self._refresh_active_spectrum_tab()

        except Exception as e:
            print(f"Error handling pixel click: {e}")

    def _on_slice_click(self, event, target):
        """Handle clicks on slice display"""
        if event.inaxes is None:
            return

        if self.active_shape != "*":
            return

        ax = self.slice1_plotter.ax if target == "left" else self.slice2_plotter.ax
        if event.inaxes != ax:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        self._handle_pixel_click(x, y, target)

    def _on_spec_click(self, event):
        """Handle clicks on the spectrum display"""
        if event.inaxes != self.spec_plotter.ax:
            return

        # LEFT CLICK: Check if clicking near a plotted spectral line to deselect it
        if event.button == 1 and hasattr(self, 'plotted_lines') and self.plotted_lines:
            # Click tolerance in SCREEN pixels (not wavelength units) so it stays
            # small and consistent regardless of x-axis zoom level - a fixed
            # wavelength tolerance would span a huge fraction of the visible
            # plot when zoomed in, making it very easy to graze and deselect a
            # line while just trying to start a zoom-drag.
            CLICK_HIT_RADIUS_PX = 5
            mouse_x = event.xdata
            if mouse_x is not None and event.x is not None:
                # Find nearest line within threshold
                nearest_row = None
                min_dist = float('inf')
                ax = self.spec_plotter.ax

                for row in self.plotted_lines:
                    from .species_dialog import SpeciesDialog
                    λ = SpeciesDialog.get_row_wavelength(row)
                    λ_disp_x = ax.transData.transform((λ, 0))[0]
                    dist = abs(λ_disp_x - event.x)
                    if dist < CLICK_HIT_RADIUS_PX and dist < min_dist:
                        min_dist = dist
                        nearest_row = row
                
                if nearest_row:
                    # Found a line to deselect
                    line_wave = SpeciesDialog.get_row_wavelength(nearest_row)
                    species = nearest_row.get('species', 'Unknown')
                    print(f"[OK] Deselected: {species} at {line_wave:.6f} um")
                    
                    # Remove from plotted lines by wavelength comparison with tolerance
                    self.plotted_lines = [
                        r for r in self.plotted_lines 
                        if abs(SpeciesDialog.get_row_wavelength(r) - line_wave) > 1e-6
                    ]
                    
                    # Track excluded wavelength for CO flux
                    if not hasattr(self, 'co_flux_excluded_wavelengths'):
                        self.co_flux_excluded_wavelengths = set()
                    self.co_flux_excluded_wavelengths.add(line_wave)
                    
                    # Replot spectrum without this line
                    self.plot_selected_lines(self.plotted_lines)
                    
                    # Remove its Gaussian fit if present
                    if hasattr(self, 'multi_gauss_results'):
                        self.multi_gauss_results = [
                            fit_info for fit_info in self.multi_gauss_results
                            if abs(SpeciesDialog.get_row_wavelength(fit_info['line']) - line_wave) > 1e-6
                        ]
                        # Replot remaining Gaussian fits
                        if self.multi_gauss_results:
                            self._plot_multi_gaussian_on_spectrum(self.multi_gauss_results)
                    
                    return

        if hasattr(self, 'subtract_mode') and self.subtract_mode == 'selecting' and event.xdata:
            self._mark_subtraction_boundary(event.xdata)
            return
        
        # Handle Gaussian boundary selection first (takes priority)
        if self.gauss_mode in ['left', 'peak', 'right', 'noise_left', 'noise_right'] and event.xdata is not None:
            self._mark_gaussian_boundary(event.xdata)
            return
            
        if not self.active_target:
            return
            
        if not hasattr(self, 'cube_data') or not hasattr(self.cube_data, 'wavelength'):
            return

        click_x = event.xdata
        if click_x is None:
            return

        # Find nearest wavelength index
        wavelength = _extract_wavelength_value(self.cube_data.wavelength)
        if wavelength is None:
            idx = int(click_x)
        else:
            idx = (np.abs(wavelength - click_x)).argmin()

        self._set_slice_index(idx, self.active_target)

    def _set_slice_index(self, idx, target):
        """Set the left/right slice to wavelength index `idx` and refresh
        every synced UI element - the spectrum indicator line, the slice
        image, the wavelength spin box, and any active region selection.

        This is the single place clicking the spectrum, clicking a slice
        pixel, and typing into left/right_wave_input all funnel through, so
        they stay in sync with each other no matter which one is used.
        """
        if target == "left":
            self.last_idx_left = idx
        else:
            self.last_idx_right = idx

        # Update the vertical line in the spectrum plot
        self.spec_plotter.update_slice_indicator(idx, target)
        self._mark_slice_indicator_set(target)
        try:
            self.spec_canvas.draw()
        except Exception:
            pass

        # Display the new slice
        self._display_slice(idx, target)

        # Snap the wavelength spin box to this slice's exact wavelength
        wavelength = _extract_wavelength_value(self.cube_data.wavelength)
        if wavelength is not None:
            spin = self.left_wave_input if target == "left" else self.right_wave_input
            spin.blockSignals(True)
            spin.setValue(float(wavelength[idx]))
            spin.blockSignals(False)

        # Update any active region selections
        if target == "left" and self.last_selection_left:
            if self.last_selection_left["shape"] == "*":
                x, y = self.last_selection_left["coords"]
                self._handle_pixel_click(x, y, "left")
            else:
                xmin, xmax, ymin, ymax = self.last_selection_left["coords"]
                mask = self.last_selection_left["mask"]
                self._recreate_region(xmin, xmax, ymin, ymax, mask, "left")

        elif target == "right" and self.last_selection_right:
            if self.last_selection_right["shape"] == "*":
                x, y = self.last_selection_right["coords"]
                self._handle_pixel_click(x, y, "right")
            else:
                xmin, xmax, ymin, ymax = self.last_selection_right["coords"]
                mask = self.last_selection_right["mask"]
                self._recreate_region(xmin, xmax, ymin, ymax, mask, "right")

    def _on_wave_input_changed(self, target):
        """Handle the user typing/stepping a value in left/right_wave_input -
        jump that slice to the closest wavelength actually in the cube."""
        if not hasattr(self, 'cube_data') or not hasattr(self.cube_data, 'wavelength'):
            return
        wavelength = _extract_wavelength_value(self.cube_data.wavelength)
        if wavelength is None:
            return
        spin = self.left_wave_input if target == "left" else self.right_wave_input
        idx = int(np.abs(wavelength - spin.value()).argmin())
        self._set_slice_index(idx, target)

    def _select_target(self, target):
        """Handle slice selection mode"""
        if target == "left":
            self.set_right_btn.setChecked(False)
            self.active_target = "left" if self.set_left_btn.isChecked() else None
        else:
            self.set_left_btn.setChecked(False)
            self.active_target = "right" if self.set_right_btn.isChecked() else None

    def _recreate_region(self, xmin, xmax, ymin, ymax, mask, target):
        """Recreate a region selection's visual patch after changing slices.

        Delegates to the same RegionSelector.create_region/on_lasso_select
        pipeline used for an organic draw or a copy_selection() (rather than
        hand-building a static patch), so the shape type is always correct
        (read from the stored selection, not selector.active_shape - which is
        always None here since activate_selector(None) resets it right after
        any selection completes), lasso is handled, and the result gets a
        fresh (unconnected) InteractivePatch instead of one pointing at a
        stale, already-removed artist.
        """
        sel = self.last_selection_left if target == "left" else self.last_selection_right
        if not sel:
            return
        shape = sel.get("shape")

        selector = self.region_selector_left if target == "left" else self.region_selector_right

        # Drop any existing interactive patch before rebuilding - its patch
        # reference is about to become stale the moment a new artist is made.
        if selector.interactive_patch is not None:
            selector.interactive_patch.disconnect()
            selector.interactive_patch = None

        selector.active_shape = shape
        if shape == "✏":
            verts = getattr(sel.get("patch"), "lasso_verts", None)
            if verts is not None:
                selector.on_lasso_select(verts)
            else:
                # No stored vertices to rebuild from - fall back to the
                # selection's bounding box as a rectangle.
                selector.active_shape = "▭"
                selector.create_region((xmin, xmax, ymin, ymax))
        else:
            selector.create_region((xmin, xmax, ymin, ymax))
        selector.active_shape = None

        # create_region/on_lasso_select call back into _handle_region_select,
        # which redraws the slice/spectrum, updates last_selection_*, and
        # transitions region_state to "placed".



    def apply_doppler_shift(self):
        """Preview a multiplicative Doppler shift for whichever spectrum tab
        is currently active (Full, Left, or Right) - never the others.

        Uses formula: lambda_0 = lambda / (1 + v/c), composed on top of that
        tab's already-applied shift factor (if any), so repeated previews
        refine incrementally rather than always starting from scratch.
        """
        if not hasattr(self, 'original_spectrum') or self.original_spectrum is None:
            QMessageBox.warning(self, "No Spectrum", "Load a data cube first.")
            return
        if getattr(self, 'base_wavelength', None) is None:
            QMessageBox.warning(self, "No Wavelength Data",
                              "Wavelength data is not available for this cube.")
            return

        active_tab = self.active_spectrum_tab
        tab_label = active_tab.capitalize()

        if active_tab in ("left", "right"):
            sel = self.last_selection_left if active_tab == "left" else self.last_selection_right
            if not sel:
                QMessageBox.warning(self, "No Selection",
                                  f"Select a region on the {tab_label} side first.")
                return

        # Ask user for velocity in km/s
        velocity, ok = QInputDialog.getDouble(
            self,
            f"Doppler Shift (Multiplicative) - {tab_label}",
            "Enter systemic velocity (km/s):",
            value=0.0,
            decimals=2
        )
        if not ok:
            return

        c = 299792.458  # Speed of light in km/s
        current_factor = self.spectrum_shift_factor.get(active_tab, 1.0)
        new_factor = current_factor * (1 / (1 + velocity / c))

        self.spectrum_shift_preview[active_tab] = {"velocity": velocity, "factor": new_factor}

        # _refresh_active_spectrum_tab draws the orange preview overplotted
        # against this tab's own trace/color, applies Normalize consistently,
        # and enables the Apply Shift / Cancel Preview buttons.
        self._refresh_active_spectrum_tab()

        print(f"[DEBUG] Previewing Doppler shift on '{active_tab}': v={velocity} km/s (cumulative factor={new_factor:.6f})")

    def _apply_shift(self):
        """Commit whichever tab's pending shift preview is active.

        Each tab's shift is independent - applying Left's preview never
        touches Full or Right. The trace stays in its own original color
        (orange is a preview-only cue, not a permanent state). If the active
        tab is Full, cube_data.wavelength is also kept in sync, since
        Gaussian fitting/continuum/HITRAN/CO-flux all read it directly and
        haven't been migrated to ask "which tab" yet (see plan).
        """
        active_tab = self.active_spectrum_tab
        preview = self.spectrum_shift_preview.get(active_tab)
        if not preview:
            QMessageBox.warning(
                self,
                "No Shift Applied",
                "Please preview a Doppler shift first using the 'Shift' button."
            )
            return

        tab_label = active_tab.capitalize()
        reply = QMessageBox.question(
            self,
            "Apply Shift",
            f"Apply this Doppler shift to the {tab_label} spectrum (v={preview['velocity']:.1f} km/s)?\n\n"
            "This will be permanent until reverted.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Incremental factor from THIS specific shift (needed for
        # shift_plot_elements, which moves Gaussian fit artists by a
        # relative amount, not an absolute position).
        old_factor = self.spectrum_shift_factor.get(active_tab, 1.0)
        new_factor = preview["factor"]
        incremental_factor = new_factor / old_factor if old_factor else new_factor

        self.spectrum_shift_factor[active_tab] = new_factor
        self.spectrum_shift_preview[active_tab] = None

        if active_tab == "full":
            new_wavelength = self._get_tab_wavelength("full")
            self.cube_data.wavelength = new_wavelength.copy()
            self.spec_plotter.wavelength = new_wavelength.copy()
            # Gaussian fit artists are drawn against Full's axis only
            # (unmigrated feature - see plan's deferred section).
            self.spec_plotter.shift_plot_elements(incremental_factor)

        # Auto-fit the x-axis to this tab's new extent (with the same ~5%
        # padding the unshifted view gets), rather than rescaling whatever
        # the previous view window happened to be - the wavelength axis
        # genuinely compresses/expands under this multiplicative shift, so
        # keeping an old (now-mismatched) view window looks like part of the
        # spectrum went missing.
        self._fit_xlim_to(self._get_tab_wavelength(active_tab))

        # Recomputes proper y-limits from the actual (now-shifted) data,
        # redraws this tab's trace in its own color (not orange - that was
        # only ever a preview cue) at the new position, updates its
        # "(Shifted)" label, and rebuilds the legend.
        self._refresh_active_spectrum_tab()

        QMessageBox.information(
            self,
            "Shift Applied",
            f"Shift applied to the {tab_label} spectrum."
        )

        print(f"[DEBUG] Shift applied to '{active_tab}': cumulative factor={new_factor:.6f}")

    def _cancel_shift_preview(self):
        """Cancel whichever tab's pending shift preview is active, without
        applying it - removes the orange overplot, leaves the applied
        factor (if any) untouched."""
        active_tab = self.active_spectrum_tab
        if not self.spectrum_shift_preview.get(active_tab):
            return
        self.spectrum_shift_preview[active_tab] = None
        self._refresh_active_spectrum_tab()
        print(f"[DEBUG] Cancelled shift preview on '{active_tab}'")

    def _revert_shift(self):
        """Revert whichever tab is active back to its original (unshifted)
        wavelength - a full reset to factor 1.0, not a step-wise undo."""
        active_tab = self.active_spectrum_tab
        if self.spectrum_shift_factor.get(active_tab, 1.0) == 1.0:
            return

        tab_label = active_tab.capitalize()
        reply = QMessageBox.question(
            self,
            "Revert Shift",
            f"Revert the {tab_label} spectrum back to its original (unshifted) wavelength?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.spectrum_shift_factor[active_tab] = 1.0
        self.spectrum_shift_preview[active_tab] = None

        if active_tab == "full":
            self.cube_data.wavelength = np.asarray(self.base_wavelength, dtype=float).copy()
            self.spec_plotter.wavelength = np.asarray(self.base_wavelength, dtype=float).copy()

        self._fit_xlim_to(self._get_tab_wavelength(active_tab))

        self._refresh_active_spectrum_tab()
        print(f"[DEBUG] Reverted shift on '{active_tab}'")

    def _open_colormap_dialog(self, target):
        dialog = ColorMapDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            cmap = dialog.selected_cmap
            if target == 'left':
                self.slice1_cmap = cmap
                self.slice1_plotter.set_cmap(cmap)
            else:
                self.slice2_cmap = cmap
                self.slice2_plotter.set_cmap(cmap)

    def copy_colormap(self, source, target):
        src_cmap = getattr(self, 'slice1_cmap' if source == 'left' else 'slice2_cmap', None)
        if not src_cmap:
            return
        if target == 'left':
            self.slice1_cmap = src_cmap
            self.slice1_plotter.set_cmap(src_cmap)
        else:
            self.slice2_cmap = src_cmap
            self.slice2_plotter.set_cmap(src_cmap)

    def _change_scale(self, target):
        """Change the intensity scaling (linear, log, sqrt, power) for a slice"""
        scale_combo = self.scale_combo_left if target == 'left' else self.scale_combo_right
        scale_type = scale_combo.currentText()
        
        # Store the scale choice
        if target == 'left':
            self.scale_left = scale_type
        else:
            self.scale_right = scale_type
        
        # Redisplay the current slice with the new scale
        if hasattr(self, 'cube_data') and self.cube_data is not None:
            idx = self.last_idx_left if target == 'left' else self.last_idx_right
            self._display_slice(idx, target)

    def _contam_rows(self):
        """
        Convert the loaded contaminating-line catalog into row dicts shaped like
        HITRAN/custom detection-line rows (tagged 'source': 'Contaminant'), so
        they can share the same Lines table, plotting, and hover code as
        detection lines instead of a separate parallel path.

        Same optional-column treatment as load_custom_line_catalog(): only
        'species'/'wavelength' are required of a contaminant CSV, but if it
        also happens to carry transition/a_coeff/e_upper/isotopologue(_name)
        columns, those are shown too -- there's no reason to hide real
        metadata just because a line is tagged as a contaminant rather than a
        detection.
        """
        if self.contam_lines is None or len(self.contam_lines) == 0:
            return []
        rows = []
        for _, row in self.contam_lines.iterrows():
            entry = {
                'model': 'Contaminant',
                'source': 'Contaminant',
                'wavelength': float(row['line_wave_um']),
                'wavelength_unit': 'µm',
                # No fallback -- load_contaminating_lines_catalog() now requires
                # every row to have a real species, same as custom detection lines.
                'species': row['species'],
            }
            transition = row.get('transition')
            if pd.notna(transition):
                entry['transition'] = str(transition)
            for float_col in ('a_coeff', 'e_upper'):
                value = row.get(float_col)
                if pd.notna(value):
                    try:
                        entry[float_col] = float(value)
                    except (TypeError, ValueError):
                        pass
            iso_num = row.get('isotopologue')
            if pd.notna(iso_num):
                try:
                    entry['isotopologue'] = int(iso_num)
                except (TypeError, ValueError):
                    pass
            # Not run through format_isotopologue_name() -- same reasoning as
            # load_custom_line_catalog(): that assumes HITRAN's own bracket
            # notation and would misformat plain user text. Shown as given.
            iso_name = row.get('isotopologue_name')
            if pd.notna(iso_name):
                entry['isotopologue_name'] = str(iso_name)
            rows.append(entry)
        return rows

    def _combined_line_rows(self):
        """All currently loaded lines -- detection (HITRAN/custom) plus any
        loaded contaminants -- combined and wavelength-sorted, for the shared
        Lines selection dialog."""
        rows = list(getattr(self, 'hitran_lines', None) or []) + self._contam_rows()
        rows.sort(key=lambda r: float(r.get('lambda_micron') or r.get('wavelength') or 0.0))
        return rows

    def show_available_species(self):
        """
        Show the combined Lines table: detection lines (HITRAN or custom-imported)
        plus any loaded contaminating lines, with "Detection Only"/"Contaminating
        Only" buttons to select either set separately.
        """
        all_lines = self._combined_line_rows()

        if not all_lines:
            QMessageBox.warning(
                self, "No Lines Loaded",
                "Fetch HITRAN lines, import a custom line list, or load contaminating lines first."
            )
            return

        # Show selection dialog
        dlg = SpeciesDialog(all_lines, parent=self, plotted_rows=self.plotted_lines)
        result = dlg.exec_()

        if result == QDialog.Accepted:
            selected_rows = dlg.get_selected_rows()
            if not selected_rows:
                QMessageBox.information(self, "No Selection", "No lines selected.")
                return
            self.plot_selected_lines(selected_rows)

    def plot_selected_lines(self, rows):
        """Replace the full set of plotted lines (detection + contaminant alike)
        with exactly what's checked in the Lines dialog, then redraw."""
        self.plotted_lines = rows.copy()

        # Sync the excluded wavelengths set:
        # Remove any wavelengths from excluded that are now being plotted (re-selection)
        if hasattr(self, 'co_flux_excluded_wavelengths') and self.co_flux_excluded_wavelengths:
            plotted_wavelengths = set(
                SpeciesDialog.get_row_wavelength(r) for r in rows
            )
            # Remove from excluded any that are now plotted
            self.co_flux_excluded_wavelengths = {
                w for w in self.co_flux_excluded_wavelengths
                if not any(abs(w - pw) < 1e-6 for pw in plotted_wavelengths)
            }

        self._draw_plotted_lines()

    def plot_contaminating_lines(self):
        """
        Merge the loaded contaminating-line catalog into the currently-plotted
        lines -- replacing only the previously-plotted contaminant markers,
        leaving any detection lines already on the spectrum untouched -- and
        redraw. Called right after a contaminating-lines CSV loads.
        """
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            return
        if not hasattr(self, 'plotted_lines'):
            self.plotted_lines = []

        non_contam = [r for r in self.plotted_lines if r.get('source') != 'Contaminant']
        self.plotted_lines = non_contam + self._contam_rows()

        self._draw_plotted_lines()

    def _draw_plotted_lines(self):
        """
        Redraw every currently-plotted line marker from self.plotted_lines.
        Detection lines (HITRAN/Custom) use the darkred/orange scheme; lines
        tagged 'source': 'Contaminant' use black/dimgray instead, so the two
        stay visually distinct despite sharing one plotted-lines list.
        """
        ax = self.spec_plotter.ax

        for artist in self.plotted_artists:
            try:
                artist.remove()
            except Exception as e:
                print("DEBUG: could not remove artist", e)
        self.plotted_artists = []

        # Get cube wavelength unit
        cube_unit = getattr(self.cube_data, 'wave_unit', 'µm') if hasattr(self, 'cube_data') else 'µm'

        for row in self.plotted_lines:
            λ = SpeciesDialog.get_row_wavelength(row)

            # Convert wavelength to cube units if needed
            source_unit = row.get('wavelength_unit', 'µm')
            try:
                if cube_unit != source_unit and are_units_compatible(source_unit, cube_unit):
                    λ_converted = convert_wavelength(λ, source_unit, cube_unit)
                    print(f"[DEBUG] Converted line {row.get('species', 'Unknown')}: {λ:.6f} {source_unit} -> {λ_converted:.6f} {cube_unit}")
                    λ = λ_converted
            except Exception as e:
                print(f"[DEBUG] Unit conversion failed for {row.get('species', 'Unknown')}: {e}")
                # Use original wavelength if conversion fails

            is_contam = row.get('source') == 'Contaminant'
            line_color = "black" if is_contam else "darkred"
            line_alpha = 0.7 if is_contam else 0.6
            text_color = "black" if is_contam else "orange"

            line = ax.axvline(λ, color=line_color, linestyle="--", alpha=line_alpha, picker=5)
            # Store row data in the line for hover access
            line.row_data = row
            # Blended transform (x in data coords, y in axes-fraction coords) instead of a
            # one-time ax.get_ylim()[1] snapshot -- keeps the label flush with the axes' top
            # edge even if the y-limit later changes (e.g. via _apply_shared_y_scale), rather
            # than leaving it anchored to a now-stale, possibly lower, y value. y=1.01 (not
            # 1.0) so the label's first character clears the top spine instead of sitting
            # right on top of it.
            text = ax.text(
                λ, 1.01, row["species"], transform=ax.get_xaxis_transform(),
                rotation=90, va="bottom", ha="center",
                fontsize=7, color=text_color
            )
            self.plotted_artists.extend([line, text])

        self.spec_canvas.draw()

    def _on_spec_hover(self, event):
        """Handle hover events over spectrum plot to show line info."""

        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            return
        if event.inaxes != self.spec_plotter.ax:
            return

        if not hasattr(self, 'plotted_lines') or not self.plotted_lines:
            return

        # Check if hovering near any plotted lines
        hover_threshold = 0.01  # wavelength units tolerance
        mouse_x = event.xdata
        if mouse_x is None:
            return

        # Find nearest line within threshold -- detection lines and contaminant
        # markers share plotted_lines, so they compete on the same distance check
        nearest_row = None
        min_dist = float('inf')

        for row in self.plotted_lines:
            λ = SpeciesDialog.get_row_wavelength(row)
            dist = abs(λ - mouse_x)
            if dist < hover_threshold and dist < min_dist:
                min_dist = dist
                nearest_row = row

        if nearest_row:
            # Get cube wavelength unit
            cube_unit = getattr(self.cube_data, 'wave_unit', 'µm') if hasattr(self, 'cube_data') else 'µm'

            species = nearest_row.get('species', 'Unknown')
            wavelength = nearest_row.get('wavelength', 0)
            source_unit = nearest_row.get('wavelength_unit', 'µm')

            # Same precision as the Lines table (species_dialog.py's wavelength
            # column uses .6f) -- not rounded to fewer digits here.
            if cube_unit != source_unit:
                try:
                    wavelength_converted = convert_wavelength(wavelength, source_unit, cube_unit)
                    wavelength_str = f"λ: {wavelength:.6f} {source_unit} ({wavelength_converted:.6f} {cube_unit})"
                except Exception:
                    wavelength_str = f"λ: {wavelength:.6f} {source_unit}"
            else:
                wavelength_str = f"λ: {wavelength:.6f} {source_unit}"

            # Detection and contaminant rows are formatted identically -- a
            # contaminant CSV can carry transition/a_coeff/e_upper/isotopologue
            # columns too (see _contam_rows()), so there's no reason to hard-code
            # a stripped-down tooltip just because a line is tagged 'Contaminant'.
            # Whatever's actually missing already shows "N/A" below.
            iso_name = nearest_row.get('isotopologue_name')
            iso_num = nearest_row.get('isotopologue')
            if iso_name:
                species = f"{species} ({iso_name})"
            elif iso_num:
                species = f"{species} (iso {iso_num})"
            transition = nearest_row.get('transition', 'N/A')
            # Default to None (not 0) -- 0 is a real value (e.g. E_upper=0 for a
            # ground-state line), so a genuinely missing field (e.g. a custom-
            # imported or contaminant line whose CSV didn't include these optional
            # columns) must format as "N/A" rather than silently print as "0.00e+00".
            a_coeff = nearest_row.get('a_coeff')
            e_upper = nearest_row.get('e_upper')
            a_coeff_text = f"{a_coeff:.2e} s⁻¹" if a_coeff is not None else "N/A"
            e_upper_text = f"{e_upper:.2f} K" if e_upper is not None else "N/A"

            tooltip_text = (
                f"Species: {species}\n"
                f"{wavelength_str}\n"
                f"Transition: {transition}\n"
                f"A-coeff: {a_coeff_text}\n"
                f"E_upper: {e_upper_text}"
            )

            # Use QToolTip for floating box - convert matplotlib event coords to global screen coords
            cursor_pos = self.spec_canvas.mapToGlobal(QPoint(int(event.x), int(self.spec_canvas.height() - event.y)))
            QToolTip.showText(cursor_pos, tooltip_text, self.spec_canvas)
        else:
            # Hide tooltip when not hovering over a line
            QToolTip.hideText()

    def refresh_view(self):
        """
        Complete reset: Clear all plots, selections, normalizations, and Gaussian fits.
        Reload spectrum from original data.
        """
        print("[DEBUG] === REFRESH VIEW: Complete Reset ===")
        
        if not hasattr(self, 'cube_data') or self.cube_data is None:
            print("[DEBUG] No cube data loaded")
            return
        
        # Reset state variables
        self.last_idx_left = 0
        self.last_idx_right = 0
        self.last_selection_left = None
        self.last_selection_right = None
        self.active_shape = None
        self.active_target = None
        
        # Reset Gaussian fitting state
        self.gauss_left_x = None
        self.gauss_peak_x = None
        self.gauss_right_x = None
        self.gauss_noise_left_x = None
        self.gauss_noise_right_x = None
        self.gauss_mode = None
        if hasattr(self, 'gauss_left_marker') and self.gauss_left_marker:
            try:
                self.gauss_left_marker.remove()
            except Exception:
                pass
        if hasattr(self, 'gauss_right_marker') and self.gauss_right_marker:
            try:
                self.gauss_right_marker.remove()
            except Exception:
                pass
        if hasattr(self, 'gauss_noise_left_marker') and self.gauss_noise_left_marker:
            try:
                self.gauss_noise_left_marker.remove()
            except Exception:
                pass
        if hasattr(self, 'gauss_noise_right_marker') and self.gauss_noise_right_marker:
            try:
                self.gauss_noise_right_marker.remove()
            except Exception:
                pass
        self.gauss_left_marker = None
        self.gauss_right_marker = None
        self.gauss_noise_left_marker = None
        self.gauss_noise_right_marker = None

        # Reset per-tab Gaussian boundary state and the map-fit cache - a new
        # cube invalidates both (stale marker artists belong to the old axes,
        # and a cached map/tab key would otherwise wrongly appear "current").
        self.gauss_boundaries = {"full": {}, "left": {}, "right": {}}
        self.gauss_cache = None
        self.gauss_cache_tab = None
        self.gauss_cache_lam_min = None
        self.gauss_cache_lam_max = None
        self.gauss_cache_noise_min = None
        self.gauss_cache_noise_max = None

        # Clear any plotted continuum line
        if hasattr(self, 'continuum_line') and self.continuum_line:
            try:
                self.continuum_line.remove()
            except Exception:
                pass
        self.continuum_line = None
        
        # Uncheck all buttons
        self.set_left_btn.setChecked(False)
        self.set_right_btn.setChecked(False)
        self.gauss_left_btn.setChecked(False)
        self.gauss_right_btn.setChecked(False)
        
        # Clear all plotted lines (detection + contaminant)
        if hasattr(self, 'plotted_artists'):
            for artist in self.plotted_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            self.plotted_artists = []
            self.plotted_lines = []

        # Clear spectrum plot and reload
        if hasattr(self, 'spec_plotter') and self.spec_plotter is not None:
            # Clear all lines and artists
            self.spec_plotter.ax.clear()
            
            # Clear Gaussian fit artists
            if hasattr(self.spec_plotter, 'gauss_fit_artists'):
                for artist in self.spec_plotter.gauss_fit_artists:
                    try:
                        artist.remove()
                    except Exception:
                        pass
                self.spec_plotter.gauss_fit_artists = []
            
            # Reload spectrum from original
            if hasattr(self, 'original_spectrum') and self.original_spectrum is not None:
                self.spec_plotter.plot_spectrum(self.original_spectrum)

            # ax.clear() above dropped the dedicated Left/Right tab lines too -
            # force them to be recreated, and show whatever the active tab
            # should now display (last_selection_left/right were reset above,
            # so Left/Right will correctly come back empty unless Full is active).
            self._spec_tab_line_left = None
            self._spec_tab_line_right = None
            self._reset_slice_indicators()
            self._refresh_active_spectrum_tab()

            self.spec_canvas.draw()
        
        # Reset slice displays to index 0
        if hasattr(self, 'slice1_plotter') and self.slice1_plotter is not None:
            # Clear patches and markers
            if hasattr(self.slice1_plotter.ax, 'area_artists'):
                for artist in self.slice1_plotter.ax.area_artists:
                    try:
                        artist.remove()
                    except Exception:
                        pass
                self.slice1_plotter.ax.area_artists = []
            # Remove handle scatter collections
            for collection in list(self.slice1_plotter.ax.collections):
                try:
                    if hasattr(collection, '_is_handle_collection') and collection._is_handle_collection:
                        collection.remove()
                except Exception:
                    pass
            if hasattr(self.slice1_plotter.ax, 'pixel_marker'):
                for marker in self.slice1_plotter.ax.pixel_marker:
                    try:
                        marker.remove()
                    except Exception:
                        pass
                self.slice1_plotter.ax.pixel_marker = []
            
            self._display_slice(0, "left")
        
        if hasattr(self, 'slice2_plotter') and self.slice2_plotter is not None:
            # Clear patches and markers
            if hasattr(self.slice2_plotter.ax, 'area_artists'):
                for artist in self.slice2_plotter.ax.area_artists:
                    try:
                        artist.remove()
                    except Exception:
                        pass
                self.slice2_plotter.ax.area_artists = []
            # Remove handle scatter collections
            for collection in list(self.slice2_plotter.ax.collections):
                try:
                    if hasattr(collection, '_is_handle_collection') and collection._is_handle_collection:
                        collection.remove()
                except Exception:
                    pass
            if hasattr(self.slice2_plotter.ax, 'pixel_marker'):
                for marker in self.slice2_plotter.ax.pixel_marker:
                    try:
                        marker.remove()
                    except Exception:
                        pass
                self.slice2_plotter.ax.pixel_marker = []
            
            self._display_slice(0, "right")
        
        # Reset stats labels
        self.stats1_label.setText("Slice 1 Stats:\n-")
        self.stats2_label.setText("Slice 2 Stats:\n-")
        
        # Deactivate region selectors and drop any live interactive patch
        # (previously only activate_selector(None) was called here, which
        # stops the mpl selector widget but leaves a connected InteractivePatch
        # wired to the canvas)
        if hasattr(self, 'region_selector_left'):
            if self.region_selector_left.interactive_patch is not None:
                self.region_selector_left.interactive_patch.disconnect()
                self.region_selector_left.interactive_patch = None
            self.region_selector_left.activate_selector(None)
        if hasattr(self, 'region_selector_right'):
            if self.region_selector_right.interactive_patch is not None:
                self.region_selector_right.interactive_patch.disconnect()
                self.region_selector_right.interactive_patch = None
            self.region_selector_right.activate_selector(None)

        # Uncheck shape buttons
        if hasattr(self, 'shape_buttons'):
            for side in ["left", "right"]:
                for btn in self.shape_buttons[side].values():
                    btn.setChecked(False)

        # Reset region-selection state so shape/Edit-Shape buttons don't stay
        # disabled from a stale "drawing"/"placed"/"editing" state.
        if hasattr(self, 'region_state'):
            for side in ["left", "right"]:
                self._set_region_state(side, "idle")

        print("[DEBUG] Refresh complete - all displays reset to initial state")

    def _current_wavelength_range(self):
        """
        Read + validate the Min/Max wavelength (µm) fields shared by the
        HITRAN/custom-lines/contaminating-lines loaders. Pops a warning dialog
        and returns None on invalid input (not a valid number, or min >= max).
        """
        try:
            wavelength_min = float(self.min_wave_input.text())
            wavelength_max = float(self.max_wave_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Enter valid wavelength numbers.")
            return None

        if wavelength_min >= wavelength_max:
            QMessageBox.warning(self, "Invalid Range", "Min wavelength must be less than Max.")
            return None

        return wavelength_min, wavelength_max

    def _line_range_edge_pad(self):
        """
        Safety margin (µm) added around the Min/Max wavelength range when filtering
        custom/contaminating lines at load time, so a line just outside the typed
        range but still close enough to bleed into an edge line's own Gaussian isn't
        silently dropped from the catalog entirely (which -- unlike display -- would
        also remove it from what the fitter can use as a template).

        Computed the same way fit_hitran_lines_collective() pads its own fit window
        (4x the instrument's line sigma, from _line_sigma()) rather than a separate
        guessed-at constant, so "close enough to matter to the fit" means the same
        thing in both places. Returns 0.0 (no padding) if no cube is loaded yet --
        there's no spectrum to reason about a sigma from.
        """
        if not hasattr(self, 'cube_data') or self.cube_data is None:
            return 0.0
        try:
            from .co_flux_extractor import _line_sigma
            wave = np.asarray(_extract_wavelength_value(self.cube_data.wavelength), dtype=float)
            dx = abs(float(np.median(np.diff(wave))))  # magnitude only -- a descending
            # wavelength array must not flip this into a negative (range-shrinking) pad
            header = getattr(self.cube_data, 'header', None)
            edge_sigmas = [
                _line_sigma(float(np.nanmin(wave)), header, dx, 2.5),
                _line_sigma(float(np.nanmax(wave)), header, dx, 2.5),
            ]
            return 4 * max(edge_sigmas)
        except Exception:
            return 0.0

    def _reset_lines_state(self):
        """
        Clear everything downstream of the current detection-line catalog --
        plotted line markers, any multi-Gaussian fit, and the loaded
        contaminating-line catalog -- whenever the detection-line source
        changes (a new HITRAN fetch or custom import). Otherwise a stale fit
        or a contaminant catalog filtered/padded around the *previous*
        wavelength range would silently stick around next to a newly loaded,
        unrelated set of lines. Contaminants specifically must be reloaded
        for the new range rather than carried over -- their fit-time
        correctness depends on being filtered to the range actually in use
        (see _load_contaminating_lines_dialog()).
        """
        if hasattr(self, 'clear_gaussian_fit'):
            self.clear_gaussian_fit()

        if hasattr(self, 'plotted_artists'):
            for artist in self.plotted_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            self.plotted_artists = []
            self.plotted_lines = []

        self.contam_lines = None

        if hasattr(self, 'spec_canvas'):
            self.spec_canvas.draw()

    def fetch_hitran_lines(self):
        """
        Fetch HITRAN lines based on user input from the GUI.
        Now includes transition level (vibrational quantum number) filtering.
        """
        try:
            wave_range = self._current_wavelength_range()
            if wave_range is None:
                return
            wavelength_min, wavelength_max = wave_range
            
            mol_map = get_available_molecules()
            molecules = sorted(mol_map.keys())
            molecule, ok = QInputDialog.getItem(
                self,
                "Select Molecule",
                "Choose a HITRAN molecule:",
                molecules,
                0,
                True  # editable so they can type
            )
            if not ok:
                return
            molecule = molecule.strip().upper()
            if molecule == "":
                QMessageBox.warning(self, "Invalid Input", "Molecule cannot be empty.")
                return
            if molecule not in mol_map:
                QMessageBox.warning(self, "Unknown Molecule", f"'{molecule}' is not in the HITRAN list.")
                return

            # Isotopologue selection -- optional: someone unsure which isotopologue
            # a line belongs to can search every isotopologue HITRAN has for this
            # molecule instead of having to guess a number up front. Each result
            # row stays individually tagged with its own isotopologue afterward.
            know_iso = QMessageBox.question(
                self,
                "Isotopologue",
                "Do you know the isotopologue number for this molecule?\n\n"
                "Choose No to search all isotopologues at once instead.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            ) == QMessageBox.Yes

            if know_iso:
                isotopologue, ok = QInputDialog.getInt(
                    self,
                    "Isotopologue",
                    "Enter isotopologue number:",
                    1,
                    1,
                    100,
                    1
                )
                if not ok:
                    return
            else:
                isotopologue = None

            # Ask if user wants to filter by transition numbers (optional)
            filter_quantum = QMessageBox.question(
                self,
                "Transition Number Filtering",
                "Do you want to filter by transition numbers for this molecule?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            ) == QMessageBox.Yes

            # Cache refresh option
            refresh_cache = QMessageBox.question(
                self,
                "Cache Option",
                "Refresh the HITRAN cache for this query?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            ) == QMessageBox.Yes

            # Fetch HITRAN data
            results = hitran_gui_interface(
                wavelength_min, wavelength_max, molecule, isotopologue, refresh_cache
            )

            if not results:
                QMessageBox.information(self, "No Matches", "No HITRAN lines found in this range.")
                return

            # Filter by transition level based on molecule (molecule-specific logic)
            if filter_quantum:
                filtered_results = []

                if molecule.upper() == 'CO' or molecule.upper() == 'H2':
                    # CO transition filtering
                    v_upper, ok = QInputDialog.getInt(
                        self,
                        "CO Upper Transition Number",
                        "Enter upper vibrational state:\n(e.g., 1 for v=1→0)",
                        1,
                        0,
                        100,
                        1
                    )
                    if not ok:
                        return

                    v_lower, ok = QInputDialog.getInt(
                        self,
                        "CO Lower Transition Number",
                        "Enter lower vibrational state:\n(e.g., 0 for v=1→0)",
                        0,
                        0,
                        100,
                        1
                    )
                    if not ok:
                        return

                    # CO uses regex: r'v=(\d+)->(\d+)' to extract and match vibrational levels
                    for line in results:
                        transition = line.get('transition', '')
                        if transition and '->' in transition:
                            try:
                                v_match = re.search(r'v=(\d+)->(\d+)', transition)
                                if v_match:
                                    line_v_upper = int(v_match.group(1))
                                    line_v_lower = int(v_match.group(2))
                                    if line_v_upper == v_upper and line_v_lower == v_lower:
                                        filtered_results.append(line)
                            except (ValueError, AttributeError):
                                filtered_results.append(line)
                        else:
                            filtered_results.append(line)

                    if not filtered_results:
                        QMessageBox.information(
                            self,
                            "No Matches",
                            f"No HITRAN lines found with v={v_upper}->{v_lower} transition."
                        )
                        return

                elif molecule.upper() == 'H2O':
                    # H2O transition filtering
                    h2o_upper_text, ok = QInputDialog.getText(
                        self,
                        "H₂O Upper Transition Numbers",
                        "Enter upper state (v Ka Kc):\n(e.g., 0 3 0 for 0 3 0→0 1 0)",
                        QLineEdit.Normal,
                        "0 3 0"
                    )
                    if not ok or not h2o_upper_text:
                        return

                    h2o_lower_text, ok = QInputDialog.getText(
                        self,
                        "H₂O Lower Transition Numbers",
                        "Enter lower state (v Ka Kc):\n(e.g., 0 1 0 for 0 3 0→0 1 0)",
                        QLineEdit.Normal,
                        "0 1 0"
                    )
                    if not ok or not h2o_lower_text:
                        return

                    # H2O uses simple string matching: "0 3 0->0 1 0"
                    h2o_pattern = f"{h2o_upper_text}->{h2o_lower_text}"
                    for line in results:
                        transition = line.get('transition', '')
                        if transition and h2o_pattern in transition:
                            filtered_results.append(line)

                    if not filtered_results:
                        QMessageBox.information(
                            self,
                            "No Matches",
                            f"No H₂O lines found matching {h2o_pattern} transition."
                        )
                        return

                elif molecule.upper() == 'OH':
                    # OH transition filtering
                    oh_upper_text, ok = QInputDialog.getText(
                        self,
                        "OH Upper Transition State",
                        "Enter upper state:\n(e.g., X1/2   3 for v=X1/2   3→X3/2   2)",
                        QLineEdit.Normal,
                        "X1/2   3"
                    )
                    if not ok or not oh_upper_text:
                        return

                    oh_lower_text, ok = QInputDialog.getText(
                        self,
                        "OH Lower Transition State",
                        "Enter lower state:\n(e.g., X3/2   2 for v=X1/2   3→X3/2   2)",
                        QLineEdit.Normal,
                        "X3/2   2"
                    )
                    if not ok or not oh_lower_text:
                        return

                    # OH uses simple string matching: "X1/2   3->X3/2   2"
                    oh_pattern = f"{oh_upper_text}->{oh_lower_text}"
                    for line in results:
                        transition = line.get('transition', '')
                        if transition and oh_pattern in transition:
                            filtered_results.append(line)

                    if not filtered_results:
                        QMessageBox.information(
                            self,
                            "No Matches",
                            f"No OH lines found matching {oh_pattern} transition."
                        )
                        return

                results = filtered_results

            # Mark HITRAN lines with source
            for line in results:
                line['source'] = 'HITRAN'

            # A fresh fetch replaces the working line set entirely -- clear any
            # stale fit/plotted markers/contaminant catalog from before it.
            self._reset_lines_state()

            # Store HITRAN lines
            self.hitran_lines = results

            # Show HITRAN results (plus any loaded contaminants) in selection dialog
            dlg = SpeciesDialog(self._combined_line_rows(), parent=self, plotted_rows=self.plotted_lines)
            if dlg.exec_() == QDialog.Accepted:
                selected_rows = dlg.get_selected_rows()
                if not selected_rows:
                    QMessageBox.information(self, "No Selection", "No lines selected for plotting.")
                    return
                self.plot_selected_lines(selected_rows)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while fetching HITRAN lines:\n{e}")

    def _load_custom_lines_dialog(self):
        """
        Load a user-supplied line list CSV as a stand-in for a live HITRAN fetch
        (e.g. no internet access). 'wavelength' and 'species' are required; any
        other HITRAN-style columns (transition, a_coeff, e_upper, isotopologue,
        isotopologue_name) are optional and only shown where present. Feeds into
        the same Lines table / plotting flow as fetch_hitran_lines(), replacing
        whatever was previously loaded there. Only lines within the current
        Min/Max wavelength range are kept -- exactly, no edge padding, same as
        a live HITRAN fetch (which queries that exact range) -- a catalog
        covering a wider band than what's being searched shouldn't dump
        irrelevant lines into the table. Unlike contaminating lines, a
        detection line just outside the range has no fit-suppression role
        that would justify padding it back in.

        Rejected outright if more than one species falls in range: a live
        HITRAN fetch is always single-molecule by construction (one QInputDialog
        selection), and fit_hitran_lines_collective()'s single shared velocity
        shift (estimated from these lines' own peaks) is only physically
        meaningful for one species at a time -- pooling two species' peaks
        together would silently average across what could be two unrelated
        offsets. Contaminating lines have no such restriction (see
        _load_contaminating_lines_dialog()) since they're never pooled into
        that shift estimate.
        """
        wave_range = self._current_wavelength_range()
        if wave_range is None:
            return
        wavelength_min, wavelength_max = wave_range

        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_path = os.path.join(base_path, 'data')

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Custom Line List CSV",
            default_path,
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if not file_path:
            return  # User cancelled

        try:
            rows = load_custom_line_catalog(file_path)
            if not rows:
                QMessageBox.critical(
                    self,
                    "Invalid CSV Format",
                    "❌ Couldn't load any lines from this file.\n\n"
                    "The CSV MUST have columns named 'wavelength' (µm) and 'species' "
                    "(case-sensitive), with both filled in for every row you want loaded."
                )
                return

            total_loaded = len(rows)
            rows = [r for r in rows if wavelength_min <= r['wavelength'] <= wavelength_max]
            if not rows:
                QMessageBox.information(
                    self, "No Lines In Range",
                    f"Loaded {total_loaded} line(s) from this file, but none fall within "
                    f"the current range ({wavelength_min:.6f}-{wavelength_max:.6f} µm)."
                )
                return

            distinct_species = sorted({r['species'] for r in rows})
            if len(distinct_species) > 1:
                QMessageBox.critical(
                    self, "Multiple Species Found",
                    "❌ This file has more than one species within the current wavelength "
                    f"range: {', '.join(distinct_species)}.\n\n"
                    "Detection lines must be a single species at a time -- the multi-Gaussian "
                    "fit's velocity-shift estimate is pooled across every selected detection "
                    "line, which only makes sense for one species. Split this into one file "
                    "per species, or use \"Contam. Lines\" instead if some of these are meant "
                    "to be excluded from the fit rather than detected."
                )
                return

            print(
                f"[OK] Loaded {len(rows)} of {total_loaded} custom lines from "
                f"{os.path.basename(file_path)} within {wavelength_min:.6f}-{wavelength_max:.6f} µm"
            )

            # A fresh import replaces the working line set entirely -- clear any
            # stale fit/plotted markers/contaminant catalog from before it.
            self._reset_lines_state()

            self.hitran_lines = rows

            dlg = SpeciesDialog(self._combined_line_rows(), parent=self, plotted_rows=self.plotted_lines)
            if dlg.exec_() == QDialog.Accepted:
                selected_rows = dlg.get_selected_rows()
                if not selected_rows:
                    QMessageBox.information(self, "No Selection", "No lines selected for plotting.")
                    return
                self.plot_selected_lines(selected_rows)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load custom line list:\n{e}")

    def _load_contaminating_lines_dialog(self):
        """
        Allow user to load an optional contaminating lines CSV file.
        This should be done AFTER setting wavelength range and loading HITRAN.

        Filtered down to the current Min/Max wavelength range (plus a small
        edge margin -- see _line_range_edge_pad()) before storing, same as
        custom detection lines -- a catalog covering a wider band than what's
        actually being analyzed (in either the multi-Gaussian fit or CO Flux
        Map, both of which are themselves scoped to this same range in the
        GUI) shouldn't clutter the table/plot or slow down fitting with
        contaminants that can never matter here.
        """
        wave_range = self._current_wavelength_range()
        if wave_range is None:
            return
        wavelength_min, wavelength_max = wave_range
        pad = self._line_range_edge_pad()
        wavelength_min, wavelength_max = wavelength_min - pad, wavelength_max + pad

        # Suggest default location
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_path = os.path.join(base_path, 'data', 'CO')

        # Open file browser
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Contaminating Lines CSV",
            default_path,
            "CSV Files (*.csv);;All Files (*.*)"
        )

        if not file_path:
            return  # User cancelled

        try:
            # Load the contaminating lines catalog
            loaded = load_contaminating_lines_catalog(file_path)

            if loaded is not None and len(loaded) > 0:
                total_loaded = len(loaded)
                in_range = loaded[
                    (loaded['line_wave_um'] >= wavelength_min) & (loaded['line_wave_um'] <= wavelength_max)
                ].reset_index(drop=True)
                if len(in_range) == 0:
                    QMessageBox.information(
                        self, "No Lines In Range",
                        f"Loaded {total_loaded} line(s) from this file, but none fall within "
                        f"the current range ({wavelength_min:.6f}-{wavelength_max:.6f} µm)."
                    )
                    self.contam_lines = None
                    return
                self.contam_lines = in_range

                print(
                    f"[OK] Loaded {len(self.contam_lines)} of {total_loaded} contaminating lines from "
                    f"{os.path.basename(file_path)} within {wavelength_min:.6f}-{wavelength_max:.6f} µm"
                )

                # Show wavelength range of loaded lines
                min_wave = self.contam_lines['line_wave_um'].min()
                max_wave = self.contam_lines['line_wave_um'].max()
                print(f"  Wavelength range: {min_wave:.6f} - {max_wave:.6f} um")

                # Mark them on the spectrum right away (dark grey), same as HITRAN selections
                self.plot_contaminating_lines()

                QMessageBox.information(
                    self,
                    "Success",
                    f"✓ Loaded {len(self.contam_lines)} of {total_loaded} contaminating lines "
                    f"(within {wavelength_min:.6f}-{wavelength_max:.6f} µm)\n\n"
                    f"Wavelength range: {min_wave:.6f} - {max_wave:.6f} μm\n\n"
                    f"These lines will be used when extracting CO flux."
                )
            else:
                QMessageBox.warning(self, "Empty File", "The CSV file is empty or has no valid lines.")
                self.contam_lines = None

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading File",
                f"Failed to load contaminating lines:\n{e}"
            )
            self.contam_lines = None

    def _spectrum_home_reset(self):
        """Custom home action for spectrum: show original spectrum and clear overlays"""
        print("[DEBUG] Spectrum home button clicked - restoring original spectrum")
        
        # Restore original spectrum if it has been modified
        if self.original_spectrum_backup is not None:
            self._restore_original_spectrum()
        
        # Call the original NavigationToolbar home method to reset zoom
        # Temporarily unset the custom action to avoid recursion
        saved_action = self.spec_toolbar.custom_home_action
        self.spec_toolbar.custom_home_action = None
        self.spec_toolbar.home()
        self.spec_toolbar.custom_home_action = saved_action
        
        # Also clear Gaussian fit visualization
        self.clear_gaussian_fit()

        # Clear any plotted line markers/labels (detection + contaminant)
        if hasattr(self, 'plotted_artists') and self.plotted_artists:
            for artist in self.plotted_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            self.plotted_artists = []
            self.plotted_lines = []
            self.spec_canvas.draw()

    def _save_results_ascii(self):
        """Save the full continuum-subtraction result to an ASCII file.

        Columns: wavelength, original_spectrum, fitted_continuum, subtracted
        Missing columns (e.g. no continuum fitted) are written as NaN.
        """
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "No spectrum loaded.")
            return

        wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
        if wavelength is None:
            QMessageBox.warning(self, "No Data", "No wavelength data available.")
            return

        # Original spectrum = the pre-subtraction values.  If a subtraction
        # has been applied, the backup holds the original; otherwise the
        # cube's current spectrum IS the original.
        if getattr(self, 'original_spectrum_backup', None) is not None:
            original = self.original_spectrum_backup
        else:
            original = self.cube_data.spectrum if hasattr(self, 'cube_data') and self.cube_data is not None else None

        if original is None:
            QMessageBox.warning(self, "No Data", "No spectrum to save.")
            return

        # Continuum may not exist if the user never fitted one.
        if hasattr(self, 'continuum_fitted') and self.continuum_fitted is not None:
            continuum = self.continuum_fitted
        else:
            continuum = np.full_like(original, np.nan, dtype=float)

        # Subtracted column = original - continuum, where the continuum is
        # finite.  Where the continuum is NaN, the subtracted result is NaN.
        subtracted = np.where(
            np.isfinite(continuum), original - continuum, np.nan
        )

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results as ASCII",
            "results.txt",
            "Text files (*.txt);;CSV files (*.csv);;All files (*)"
        )
        if not file_path:
            return

        try:
            wave_unit = self.cube_data.wave_unit if hasattr(self.cube_data, 'wave_unit') else "Unknown"
            flux_unit = (self.cube_data.header.get('BUNIT', 'Unknown')
                         if hasattr(self.cube_data, 'header') else 'Unknown')
            instrument = self.cube_data.instrument if hasattr(self.cube_data, 'instrument') else "Unknown"

            with open(file_path, 'w') as f:
                f.write("# Results Export from Cube Explorer\n")
                f.write(f"# Instrument: {instrument}\n")
                f.write("# Type: full_results\n")
                f.write(f"# Wavelength unit: {wave_unit}\n")
                f.write(f"# Flux unit: {flux_unit}\n")
                f.write("#\n")
                f.write(f"# Wavelength ({wave_unit})\tOriginal\tContinuum\tSubtracted\n")
                for w, o, c, s in zip(wavelength, original, continuum, subtracted):
                    f.write(f"{w:.6e}\t{o:.6e}\t{c:.6e}\t{s:.6e}\n")

            QMessageBox.information(self, "Success", f"Results saved to:\n{file_path}")
            print(f"Results saved to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results:\n{e}")
            print(f"Error saving results: {e}")

    def open_2d_window(self):
        try:
            from .ifu_analyzer_2d import Ifu_analyzer_2d_window  # new file/module

            two_d_window = Ifu_analyzer_2d_window()
            two_d_window.resize(self.width(), self.height())
            two_d_window.show()

            if not hasattr(self, "child_windows") or self.child_windows is None:
                self.child_windows = []
            self.child_windows.append(two_d_window)

        except Exception as e:
            QMessageBox.warning(self, "2D Window Error", f"Could not open 2D window:\n{e}")

    def open_rotational_diagram(self):
        try:
            from .gui_rotational_diagram import RotationalDiagramWindow

            rot_diag_window = RotationalDiagramWindow(parent_gui=self)
            rot_diag_window.show()

            if not hasattr(self, "child_windows") or self.child_windows is None:
                self.child_windows = []
            self.child_windows.append(rot_diag_window)

        except Exception as e:
            QMessageBox.warning(self, "Rotational Diagram Error", f"Could not open rotational diagram window:\n{e}")


