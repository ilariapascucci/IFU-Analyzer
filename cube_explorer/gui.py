"""
Main GUI class for the Cube Explorer application
"""

import os
import re
import gc
import warnings
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm, PowerNorm, Normalize

# Suppress FITS warning messages about header formatting
warnings.filterwarnings('ignore', category=Warning, module='astropy.wcs.wcs')
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QMessageBox, QTextEdit, QLineEdit, QInputDialog, QDialog, QToolTip, QComboBox,
    QListWidget, QListWidgetItem, QSplitter, QFrame, QProgressDialog
)
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from PyQt5.QtWidgets import QApplication

from scipy.ndimage import median_filter
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks  # NEW: Import find_peaks

from .data_loader import CubeData
from .visualization import SpectrumPlotter, SlicePlotter
from .colormap_dialog import ColorMapDialog
from .region_selector import RegionSelector
from .continuum_dialog import ContinuumParametersDialog
from .custom_continuum import CustomContinuumSelector
from .line_catalog import get_available_molecules, normalize_species_input, hitran_gui_interface, load_CO_line_catalog, load_contaminating_lines_catalog
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



class CubeExplorer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cube Explorer")
        self.cube_data = CubeData()
        self._init_ui()
        # Load CO line catalogs on startup
        self._load_CO_catalogs()

    def closeEvent(self, event):
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
        
        print('\n>>> Loading CO Catalogues...')
        self.CO_lines = load_CO_line_catalog(CO_path)
        # Contaminating lines are now optional - users can load them separately after HITRAN parameters are set
        self.contam_lines = None
        
        if self.CO_lines is not None:
            print(f'✓ CO lines: {len(self.CO_lines)} entries')
            print(f'✓ Contaminating lines: OPTIONAL (can be loaded later)\n')
        else:
            print('⚠️  Warning: Failed to load CO lines catalog\n')
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
                except:
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
        
        self.active_target = None
        
        # Clear Doppler shifted line if it exists
        if hasattr(self, 'shifted_line') and self.shifted_line is not None:
            try:
                self.shifted_line.remove()
            except:
                pass
            self.shifted_line = None
        
        # Reset Doppler shift state
        self.shifted_spectrum_data = None
        self.shifted_wavelength_data = None
        self.shifted_velocity = None
        self.is_spectrum_shifted = False
        if hasattr(self, 'apply_shift_btn'):
            self.apply_shift_btn.setEnabled(False)
        
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
        
        self.instrument_label = QLabel("Instrument: -")
        self.instrument_label.setStyleSheet("font-weight: bold;")

        self.controls_layout = QHBoxLayout()
        self.controls_layout.addWidget(self.open_btn)
        self.controls_layout.addWidget(self.header_btn)
        self.controls_layout.addWidget(self.refresh_btn)
        self.controls_layout.addWidget(self.new_window_btn)
        self.controls_layout.addWidget(self.instrument_label)
        self.controls_layout.addStretch()

    def _init_plots(self):
        """Initialize matplotlib figures and canvases"""
        # Left slice
        self.slice1_fig = Figure(figsize=(4, 4), dpi=100)
        self.slice1_canvas = FigureCanvas(self.slice1_fig)
        self.slice1_plotter = SlicePlotter(self.slice1_fig)
        self.slice1_toolbar = CustomNavigationToolbar(self.slice1_canvas, self)
        self.slice1_canvas.mpl_connect("button_press_event", lambda e: self._on_slice_click(e, "left"))
        
        # Right slice
        self.slice2_fig = Figure(figsize=(4, 4), dpi=100)
        self.slice2_canvas = FigureCanvas(self.slice2_fig)
        self.slice2_plotter = SlicePlotter(self.slice2_fig)
        self.slice2_toolbar = CustomNavigationToolbar(self.slice2_canvas, self)
        self.slice2_canvas.mpl_connect("button_press_event", lambda e: self._on_slice_click(e, "right"))

        # Spectrum
        self.spec_fig = Figure(figsize=(8, 3), dpi=100)
        self.spec_canvas = FigureCanvas(self.spec_fig)
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
        self.shift_btn = QPushButton("Shift")
        self.apply_shift_btn = QPushButton("Apply Shift")
        
        
        # Gaussian fitting buttons
        self.gauss_left_btn = QPushButton("Left")
        self.gauss_peak_btn = QPushButton("Peak")
        self.gauss_right_btn = QPushButton("Right")
        self.gauss_noise_left_btn = QPushButton("Noise L")
        self.gauss_noise_right_btn = QPushButton("Noise R")
        self.gauss_fit_btn = QPushButton("Fit Gaussian")
        self.gauss_map_btn = QPushButton("Fit Gaussian Map")
        self.multi_gauss_fit_btn = QPushButton("Multi-Gaussian Fit")
        self.co_flux_map_btn = QPushButton("CO Flux Map")
        
        self.auto_continuum_toggle = QPushButton("Automatic Continuum")
        self.custom_continuum_toggle = QPushButton("Custom Continuum")

        self.auto_continuum_toggle.setCheckable(True)
        self.custom_continuum_toggle.setCheckable(True)

        # Style for toggle buttons
        toggle_style = """
            QPushButton {
                background-color: #e0e0e0;
                border: 2px solid #888;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #4a90e2;
                color: white;
                border: 2px solid #2d73c5;
            }
            QPushButton:hover {
                border: 2px solid #4a90e2;
            }
        """
        self.auto_continuum_toggle.setStyleSheet(toggle_style)
        self.custom_continuum_toggle.setStyleSheet(toggle_style)

        # Action buttons (initially hidden)
        self.continuum_line_btn = QPushButton("Continuum Line")
        self.subtract_continuum_btn = QPushButton("Subtract Continuum")
        self.custom_subtract_btn = QPushButton("Custom Subtract")
        self.custom_ok_btn = QPushButton("OK")

        self.save_spectrum_btn = QPushButton("Save Spectrum")

        # Make action buttons checkable
        self.continuum_line_btn.setCheckable(True)
        self.subtract_continuum_btn.setCheckable(True)
        self.custom_subtract_btn.setCheckable(True)

        # Connect toggle buttons
        self.auto_continuum_toggle.clicked.connect(self.on_auto_continuum_toggle)
        self.custom_continuum_toggle.clicked.connect(self.on_custom_continuum_toggle)

        # Connect action buttons (keep existing connections)
        self.continuum_line_btn.clicked.connect(self._on_continuum_line_clicked)
        self.custom_subtract_btn.clicked.connect(self._activate_custom_continuum_subtraction)
        self.custom_ok_btn.clicked.connect(self.on_custom_ok_clicked)

        # Hide all action buttons initially
        self.continuum_line_btn.hide()
        self.subtract_continuum_btn.hide()
        self.custom_subtract_btn.hide()
        self.custom_ok_btn.hide()

        # Tooltips
        self.auto_continuum_toggle.setToolTip("Toggle automatic continuum fitting mode")
        self.custom_continuum_toggle.setToolTip("Toggle custom continuum point selection mode")
        self.continuum_line_btn.setToolTip("Fit and display continuum line")
        self.subtract_continuum_btn.setToolTip("Subtract the fitted continuum")
        self.custom_subtract_btn.setToolTip("Manually select continuum points and fit")
        self.custom_ok_btn.setToolTip("Calculate and apply custom continuum")
        self.save_spectrum_btn.setToolTip("Save current spectrum to file")

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
        self.shift_btn.clicked.connect(self.apply_doppler_shift)
        self.apply_shift_btn.clicked.connect(self._apply_shift)
        self.apply_shift_btn.setEnabled(False)
        self.apply_shift_btn.setToolTip("Replace original spectrum with the shifted version")
        self.gauss_left_btn.clicked.connect(self._activate_gaussian_left)
        self.gauss_peak_btn.clicked.connect(self._activate_gaussian_peak)
        self.gauss_right_btn.clicked.connect(self._activate_gaussian_right)
        self.gauss_noise_left_btn.clicked.connect(self._activate_gaussian_noise_left)
        self.gauss_noise_right_btn.clicked.connect(self._activate_gaussian_noise_right)
        self.gauss_fit_btn.clicked.connect(self._fit_gaussian)
        self.gauss_map_btn.clicked.connect(self._fit_gaussian_map)
        self.multi_gauss_fit_btn.clicked.connect(self._fit_multi_gaussian)
        self.co_flux_map_btn.clicked.connect(self._extract_co_flux_map)
        self.save_spectrum_btn.clicked.connect(self._save_spectrum_ascii)
        self.continuum_line_btn.clicked.connect(self._on_continuum_line_clicked)
        self.subtract_continuum_btn.clicked.connect(self._activate_continuum_subtraction)
        self.custom_subtract_btn.clicked.connect(self._activate_custom_continuum_subtraction)
        
        # Set tooltips for all buttons
        self.set_left_btn.setToolTip("Click on the left slice to set the wavelength from that pixel")
        self.set_right_btn.setToolTip("Click on the right slice to set the wavelength from that pixel")
        self.shift_btn.setToolTip("Apply Doppler shift to the spectrum based on velocity")
        self.gauss_left_btn.setToolTip("Click on spectrum to set left boundary of Gaussian fit")
        self.gauss_peak_btn.setToolTip("Click on spectrum to set peak center of Gaussian fit")
        self.gauss_right_btn.setToolTip("Click on spectrum to set right boundary of Gaussian fit")
        self.gauss_fit_btn.setToolTip("Fit a Gaussian to the selected line")
        self.save_spectrum_btn.setToolTip("Save the current spectrum to an ASCII file")
        self.continuum_line_btn.setToolTip("Toggle continuum subtraction mode")
        self.subtract_continuum_btn.setToolTip("Subtract the fitted continuum from the spectrum")
        self.gauss_map_btn.setToolTip("Fit Gaussian to all pixels using Left/Right boundaries")
        self.multi_gauss_fit_btn.setToolTip("Fit Gaussians to all selected HITRAN lines using current continuum")
        self.gauss_noise_left_btn.setToolTip("Select left boundary of a line-free noise region")
        self.gauss_noise_right_btn.setToolTip("Select right boundary of a line-free noise region")
        self.custom_subtract_btn.setToolTip("Manually select continuum points")
        self.co_flux_map_btn.setToolTip("Fit multiple lines and and creat an integrated flux map")


        # Wavelength range inputs (for line search)
        self.min_wave_input = QLineEdit(self)
        self.min_wave_input.setPlaceholderText("Min λ")
        self.max_wave_input = QLineEdit(self)
        self.max_wave_input.setPlaceholderText("Max λ")
        self.ok_wave_button = QPushButton("OK")
        self.ok_wave_button.clicked.connect(self.apply_wavelength_range)
        
        # HITRAN collapsible group buttons
        self.hitran_btn = QPushButton("HITRAN")
        self.hitran_btn.setToolTip("Fetch HITRAN lines with transition level filtering")
        self.hitran_btn.clicked.connect(self.fetch_hitran_lines)
        
        self.load_contam_btn = QPushButton("Load Contaminating Lines")
        self.load_contam_btn.setToolTip("(Optional) Load CSV file with contaminating lines")
        self.load_contam_btn.clicked.connect(self._load_contaminating_lines_dialog)
        
        self.show_species_btn = QPushButton("SPECIES")
        self.show_species_btn.setToolTip("Show available HITRAN species")
        self.show_species_btn.clicked.connect(self.show_available_species)

    def _init_shape_controls(self):
        """Initialize shape selection buttons"""
        # Helper function to create shape buttons
        def make_shape_button(symbol, target, callback=None):
            btn = QPushButton(symbol)
            btn.setCheckable(True)
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
            "⬭": "Polygon selection",
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

        # Add copy buttons
        self.copy_btn_l = make_shape_button("⎘", "left", 
            callback=lambda: self.copy_selection(source="left", target="right"))
        self.copy_btn_l.setToolTip("Copy selection from left slice to right")
        self.copy_btn_r = make_shape_button("⎘", "right",
            callback=lambda: self.copy_selection(source="right", target="left"))
        self.copy_btn_r.setToolTip("Copy selection from right slice to left")
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
        
        # Group 1: Slice Selection
        self.slice_group = CollapsibleButtonGroup("Slice Selection")
        self.slice_group.add_button(self.set_left_btn)
        self.slice_group.add_button(self.set_right_btn)
        collapsible_layout.addWidget(self.slice_group)
        
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
        
        # Group 4: HITRAN (wavelength inputs + action buttons + CO flux map)
        self.hitran_group = CollapsibleButtonGroup("HITRAN")
        self.hitran_group.add_button(self.min_wave_input)
        self.hitran_group.add_button(self.max_wave_input)
        self.hitran_group.add_button(self.hitran_btn)
        self.hitran_group.add_button(self.load_contam_btn)
        self.hitran_group.add_button(self.multi_gauss_fit_btn)
        self.hitran_group.add_button(self.show_species_btn)
        self.hitran_group.add_button(self.co_flux_map_btn)
        collapsible_layout.addWidget(self.hitran_group)
        
        # Group 5: Shifting
        self.shifting_group = CollapsibleButtonGroup("Shifting")
        self.shifting_group.add_button(self.shift_btn)
        self.shifting_group.add_button(self.apply_shift_btn)
        collapsible_layout.addWidget(self.shifting_group)
        
        collapsible_layout.addStretch()
        
        main_layout.addLayout(collapsible_layout)
        
        # ===================================================================

        # Add spectrum at bottom with expanded vertical space
        main_layout.addWidget(self.spec_canvas, 2)
        main_layout.addWidget(self.spec_toolbar)
        
        # Add save spectrum button separately
        self.continuum_action_layout = QHBoxLayout()
        self.continuum_action_layout.addStretch()
        self.continuum_action_layout.addWidget(self.continuum_line_btn)
        self.continuum_action_layout.addWidget(self.subtract_continuum_btn)
        self.continuum_action_layout.addWidget(self.custom_subtract_btn)
        self.continuum_action_layout.addWidget(self.custom_ok_btn)
        self.continuum_action_layout.addWidget(self.save_spectrum_btn)
        self.continuum_action_layout.addStretch()
        main_layout.addLayout(self.continuum_action_layout)

        self.setLayout(main_layout)


    def _on_continuum_line_clicked(self):
        """Handle Continuum Line button toggle - show parameter dialog then fit and display continuum"""
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            self.continuum_line_btn.setChecked(False)
            return

        # FIXED: Check BOTH possible attribute names
        cube = None
        if hasattr(self, 'cube_data'):
            cube = self.cube_data
        elif hasattr(self, 'cubedata'):
            cube = self.cubedata

        if cube is None or cube.spectrum is None:
            QMessageBox.warning(self, "No Spectrum", "No spectrum data available.")
            self.continuum_line_btn.setChecked(False)
            return

        if self.continuum_line_btn.isChecked():
            # If Subtract Continuum is active, turn it off first and restore original
            if self.subtract_continuum_btn.isChecked():
                self.restore_original_spectrum()
                self.subtract_continuum_btn.setChecked(False)

            # Show parameter dialog
            dialog = ContinuumParametersDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                # Get parameters from dialog
                params = dialog.get_parameters()
                # Fit and display continuum with custom parameters
                self._fit_and_display_continuum(params)
            else:
                # User cancelled, uncheck the button
                self.continuum_line_btn.setChecked(False)
        else:
            # Unchecking - remove continuum line
            if hasattr(self, 'continuum_line') and self.continuum_line is not None:
                try:
                    self.continuum_line.remove()
                    print("[DEBUG] Continuum line visual removed")
                except Exception as e:
                    print(f"[DEBUG] Error removing continuum line: {e}")
                self.continuum_line = None

            # Only remove visual, keep continuum_fitted
            if hasattr(self, 'spec_plotter') and self.spec_plotter is not None:
                self.spec_plotter.ax.legend(loc='upper right', fontsize=8)
                self.spec_canvas.draw()

            # Update Subtract Continuum button state
            if hasattr(self, 'continuum_subtracted') and self.continuum_subtracted:
                if hasattr(self, 'restore_original_spectrum_internal'):
                    self.restore_original_spectrum_internal()
                if hasattr(self, 'subtract_continuum_btn'):
                    self.subtract_continuum_btn.setEnabled(False)
                    self.subtract_continuum_btn.setChecked(False)

    def _fit_and_display_continuum(self, params = None):
        """Fit continuum using peak detection and median filter - improved accuracy"""
        if params is None:
            params = {
                'window_size': 51,
                'sigma_threshold': 2.0,
                'peak_width': 1,
                'peak_distance': 3,
                'peak_exclusion': 3,
                'iterations': 2,
                'smoothing_factor': 5
            }
        try:            
            # Get spectrum data
            wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)

            # Get correct cube data object
            cube = getattr(self, 'cube_data', None) or getattr(self, 'cubedata', None)
            if cube is None:
                QMessageBox.warning(self, "No Data", "Cube data not available.")
                self.continuum_line_btn.setChecked(False)
                return
            flux = cube.spectrum
            
            if wavelength is None or flux is None:
                QMessageBox.warning(self, "No Data", "Spectrum data not available.")
                self.continuum_line_btn.setChecked(False)
                return
            
            # Back up original spectrum for home button to restore
            if self.original_spectrum_backup is None:
                self.original_spectrum_backup = cube.spectrum.copy()
                print("[DEBUG] Backed up original spectrum for continuum fitting")
            
            # Remove NaN and infinite values
            mask = np.isfinite(wavelength) & np.isfinite(flux)
            wavelength_clean = wavelength[mask]
            flux_clean = flux[mask]
            
            if len(wavelength_clean) < 10:
                QMessageBox.warning(self, "Insufficient Data", "Not enough valid data points to fit continuum.")
                self.continuum_line_btn.setChecked(False)
                return
            
            print("[DEBUG] Fitting continuum using peak detection method...")
            
            # STEP 1: Initial median filter to smooth noise
            window_size = params['window_size']  # Must be odd
            flux_smoothed = median_filter(flux_clean, size=window_size, mode='reflect')
            
            # STEP 2: Find peaks (emission lines) in the spectrum
            # Peaks are points significantly above the local baseline
            # prominence: how much a peak stands out
            # width: minimum width of peaks
            # distance: minimum spacing between peaks
            peaks, properties = find_peaks(
                flux_clean,
                prominence=np.std(flux_clean) * params['sigma_threshold'],
                width=params['peak_width'],
                distance=params['peak_distance']
            )

            print(f"[DEBUG] Found {len(peaks)} emission peaks to exclude")
            
            # STEP 3: Create mask to exclude peaks and surrounding region
            # Exclude peaks + 3 pixels on each side (handles wings of emission lines)
            exclude_mask = np.zeros(len(flux_clean), dtype=bool)
            for peak_idx in peaks:
                # Exclude 3 pixels left and right of each peak
                exclusion = params['peak_exclusion']
                left_edge = max(0, peak_idx - exclusion)
                right_edge = min(len(flux_clean), peak_idx + exclusion + 1)  # +4 because of Python slicing
                exclude_mask[left_edge:right_edge] = True
            
            # STEP 4: Fit continuum using only non-peak regions
            # Create a copy where peaks are masked
            flux_for_continuum = flux_clean.copy()
            
            # Interpolate over excluded regions using nearby continuum points
            good_indices = np.where(~exclude_mask)[0]
            if len(good_indices) < 10:
                print("[DEBUG] Too many peaks excluded, falling back to sigma clipping")
                # Fallback to original method if too much excluded
                exclude_mask = np.zeros(len(flux_clean), dtype=bool)
                good_indices = np.arange(len(flux_clean))
            
            # Apply median filter to good (non-peak) regions
            flux_good = flux_clean.copy()
            flux_good[exclude_mask] = np.nan
            
            # Interpolate over NaN (peak) regions
            if np.any(exclude_mask):
                flux_good[exclude_mask] = np.interp(
                    wavelength_clean[exclude_mask],
                    wavelength_clean[~exclude_mask],
                    flux_clean[~exclude_mask]
                )
            
            # Apply median filter to the interpolated data
            continuum = median_filter(flux_good, size=window_size, mode='reflect')
            
            # STEP 5: Iterative sigma clipping for fine-tuning
            # Now clip any remaining outliers (fainter lines missed by peak detection)
            for iteration in range(params['iterations']):  # Reduced to 2 iterations since we already removed peaks
                residuals = flux_clean - continuum
                sigma = np.std(residuals[~exclude_mask])  # Only use non-peak regions for sigma
                
                # Only clip points ABOVE continuum (emission) that weren't already excluded
                outliers = (residuals > 2.0 * sigma) & (~exclude_mask)
                
                if np.sum(outliers) == 0:
                    break  # No more outliers found
                
                # Update mask
                exclude_mask |= outliers
                
                # Re-interpolate and filter
                flux_good = flux_clean.copy()
                flux_good[exclude_mask] = np.interp(
                    wavelength_clean[exclude_mask],
                    wavelength_clean[~exclude_mask],
                    flux_clean[~exclude_mask]
                )
                continuum = median_filter(flux_good, size=window_size, mode='reflect')
            
            # STEP 6: Final smoothing with spline
            try:
                # Fit spline only to non-peak regions for better accuracy
                spline = UnivariateSpline(
                    wavelength_clean[~exclude_mask], 
                    continuum[~exclude_mask], 
                    s=len(wavelength_clean) / params['smoothing_factor'],  # Reduced smoothing factor for better fit
                    k=3
                )
                continuum_smooth = spline(wavelength_clean)
            except Exception as e:
                print(f"[DEBUG] Spline smoothing failed, using median filter result: {e}")
                continuum_smooth = continuum
            
            # Create full continuum array matching original wavelength
            continuum_values = np.full_like(wavelength, np.nan)
            continuum_values[mask] = continuum_smooth
            
            # Store the fitted continuum
            self.continuum_fitted = continuum_values
            
            # Plot the continuum line
            ax = self.spec_plotter.ax
            self.continuum_line = ax.plot(wavelength, continuum_values,
                                        color='orange',
                                        linewidth=2,
                                        linestyle='-',
                                        label='Continuum',
                                        zorder=10)[0]
            
            # Update legend
            ax.legend(loc='upper right', fontsize=8)
            self.spec_canvas.draw()
            
            print(f"[DEBUG] Continuum fitted successfully using peak detection + median filter")
            print(f"[DEBUG] Excluded {np.sum(exclude_mask)} pixels ({100*np.sum(exclude_mask)/len(flux_clean):.1f}%)")
            
        except ImportError as ie:
            QMessageBox.critical(self, "Missing Library",
                            f"Required library missing: {str(ie)}\n\n"
                            "Install with:\n"
                            "pip install scipy")
            self.continuum_line_btn.setChecked(False)
        except Exception as e:
            QMessageBox.critical(self, "Fitting Error",
                            f"Error fitting continuum:\n{str(e)}")
            self.continuum_line_btn.setChecked(False)
            print(f"[DEBUG] Continuum fitting error: {e}")
            traceback.print_exc()
        if hasattr(self, 'subtract_continuum_btn'):
            self.subtract_continuum_btn.setEnabled(True)
            print("[DEBUG] Continuum fitted successfully - Subtract button enabled")
        

    def _remove_continuum_line(self):
        """Remove the continuum line from the spectrum plot"""
        if self.continuum_line is not None:
            try:
                self.continuum_line.remove()
                print("DEBUG: Continuum line removed")
            except Exception as e:
                print(f"DEBUG: Error removing continuum line: {e}")
            self.continuum_line = None
            # DO NOT clear continuum_fitted here - we need it for subtraction!
            # self.continuum_fitted = None  # REMOVED THIS LINE

        # Update legend
        if hasattr(self, 'spec_plotter') and self.spec_plotter is not None:
            self.spec_plotter.ax.legend(loc='upper right', fontsize=8)
            self.spec_canvas.draw()

        # If spectrum was subtracted, restore it
        if self.continuum_subtracted:
            self._restore_original_spectrum_internal()            
            self.subtract_continuum_btn.setChecked(False)

    def _activate_custom_continuum_subtraction(self):
        """
        Activate or deactivate custom continuum subtraction with proper toggle behavior
        """
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            self.custom_subtract_btn.setChecked(False)
            return

        if not hasattr(self, 'cube_data') or self.cube_data.spectrum is None:
            QMessageBox.warning(self, "No Spectrum", "No spectrum data available.")
            self.custom_subtract_btn.setChecked(False)
            return

        if self.custom_subtract_btn.isChecked():
            # ==================== ACTIVATION MODE ====================

            # Check if spectrum is already custom-subtracted
            if self.continuum_subtracted and hasattr(self, 'custom_continuum_applied') and self.custom_continuum_applied:
                # Already subtracted, clicking again should RESTORE
                print("DEBUG: Custom subtract already applied, restoring original")
                self._restore_from_custom_subtract()
                self.custom_subtract_btn.setChecked(False)
                return

            # Check if continuum line exists first (ALWAYS check)
            if not hasattr(self, 'continuum_fitted') or self.continuum_fitted is None:
                QMessageBox.warning(
                    self, 
                    "No Continuum Line Set",
                    "Please create the continuum line first using the 'Continuum Line' button.\n\n"
                    "Steps:\n"
                    "1. Click 'Continuum Line' button to auto-fit the continuum\n"
                    "2. Then click 'Custom Subtract' to refine it with manual points\n\n"
                    "The continuum line will be your starting reference for custom subtraction."
                )
                self.custom_subtract_btn.setChecked(False)
                return

            print("DEBUG: Continuum line detected - activating custom subtraction")

            # If already subtracted via regular subtract, restore first
            if self.continuum_subtracted and self.original_spectrum_backup is not None:
                self.cube_data.spectrum = self.original_spectrum_backup.copy()
                self.continuum_subtracted = False
                print("DEBUG: Restored original spectrum for custom refinement")

            # Turn off Subtract Continuum button if active
            if self.subtract_continuum_btn.isChecked():
                self.subtract_continuum_btn.setChecked(False)

            # DON'T turn off Continuum Line button - keep it visible during point selection!
            # This is KEY: user needs to see the continuum while placing points

            # Back up original spectrum for home button to restore
            if self.original_spectrum_backup is None:
                self.original_spectrum_backup = self.cube_data.spectrum.copy()
                print("[DEBUG] Backed up original spectrum for custom continuum fitting")

            # Create or reset selector with current spectrum data
            # FIXED: Disable Enter key by NOT passing on_finalize callback
            if self.custom_continuum_selector is None:
                self.custom_continuum_selector = CustomContinuumSelector(
                    ax=self.spec_plotter.ax,
                    canvas=self.spec_canvas,
                    wavelength=_extract_wavelength_value(self.spec_plotter.wavelength),
                    flux=self.cube_data.spectrum,
                    parent_widget=self
                    # NOTE: NOT passing on_finalize parameter - Enter key disabled
                )
            else:
                # Update with current spectrum data (important after zoom/pan)
                self.custom_continuum_selector.flux = self.cube_data.spectrum
                self.custom_continuum_selector.wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)

            # Activate it (WITHOUT Enter key functionality)
            self.custom_continuum_selector.activate()
            self.custom_continuum_active = True
            print("DEBUG: Custom continuum selector activated - click on the blue spectrum to add points")
            print("DEBUG: Points will SNAP to nearest spectrum data point")
            print("DEBUG: Recommended: ~20 points for best results")
            print("DEBUG: Press OK button (NOT Enter) to fit and subtract, ESC to cancel")

        else:
            # ==================== DEACTIVATION MODE ====================

            # Check if custom subtract was applied
            if self.continuum_subtracted and hasattr(self, 'custom_continuum_applied') and self.custom_continuum_applied:
                # Restore original spectrum
                print("DEBUG: Deactivating - restoring original spectrum")
                self._restore_from_custom_subtract()
            else:
                # Just deactivate selector without restoration
                if self.custom_continuum_selector is not None:
                    self.custom_continuum_selector.deactivate()
                self.custom_continuum_active = False
                print("DEBUG: Custom continuum subtraction deactivated")

    def _restore_from_custom_subtract(self):
        """
        Restore the original spectrum from custom subtract
        """
        if self.original_spectrum_backup is None:
            print("DEBUG: No backup to restore")
            return

        try:
            # Restore original spectrum
            self.cube_data.spectrum = self.original_spectrum_backup.copy()

            # Replot
            self.spec_plotter.plot_spectrum(self.cube_data.spectrum)

            # Restore continuum line if continuum_fitted exists and button should be on
            if self.continuum_fitted is not None and self.continuum_line_btn.isChecked():
                wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
                ax = self.spec_plotter.ax
                self.continuum_line = ax.plot(
                    wavelength, 
                    self.continuum_fitted,
                    color='orange',
                    linewidth=2,
                    linestyle='-',
                    label='Continuum',
                    zorder=10
                )[0]

            # Update legend and redraw
            self.spec_plotter.ax.legend(loc='upper right', fontsize=8)
            self.spec_canvas.draw()

            # Reset flags
            self.continuum_subtracted = False
            self.custom_continuum_applied = False

            print("DEBUG: Original spectrum restored from custom subtract")

            QMessageBox.information(
                self,
                "Restored",
                "Original spectrum has been restored."
            )

        except Exception as e:
            print(f"DEBUG: Error restoring spectrum: {e}")



    def _perform_global_continuum_subtraction(self):
        """
        Subtract the fitted continuum from the entire spectrum with robust scaling.
        """
        try:
            # Backup original spectrum if not already done
            if self.original_spectrum_backup is None:
                self.original_spectrum_backup = self.cube_data.spectrum.copy()
                print("DEBUG: Backed up original spectrum")

            # Extract numeric values safely
            wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
            flux = self.original_spectrum_backup.copy()
            continuum = self.continuum_fitted

            # Check for NoneType before accessing .shape
            if continuum is None:
                QMessageBox.warning(self, "Error", "No continuum data found. Fit a continuum first.")
                self.subtract_continuum_btn.setChecked(False)
                return

            print(f"DEBUG: flux shape {flux.shape}, continuum shape {continuum.shape}")

            if flux.shape != continuum.shape:
                QMessageBox.critical(
                    self, 
                    "Shape Mismatch",
                    f"Cannot subtract: flux shape {flux.shape} != continuum shape {continuum.shape}.\n\n"
                    f"Please refit the continuum."
                )
                self.subtract_continuum_btn.setChecked(False)
                return

            # Perform subtraction
            flux_subtracted = flux - continuum
            self.cube_data.spectrum = flux_subtracted

            # Replot the spectrum
            ax = self.spec_plotter.ax
            ax.clear()

            # Plot subtracted spectrum
            ax.plot(wavelength, flux_subtracted, 'b-', label='Continuum-Subtracted Spectrum', 
                    linewidth=1, clip_on=True)

            # *** ADD ZERO REFERENCE LINE (light grey dotted) ***
            ax.axhline(0, color='lightgrey', linestyle=':', linewidth=1.5, 
                    alpha=0.8, label='Zero Level', zorder=3)

            # Calculate robust y-limits using percentiles
            valid_flux = flux_subtracted[np.isfinite(flux_subtracted)]
            if len(valid_flux) > 0:
                y_min = np.percentile(valid_flux, 1)
                y_max = np.percentile(valid_flux, 99)
                margin = (y_max - y_min) * 0.10
                ax.set_ylim(y_min - margin, y_max + margin)

            # Set limits
            ax.set_xlim(wavelength.min(), wavelength.max())

            # Labels
            wave_unit = getattr(self.cube_data, 'wave_unit', 'µm')
            flux_unit = getattr(self.cube_data, 'flux_unit', 'MJy/sr')
            ax.set_xlabel(f'Wavelength ({wave_unit})')
            ax.set_ylabel(f'Flux ({flux_unit})')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Force clipping
            for line in ax.get_lines():
                line.set_clip_on(True)

            # Redraw
            self.spec_canvas.draw()
            QApplication.processEvents()

            # Set flags
            self.continuum_subtracted = True
            self.global_continuum_applied = True  # NEW FLAG
            self.auto_continuum_subtracted = True  # Flag for CO Flux Map check

            print("DEBUG: Global continuum subtraction complete")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Subtraction failed: {str(e)}")
            self.subtract_continuum_btn.setChecked(False)
            traceback.print_exc()

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

    def _perform_custom_subtraction(self, continuum):
        """
        Subtract custom fitted continuum from spectrum and display with zero line
        """
        try:
            # Backup original if not already done
            if self.original_spectrum_backup is None:
                self.original_spectrum_backup = self.cube_data.spectrum.copy()
                print("DEBUG: Backed up original spectrum")

            # Perform subtraction
            subtracted = self.original_spectrum_backup - continuum

            # Update the spectrum
            self.cube_data.spectrum = subtracted

            # Get wavelength for plotting
            wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)

            # Replot with custom styling
            ax = self.spec_plotter.ax
            ax.clear()

            # Plot subtracted spectrum
            ax.plot(wavelength, subtracted, color='blue', linewidth=1.5, 
                    label='Custom Continuum-Subtracted Spectrum', clip_on=True, zorder=5, antialiased=True)

            # *** ADD ZERO REFERENCE LINE (light grey dotted) ***
            ax.axhline(0, color='lightgrey', linestyle=':', linewidth=1.5, 
                    alpha=0.8, label='Zero Level', zorder=3)

            # Calculate robust y-limits
            valid_flux = subtracted[np.isfinite(subtracted)]
            if len(valid_flux) > 0:
                y_min = np.percentile(valid_flux, 1)
                y_max = np.percentile(valid_flux, 99)
                margin = (y_max - y_min) * 0.10
                ax.set_ylim(y_min - margin, y_max + margin)

            # Set x-limits
            ax.set_xlim(wavelength.min(), wavelength.max())

            # Labels
            wave_unit = getattr(self.cube_data, 'wave_unit', 'µm')
            flux_unit = getattr(self.cube_data, 'flux_unit', 'MJy/sr')
            ax.set_xlabel(f'Wavelength ({wave_unit})', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'Relative Flux ({flux_unit})', fontsize=11, fontweight='bold')
            ax.set_title('Custom Continuum Subtraction Result', fontsize=12, fontweight='bold', pad=10)

            # Grid and legend
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.legend(loc='upper right', fontsize=8)

            # Force clipping
            for line in ax.get_lines():
                line.set_clip_on(True)

            # Redraw
            self.spec_canvas.draw()

            # Mark as subtracted with custom continuum
            self.continuum_subtracted = True
            self.custom_continuum_applied = True  # NEW FLAG

            print("DEBUG: Custom continuum subtraction complete")
            print(f"DEBUG: Subtracted flux range: {np.nanmin(subtracted):.3e} to {np.nanmax(subtracted):.3e}")

            QMessageBox.information(
                self,
                "Subtraction Complete",
                f"Custom continuum subtracted successfully!\n\n"
                f"Result: I_subtracted(λ) = I_observed(λ) - I_continuum(λ)\n\n"
                f"Click 'Custom Subtract' again to restore original spectrum."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Subtraction Error",
                f"Error subtracting continuum:\n{str(e)}"
            )
            print(f"DEBUG: Subtraction error: {e}")
            traceback.print_exc()


    def _restore_original_spectrum(self):
        """
        Restore the original spectrum before continuum subtraction
        """
        if self.original_spectrum_backup is None:
            print("DEBUG: No backup to restore")
            return

        try:
            # Restore original spectrum
            self.cube_data.spectrum = self.original_spectrum_backup.copy()

            # Replot
            self.spec_plotter.plot_spectrum(self.cube_data.spectrum)

            # Restore continuum line if continuum_fitted exists and button should be on
            if self.continuum_fitted is not None and self.continuum_line_btn.isChecked():
                wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
                ax = self.spec_plotter.ax
                self.continuum_line = ax.plot(
                    wavelength, 
                    self.continuum_fitted,
                    color='orange',
                    linewidth=2,
                    linestyle='-',
                    label='Continuum',
                    zorder=10
                )[0]

            # Remove continuum line visual if button is not checked
            if self.continuum_line is not None and not self.continuum_line_btn.isChecked():
                try:
                    self.continuum_line.remove()
                    print("DEBUG: Removed continuum line visual during restore")
                except Exception as e:
                    print(f"DEBUG: Error removing continuum line: {e}")
                self.continuum_line = None

            # Update legend and redraw
            self.spec_plotter.ax.legend(loc='upper right', fontsize=8)
            self.spec_canvas.draw()

            # Reset flags
            self.continuum_subtracted = False
            self.global_continuum_applied = False  # Reset flag

            print("DEBUG: Original spectrum restored")

        except Exception as e:
            print(f"DEBUG: Error restoring spectrum: {e}")

    def _init_state(self):
        """Initialize state variables"""
        self.active_shape = None
        self.active_target = None
        self.last_idx_left = 0
        self.last_idx_right = 0
        self.last_selection_left = None
        self.last_selection_right = None
        self.hitran_lines = []  # Store HITRAN lines separately
        self.plotted_lines = []  # Store currently plotted spectral lines
        self.plotted_artists = []  # Store matplotlib artist objects for plotted lines
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
        
        # Doppler shift state
        self.shifted_line = None  # Store shifted spectrum line artist
        self.shifted_spectrum_data = None  # Store shifted spectrum data for application
        self.shifted_wavelength_data = None  # Store shifted wavelength data for application
        self.shifted_velocity = None  # Store the velocity used for shifting
        self.is_spectrum_shifted = False  # Track if current spectrum is a shifted version

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
            except Exception as plot_error:
                QMessageBox.warning(self, "Plot Error", f"Error plotting spectrum: {str(plot_error)}")
            
            try:
                # Display initial slices
                self._display_slice(0, "left")
                self._display_slice(0, "right")
            except Exception as slice_error:
                QMessageBox.warning(self, "Display Error", f"Error displaying slices: {str(slice_error)}")

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

    def apply_wavelength_range(self):
        """Triggered when user confirms wavelength range (OK button - currently not visible)"""
        # Read min/max wavelength
        try:
            min_wave = float(self.min_wave_input.text())
            max_wave = float(self.max_wave_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Enter valid wavelength numbers.")
            return

        if min_wave >= max_wave:
            QMessageBox.warning(self, "Invalid Range", "Min wavelength must be less than max wavelength.")
            return

        # Confirm range is set
        QMessageBox.information(
            self, "Range Set", 
            f"Wavelength range set to {min_wave:.3f} - {max_wave:.3f} µm\n\n"
            "Use HITRAN button to fetch spectral lines."
        )


    def on_span_selected(self, xmin, xmax):
        self.min_wave_input.setText(f"{xmin:.3f}")
        self.max_wave_input.setText(f"{xmax:.3f}")

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
                title = f"{target.capitalize()} Slice (idx={idx}) @ {wave:.1f} Å"
            elif self.cube_data.wave_unit == "µm":
                title = f"{target.capitalize()} Slice (idx={idx}) @ {wave:.3f} µm"
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
        else:
            selector.activate_selector(None)

    def _handle_region_select(self, mask, patch, coords, target):
        """Handle region selection completion"""
        # Store selection (coords are xmin,xmax,ymin,ymax)
        xmin, xmax, ymin, ymax = coords

        # Determine which slice index and image to use
        idx = self.last_idx_left if target == "left" else self.last_idx_right
        image = self.cube_data.get_slice(idx)
        if image is None:
            return

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

        # Save selection with full mask and label
        selection = {
            "shape": self.active_shape,
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

        # Remove all handle scatter collections so they don't linger
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
        selector = self.region_selector_left if target == "left" else self.region_selector_right
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
            unit_str = f" [{unit}]" if unit else ""
            
            # Create region stats with sum and std
            region_stats = [
                f"Selected Area Stats{unit_str}:",
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

        # Compute and plot region spectrum on top of original
        region_spectrum = self.cube_data.compute_region_spectrum(full_mask)
        if region_spectrum is not None and hasattr(self, 'spec_plotter') and self.spec_plotter is not None:
            # Multiply mean by n_pixels to get summed spectrum comparable to the main spectrum
            try:
                region_sum = region_spectrum * float(n_pixels) if n_pixels is not None else region_spectrum
            except Exception:
                region_sum = region_spectrum

            self.spec_plotter.plot_region_spectrum(
                region_sum,
                color='cyan' if target == "left" else 'magenta',
                label=selection['label']
            )
            try:
                # show a quick sample of values for debugging
                first_vals = region_spectrum[:8] if len(region_spectrum) >= 8 else region_spectrum
                print(f"REGION SPECTRUM [{target}] n_pixels={n_pixels} first_vals={first_vals[:8]}")
            except Exception:
                pass
            try:
                self.spec_canvas.draw()
            except Exception as e:
                print(f"[DEBUG] Error updating spectrum canvas: {e}")
        
        # Ensure the slice canvas is redrawn to show handles (use draw() for live updates)
        try:
            if target == "left":
                self.slice1_canvas.draw()
            else:
                self.slice2_canvas.draw()
        except Exception as e:
            print(f"[DEBUG] Error updating slice canvas: {e}")

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
        if target == "left":
            sel = self.last_selection_left
            plotter = self.slice1_plotter
        else:
            sel = self.last_selection_right
            plotter = self.slice2_plotter

        if not sel:
            return

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

        # Remove region spectrum line from the spectrum plotter (side-specific label)
        try:
            label = sel.get('label') if isinstance(sel, dict) else None
            if label and hasattr(self, 'spec_plotter') and label in getattr(self.spec_plotter, 'region_lines', {}):
                try:
                    self.spec_plotter.region_lines[label].remove()
                except Exception:
                    pass
                try:
                    del self.spec_plotter.region_lines[label]
                except Exception:
                    pass
        except Exception:
            pass

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
                if target == "left":
                    self.last_selection_left = {"shape": "*", "coords": (x, y), "mask": None}
                else:
                    self.last_selection_right = {"shape": "*", "coords": (x, y), "mask": None}

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
                        stats_text.append(f"λ={wave:.2f} Å")
                    elif self.cube_data.wave_unit == "µm":
                        stats_text.append(f"λ={wave:.3f} µm")
                    elif self.cube_data.wave_unit == "km/s":
                        stats_text.append(f"v={wave:.1f} km/s")
                    else:
                        stats_text.append(f"λ={wave:.3f}")

                stats_label.setText("\n".join(stats_text))

                # When clicking a pixel remove any existing region patches (exclusive selection)
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
                
                # Compute and plot single-pixel spectrum (fast path)
                region_spectrum = self.cube_data.get_pixel_spectrum(x, y)
                if region_spectrum is not None and hasattr(self, 'spec_plotter') and self.spec_plotter is not None:
                    # Remove any existing 'Selection' region line (visualizer will handle canonical label)
                    try:
                        # remove any previous region line for this side
                        prev_label = f'{target.capitalize()} Selection'
                        if prev_label in getattr(self.spec_plotter, 'region_lines', {}):
                            self.spec_plotter.region_lines[prev_label].remove()
                            del self.spec_plotter.region_lines[prev_label]
                    except Exception:
                        pass

                    self.spec_plotter.plot_region_spectrum(
                        region_spectrum,
                        color='cyan' if target == "left" else 'magenta',
                        label=f'{target.capitalize()} Selection'
                    )
                    try:
                        self.spec_canvas.draw()
                    except Exception:
                        pass
                    
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
            hover_threshold = 0.01  # wavelength units tolerance
            mouse_x = event.xdata
            if mouse_x is not None:
                # Find nearest line within threshold
                nearest_row = None
                min_dist = float('inf')
                
                for row in self.plotted_lines:
                    from .species_dialog import SpeciesDialog
                    λ = SpeciesDialog.get_row_wavelength(row)
                    dist = abs(λ - mouse_x)
                    if dist < hover_threshold and dist < min_dist:
                        min_dist = dist
                        nearest_row = row
                
                if nearest_row:
                    # Found a line to deselect
                    line_wave = SpeciesDialog.get_row_wavelength(nearest_row)
                    species = nearest_row.get('species', 'Unknown')
                    print(f"✓ Deselected: {species} at {line_wave:.6f} μm")
                    
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

        # Store the index
        if self.active_target == "left":
            self.last_idx_left = idx
        else:
            self.last_idx_right = idx

        # Update the vertical line in the spectrum plot
        self.spec_plotter.update_slice_indicator(idx, self.active_target)
        try:
            self.spec_canvas.draw()
        except Exception:
            pass

        # Display the new slice
        self._display_slice(idx, self.active_target)
        
        # Update any active region selections
        if self.active_target == "left" and self.last_selection_left:
            self.left_region_spectrum = self.cube_data.compute_region_spectrum(
                self.last_selection_left["mask"])
            if self.last_selection_left["shape"] == "*":
                x, y = self.last_selection_left["coords"]
                self._handle_pixel_click(x, y, "left")
            else:
                xmin, xmax, ymin, ymax = self.last_selection_left["coords"]
                mask = self.last_selection_left["mask"]
                self._recreate_region(xmin, xmax, ymin, ymax, mask, "left")
                
        elif self.active_target == "right" and self.last_selection_right:
            self.right_region_spectrum = self.cube_data.compute_region_spectrum(
                self.last_selection_right["mask"])
            if self.last_selection_right["shape"] == "*":
                x, y = self.last_selection_right["coords"]
                self._handle_pixel_click(x, y, "right")
            else:
                xmin, xmax, ymin, ymax = self.last_selection_right["coords"]
                mask = self.last_selection_right["mask"]
                self._recreate_region(xmin, xmax, ymin, ymax, mask, "right")

    def _select_target(self, target):
        """Handle slice selection mode"""
        if target == "left":
            self.set_right_btn.setChecked(False)
            self.active_target = "left" if self.set_left_btn.isChecked() else None
        else:
            self.set_left_btn.setChecked(False)
            self.active_target = "right" if self.set_right_btn.isChecked() else None

    def _recreate_region(self, xmin, xmax, ymin, ymax, mask, target):
        """Recreate a region selection after changing slices"""
        if not hasattr(self, 'active_shape'):
            return
            
        # Store current shape
        current_shape = self.active_shape
        
        # Get correct selector and plotter
        selector = self.region_selector_left if target == "left" else self.region_selector_right
        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
        
        # Create appropriate shape based on the original selection type
        if selector.active_shape in ["◯", "⬭"]:  # Circle or ellipse
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            if selector.active_shape == "◯":  # Circle
                r = min((xmax - xmin)/2, (ymax - ymin)/2)
                patch = patches.Circle((cx, cy), r, edgecolor='cyan', facecolor='none', lw=1.5)
            else:  # Ellipse
                a, b = (xmax - xmin)/2, (ymax - ymin)/2
                patch = patches.Ellipse((cx, cy), 2*a, 2*b, edgecolor='cyan', facecolor='none', lw=1.5)
        else:  # Rectangle or default
            patch = patches.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin,
                edgecolor='cyan', facecolor='none', lw=1.5
            )
        
        # Clear old patches
        if hasattr(plotter.ax, 'area_artists'):
            for artist in plotter.ax.area_artists:
                artist.remove()
            plotter.ax.area_artists = []

        # Remove all handle scatter collections
        for collection in list(plotter.ax.collections):
            try:
                if hasattr(collection, '_is_handle_collection') and collection._is_handle_collection:
                    collection.remove()
            except Exception:
                pass
        
        # Add new patch
        plotter.ax.add_patch(patch)
        if not hasattr(plotter.ax, 'area_artists'):
            plotter.ax.area_artists = []
        plotter.ax.area_artists.append(patch)
        try:
            if target == "left":
                self.slice1_canvas.draw()
            else:
                self.slice2_canvas.draw()
        except Exception:
            pass
        
        # Update the spectrum for the region
        region_spectrum = self.cube_data.compute_region_spectrum(mask)
        if region_spectrum is not None:
            # visualization will plot this region with a side-specific label
            self.spec_plotter.plot_region_spectrum(
                region_spectrum,
                color='cyan' if target == "left" else 'magenta',
                label=f'{target.capitalize()} Selection'
            )
            try:
                self.spec_canvas.draw()
            except Exception:
                pass



    def apply_doppler_shift(self):
        """Apply multiplicative Doppler shift to spectrum based on user-provided velocity.
        
        Uses formula: lambda_0 = lambda / (1 + v/c)
        This scales the entire wavelength array uniformly.
        """
        
        # Check if spectrum is loaded
        if not hasattr(self, 'original_spectrum') or self.original_spectrum is None:
            QMessageBox.warning(self, "No Spectrum", "Load a data cube first.")
            return
        
        # Check if wavelength data is available
        wavelength_values = _extract_wavelength_value(self.spec_plotter.wavelength)
        if wavelength_values is None or len(wavelength_values) == 0:
            QMessageBox.warning(self, "No Wavelength Data", 
                              "Wavelength data is not available for this cube.")
            return
        
        # Ask user for velocity in km/s
        velocity, ok = QInputDialog.getDouble(
            self, 
            "Doppler Shift (Multiplicative)",
            "Enter systemic velocity (km/s):",
            value=0.0,
            decimals=2
        )
        
        if not ok:
            return
        
        # Constants
        c = 299792.458  # Speed of light in km/s
        
        # Get current wavelength and spectrum
        wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
        
        # Use current spectrum
        current_spectrum = self.original_spectrum
        
        # MULTIPLICATIVE METHOD: Global wavelength scaling
        # lambda_0 = lambda / (1 + v/c)
        shifted_wavelength = wavelength / (1 + velocity / c)
        
        # Use current spectrum (shifted spectrum will be at shifted wavelength positions)
        shifted_spectrum = current_spectrum
        
        # Remove any existing shifted spectrum line
        if hasattr(self, 'shifted_line') and self.shifted_line is not None:
            try:
                self.shifted_line.remove()
            except:
                pass
        
        # Store the shifted spectrum data and wavelength for later application
        self.shifted_spectrum_data = shifted_spectrum.copy()
        self.shifted_wavelength_data = shifted_wavelength.copy()
        self.shifted_velocity = velocity
        # Reset the shifted flag since this is a preview
        self.is_spectrum_shifted = False
        
        # Get current y-axis ticks and limits before plotting
        y_ticks = self.spec_plotter.ax.get_yticks()
        y_limits = self.spec_plotter.ax.get_ylim()
        
        # Plot shifted spectrum in purple at shifted wavelength positions
        self.shifted_line, = self.spec_plotter.ax.plot(
            shifted_wavelength, 
            shifted_spectrum,
            'purple', 
            linewidth=1.5, 
            alpha=0.8, 
            label=f'Shifted (v={velocity:.1f} km/s)',
            zorder=4
        )
        
        # Explicitly set y-axis ticks and limits to original values
        self.spec_plotter.ax.set_yticks(y_ticks)
        self.spec_plotter.ax.set_ylim(y_limits)
        
        # Enable the Apply Shift button
        self.apply_shift_btn.setEnabled(True)
        
        # Update legend and redraw
        self.spec_plotter.ax.legend(loc='upper right', fontsize=8)
        self.spec_canvas.draw()
        
        print(f"[DEBUG] Applied multiplicative Doppler shift: v={velocity} km/s")
    
    def _apply_shift(self):
        """Apply the shifted spectrum as the new original spectrum.
        
        This replaces the original spectrum with the shifted one, allowing
        the user to fit Gaussians to the shifted spectrum without the original
        overlay. Preserves y-axis limits and changes spectrum color to purple.
        """
        
        # Check if a shift has been applied
        if not hasattr(self, 'shifted_spectrum_data') or self.shifted_spectrum_data is None:
            QMessageBox.warning(
                self, 
                "No Shift Applied",
                "Please apply a Doppler shift first using the 'Shift' button."
            )
            return
        
        # Confirm with user
        if self.shifted_velocity is not None:
            reply = QMessageBox.question(
                self,
                "Apply Shift",
                f"Replace original spectrum with shifted version (v={self.shifted_velocity:.1f} km/s)?\n\n"
                "The original spectrum will be replaced and the shift will be permanent.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        # Preserve current axis limits and ticks before updating spectrum
        y_limits = self.spec_plotter.ax.get_ylim()
        y_ticks = self.spec_plotter.ax.get_yticks()
        x_limits = self.spec_plotter.ax.get_xlim()
        x_ticks = self.spec_plotter.ax.get_xticks()
        velocity = self.shifted_velocity  # Store velocity before clearing it
        
        # Replace original spectrum with shifted spectrum
        self.original_spectrum = self.shifted_spectrum_data.copy()
        
        # Also update the wavelength array to the shifted wavelength
        if hasattr(self, 'shifted_wavelength_data') and self.shifted_wavelength_data is not None:
            self.cube_data.wavelength = self.shifted_wavelength_data.copy()
            self.spec_plotter.wavelength = self.shifted_wavelength_data.copy()
        
        # Remove the shifted spectrum line from the plot
        if hasattr(self, 'shifted_line') and self.shifted_line is not None:
            try:
                self.shifted_line.remove()
            except:
                pass
            self.shifted_line = None
        
        # Clear shifted spectrum data
        self.shifted_spectrum_data = None
        self.shifted_wavelength_data = None
        self.shifted_velocity = None
        
        # Mark that current spectrum is shifted
        self.is_spectrum_shifted = True
        
        # Disable the Apply Shift button
        self.apply_shift_btn.setEnabled(False)
        
        # Update the spectrum display (replot the new original)
        self.spec_plotter.update_spectrum(self.original_spectrum)
        
        # Adjust x-limits to account for the wavelength shift
        # Since wavelength was shifted by: wavelength / (1 + v/c)
        # We need to apply the same shift to the x-axis limits
        c = 299792.458  # Speed of light in km/s
        shift_factor = 1 / (1 + velocity / c)
        adjusted_x_min = x_limits[0] * shift_factor
        adjusted_x_max = x_limits[1] * shift_factor
        
        # Restore axis limits and ticks after update
        self.spec_plotter.ax.set_ylim(y_limits)
        self.spec_plotter.ax.set_yticks(y_ticks)
        self.spec_plotter.ax.set_xlim(adjusted_x_min, adjusted_x_max)
        self.spec_plotter.ax.set_xticks(x_ticks)
        
        # Shift all plot elements (Gaussian fit, slice indicators) by the same factor
        self.spec_plotter.shift_plot_elements(shift_factor)
        
        # Change the spectrum line color to purple to indicate it's shifted
        if hasattr(self.spec_plotter, 'global_line') and self.spec_plotter.global_line is not None:
            self.spec_plotter.global_line.set_color('purple')
        
        # Update legend label to indicate this is a shifted spectrum
        if hasattr(self.spec_plotter, 'global_line') and self.spec_plotter.global_line is not None:
            self.spec_plotter.global_line.set_label('Shifted Spectrum')
        
        # Update legend and redraw
        self.spec_plotter.ax.legend(loc='upper right', fontsize=8)
        self.spec_canvas.draw()
        
        QMessageBox.information(
            self,
            "Shift Applied",
            "Original spectrum has been replaced with the shifted version.\n"
            "You can now fit Gaussians to the shifted spectrum."
        )
        
        print(f"[DEBUG] Shift applied - original spectrum replaced with shifted spectrum")
    
    def _find_nearest_line_in_range(self, lam_min, lam_max):
        """Find the nearest spectral line within a wavelength range.
        
        Returns:
            float: Wavelength of the nearest line in µm, or None if not found
        """
        # Optional: Can return a HITRAN line if available, but not required for Gaussian fitting
        if not hasattr(self, 'hitran_lines') or not self.hitran_lines:
            return None

        mid = (lam_min + lam_max) / 2.0
        line_wavelengths = []
        for row in self.hitran_lines:
            try:
                if 'lambda_micron' in row:
                    lam = float(row['lambda_micron'])
                elif 'wavelength' in row:
                    lam = float(row['wavelength'])
                else:
                    continue
                if lam_min <= lam <= lam_max:
                    line_wavelengths.append(lam)
            except (ValueError, KeyError):
                continue

        if not line_wavelengths:
            return None

        line_wavelengths = np.array(line_wavelengths)
        closest_idx = np.argmin(np.abs(line_wavelengths - mid))
        return float(line_wavelengths[closest_idx])

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

    def _activate_continuum_subtraction(self):
        """
        Toggle continuum subtraction for the entire spectrum (AUTOMATIC - no point selection)
        Now works as a proper toggle button with continuum line check every time
        """
        print("\n" + "="*80)
        print("[DEBUG] _activate_continuum_subtraction CALLED")
        print("="*80)

        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            print("[DEBUG] No spec_plotter")
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            self.subtract_continuum_btn.setChecked(False)
            return

        if not hasattr(self, 'cube_data') or self.cube_data.spectrum is None:
            print("[DEBUG] No cube_data or spectrum")
            QMessageBox.warning(self, "No Spectrum", "No spectrum data available.")
            self.subtract_continuum_btn.setChecked(False)
            return

        print(f"[DEBUG] Button isChecked: {self.subtract_continuum_btn.isChecked()}")

        if self.subtract_continuum_btn.isChecked():
            print("[DEBUG] ACTIVATION MODE - Attempting subtraction")

            # CRITICAL: ALWAYS check if continuum line exists first (EVERY TIME)
            print(f"[DEBUG] hasattr continuum_fitted: {hasattr(self, 'continuum_fitted')}")
            if hasattr(self, 'continuum_fitted'):
                print(f"[DEBUG] continuum_fitted is None: {self.continuum_fitted is None}")
                if self.continuum_fitted is not None:
                    print(f"[DEBUG] continuum_fitted shape: {self.continuum_fitted.shape}")

            if not hasattr(self, 'continuum_fitted') or self.continuum_fitted is None:
                print("[DEBUG] No continuum fitted - showing warning")
                QMessageBox.warning(
                    self,
                    "No Continuum Line Set",
                    "Please create the continuum line first using the 'Continuum Line' button.\n\n"
                    "Steps:\n"
                    "1. Click 'Continuum Line' button to fit the continuum\n"
                    "2. Then click 'Subtract Continuum' to subtract it\n\n"
                    "The continuum must be fitted before subtraction."
                )
                self.subtract_continuum_btn.setChecked(False)
                return

            print("[DEBUG] Continuum line found - proceeding with subtraction")

            # Check if already subtracted (toggle to restore)
            print(f"[DEBUG] continuum_subtracted: {hasattr(self, 'continuum_subtracted') and self.continuum_subtracted}")
            if self.continuum_subtracted and hasattr(self, 'global_continuum_applied') and self.global_continuum_applied:
                print("[DEBUG] Already subtracted, restoring original")
                self._restore_original_spectrum()
                self.subtract_continuum_btn.setChecked(False)
                return

            # Check if custom subtract is active or applied
            if self.custom_subtract_btn.isChecked() or (hasattr(self, 'custom_continuum_applied') and self.custom_continuum_applied):
                print("[DEBUG] Custom subtract active - asking user")
                reply = QMessageBox.question(
                    self,
                    "Custom Subtract Active",
                    "Using automatic subtraction will remove your custom subtraction progress.\n\n"
                    "Are you sure you want to continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    print("[DEBUG] User cancelled")
                    self.subtract_continuum_btn.setChecked(False)
                    return
                else:
                    # User confirmed, restore and proceed
                    if hasattr(self, 'custom_continuum_applied') and self.custom_continuum_applied:
                        self._restore_from_custom_subtract()
                    self.custom_subtract_btn.setChecked(False)

            # DON'T turn off the continuum line button - let it stay visible
            print("[DEBUG] About to call _perform_global_continuum_subtraction()")

            # Perform the automatic global subtraction
            self._perform_global_continuum_subtraction()

            print("[DEBUG] Returned from _perform_global_continuum_subtraction()")
            print(f"[DEBUG] continuum_subtracted should now be True")

            # Button stays checked after subtraction
            self.subtract_continuum_btn.setChecked(True)
            print("[DEBUG] Subtraction complete")

        else:
            # DEACTIVATION MODE (RESTORE)
            print("[DEBUG] DEACTIVATION MODE - Restoring original")
            self._restore_original_spectrum()

        print("="*80)
        print("[DEBUG] _activate_continuum_subtraction FINISHED")
        print("="*80 + "\n")

    def _init_line_search_controls(self):
        """Initialize the min/max inputs and buttons (for legacy support)."""
        # Note: These are also created in _init_plots and used in the HITRAN collapsible group
        # This method is kept for compatibility but the layout is not currently used
        pass

    def show_available_species(self):
        """
        Show available HITRAN species.
        This is the main species viewing button.
        """
        # Check if HITRAN data exists
        if not hasattr(self, 'hitran_lines') or not self.hitran_lines:
            QMessageBox.warning(self, "No HITRAN Data", "Click the HITRAN button first to fetch HITRAN lines.")
            return
        
        # Use only HITRAN lines
        all_lines = list(self.hitran_lines)
        
        if not all_lines:
            QMessageBox.information(self, "No Data", "No spectral lines available.")
            return
        
        # Sort by wavelength
        all_lines.sort(key=lambda r: float(r.get('lambda_micron') or r.get('wavelength') or 0.0))
        
        # Show selection dialog
        dlg = SpeciesDialog(all_lines, parent=self, plotted_rows=self.plotted_lines)
        result = dlg.exec_()
        
        if result == QDialog.Accepted:
            selected_rows = dlg.get_selected_rows()
            if not selected_rows:
                QMessageBox.information(self, "No Selection", "No lines selected.")
                return
            self.plot_selected_lines(selected_rows)

    def on_species_button_clicked(self):
        species = self.species_combo.currentText()
        lam_min = float(self.min_edit.text())
        lam_max = float(self.max_edit.text())

        # Get filtered rows from catalog
        lines = self.catalog.filter_species_wavelength(species, lam_min, lam_max)

        # Create dialog, passing currently plotted lines for checkbox sync
        dlg = SpeciesDialog(lines, parent=self, plotted_rows=self.plotted_lines)
        result = dlg.exec_()

        if result == QDialog.Accepted:
            selected_rows = dlg.get_selected_rows()
            
            # Update the spectrum plot with selected lines
            self.plot_selected_lines(selected_rows)

    def plot_selected_lines(self, rows):
        ax = self.spec_plotter.ax

        # Remove old lines/text
        for artist in self.plotted_artists:
            try:
                artist.remove()
            except Exception as e:
                print("DEBUG: could not remove artist", e)
        self.plotted_artists = []

        # Update currently plotted rows
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
        
        # Get cube wavelength unit
        cube_unit = getattr(self.cube_data, 'wave_unit', 'µm') if hasattr(self, 'cube_data') else 'µm'

        # Plot new selection with hover metadata
        for row in rows:
            λ = SpeciesDialog.get_row_wavelength(row)
            
            # Determine source unit (HITRAN)
            source_unit = row.get('wavelength_unit', 'µm')
            
            # Convert wavelength to cube units if needed
            try:
                if cube_unit != source_unit and are_units_compatible(source_unit, cube_unit):
                    λ_converted = convert_wavelength(λ, source_unit, cube_unit)
                    print(f"[DEBUG] Converted line {row.get('species', 'Unknown')}: {λ:.4f} {source_unit} -> {λ_converted:.4f} {cube_unit}")
                    λ = λ_converted
            except Exception as e:
                print(f"[DEBUG] Unit conversion failed for {row.get('species', 'Unknown')}: {e}")
                # Use original wavelength if conversion fails
            
            # Use HITRAN color (darkred)
            line = ax.axvline(λ, color="darkred", linestyle="--", alpha=0.6, picker=5)
            # Store row data in the line for hover access
            line.row_data = row
            text = ax.text(
                λ, ax.get_ylim()[1],
                row["species"], rotation=90, va="bottom", ha="center",
                fontsize=7, color="orange"
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
        
        # Find nearest line within threshold
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
            
            # Format tooltip with line information
            species = nearest_row.get('species', 'Unknown')
            wavelength = nearest_row.get('wavelength', 0)
            source_unit = nearest_row.get('wavelength_unit', 'µm')
            transition = nearest_row.get('transition', 'N/A')
            a_coeff = nearest_row.get('a_coeff', 0)
            e_upper = nearest_row.get('e_upper', 0)
            
            # Show both original and converted wavelength if units differ
            if cube_unit != source_unit:
                try:
                    wavelength_converted = convert_wavelength(wavelength, source_unit, cube_unit)
                    wavelength_str = f"λ: {wavelength:.4f} {source_unit} ({wavelength_converted:.4f} {cube_unit})"
                except:
                    wavelength_str = f"λ: {wavelength:.4f} {source_unit}"
            else:
                wavelength_str = f"λ: {wavelength:.4f} {source_unit}"
            
            tooltip_text = (
                f"Species: {species}\n"
                f"{wavelength_str}\n"
                f"Transition: {transition}\n"
                f"A-coeff: {a_coeff:.2e} s⁻¹\n"
                f"E_upper: {e_upper:.2f} K"
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
            except:
                pass
        if hasattr(self, 'gauss_right_marker') and self.gauss_right_marker:
            try:
                self.gauss_right_marker.remove()
            except:
                pass
        if hasattr(self, 'gauss_noise_left_marker') and self.gauss_noise_left_marker:
            try:
                self.gauss_noise_left_marker.remove()
            except:
                pass
        if hasattr(self, 'gauss_noise_right_marker') and self.gauss_noise_right_marker:
            try:
                self.gauss_noise_right_marker.remove()
            except:
                pass
        self.gauss_left_marker = None
        self.gauss_right_marker = None
        self.gauss_noise_left_marker = None
        self.gauss_noise_right_marker = None
        
        # Clear any plotted continuum line
        if hasattr(self, 'continuum_line') and self.continuum_line:
            try:
                self.continuum_line.remove()
            except:
                pass
        self.continuum_line = None
        
        # Uncheck all buttons
        self.set_left_btn.setChecked(False)
        self.set_right_btn.setChecked(False)
        self.gauss_left_btn.setChecked(False)
        self.gauss_right_btn.setChecked(False)
        
        # Clear all plotted lines (HITRAN)
        if hasattr(self, 'plotted_artists'):
            for artist in self.plotted_artists:
                try:
                    artist.remove()
                except:
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
                    except:
                        pass
                self.spec_plotter.gauss_fit_artists = []
            
            # Clear region lines
            if hasattr(self.spec_plotter, 'region_lines'):
                for line in self.spec_plotter.region_lines.values():
                    try:
                        line.remove()
                    except:
                        pass
                self.spec_plotter.region_lines = {}
            
            # Reload spectrum from original
            if hasattr(self, 'original_spectrum') and self.original_spectrum is not None:
                self.spec_plotter.plot_spectrum(self.original_spectrum)
            self.spec_canvas.draw()
        
        # Reset slice displays to index 0
        if hasattr(self, 'slice1_plotter') and self.slice1_plotter is not None:
            # Clear patches and markers
            if hasattr(self.slice1_plotter.ax, 'area_artists'):
                for artist in self.slice1_plotter.ax.area_artists:
                    try:
                        artist.remove()
                    except:
                        pass
                self.slice1_plotter.ax.area_artists = []
            # Remove handle scatter collections
            for collection in list(self.slice1_plotter.ax.collections):
                try:
                    if hasattr(collection, '_is_handle_collection') and collection._is_handle_collection:
                        collection.remove()
                except:
                    pass
            if hasattr(self.slice1_plotter.ax, 'pixel_marker'):
                for marker in self.slice1_plotter.ax.pixel_marker:
                    try:
                        marker.remove()
                    except:
                        pass
                self.slice1_plotter.ax.pixel_marker = []
            
            self._display_slice(0, "left")
        
        if hasattr(self, 'slice2_plotter') and self.slice2_plotter is not None:
            # Clear patches and markers
            if hasattr(self.slice2_plotter.ax, 'area_artists'):
                for artist in self.slice2_plotter.ax.area_artists:
                    try:
                        artist.remove()
                    except:
                        pass
                self.slice2_plotter.ax.area_artists = []
            # Remove handle scatter collections
            for collection in list(self.slice2_plotter.ax.collections):
                try:
                    if hasattr(collection, '_is_handle_collection') and collection._is_handle_collection:
                        collection.remove()
                except:
                    pass
            if hasattr(self.slice2_plotter.ax, 'pixel_marker'):
                for marker in self.slice2_plotter.ax.pixel_marker:
                    try:
                        marker.remove()
                    except:
                        pass
                self.slice2_plotter.ax.pixel_marker = []
            
            self._display_slice(0, "right")
        
        # Reset stats labels
        self.stats1_label.setText("Slice 1 Stats:\n-")
        self.stats2_label.setText("Slice 2 Stats:\n-")
        
        # Deactivate region selectors
        if hasattr(self, 'region_selector_left'):
            self.region_selector_left.activate_selector(None)
        if hasattr(self, 'region_selector_right'):
            self.region_selector_right.activate_selector(None)
        
        # Uncheck shape buttons
        if hasattr(self, 'shape_buttons'):
            for side in ["left", "right"]:
                for btn in self.shape_buttons[side].values():
                    btn.setChecked(False)
        
        print("[DEBUG] Refresh complete - all displays reset to initial state")

    def fetch_hitran_lines(self):
        """
        Fetch HITRAN lines based on user input from the GUI.
        Now includes transition level (vibrational quantum number) filtering.
        """
        try:
            # Get wavelength range from GUI inputs
            try:
                wavelength_min = float(self.min_wave_input.text())
                wavelength_max = float(self.max_wave_input.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Enter valid wavelength numbers.")
                return

            if wavelength_min >= wavelength_max:
                QMessageBox.warning(self, "Invalid Range", "Min wavelength must be less than Max.")
                return
            
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

            # Isotopologue selection
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

            # Transition level - upper vibrational quantum number
            v_upper, ok = QInputDialog.getInt(
                self,
                "Transition Level - Upper",
                "Enter upper vibrational level:\n(e.g., 1 for v=1->0)",
                1,
                0,
                100,
                1
            )
            if not ok:
                return

            # Transition level - lower vibrational quantum number
            v_lower, ok = QInputDialog.getInt(
                self,
                "Transition Level - Lower",
                "Enter lower vibrational level:\n(e.g., 0 for v=1->0)",
                0,
                0,
                100,
                1
            )
            if not ok:
                return

            # Cache refresh option
            refresh_cache = QMessageBox.question(
                self,
                "Cache Option",
                "Refresh the HITRAN cache for this query?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            ) == QMessageBox.Yes

            # Fetch HITRAN data
            results = hitran_gui_interface(
                wavelength_min, wavelength_max, molecule, isotopologue, refresh_cache
            )

            if not results:
                QMessageBox.information(self, "No Matches", "No HITRAN lines found in this range.")
                return

            # Filter by transition level (v values)
            filtered_results = []
            for line in results:
                transition = line.get('transition', '')
                # Parse v values from transition string like "v=1->0, J=23->22"
                if transition and '->' in transition:
                    try:
                        # Extract v values from format "v=X->Y"
                        v_match = re.search(r'v=(\d+)->(\d+)', transition)
                        if v_match:
                            line_v_upper = int(v_match.group(1))
                            line_v_lower = int(v_match.group(2))
                            
                            # Keep line if v values match the requested ones
                            if line_v_upper == v_upper and line_v_lower == v_lower:
                                filtered_results.append(line)
                    except (ValueError, AttributeError):
                        # If parsing fails, keep the line anyway
                        filtered_results.append(line)
                else:
                    # If no transition info, keep the line
                    filtered_results.append(line)
            
            if not filtered_results:
                QMessageBox.information(
                    self, 
                    "No Matches", 
                    f"No HITRAN lines found with v={v_upper}->{v_lower} transition."
                )
                return

            # Mark HITRAN lines with source
            for line in filtered_results:
                line['source'] = 'HITRAN'
            
            # Store HITRAN lines
            self.hitran_lines = filtered_results

            # Sort by wavelength (ascending)
            filtered_results.sort(key=lambda r: float(r.get('lambda_micron') or r.get('wavelength') or 0.0))

            # Show HITRAN results in selection dialog
            dlg = SpeciesDialog(filtered_results, parent=self, plotted_rows=self.plotted_lines)
            if dlg.exec_() == QDialog.Accepted:
                selected_rows = dlg.get_selected_rows()
                if not selected_rows:
                    QMessageBox.information(self, "No Selection", "No lines selected for plotting.")
                    return
                self.plot_selected_lines(selected_rows)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while fetching HITRAN lines:\n{e}")

    def _load_contaminating_lines_dialog(self):
        """
        Allow user to load an optional contaminating lines CSV file.
        This should be done AFTER setting wavelength range and loading HITRAN.
        """
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
            self.contam_lines = load_contaminating_lines_catalog(file_path)
            
            if self.contam_lines is not None and len(self.contam_lines) > 0:
                # Validate that loading succeeded (the CSV had the required 'wavelength' column)
                if 'line_wave_um' not in self.contam_lines.columns:
                    QMessageBox.critical(
                        self,
                        "Invalid CSV Format",
                        "❌ The contaminating lines CSV file is missing the required 'wavelength' column.\n\n"
                        "The CSV MUST have a column named 'wavelength' (case-sensitive)."
                    )
                    self.contam_lines = None
                    return
                
                print(f"✓ Loaded {len(self.contam_lines)} contaminating lines from {os.path.basename(file_path)}")
                
                # Show wavelength range of loaded lines
                min_wave = self.contam_lines['line_wave_um'].min()
                max_wave = self.contam_lines['line_wave_um'].max()
                print(f"  Wavelength range: {min_wave:.4f} - {max_wave:.4f} μm")
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"✓ Loaded {len(self.contam_lines)} contaminating lines\n\n"
                    f"Wavelength range: {min_wave:.4f} - {max_wave:.4f} μm\n\n"
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

    def _activate_gaussian_left(self):
        """Activate selection mode for left boundary of Gaussian fit"""
        if self.gauss_left_btn.isChecked():
            self.gauss_mode = 'left'
            self.gauss_peak_btn.setChecked(False)
            self.gauss_right_btn.setChecked(False)
            self.gauss_noise_left_btn.setChecked(False)
            self.gauss_noise_right_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _activate_gaussian_peak(self):
        """Activate selection mode for peak of Gaussian fit"""
        if self.gauss_peak_btn.isChecked():
            self.gauss_mode = 'peak'
            self.gauss_left_btn.setChecked(False)
            self.gauss_right_btn.setChecked(False)
            self.gauss_noise_left_btn.setChecked(False)
            self.gauss_noise_right_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _activate_gaussian_right(self):
        """Activate selection mode for right boundary of Gaussian fit"""
        if self.gauss_right_btn.isChecked():
            self.gauss_mode = 'right'
            self.gauss_left_btn.setChecked(False)
            self.gauss_peak_btn.setChecked(False)
            self.gauss_noise_left_btn.setChecked(False)
            self.gauss_noise_right_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _activate_gaussian_noise_left(self):
        """Activate selection mode for left boundary of noise region"""
        if self.gauss_noise_left_btn.isChecked():
            self.gauss_mode = 'noise_left'
            self.gauss_left_btn.setChecked(False)
            self.gauss_peak_btn.setChecked(False)
            self.gauss_right_btn.setChecked(False)
            self.gauss_noise_right_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _activate_gaussian_noise_right(self):
        """Activate selection mode for right boundary of noise region"""
        if self.gauss_noise_right_btn.isChecked():
            self.gauss_mode = 'noise_right'
            self.gauss_left_btn.setChecked(False)
            self.gauss_peak_btn.setChecked(False)
            self.gauss_right_btn.setChecked(False)
            self.gauss_noise_left_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _mark_gaussian_boundary(self, x_pos):
        """Mark a boundary point for Gaussian fitting"""
        ax = self.spec_plotter.ax
        
        if self.gauss_mode == 'left':
            # Clear previous left marker if exists
            if hasattr(self, 'gauss_left_marker') and self.gauss_left_marker:
                try:
                    self.gauss_left_marker.remove()
                except:
                    pass
            
            # Draw new left marker (green dashed, thin)
            self.gauss_left_marker = ax.axvline(x_pos, color='green', linestyle='--', linewidth=1.5, label='Left', alpha=0.7)
            self.gauss_left_x = x_pos
            self.gauss_left_btn.setChecked(False)
            self.gauss_mode = None
            print(f"[DEBUG] Left boundary set at λ = {x_pos:.4f}")
            
        elif self.gauss_mode == 'peak':
            # Clear previous peak marker if exists
            if hasattr(self, 'gauss_peak_marker') and self.gauss_peak_marker:
                try:
                    self.gauss_peak_marker.remove()
                except:
                    pass
            
            # Draw new peak marker (red dotted, very thin - distinguish from left/right)
            self.gauss_peak_marker = ax.axvline(x_pos, color='red', linestyle=':', linewidth=1.0, label='Peak', alpha=0.8)
            self.gauss_peak_x = x_pos
            self.gauss_peak_btn.setChecked(False)
            self.gauss_mode = None
            print(f"[DEBUG] Peak set at λ = {x_pos:.4f}")
            
        elif self.gauss_mode == 'right':
            # Clear previous right marker if exists
            if hasattr(self, 'gauss_right_marker') and self.gauss_right_marker:
                try:
                    self.gauss_right_marker.remove()
                except:
                    pass
            
            # Draw new right marker (purple dashed, thin)
            self.gauss_right_marker = ax.axvline(x_pos, color='purple', linestyle='--', linewidth=1.5, label='Right', alpha=0.7)
            self.gauss_right_x = x_pos
            self.gauss_right_btn.setChecked(False)
            self.gauss_mode = None
            print(f"[DEBUG] Right boundary set at λ = {x_pos:.4f}")

        elif self.gauss_mode == 'noise_left':
            # Clear previous noise-left marker if exists
            if hasattr(self, 'gauss_noise_left_marker') and self.gauss_noise_left_marker:
                try:
                    self.gauss_noise_left_marker.remove()
                except:
                    pass

            # Draw new noise-left marker (orange dashed, thin)
            self.gauss_noise_left_marker = ax.axvline(x_pos, color='orange', linestyle='--', linewidth=1.3, label='Noise L', alpha=0.7)
            self.gauss_noise_left_x = x_pos
            self.gauss_noise_left_btn.setChecked(False)
            self.gauss_mode = None
            print(f"[DEBUG] Noise left boundary set at λ = {x_pos:.4f}")

        elif self.gauss_mode == 'noise_right':
            # Clear previous noise-right marker if exists
            if hasattr(self, 'gauss_noise_right_marker') and self.gauss_noise_right_marker:
                try:
                    self.gauss_noise_right_marker.remove()
                except:
                    pass

            # Draw new noise-right marker (brown dashed, thin)
            self.gauss_noise_right_marker = ax.axvline(x_pos, color='sienna', linestyle='--', linewidth=1.3, label='Noise R', alpha=0.7)
            self.gauss_noise_right_x = x_pos
            self.gauss_noise_right_btn.setChecked(False)
            self.gauss_mode = None
            print(f"[DEBUG] Noise right boundary set at λ = {x_pos:.4f}")
        
        self.spec_canvas.draw()

    def _fit_gaussian(self):
        """Perform Gaussian fit to selected spectral line region"""
        # Validation
        if self.gauss_left_x is None or self.gauss_right_x is None:
            QMessageBox.warning(
                self, "Missing Boundaries",
                "Please select LEFT, PEAK, and RIGHT boundaries in order.\n\n"
                "1. Click 'Left' button, then click on spectrum (left edge)\n"
                "2. Click 'Peak' button, then click on the peak (REQUIRED)\n"
                "3. Click 'Right' button, then click on spectrum (right edge)\n"
                "4. Click 'Fit Gaussian'"
            )
            return
        
        # REQUIRE peak selection
        if self.gauss_peak_x is None:
            QMessageBox.warning(
                self, "Peak Selection Required",
                "You MUST select a peak to fit.\n\n"
                "Please:\n"
                "1. Click 'Peak' button\n"
                "2. Click on the peak in the spectrum\n"
                "3. Then click 'Fit Gaussian'"
            )
            return
        
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            return
        
        if not hasattr(self, 'cube_data') or self.cube_data is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            return
        
        # Get spectrum data from cube_data
        spectrum_x = _extract_wavelength_value(self.spec_plotter.wavelength)
        spectrum_y = self.cube_data.spectrum
        
        if spectrum_x is None or spectrum_y is None:
            QMessageBox.warning(self, "No Spectrum", "No spectrum data available.")
            return
        
        if len(spectrum_x) != len(spectrum_y):
            QMessageBox.warning(self, "Data Mismatch", "Wavelength and flux arrays have different lengths.")
            return
        
        # Check how many points are in the fitting range
        lam_min = min(self.gauss_left_x, self.gauss_right_x)
        lam_max = max(self.gauss_left_x, self.gauss_right_x)
        
        # Handle Quantity objects from Phase 1
        if isinstance(spectrum_x, Quantity):
            # Extract values for comparison
            spectrum_x_vals = spectrum_x.value
        else:
            spectrum_x_vals = spectrum_x
        
        fit_mask = (spectrum_x_vals >= lam_min) & (spectrum_x_vals <= lam_max)
        n_points_in_fit = np.count_nonzero(fit_mask)
        
        # If fitting range has < 15 points, REQUIRE noise region
        if n_points_in_fit < 15:
            if self.gauss_noise_left_x is None or self.gauss_noise_right_x is None:
                QMessageBox.warning(
                    self, "Noise Region Required",
                    f"Your fitting range contains only {n_points_in_fit} data points.\n\n"
                    "For narrow ranges (<15 points), you MUST also define a noise region "
                    "to get a reliable noise estimate.\n\n"
                    "Please:\n"
                    "1. Click 'Noise L' button, then click on spectrum (line-free region)\n"
                    "2. Click 'Noise R' button, then click on spectrum\n"
                    "3. Try fitting again"
                )
                return
        
        # Noise region validation: require BOTH boundaries or NEITHER
        # If only one is set, treat as if neither are set
        if (self.gauss_noise_left_x is None) ^ (self.gauss_noise_right_x is None):
            self.gauss_noise_left_x = None
            self.gauss_noise_right_x = None

        noise_min = None
        noise_max = None
        if self.gauss_noise_left_x is not None and self.gauss_noise_right_x is not None:
            noise_min = min(self.gauss_noise_left_x, self.gauss_noise_right_x)
            noise_max = max(self.gauss_noise_left_x, self.gauss_noise_right_x)

        # Use HITRAN line to set lambda_0 if available
        self.lambda_0 = None
        if hasattr(self, 'hitran_lines') and self.hitran_lines:
            self.lambda_0 = self._find_nearest_line_in_range(lam_min, lam_max)

        # Perform fit
        try:
            result = fit_gaussian_to_line(
                self.gauss_left_x, self.gauss_right_x,
                spectrum_x, spectrum_y,
                peak_x=self.gauss_peak_x,  # Pass optional peak for better initial guess
                noise_min=noise_min,
                noise_max=noise_max,
                header=self.cube_data.header,  # Pass header for unit conversion
                wave_unit=self.cube_data.wave_unit
            )
            
            if not result.success:
                QMessageBox.warning(self, "Fit Failed", result.message)
                return
            
            # Display results
            self._display_gaussian_results(result)
            
            # Plot the fit on the spectrum using visualization module
            self.spec_plotter.plot_gaussian_fit(result)
            self.spec_canvas.draw_idle()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gaussian fitting error:\n{str(e)}")

    def _display_gaussian_results(self, result: GaussianFitResult):
        """Display Gaussian fit results in a dialog"""        
        # Get units from cube data
        wave_unit_raw = getattr(self.cube_data, 'wave_unit', 'spectral units')
        flux_unit_raw = getattr(self.cube_data, 'flux_unit', 'flux units')
        
        # Clean units for display
        wave_unit = clean_wave_unit(wave_unit_raw)
        flux_unit = clean_flux_unit(flux_unit_raw)
        
        # Determine appropriate unit names
        if wave_unit in ["Å", "µm", "nm"]:
            axis_name = "Wavelength"
        elif "km/s" in wave_unit or "m/s" in wave_unit:
            axis_name = "Velocity"
        else:
            axis_name = "Spectral axis"
        
        # For area, don't repeat the wavelength unit if it's already in flux_unit
        # E.g., if flux_unit is "erg s⁻¹ cm⁻² Å⁻¹", area should be "erg s⁻¹ cm⁻²" (integrated over Å)
        if wave_unit and (wave_unit in flux_unit or wave_unit.replace('Å', 'Angstrom') in flux_unit_raw):
            # Flux unit already includes per-wavelength, so area is flux integrated
            # Remove the "per wavelength" part for area
            area_unit = flux_unit.replace(f' {wave_unit}⁻¹', '').replace(f'/{wave_unit}', '')
        else:
            # Standard case: area = flux × wavelength
            area_unit = f"{flux_unit}·{wave_unit}"
        
        # For baseline slope
        slope_unit = f"{flux_unit}/{wave_unit}" if wave_unit not in flux_unit else flux_unit
        
        # Format results text with dynamic units
        results_text = f"""
GAUSSIAN FIT RESULTS
{'='*60}

FITTED PARAMETERS:
  Center ({axis_name}):  {result.center:.6f} {wave_unit}
  Amplitude:             {result.amplitude:.4e} {flux_unit}
  Sigma (σ):             {result.sigma:.6f} {wave_unit}
  FWHM:                  {result.fwhm:.6f} {wave_unit}

INTEGRATED QUANTITIES:
  Gaussian Area:         {result.area:.4e} {area_unit}
  Equivalent Width:      {result.area/np.mean(result.baseline):.4e} {wave_unit}

BASELINE:
  Slope:                 {result.baseline_slope:.4e} {slope_unit}
  Left endpoint:         {result.baseline_intercept:.4e} {flux_unit}

FIT QUALITY:
  R² (coefficient):      {result.r_squared:.4f}
  χ² (chi-squared):      {result.chi_squared:.4e}
  Data points:           {len(result.x_data)}

STATUS: {result.message}
{'='*60}

INTERPRETATION:
- Center: Peak {axis_name.lower()} of the spectral line
- FWHM: Full Width at Half Maximum (line width)
- Area: Integrated flux (proportional to line strength)
- R²: How well the Gaussian fits (1.0 = perfect, >0.9 = good)
- Equivalent Width: Area divided by continuum level
"""
        
        # Create dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Gaussian Fit Results")
        dlg.setMinimumSize(600, 500)
        
        layout = QVBoxLayout()
        
        # Text display
        text_edit = QTextEdit()
        text_edit.setPlainText(results_text)
        text_edit.setReadOnly(True)
        text_edit.setFontFamily("Courier")
        layout.addWidget(text_edit)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        
        dlg.setLayout(layout)
        dlg.exec_()

    def clear_gaussian_fit(self):
        """Clear all Gaussian fit visualization elements from the spectrum"""
        try:
            # Remove Gaussian fit artists (baseline, fit line, fill, text)
            if hasattr(self.spec_plotter, 'gauss_fit_artists'):
                for artist in self.spec_plotter.gauss_fit_artists:
                    try:
                        artist.remove()
                    except:
                        pass
                self.spec_plotter.gauss_fit_artists = []
                print("[DEBUG] Cleared Gaussian fit artists")
            
            # Remove individual Gaussian boundary markers (vertical lines)
            marker_attrs = ['gauss_left_marker', 'gauss_peak_marker', 'gauss_right_marker', 
                           'gauss_noise_left_marker', 'gauss_noise_right_marker']
            for attr in marker_attrs:
                if hasattr(self, attr):
                    marker = getattr(self, attr)
                    if marker is not None:
                        try:
                            marker.remove()
                        except:
                            pass
                    setattr(self, attr, None)
            print("[DEBUG] Cleared Gaussian boundary markers (vertical lines)")
            
            # Remove markers from list (for compatibility)
            for marker in self.gauss_markers:
                try:
                    marker.remove()
                except:
                    pass
            self.gauss_markers = []
            print("[DEBUG] Cleared Gaussian marker list")
            
            # Reset Gaussian state variables
            self.gauss_left_x = None
            self.gauss_peak_x = None
            self.gauss_right_x = None
            self.gauss_noise_left_x = None
            self.gauss_noise_right_x = None
            print("[DEBUG] Reset Gaussian state variables")
            
            # Uncheck all Gaussian buttons
            self.gauss_left_btn.setChecked(False)
            self.gauss_peak_btn.setChecked(False)
            self.gauss_right_btn.setChecked(False)
            self.gauss_noise_left_btn.setChecked(False)
            self.gauss_noise_right_btn.setChecked(False)
            
            # Clear multi-Gaussian fits if present
            if hasattr(self, 'multi_gauss_artists'):
                for artist in self.multi_gauss_artists:
                    try:
                        artist.remove()
                    except:
                        pass
                self.multi_gauss_artists = []
                print("[DEBUG] Cleared multi-Gaussian artists")
            
            self.multi_gauss_results = None
            
            # Update legend and redraw
            self.spec_plotter.ax.legend(loc='upper right', fontsize=8)
            self.spec_canvas.draw()
            
        except Exception as e:
            print(f"[DEBUG] Error clearing Gaussian fit: {e}")

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

    def _fit_multi_gaussian(self):
        """
        Fit Gaussians to each selected HITRAN line individually.
        Same process as single Gaussian, but repeat for each line.
        Uses the pre-computed automatic continuum as baseline.
        """
        # Validation: lines selected
        if not hasattr(self, 'plotted_lines') or not self.plotted_lines:
            QMessageBox.warning(
                self,
                "No Lines Selected",
                "Please first select HITRAN lines to fit."
            )
            return
        
        # Validation: data loaded
        if not hasattr(self, 'cube_data') or self.cube_data is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            return
        
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            return
        
        # Validation: continuum computed
        if not hasattr(self, 'continuum_fitted') or self.continuum_fitted is None:
            QMessageBox.warning(
                self,
                "No Continuum",
                "Please compute the continuum first before fitting lines."
            )
            return
        
        # Get spectrum data
        spectrum_x = _extract_wavelength_value(self.spec_plotter.wavelength)
        spectrum_y = self.cube_data.spectrum
        continuum_y = self.continuum_fitted
        
        if spectrum_x is None or spectrum_y is None:
            QMessageBox.warning(self, "No Spectrum", "No spectrum data available.")
            return
        
        if len(spectrum_x) != len(spectrum_y) or len(spectrum_x) != len(continuum_y):
            QMessageBox.warning(self, "Data Mismatch", "Wavelength, flux, and continuum arrays have different lengths.")
            return
        
        # Fit each HITRAN line individually
        fit_results = []
        
        for line in self.plotted_lines:
            try:
                # Get line center wavelength
                line_center = line.get('lambda_micron') or line.get('wavelength')
                if line_center is None:
                    continue
                
                line_center = float(line_center)
                
                # Auto-detect boundaries: find where signal drops below ~30% of peak
                x1, x2 = auto_detect_boundaries(
                    line_center, spectrum_x, spectrum_y,
                    continuum_y=continuum_y,
                    threshold_percent=30.0
                )
                
                # Get indices for this region
                region_mask = (spectrum_x >= x1) & (spectrum_x <= x2)
                continuum_region = continuum_y[region_mask]
                
                # Subtract the auto-continuum to get line-only data
                spectrum_region = spectrum_y[region_mask]
                spectrum_line_only = spectrum_region - continuum_region
                spectrum_x_region = spectrum_x[region_mask]
                
                # Fit Gaussian to continuum-subtracted spectrum
                # The baseline will naturally be ~0 (the continuum level was subtracted)
                result = fit_gaussian_to_line(
                    x1, x2,
                    spectrum_x_region, spectrum_line_only,  # Use continuum-subtracted data
                    peak_x=line_center,
                    noise_min=None,
                    noise_max=None,
                    header=self.cube_data.header,
                    wave_unit=self.cube_data.wave_unit
                )
                
                if result.success:
                    # Replace the linear baseline with the actual auto-continuum array
                    # so the baseline in plots and calculations reflects the true continuum
                    result.baseline = continuum_region.copy()
                    
                    fit_results.append({
                        'line': line,
                        'result': result
                    })
            
            except Exception as e:
                print(f"Error fitting line {line.get('species', 'Unknown')}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not fit_results:
            QMessageBox.warning(
                self,
                "Fit Failed",
                "Could not fit any HITRAN lines. Check their positions and wavelength ranges."
            )
            return
        
        print(f"[DEBUG] Successfully fitted {len(fit_results)} lines with auto-continuum baseline")
        
        # Store results for later use
        self.multi_gauss_results = fit_results
        
        # Plot all fits on spectrum and show results
        self._plot_multi_gaussian_on_spectrum(fit_results)
        self._show_multi_gaussian_results(fit_results)

    def _plot_multi_gaussian_on_spectrum(self, fit_results):
        """
        Plot each fitted Gaussian on the spectrum.
        Plot all 10 at once, don't call plot_gaussian_fit repeatedly (it clears each time).
        fit_results is a list of {'line': line_dict, 'result': GaussianFitResult}
        """
        ax = self.spec_plotter.ax
        
        # Store artists for later clearing  
        if not hasattr(self, 'multi_gauss_artists'):
            self.multi_gauss_artists = []
        
        # Clear old artists if any
        for artist in self.multi_gauss_artists:
            try:
                artist.remove()
            except:
                pass
        self.multi_gauss_artists = []
        
        # Normalize if needed (same logic as single Gaussian)
        if hasattr(self.spec_plotter, 'global_line') and self.spec_plotter.global_line is not None:
            global_y = self.spec_plotter.global_line.get_ydata()
            global_median = np.nanmedian(global_y[np.isfinite(global_y)])
            if 0.5 < global_median < 2.0:
                fit_median = np.nanmedian(np.concatenate([fit['result'].y_data for fit in fit_results]))
                norm_factor = fit_median if fit_median > 0 else 1.0
            else:
                norm_factor = 1.0
        else:
            norm_factor = 1.0
        
        # Plot each fitted Gaussian
        for idx, fit_info in enumerate(fit_results):
            result = fit_info['result']
            
            # Normalize
            baseline_norm = result.baseline / norm_factor
            y_fit_norm = result.y_fit / norm_factor
            
            # Plot baseline as dashed line (only for first fit to avoid legend clutter)
            if idx == 0:
                baseline_line, = ax.plot(
                    result.x_data, baseline_norm,
                    'k--', linewidth=2, alpha=0.7, label='Baseline', zorder=10
                )
                self.multi_gauss_artists.append(baseline_line)
            
            # Plot fitted Gaussian as red line
            gauss_line, = ax.plot(
                result.x_data, baseline_norm + y_fit_norm,
                'r-', linewidth=3, alpha=0.9, label='Gaussian Fit' if idx == 0 else '', zorder=11
            )
            self.multi_gauss_artists.append(gauss_line)
            
            # Shade area under Gaussian
            fill = ax.fill_between(
                result.x_data, baseline_norm, baseline_norm + y_fit_norm,
                color='red', alpha=0.2, zorder=9
            )
            self.multi_gauss_artists.append(fill)
        
        # Update legend and redraw
        ax.legend(loc='upper right', fontsize=8)
        self.spec_canvas.draw_idle()
        print(f"[DEBUG] Plotted {len(fit_results)} Gaussian fits on spectrum")

    def _show_multi_gaussian_results(self, fit_results):
        """
        Show results dialog with statistics for all fitted lines.
        fit_results is a list of {'line': line_dict, 'result': GaussianFitResult}
        """
        # Get units from cube data
        wave_unit_raw = getattr(self.cube_data, 'wave_unit', 'µm')
        flux_unit_raw = getattr(self.cube_data, 'flux_unit', 'MJy/sr')
        
        # Clean units for display
        wave_unit = clean_wave_unit(wave_unit_raw)
        flux_unit = clean_flux_unit(flux_unit_raw)
        
        # Area unit: flux integrated over wavelength
        area_unit = f"{flux_unit}·{wave_unit}"
        
        # Build results text with all fitted lines
        results_text = f"""
MULTI-GAUSSIAN FIT RESULTS
{'='*70}

Successfully fitted {len(fit_results)} lines

FITTED LINES:
{'='*70}
"""
        
        for idx, fit_info in enumerate(fit_results, 1):
            result = fit_info['result']
            line = fit_info['line']
            species = line.get('species', 'Unknown')
            
            results_text += f"""
Line {idx}: {species}
  Center (λ):            {result.center:.6f} {wave_unit}
  Amplitude:             {result.amplitude:.4e} {flux_unit}
  Sigma (σ):             {result.sigma:.6f} {wave_unit}
  FWHM:                  {result.fwhm:.6f} {wave_unit}
  Integrated Area:       {result.area:.4e} {area_unit}
  R²:                    {result.r_squared:.4f}
  {'─'*60}
"""
        
        results_text += f"""
OVERALL:
  Total lines fitted:    {len(fit_results)}
  
Note: Each line was fitted individually using the standard Gaussian fitting process.
Red shaded areas are now visible on the spectrum for all lines.
"""
        
        # Create dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Multi-Gaussian Fit Results")
        dlg.setMinimumSize(750, 600)
        
        layout = QVBoxLayout()
        
        # Text display with scrollbar
        text_display = QTextEdit()
        text_display.setPlainText(results_text)
        text_display.setReadOnly(True)
        text_display.setFont(QFont('Courier', 9))
        layout.addWidget(text_display)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(lambda: dlg.accept())
        
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        dlg.setLayout(layout)
        dlg.exec_()
        """
        Show results dialog with statistics for all fitted lines.
        fit_results is a list of {'line': line_dict, 'result': GaussianFitResult}
        """
        # Get units from cube data
        wave_unit_raw = getattr(self.cube_data, 'wave_unit', 'µm')
        flux_unit_raw = getattr(self.cube_data, 'flux_unit', 'MJy/sr')
        
        # Clean units for display
        wave_unit = clean_wave_unit(wave_unit_raw)
        flux_unit = clean_flux_unit(flux_unit_raw)
        
        # Area unit: flux integrated over wavelength
        area_unit = f"{flux_unit}·{wave_unit}"
        
        # Build results text with all fitted lines
        results_text = f"""
MULTI-GAUSSIAN FIT RESULTS
{'='*70}

Successfully fitted {len(fit_results)} lines

FITTED LINES:
{'='*70}
"""
        
        for idx, fit_info in enumerate(fit_results, 1):
            result = fit_info['result']
            line = fit_info['line']
            species = line.get('species', 'Unknown')
            
            results_text += f"""
Line {idx}: {species}
  Center (λ):            {result.center:.6f} {wave_unit}
  Amplitude:             {result.amplitude:.4e} {flux_unit}
  Sigma (σ):             {result.sigma:.6f} {wave_unit}
  FWHM:                  {result.fwhm:.6f} {wave_unit}
  Integrated Area:       {result.area:.4e} {area_unit}
  R²:                    {result.r_squared:.4f}
  {'─'*60}
"""
        
        results_text += f"""
OVERALL:
  Total lines fitted:    {len(fit_results)}
  
Note: Each line was fitted individually using the standard Gaussian fitting process.
Red shaded areas are now visible on the spectrum for all lines.
"""
        
        # Create dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Multi-Gaussian Fit Results")
        dlg.setMinimumSize(750, 600)
        
        layout = QVBoxLayout()
        
        # Text display with scrollbar
        text_display = QTextEdit()
        text_display.setPlainText(results_text)
        text_display.setReadOnly(True)
        text_display.setFont(QFont('Courier', 9))
        layout.addWidget(text_display)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(lambda: dlg.accept())
        
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        dlg.setLayout(layout)
        dlg.exec_()

    def _fit_gaussian_map(self):
        """Fit Gaussian to all pixels in the cube using the selected boundaries (runs in background)"""        
        # Validation
        if self.gauss_left_x is None or self.gauss_right_x is None:
            QMessageBox.warning(
                self, "Missing Boundaries",
                "Please select both LEFT and RIGHT boundaries first.\n\n"
                "1. Click 'Left' button, then click on spectrum\n"
                "2. Click 'Right' button, then click on spectrum\n"
                "3. Click 'Fit Gaussian Map'"
            )
            return
        
        if not hasattr(self, 'cube_data') or self.cube_data is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            return
        
        # Get cube dimensions
        cube = self.cube_data.cube  # 3D array (n_wave, n_y, n_x)
        wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
        
        if cube is None or wavelength is None:
            QMessageBox.warning(self, "No Data", "No cube data available.")
            return
        
        n_wave, n_y, n_x = cube.shape
        total_pixels = n_y * n_x
        
        # Get wavelength range
        lam_min = min(self.gauss_left_x, self.gauss_right_x)
        lam_max = max(self.gauss_left_x, self.gauss_right_x)
        
        # Check how many points are in the fitting range
        fit_mask = (wavelength >= lam_min) & (wavelength <= lam_max)
        n_points_in_fit = np.count_nonzero(fit_mask)
        
        # Warn if fit region is very narrow (< 50 points)
        if n_points_in_fit < 50:
            reply = QMessageBox.warning(
                self, "Narrow Fit Region",
                f"Your fitting range contains only {n_points_in_fit} data points.\n\n"
                "This is very narrow and lacks continuum context for robust fitting.\n"
                "For best results, select a wider region that includes line shoulders "
                "and continuum on both sides (≥50 points recommended).\n\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # ALWAYS require noise region for map fitting (for robust noise estimation)
        if self.gauss_noise_left_x is None or self.gauss_noise_right_x is None:
            QMessageBox.warning(
                self, "Noise Region Required for Map Fitting",
                "For reliable per-pixel Gaussian fits, you MUST define a noise region\n"
                "(a line-free area where we can measure the continuum noise level).\n\n"
                "Please:\n"
                "1. Click 'Noise L' button, then click on spectrum (line-free region)\n"
                "2. Click 'Noise R' button, then click on spectrum\n"
                "3. Try fitting the map again"
            )
            return

        # Noise region validation: require BOTH boundaries or NEITHER
        # If only one is set, treat as if neither are set
        if (self.gauss_noise_left_x is None) ^ (self.gauss_noise_right_x is None):
            self.gauss_noise_left_x = None
            self.gauss_noise_right_x = None

        noise_min = None
        noise_max = None
        if self.gauss_noise_left_x is not None and self.gauss_noise_right_x is not None:
            noise_min = min(self.gauss_noise_left_x, self.gauss_noise_right_x)
            noise_max = max(self.gauss_noise_left_x, self.gauss_noise_right_x)
        
        # Check if we have cached results for this exact wavelength range
        if (self.gauss_cache is not None and 
            self.gauss_cache_lam_min == lam_min and 
            self.gauss_cache_lam_max == lam_max and
            self.gauss_cache_noise_min == noise_min and
            self.gauss_cache_noise_max == noise_max):
            # Use cached results - no need to refit
            QMessageBox.information(
                self, "Using Cached Results",
                f"Using previously fitted Gaussian map for wavelength range:\n"
                f"[{lam_min:.4f}, {lam_max:.4f}]\n\n"
                f"Select new boundaries to refit."
            )
            self._on_gaussian_map_complete(self.gauss_cache)
            return
        
        # Use standard 3-sigma detection threshold
        fac_above_stdres = 3.0
        
        # Confirm with user
        noise_text = "None" if noise_min is None else f"[{noise_min:.4f}, {noise_max:.4f}]"
        reply = QMessageBox.question(
            self, "Fit Gaussian Map",
            f"This will fit Gaussians to {total_pixels:,} pixels.\n"
            f"Wavelength range: [{self.gauss_left_x:.4f}, {self.gauss_right_x:.4f}]\n"
            f"Noise range: {noise_text}\n"
            f"Detection threshold: {fac_above_stdres:.1f}σ\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return

        # Create non-modal progress dialog (allows interaction)
        self.gauss_progress = QProgressDialog(
            "Fitting Gaussians...", 
            "Hide", 0, 0, self
        )
        self.gauss_progress.setWindowModality(Qt.NonModal)
        self.gauss_progress.setWindowTitle("Gaussian Fitting (Background)")
        self.gauss_progress.setMinimumDuration(0)
        self.gauss_progress.setCancelButton(None)  # Can't cancel once started
        self.gauss_progress.show()

        self.lambda_0 = None
        if hasattr(self, 'hitran_lines') and self.hitran_lines:
            self.lambda_0 = self._find_nearest_line_in_range(lam_min, lam_max)
        
        # Create and start background worker
        self.gauss_worker = GaussianMapWorker(
            cube=cube,
            wavelength=wavelength,
            lam_min=lam_min,
            lam_max=lam_max,
            n_jobs=8,
            fac_above_stdres=fac_above_stdres,
            noise_min=noise_min,
            noise_max=noise_max,
            header=self.cube_data.header,  # Pass header for unit conversion
            wave_unit=self.cube_data.wave_unit,
            flux_unit=self.cube_data.flux_unit,  # Pass flux unit for CGS conversion validation
            lambda_0=self.lambda_0
        )
        self.gauss_worker.finished.connect(self._on_gaussian_map_complete)
        self.gauss_worker.error.connect(self._on_gaussian_map_error)
        self.gauss_worker.start()
    
    def _on_gaussian_map_complete(self, results):
        """Called when background Gaussian fitting completes"""
        self.gauss_progress.close()
        QMessageBox.information(
            self, "Fitting Complete",
            "Gaussian map fitting finished successfully!"
        )
        # Cache the results for reuse if wavelength range hasn't changed
        self.gauss_cache = results
        self.gauss_cache_lam_min = min(self.gauss_left_x, self.gauss_right_x)
        self.gauss_cache_lam_max = max(self.gauss_left_x, self.gauss_right_x)
        if self.gauss_noise_left_x is not None and self.gauss_noise_right_x is not None:
            self.gauss_cache_noise_min = min(self.gauss_noise_left_x, self.gauss_noise_right_x)
            self.gauss_cache_noise_max = max(self.gauss_noise_left_x, self.gauss_noise_right_x)
        else:
            self.gauss_cache_noise_min = None
            self.gauss_cache_noise_max = None
        self._show_gaussian_map_results(results)
    
    def _on_gaussian_map_error(self, error_msg):
        """Called when background Gaussian fitting fails"""
        self.gauss_progress.close()
        QMessageBox.critical(self, "Error", f"Gaussian map fitting error:\n{error_msg}")
    
    def _show_gaussian_map_results(self, results: dict):
        """Display Gaussian parameter maps in a new window"""
        dlg = GaussianMapResultsDialog(
            self, 
            results, 
            self.cube_data, 
            self.gauss_left_x, 
            self.gauss_right_x,
            wavelength=self.spec_plotter.wavelength  # Pass wavelength for pixel diagnostics
        )
        dlg.exec_()

    def _save_spectrum_ascii(self):
        """Save the current spectrum displayed in the bottom panel to ASCII file"""
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "No spectrum loaded to save.")
            return
        
        if _extract_wavelength_value(self.spec_plotter.wavelength) is None:
            QMessageBox.warning(self, "No Data", "No wavelength data available.")
            return
        
        # Get the current spectrum data (use current plotted spectrum)
        wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
        
        # Use current spectrum from plotter
        spectrum = self.spec_plotter.current_spectrum if hasattr(self.spec_plotter, 'current_spectrum') else self.cube_data.spectrum
        spectrum_type = "original"
        
        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Spectrum as ASCII",
            "spectrum.txt",
            "Text files (*.txt);;CSV files (*.csv);;All files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Prepare header
            wave_unit = self.cube_data.wave_unit if hasattr(self.cube_data, 'wave_unit') else "Unknown"
            instrument = self.cube_data.instrument if hasattr(self.cube_data, 'instrument') else "Unknown"
            
            # Write ASCII file with header
            with open(file_path, 'w') as f:
                f.write(f"# Spectrum Export from Cube Explorer\n")
                f.write(f"# Instrument: {instrument}\n")
                f.write(f"# Type: {spectrum_type}\n")
                f.write(f"# Wavelength unit: {wave_unit}\n")
                f.write(f"# Flux unit: {self.cube_data.header.get('BUNIT', 'Unknown') if hasattr(self.cube_data, 'header') else 'Unknown'}\n")
                f.write(f"#\n")
                f.write(f"# Wavelength ({wave_unit})\tFlux\n")
                
                # Write data
                for w, s in zip(wavelength, spectrum):
                    f.write(f"{w:.6e}\t{s:.6e}\n")
            
            QMessageBox.information(self, "Success", f"Spectrum saved to:\n{file_path}")
            print(f"Spectrum saved to {file_path}")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save spectrum:\n{str(e)}")
            print(f"Error saving spectrum: {e}")

    def on_auto_continuum_toggle(self):
        """Handle Automatic Continuum toggle button"""
        if self.auto_continuum_toggle.isChecked():
            # Check if custom continuum is active with progress
            if self.custom_continuum_toggle.isChecked():
                if self.has_custom_progress():
                    reply = QMessageBox.question(
                        self, 
                        "Switch Mode?",
                        "Switching to Automatic Continuum will clear your Custom Continuum progress. Continue?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        self.auto_continuum_toggle.setChecked(False)
                        return
                    else:
                        self.clear_custom_progress()

                # Uncheck custom toggle
                self.custom_continuum_toggle.setChecked(False)

            # Show automatic continuum buttons
            self.show_automatic_buttons()
        else:
            # Check if there's progress before turning off
            if self.has_automatic_progress():
                reply = QMessageBox.question(
                    self,
                    "Turn Off?",
                    "Turning off Automatic Continuum will clear your progress. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    self.auto_continuum_toggle.setChecked(True)
                    return
                else:
                    self.clear_automatic_progress()

            # Hide automatic buttons
            self.hide_all_action_buttons()


    def on_custom_continuum_toggle(self):
        """Handle Custom Continuum toggle button"""
        if self.custom_continuum_toggle.isChecked():
            # Check if automatic continuum is active with progress
            if self.auto_continuum_toggle.isChecked():
                if self.has_automatic_progress():
                    reply = QMessageBox.question(
                        self,
                        "Switch Mode?",
                        "Switching to Custom Continuum will clear your Automatic Continuum progress. Continue?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        self.custom_continuum_toggle.setChecked(False)
                        return
                    else:
                        self.clear_automatic_progress()

                # Uncheck automatic toggle
                self.auto_continuum_toggle.setChecked(False)

            # Show custom continuum buttons
            self.show_custom_buttons()
        else:
            # Check if there's progress before turning off
            if self.has_custom_progress():
                reply = QMessageBox.question(
                    self,
                    "Turn Off?",
                    "Turning off Custom Continuum will clear your progress. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    self.custom_continuum_toggle.setChecked(True)
                    return
                else:
                    self.clear_custom_progress()

            # Hide custom buttons
            self.hide_all_action_buttons()


    def show_automatic_buttons(self):
        """Show only automatic continuum action buttons"""
        self.hide_all_action_buttons()
        self.continuum_line_btn.show()
        self.subtract_continuum_btn.show()


    def show_custom_buttons(self):
        """Show only custom continuum action buttons"""
        self.hide_all_action_buttons()
        self.continuum_line_btn.show()  # Also in custom mode
        self.custom_subtract_btn.show()
        self.custom_ok_btn.show()


    def hide_all_action_buttons(self):
        """Hide all continuum action buttons"""
        self.continuum_line_btn.hide()
        self.subtract_continuum_btn.hide()
        self.custom_subtract_btn.hide()
        self.custom_ok_btn.hide()


    def has_automatic_progress(self):
        """Check if there's any automatic continuum progress"""
        has_progress = (
            self.continuum_line_btn.isChecked() or
            self.subtract_continuum_btn.isChecked() or
            (hasattr(self, 'continuum_fitted') and self.continuum_fitted is not None) or
            (hasattr(self, 'continuum_subtracted') and self.continuum_subtracted)
        )
        return has_progress


    def has_custom_progress(self):
        """Check if there's any custom continuum progress"""
        has_progress = (
            self.custom_subtract_btn.isChecked() or
            (hasattr(self, 'custom_continuum_active') and self.custom_continuum_active) or
            (hasattr(self, 'custom_continuum_applied') and self.custom_continuum_applied)
        )
        return has_progress


    def clear_automatic_progress(self):
        """Clear all automatic continuum progress"""
        try:
            # Uncheck buttons
            self.continuum_line_btn.setChecked(False)
            self.subtract_continuum_btn.setChecked(False)

            # Remove continuum line visual
            if hasattr(self, 'continuum_line') and self.continuum_line is not None:
                try:
                    self.continuum_line.remove()
                except:
                    pass
                self.continuum_line = None

            # Restore original spectrum if subtracted
            if hasattr(self, 'continuum_subtracted') and self.continuum_subtracted:
                self.restore_original_spectrum()

            # Clear fitted continuum
            self.continuum_fitted = None
            self.continuum_subtracted = False

            if hasattr(self, 'global_continuum_applied'):
                self.global_continuum_applied = False

            print("[DEBUG] Automatic continuum progress cleared")
        except Exception as e:
            print(f"[DEBUG] Error clearing automatic progress: {e}")


    def clear_custom_progress(self):
        """Clear all custom continuum progress"""
        try:
            # Uncheck button
            self.custom_subtract_btn.setChecked(False)

            # Deactivate custom continuum selector
            if hasattr(self, 'custom_continuum_selector') and self.custom_continuum_selector is not None:
                self.custom_continuum_selector.clear_all()
                self.custom_continuum_selector.deactivate()

            # Restore original spectrum if applied
            if hasattr(self, 'custom_continuum_applied') and self.custom_continuum_applied:
                self.restore_from_custom_subtract()

            self.custom_continuum_active = False
            self.custom_continuum_applied = False

            print("[DEBUG] Custom continuum progress cleared")
        except Exception as e:
            print(f"[DEBUG] Error clearing custom progress: {e}")


    def on_custom_ok_clicked(self):
        """Handle OK button click in custom continuum mode"""
        if not self.custom_continuum_active:
            QMessageBox.warning(self, "Not Active", "Custom continuum is not active. Click 'Custom Subtract' first.")
            return

        if not hasattr(self, 'custom_continuum_selector') or self.custom_continuum_selector is None:
            QMessageBox.warning(self, "No Selector", "Custom continuum selector not initialized.")
            return

        # DIAGNOSTIC: Check what attributes the selector actually has
        print(f"[DEBUG] CustomContinuumSelector attributes: {dir(self.custom_continuum_selector)}")

        # Try different possible attribute names
        num_points = 0
        if hasattr(self.custom_continuum_selector, 'points'):
            num_points = len(self.custom_continuum_selector.points)
            print(f"[DEBUG] Found 'points' attribute with {num_points} points")
        elif hasattr(self.custom_continuum_selector, 'selected_points'):
            num_points = len(self.custom_continuum_selector.selected_points)
            print(f"[DEBUG] Found 'selected_points' attribute with {num_points} points")
        elif hasattr(self.custom_continuum_selector, 'continuum_points'):
            num_points = len(self.custom_continuum_selector.continuum_points)
            print(f"[DEBUG] Found 'continuum_points' attribute with {num_points} points")
        elif hasattr(self.custom_continuum_selector, '_points'):
            num_points = len(self.custom_continuum_selector._points)
            print(f"[DEBUG] Found '_points' attribute with {num_points} points")
        else:
            # Last resort: check all list-like attributes
            for attr_name in dir(self.custom_continuum_selector):
                if not attr_name.startswith('_'):
                    attr = getattr(self.custom_continuum_selector, attr_name)
                    if isinstance(attr, (list, tuple)) and len(attr) > 0:
                        print(f"[DEBUG] Found possible points list: {attr_name} with {len(attr)} items")
                        num_points = len(attr)
                        break

            if num_points == 0:
                QMessageBox.critical(
                    self,
                    "Attribute Error",
                    "CustomContinuumSelector does not have a valid points attribute.\n\n"
                    f"Available attributes:\n{', '.join([a for a in dir(self.custom_continuum_selector) if not a.startswith('_')])}"
                )
                return

        if num_points < 3:
            QMessageBox.warning(
                self,
                "Not Enough Points",
                f"You need at least 3 points to fit a continuum. You have {num_points} point(s).\n\n"
                "Click on the spectrum to add continuum points, then click OK."
            )
            return

        print(f"[DEBUG] OK button: Fitting custom continuum with {num_points} points")

        # Fit and apply the custom continuum
        try:
            continuum = self.custom_continuum_selector.fit_continuum()
            if continuum is not None:
                self._perform_custom_subtraction(continuum)
                QMessageBox.information(
                    self,
                    "Success",
                    f"Custom continuum calculated and applied using {num_points} points."
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply custom continuum: {str(e)}")
            traceback.print_exc()
            return

        # Deactivate the selector
        self.custom_continuum_selector.deactivate()
        self.custom_continuum_active = False

    # ================== CO Flux Extraction Methods ==================
    
    def _extract_co_flux_map(self):
        """
        Extract CO rovibrational flux from entire cube.
        Includes validation checks and preview before processing.
        """
        # Validate cube
        if not hasattr(self, 'cube_data') or self.cube_data is None or self.cube_data.cube is None:
            QMessageBox.warning(self, "No Data", "Load a FITS cube first")
            return
        
        # Validate catalogs
        if not hasattr(self, 'CO_lines') or self.CO_lines is None:
            QMessageBox.warning(self, "Missing Catalog", "CO line catalog not loaded")
            return
        
        # Contaminating lines are optional - user can load them or skip
        if not hasattr(self, 'contam_lines'):
            self.contam_lines = None
        
        # Step 1: Require automatic continuum subtraction for CO flux extraction
        reply = QMessageBox.question(
            self,
            "Automatic Continuum Required",
            "CO Flux extraction requires automatic continuum subtraction.\n\n"
            "Did you already run:\n"
            "  Continuum Tools > Automatic Continuum > Continuum Line?\n\n"
            "If NOT, please do that first before proceeding.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            print("CO Flux Map extraction cancelled - automatic continuum required")
            return
        
        # Step 2: Get wavelength range from GUI
        try:
            wave_min = float(self.min_wave_input.text())
            wave_max = float(self.max_wave_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Wavelength", "Enter valid min/max wavelength values.")
            return
        
        if wave_min >= wave_max:
            QMessageBox.warning(self, "Invalid Range", "Min wavelength must be less than Max.")
            return
        
        # Check if we have cached results with the same parameters
        if hasattr(self, 'co_flux_result_cache') and hasattr(self, 'co_flux_params_cache'):
            # Check if parameters match
            params_match = (
                self.co_flux_params_cache.get('wave_min') == wave_min and
                self.co_flux_params_cache.get('wave_max') == wave_max and
                self.co_flux_params_cache.get('subtract_continuum') == (hasattr(self, 'auto_continuum_subtracted') and self.auto_continuum_subtracted)
            )
            # Check if HITRAN lines match
            hitran_match = (
                hasattr(self, 'co_flux_hitran_cache') and
                len(self.co_flux_hitran_cache) == len(self.hitran_lines)
            )
            
            if params_match and hitran_match:
                # Use cached results
                reply = QMessageBox.question(
                    self,
                    "Cached Results Available",
                    "Results from the last extraction with these parameters are cached.\n\n"
                    "Click 'Yes' to reopen the cached results (instant)\n"
                    "Click 'No' to re-extract with current data",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    results_dialog = COFluxMapResultsDialog(
                        self.co_flux_result_cache['flux_map'],
                        self.co_flux_result_cache['detection_map'],
                        header=self.co_flux_result_cache.get('header'),
                        wave_range=self.co_flux_result_cache.get('wave_range'),
                        wavelength=getattr(self, 'co_flux_wavelength', None),
                        cube_data=self.cube_data,
                        CO_catalog=getattr(self, 'co_flux_catalog', None),
                        contam_catalog=getattr(self, 'co_flux_contam_catalog', None),
                        subtract_continuum=getattr(self, 'co_flux_subtract_continuum', False),
                        parent=self,
                        excluded_wavelengths=getattr(self, 'co_flux_excluded_wavelengths', set())
                    )
                    results_dialog.exec_()
                    return
        
        # Step 3: Get HITRAN lines for the wavelength range
        if not hasattr(self, 'hitran_lines') or not self.hitran_lines:
            QMessageBox.warning(
                self,
                "No HITRAN Data",
                "Click the HITRAN button first to fetch HITRAN lines for the wavelength range."
            )
            return
        
        # Filter HITRAN lines by wavelength range
        hitran_in_range = [
            line for line in self.hitran_lines
            if wave_min <= float(line.get('wavelength', 0)) <= wave_max
        ]
        
        if not hitran_in_range:
            QMessageBox.warning(
                self,
                "No HITRAN Lines",
                f"No HITRAN lines found in wavelength range {wave_min:.4f} - {wave_max:.4f} μm"
            )
            return
        
        print(f"✓ Found {len(hitran_in_range)} HITRAN lines in range {wave_min:.4f} - {wave_max:.4f} μm")
        
        # Step 4: Filter lines for quality
        # a. Keep only HITRAN lines (ignore contaminants already done)
        # b. Remove close duplicates (keep highest peak)
        # c. Remove lines with no peak
        filtered_lines = self._filter_co_flux_lines(hitran_in_range)
        
        if not filtered_lines:
            QMessageBox.warning(self, "No Lines", "No valid lines after filtering.")
            return
        
        print(f"✓ After filtering: {len(filtered_lines)} lines")
        
        # Step 5: Show preview from a representative pixel (center pixel)
        cube_shape = self.cube_data.cube.shape
        center_y = cube_shape[1] // 2
        center_x = cube_shape[2] // 2
        
        print(f"  Showing preview from center pixel ({center_x}, {center_y})...")
        
        # Get spectrum from center pixel
        spectrum_flux = self.cube_data.get_pixel_spectrum(center_x, center_y)
        spectrum_wave = self.cube_data.wavelength
        if hasattr(spectrum_wave, 'unit'):
            spectrum_wave = spectrum_wave.value
        
        # Step 6: Ask for confirmation with preview
        confirmation = self._show_co_flux_confirmation(
            filtered_lines, spectrum_wave, spectrum_flux, center_x, center_y
        )
        
        if confirmation != QMessageBox.Yes:
            print("CO Flux Map extraction cancelled by user")
            return
        
        # Step 7: Proceed with full extraction
        import os
        try:
            cpu_count = os.cpu_count() or 8
            # Reduce workers for CPU-bound continuum fitting (median filter + peak detection is expensive)
            # Use fewer workers to avoid context switching overhead
            n_jobs = min(cpu_count, 8)
        except:
            n_jobs = 8
        
        # CRITICAL: Use the filtered HITRAN lines if available, NOT the static CO catalog
        # When user clicks HITRAN, we get isotopologue-filtered lines in self.hitran_lines
        # Use those instead of the full CO_lines catalog which has all isotopologues
        if hasattr(self, 'hitran_lines') and self.hitran_lines:
            # Convert HITRAN data format (list of dicts) to DataFrame with 'line_wave_um' column
            import pandas as pd
            hitran_df = pd.DataFrame(self.hitran_lines)
            # Rename 'wavelength' to 'line_wave_um' if needed
            if 'wavelength' in hitran_df.columns and 'line_wave_um' not in hitran_df.columns:
                hitran_df = hitran_df.rename(columns={'wavelength': 'line_wave_um'})
            use_catalog = hitran_df
        else:
            use_catalog = self.CO_lines
        
        # Store parameters for later use in result dialog
        self.co_flux_wavelength = self.cube_data.wavelength
        self.co_flux_catalog = use_catalog
        self.co_flux_contam_catalog = self.contam_lines
        self.co_flux_wave_min = wave_min
        self.co_flux_wave_max = wave_max
        
        # Use continuum subtraction - ALWAYS enabled (user warned to run automatic continuum first)
        use_continuum_subtraction = True  # Hardcoded to True as per user requirement
        self.co_flux_subtract_continuum = use_continuum_subtraction
        
        print(f"\n{'='*70}")
        print(f">>> CO FLUX MAP EXTRACTION PARAMETERS")
        print(f"{'='*70}")
        print(f"Wavelength range: {wave_min:.4f} - {wave_max:.4f} μm")
        print(f"CO line catalog: {len(use_catalog)} lines")
        print(f"Continuum subtraction: ENABLED")
        print(f"  Method: Median filter + peak detection")
        print(f"  Parameters: window=51, sigma=2.0, peak_dist=3, iterations=2")
        print(f"{'='*70}\n")
        
        self.co_flux_worker = COFluxMapWorker(
            self.cube_data,
            use_catalog,
            self.contam_lines,
            n_jobs=n_jobs,
            subtract_continuum=use_continuum_subtraction,
            continuum_poly_deg=3,  # RANSAC polynomial degree
            wave_min=wave_min,
            wave_max=wave_max,
            excluded_wavelengths=getattr(self, 'co_flux_excluded_wavelengths', set())
        )
        
        self.co_flux_progress = QProgressDialog(
            "Extracting CO flux from all pixels...", "Cancel", 0, 0, self
        )
        self.co_flux_progress.setWindowModality(Qt.WindowModal)
        self.co_flux_progress.canceled.connect(self._cancel_co_flux_extraction)
        
        self.co_flux_worker.progress.connect(self._update_co_flux_progress)
        self.co_flux_worker.finished.connect(self._on_co_flux_map_complete)
        self.co_flux_worker.error.connect(self._on_co_flux_map_error)
        
        self.co_flux_worker.start()
    
    def _filter_co_flux_lines(self, lines):
        """
        Filter CO lines for quality:
        - Remove duplicates (keep only highest peak among close lines)
        - Remove lines with no peak
        - Keep only HITRAN lines
        """
        filtered = []
        distance_threshold = 0.002  # 0.002 μm as close threshold
        
        # Sort by wavelength
        sorted_lines = sorted(lines, key=lambda x: float(x.get('wavelength', 0)))
        
        skip_indices = set()
        for i, line in enumerate(sorted_lines):
            if i in skip_indices:
                continue
            
            wave = float(line.get('wavelength', 0))
            peak = line.get('a_coeff') or line.get('intensity')  # Various peak metrics
            
            # Filter c: Skip lines with no peak
            if not peak or peak == 0:
                print(f"  Skipping line at {wave:.6f} μm (no peak)")
                continue
            
            # Filter b: Check for close neighbors and keep only highest
            neighbors = []
            for j in range(i + 1, len(sorted_lines)):
                neighbor_wave = float(sorted_lines[j].get('wavelength', 0))
                if abs(neighbor_wave - wave) > distance_threshold:
                    break
                neighbor_peak = sorted_lines[j].get('a_coeff') or sorted_lines[j].get('intensity')
                if neighbor_peak:
                    neighbors.append((j, neighbor_peak))
            
            # Keep only if this is the highest peak among neighbors
            is_highest = True
            if neighbors:
                max_neighbor_peak = max(n[1] for n in neighbors)
                if peak < max_neighbor_peak:
                    is_highest = False
                    skip_indices.update(n[0] for n in neighbors)
                else:
                    # Mark neighbors to skip
                    skip_indices.update(n[0] for n in neighbors)
            
            if is_highest:
                filtered.append(line)
        
        return filtered
    
    def _show_co_flux_confirmation(self, lines, spectrum_wave, spectrum_flux, pixel_x, pixel_y):
        """
        Show preview of gaussians on a representative pixel and ask for confirmation.
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        
        # Apply continuum subtraction if enabled (same as extraction will do)
        use_continuum_subtraction = hasattr(self, 'auto_continuum_subtracted') and self.auto_continuum_subtracted
        preview_flux = spectrum_flux.copy()
        
        if use_continuum_subtraction:
            try:
                from .co_flux_extractor import fit_continuum_median_peaks
                continuum = fit_continuum_median_peaks(spectrum_wave, preview_flux)
                if continuum is not None and not np.all(np.isnan(continuum)):
                    preview_flux = preview_flux - continuum
            except Exception:
                pass  # Use original flux if continuum fit fails
        
        # Create dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("CO Flux Map Preview")
        dlg.setGeometry(100, 100, 1000, 600)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(
            f"<b>CO Flux Map Preview</b><br>"
            f"Pixel ({pixel_x}, {pixel_y}) | "
            f"{len(lines)} filtered lines | "
            f"Wavelength range: {float(self.min_wave_input.text()):.4f} - {float(self.max_wave_input.text()):.4f} μm"
        )
        layout.addWidget(title)
        
        # Create matplotlib figure for preview
        fig = Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        spectrum_label = 'Continuum-Subtracted Spectrum' if use_continuum_subtraction else 'Spectrum'
        ax.plot(spectrum_wave, preview_flux, 'k-', linewidth=1.2, label=spectrum_label, alpha=0.8)
        
        # Mark detected lines
        wave_min = float(self.min_wave_input.text())
        wave_max = float(self.max_wave_input.text())
        
        for line in lines:
            wave = float(line.get('wavelength', 0))
            if wave_min <= wave <= wave_max:
                ax.axvline(wave, color='red', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Flux')
        ax.set_title(f'CO Line Positions ({len(lines)} lines)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        canvas = FigureCanvas(fig)
        
        # Create plot layout with toolbar for zoom/pan interaction
        plot_layout = QVBoxLayout()
        toolbar = NavigationToolbar(canvas, dlg)
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(canvas)
        
        # Add plot to main layout
        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)
        layout.addWidget(plot_widget)
        
        # Info text
        info_text = QLabel(
            f"<b>Filtered Lines Summary:</b><br>"
            f"• Total Lines: {len(lines)}<br>"
            f"• Filters Applied: HITRAN only, highest peak among close lines, no zero-peak lines<br><br>"
            f"<b>Next Steps:</b><br>"
            f"1. Gaussians will be fitted to each line at each pixel<br>"
            f"2. Integrated flux map will be calculated<br>"
            f"3. Results will be displayed<br><br>"
            f"<b>Proceed with extraction?</b>"
        )
        layout.addWidget(info_text)
        
        # Buttons
        btn_layout = QHBoxLayout()
        yes_btn = QPushButton("YES - Extract Map")
        no_btn = QPushButton("NO - Cancel")
        yes_btn.clicked.connect(dlg.accept)
        no_btn.clicked.connect(dlg.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(yes_btn)
        btn_layout.addWidget(no_btn)
        layout.addLayout(btn_layout)
        
        dlg.setLayout(layout)
        
        result = dlg.exec_()
        return QMessageBox.Yes if result == QDialog.Accepted else QMessageBox.No
    
    def _update_co_flux_progress(self, current, total):
        """Update progress."""
        self.co_flux_progress.setMaximum(total)
        self.co_flux_progress.setValue(current)
        self.co_flux_progress.setLabelText(f"Extracting CO flux: {current}/{total} pixels")
    
    def _on_co_flux_map_complete(self, result):
        """Handle successful extraction."""
        self.co_flux_progress.close()
        
        # Cache the results for later reopening
        self.co_flux_result_cache = result
        self.co_flux_hitran_cache = self.hitran_lines.copy() if hasattr(self, 'hitran_lines') else []
        self.co_flux_params_cache = {
            'wave_min': float(self.min_wave_input.text()),
            'wave_max': float(self.max_wave_input.text()),
            'subtract_continuum': hasattr(self, 'auto_continuum_subtracted') and self.auto_continuum_subtracted,
        }
        
        # Show results with export option
        results_dialog = COFluxMapResultsDialog(
            result['flux_map'],
            result['detection_map'],
            header=result.get('header'),
            wave_range=result.get('wave_range'),
            wavelength=getattr(self, 'co_flux_wavelength', None),
            cube_data=self.cube_data,
            CO_catalog=getattr(self, 'co_flux_catalog', None),
            contam_catalog=getattr(self, 'co_flux_contam_catalog', None),
            subtract_continuum=getattr(self, 'co_flux_subtract_continuum', False),
            parent=self,
            excluded_wavelengths=getattr(self, 'co_flux_excluded_wavelengths', set())
        )
        results_dialog.exec_()
    
    def _on_co_flux_map_error(self, error_msg):
        """Handle extraction error."""
        self.co_flux_progress.close()
        if "cancelled" not in error_msg.lower():
            QMessageBox.critical(self, "Extraction Error", error_msg)
        else:
            QMessageBox.information(self, "Extraction Cancelled", "CO flux extraction was cancelled")
    
    def _cancel_co_flux_extraction(self):
        """Cancel the ongoing extraction."""
        if hasattr(self, 'co_flux_worker') and self.co_flux_worker:
            self.co_flux_worker.request_cancel()
            self.co_flux_progress.setLabelText("Cancelling extraction...")
    
    def _show_co_flux_map_results(self, result):
        """Display results (called from _on_co_flux_map_complete)."""
        pass  # Handled in _on_co_flux_map_complete directly