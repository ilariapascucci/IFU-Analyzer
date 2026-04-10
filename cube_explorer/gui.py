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
from .gui_continuum import GuiContinuumMixin
from .gui_gaussian import GuiGaussianMixin
from .gui_coflux import GuiCOFluxMixin
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



class CubeExplorer(QWidget, GuiContinuumMixin, GuiGaussianMixin, GuiCOFluxMixin):
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

        self.two_d_btn = QPushButton("2D")
        self.two_d_btn.setToolTip("Open a 2D FITS analyzer window")
        self.two_d_btn.clicked.connect(self.open_2d_window)
        
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
        self.custom_delete_btn = QPushButton("Delete Point")
        self.custom_ok_btn = QPushButton("OK")

        self.custom_save_pts_btn = QPushButton("Save Points")
        self.custom_load_pts_btn = QPushButton("Load Points")

        self.custom_save_pts_btn.setToolTip("Save custom continuum points to a .txt file")
        self.custom_load_pts_btn.setToolTip("Load custom continuum points from a .txt file")

        self.save_spectrum_btn = QPushButton("Save Spectrum")
        self.save_continuum_btn = QPushButton("Save Continuum")

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
        self.save_spectrum_btn.setToolTip("Save current spectrum to file")
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
        self.save_continuum_btn.clicked.connect(self._save_continuum_ascii)
        self.subtract_continuum_btn.clicked.connect(self._activate_continuum_subtraction)
        
        # Set tooltips for all buttons
        self.set_left_btn.setToolTip("Click on the left slice to set the wavelength from that pixel")
        self.set_right_btn.setToolTip("Click on the right slice to set the wavelength from that pixel")
        self.shift_btn.setToolTip("Apply Doppler shift to the spectrum based on velocity")
        self.gauss_left_btn.setToolTip("Click on spectrum to set left boundary of Gaussian fit")
        self.gauss_peak_btn.setToolTip("Click on spectrum to set peak center of Gaussian fit")
        self.gauss_right_btn.setToolTip("Click on spectrum to set right boundary of Gaussian fit")
        self.gauss_fit_btn.setToolTip("Fit a Gaussian to the selected line")
        self.save_spectrum_btn.setToolTip("Save the current spectrum to an ASCII file")
        self.save_continuum_btn.setToolTip("Save the fitted continuum to an ASCII file")
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
        self.continuum_action_layout.addWidget(self.custom_delete_btn)
        self.continuum_action_layout.addWidget(self.custom_save_pts_btn)
        self.continuum_action_layout.addWidget(self.custom_load_pts_btn)
        self.continuum_action_layout.addWidget(self.custom_ok_btn)
        self.continuum_action_layout.addWidget(self.save_spectrum_btn)
        self.continuum_action_layout.addWidget(self.save_continuum_btn)
        self.continuum_action_layout.addStretch()
        main_layout.addLayout(self.continuum_action_layout)

        self.setLayout(main_layout)        

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

    def _save_continuum_ascii(self):
        """Save the fitted continuum to an ASCII file"""
        # Ensure a continuum has actually been fitted
        if not hasattr(self, 'continuum_fitted') or self.continuum_fitted is None:
            QMessageBox.warning(self, "No Data", "No continuum has been fitted yet. Please fit a continuum first.")
            return

        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "No spectrum loaded.")
            return
        
        wavelength = _extract_wavelength_value(self.spec_plotter.wavelength)
        if wavelength is None:
            QMessageBox.warning(self, "No Data", "No wavelength data available.")
            return
        
        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Continuum as ASCII",
            "continuum.txt",
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
                f.write(f"# Continuum Export from Cube Explorer\n")
                f.write(f"# Instrument: {instrument}\n")
                f.write(f"# Type: fitted_continuum\n")
                f.write(f"# Wavelength unit: {wave_unit}\n")
                f.write(f"# Flux unit: {self.cube_data.header.get('BUNIT', 'Unknown') if hasattr(self.cube_data, 'header') else 'Unknown'}\n")
                f.write(f"#\n")
                f.write(f"# Wavelength ({wave_unit})\tContinuum Flux\n")
                
                # Write data (skipping any NaN values)
                for w, c in zip(wavelength, self.continuum_fitted):
                    if np.isfinite(c):
                        f.write(f"{w:.6e}\t{c:.6e}\n")
            
            QMessageBox.information(self, "Success", f"Continuum saved to:\n{file_path}")
            print(f"Continuum saved to {file_path}")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save continuum:\n{str(e)}")
            print(f"Error saving continuum: {e}")

    def open_2d_window(self):
        try:
            from .ifu_analyzer_2d import Ifu_analyzer_2d_window  # new file/module

            two_d_window = Ifu_analyzer_2d_window()
            two_d_window.resize(self.width(), self.height())
            two_d_window.show()

            if not hasattr(self, "childwindows") or self.childwindows is None:
                self.childwindows = []
            self.childwindows.append(two_d_window)

        except Exception as e:
            QMessageBox.warning(self, "2D Window Error", f"Could not open 2D window:\n{e}")


