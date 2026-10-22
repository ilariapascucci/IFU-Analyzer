"""
Main GUI class for the Cube Explorer application
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suppress FITS warning messages about header formatting
warnings.filterwarnings('ignore', category=Warning, module='astropy.wcs.wcs')
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QMessageBox, QTextEdit, QInputDialog, QDialog, QToolTip,
    QListWidget, QListWidgetItem, QFrame, QDoubleSpinBox,
    QSizePolicy
)
from PyQt5.QtCore import QPoint, QPropertyAnimation, QEasingCurve
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .data_loader import CubeData
from .gui_continuum import GuiContinuumMixin
from .gui_gaussian import GuiGaussianMixin
from .gui_flux_map import GuiFluxMapMixin
from .gui_spectrum import GuiSpectrumMixin
from .gui_region_selection import RegionSelectionMixin
from .gui_mixins import DualPanelViewMixin
from .visualization import SpectrumPlotter, SlicePlotter
from .colormap_dialog import ColorMapDialog
from .scale_dialog import ScaleDialog, ScaleSettings, scaling_stats_text
from .region_selector import RegionSelector
from .utils import make_2d_wcs
from . import region_io
from .line_catalog import get_available_molecules, hitran_gui_interface, load_contaminating_lines_catalog, load_custom_line_catalog
from .transition_patterns import TRANSITION_PATTERNS
from .molecule_transition_filters import apply_transition_filter
from .species_dialog import SpeciesDialog
from .unit_converter import convert_wavelength, are_units_compatible, clean_flux_unit, clean_wave_unit, doppler_velocity_to_wavelength_factor, extract_value_from_quantity


class CollapsibleButtonGroup(QWidget):
    """
    A collapsible button group that expands/collapses with animation.
    Uses snake_case for all method and variable names.
    Supports accordion behavior - only one group can be expanded at a time.
    """
    # Class-level list to track all instances for accordion behavior
    _all_groups = []
    
    def __init__(self, title, parent=None):
        """Build a collapsible group and register it for accordion behavior.

        Args:
            title (str): Text label shown on the toggle button.
            parent (QWidget): Parent widget, or None for a top-level group.

        Returns:
            None
        """
        super().__init__(parent)
        self.is_expanded = False
        self.title = title
        # Register this instance globally
        CollapsibleButtonGroup._all_groups.append(self)
        self.init_ui(title)

    def init_ui(self, title):
        """Build the toggle button and the collapsible button container.

        Args:
            title (str): Text label shown on the toggle button.

        Returns:
            None
        """
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
        """Add a button to this collapsible group.

        Args:
            button (QPushButton): Button to add to the group's container.

        Returns:
            None
        """
        # Use default button styling (no custom stylesheet)
        self.button_layout.addWidget(button)

    def close_other_groups(self):
        """Close all other collapsible groups in this window (accordion behavior).

        Args:
            None

        Returns:
            None
        """
        # Groups in other windows keep their own open state.
        for group in CollapsibleButtonGroup._all_groups:
            if group is not self and group.is_expanded and group.window() is self.window():
                group.close()

    def close(self):
        """Close/collapse this group.

        Args:
            None

        Returns:
            None
        """
        if self.is_expanded:
            self.toggle_btn.setChecked(False)
            self.toggle()

    def toggle(self):
        """Toggle expand/collapse with animation.

        Args:
            None

        Returns:
            None
        """
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
        """Build the toolbar and set every custom action callback to None.

        Args:
            canvas (FigureCanvas): Matplotlib canvas the toolbar controls.
            parent (QWidget): Parent widget, or None for a top-level toolbar.
            coordinates (bool): Whether to show the cursor position readout.

        Returns:
            None
        """
        super().__init__(canvas, parent, coordinates)
        # Initialize custom action callbacks
        self.custom_home_action = None
        self.custom_back_action = None
        self.custom_forward_action = None
        self.custom_pan_action = None
        self.custom_zoom_action = None
        self.custom_save_action = None
    
    def home(self, *args):
        """Run the home button's custom action if set, else the default.

        Args:
            *args: Positional arguments matplotlib passes to the handler.

        Returns:
            None
        """
        if self.custom_home_action is not None:
            self.custom_home_action(*args)
        else:
            super().home(*args)

    def back(self, *args):
        """Run the back button's custom action if set, else the default.

        Args:
            *args: Positional arguments matplotlib passes to the handler.

        Returns:
            None
        """
        if self.custom_back_action is not None:
            self.custom_back_action(*args)
        else:
            super().back(*args)

    def forward(self, *args):
        """Run the forward button's custom action if set, else the default.

        Args:
            *args: Positional arguments matplotlib passes to the handler.

        Returns:
            None
        """
        if self.custom_forward_action is not None:
            self.custom_forward_action(*args)
        else:
            super().forward(*args)

    def pan(self, *args):
        """Run the pan button's custom action if set, else the default.

        Args:
            *args: Positional arguments matplotlib passes to the handler.

        Returns:
            None
        """
        if self.custom_pan_action is not None:
            self.custom_pan_action(*args)
        else:
            super().pan(*args)

    def zoom(self, *args):
        """Run the zoom button's custom action if set, else the default.

        Args:
            *args: Positional arguments matplotlib passes to the handler.

        Returns:
            None
        """
        if self.custom_zoom_action is not None:
            self.custom_zoom_action(*args)
        else:
            super().zoom(*args)

    def save_figure(self, *args):
        """Run the save button's custom action if set, else the default.

        Args:
            *args: Positional arguments matplotlib passes to the handler.

        Returns:
            None
        """
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
        """Create the spin box with no wavelength grid set yet.

        Args:
            parent (QWidget): Parent widget, or None for a top-level box.

        Returns:
            None
        """
        super().__init__(parent)
        self._wavelengths = None  # sorted ascending numpy array, or None

    def set_wavelength_grid(self, wavelengths):
        """Set the wavelength grid the spin arrows and keys step through.

        Args:
            wavelengths (numpy.ndarray): Sorted ascending wavelength values.

        Returns:
            None
        """
        self._wavelengths = wavelengths

    def stepBy(self, steps):
        """Move the value to the next/previous wavelength in the grid.

        Args:
            steps (int): Number of grid steps to move; negative steps back.

        Returns:
            None
        """
        wl = self._wavelengths
        if wl is None or len(wl) == 0:
            super().stepBy(steps)
            return
        idx = int(np.abs(wl - self.value()).argmin())
        idx = max(0, min(len(wl) - 1, idx + steps))
        self.setValue(float(wl[idx]))


class CubeExplorer(QWidget, GuiContinuumMixin, GuiGaussianMixin, GuiFluxMapMixin, GuiSpectrumMixin,
                   RegionSelectionMixin, DualPanelViewMixin):
    def __init__(self):
        """Build the main window.

        Args:
            None

        Returns:
            None
        """
        super().__init__()
        self.setWindowTitle("Cube Explorer")
        self.cube_data = CubeData()
        self._init_ui()
        # The user loads contaminating lines later, so there are none at startup.
        self.contam_lines = None

    def closeEvent(self, event):
        """Cancel any running Gaussian map fit and free cube memory on close.

        Args:
            event (QCloseEvent): Close event Qt sends to the widget.

        Returns:
            None
        """
        # A Gaussian map fit runs in its own QThread that keeps computing after
        # this window closes unless told to stop; cancel it and wait briefly.
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

    def _clear_all_displays(self):
        """Clear all plots, selections, and stats when loading a new cube.

        Args:
            None

        Returns:
            None
        """
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
        for btn in (self.region_save_slice_btn, self.region_save_slice_masked_btn, self.region_save_cube_btn,
                    self.region_save_wcs_btn, self.region_import_wcs_btn):
            btn.setEnabled(False)
        self.original_spectrum = None
        print("[DEBUG] Resetting selections")

        print("[DEBUG] Clearing continuum-related data...")

        if hasattr(self, 'custom_continuum_selector') and self.custom_continuum_selector is not None:
            self.custom_continuum_selector.clear_all()
            self.custom_continuum_selector.deactivate()
            self.custom_continuum_selector = None
            self.custom_continuum_active = False
            self._custom_continuum_tab = None

        # Uncheck custom subtract button
        if hasattr(self, 'custom_subtract_btn'):
            self.custom_subtract_btn.setChecked(False)

        # Reset per-tab continuum state - a new cube invalidates every tab's
        # fitted/subtracted continuum, not just whichever tab was active.
        self.continuum_fitted = {"full": None, "left": None, "right": None}
        self.continuum_subtracted = {"full": False, "left": False, "right": False}
        self.continuum_mode_applied = {"full": None, "left": None, "right": None}

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
        """Build every UI section in order and set up initial state.

        Args:
            None

        Returns:
            None
        """
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
        """Initialize file control buttons and labels.

        Args:
            None

        Returns:
            None
        """
        self.open_btn = QPushButton("Open FITS")
        self.open_btn.clicked.connect(self.load_cube)
        self.open_btn.setToolTip("Browse and load a FITS cube file")

        self.header_btn = QPushButton("Header")
        self.header_btn.clicked.connect(self.show_header)
        self.header_btn.setToolTip("Display the FITS header information")

        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.clicked.connect(self.refresh_view)
        self.refresh_btn.setToolTip(
            "Reset the view. This clears the selections, plotted lines, Gaussian fits, "
            "and continuum state. It reloads the spectrum.")

        # New Window button: spawn another CubeExplorer window
        self.new_window_btn = QPushButton("🗗")
        self.new_window_btn.setToolTip("Open a new Cube Explorer window")
        self.new_window_btn.clicked.connect(self._open_new_window)

        self.two_d_btn = QPushButton("2D")
        self.two_d_btn.setToolTip("Open a 2D FITS analyzer window")
        self.two_d_btn.clicked.connect(self.open_2d_window)

        self.rot_diag_btn = QPushButton("Rotational Diagram")
        self.rot_diag_btn.setToolTip("Open the rotational diagram analyzer for H2 or CO.")
        self.rot_diag_btn.clicked.connect(self.open_rotational_diagram)
        
        self.instrument_label = QLabel("Instrument: -")
        self.instrument_label.setStyleSheet("font-weight: bold;")

        def separator():
            """Build a thin vertical line that splits the top-bar button groups.

            Args:
                None

            Returns:
                line (QFrame): The vertical separator line.
            """
            line = QFrame()
            line.setFixedWidth(1)
            line.setStyleSheet("background-color: palette(mid);")
            return line

        self.controls_layout = QHBoxLayout()
        self.controls_layout.addWidget(self.open_btn)
        self.controls_layout.addWidget(self.header_btn)
        self.controls_layout.addWidget(separator())
        self.controls_layout.addWidget(self.refresh_btn)
        self.controls_layout.addWidget(self.new_window_btn)
        self.controls_layout.addWidget(separator())
        self.controls_layout.addWidget(self.rot_diag_btn)
        self.controls_layout.addWidget(self.two_d_btn)
        self.controls_layout.addWidget(separator())
        self.controls_layout.addWidget(self.instrument_label)
        self.controls_layout.addStretch()

    def _init_plots(self):
        """Initialize matplotlib figures and canvases.

        Args:
            None

        Returns:
            None
        """
        # Left slice
        self.slice1_fig = Figure(figsize=(4, 4), dpi=100)
        self.slice1_canvas = FigureCanvas(self.slice1_fig)
        self.slice1_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.slice1_canvas.setMinimumHeight(200)
        self.slice1_plotter = SlicePlotter(self.slice1_fig)
        self.slice1_toolbar = CustomNavigationToolbar(self.slice1_canvas, self)
        self.slice1_canvas.mpl_connect("button_press_event", lambda e: self._on_slice_click(e, "left"))
        self.slice1_canvas.mpl_connect("motion_notify_event", lambda e: self._on_slice_hover(e, "left"))

        # Right slice
        self.slice2_fig = Figure(figsize=(4, 4), dpi=100)
        self.slice2_canvas = FigureCanvas(self.slice2_fig)
        self.slice2_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.slice2_canvas.setMinimumHeight(200)
        self.slice2_plotter = SlicePlotter(self.slice2_fig)
        self.slice2_toolbar = CustomNavigationToolbar(self.slice2_canvas, self)
        self.slice2_canvas.mpl_connect("button_press_event", lambda e: self._on_slice_click(e, "right"))
        self.slice2_canvas.mpl_connect("motion_notify_event", lambda e: self._on_slice_hover(e, "right"))

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

        # Live cursor readout under each slice (Pixel + Value)
        self.hover1_label = QLabel("Cursor Info:\n-")
        self.hover1_label.setMinimumWidth(150)
        self.hover1_label.setWordWrap(True)
        self.hover1_label.setStyleSheet("font-size: 10pt; color: #333;")

        self.hover2_label = QLabel("Cursor Info:\n-")
        self.hover2_label.setMinimumWidth(150)
        self.hover2_label.setWordWrap(True)
        self.hover2_label.setStyleSheet("font-size: 10pt; color: #333;")

        # Slice selection buttons
        self.set_left_btn = QPushButton("Set Left Slice")
        self.set_right_btn = QPushButton("Set Right Slice")

        # Typing a wavelength (or using the spin arrows) jumps that slice to
        # the closest available value, kept in sync with clicking the plots.
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
        self.gauss_left_btn = QPushButton("Fit Left")
        self.gauss_peak_btn = QPushButton("Peak")
        self.gauss_right_btn = QPushButton("Fit Right")
        self.gauss_noise_left_btn = QPushButton("Noise Left")
        self.gauss_noise_right_btn = QPushButton("Noise Right")
        self.gauss_fit_btn = QPushButton("Fit Gaussian")
        self.gauss_map_btn = QPushButton("Fit Gaussian Map")
        self.multi_gauss_fit_btn = QPushButton("Multi-Gauss")
        self.flux_map_btn = QPushButton("Flux Map")
        
        self.auto_continuum_toggle = QPushButton("Automatic Continuum")
        self.custom_continuum_toggle = QPushButton("Custom Continuum")
        self.save_continuum_cubes_btn = QPushButton("Save Continuum Cubes")
        self.save_continuum_cubes_btn.setToolTip(
            "Fit the continuum at every pixel, then save the continuum cube and the continuum-subtracted cube")
        self.save_continuum_cubes_btn.clicked.connect(self._save_continuum_cubes)

        self.auto_continuum_toggle.setCheckable(True)
        self.custom_continuum_toggle.setCheckable(True)

        # Action buttons (initially hidden)
        self.continuum_line_btn = QPushButton("Fit Continuum")
        self.subtract_continuum_btn = QPushButton("Subtract Continuum")
        self.custom_subtract_btn = QPushButton("Custom Subtract")
        self.custom_delete_btn = QPushButton("Delete Point")
        self.custom_ok_btn = QPushButton("OK")

        self.custom_save_pts_btn = QPushButton("Save Points")
        self.custom_load_pts_btn = QPushButton("Load Points")

        self.custom_save_pts_btn.setToolTip("Save custom continuum points to a .txt file")
        self.custom_load_pts_btn.setToolTip("Load custom continuum points from a .txt file")

        self.save_results_btn = QPushButton("Save Spectrum")

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
        self.custom_delete_btn.hide()

        # Tooltips
        self.auto_continuum_toggle.setToolTip("Toggle automatic continuum fitting mode")
        self.custom_continuum_toggle.setToolTip("Toggle custom continuum point selection mode")
        self.continuum_line_btn.setToolTip("Fit and display continuum line")
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
        self.flux_map_btn.clicked.connect(self._extract_flux_map)
        self.save_results_btn.clicked.connect(self._save_results_ascii)
        self.subtract_continuum_btn.clicked.connect(self._activate_continuum_subtraction)
        
        # Set tooltips for all buttons
        self.set_left_btn.setToolTip("Click on the left slice to set the wavelength from that pixel")
        self.set_right_btn.setToolTip("Click on the right slice to set the wavelength from that pixel")
        self.shift_btn.setToolTip("Preview a Doppler shift for the active tab's spectrum")
        self.gauss_left_btn.setToolTip("Click the spectrum to set the left edge of the Gaussian fit.")
        self.gauss_peak_btn.setToolTip("Click the spectrum to set the peak of the Gaussian fit.")
        self.gauss_right_btn.setToolTip("Click the spectrum to set the right edge of the Gaussian fit.")
        self.gauss_fit_btn.setToolTip("Fit a Gaussian to the selected line")
        self.save_results_btn.setToolTip(
            "Save wavelength, original spectrum, fitted continuum, "
            "and continuum-subtracted spectrum to a single ASCII file."
        )
        self.subtract_continuum_btn.setToolTip("Subtract the fitted continuum from the spectrum")
        self.gauss_map_btn.setToolTip(
            "Fit a Gaussian at every pixel. This uses Fit Left, Fit Right, Noise Left, and Noise Right.")
        self.multi_gauss_fit_btn.setToolTip("Fit Gaussians to all selected HITRAN lines using current continuum")
        self.gauss_noise_left_btn.setToolTip("Click the spectrum to set the left edge of a line-free noise region.")
        self.gauss_noise_right_btn.setToolTip("Click the spectrum to set the right edge of a line-free noise region.")
        self.flux_map_btn.setToolTip("Fit multiple lines and create an integrated flux map")


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
        self.hitran_btn = QPushButton("Fetch HITRAN")
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

        self.mark_contam_btn = QPushButton("Mark Contaminant")
        self.mark_contam_btn.setCheckable(True)
        self.mark_contam_btn.setToolTip(
            "Click, then click a feature on the spectrum to mark it as a contaminant line -- "
            "no HITRAN match needed (shown as species \"N/A\"). Stays on so you can mark "
            "several in a row; click it again to stop. To remove one, uncheck it in \"Lines\" "
            "or click its marker on the spectrum."
        )
        self.mark_contam_btn.toggled.connect(self._toggle_mark_contam_mode)

        self.show_species_btn = QPushButton("Select Lines")
        self.show_species_btn.setToolTip("Choose the lines to plot on the spectrum.")
        self.show_species_btn.clicked.connect(self.show_available_species)

    def _init_shape_controls(self):
        """Initialize shape selection buttons.

        Args:
            None

        Returns:
            None
        """
        # Helper function to create shape buttons
        def make_shape_button(symbol, target, callback=None, checkable=True):
            """Build one square icon button wired to a shape or callback.

            Args:
                symbol (str): Unicode glyph shown as the button label.
                target (str): Side the button applies to ('left' or 'right').
                callback (callable): Click handler; defaults to _set_shape.
                checkable (bool): Whether the button stays pressed when set.

            Returns:
                btn (QPushButton): The configured button.
            """
            btn = QPushButton(symbol)
            btn.setCheckable(checkable)
            btn.setFixedSize(35, 35)
            btn.setStyleSheet("QPushButton { font-size: 18pt; }")
            if callback is None:
                btn.clicked.connect(lambda _, s=symbol: self._set_shape(s, target=target))
            else:
                btn.clicked.connect(callback)
            return btn

        # Create left and right shape buttons
        shapes = ["*", "▭", "○", "⬭", "✏"]
        shape_tooltips = {
            "*": "Point selection",
            "▭": "Rectangle selection",
            "○": "Circle selection",
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
        self.color_btn_left.setFixedSize(35, 35)
        self.color_btn_left.setToolTip("Change left slice color scale")
        self.color_btn_left.clicked.connect(lambda: self._open_colormap_dialog('left'))

        self.color_btn_right = QPushButton("🌈")
        self.color_btn_right.setFixedSize(36, 36)
        self.color_btn_right.setToolTip("Change right slice color scale")
        self.color_btn_right.clicked.connect(lambda: self._open_colormap_dialog('right'))

        # Scale selection buttons - square icon buttons that open the scale/
        # range dialog, matching the other square buttons.
        self.scale_btn_left = self._make_scale_button('left')
        self.scale_btn_right = self._make_scale_button('right')

        # One copy button per panel. It opens a menu to copy the selection,
        # colormap, or scale/range from the other panel.
        self.copy_menu_btn_l = self._make_copy_button('left')
        self.copy_menu_btn_r = self._make_copy_button('right')

        # Edit Shape toggle buttons (enable drag/resize on a placed shape)
        self.edit_shape_btn_l = QPushButton("✎")
        self.edit_shape_btn_l.setCheckable(True)
        self.edit_shape_btn_l.setFixedSize(35, 35)
        self.edit_shape_btn_l.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.edit_shape_btn_l.setToolTip("Toggle Edit Shape (drag/resize)")
        self.edit_shape_btn_l.setEnabled(False)
        self.edit_shape_btn_l.clicked.connect(lambda: self._toggle_edit_shape("left"))

        self.edit_shape_btn_r = QPushButton("✎")
        self.edit_shape_btn_r.setCheckable(True)
        self.edit_shape_btn_r.setFixedSize(35, 35)
        self.edit_shape_btn_r.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.edit_shape_btn_r.setToolTip("Toggle Edit Shape (drag/resize)")
        self.edit_shape_btn_r.setEnabled(False)
        self.edit_shape_btn_r.clicked.connect(lambda: self._toggle_edit_shape("right"))

        # Erase (clear) selection buttons
        self.erase_btn_l = QPushButton("✖")
        self.erase_btn_l.setFixedSize(35, 35)
        self.erase_btn_l.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.erase_btn_l.setToolTip("Clear selection on left slice")
        self.erase_btn_l.clicked.connect(lambda: self.clear_region("left"))

        self.erase_btn_r = QPushButton("✖")
        self.erase_btn_r.setFixedSize(35, 35)
        self.erase_btn_r.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.erase_btn_r.setToolTip("Clear selection on right slice")
        self.erase_btn_r.clicked.connect(lambda: self.clear_region("right"))

        # Save/import actions live in the "Region" group (_setup_main_layout),
        # not here -- each one asks which panel (left/right) to use.
        self.region_save_slice_btn = QPushButton("Save Slice")
        self.region_save_slice_btn.setToolTip("Save one panel's current slice as a standalone 2D FITS")
        self.region_save_slice_btn.clicked.connect(self._region_save_slice)

        self.region_save_slice_masked_btn = QPushButton("Save Slice (Masked)")
        self.region_save_slice_masked_btn.setToolTip(
            "Save one panel's current slice as a 2D FITS, with everything outside its region set to NaN")
        self.region_save_slice_masked_btn.clicked.connect(self._region_save_slice_masked)

        self.region_save_cube_btn = QPushButton("Save Cube (Masked)")
        self.region_save_cube_btn.setToolTip(
            "Save the full cube, with everything outside one panel's region set to NaN")
        self.region_save_cube_btn.clicked.connect(self._region_save_cube_masked)

        self.region_save_wcs_btn = QPushButton("Save Region")
        self.region_save_wcs_btn.setToolTip(
            "Save one panel's region as RA/Dec sky coordinates, so it can be reconstructed on a different cube")
        self.region_save_wcs_btn.clicked.connect(self._region_save_wcs_text)

        self.region_import_wcs_btn = QPushButton("Import Region")
        self.region_import_wcs_btn.setToolTip(
            "Load a region previously saved as WCS text onto one panel, at the same sky position")
        self.region_import_wcs_btn.clicked.connect(self._region_import_wcs_text)

        for btn in (self.region_save_slice_btn, self.region_save_slice_masked_btn, self.region_save_cube_btn,
                    self.region_save_wcs_btn, self.region_import_wcs_btn):
            btn.setEnabled(False)

    def _setup_main_layout(self):
        """Set up the main application layout.

        Args:
            None

        Returns:
            None
        """
        main_layout = QVBoxLayout()

        # Add controls at top
        main_layout.addLayout(self.controls_layout)

        # Top area: left and right halves
        top_layout = QHBoxLayout()

        # LEFT HALF
        # Left stats column (will take 25% of left half)
        left_stats_layout = QVBoxLayout()
        left_stats_layout.addWidget(self.stats1_label)
        left_stats_layout.addSpacing(12)
        left_stats_layout.addWidget(self.hover1_label)
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
        left_sidebar.addWidget(self.scale_btn_left)
        for shape in ["*", "▭", "○", "⬭", "✏"]:
            left_sidebar.addWidget(self.shape_buttons["left"][shape])
        left_sidebar.addWidget(self.copy_menu_btn_l)
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
        right_stats_layout.addSpacing(12)
        right_stats_layout.addWidget(self.hover2_label)
        right_stats_layout.addStretch()

        right_combined = QHBoxLayout()

        right_sidebar = QVBoxLayout()
        right_sidebar.addWidget(self.color_btn_right)
        right_sidebar.addWidget(self.scale_btn_right)
        for shape in ["*", "▭", "○", "⬭", "✏"]:
            right_sidebar.addWidget(self.shape_buttons["right"][shape])
        right_sidebar.addWidget(self.copy_menu_btn_r)
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

        # --- Collapsible button groups ---
        # Create collapsible button layout
        collapsible_layout = QHBoxLayout()
        collapsible_layout.setSpacing(0)
        collapsible_layout.setContentsMargins(0, 0, 0, 0)
        
        # Group 1: Region (save/import for slices and regions -- each
        # button asks which panel, left or right, to use).
        self.region_group = CollapsibleButtonGroup("Region")
        self.region_group.add_button(self.region_save_slice_btn)
        self.region_group.add_button(self.region_save_slice_masked_btn)
        self.region_group.add_button(self.region_save_cube_btn)
        self.region_group.add_button(self.region_save_wcs_btn)
        self.region_group.add_button(self.region_import_wcs_btn)
        collapsible_layout.addWidget(self.region_group)

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
        self.continuum_group.add_button(self.save_continuum_cubes_btn)
        collapsible_layout.addWidget(self.continuum_group)

        # Group 4: HITRAN (wavelength inputs + action buttons + flux map)
        self.hitran_group = CollapsibleButtonGroup("HITRAN")
        self.hitran_group.add_button(QLabel("λ min (µm):"))
        self.hitran_group.add_button(self.min_wave_input)
        self.hitran_group.add_button(QLabel("λ max (µm):"))
        self.hitran_group.add_button(self.max_wave_input)
        self.hitran_group.add_button(self.hitran_btn)
        self.hitran_group.add_button(self.load_custom_lines_btn)
        self.hitran_group.add_button(self.load_contam_btn)
        self.hitran_group.add_button(self.mark_contam_btn)
        self.hitran_group.add_button(self.multi_gauss_fit_btn)
        self.hitran_group.add_button(self.show_species_btn)
        self.hitran_group.add_button(self.flux_map_btn)
        collapsible_layout.addWidget(self.hitran_group)

        collapsible_layout.addStretch()  # Push all groups to the left
        
        main_layout.addLayout(collapsible_layout)

        # Slice selection, spectrum tabs, and shifting share one row here
        # (not the toolbar above) since all three act on the spectrum itself.
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
        """Cancel custom continuum selection.

        Args:
            None

        Returns:
            None
        """
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
        """Initialize state variables.

        Args:
            None

        Returns:
            None
        """
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
        self.slice1_scale = ScaleSettings()
        self.slice2_scale = ScaleSettings()

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
        
        # Continuum subtraction state is independent per tab: each tab derives
        # its own raw flux via _get_tab_raw_flux, so there is no shared backup.
        self.continuum_fitted = {"full": None, "left": None, "right": None}
        self.continuum_subtracted = {"full": False, "left": False, "right": False}
        self.continuum_mode_applied = {"full": None, "left": None, "right": None}  # 'auto' | 'custom' | None
        self.custom_continuum_selector = None
        self.custom_continuum_active = False
        # Tab a custom-continuum session belongs to, captured at session start
        # so it stays correct even if the active tab changes before confirming.
        self._custom_continuum_tab = None
        self.subtract_mode = None  # 'left' or 'right'
        self.subtract_left_x = None
        self.subtract_right_x = None
        self.subtract_markers = []  # Store marker artists

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
        """Open a file dialog and load the selected FITS file.

        Args:
            None

        Returns:
            None
        """
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

            # Permanent, never-mutated reference wavelength for the per-tab shift
            # system; each tab's displayed wavelength derives from this array.
            self.base_wavelength = np.array(extract_value_from_quantity(self.cube_data.wavelength), dtype=float)

            # Update instrument label with filename
            filename = os.path.basename(self.cube_data.filename)
            self.instrument_label.setText(f"Instrument: {self.cube_data.instrument} [{filename}]")

            # Clear spec AND TICKS from previous spec
            self.spec_fig.clf()

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
                # Reset to the Full tab so a previously-active Left/Right tab
                # doesn't keep pointing at the old cube's stale selection.
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
            wavelength = extract_value_from_quantity(self.cube_data.wavelength)
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

            self._fill_wave_range_inputs()

            for btn in (self.region_save_slice_btn, self.region_save_slice_masked_btn, self.region_save_cube_btn,
                        self.region_save_wcs_btn, self.region_import_wcs_btn):
                btn.setEnabled(True)

        except MemoryError:
            QMessageBox.critical(self, "Memory Error", 
                "Not enough memory to load this file. Try closing other applications or using a smaller file.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
            print(f"Detailed error: {e}")  # For debugging

    def _fill_wave_range_inputs(self):
        """Set the Min/Max wavelength boxes to the loaded cube's full range, in micron.

        Args:
            None

        Returns:
            None
        """
        wavelength = np.asarray(extract_value_from_quantity(self.cube_data.wavelength), dtype=float)
        wavelength = wavelength[np.isfinite(wavelength)]
        if wavelength.size == 0:
            return
        try:
            low = convert_wavelength(float(wavelength.min()), self.cube_data.wave_unit, "um")
            high = convert_wavelength(float(wavelength.max()), self.cube_data.wave_unit, "um")
        except ValueError:
            # A velocity axis has no micron range; the boxes keep their defaults.
            print(f"[INFO] Min/Max wavelength unchanged: the spectral axis is in {self.cube_data.wave_unit}.")
            return
        self.min_wave_input.setValue(low)
        self.max_wave_input.setValue(high)

    def _open_new_window(self):
        """Open a new independent CubeExplorer window.

        Args:
            None

        Returns:
            None
        """
        try:
            new_win = CubeExplorer()
            new_win.resize(self.width(), self.height())
            new_win.show()
            # Keep a reference so it isn't garbage collected
            self.child_windows.append(new_win)
        except Exception as e:
            QMessageBox.warning(self, "New Window Error", f"Could not open a new window:\n{e}")


    def show_header(self):
        """Display FITS headers with HDU selection dialog.

        Args:
            None

        Returns:
            None
        """
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
            """Show the header for the HDU currently selected in the list.

            Args:
                None

            Returns:
                None
            """
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
        """Display the header for a specific HDU.

        Args:
            hdu_index (int): Index of the HDU to display within hdul.
            hdul (astropy.io.fits.HDUList): Open FITS HDU list.
            hdu_name (str): Label shown in the header window's title.

        Returns:
            None
        """
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
        """Display a wavelength/velocity slice with proper title and intensity units.

        Args:
            idx (int): Index of the spectral slice to display.
            target (str): Panel to update, 'left' or 'right'.

        Returns:
            None
        """
        image = self.cube_data.get_slice(idx)
        
        # Select plotter and stats label
        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
        stats_label = self.stats1_label if target == "left" else self.stats2_label

        # Title with spectral value
        if hasattr(self.cube_data, 'wavelength') and self.cube_data.wavelength is not None:
            # Extract wavelength value, handling Quantity objects
            wave = extract_value_from_quantity(self.cube_data.wavelength[idx])
            if self.cube_data.wave_unit == "Å":
                title = f"{target.capitalize()} Slice (idx={idx}) @ {wave:.6f} {clean_wave_unit(self.cube_data.wave_unit)}"
            elif self.cube_data.wave_unit == "um":
                title = f"{target.capitalize()} Slice (idx={idx}) @ {wave:.6f} {clean_wave_unit(self.cube_data.wave_unit)}"
            elif self.cube_data.wave_unit == "km/s":
                title = f"{target.capitalize()} Slice (idx={idx}) @ {wave:.1f} km/s"
            else:
                title = f"{target.capitalize()} Slice @ idx={idx}"
        else:
            title = f"{target.capitalize()} Slice @ idx={idx}"
        
        # Update plot using stored colormap preference per slice if available
        cmap = getattr(self, 'slice1_cmap' if target == 'left' else 'slice2_cmap', None)
        scale_settings = self.slice1_scale if target == 'left' else self.slice2_scale
        # Get the flux unit for colorbar display
        flux_unit_raw = getattr(self.cube_data, "flux_unit", "")
        flux_unit_clean = clean_flux_unit(flux_unit_raw) if flux_unit_raw else ""
        plotter.plot_slice(image, title, cmap=cmap, settings=scale_settings,
                           flux_unit=flux_unit_clean)
        
        # Intensity stats with units -- always the data's own true linear
        # values, regardless of the active display scale/range (see Scaling below).
        mn, mx, mean = np.nanmin(image), np.nanmax(image), np.nanmean(image)
        total_sum, std = np.nansum(image), np.nanstd(image)
        unit_raw = getattr(self.cube_data, "flux_unit", "arbitrary units")  # fallback if flux_unit missing
        unit = clean_flux_unit(unit_raw)
        # Scaling: the color range the plot actually uses -- distinct from the
        # data stats above, which never change with scale/range.
        stats_label.setText(
            f"Intensity (data)\n[{unit}]:\n"
            f"min={mn:.3e}\n"
            f"max={mx:.3e}\n"
            f"mean={mean:.3e}\n"
            f"sum={total_sum:.3e}\n"
            f"std={std:.3e}\n"
            f"\n"
            + scaling_stats_text(image, scale_settings)
        )

        # Force redraw
        if target == "left":
            self.slice1_canvas.draw()
            self.last_idx_left = idx
        else:
            self.slice2_canvas.draw()
            self.last_idx_right = idx

    # ----- Region selection: the shared controller lives in RegionSelectionMixin.
    # ----- These hooks feed it the 3D window's cube-backed data. ----------------
    def _region_image(self, target):
        """Return the 2D array currently shown in the given panel.

        Args:
            target (str): Panel to read from, 'left' or 'right'.

        Returns:
            image (numpy.ndarray): The displayed slice, or None if unset.
        """
        idx = self.last_idx_left if target == "left" else self.last_idx_right
        return self.cube_data.get_slice(idx)

    def _region_flux_unit(self, target):
        """Return the cleaned flux-unit string for the given panel.

        Args:
            target (str): Panel to read from, 'left' or 'right'.

        Returns:
            unit (str): Cleaned flux-unit label, or '' when unknown.
        """
        unit_raw = getattr(self.cube_data, "flux_unit", "")
        return clean_flux_unit(unit_raw) if unit_raw else ""

    def _region_stats_placeholder(self, target):
        """Return the placeholder stats-label text for the given panel.

        Args:
            target (str): Panel to build the placeholder for, 'left' or 'right'.

        Returns:
            text (str): Placeholder label text shown before any stats exist.
        """
        return "Slice 1 Stats:\n-" if target == "left" else "Slice 2 Stats:\n-"

    def _region_slice_stat_keep(self, line):
        """Keep the whole-slice Scaling/vmin/vmax lines above region stats.

        Args:
            line (str): One line of the current stats label text.

        Returns:
            keep (bool): True for a Scaling/vmin/vmax line.
        """
        return line.startswith(("Scaling:", "vmin=", "vmax="))

    # Pixel readout omits RA/Dec (sky position is often unavailable here) and
    # wavelength (already shown on the spectrum and slice title).

    def _region_on_change(self, target, kind):
        """React to a region change: redraw the spectrum and update buttons.

        Args:
            target (str): Panel the change happened on, 'left' or 'right'.
            kind (str): Kind of change (e.g. 'clear', 'select').

        Returns:
            None
        """
        if kind == "clear":
            try:
                self.spec_canvas.draw()
            except Exception:
                pass
            # clear_region already blanked the stats label to a placeholder;
            # restore the real Intensity/Scaling stats rather than leaving it stale.
            if getattr(self, "cube_data", None) is not None:
                idx = self.last_idx_left if target == "left" else self.last_idx_right
                self._display_slice(idx, target)
        if getattr(self, "active_spectrum_tab", None) == target:
            self._refresh_active_spectrum_tab()

    def _selection_to_full_mask(self, target, sel):
        """Build a full-image mask from a selection dict.

        Args:
            target (str): Panel the selection belongs to, 'left' or 'right'.
            sel (dict): Selection dict from last_selection_left/right.

        Returns:
            mask (numpy.ndarray): Boolean mask, same spatial shape as the cube.
        """
        if sel.get("shape") == "*":
            # Point selections store coords, not a mask -- build a single-pixel one.
            x, y = sel["coords"]
            x, y = int(round(x)), int(round(y))
            cube = self.cube_data.cube
            spatial_shape = cube.shape[1:] if self.cube_data.spec_axis == 0 else \
                tuple(s for i, s in enumerate(cube.shape) if i != self.cube_data.spec_axis)
            mask = np.zeros(spatial_shape, dtype=bool)
            mask[y, x] = True
            return mask
        return sel["mask"]

    def _export_region_cube(self, target):
        """Save the cube to a new FITS file with spaxels outside the
        selected region (or, for a Point selection, outside that single
        pixel) set to NaN.

        Args:
            target (str): Panel whose selection defines the region, 'left'
                or 'right'.

        Returns:
            None
        """
        sel = self.last_selection_left if target == "left" else self.last_selection_right
        if not sel:
            QMessageBox.warning(
                self, "Region Selection Required",
                f"Select a region or pixel on the {target.capitalize()} side first."
            )
            return
        mask = self._selection_to_full_mask(target, sel)

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Region as Cube", "", "FITS files (*.fits)")
        if not path:
            return
        if not path.lower().endswith(".fits"):
            path += ".fits"

        try:
            self.cube_data.export_region_cube(mask, path)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not export region cube:\n{e}")
            return
        QMessageBox.information(self, "Export Complete", f"Region cube saved to:\n{path}")

    def _region_save_slice(self):
        """Save one panel's current slice as a standalone 2D FITS file.

        Args:
            None

        Returns:
            None
        """
        if getattr(self.cube_data, "cube", None) is None:
            QMessageBox.warning(self, "No Cube Loaded", "Load a cube first.")
            return
        target = self._ask_target_panel("Save Slice", "Which panel's slice would you like to save?")
        if target is None:
            return
        idx = self.last_idx_left if target == "left" else self.last_idx_right

        path, _ = QFileDialog.getSaveFileName(self, "Save Slice as 2D FITS", "", "FITS files (*.fits)")
        if not path:
            return
        if not path.lower().endswith(".fits"):
            path += ".fits"
        try:
            self.cube_data.export_slice_fits(idx, path)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not export slice:\n{e}")
            return
        QMessageBox.information(self, "Export Complete", f"Slice saved to:\n{path}")

    def _region_save_slice_masked(self):
        """Save one panel's current slice as a 2D FITS file, with pixels
        outside that panel's region set to NaN.

        Args:
            None

        Returns:
            None
        """
        if getattr(self.cube_data, "cube", None) is None:
            QMessageBox.warning(self, "No Cube Loaded", "Load a cube first.")
            return
        target = self._ask_target_panel("Save Slice (Masked)", "Which panel's region would you like to save?")
        if target is None:
            return
        sel = self.last_selection_left if target == "left" else self.last_selection_right
        if not sel:
            QMessageBox.warning(
                self, "Region Selection Required",
                f"Select a region or pixel on the {target.capitalize()} side first.")
            return
        mask = self._selection_to_full_mask(target, sel)
        idx = self.last_idx_left if target == "left" else self.last_idx_right

        path, _ = QFileDialog.getSaveFileName(self, "Save Slice (Masked) as 2D FITS", "", "FITS files (*.fits)")
        if not path:
            return
        if not path.lower().endswith(".fits"):
            path += ".fits"
        try:
            self.cube_data.export_slice_fits(idx, path, mask=mask)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not export slice:\n{e}")
            return
        QMessageBox.information(self, "Export Complete", f"Masked slice saved to:\n{path}")

    def _region_save_cube_masked(self):
        """Export one panel's region as a NaN-masked cube (unchanged from
        the existing _export_region_cube behavior).

        Args:
            None

        Returns:
            None
        """
        if getattr(self.cube_data, "cube", None) is None:
            QMessageBox.warning(self, "No Cube Loaded", "Load a cube first.")
            return
        target = self._ask_target_panel(
            "Save Cube (Masked)", "Which panel's region would you like to export as a cube?")
        if target is None:
            return
        self._export_region_cube(target)

    def _region_save_wcs_text(self):
        """Save one panel's region as RA/Dec sky coordinates in a text
        file, so it can be placed on a different cube later.

        Args:
            None

        Returns:
            None
        """
        if getattr(self.cube_data, "cube", None) is None:
            QMessageBox.warning(self, "No Cube Loaded", "Load a cube first.")
            return
        target = self._ask_target_panel("Save Region as WCS Text", "Which panel's region would you like to save?")
        if target is None:
            return
        sel = self.last_selection_left if target == "left" else self.last_selection_right
        if not sel:
            QMessageBox.warning(
                self, "Region Selection Required",
                f"Select a region or pixel on the {target.capitalize()} side first.")
            return

        wcs2d = make_2d_wcs(self.cube_data.header)
        if wcs2d is None or not getattr(wcs2d, "has_celestial", False):
            QMessageBox.critical(
                self, "No WCS Available",
                "This cube has no usable celestial WCS, so its region can't be saved as sky coordinates.")
            return

        shape = sel.get("shape")
        lasso_verts = getattr(sel.get("patch"), "lasso_verts", None) if shape == "✏" else None
        try:
            result = region_io.region_shape_to_sky(shape, sel["coords"], wcs2d, lasso_verts=lasso_verts)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not convert region to sky coordinates:\n{e}")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Region as WCS Text", "", "Text files (*.txt)")
        if not path:
            return
        if not path.lower().endswith((".txt", ".csv")):
            path += ".txt"

        source_desc = os.path.basename(self.cube_data.filename or "")
        try:
            region_io.write_region_wcs_txt(
                path, shape, source_desc, result["pixel_points"], result["sky_points"],
                result.get("sky_center"), result.get("sky_axis_x"), result.get("sky_axis_y"))
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not write region file:\n{e}")
            return
        QMessageBox.information(self, "Export Complete", f"Region saved to:\n{path}")

    def _region_import_wcs_text(self):
        """Load a region saved as WCS text and place it on one panel,
        using this cube's own WCS to match the same sky position.

        Args:
            None

        Returns:
            None
        """
        if getattr(self.cube_data, "cube", None) is None:
            QMessageBox.warning(self, "No Cube Loaded", "Load a cube first.")
            return
        target = self._ask_target_panel("Import Region", "Which panel should the imported region be placed on?")
        if target is None:
            return

        wcs2d = make_2d_wcs(self.cube_data.header)
        if wcs2d is None or not getattr(wcs2d, "has_celestial", False):
            QMessageBox.critical(
                self, "No WCS Available",
                "This cube has no usable celestial WCS, so a saved region can't be placed on it.")
            return

        image = self._region_image(target)
        if image is None:
            QMessageBox.warning(self, "Import Failed", "No slice is currently displayed on that panel.")
            return

        path, _ = QFileDialog.getOpenFileName(self, "Import Region as WCS Text", "", "Text files (*.txt *.csv)")
        if not path:
            return

        try:
            saved = region_io.read_region_wcs_txt(path)
            placed = region_io.sky_to_region_shape(
                saved["shape"], saved["sky_points"], saved["sky_center"],
                saved["sky_axis_x"], saved["sky_axis_y"], wcs2d, target_shape=image.shape)
        except Exception as e:
            QMessageBox.warning(self, "Import Failed", str(e))
            return

        selector = self.region_selector_left if target == "left" else self.region_selector_right
        if selector.interactive_patch is not None:
            selector.interactive_patch.disconnect()
            selector.interactive_patch = None

        shape = placed["shape"]
        if shape == "*":
            x, y = placed["coords"]
            self._handle_pixel_click(int(round(x)), int(round(y)), target)
        elif shape == "✏":
            selector.active_shape = "✏"
            selector.on_lasso_select(placed["verts"])
            selector.active_shape = None
        else:
            selector.active_shape = shape
            coords = tuple(int(round(c)) for c in placed["coords"])
            selector.create_region(coords)
            selector.active_shape = None

    def _on_slice_click(self, event, target):
        """Handle clicks on slice display.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Click event.
            target (str): Panel clicked on, 'left' or 'right'.

        Returns:
            None
        """
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

    def _on_slice_hover(self, event, target):
        """Live cursor readout (pixel + value) as the mouse moves over a slice.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Motion event.
            target (str): Panel hovered over, 'left' or 'right'.

        Returns:
            None
        """
        hover_label = self.hover1_label if target == "left" else self.hover2_label
        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
        if (event.inaxes != plotter.ax or event.xdata is None or event.ydata is None
                or not getattr(self, "cube_data", None)):
            hover_label.setText("Cursor Info:\n-")
            return
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        image = self._region_image(target)
        if image is None or not (0 <= iy < image.shape[0] and 0 <= ix < image.shape[1]):
            hover_label.setText("Cursor Info:\n-")
            return
        unit = self._region_flux_unit(target)
        unit_str = f" [{unit}]" if unit else ""
        hover_label.setText(
            f"Cursor Info:\nPixel: {ix}, {iy}\nVal: {float(image[iy, ix]):.4e}{unit_str}")

    def _on_spec_click(self, event):
        """Handle clicks on the spectrum display.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Click event.

        Returns:
            None
        """
        if event.inaxes != self.spec_plotter.ax:
            return

        # "Mark Contaminant" mode takes priority over everything else below --
        # a click here always means "add a contaminant", not "deselect" or "navigate".
        if event.button == 1 and getattr(self, 'mark_contam_btn', None) and self.mark_contam_btn.isChecked() \
                and event.xdata is not None:
            self._add_manual_contaminant(event.xdata)
            return

        # LEFT CLICK: Check if clicking near a plotted spectral line to deselect it
        if event.button == 1 and hasattr(self, 'plotted_lines') and self.plotted_lines:
            # Tolerance is in screen pixels, not wavelength, so it stays small
            # and consistent across zoom; a wavelength tolerance would over-trigger when zoomed in.
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
                    
                    # Track excluded wavelength for the flux map
                    if not hasattr(self, 'flux_map_excluded_wavelengths'):
                        self.flux_map_excluded_wavelengths = set()
                    self.flux_map_excluded_wavelengths.add(line_wave)
                    
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
        wavelength = extract_value_from_quantity(self.cube_data.wavelength)
        if wavelength is None:
            idx = int(click_x)
        else:
            idx = (np.abs(wavelength - click_x)).argmin()

        self._set_slice_index(idx, self.active_target)

    def _set_slice_index(self, idx, target):
        """Set the left/right slice to wavelength index `idx` and sync every
        linked UI element. The single point spectrum clicks, slice clicks,
        and wave-input all route through to stay consistent.

        Args:
            idx (int): Wavelength index to display.
            target (str): Panel to update, 'left' or 'right'.

        Returns:
            None
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
        wavelength = extract_value_from_quantity(self.cube_data.wavelength)
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
        """Jump the slice to the closest wavelength actually in the cube
        after the user types or steps a value in the wave-input spin box.

        Args:
            target (str): Panel whose spin box changed, 'left' or 'right'.

        Returns:
            None
        """
        if not hasattr(self, 'cube_data') or not hasattr(self.cube_data, 'wavelength'):
            return
        wavelength = extract_value_from_quantity(self.cube_data.wavelength)
        if wavelength is None:
            return
        spin = self.left_wave_input if target == "left" else self.right_wave_input
        idx = int(np.abs(wavelength - spin.value()).argmin())
        self._set_slice_index(idx, target)

    def _select_target(self, target):
        """Handle slice selection mode.

        Args:
            target (str): Panel whose Set button was toggled, 'left' or
                'right'.

        Returns:
            None
        """
        if target == "left":
            self.set_right_btn.setChecked(False)
            self.active_target = "left" if self.set_left_btn.isChecked() else None
        else:
            self.set_left_btn.setChecked(False)
            self.active_target = "right" if self.set_right_btn.isChecked() else None

    def apply_doppler_shift(self):
        """Preview a multiplicative Doppler shift for the active spectrum
        tab (Full, Left, or Right) using lambda_0 = lambda / (1 + v/c),
        composed on top of that tab's existing shift factor.

        Args:
            None

        Returns:
            None
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

        current_factor = self.spectrum_shift_factor.get(active_tab, 1.0)
        new_factor = current_factor * doppler_velocity_to_wavelength_factor(velocity)

        self.spectrum_shift_preview[active_tab] = {"velocity": velocity, "factor": new_factor}

        # _refresh_active_spectrum_tab draws the orange preview over this
        # tab's own trace, applies Normalize, and enables the shift buttons.
        self._refresh_active_spectrum_tab()

        print(f"[DEBUG] Previewing Doppler shift on '{active_tab}': v={velocity} km/s (cumulative factor={new_factor:.6f})")

    def _apply_shift(self):
        """Commit whichever tab's pending shift preview is active. Each
        tab's shift is independent; the trace keeps its own color (orange
        marks preview only), and Full's cube_data.wavelength stays synced.

        Args:
            None

        Returns:
            None
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

        # Incremental factor from this shift, needed by shift_plot_elements,
        # which moves Gaussian fit artists by a relative amount.
        old_factor = self.spectrum_shift_factor.get(active_tab, 1.0)
        new_factor = preview["factor"]
        incremental_factor = new_factor / old_factor if old_factor else new_factor

        self.spectrum_shift_factor[active_tab] = new_factor
        self.spectrum_shift_preview[active_tab] = None

        # cube_data.wavelength must stay synced for Full: fitting, continuum,
        # HITRAN, and CO-flux code still read it directly, not per tab.
        if active_tab == "full":
            new_wavelength = self._get_tab_wavelength("full")
            self.cube_data.wavelength = new_wavelength.copy()
            self.spec_plotter.wavelength = new_wavelength.copy()
            # Gaussian fit artists are drawn against Full's axis only
            # (unmigrated feature - see plan's deferred section).
            self.spec_plotter.shift_plot_elements(incremental_factor)

        # Auto-fit the x-axis to this tab's new extent instead of keeping the
        # old view window, which would look like part of the spectrum went missing.
        self._fit_xlim_to(self._get_tab_wavelength(active_tab))

        # Recomputes y-limits from the shifted data, redraws the trace in its
        # own color (orange was only a preview cue), and rebuilds the legend.
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
        factor (if any) untouched.

        Args:
            None

        Returns:
            None
        """
        active_tab = self.active_spectrum_tab
        if not self.spectrum_shift_preview.get(active_tab):
            return
        self.spectrum_shift_preview[active_tab] = None
        self._refresh_active_spectrum_tab()
        print(f"[DEBUG] Cancelled shift preview on '{active_tab}'")

    def _revert_shift(self):
        """Revert whichever tab is active back to its original (unshifted)
        wavelength - a full reset to factor 1.0, not a step-wise undo.

        Args:
            None

        Returns:
            None
        """
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
        """Open the colormap picker and apply the chosen map to a slice.

        Args:
            target (str): Panel to recolor, 'left' or 'right'.

        Returns:
            None
        """
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
        """Copy one slice's colormap onto the other slice.

        Args:
            source (str): Panel to copy the colormap from, 'left' or 'right'.
            target (str): Panel to copy the colormap to, 'left' or 'right'.

        Returns:
            None
        """
        src_cmap = getattr(self, 'slice1_cmap' if source == 'left' else 'slice2_cmap', None)
        if not src_cmap:
            return
        if target == 'left':
            self.slice1_cmap = src_cmap
            self.slice1_plotter.set_cmap(src_cmap)
        else:
            self.slice2_cmap = src_cmap
            self.slice2_plotter.set_cmap(src_cmap)

    def _open_scale_dialog(self, target):
        """Open the scale, gamma, and color range dialog for a slice.

        Args:
            target (str): Panel to configure, 'left' or 'right'.

        Returns:
            None
        """
        if not hasattr(self, 'cube_data') or self.cube_data is None:
            QMessageBox.warning(self, "No Cube Loaded", "Load a cube first.")
            return

        idx = self.last_idx_left if target == 'left' else self.last_idx_right
        image = self.cube_data.get_slice(idx)
        current = self.slice1_scale if target == 'left' else self.slice2_scale

        dialog = ScaleDialog(current, image, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return

        self._store_scale_settings(target, dialog.result_settings)
        self._display_slice(idx, target)

    def copy_scale(self, source, target):
        """Copy one slice's scale type, gamma, and color range onto the other slice.

        Args:
            source (str): Panel to copy from, 'left' or 'right'.
            target (str): Panel to copy to, 'left' or 'right'.

        Returns:
            None
        """
        settings = self.slice1_scale if source == 'left' else self.slice2_scale
        self._store_scale_settings(target, settings.copy())

        if hasattr(self, 'cube_data') and self.cube_data is not None:
            idx = self.last_idx_left if target == 'left' else self.last_idx_right
            self._display_slice(idx, target)

    def _contam_rows(self):
        """Convert the contaminating-line catalog into rows shaped like
        HITRAN/custom rows (tagged 'source': 'Contaminant') so they share
        the Lines table and plotting code.

        Args:
            None

        Returns:
            rows (list[dict]): One dict per contaminating line, or an empty
                list if no contaminating catalog is loaded.
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
            # Skips format_isotopologue_name(): that assumes HITRAN's bracket
            # notation and would misformat plain user text. Shown as given.
            iso_name = row.get('isotopologue_name')
            if pd.notna(iso_name):
                entry['isotopologue_name'] = str(iso_name)
            rows.append(entry)
        return rows

    def _combined_line_rows(self):
        """All currently loaded lines -- detection (HITRAN/custom) plus any
        loaded contaminants -- combined and wavelength-sorted, for the shared
        Lines selection dialog.

        Args:
            None

        Returns:
            rows (list[dict]): Combined, wavelength-sorted line rows.
        """
        rows = list(getattr(self, 'hitran_lines', None) or []) + self._contam_rows()
        rows.sort(key=lambda r: float(r.get('lambda_micron') or r.get('wavelength') or 0.0))
        return rows

    def show_available_species(self):
        """Show the combined Lines table: detection lines (HITRAN or
        custom-imported) plus any loaded contaminating lines, with buttons
        to select either set separately.

        Args:
            None

        Returns:
            None
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
        with exactly what's checked in the Select Lines dialog, then redraw.

        Args:
            rows (list[dict]): Line rows to plot, replacing the current set.

        Returns:
            None
        """
        self.plotted_lines = rows.copy()

        # Sync the excluded wavelengths set:
        # Remove any wavelengths from excluded that are now being plotted (re-selection)
        if hasattr(self, 'flux_map_excluded_wavelengths') and self.flux_map_excluded_wavelengths:
            plotted_wavelengths = set(
                SpeciesDialog.get_row_wavelength(r) for r in rows
            )
            # Remove from excluded any that are now plotted
            self.flux_map_excluded_wavelengths = {
                w for w in self.flux_map_excluded_wavelengths
                if not any(abs(w - pw) < 1e-6 for pw in plotted_wavelengths)
            }

        self._draw_plotted_lines()

    def plot_contaminating_lines(self):
        """Merge the contaminating-line catalog into the plotted lines,
        replacing only previous contaminant markers and leaving detection
        lines untouched, then redraw. Called after a contaminating CSV loads.

        Args:
            None

        Returns:
            None
        """
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            return
        if not hasattr(self, 'plotted_lines'):
            self.plotted_lines = []

        non_contam = [r for r in self.plotted_lines if r.get('source') != 'Contaminant']
        self.plotted_lines = non_contam + self._contam_rows()

        self._draw_plotted_lines()

    def _draw_plotted_lines(self):
        """Redraw every plotted line marker from self.plotted_lines. Detection
        lines use the darkred/orange scheme; contaminant-tagged lines use
        black/dimgray so the two stay visually distinct.

        Args:
            None

        Returns:
            None
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
            # Blended transform (x data, y axes-fraction) keeps the label flush
            # with the top edge even if y-limits change later; y=1.01 clears the spine.
            text = ax.text(
                λ, 1.01, row["species"], transform=ax.get_xaxis_transform(),
                rotation=90, va="bottom", ha="center",
                fontsize=7, color=text_color
            )
            self.plotted_artists.extend([line, text])

        self.spec_canvas.draw()

    def _on_spec_hover(self, event):
        """Handle hover events over spectrum plot to show line info.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Motion event.

        Returns:
            None
        """

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

            # Detection and contaminant rows format identically: a contaminant
            # CSV can carry the same optional columns, and N/A covers what's missing.
            iso_name = nearest_row.get('isotopologue_name')
            iso_num = nearest_row.get('isotopologue')
            if iso_name:
                species = f"{species} ({iso_name})"
            elif iso_num:
                species = f"{species} (iso {iso_num})"
            transition = nearest_row.get('transition', 'N/A')
            # Defaults to None, not 0: 0 is a real value (e.g. ground-state
            # E_upper=0), so a missing field must show "N/A", not "0.00e+00".
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
        """Complete reset: clear all plots, selections, normalizations, and
        Gaussian fits, then reload the spectrum from the original data.

        Args:
            None

        Returns:
            None
        """
        print("[DEBUG] === REFRESH VIEW: Complete Reset ===")
        
        if not hasattr(self, 'cube_data') or self.cube_data is None:
            print("[DEBUG] No cube data loaded")
            return

        reply = QMessageBox.question(
            self, "Reset View",
            "Reset the view?\n\nThis clears the region selections, plotted lines, Gaussian fits, "
            "and continuum subtraction. It sets both slices to the first wavelength.\n\n"
            "It does not change the Doppler shifts or the line lists.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
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

        # Reset per-tab Gaussian boundary state and the map-fit cache: a new
        # cube invalidates both, so a cached tab key can't wrongly look current.
        self.gauss_boundaries = {"full": {}, "left": {}, "right": {}}
        self.gauss_cache = None
        self.gauss_cache_tab = None
        self.gauss_cache_lam_min = None
        self.gauss_cache_lam_max = None
        self.gauss_cache_noise_min = None
        self.gauss_cache_noise_max = None

        # Reset per-tab continuum state; the fitted/subtracted artist itself
        # is recreated lazily by _refresh_active_spectrum_tab after ax.clear().
        self.continuum_fitted = {"full": None, "left": None, "right": None}
        self.continuum_subtracted = {"full": False, "left": False, "right": False}
        self.continuum_mode_applied = {"full": None, "left": None, "right": None}
        self.continuum_line_btn.setChecked(False)
        self.subtract_continuum_btn.setChecked(False)
        self.custom_subtract_btn.setChecked(False)
        if hasattr(self, 'custom_continuum_selector') and self.custom_continuum_selector is not None:
            self.custom_continuum_selector.clear_all()
            self.custom_continuum_selector.deactivate()
            self.custom_continuum_selector = None
        self.custom_continuum_active = False
        self._custom_continuum_tab = None

        # Uncheck all buttons
        self.set_left_btn.setChecked(False)
        self.set_right_btn.setChecked(False)
        self.gauss_left_btn.setChecked(False)
        self.gauss_peak_btn.setChecked(False)
        self.gauss_right_btn.setChecked(False)
        self.gauss_noise_left_btn.setChecked(False)
        self.gauss_noise_right_btn.setChecked(False)

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

            # ax.clear() above dropped the dedicated Left/Right tab lines too;
            # force them to be recreated so the active tab displays correctly.
            self._spec_tab_line_left = None
            self._spec_tab_line_right = None
            self._continuum_line_artist = None
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
        
        # Deactivate region selectors and drop any live interactive patch --
        # activate_selector(None) alone left a connected InteractivePatch wired.
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
        """Read and validate the Min/Max wavelength (micron) fields shared
        by the HITRAN/custom-lines/contaminating-lines loaders. Pops a
        warning dialog and returns None on invalid input.

        Args:
            None

        Returns:
            wavelength_range (tuple): (min, max) wavelength in micron, or
                None if the input is invalid.
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
        """Safety margin (micron) around the Min/Max wavelength range so a
        line just outside it that still bleeds into an edge line's Gaussian
        isn't dropped from the fit-template catalog.

        Args:
            None

        Returns:
            pad (float): Padding in micron, matching the fit's own edge
                padding (4x line sigma); 0.0 if no cube is loaded.
        """
        if not hasattr(self, 'cube_data') or self.cube_data is None:
            return 0.0
        from .flux_map_extractor import _line_sigma
        wave = np.asarray(extract_value_from_quantity(self.cube_data.wavelength), dtype=float)
        header = getattr(self.cube_data, 'header', None)
        edge_sigmas = [
            _line_sigma(float(np.nanmin(wave)), header),
            _line_sigma(float(np.nanmax(wave)), header),
        ]
        # Same 4x-sigma padding fit_hitran_lines_collective() uses, so "close
        # enough to matter" means the same thing in both places.
        return 4 * max(edge_sigmas)

    def _reset_lines_state(self):
        """Clear plotted markers, the multi-Gaussian fit, and the
        contaminating-line catalog when the detection source changes, so
        stale range-filtered data does not linger.

        Args:
            None

        Returns:
            None
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
        """Fetch HITRAN lines based on user input from the GUI, including
        transition level (vibrational quantum number) filtering.

        Args:
            None

        Returns:
            None
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
            typed = molecule.strip()
            if typed == "":
                QMessageBox.warning(self, "Invalid Input", "Molecule cannot be empty.")
                return
            # Resolve case-insensitively -- mol_map keys keep HITRAN's own casing
            # (HCl, NOp, H3p), so a bare .upper() would reject them.
            molecule = {k.upper(): k for k in mol_map}.get(typed.upper())
            if molecule is None:
                QMessageBox.warning(self, "Unknown Molecule", f"'{typed}' is not in the HITRAN list.")
                return

            # Isotopologue is optional: search every isotopologue HITRAN has
            # instead of guessing one; each result stays tagged with its own.
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

            # Always False: HITRAN line data barely changes, so a local copy
            # is as good as online; fetch happens automatically if nothing is cached.
            refresh_cache = False

            # Fetch HITRAN data
            results, fetch_meta = hitran_gui_interface(
                wavelength_min, wavelength_max, molecule, isotopologue, refresh_cache
            )

            if not results:
                if fetch_meta.get('error'):
                    QMessageBox.warning(
                        self, "HITRAN Unavailable",
                        f"Could not retrieve {molecule} data for "
                        f"{wavelength_min:.3f}-{wavelength_max:.3f} µm.\n\n"
                        f"No offline copy covers this range, and the online "
                        f"fetch failed:\n{fetch_meta.get('message', 'unknown error')}"
                    )
                else:
                    QMessageBox.information(self, "No Matches", "No HITRAN lines found in this range.")
                return

            # Optional transition filter. Every molecule-specific rule lives in
            # transition_patterns.py + molecule_transition_filters.py.
            if molecule in TRANSITION_PATTERNS:
                want_filter = QMessageBox.question(
                    self, "Transition Filter",
                    f"Filter {molecule} lines by transition?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                ) == QMessageBox.Yes
                if want_filter:
                    filtered = apply_transition_filter(self, results, molecule)
                    if filtered is None:
                        return
                    results = filtered

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
        """Load a user CSV as a stand-in for a live HITRAN fetch (offline
        use). Requires 'wavelength' and 'species'; feeds the same Lines
        table as fetch_hitran_lines(), replacing whatever was loaded before.

        Args:
            None

        Returns:
            None
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
            # Kept exactly to the typed range, no edge padding -- unlike
            # contaminants, a detection line has no fit-suppression role to justify it.
            rows = [r for r in rows if wavelength_min <= r['wavelength'] <= wavelength_max]
            if not rows:
                QMessageBox.information(
                    self, "No Lines In Range",
                    f"Loaded {total_loaded} line(s) from this file, but none fall within "
                    f"the current range ({wavelength_min:.6f}-{wavelength_max:.6f} µm)."
                )
                return

            distinct_species = sorted({r['species'] for r in rows})
            # Detection must be single-species: the fit's shared velocity shift
            # (from these lines' peaks) is only meaningful for one species at a time.
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

    def _toggle_mark_contam_mode(self, checked):
        """Mutually exclusive with the Gaussian-boundary buttons -- a spectrum
        click can only mean one thing at a time, so turning this on clears
        any active boundary-picking mode (and vice versa, in gui_gaussian.py).

        Args:
            checked (bool): New checked state of mark_contam_btn.

        Returns:
            None: Clears gauss_mode and the boundary buttons' checked state.
        """
        if not checked:
            return
        self.gauss_mode = None
        for btn in (self.gauss_left_btn, self.gauss_peak_btn, self.gauss_right_btn,
                    self.gauss_noise_left_btn, self.gauss_noise_right_btn):
            btn.setChecked(False)

    def _manual_contam_rows(self):
        """Rows of self.contam_lines added by clicking the spectrum (see
        _add_manual_contaminant), not a CSV import. These survive a new CSV
        import instead of being replaced by it.

        Args: None

        Returns:
            rows (pd.DataFrame): The manually-added rows, or an empty DataFrame.
        """
        if self.contam_lines is None or 'origin' not in self.contam_lines.columns:
            return pd.DataFrame(columns=['line_wave_um', 'species', 'origin'])
        return self.contam_lines[self.contam_lines['origin'] == 'manual'].reset_index(drop=True)

    def _add_manual_contaminant(self, wavelength):
        """Add a spectrum-clicked wavelength as a contaminant with no known
        species. Merges into self.contam_lines and redraws the same way a
        CSV import does, so it gets excluded from the fit the same way.

        Args:
            wavelength (float): Wavelength (cube's own unit) that was clicked.

        Returns:
            None: Updates self.contam_lines and redraws the spectrum's markers.
        """
        if self.contam_lines is not None and len(self.contam_lines):
            if 'origin' not in self.contam_lines.columns:
                self.contam_lines['origin'] = 'csv'
            # Skip a near-duplicate click -- screen-pixel tolerance, not
            # wavelength, so it stays consistent regardless of plot zoom.
            ax = self.spec_plotter.ax
            click_x_px = ax.transData.transform((wavelength, 0))[0]
            for wl in self.contam_lines['line_wave_um'].astype(float):
                if abs(ax.transData.transform((wl, 0))[0] - click_x_px) < 5:
                    return

        new_row = pd.DataFrame([{'line_wave_um': float(wavelength), 'species': 'N/A', 'origin': 'manual'}])
        self.contam_lines = new_row if self.contam_lines is None or len(self.contam_lines) == 0 \
            else pd.concat([self.contam_lines, new_row], ignore_index=True)

        print(f"[OK] Marked contaminant at {wavelength:.6f} µm (no species -- excluded from the fit's own reported results)")
        self.plot_contaminating_lines()

    def _load_contaminating_lines_dialog(self):
        """Load an optional contaminating-lines CSV. Do this after setting
        the wavelength range and loading HITRAN lines. Filtered to the
        Min/Max range plus a small edge margin (see _line_range_edge_pad()).

        Args:
            None

        Returns:
            None
        """
        wave_range = self._current_wavelength_range()
        if wave_range is None:
            return
        wavelength_min, wavelength_max = wave_range
        try:
            pad = self._line_range_edge_pad()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not compute the line-edge padding:\n{e}")
            return
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

            if loaded is None or len(loaded) == 0:
                QMessageBox.warning(self, "Empty File", "The CSV file is empty or has no valid lines.")
                return  # Leaves self.contam_lines untouched -- a failed import isn't destructive.

            total_loaded = len(loaded)
            in_range = loaded[
                (loaded['line_wave_um'] >= wavelength_min) & (loaded['line_wave_um'] <= wavelength_max)
            ].reset_index(drop=True).copy()
            in_range['origin'] = 'csv'

            # Manually-clicked contaminants (see _add_manual_contaminant) survive a
            # new CSV import -- only the previously-loaded CSV rows get replaced.
            manual_rows = self._manual_contam_rows()

            if len(in_range) == 0:
                QMessageBox.information(
                    self, "No Lines In Range",
                    f"Loaded {total_loaded} line(s) from this file, but none fall within "
                    f"the current range ({wavelength_min:.6f}-{wavelength_max:.6f} µm)."
                )
                self.contam_lines = manual_rows if len(manual_rows) else None
                return
            self.contam_lines = pd.concat([manual_rows, in_range], ignore_index=True)

            print(
                f"[OK] Loaded {len(in_range)} of {total_loaded} contaminating lines from "
                f"{os.path.basename(file_path)} within {wavelength_min:.6f}-{wavelength_max:.6f} µm"
                + (f" (+{len(manual_rows)} manually-marked kept)" if len(manual_rows) else "")
            )

            # Show wavelength range of loaded lines
            min_wave = in_range['line_wave_um'].min()
            max_wave = in_range['line_wave_um'].max()
            print(f"  Wavelength range: {min_wave:.6f} - {max_wave:.6f} um")

            # Mark them on the spectrum right away (dark grey), same as HITRAN selections
            self.plot_contaminating_lines()

            QMessageBox.information(
                self,
                "Success",
                f"✓ Loaded {len(in_range)} of {total_loaded} contaminating lines "
                f"(within {wavelength_min:.6f}-{wavelength_max:.6f} µm)\n\n"
                f"Wavelength range: {min_wave:.6f} - {max_wave:.6f} μm\n\n"
                + (f"Kept {len(manual_rows)} manually-marked contaminant(s) from before.\n\n" if len(manual_rows) else "")
                + f"These lines will be used when extracting flux."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading File",
                f"Failed to load contaminating lines:\n{e}"
            )  # self.contam_lines is untouched -- a failed import isn't destructive.

    def _spectrum_home_reset(self):
        """Custom home action for spectrum: show original spectrum and clear overlays.

        Args:
            None

        Returns:
            None
        """
        print("[DEBUG] Spectrum home button clicked - restoring original spectrum")
        
        # Restore original spectrum if the active tab has a continuum subtracted
        if self.continuum_subtracted.get(self.active_spectrum_tab, False):
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
        """Save the active tab's continuum-subtraction result to an ASCII
        file. Columns: wavelength, original_spectrum, fitted_continuum,
        subtracted. Missing values are written as NaN.

        Args:
            None

        Returns:
            None
        """
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "No spectrum loaded.")
            return

        tab = self.active_spectrum_tab
        wavelength = self._get_tab_wavelength(tab)
        if wavelength is None:
            QMessageBox.warning(self, "No Data", "No wavelength data available.")
            return

        # Original spectrum = the active tab's own raw (pre-subtraction) flux.
        original = self._get_tab_raw_flux(tab)

        if original is None:
            QMessageBox.warning(self, "No Data", "No spectrum to save for the active tab.")
            return

        # The continuum is absent if the user never fitted one for this tab.
        continuum = self.continuum_fitted.get(tab)
        if continuum is None:
            continuum = np.full_like(original, np.nan, dtype=float)

        # Subtracted column = original - continuum, where the continuum is
        # finite.  Where the continuum is NaN, the subtracted result is NaN.
        subtracted = np.where(
            np.isfinite(continuum), original - continuum, np.nan
        )

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Spectrum as ASCII",
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
                f.write(f"# Spectrum tab: {tab}\n")
                f.write("# Type: full_results\n")
                f.write(f"# Wavelength unit: {wave_unit}\n")
                f.write(f"# Flux unit: {flux_unit}\n")
                f.write("#\n")
                f.write(f"# Wavelength ({wave_unit})\tOriginal\tContinuum\tSubtracted\n")
                for w, o, c, s in zip(wavelength, original, continuum, subtracted):
                    f.write(f"{w:.6e}\t{o:.6e}\t{c:.6e}\t{s:.6e}\n")

            QMessageBox.information(self, "Success", f"Spectrum saved to:\n{file_path}")
            print(f"Spectrum saved to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save the spectrum:\n{e}")
            print(f"Error saving spectrum: {e}")

    def open_2d_window(self):
        """Open a new standalone 2D FITS analyzer window.

        Args:
            None

        Returns:
            None
        """
        try:
            from .gui_2d import Ifu_analyzer_2d_window

            two_d_window = Ifu_analyzer_2d_window()
            two_d_window.resize(self.width(), self.height())
            two_d_window.show()

            if not hasattr(self, "child_windows") or self.child_windows is None:
                self.child_windows = []
            self.child_windows.append(two_d_window)

        except Exception as e:
            QMessageBox.warning(self, "2D Window Error", f"Could not open 2D window:\n{e}")

    def open_rotational_diagram(self):
        """Open a new rotational-diagram analyzer window for H2 or CO.

        Args:
            None

        Returns:
            None
        """
        try:
            from .gui_rotational_diagram import RotationalDiagramWindow

            rot_diag_window = RotationalDiagramWindow(parent_gui=self)
            rot_diag_window.show()

            if not hasattr(self, "child_windows") or self.child_windows is None:
                self.child_windows = []
            self.child_windows.append(rot_diag_window)

        except Exception as e:
            QMessageBox.warning(self, "Rotational Diagram Error", f"Could not open rotational diagram window:\n{e}")


