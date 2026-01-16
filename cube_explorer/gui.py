"""
Main GUI class for the Cube Explorer application
"""

import os
import gc
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm, PowerNorm, Normalize

# Suppress FITS warning messages about header formatting
warnings.filterwarnings('ignore', category=Warning, module='astropy.wcs.wcs')
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QMessageBox, QTextEdit, QLineEdit, QInputDialog, QDialog, QToolTip, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from .data_loader import CubeData
from .visualization import SpectrumPlotter, SlicePlotter
from .colormap_dialog import ColorMapDialog
from .region_selector import RegionSelector
from .line_catalog import filter_spectral_lines, normalize_species_input, hitran_gui_interface
from .species_dialog import SpeciesDialog
from .gaussian_fitter import fit_gaussian_to_line, GaussianFitResult
from .gaussian_dialogs import GaussianMapWorker, GaussianMapResultsDialog
from .unit_converter import convert_wavelength, are_units_compatible


class CubeExplorer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cube Explorer")
        self.cube_data = CubeData()
        self.selected_file = None
        self.lst_file = None
        self._init_ui()

    def closeEvent(self, event):
        try:
            self.cube_data.free_memory()
        except AttributeError:
            pass
        plt.close('all')
        gc.collect()
        event.accept()

    def _clear_all_displays(self):
        """Clear all plots, selections, and stats when loading a new cube."""
        print("=== CLEAR ALL DISPLAYS START ===")

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

        # --- Uncheck buttons ---
        for side in ["left", "right"]:
            for btn in self.shape_buttons.get(side, {}).values():
                btn.setChecked(False)
        self.set_left_btn.setChecked(False)
        self.set_right_btn.setChecked(False)
        self.normalize_btn.setChecked(False)
        self.set_range_btn.setChecked(False)
        self.active_target = None
        print("[DEBUG] Unchecking all buttons")

        # --- Reset labels ---
        self.instrument_label.setText("Instrument: -")
        self.lst_label.setText("LST Loaded: -")
        print("[DEBUG] Resetting instrument and LST labels")

        # --- Force garbage collection ---
        import gc
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
        self.open_btn.clicked.connect(self.select_file)
        
        self.load_btn = QPushButton("Load Cube")
        self.load_btn.clicked.connect(self.load_cube)

        self.header_btn = QPushButton("Header")
        self.header_btn.clicked.connect(self.show_header)

        self.lst_btn = QPushButton("Import LST")
        self.lst_btn.clicked.connect(self.select_lst)
        
        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.clicked.connect(self.refresh_view)
        self.refresh_btn.setToolTip("Refresh all plots")

        # New Window button: spawn another CubeExplorer window
        self.new_window_btn = QPushButton("🗗")
        self.new_window_btn.setToolTip("Open a new Cube Explorer window")
        self.new_window_btn.clicked.connect(self._open_new_window)
        
        self.instrument_label = QLabel("Instrument: -")
        self.instrument_label.setStyleSheet("font-weight: bold;")

        self.lst_label = QLabel("LST loaded: -")
        self.lst_label.setStyleSheet("font-weight: bold;")

        self.controls_layout = QHBoxLayout()
        self.controls_layout.addWidget(self.open_btn)
        self.controls_layout.addWidget(self.load_btn)
        self.controls_layout.addWidget(self.header_btn)
        self.controls_layout.addWidget(self.lst_btn)
        self.controls_layout.addWidget(self.refresh_btn)
        self.controls_layout.addWidget(self.new_window_btn)
        self.controls_layout.addWidget(self.instrument_label)
        self.controls_layout.addWidget(self.lst_label)
        self.controls_layout.addStretch()

    def _init_plots(self):
        """Initialize matplotlib figures and canvases"""
        # Left slice
        self.slice1_fig = Figure(figsize=(4, 4))
        self.slice1_canvas = FigureCanvas(self.slice1_fig)
        self.slice1_plotter = SlicePlotter(self.slice1_fig)
        self.slice1_toolbar = NavigationToolbar(self.slice1_canvas, self)
        self.slice1_canvas.mpl_connect("button_press_event", lambda e: self._on_slice_click(e, "left"))
        
        # Right slice
        self.slice2_fig = Figure(figsize=(4, 4))
        self.slice2_canvas = FigureCanvas(self.slice2_fig)
        self.slice2_plotter = SlicePlotter(self.slice2_fig)
        self.slice2_toolbar = NavigationToolbar(self.slice2_canvas, self)
        self.slice2_canvas.mpl_connect("button_press_event", lambda e: self._on_slice_click(e, "right"))

        # Spectrum
        self.spec_fig = Figure(figsize=(8, 3))
        self.spec_canvas = FigureCanvas(self.spec_fig)
        self.spec_plotter = None  # Will be initialized after loading data
        for ax in self.spec_fig.axes:
            ax.set_clip_on(True)
        self.spec_toolbar = NavigationToolbar(self.spec_canvas, self)
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
        self.normalize_btn = QPushButton("Normalize")
        self.set_range_btn = QPushButton("Set Range")
        
        # Gaussian fitting buttons
        self.gauss_left_btn = QPushButton("Left")
        self.gauss_right_btn = QPushButton("Right")
        self.gauss_fit_btn = QPushButton("Fit Gaussian")
        self.gauss_map_btn = QPushButton("Fit Gaussian Map")
        
        # Spectrum saving button
        self.save_spectrum_btn = QPushButton("Save Spectrum")
        self.continuum_line_btn = QPushButton("Continuum Line")
        self.subtract_continuum_btn = QPushButton("Subtract Continuum")

        self.subtract_continuum_btn.setCheckable(True)
        self.subtract_continuum_btn.setEnabled(False) 
        self.continuum_line_btn.setCheckable(True)        
        self.set_left_btn.setCheckable(True)
        self.set_right_btn.setCheckable(True)
        self.normalize_btn.setCheckable(True)
        self.set_range_btn.setCheckable(True)
        self.gauss_left_btn.setCheckable(True)
        self.gauss_right_btn.setCheckable(True)
        
        self.set_left_btn.clicked.connect(lambda: self._select_target("left"))
        self.set_right_btn.clicked.connect(lambda: self._select_target("right"))
        self.normalize_btn.clicked.connect(self.normalize_spectrum)
        self.set_range_btn.clicked.connect(self._activate_range_selector)
        self.gauss_left_btn.clicked.connect(self._activate_gaussian_left)
        self.gauss_right_btn.clicked.connect(self._activate_gaussian_right)
        self.gauss_fit_btn.clicked.connect(self._fit_gaussian)
        self.gauss_map_btn.clicked.connect(self._fit_gaussian_map)
        self.gauss_map_btn.setToolTip("Fit Gaussian to all pixels using Left/Right boundaries")
        self.save_spectrum_btn.clicked.connect(self._save_spectrum_ascii)
        self.continuum_line_btn.clicked.connect(self._on_continuum_line_clicked)
        self.subtract_continuum_btn.clicked.connect(self._activate_continuum_subtraction)



        # Wavelength range inputs (hidden until LST loaded)
        self.wavelength_label = QLabel("Wavelength range:")
        self.min_wave_input = QLineEdit(self)
        self.min_wave_input.setPlaceholderText("Min λ")
        self.max_wave_input = QLineEdit(self)
        self.max_wave_input.setPlaceholderText("Max λ")
        self.species_input = QLineEdit(self)
        self.species_input.setPlaceholderText("Species (optional)")
        self.ok_wave_button = QPushButton("OK")
        self.ok_wave_button.clicked.connect(self.apply_wavelength_range)
        self.show_lst_btn = QPushButton("SPECIES")
        self.show_lst_btn.setToolTip("Open list of lines within current range (optional species filter)")
        self.show_lst_btn.clicked.connect(self.show_filtered_lines)
        self.hitran_btn = QPushButton("HITRAN")
        self.hitran_btn.setToolTip("Fetch HITRAN lines for current wavelength range")
        self.hitran_btn.clicked.connect(self.fetch_hitran_lines)

        # Hide until LST is loaded
        for w in [self.wavelength_label, self.min_wave_input, self.max_wave_input,
                  self.species_input, self.ok_wave_button, self.show_lst_btn, self.hitran_btn]:
            w.hide()

        # Layout for wavelength input
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(self.wavelength_label)
        wl_layout.addWidget(self.min_wave_input)
        wl_layout.addWidget(self.max_wave_input)
        wl_layout.addWidget(self.species_input)
        wl_layout.addWidget(self.ok_wave_button)
        wl_layout.addWidget(self.show_lst_btn)
        wl_layout.addWidget(self.hitran_btn)
        

        self.wavelength_layout = wl_layout

    def _init_shape_controls(self):
        """Initialize shape selection buttons"""
        # Helper function to create shape buttons
        def make_shape_button(symbol, target, callback=None):
            btn = QPushButton(symbol)
            btn.setCheckable(True)
            btn.setFixedSize(36, 36)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 18pt;
                    border: 1px solid #888;
                    border-radius: 6px;
                    background-color: #f0f0f0;
                }
                QPushButton:checked {
                    background-color: #4a90e2;
                    color: white;
                    border: 2px solid #2d73c5;
                }
            """)
            if callback is None:
                btn.clicked.connect(lambda _, s=symbol: self._set_shape(s, target=target))
            else:
                btn.clicked.connect(callback)
            return btn

        # Create left and right shape buttons
        shapes = ["*", "▭", "◯", "⬭", "✏"]
        self.shape_buttons = {
            "left": {shape: make_shape_button(shape, "left") for shape in shapes},
            "right": {shape: make_shape_button(shape, "right") for shape in shapes}
        }

        # Colormap (color scale) buttons (non-shape, open dialog)
        self.color_btn_left = QPushButton("🌈")
        self.color_btn_left.setFixedSize(36, 36)
        self.color_btn_left.setToolTip("Change left slice color scale")
        self.color_btn_left.clicked.connect(lambda: self._open_colormap_dialog('left'))

        self.color_btn_right = QPushButton("🌈")
        self.color_btn_right.setFixedSize(36, 36)
        self.color_btn_right.setToolTip("Change right slice color scale")
        self.color_btn_right.clicked.connect(lambda: self._open_colormap_dialog('right'))

        # Copy colormap buttons
        self.copy_color_btn_l = QPushButton("⎘🌈")
        self.copy_color_btn_l.setFixedSize(42, 36)
        self.copy_color_btn_l.setToolTip("Copy right slice colormap to left slice")
        self.copy_color_btn_l.clicked.connect(lambda: self.copy_colormap('right','left'))

        self.copy_color_btn_r = QPushButton("⎘🌈")
        self.copy_color_btn_r.setFixedSize(42, 36)
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
            callback=lambda: self.copy_selection(source="right", target="left"))
        self.copy_btn_r = make_shape_button("⎘", "right",
            callback=lambda: self.copy_selection(source="left", target="right"))
        # Erase (clear) selection buttons
        self.erase_btn_l = QPushButton("✖")
        self.erase_btn_l.setFixedSize(36, 36)
        self.erase_btn_l.clicked.connect(lambda: self.clear_region("left"))

        self.erase_btn_r = QPushButton("✖")
        self.erase_btn_r.setFixedSize(36, 36)
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
        # Add color button at top
        left_sidebar.addWidget(self.color_btn_left)
        # Add scale dropdown
        left_sidebar.addWidget(QLabel("Scale:"))
        left_sidebar.addWidget(self.scale_combo_left)
        for shape in ["*", "▭", "◯", "⬭", "✏"]:
            left_sidebar.addWidget(self.shape_buttons["left"][shape])
        left_sidebar.addWidget(self.copy_btn_l)
        left_sidebar.addWidget(self.copy_color_btn_l)
        left_sidebar.addWidget(self.erase_btn_l)
        left_sidebar.addStretch()

        # assemble left combined region: slice (4) + sidebar (1)
        left_combined.addLayout(left_slice_layout, 4)
        left_combined.addLayout(left_sidebar, 1)

        # Wrap left half into a layout that enforces stats:combined = 1:3
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
        # Add scale dropdown
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

        # Add halves to top_layout equally
        top_layout.addLayout(left_half, 1)
        top_layout.addLayout(right_half, 1)

        main_layout.addLayout(top_layout)

        # Add slice selection buttons under the top layout
        slice_select_layout = QHBoxLayout()
        slice_select_layout.addWidget(self.set_left_btn)
        slice_select_layout.addWidget(self.set_right_btn)
        slice_select_layout.addWidget(self.normalize_btn)
        slice_select_layout.addStretch()
        
        # Add Gaussian fitting buttons
        slice_select_layout.addWidget(QLabel("Gaussian Fit:"))
        slice_select_layout.addWidget(self.gauss_left_btn)
        slice_select_layout.addWidget(self.gauss_right_btn)
        slice_select_layout.addWidget(self.gauss_fit_btn)
        slice_select_layout.addWidget(self.gauss_map_btn)
        slice_select_layout.addStretch()
        
        main_layout.addLayout(slice_select_layout)

        # Add spectrum at bottom
        main_layout.addWidget(self.spec_canvas)
        main_layout.addWidget(self.spec_toolbar)
        
        # Add save spectrum button
        spectrum_button_layout = QHBoxLayout()
        spectrum_button_layout.addStretch()
        spectrum_button_layout.addWidget(self.save_spectrum_btn)
        spectrum_button_layout.addStretch()
        spectrum_button_layout.addWidget(self.continuum_line_btn)
        spectrum_button_layout.addWidget(self.subtract_continuum_btn)        
        main_layout.addLayout(spectrum_button_layout)

        # Add wavelength range controls under the spectrum
        main_layout.addLayout(self.wavelength_layout)

        self.setLayout(main_layout)

    def _on_continuum_line_clicked(self):
        """Handle Continuum Line button toggle - fit and display continuum"""
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            self.continuum_line_btn.setChecked(False)
            return
        
        if not hasattr(self, 'cube_data') or self.cube_data.spectrum is None:
            QMessageBox.warning(self, "No Spectrum", "No spectrum data available.")
            self.continuum_line_btn.setChecked(False)
            return
        
        if self.continuum_line_btn.isChecked():
            # Fit and display continuum
            self._fit_and_display_continuum()
        else:
            # Remove continuum line
            self._remove_continuum_line()

    def _fit_and_display_continuum(self):
        """Fit continuum using median filter - follows baseline perfectly"""
        try:
            from scipy.ndimage import median_filter
            from scipy.interpolate import UnivariateSpline
            
            # Get spectrum data
            wavelength = self.spec_plotter.wavelength
            flux = self.cube_data.spectrum
            
            if wavelength is None or flux is None:
                QMessageBox.warning(self, "No Data", "Spectrum data not available.")
                self.continuum_line_btn.setChecked(False)
                return
            
            # Remove NaN and infinite values
            mask = np.isfinite(wavelength) & np.isfinite(flux)
            wavelength_clean = wavelength[mask]
            flux_clean = flux[mask]
            
            if len(wavelength_clean) < 10:
                QMessageBox.warning(self, "Insufficient Data", "Not enough valid data points to fit continuum.")
                self.continuum_line_btn.setChecked(False)
                return
            
            print("[DEBUG] Fitting continuum using median filter method...")
            
            # STEP 1: Apply median filter to remove narrow spikes
            # Window size determines smoothness - larger = smoother
            # For your data, 51 works well (must be odd number)
            window_size = 51
            flux_median_filtered = median_filter(flux_clean, size=window_size, mode='reflect')
            
            # STEP 2: Iterative sigma clipping to remove broad emission features
            continuum = flux_median_filtered.copy()
            
            for iteration in range(3):  # 3 iterations of clipping
                # Calculate residuals
                residuals = flux_clean - continuum
                
                # Calculate sigma (standard deviation) of residuals
                sigma = np.std(residuals)
                
                # Find points that are NOT outliers (within 2 sigma below continuum)
                # We only clip points ABOVE the continuum (emission lines)
                good_mask = residuals < (2.0 * sigma)
                
                # Apply another median filter only on good points
                flux_good = flux_clean.copy()
                flux_good[~good_mask] = np.nan  # Mark bad points as NaN
                
                # Interpolate over NaN values
                nan_mask = np.isnan(flux_good)
                if np.any(~nan_mask):
                    flux_good[nan_mask] = np.interp(
                        wavelength_clean[nan_mask], 
                        wavelength_clean[~nan_mask], 
                        flux_good[~nan_mask]
                    )
                
                # Apply median filter again
                continuum = median_filter(flux_good, size=window_size, mode='reflect')
            
            # STEP 3: Smooth the continuum slightly using spline
            # This removes very high-frequency noise while keeping baseline variations
            try:
                # UnivariateSpline with s parameter controls smoothing
                # s=0 means interpolation (goes through all points)
                # s>0 means smoothing (larger = more smooth)
                # For your data, s should be small to follow closely
                spline = UnivariateSpline(wavelength_clean, continuum, s=len(wavelength_clean)*10, k=3)
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
                                        linestyle='-',  # Solid line
                                        label='Continuum',
                                        zorder=10)[0]
            
            # Update legend
            ax.legend(loc='upper right', fontsize=8)
            self.spec_canvas.draw()
            
            print(f"[DEBUG] Continuum fitted successfully using median filter + sigma clipping")
            self.subtract_continuum_btn.setEnabled(True)
            
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
            import traceback
            traceback.print_exc()
        
        # Enable subtract button
        self.subtract_continuum_btn.setEnabled(True)

    def _remove_continuum_line(self):
        """Remove the continuum line from the spectrum plot"""
        if self.continuum_line is not None:
            try:
                self.continuum_line.remove()
                print("[DEBUG] Continuum line removed")
            except Exception as e:
                print(f"[DEBUG] Error removing continuum line: {e}")
            self.continuum_line = None
            self.continuum_fitted = None


            
            # Update legend
            if hasattr(self, 'spec_plotter') and self.spec_plotter is not None:
                self.spec_plotter.ax.legend(loc='upper right', fontsize=8)
                self.spec_canvas.draw()

            # If spectrum was subtracted, restore it
            if self.continuum_subtracted:
                self._restore_original_spectrum_internal()
            self.subtract_continuum_btn.setEnabled(False)
            self.subtract_continuum_btn.setChecked(False)

    def _init_state(self):
        """Initialize state variables"""
        self.active_shape = None
        self.active_target = None
        self.last_idx_left = 0
        self.last_idx_right = 0
        self.last_selection_left = None
        self.last_selection_right = None
        self.lst_plotted_lines = []
        self.lst_plotted_artists = []
        self.hitran_lines = []  # Store HITRAN lines separately
        self.lst_lines = []  # Store LST lines separately
        self.scale_left = 'Linear'
        self.scale_right = 'Linear'
        
        # Gaussian fitting state
        self.gauss_left_x = None
        self.gauss_right_x = None
        self.gauss_mode = None  # 'left' or 'right'
        self.gauss_markers = []  # Store marker artists
        
        # Normalization state
        self.is_normalized = False
        self.continuum_line = None  # Store continuum line artist
        self.continuum_fitted = None # Continuum subtraction state
        self.subtract_mode = None  # 'left' or 'right'
        self.subtract_left_x = None
        self.subtract_right_x = None
        self.subtract_markers = []  # Store marker artists

        self.continuum_subtracted = False  # Track if continuum is currently subtracted
        self.original_spectrum_backup = None

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

    def select_file(self):
        """Open file dialog to select a FITS file"""
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open FITS", filter="FITS files (*.fits *.fz *.fit)")
        if fname:
            self.selected_file = fname
            QMessageBox.information(self, "File Selected", f"Selected: {fname}")

    def load_cube(self):
        """Load the selected FITS file"""
        if not hasattr(self, "selected_file"):
            QMessageBox.warning(self, "No File", "Please open a FITS file first.")
            return

        try:
            # Clear all previous plots and selections
            self._clear_all_displays()
            
            # Free any existing memory
            if hasattr(self, 'cube_data'):
                self.cube_data.free_memory()

            # Load the data
            success = self.cube_data.load_from_file(self.selected_file)
            if not success:
                QMessageBox.warning(self, "Load Error", "Failed to load the FITS file")
                return
                
            if self.cube_data.spectrum is None or len(self.cube_data.spectrum) == 0:
                QMessageBox.warning(self, "Data Error", "Failed to compute spectrum from the data")
                return

            # Copy spectrum for normalization
            self.original_spectrum = self.cube_data.spectrum.copy()

            # Update instrument label
            self.instrument_label.setText(f"Instrument: {self.cube_data.instrument}")

            # Initialize the spectrum plotter with wavelength information
            self.spec_plotter = SpectrumPlotter(
                self.spec_fig,
                self.cube_data.wavelength,
                self.cube_data.wave_unit
            )
            self.spec_plotter.cube_data = self.cube_data  # Set cube_data for unit handling
            
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
        if not hasattr(self.cube_data, 'header') or self.cube_data.header is None:
            QMessageBox.warning(self, "No file loaded", "Please load a FITS file first.")
            return

        header = self.cube_data.header
        header_text = "\n".join(
            f"{k:8} = {v!r} / {header.comments[k]}" if k in header.comments else f"{k:8} = {v!r}" 
            for k, v in header.items()
        )

        # Keep reference to prevent garbage collection
        self.header_window = QWidget()
        self.header_window.setWindowTitle("FITS Header")
        self.header_window.setMinimumSize(600, 400)

        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setText(header_text)
        layout.addWidget(text_edit)

        self.header_window.setLayout(layout)
        self.header_window.show()

    def select_lst(self):
        """Open dialog to load .LST line list file"""
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Line List", filter="Line List (*.lst *.txt)"
        )
        if fname:
            # Read file safely
            with open(fname, "r") as f:
                self.lst_data = f.read().splitlines()

            self.lst_file = fname
            self.lst_label.setText(f".LST: LOADED ({os.path.basename(fname)})")

            QMessageBox.information(self, "LST Loaded",
                                    f"Imported line list:\n{fname}")

            # Reveal wavelength controls and OK button
            self.wavelength_label.show()
            self.min_wave_input.show()
            self.max_wave_input.show()
            self.species_input.show()
            self.ok_wave_button.show()
            self.show_lst_btn.show()
            self.hitran_btn.show()

    def apply_wavelength_range(self):
        """Triggered when user confirms wavelength range and species"""
        if not hasattr(self, "lst_file") or not self.lst_file:
            QMessageBox.warning(self, "No LST File", "Please load a .LST file first.")
            return

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

        # Read species (optional)
        species_txt = self.species_input.text().strip()
        species = normalize_species_input(species_txt) if species_txt else None

        if species == "":
            species = None  # plot all if empty

        # Don't plot lines automatically - wait for user to click SPECIES button
        # and select which rows to plot in the dialog
        QMessageBox.information(
            self, "Range Set", 
            f"Wavelength range set to {min_wave:.3f} - {max_wave:.3f} µm\n"
            f"Species: {species or 'All'}\n\n"
            "Click the SPECIES button to view and select lines to plot."
        )


    def plot_lst_lines(self, min_wave, max_wave, species=None):
        """Plot vertical lines from LST file within a wavelength range and optional species"""
        if not hasattr(self, "lst_file") or not self.lst_file:
            print("No LST file loaded")
            return

        # Filter spectral lines
        results, species_counts = filter_spectral_lines(
            self.lst_data, min_wave, max_wave, species
        )

        # Debug prints
        print(f"Filtering for species: '{species}' in range {min_wave}-{max_wave}")
        print(f"Available species in this range: {list(species_counts.keys())}")
        print(f"Number of matched lines: {len(results)}")

        if not results:
            QMessageBox.information(
                self, "No Matches",
                f"No spectral lines found for '{species or 'all'}' in this range."
            )
            return

        # Use spectrum axis
        ax = self.spec_plotter.ax

        # Remove previous LST lines
        for line in ax.lines[:]:
            if getattr(line, "_from_lst", False):
                line.remove()

        # Plot new lines (no text labels)
        for row in results:
            try:
                wavelength = SpeciesDialog.get_row_wavelength(row)
                vline = ax.axvline(wavelength, color="red", linestyle="--", alpha=0.6)
                vline._from_lst = True
            except Exception:
                continue

        self.spec_canvas.draw_idle()

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
            wave = self.cube_data.wavelength[idx]
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
        plotter.plot_slice(image, title, cmap=cmap, scale=scale)
        
        # Intensity stats with units
        mn, mx, mean = np.nanmin(image), np.nanmax(image), np.nanmean(image)
        from .unit_converter import clean_flux_unit
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
            "label": f'{target.capitalize()} Selection'
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
            from .unit_converter import clean_flux_unit
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
            except Exception:
                pass

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
        else:
            # Handle region selection
            selector = (
                self.region_selector_left if target == "left" else self.region_selector_right
            )
            selector.active_shape = source_selection["shape"]
            selector.create_region(source_selection["coords"])

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
                    wave = self.cube_data.wavelength[idx]
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

        if hasattr(self, 'subtract_mode') and self.subtract_mode == 'selecting' and event.xdata:
            self._mark_subtraction_boundary(event.xdata)
            return
        
        # Handle Gaussian boundary selection first (takes priority)
        if self.gauss_mode in ['left', 'right'] and event.xdata is not None:
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
        wavelength = self.cube_data.wavelength
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

    def normalize_spectrum(self):
        """Toggle continuum normalization of spectra."""
        
        # Check if spectrum is loaded
        if not hasattr(self, 'original_spectrum') or self.original_spectrum is None:
            QMessageBox.warning(self, "No Spectrum", "Load a data cube first.")
            self.normalize_btn.setChecked(False)
            return
        
        # Toggle normalization
        if self.normalize_btn.isChecked():
            self._apply_normalization()
        else:
            self._remove_normalization()
    
    def _apply_normalization(self):
        """Apply continuum normalization to all spectra."""
        
        # Check if at least one region is selected
        has_region = (self.last_selection_left is not None or 
                     self.last_selection_right is not None)
        
        if not has_region:
            QMessageBox.warning(
                self, "No Region Selected",
                "Please select at least one region (left or right) before normalizing.\n\n"
                "The selected region will be used to estimate the continuum."
            )
            self.normalize_btn.setChecked(False)
            return
        
        # Get the selected region spectrum for continuum estimation
        if self.last_selection_left is not None:
            region_spec = self.cube_data.compute_region_spectrum(
                self.last_selection_left["mask"]
            )
        else:
            region_spec = self.cube_data.compute_region_spectrum(
                self.last_selection_right["mask"]
            )
        
        if region_spec is None:
            QMessageBox.warning(self, "Error", "Could not compute region spectrum.")
            self.normalize_btn.setChecked(False)
            return
        
        # Estimate continuum using percentile filtering (removes emission lines)
        # Use 50th percentile in sliding window - robust to outliers
        from scipy.ndimage import percentile_filter
        window_size = max(5, len(region_spec) // 20)  # Adaptive window
        if window_size % 2 == 0:
            window_size += 1  # Must be odd
        
        continuum = percentile_filter(region_spec, 50, size=window_size)
        
        # Smooth the continuum to avoid artifacts
        from scipy.ndimage import uniform_filter1d
        continuum = uniform_filter1d(continuum, size=max(3, window_size // 3))
        
        # Avoid division by zero
        continuum[continuum <= 0] = np.nanmedian(continuum[continuum > 0])
        
        # Store continuum for later
        self.continuum = continuum
        
        # Normalize global spectrum
        normalized_global = self.original_spectrum / continuum
        self.spec_plotter.update_spectrum(normalized_global)
        
        # Plot the actual continuum curve (before normalization) for reference
        wavelength = self.spec_plotter.wavelength
        if self.continuum_line is not None:
            try:
                self.continuum_line.remove()
            except:
                pass
        
        # Plot the continuum curve itself (not normalized)
        # This shows what baseline was used for normalization
        self.continuum_line, = self.spec_plotter.ax.plot(
            wavelength, continuum,
            'g--', linewidth=2, alpha=0.7, label='Continuum (fit)', zorder=5
        )
        
        # Update region spectra if they exist
        for side, attr, color, label in [
            ('left', 'left_region_spectrum', 'cyan', 'Left Selection'),
            ('right', 'right_region_spectrum', 'magenta', 'Right Selection')
        ]:
            selected = getattr(self, f'last_selection_{side}', None)
            if selected is not None:
                # Recompute and plot normalized region spectrum
                mask = selected.get("mask")
                if mask is not None:
                    region_spec = self.cube_data.compute_region_spectrum(mask)
                    if region_spec is not None:
                        normalized_region = region_spec / continuum
                        self.spec_plotter.plot_region_spectrum(
                            normalized_region,
                            color=color,
                            label=label
                        )
        
        # Update legend and redraw
        self.spec_plotter.ax.legend(loc='upper right', fontsize=8)
        self.spec_canvas.draw()
        self.is_normalized = True
        
        print("[DEBUG] Continuum normalization applied")
    
    def _remove_normalization(self):
        """Remove normalization and restore original spectra."""
        
        print("[DEBUG] Removing normalization...")
        
        # Remove continuum line
        if self.continuum_line is not None:
            try:
                self.continuum_line.remove()
                print("[DEBUG] Removed continuum line")
            except Exception as e:
                print(f"[DEBUG] Error removing continuum line: {e}")
            self.continuum_line = None
        
        # Clear existing region lines to prevent duplicates
        if hasattr(self.spec_plotter, 'region_lines'):
            for label, line in list(self.spec_plotter.region_lines.items()):
                try:
                    line.remove()
                except:
                    pass
            self.spec_plotter.region_lines = {}
        
        # Restore original global spectrum
        print(f"[DEBUG] Restoring spectrum, shape: {self.original_spectrum.shape}")
        self.spec_plotter.update_spectrum(self.original_spectrum)
        
        # Restore region spectra if they exist
        for side, color, label in [
            ('left', 'cyan', 'Left Selection'),
            ('right', 'magenta', 'Right Selection')
        ]:
            selected = getattr(self, f'last_selection_{side}', None)
            if selected is not None:
                mask = selected.get("mask")
                if mask is not None:
                    region_spec = self.cube_data.compute_region_spectrum(mask)
                    if region_spec is not None:
                        print(f"[DEBUG] Restoring {side} region spectrum")
                        self.spec_plotter.plot_region_spectrum(
                            region_spec,
                            color=color,
                            label=label
                        )
        
        # Update legend and redraw
        self.spec_plotter.ax.legend(loc='upper right', fontsize=8)
        self.spec_canvas.draw_idle()
        self.is_normalized = False
        
        print("[DEBUG] Normalization removed successfully")

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

    def _activate_range_selector(self):
        if self.set_range_btn.isChecked():
            self.active_target = "range"
            if hasattr(self, 'spec_plotter') and self.spec_plotter:
                self.span = SpanSelector(
                    self.spec_plotter.ax,
                    self.on_span_selected,
                    "horizontal",
                    useblit=True,
                    props=dict(alpha=0.3, facecolor='orange')
                )
        else:
            if hasattr(self, 'span'):
                self.span.disconnect_events()
                self.span = None

    def _activate_continuum_subtraction(self):
        """Activate continuum subtraction mode - user selects two points"""
        # Check if continuum is fitted
        if not hasattr(self, 'continuum_fitted') or self.continuum_fitted is None:
            QMessageBox.warning(self, "No Continuum", 
                            "Please fit a continuum first (click 'Continuum Line' button).")
            self.subtract_continuum_btn.setChecked(False)
            return
        
        if self.subtract_continuum_btn.isChecked():
            # Activate selection mode
            print("[DEBUG] Continuum subtraction mode activated. Click two points on spectrum.")
            QMessageBox.information(self, "Select Region", 
                                "Click TWO points on the spectrum to select the wavelength range for subtraction.\n\n"
                                "The order doesn't matter - left/right will be determined automatically.")
            self.subtract_mode = 'selecting'
        else:
            # Cancel mode and clear markers
            self._clear_subtraction_markers()
            self.subtract_mode = None
            self.subtract_left_x = None
            self.subtract_right_x = None
            print("[DEBUG] Continuum subtraction mode cancelled")

    def _mark_subtraction_boundary(self, xpos):
        """Mark a boundary point for continuum subtraction"""
        ax = self.spec_plotter.ax
        
        # If we don't have a left point yet, this is the first point
        if self.subtract_left_x is None and self.subtract_right_x is None:
            # First point - mark as "Point 1"
            marker = ax.axvline(xpos, color='red', linestyle='--', linewidth=2, label='Point 1')
            self.subtract_markers.append(marker)
            self.subtract_left_x = xpos  # Temporarily store here
            print(f"[DEBUG] First point marked at {xpos:.4f}")
            self.spec_canvas.draw()
            
        elif self.subtract_right_x is None:
            # Second point - now determine left/right
            second_x = xpos
            first_x = self.subtract_left_x
            
            # Determine which is left and which is right
            if second_x < first_x:
                # User clicked right-to-left
                self.subtract_left_x = second_x
                self.subtract_right_x = first_x
            else:
                # User clicked left-to-right
                self.subtract_right_x = second_x
                # self.subtract_left_x already set
            
            # Remove old markers
            for marker in self.subtract_markers:
                try:
                    marker.remove()
                except:
                    pass
            self.subtract_markers = []
            
            # Draw final left/right markers
            left_marker = ax.axvline(self.subtract_left_x, color='green', linestyle='-', linewidth=2, label='Left')
            right_marker = ax.axvline(self.subtract_right_x, color='orange', linestyle='-', linewidth=2, label='Right')
            self.subtract_markers = [left_marker, right_marker]
            
            print(f"[DEBUG] Region selected: {self.subtract_left_x:.4f} to {self.subtract_right_x:.4f}")
            
            # Update legend
            ax.legend(loc='upper right', fontsize=8)
            self.spec_canvas.draw()
            
            # Perform the subtraction
            self.subtract_continuum_btn.setChecked(False)
            self.subtract_mode = None
            self._perform_regional_continuum_subtraction()

    def _perform_regional_continuum_subtraction(self):
        """Subtract continuum in selected range"""
        try:
            wavelength = self.spec_plotter.wavelength
            flux = self.cube_data.spectrum
            continuum = self.continuum_fitted
            
            # Find indices
            left_idx = np.argmin(np.abs(wavelength - self.subtract_left_x))
            right_idx = np.argmin(np.abs(wavelength - self.subtract_right_x))
            
            # Subtract ONLY in range
            flux[left_idx:right_idx+1] -= continuum[left_idx:right_idx+1]
            
            # Update the spectrum data
            self.cube_data.spectrum = flux
            
            # REPLOT the spectrum (since update_spectrum doesn't exist)
            ax = self.spec_plotter.ax
            ax.clear()
            
            # Calculate FULL y-range before plotting
            y_min = np.nanmin(flux)
            y_max = np.nanmax(flux)
            y_margin = (y_max - y_min) * 0.05
            
            # Plot the spectrum with clipping
            ax.plot(wavelength, flux, 'b-', label='Total Spectrum', linewidth=1, clip_on=True)
            
            # Set fixed limits
            ax.set_xlim(wavelength.min(), wavelength.max())
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            
            # Labels and legend
            ax.set_xlabel(f'Wavelength [{self.cube_data.wave_unit}]')
            ax.set_ylabel('Flux [MJy sr⁻¹]')
            ax.legend(loc='upper right', fontsize=8)
            
            # Force clipping on all lines
            for line in ax.get_lines():
                line.set_clip_on(True)
            
            # Redraw
            self.spec_canvas.draw()
            
            print(f"[DEBUG] Subtracted in [{self.subtract_left_x:.4f}, {self.subtract_right_x:.4f}]")
            print(f"[DEBUG] Y-axis: {y_min:.1f} to {y_max:.1f}")
            
            QMessageBox.information(self, "Done", 
                                f"Continuum subtracted:\n"
                                f"{self.subtract_left_x:.4f} to {self.subtract_right_x:.4f}\n"
                                f"Y-range: {y_min:.1f} to {y_max:.1f}")
            
            self._clear_subtraction_markers()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed:\n{str(e)}")
            print(f"[DEBUG] Error: {e}")
            import traceback
            traceback.print_exc()



    def _clear_subtraction_markers(self):
        """Clear the subtraction boundary markers"""
        for marker in self.subtract_markers:
            try:
                marker.remove()
            except:
                pass
        self.subtract_markers = []
        if hasattr(self, 'spec_canvas'):
            self.spec_canvas.draw()

    def _init_line_search_controls(self):
        """Initialize the min/max/species inputs and buttons."""
        self.min_wave_input = QLineEdit()
        self.max_wave_input = QLineEdit()
        self.species_input = QLineEdit()
        
        self.min_wave_input.setPlaceholderText("Min λ (µm)")
        self.max_wave_input.setPlaceholderText("Max λ (µm)")
        self.species_input.setPlaceholderText("Species")

        self.ok_wave_button = QPushButton("OK")
        self.ok_wave_button.clicked.connect(self.apply_wavelength_range)

        self.show_lst_btn = QPushButton("SPECIES")
        self.show_lst_btn.clicked.connect(self.show_filtered_lines)

        # HITRAN fetch button
        self.hitran_btn = QPushButton("HITRAN")
        self.hitran_btn.setToolTip("Fetch spectral lines from HITRAN database")
        self.hitran_btn.clicked.connect(self.fetch_hitran_lines)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Wavelength range:"))
        layout.addWidget(self.min_wave_input)
        layout.addWidget(self.max_wave_input)
        layout.addWidget(self.species_input)
        layout.addWidget(self.ok_wave_button)
        layout.addWidget(self.show_lst_btn)
        layout.addWidget(self.hitran_btn)

        self.line_search_layout = layout

    def enable_wavelength_selector(self):
        def on_select(xmin, xmax):
            self.find_spectral_lines(xmin, xmax)
        self.span = SpanSelector(
            self.spec_plotter.ax, on_select, "horizontal",
            useblit=True, props=dict(alpha=0.3, facecolor='orange')
        )

    def on_spectrum_region_selected(self, start_lambda, end_lambda):
        lines, species_counts = filter_spectral_lines(start_lambda, end_lambda)
        print(f"Found {len(lines)} lines between {start_lambda} and {end_lambda}")

        # Plot lines on top of spectrum
        for row in lines:
            self.spec_plotter.ax.axvline(SpeciesDialog.get_row_wavelength(row), color="red", linestyle="--", alpha=0.4)
            self.spec_plotter.ax.text(SpeciesDialog.get_row_wavelength(row), 1.02, row["species"], rotation=90, va="bottom", ha="center", fontsize=7, color="orange")
        self.spec_canvas.draw()

    def find_spectral_lines(self, wavelength_min=None, wavelength_max=None):
        """Search and overlay spectral lines within a wavelength range."""
        if not hasattr(self, "lst_file"):
            QMessageBox.warning(self, "No LST File", "Please import an LST file first.")
            return

        # Get input wavelengths if not passed from SpanSelector
        if wavelength_min is None or wavelength_max is None:
            try:
                wavelength_min = float(self.lambda_min_input.text())
                wavelength_max = float(self.lambda_max_input.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Enter valid wavelength numbers.")
                return

        # Ask user for a species (optional)
        species, ok = QInputDialog.getText(self, "Select Species", "Enter species (e.g., H2, [Fe II]):")
        if not ok:
            return

        # Search lines
        lines, _ = filter_spectral_lines(self.lst_data, wavelength_min, wavelength_max, species)

        if not lines:
            QMessageBox.information(self, "No Matches", "No lines found in this range.")
            return

        # Plot vertical markers
        ax = self.spec_plotter.ax
        for row in lines:
            λ = SpeciesDialog.get_row_wavelength(row)
            ax.axvline(λ, color="red", linestyle="--", alpha=0.4)
            ax.text(λ, 1.02, row["species"], rotation=90, va="bottom", ha="center", fontsize=7, color="orange")

        self.spec_canvas.draw()


    def show_filtered_lines(self):
        # 1. Filter LST lines
        species_text = self.species_input.text().strip()
        species = normalize_species_input(species_text) if species_text else None

        # Use self.lst_file (file path) instead of self.lst_data (file contents)
        lst_lines, _ = filter_spectral_lines(self.lst_file,
                                         float(self.min_wave_input.text()),
                                         float(self.max_wave_input.text()), species)

        # Mark LST lines with source
        for line in lst_lines:
            line['source'] = 'LST'
        
        # Store LST lines for future reference
        self.lst_lines = lst_lines

        # 2. Filter HITRAN lines to current wavelength range
        min_wave = float(self.min_wave_input.text())
        max_wave = float(self.max_wave_input.text())
        filtered_hitran = []
        for line in self.hitran_lines:
            wave = float(line.get('lambda_micron') or line.get('wavelength') or 0.0)
            if min_wave <= wave <= max_wave:
                filtered_hitran.append(line)
        
        # 3. Merge with filtered HITRAN lines
        all_lines = lst_lines + filtered_hitran
        
        # Sort by wavelength (ascending)
        all_lines.sort(key=lambda r: float(r.get('lambda_micron') or r.get('wavelength') or 0.0))
        
        if not all_lines:
            QMessageBox.information(self, "No Matches", "No spectral lines found.")
            return

        # 4. If this is the first time (nothing plotted yet), plot all lines first
        if not self.lst_plotted_lines:
            self.plot_selected_lines(all_lines)

        # 5. Show merged dialog
        dlg = SpeciesDialog(all_lines, parent=self, plotted_rows=self.lst_plotted_lines)
        result = dlg.exec_()

        # 6. Get selected rows and plot only those
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

        # Create dialog
        dlg = SpeciesDialog(lines, parent=self, plotted_rows=self.lst_plotted_lines)
        result = dlg.exec_()

        if result == QDialog.Accepted:
            selected_rows = dlg.get_selected_rows()

            # Extract wavelengths for plotting
            wavelengths = [row["wavelength"] for row in selected_rows]

            self.display.draw_vertical_lines(wavelengths)

    def plot_selected_lines(self, rows):
        ax = self.spec_plotter.ax

        # Remove old lines/text
        for artist in self.lst_plotted_artists:
            try:
                artist.remove()
            except Exception as e:
                print("DEBUG: could not remove artist", e)
        self.lst_plotted_artists = []

        # Update currently plotted rows
        self.lst_plotted_lines = rows.copy()
        
        # Get cube wavelength unit
        cube_unit = getattr(self.cube_data, 'wave_unit', 'µm') if hasattr(self, 'cube_data') else 'µm'

        # Plot new selection with hover metadata
        for row in rows:
            λ = SpeciesDialog.get_row_wavelength(row)
            
            # Determine source unit (LST files are typically in µm, HITRAN varies)
            is_hitran = row.get('source') == 'HITRAN'
            source_unit = row.get('wavelength_unit', 'µm')  # Default to µm for LST
            
            # Convert wavelength to cube units if needed
            try:
                if cube_unit != source_unit and are_units_compatible(source_unit, cube_unit):
                    λ_converted = convert_wavelength(λ, source_unit, cube_unit)
                    print(f"[DEBUG] Converted line {row.get('species', 'Unknown')}: {λ:.4f} {source_unit} -> {λ_converted:.4f} {cube_unit}")
                    λ = λ_converted
            except Exception as e:
                print(f"[DEBUG] Unit conversion failed for {row.get('species', 'Unknown')}: {e}")
                # Use original wavelength if conversion fails
            
            # Use darker red for HITRAN lines, regular red for LST lines
            line_color = "darkred" if is_hitran else "red"
            line_alpha = 0.6 if is_hitran else 0.4
            
            line = ax.axvline(λ, color=line_color, linestyle="--", alpha=line_alpha, picker=5)
            # Store row data in the line for hover access
            line.row_data = row
            text = ax.text(
                λ, ax.get_ylim()[1],
                row["species"], rotation=90, va="bottom", ha="center",
                fontsize=7, color="orange"
            )
            self.lst_plotted_artists.extend([line, text])

        self.spec_canvas.draw()

    def _on_spec_hover(self, event):
        """Handle hover events over spectrum plot to show line info."""

        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            return
        if event.inaxes != self.spec_plotter.ax:
            return
        
        if not hasattr(self, 'lst_plotted_lines') or not self.lst_plotted_lines:
            return
        
        # Check if hovering near any plotted lines
        hover_threshold = 0.01  # wavelength units tolerance
        mouse_x = event.xdata
        if mouse_x is None:
            return
        
        # Find nearest line within threshold
        nearest_row = None
        min_dist = float('inf')
        
        for row in self.lst_plotted_lines:
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
            from PyQt5.QtCore import QPoint
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
        self.gauss_right_x = None
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
        self.gauss_left_marker = None
        self.gauss_right_marker = None
        
        # Reset normalization
        if hasattr(self, 'is_normalized') and self.is_normalized:
            self.normalize_btn.setChecked(False)
            self.is_normalized = False
        if hasattr(self, 'continuum_line') and self.continuum_line:
            try:
                self.continuum_line.remove()
            except:
                pass
        self.continuum_line = None
        
        # Uncheck all buttons
        self.set_left_btn.setChecked(False)
        self.set_right_btn.setChecked(False)
        self.normalize_btn.setChecked(False)
        self.gauss_left_btn.setChecked(False)
        self.gauss_right_btn.setChecked(False)
        
        # Clear all plotted lines (LST, HITRAN)
        if hasattr(self, 'lst_plotted_artists'):
            for artist in self.lst_plotted_artists:
                try:
                    artist.remove()
                except:
                    pass
            self.lst_plotted_artists = []
            self.lst_plotted_lines = []
        
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

            # Molecule selection from known HITRAN list
            try:
                from .line_catalog import get_available_molecules, hitran_gui_interface
            except Exception:
                # Fallback absolute import
                from cube_explorer.line_catalog import get_available_molecules, hitran_gui_interface

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

            # Mark HITRAN lines with source
            for line in results:
                line['source'] = 'HITRAN'
            
            # Store HITRAN lines (append to existing if any)
            self.hitran_lines = results

            # Merge with LST lines for display
            all_lines = self.lst_lines + self.hitran_lines
            
            # Sort by wavelength (ascending)
            all_lines.sort(key=lambda r: float(r.get('lambda_micron') or r.get('wavelength') or 0.0))

            # If this is the first time (nothing plotted yet), plot all lines first
            if not self.lst_plotted_lines:
                self.plot_selected_lines(all_lines)

            # Show merged selection dialog
            dlg = SpeciesDialog(all_lines, parent=self, plotted_rows=self.lst_plotted_lines)
            if dlg.exec_() == QDialog.Accepted:
                selected_rows = dlg.get_selected_rows()
                if not selected_rows:
                    QMessageBox.information(self, "No Selection", "No lines selected for plotting.")
                    return
                self.plot_selected_lines(selected_rows)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while fetching HITRAN lines:\n{e}")

    def _activate_gaussian_left(self):
        """Activate selection mode for left boundary of Gaussian fit"""
        if self.gauss_left_btn.isChecked():
            self.gauss_mode = 'left'
            self.gauss_right_btn.setChecked(False)
        else:
            self.gauss_mode = None

    def _activate_gaussian_right(self):
        """Activate selection mode for right boundary of Gaussian fit"""
        if self.gauss_right_btn.isChecked():
            self.gauss_mode = 'right'
            self.gauss_left_btn.setChecked(False)
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
            
            # Draw new left marker (blue vertical line)
            self.gauss_left_marker = ax.axvline(x_pos, color='blue', linestyle=':', linewidth=2, label='Left')
            self.gauss_left_x = x_pos
            self.gauss_left_btn.setChecked(False)
            self.gauss_mode = None
            print(f"[DEBUG] Left boundary set at λ = {x_pos:.4f}")
            
        elif self.gauss_mode == 'right':
            # Clear previous right marker if exists
            if hasattr(self, 'gauss_right_marker') and self.gauss_right_marker:
                try:
                    self.gauss_right_marker.remove()
                except:
                    pass
            
            # Draw new right marker (green vertical line)
            self.gauss_right_marker = ax.axvline(x_pos, color='green', linestyle=':', linewidth=2, label='Right')
            self.gauss_right_x = x_pos
            self.gauss_right_btn.setChecked(False)
            self.gauss_mode = None
            print(f"[DEBUG] Right boundary set at λ = {x_pos:.4f}")
        
        self.spec_canvas.draw()

    def _fit_gaussian(self):
        """Perform Gaussian fit to selected spectral line region"""
        # Validation
        if self.gauss_left_x is None or self.gauss_right_x is None:
            QMessageBox.warning(
                self, "Missing Boundaries",
                "Please select both LEFT and RIGHT boundaries first.\n\n"
                "1. Click 'Left' button, then click on spectrum\n"
                "2. Click 'Right' button, then click on spectrum\n"
                "3. Click 'Fit Gaussian'"
            )
            return
        
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            return
        
        if not hasattr(self, 'cube_data') or self.cube_data is None:
            QMessageBox.warning(self, "No Data", "Load a data cube first.")
            return
        
        # Get spectrum data from cube_data
        spectrum_x = self.spec_plotter.wavelength
        spectrum_y = self.cube_data.spectrum
        
        if spectrum_x is None or spectrum_y is None:
            QMessageBox.warning(self, "No Spectrum", "No spectrum data available.")
            return
        
        if len(spectrum_x) != len(spectrum_y):
            QMessageBox.warning(self, "Data Mismatch", "Wavelength and flux arrays have different lengths.")
            return
        
        # Perform fit
        try:
            result = fit_gaussian_to_line(
                self.gauss_left_x, self.gauss_right_x,
                spectrum_x, spectrum_y
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
        from PyQt5.QtWidgets import QTextEdit
        from .unit_converter import clean_wave_unit, clean_flux_unit
        
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

    def _fit_gaussian_map(self):
        """Fit Gaussian to all pixels in the cube using the selected boundaries (runs in background)"""
        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import Qt
        
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
        wavelength = self.spec_plotter.wavelength
        
        if cube is None or wavelength is None:
            QMessageBox.warning(self, "No Data", "No cube data available.")
            return
        
        n_wave, n_y, n_x = cube.shape
        total_pixels = n_y * n_x
        
        # Ask user for sigma threshold
        fac_above_stdres, ok = QInputDialog.getDouble(
            self,
            "Detection Threshold",
            "Enter detection threshold (standard deviations above noise):\n\n"
            "Common values:\n"
            "  • 3σ (standard, ~99.7% confidence)\n"
            "  • 2σ (more detections, ~95% confidence)\n"
            "  • 5σ (very conservative, ~99.9999% confidence)",
            value=3.0,
            min=0.1,
            max=10.0,
            decimals=1
        )
        
        if not ok:
            return  # User cancelled
        
        # Confirm with user
        reply = QMessageBox.question(
            self, "Fit Gaussian Map",
            f"This will fit Gaussians to {total_pixels:,} pixels.\n"
            f"Wavelength range: [{self.gauss_left_x:.4f}, {self.gauss_right_x:.4f}]\n"
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
        
        # Create and start background worker
        self.gauss_worker = GaussianMapWorker(
            cube=cube,
            wavelength=wavelength,
            lam_min=min(self.gauss_left_x, self.gauss_right_x),
            lam_max=max(self.gauss_left_x, self.gauss_right_x),
            n_jobs=8,
            fac_above_stdres=fac_above_stdres
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
            self.gauss_right_x
        )
        dlg.exec_()

    def _save_spectrum_ascii(self):
        """Save the current spectrum displayed in the bottom panel to ASCII file"""
        if not hasattr(self, 'spec_plotter') or self.spec_plotter is None:
            QMessageBox.warning(self, "No Data", "No spectrum loaded to save.")
            return
        
        if self.spec_plotter.wavelength is None:
            QMessageBox.warning(self, "No Data", "No wavelength data available.")
            return
        
        # Get the current spectrum data (use current plotted spectrum)
        wavelength = self.spec_plotter.wavelength
        
        # Determine which spectrum to save (normalized or original)
        if self.is_normalized and self.original_spectrum is not None:
            spectrum = self.spec_plotter.current_spectrum if hasattr(self.spec_plotter, 'current_spectrum') else self.cube_data.spectrum
            spectrum_type = "normalized"
        else:
            spectrum = self.cube_data.spectrum
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
