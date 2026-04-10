"""
Main GUI class for the Dual 2D FITS Viewer
"""

import os
import gc
import warnings
import traceback
import numpy as np
import matplotlib.patches as patches
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import rotate

# Suppress FITS warning messages about header formatting
warnings.filterwarnings('ignore', category=Warning, module='astropy.wcs.wcs')
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QMessageBox, QComboBox, QDialog, QDoubleSpinBox, QApplication, QTextEdit, QInputDialog, QLineEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from .visualization import SlicePlotter
from .colormap_dialog import ColorMapDialog
from .region_selector import RegionSelector

class Ifu_analyzer_2d_window(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("2D FITS Dual Viewer")
        
        self.setWindowFlags(Qt.Window)
        self.setMinimumSize(1200, 600)
        
        self.image_left = None
        self.header_left = None
        self.image_right = None
        self.header_right = None
        
        self._init_state()
        self._init_ui()

    def closeEvent(self, event):
        self.image_left = None
        self.image_right = None
        self.base_image_left = None
        self.base_image_right = None
        gc.collect()
        event.accept()

    def _init_state(self):
        """Initialize state variables"""
        self.active_shape = None
        self.active_target = None
        self.last_selection_left = None
        self.last_selection_right = None
        self.last_contour_right_wcs = None
        self.last_contour_right_level = None
        self.scale_left = 'Linear'
        self.scale_right = 'Linear'
        self.slice1_cmap = 'magma'
        self.slice2_cmap = 'magma'
        
        self.wcs_left = None
        self.wcs_right = None
        self.base_image_left = None
        self.base_header_left = None
        self.cum_angle_left = 0.0
        
        self.base_image_right = None
        self.base_header_right = None
        self.cum_angle_right = 0.0
        
        # Contour tracking variables for saving
        self.last_contour_left_img = None
        self.last_contour_left_wcs = None
        self.last_contour_left_level = None
        
        self.last_contour_right_img = None
        self.last_contour_right_wcs = None
        self.last_contour_right_level = None

    def _init_ui(self):
        # Stats labels - WordWrap enabled to prevent text cutoff
        self.stats1_label = QLabel("Image Stats:\n-")
        self.stats1_label.setMinimumWidth(150)
        self.stats1_label.setWordWrap(True)
        self.stats1_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        
        self.stats2_label = QLabel("Image Stats:\n-")
        self.stats2_label.setMinimumWidth(150)
        self.stats2_label.setWordWrap(True)
        self.stats2_label.setStyleSheet("font-size: 10pt; font-weight: bold;")

        self.hover1_label = QLabel("Cursor Info:\n-")
        self.hover1_label.setMinimumWidth(150)
        self.hover1_label.setWordWrap(True)
        self.hover1_label.setStyleSheet("font-size: 10pt; color: #333;")
        
        self.hover2_label = QLabel("Cursor Info:\n-")
        self.hover2_label.setMinimumWidth(150)
        self.hover2_label.setWordWrap(True)
        self.hover2_label.setStyleSheet("font-size: 10pt; color: #333;")

        self._init_plots()
        self._init_shape_controls()
        self._setup_main_layout()

    def _init_plots(self):
        self.slice1_fig = Figure(figsize=(5, 5), dpi=100)
        self.slice1_canvas = FigureCanvas(self.slice1_fig)
        self.slice1_plotter = SlicePlotter(self.slice1_fig)
        self.slice1_toolbar = NavigationToolbar(self.slice1_canvas, self)
        self.slice1_canvas.mpl_connect("button_press_event", lambda e: self._on_slice_click(e, "left"))
        self.slice1_canvas.mpl_connect("motion_notify_event", lambda e: self._on_mouse_hover(e, "left"))
        
        self.slice2_fig = Figure(figsize=(5, 5), dpi=100)
        self.slice2_canvas = FigureCanvas(self.slice2_fig)
        self.slice2_plotter = SlicePlotter(self.slice2_fig)
        self.slice2_toolbar = NavigationToolbar(self.slice2_canvas, self)
        self.slice2_canvas.mpl_connect("button_press_event", lambda e: self._on_slice_click(e, "right"))
        self.slice2_canvas.mpl_connect("motion_notify_event", lambda e: self._on_mouse_hover(e, "right"))

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

    def _init_shape_controls(self):
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

        shapes = ["*", "▭", "◯", "⬭", "✏"]
        self.shape_buttons = {
            "left": {shape: make_shape_button(shape, "left") for shape in shapes},
            "right": {shape: make_shape_button(shape, "right") for shape in shapes}
        }

        self.color_btn_left = QPushButton("🌈")
        self.color_btn_left.setFixedSize(40, 40)
        self.color_btn_left.clicked.connect(lambda: self._open_colormap_dialog('left'))

        self.color_btn_right = QPushButton("🌈")
        self.color_btn_right.setFixedSize(40, 40)
        self.color_btn_right.clicked.connect(lambda: self._open_colormap_dialog('right'))

        self.scale_combo_left = QComboBox()
        self.scale_combo_left.addItems(["Linear", "Log", "Sqrt", "Power (γ=2)"])
        self.scale_combo_left.currentTextChanged.connect(lambda: self._change_scale('left'))
        
        self.scale_combo_right = QComboBox()
        self.scale_combo_right.addItems(["Linear", "Log", "Sqrt", "Power (γ=2)"])
        self.scale_combo_right.currentTextChanged.connect(lambda: self._change_scale('right'))

        self.erase_btn_l = QPushButton("✖")
        self.erase_btn_l.setFixedSize(40, 40)
        self.erase_btn_l.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.erase_btn_l.clicked.connect(lambda: self.clear_region("left"))

        self.erase_btn_r = QPushButton("✖")
        self.erase_btn_r.setFixedSize(40, 40)
        self.erase_btn_r.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.erase_btn_r.clicked.connect(lambda: self.clear_region("right"))

        # =========================================================
        # BOTTOM PANEL CONTROLS (STYLED LARGER, NO BOLD)
        # =========================================================
        control_style = "font-size: 10pt; padding: 4px;"
        btn_style = "QPushButton { font-size: 11pt; padding: 8px 16px; }"

        # ----- UNIFIED ROTATION CONTROLS -----
        self.rot_dir = QComboBox()
        self.rot_dir.addItems(["Counter-Clockwise", "Clockwise"])
        self.rot_dir.setStyleSheet(control_style)
        
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(0.0, 360.0)
        self.angle_spin.setValue(90.0)
        self.angle_spin.setSuffix(" °")
        self.angle_spin.setStyleSheet(control_style)
        
        self.rotate_btn = QPushButton("Rotate")
        self.rotate_btn.setStyleSheet(btn_style)
        self.rotate_btn.clicked.connect(self._prompt_and_rotate)
        
        self.save_btn = QPushButton("Save FITS")
        self.save_btn.setStyleSheet(btn_style)
        self.save_btn.clicked.connect(self._prompt_and_save)

        # ----- NEW CONTOUR CONTROLS -----
        self.contour_btn = QPushButton("Plot Contour")
        self.contour_btn.setStyleSheet(btn_style)
        self.contour_btn.clicked.connect(self._prompt_and_contour)

        self.superimpose_btn = QPushButton("Superimpose WCS")
        self.superimpose_btn.setStyleSheet(btn_style)
        self.superimpose_btn.clicked.connect(self._prompt_and_superimpose)
        
        self.save_contour_btn = QPushButton("Save Contour Mask")
        self.save_contour_btn.setStyleSheet(btn_style)
        self.save_contour_btn.clicked.connect(self._prompt_and_save_contour)

        self.load_contour_btn = QPushButton("Load Saved Contour")
        self.load_contour_btn.setStyleSheet(btn_style)
        self.load_contour_btn.clicked.connect(self._prompt_and_load_contour)

        self.clear_contour_btn = QPushButton("Clear Contours")
        self.clear_contour_btn.setStyleSheet(btn_style)
        self.clear_contour_btn.clicked.connect(self._prompt_and_clear_contours)

    def _prompt_and_rotate(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Select Target")
        msg.setText("Which slice would you like to rotate?")
        btn_left = msg.addButton("Left", QMessageBox.ActionRole)
        btn_right = msg.addButton("Right", QMessageBox.ActionRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        
        msg.exec_()
        
        if msg.clickedButton() == btn_left:
            self.rotate_image("left")
        elif msg.clickedButton() == btn_right:
            self.rotate_image("right")

    def _prompt_and_save(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Select Target")
        msg.setText("Which slice would you like to save?")
        btn_left = msg.addButton("Left", QMessageBox.ActionRole)
        btn_right = msg.addButton("Right", QMessageBox.ActionRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        
        msg.exec_()
        
        if msg.clickedButton() == btn_left:
            self.save_fits("left")
        elif msg.clickedButton() == btn_right:
            self.save_fits("right")

    def _prompt_and_contour(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Plot WCS Contour")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        # 1. Target Slice
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Plot contour ON:"))
        target_combo = QComboBox()
        target_combo.addItems(["Left Slice", "Right Slice"])
        target_layout.addWidget(target_combo)
        layout.addLayout(target_layout)

        # 2. Source Image
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Generate contour FROM:"))
        source_combo = QComboBox()
        source_combo.addItems(["Same Slice (Self)", "Other Slice (Match WCS)"])
        source_layout.addWidget(source_combo)
        layout.addLayout(source_layout)

        # 3. Background Std — LOCAL variable, not self.sigma_input
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Background Std (blank = Astropy auto):"))
        sigma_input = QLineEdit()          # ← local, not self.
        sigma_input.setPlaceholderText("Auto")
        sigma_input.setMinimumWidth(120)
        sigma_layout.addWidget(sigma_input)
        layout.addLayout(sigma_layout)

        # 4. Sigma levels — comma-separated multipliers
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Sigma levels (e.g. 3, 5, 10):"))
        level_input = QLineEdit()          # ← local, not self.
        level_input.setText("3, 5, 10")
        level_input.setPlaceholderText("e.g. 3, 5, 10")
        level_input.setMinimumWidth(120)
        level_layout.addWidget(level_input)
        layout.addLayout(level_layout)

        layout.addWidget(QLabel(
            "<i>Each level N draws a filled band at N × σ ± 0.05σ above background median.</i>"
        ))

        # 5. Color
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Contour Color:"))
        color_combo = QComboBox()
        color_combo.addItems(["white", "red", "green", "cyan", "magenta", "yellow", "black"])
        color_layout.addWidget(color_combo)
        layout.addLayout(color_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Plot")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        if dialog.exec_() == QDialog.Accepted:
            target      = "left" if target_combo.currentIndex() == 0 else "right"
            source      = "self" if source_combo.currentIndex() == 0 else "other"
            sigma_str   = sigma_input.text().strip()    # ← read from local var
            levels_str  = level_input.text().strip()    # ← read from local var
            color       = color_combo.currentText()
            self._apply_contour(target, source, sigma_str, levels_str, color)

    def _apply_contour(self, target, source, sigma_str, levels_str, color):
        target_plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter

        def get_2d_wcs(header):
            if header is None: return None
            w = WCS(header)
            while w.pixel_n_dim > 2:
                w = w.dropaxis(2)
            return w

        target_wcs = get_2d_wcs(self.header_left if target == 'left' else self.header_right)

        if source == 'self':
            source_img = self.image_left  if target == 'left' else self.image_right
            source_wcs = target_wcs
        else:
            source_img = self.image_right if target == 'left' else self.image_left
            source_wcs = get_2d_wcs(self.header_right if target == 'left' else self.header_left)

        if source_img is None:
            QMessageBox.warning(self, "Missing Data", "Source image is not loaded.")
            return
        if source == 'other' and (self.header_left is None or self.header_right is None):
            QMessageBox.warning(self, "Missing Data", "Both images must be loaded.")
            return

        # ── Parse sigma levels ────────────────────────────────────────────────
        try:
            sigma_levels = [float(x.strip()) for x in levels_str.split(',') if x.strip()]
        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                                "Levels must be comma-separated numbers (e.g. 3, 5, 10).")
            return
        if not sigma_levels:
            QMessageBox.warning(self, "Invalid Input", "Enter at least one sigma level.")
            return

        # ── Calculate background stats ────────────────────────────────────────
        valid = source_img[np.isfinite(source_img)]
        if valid.size == 0:
            return
        _, median_val, astropy_std = sigma_clipped_stats(valid, sigma=3.0)

        if sigma_str:
            try:
                bkg_std = float(sigma_str)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input",
                                    "Background Std must be a valid number.")
                return
        else:
            bkg_std = astropy_std

        # ── Build band-pairs: each level N → [N-0.05, N+0.05] × std + median ─
        band_levels = sorted([median_val + n * bkg_std for n in sigma_levels])

        ax = target_plotter.ax
        if not hasattr(target_plotter, 'contours'):
            target_plotter.contours = []

        # Track for save/clear
        if target == 'left':
            self.last_contour_left_img   = source_img
            self.last_contour_left_wcs   = source_wcs
            self.last_contour_left_level = band_levels[0]
        else:
            self.last_contour_right_img   = source_img
            self.last_contour_right_wcs   = source_wcs
            self.last_contour_right_level = band_levels[0]

        try:
            kwargs = {'levels': band_levels, 'alpha': 0.8}
            if source == 'other':
                kwargs['transform'] = ax.get_transform(source_wcs)
            cset = ax.contour(source_img, colors=[color], linewidths=1.5, **kwargs)
            
            target_plotter.contours.append(cset)
            target_plotter.current_contour = cset

            if target == 'left': self.slice1_canvas.draw()
            else:                 self.slice2_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Contour Error",
                                f"Could not generate contours:\n{str(e)}")
            traceback.print_exc()

    def _prompt_and_save_contour(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Save Contour")
        msg.setText("Which slice's contour would you like to save?")
        btn_left  = msg.addButton("Left",   QMessageBox.ActionRole)
        btn_right = msg.addButton("Right",  QMessageBox.ActionRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.exec_()
        if msg.clickedButton() == btn_left:
            self._save_contour_txt("left")
        elif msg.clickedButton() == btn_right:
            self._save_contour_txt("right")

    def _save_contour_txt(self, target):
        plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter
        wcs_obj = self.wcs_left       if target == 'left' else self.wcs_right
        cset    = getattr(plotter, 'current_contour', None)

        if cset is None:
            QMessageBox.warning(self, "No Contour",
                                f"No active contour found on the {target} panel.")
            return

        fname, _ = QFileDialog.getSaveFileName(
            self, f"Save Contour ({target.capitalize()})",
            filter="Text files (*.txt);;CSV files (*.csv)")
        if not fname:
            return
        if not (fname.endswith('.txt') or fname.endswith('.csv')):
            fname += '.txt'

        try:
            rows = ["# IFU Analyzer 2D — Contour Export\n",
                    "# level, x_pixel, y_pixel, RA_deg, Dec_deg\n"]

            collections = cset.collections if hasattr(cset, 'collections') else [cset]
            levels      = list(cset.levels) if hasattr(cset, 'levels') else []

            for i, coll in enumerate(collections):
                level = levels[i] if i < len(levels) else i
                for path in coll.get_paths():
                    for x, y in path.vertices:
                        ra_str = dec_str = ''
                        if wcs_obj is not None:
                            try:
                                sky    = wcs_obj.pixel_to_world(x, y)
                                ra_str  = f"{sky.ra.deg:.8f}"
                                dec_str = f"{sky.dec.deg:.8f}"
                            except Exception:
                                pass
                        rows.append(f"{level:.6e}, {x:.4f}, {y:.4f}, {ra_str}, {dec_str}\n")

            with open(fname, 'w') as f:
                f.writelines(rows)

            QMessageBox.information(self, "Saved", f"Contour coordinates saved to:\n{fname}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save contour:\n{str(e)}")
            traceback.print_exc()

    def _prompt_and_load_contour(self):
        """Prompt user for target slice, mask file, and color to overlay a saved contour."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Load Saved Contour")
        msg.setText("Which slice would you like to plot the loaded contour ON?")
        btn_left = msg.addButton("Left", QMessageBox.ActionRole)
        btn_right = msg.addButton("Right", QMessageBox.ActionRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        
        msg.exec_()
        
        if msg.clickedButton() == btn_left:
            target = "left"
        elif msg.clickedButton() == btn_right:
            target = "right"
        else:
            return
            
        # FIX: Now looks for the TXT files we generated, instead of FITS
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Contour Text File", filter="Text files (*.txt *.csv)"
        )
        if not fname:
            return
            
        colors = ["white", "red", "green", "cyan", "magenta", "yellow", "black"]
        color, ok = QInputDialog.getItem(
            self, "Contour Color", "Select contour line color:", colors, 0, False
        )
        
        if ok and color:
            self._apply_loaded_contour(target, fname, color)

        
    def _apply_loaded_contour(self, target, fname, color):
        """Loads a saved text mask and plots it as smooth lines."""
        target_plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter
        target_img = self.image_left if target == 'left' else self.image_right
        
        if target_img is None:
            QMessageBox.warning(self, "No Target Image", f"Load an image in the {target} panel first.")
            return

        try:
            # Parse the TXT file generated by _save_contour_txt
            data = np.genfromtxt(fname, delimiter=',', comments='#')
            if data.ndim == 1: data = data.reshape(1, -1)
            
            # Extract x and y coordinates (columns 1 and 2)
            x_pts = data[:, 1]
            y_pts = data[:, 2]
            
            # MAGIC FIX: Prevent drawing lines BETWEEN separate contour islands.
            # Calculate distance between consecutive points. If the jump is > 5 pixels,
            # it means we are jumping to a new contour ring. We insert a NaN to break the line.
            dist = np.sqrt(np.diff(x_pts)**2 + np.diff(y_pts)**2)
            break_indices = np.where(dist > 5.0)[0] + 1 
            
            x_plot = np.insert(x_pts, break_indices, np.nan)
            y_plot = np.insert(y_pts, break_indices, np.nan)
            
            ax = target_plotter.ax

            if not hasattr(target_plotter, 'contours'):
                target_plotter.contours = []
            
            # Plot as a continuous line (with calculated breaks for separate islands)
            contour_line, = ax.plot(x_plot, y_plot, color=color, linewidth=1.5, alpha=0.8, zorder=10)
            
            # Wrap the Line2D object in a dummy object so the clear_contours function 
            # (which expects collections) doesn't crash when trying to remove it later.
            class DummyContourSet:
                def __init__(self, line):
                    self.collections = [line]
            
            target_plotter.contours.append(DummyContourSet(contour_line))
            target_plotter.current_contour = DummyContourSet(contour_line)
            
            # Redraw canvas
            if target == 'left': self.slice1_canvas.draw()
            else: self.slice2_canvas.draw()
            
            QMessageBox.information(self, "Loaded", f"Successfully loaded contour from {os.path.basename(fname)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to plot saved contour:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _setup_main_layout(self):
        """Set up the main application layout with a fully modular, future-proof bottom panel"""
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        # ================= LEFT HALF =================
        left_half = QHBoxLayout()
        
        self.open_left_btn = QPushButton("Open 2D FITS (Left)")
        self.open_left_btn.clicked.connect(lambda: self.load_2d_fits("left"))
        
        self.header_left_btn = QPushButton("Header")
        self.header_left_btn.clicked.connect(lambda: self.show_header("left"))
        
        # Left Stats strictly bounded so they don't stretch Canvas space
        left_stats_layout = QVBoxLayout()
        left_stats_layout.addWidget(self.open_left_btn)
        left_stats_layout.addWidget(self.header_left_btn)
        left_stats_layout.addWidget(self.stats1_label)
        left_stats_layout.addSpacing(15) 
        left_stats_layout.addWidget(self.hover1_label)
        left_stats_layout.addStretch()
        
        left_stats_widget = QWidget()
        left_stats_widget.setLayout(left_stats_layout)
        left_stats_widget.setFixedWidth(180)

        # Left Canvas allowed to freely resize
        left_slice_layout = QVBoxLayout()
        left_slice_layout.addWidget(self.slice1_canvas)
        left_slice_layout.addWidget(self.slice1_toolbar)
        
        left_slice_widget = QWidget()
        left_slice_widget.setLayout(left_slice_layout)

        # Left Sidebar strictly bounded
        left_sidebar = QVBoxLayout()
        left_sidebar.addWidget(self.color_btn_left)
        left_sidebar.addWidget(QLabel("Scale:"))
        left_sidebar.addWidget(self.scale_combo_left)
        for shape in ["*", "▭", "◯", "⬭", "✏"]:
            left_sidebar.addWidget(self.shape_buttons["left"][shape])
        left_sidebar.addWidget(self.erase_btn_l)
        left_sidebar.addStretch()

        left_sidebar_widget = QWidget()
        left_sidebar_widget.setLayout(left_sidebar)
        left_sidebar_widget.setFixedWidth(120)

        left_half.addWidget(left_stats_widget, 0)
        left_half.addWidget(left_slice_widget, 1)
        left_half.addWidget(left_sidebar_widget, 0)

        # ================= RIGHT HALF =================
        right_half = QHBoxLayout()
        
        self.open_right_btn = QPushButton("Open 2D FITS (Right)")
        self.open_right_btn.clicked.connect(lambda: self.load_2d_fits("right"))
        
        self.header_right_btn = QPushButton("Header")
        self.header_right_btn.clicked.connect(lambda: self.show_header("right"))

        # Right Stats strictly bounded
        right_stats_layout = QVBoxLayout()
        right_stats_layout.addWidget(self.open_right_btn)
        right_stats_layout.addWidget(self.header_right_btn)
        right_stats_layout.addWidget(self.stats2_label)
        right_stats_layout.addSpacing(15)
        right_stats_layout.addWidget(self.hover2_label)
        right_stats_layout.addStretch()
        
        right_stats_widget = QWidget()
        right_stats_widget.setLayout(right_stats_layout)
        right_stats_widget.setFixedWidth(180)

        # Right Canvas allowed to freely resize
        right_slice_layout = QVBoxLayout()
        right_slice_layout.addWidget(self.slice2_canvas)
        right_slice_layout.addWidget(self.slice2_toolbar)
        
        right_slice_widget = QWidget()
        right_slice_widget.setLayout(right_slice_layout)

        # Right Sidebar strictly bounded
        right_sidebar = QVBoxLayout()
        right_sidebar.addWidget(self.color_btn_right)
        right_sidebar.addWidget(QLabel("Scale:"))
        right_sidebar.addWidget(self.scale_combo_right)
        for shape in ["*", "▭", "◯", "⬭", "✏"]:
            right_sidebar.addWidget(self.shape_buttons["right"][shape])
        right_sidebar.addWidget(self.erase_btn_r)
        right_sidebar.addStretch()

        right_sidebar_widget = QWidget()
        right_sidebar_widget.setLayout(right_sidebar)
        right_sidebar_widget.setFixedWidth(120)

        right_half.addWidget(right_stats_widget, 0)
        right_half.addWidget(right_slice_widget, 1)
        right_half.addWidget(right_sidebar_widget, 0)

        # Add halves to top layout
        top_layout.addLayout(left_half, 1)
        top_layout.addLayout(right_half, 1)

        # ================= BOTTOM CONTROLS =================
        self.bottom_panel = QWidget()
        bottom_layout = QHBoxLayout(self.bottom_panel)

        lbl_style = "font-size: 11pt; color: #333;"

        # Left controls (Rotate)
        rotate_widget = QWidget()
        rotate_layout = QVBoxLayout(rotate_widget)
        rotate_layout.setContentsMargins(0,0,0,0)
        rotate_lbl = QLabel("Actions:")
        rotate_lbl.setStyleSheet(lbl_style)
        rotate_layout.addWidget(rotate_lbl)
        rotate_layout.addWidget(self.rot_dir)
        rotate_layout.addWidget(self.angle_spin)
        rotate_layout.addWidget(self.rotate_btn)
        
        # Right controls (Contours)
        contour_widget = QWidget()
        contour_layout = QVBoxLayout(contour_widget)
        contour_layout.setContentsMargins(0,0,0,0)
        contour_lbl = QLabel("Contours:")
        contour_lbl.setStyleSheet(lbl_style)
        contour_layout.addWidget(contour_lbl)
        contour_layout.addWidget(self.contour_btn)
        contour_layout.addWidget(self.superimpose_btn)
        contour_layout.addWidget(self.save_contour_btn)
        contour_layout.addWidget(self.load_contour_btn)
        contour_layout.addWidget(self.clear_contour_btn)

        # Middle controls (Save FITS)
        save_widget = QWidget()
        save_layout = QVBoxLayout(save_widget)
        save_layout.setContentsMargins(0,0,0,0)
        save_layout.addStretch() # Push the button down to align with the bottom of the other stacks
        save_layout.addWidget(self.save_btn)

        # Spacing logic adjusted to push everything left based on user feedback
        bottom_layout.addStretch(1) # Smallest stretch on the left to pull controls leftward
        bottom_layout.addWidget(rotate_widget)
        bottom_layout.addSpacing(30)
        bottom_layout.addWidget(contour_widget)
        
        bottom_layout.addStretch(3) # Shifts Save FITS leftward, roughly aligning it between the canvases
        bottom_layout.addWidget(save_widget, alignment=Qt.AlignBottom)
        bottom_layout.addStretch(6) # Larger stretch on right pushes everything left

        # ================= ASSEMBLE MAIN LAYOUT =================
        main_layout.addLayout(top_layout, 2)
        main_layout.addStretch(1) # Allows canvases to remain vertically dominant/square
        main_layout.addWidget(self.bottom_panel, 0)

        self.setLayout(main_layout)

    def show_header(self, target):
        header = self.header_left if target == "left" else self.header_right
        
        if header is None:
            QMessageBox.warning(self, "No Image", "Please load an image first to view its header.")
            return

        header_text = "\n".join(
            f"{k:8} = {v!r} / {header.comments[k]}" if k in header.comments else f"{k:8} = {v!r}" 
            for k, v in header.items()
        )
        
        header_window = QWidget()
        header_window.setWindowTitle(f"FITS Header ({target.capitalize()})")
        header_window.setMinimumSize(600, 500)
        
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFontFamily("Courier")
        text_edit.setText(header_text)
        layout.addWidget(text_edit)
        
        header_window.setLayout(layout)
        header_window.show()
        
        if target == "left":
            self.header_window_left = header_window
        else:
            self.header_window_right = header_window

    def rebuild_axes(self, target, header):
        plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter
        canvas  = self.slice1_canvas  if target == 'left' else self.slice2_canvas

        plotter.fig.clf()
        ax_rect   = [0.25, 0.20, 0.50, 0.66]
        cbar_rect = [0.80, 0.20, 0.03, 0.66]

        wcs_for_projection = None

        # ── Step 1: Build a clean, strictly-2D, SIP-free WCS ──────────────────
        try:
            ch = header.copy()
            ch['NAXIS']   = 2
            ch['WCSAXES'] = 2

            # Remove every 3-D / spectral / distortion key that breaks WCSAxes
            bad_keys = [
                'NAXIS3', 'NAXIS4',
                'CTYPE3', 'CRVAL3', 'CRPIX3', 'CDELT3', 'CUNIT3',
                'PC1_3',  'PC2_3',  'PC3_1',  'PC3_2',  'PC3_3',
                'CD1_3',  'CD2_3',  'CD3_1',  'CD3_2',  'CD3_3',
                'A_ORDER','B_ORDER','AP_ORDER','BP_ORDER',
            ]
            # Dynamically collect SIP coefficient keys (A_i_j, B_i_j, AP_i_j, BP_i_j)
            sip_keys = [k for k in ch.keys()
                        if k.startswith(('A_', 'B_', 'AP_', 'BP_')) and k != 'BUNIT']
            bad_keys.extend(sip_keys)
            for k in bad_keys:
                if k in ch:
                    del ch[k]

            # Strip '-SIP' suffix from CTYPE1 / CTYPE2 if present
            for ct in ('CTYPE1', 'CTYPE2'):
                if ct in ch and '-SIP' in str(ch[ct]):
                    ch[ct] = str(ch[ct]).replace('-SIP', '')

            wcs_obj = WCS(ch, naxis=2)

            wcs_for_projection = wcs_obj.celestial if wcs_obj.has_celestial else wcs_obj
            print(f"DEBUG rebuild_axes [{target}]: WCS parsed OK → {wcs_for_projection}")

        except Exception as e:
            print(f"DEBUG rebuild_axes [{target}]: WCS parse FAILED → {e}")
            wcs_for_projection = None
            
        # Store so hover readout can use it
        if target == 'left':
            self.wcs_left  = wcs_for_projection
        else:
            self.wcs_right = wcs_for_projection

        # ── Step 2: Create axes (WCSAxes if possible, plain fallback) ──────────
        try:
            if wcs_for_projection is None:
                raise ValueError("No valid WCS available for projection")
            plotter.ax = plotter.fig.add_axes(ax_rect, projection=wcs_for_projection)
            print(f"DEBUG rebuild_axes [{target}]: WCSAxes created ✓")
        except Exception as e:
            print(f"DEBUG rebuild_axes [{target}]: WCSAxes FAILED, using plain axes → {e}")
            plotter.ax = plotter.fig.add_axes(ax_rect)

        plotter.cbar_ax = plotter.fig.add_axes(cbar_rect)
        plotter.cbar_ax.set_visible(False)

        if target == 'left':
            self.region_selector_left = RegionSelector(
                plotter.ax, canvas,
                lambda mask, patch, coords: self._handle_region_select(mask, patch, coords, 'left'))
        else:
            self.region_selector_right = RegionSelector(
                plotter.ax, canvas,
                lambda mask, patch, coords: self._handle_region_select(mask, patch, coords, 'right'))

    def load_2d_fits(self, target):
        fname, _ = QFileDialog.getOpenFileName(
            self, f"Open 2D FITS ({target.capitalize()})", filter="FITS files (*.fits *.fz *.fit)")
        
        if not fname:
            return

        try:
            self.clear_region(target)
            
            with fits.open(fname) as hdul:
                data = None
                header = None
                for hdu in hdul:
                    if hdu.data is not None:
                        data = hdu.data
                        header = hdu.header
                        break
                
                if data is None:
                    QMessageBox.warning(self, "Load Error", "No image data found in FITS file.")
                    return

                data = np.squeeze(data)
                if data.ndim != 2:
                    QMessageBox.warning(self, "Dimension Error", 
                                        f"Expected a 2D image, but got a {data.ndim}D array.\n"
                                        "Please select a 2D FITS file.")
                    return

                # --- NEW CLEANUP BLOCK: Remove dangling 3D JWST keys ---
                header['NAXIS'] = 2
                if 'WCSAXES' in header:
                    header['WCSAXES'] = 2
                
                bad_keys = ['NAXIS3', 'NAXIS4','CTYPE3', 'CRVAL3', 'CRPIX3', 'CDELT3', 'CUNIT3', 
                            'PC1_3', 'PC2_3', 'PC3_1', 'PC3_2', 'PC3_3',
                            'CD1_3', 'CD2_3', 'CD3_1', 'CD3_2', 'CD3_3']
                for k in bad_keys:
                    if k in header:
                        del header[k]
                # -------------------------------------------------------

                if target == "left":
                    self.base_image_left = data.copy()
                    self.base_header_left = header.copy()
                    self.cum_angle_left = 0.0
                    self.image_left = data
                    self.header_left = header
                else:
                    self.base_image_right = data.copy()
                    self.base_header_right = header.copy()
                    self.cum_angle_right = 0.0
                    self.image_right = data
                    self.header_right = header

                self.rebuild_axes(target, header)
                filename = os.path.basename(fname)
                self.display_image(target, title=filename)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading FITS file:\n{str(e)}")
            traceback.print_exc()

    def rotate_image(self, target):
        base_image = self.base_image_left if target == "left" else self.base_image_right
        base_header = self.base_header_left if target == "left" else self.base_header_right
        
        if base_image is None or base_header is None:
            QMessageBox.warning(self, "No Image", f"Please load an image into the {target} panel first.")
            return
            
        angle = self.angle_spin.value()
        if angle == 0.0:
            return
            
        if self.rot_dir.currentText() == "Clockwise":
            angle = -angle

        if target == "left":
            self.cum_angle_left += angle
            total_angle = self.cum_angle_left
        else:
            self.cum_angle_right += angle
            total_angle = self.cum_angle_right

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            wcs_in_full = WCS(base_header)
            if not wcs_in_full.has_celestial:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "WCS Error", "Image does not have a valid celestial WCS.")
                return
                
            if wcs_in_full.pixel_n_dim > 2:
                wcs_in = wcs_in_full.celestial
            else:
                wcs_in = wcs_in_full
                
            wcs_out = wcs_in.deepcopy()
            theta = np.deg2rad(total_angle)
            
            array_out = rotate(base_image, total_angle, reshape=True, cval=np.nan, order=1)
            
            cos_t, sin_t = np.cos(-theta), np.sin(-theta)
            R_inv = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            
            if hasattr(wcs_out.wcs, 'cd'):
                wcs_out.wcs.cd = np.dot(wcs_out.wcs.cd, R_inv)
            elif hasattr(wcs_out.wcs, 'pc'):
                wcs_out.wcs.pc = np.dot(wcs_out.wcs.pc, R_inv)
            else:
                wcs_out.wcs.pc = R_inv
                
            ny, nx = base_image.shape
            new_ny, new_nx = array_out.shape
            
            cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
            new_cx, new_cy = (new_nx - 1) / 2.0, (new_ny - 1) / 2.0
            
            crpix_0 = wcs_in.wcs.crpix - 1
            dx, dy = crpix_0[0] - cx, crpix_0[1] - cy
            
            cos_r, sin_r = np.cos(theta), np.sin(theta)
            new_dx = dx * cos_r - dy * sin_r
            new_dy = dx * sin_r + dy * cos_r
            
            wcs_out.wcs.crpix = np.array([new_cx + new_dx, new_cy + new_dy]) + 1
            
            new_header = base_header.copy()
            for key in list(wcs_in_full.to_header().keys()):
                if key in new_header:
                    del new_header[key]
                    
            wcs_header = wcs_out.to_header()
            for key in wcs_header:
                new_header[key] = wcs_header[key]
            
            new_header['NAXIS'] = 2
            new_header['NAXIS1'] = new_nx
            new_header['NAXIS2'] = new_ny
            if 'NAXIS3' in new_header:
                del new_header['NAXIS3']
            if 'WCSAXES' in new_header:
                new_header['WCSAXES'] = 2
                    
            if target == "left":
                self.image_left = array_out
                self.header_left = new_header
            else:
                self.image_right = array_out
                self.header_right = new_header
                
            self.rebuild_axes(target, new_header)
            self.clear_region(target)
            
            display_angle = total_angle % 360 if total_angle >= 0 else -(-total_angle % 360)
            self.display_image(target, title=f"Rotated {display_angle:.1f}° Total")
            
        except Exception as e:
            QMessageBox.critical(self, "Rotation Error", f"An error occurred during rotation:\n{str(e)}")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

    def save_fits(self, target):
            image = self.image_left if target == "left" else self.image_right
            header = self.header_left if target == "left" else self.header_right
            
            if image is None or header is None:
                QMessageBox.warning(self, "No Image", f"There is no image currently loaded in the {target} panel to save.")
                return
                
            fname, _ = QFileDialog.getSaveFileName(
                self, f"Save FITS ({target.capitalize()})", filter="FITS files (*.fits)"
            )
            if not fname:
                return
                
            if not fname.lower().endswith('.fits'):
                fname += '.fits'
                
            try:
                hdu = fits.PrimaryHDU(data=image, header=header)
                hdu.writeto(fname, overwrite=True)
                QMessageBox.information(self, "Success", f"FITS successfully saved to:\n{fname}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save FITS file:\n{str(e)}")

    def display_image(self, target, title=''):
        """Displays the 2D image with chosen scaling, colormap, and WCS grid."""
        image  = self.image_left   if target == 'left' else self.image_right
        header = self.header_left  if target == 'left' else self.header_right
        if image is None:
            return

        plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter
        stats_label = self.stats1_label if target == 'left' else self.stats2_label
        cmap  = getattr(self, 'slice1_cmap' if target == 'left' else 'slice2_cmap', 'magma')
        scale = self.scale_left if target == 'left' else self.scale_right

        ax = plotter.ax
        for artist in list(ax.images):
            artist.remove()
        for coll in list(ax.collections):
            coll.remove()
        for line in list(ax.lines):
            line.remove()
        ax.set_title('')

        # 1. Calculate Intensity Normalization
        finite = image[np.isfinite(image)]
        vmin   = float(np.nanmin(finite)) if finite.size else 0.0
        vmax   = float(np.nanmax(finite)) if finite.size else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0

        from matplotlib.colors import LogNorm, PowerNorm, Normalize
        if scale == 'Log':
            # Log scale requires strictly positive values
            vmin_log = max(vmin, 1e-10)
            norm = LogNorm(vmin=vmin_log, vmax=vmax)
        elif scale == 'Sqrt':
            vmin_sqrt = max(vmin, 0.0)
            norm = PowerNorm(gamma=0.5, vmin=vmin_sqrt, vmax=vmax)
        elif scale == 'Power (γ=2)':
            norm = PowerNorm(gamma=2.0, vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        # 2. Render the Image
        im = ax.imshow(image, origin='lower', cmap=cmap, norm=norm,
               interpolation='nearest', aspect='auto')
        ax.set_title('', pad=4)
        # 3. WCS Grid and Labels
        unit_str = ""
        if header is not None:
            if 'BUNIT' in header:
                unit_str = f" [{header['BUNIT']}]"
                
            # Only attempt grid if the axes was built as a WCSAxes object
            if hasattr(ax, 'coords'):
                try:
                    ax.set_axis_on()
                    ax.coords.grid(color='white', alpha=0.5, linestyle='solid')

                    try:
                        ra  = ax.coords['ra']
                        dec = ax.coords['dec']
                    except (KeyError, TypeError):
                        ra, dec = ax.coords[0], ax.coords[1]

                    # Choose formatter precision based on FOV — let Astropy auto-pick spacing
                    import astropy.units as u
                    cdelt    = abs(header.get('CDELT1', header.get('CD1_1', 0.0)))
                    ny, nx   = image.shape
                    fov_amin = cdelt * max(nx, ny) * 60.0   # arcmin

                    if fov_amin > 10.0:
                        ra.set_major_formatter('hh:mm:ss')
                        dec.set_major_formatter('dd:mm:ss')
                    elif fov_amin > 1.0:
                        ra.set_major_formatter('hh:mm:ss.s')
                        dec.set_major_formatter('dd:mm:ss')
                    else:
                        # Sub-arcmin field — needs sub-second RA precision
                        ra.set_major_formatter('hh:mm:ss.ss')
                        dec.set_major_formatter('dd:mm:ss.s')

                    # ← NO set_ticks() call — Astropy auto-locates ticks for the current FOV

                    ra.set_axislabel('RA',   minpad=0.5)
                    dec.set_axislabel('Dec', minpad=0.5)
                    ra.set_ticklabel_position('tblr')
                    dec.set_ticklabel_position('tblr')
                    ra.set_axislabel_position('b')
                    dec.set_axislabel_position('l')
                    ra.set_ticks_visible(True)
                    dec.set_ticks_visible(True)
                    ra.set_ticklabel_visible(True)
                    dec.set_ticklabel_visible(True)

                    ra.set_ticklabel(color='black', size=7, rotation='vertical', pad=2)
                    dec.set_ticklabel(color='black', size=7)
                    ra.set_ticks(color='black', width=1)
                    dec.set_ticks(color='black', width=1)
                    ra.set_axislabel('RA',  color='black', size=8)
                    dec.set_axislabel('Dec', color='black', size=8)

                    print(f"DEBUG WCS grid: fov={fov_amin:.3f} arcmin, auto-spacing enabled")

                except Exception as e:
                    QMessageBox.warning(self, "Grid Error", f"Failed to draw WCS grid:\n{str(e)}")
                    traceback.print_exc()
                print(f"DEBUG coords type: {type(ax.coords)}")
                print(f"DEBUG ax position: {ax.get_position()}")

        # 4. Colorbar Management
        if not hasattr(plotter, 'cbar_ax') or plotter.cbar_ax is None:
            # Defined placement: [left, bottom, width, height] in figure coordinates
            plotter.cbar_ax = plotter.fig.add_axes([0.84, 0.10, 0.04, 0.82])

        plotter.cbar_ax.cla()
        plotter.cbar_ax.set_visible(True)

        cbar = plotter.fig.colorbar(im, cax=plotter.cbar_ax)
        cbar.ax.tick_params(labelsize=7)

        if header is not None and 'BUNIT' in header:
            cbar.set_label(header['BUNIT'], size=9, weight='bold', labelpad=5)
            
        plotter.cbar = cbar

        # 5. Update Statistics Panel
        mn   = np.nanmin(image)
        mx   = np.nanmax(image)
        mean = np.nanmean(image)
        stats_label.setText(
            f"Image Stats{unit_str}\n"
            f"min={mn:.3e}\n"
            f"max={mx:.3e}\n"
            f"mean={mean:.3e}"
        )

        # 6. Refresh Canvas
        if target == 'left':
            self.slice1_canvas.draw()
        else:
            self.slice2_canvas.draw()

    def _on_mouse_hover(self, event, target):
        """Handle dynamic updates of mouse coordinate position and value mapping"""
        plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter
        image = self.image_left if target == 'left' else self.image_right
        header = self.header_left if target == 'left' else self.header_right
        hover_label = self.hover1_label if target == 'left' else self.hover2_label

        if image is None or event.inaxes != plotter.ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        ix, iy = int(round(x)), int(round(y))
        
        # Pull actual intensity level
        if 0 <= iy < image.shape[0] and 0 <= ix < image.shape[1]:
            val = float(image[iy, ix])
        else:
            val = float('nan')

        # Translate Matplotlib X,Y coordinate to World Coordinate RA/DEC
        wcs_str = ""
        wcs_obj = self.wcs_left if target == 'left' else self.wcs_right
        
        if wcs_obj is not None and wcs_obj.has_celestial:
            try:
                wcs2d = wcs_obj.celestial if wcs_obj.pixel_n_dim > 2 else wcs_obj
                world = wcs2d.pixel_to_world(x, y)
                ra = world.ra.to_string(unit='hour', sep='hms', precision=2, pad=True)
                dec = world.dec.to_string(unit='deg', sep='dms', precision=1, alwayssign=True, pad=True)
                wcs_str = f"RA: {ra}\nDec: {dec}\n"
            except Exception:
                pass

        unit_str = ""
        if header is not None and 'BUNIT' in header:
            unit_str = f" [{header['BUNIT']}]"

        # Update the live UI label directly
        text = f"Cursor Info:\nPixel: {ix}, {iy}\n{wcs_str}Val: {val:.4e}{unit_str}"
        hover_label.setText(text)

    def _set_shape(self, shape, target):
        if self.active_shape == shape and self.active_target == target:
            self.active_shape = None
            self.active_target = None
        else:
            self.active_shape = shape
            self.active_target = target

        for side in ["left", "right"]:
            for btn_shape, btn in self.shape_buttons[side].items():
                btn.setChecked(side == target and btn_shape == shape and self.active_shape is not None)

        selector = self.region_selector_left if target == "left" else self.region_selector_right
        selector.activate_selector(shape if self.active_shape else None)

    def _handle_region_select(self, mask, patch, coords, target):
        image = self.image_left if target == "left" else self.image_right
        header = self.header_left if target == "left" else self.header_right
        if image is None: return

        xmin, xmax, ymin, ymax = coords

        if mask.shape == image.shape:
            full_mask = mask
        else:
            full_mask = np.zeros_like(image, dtype=bool)
            h_mask, w_mask = mask.shape
            y0, x0 = int(ymin), int(xmin)
            y1, x1 = min(y0 + h_mask, image.shape[0]), min(x0 + w_mask, image.shape[1])
            full_mask[y0:y1, x0:x1] = mask[:(y1-y0), :(x1-x0)]

        selection = {"shape": self.active_shape, "coords": coords, "mask": full_mask, "patch": patch}
        
        if target == "left": self.last_selection_left = selection
        else: self.last_selection_right = selection

        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
        
        if hasattr(plotter.ax, 'area_artists'):
            for artist in getattr(plotter.ax, 'area_artists'):
                try: artist.remove()
                except: pass
        plotter.ax.area_artists = []

        plotter.ax.add_patch(patch)
        plotter.ax.area_artists.append(patch)
        
        selector = self.region_selector_left if target == "left" else self.region_selector_right
        if selector.selector is not None:
            try: selector.selector.set_visible(False)
            except: pass
        
        self.active_shape = None
        self.active_target = None
        for side in ["left", "right"]:
            for btn in self.shape_buttons[side].values():
                btn.setChecked(False)
        selector.activate_selector(None)

        selected_area = image[full_mask]
        if selected_area.size > 0:
            mn, mx, mean = np.nanmin(selected_area), np.nanmax(selected_area), np.nanmean(selected_area)
            total_sum, std = np.nansum(selected_area), np.nanstd(selected_area)
            
            stats_label = self.stats1_label if target == "left" else self.stats2_label
            current_text = stats_label.text().split("\n")
            slice_stats = [line for line in current_text if line.startswith("Image Stats")]
            
            unit_str = ""
            if header is not None and 'BUNIT' in header:
                unit_str = f" [{header['BUNIT']}]"
                
            region_stats = [
                f"Selected Area Stats{unit_str}:",
                f"min={mn:.3e}", f"max={mx:.3e}", f"mean={mean:.3e}",
                f"sum={total_sum:.3e}", f"std={std:.3e}"
            ]
            combined_stats = slice_stats + [""] + region_stats if slice_stats else region_stats
            stats_label.setText("\n".join(combined_stats))

        if target == "left": self.slice1_canvas.draw()
        else: self.slice2_canvas.draw()

    def clear_region(self, target):
        if target == "left":
            self.last_selection_left = None
            plotter = self.slice1_plotter
        else:
            self.last_selection_right = None
            plotter = self.slice2_plotter

        # Clear physical shapes
        if hasattr(plotter.ax, 'area_artists'):
            for artist in plotter.ax.area_artists:
                try: artist.remove()
                except: pass
            plotter.ax.area_artists = []

        # Clear pixel marks
        if hasattr(plotter.ax, 'pixel_marker'):
            for marker in plotter.ax.pixel_marker:
                try: marker.remove()
                except: pass
            plotter.ax.pixel_marker = []
            
        # Clear contours (compatible with Matplotlib 3.8+)
        if hasattr(plotter, 'current_contour') and plotter.current_contour:
            if hasattr(plotter.current_contour, 'collections'):
                for c in plotter.current_contour.collections:
                    try: c.remove()
                    except: pass
            else:
                try: plotter.current_contour.remove()
                except: pass
            plotter.current_contour = None
            
        if target == "left":
            self.last_contour_left_img = None
            self.last_contour_left_wcs = None
            self.last_contour_left_level = None
        else:
            self.last_contour_right_img = None
            self.last_contour_right_wcs = None
            self.last_contour_right_level = None

        if target == "left" and self.image_left is not None:
            self.display_image("left", "Left Image")
        elif target == "right" and self.image_right is not None:
            self.display_image("right", "Right Image")

    def _handle_pixel_click(self, x, y, target):
        image = self.image_left if target == "left" else self.image_right
        header = self.header_left if target == "left" else self.header_right
        if image is None: return
                
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            intensity = float(image[y, x])
            
            if target == "left": self.last_selection_left = {"shape": "*", "coords": (x, y), "mask": None}
            else: self.last_selection_right = {"shape": "*", "coords": (x, y), "mask": None}

            stats_label = self.stats1_label if target == "left" else self.stats2_label
            
            unit_str = ""
            if header is not None and 'BUNIT' in header:
                unit_str = f" {header['BUNIT']}"
                
            stats_label.setText(f"Pixel: (x={x}, y={y})\nIntensity={intensity:.3e}{unit_str}")

            plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
            
            if hasattr(plotter.ax, 'area_artists'):
                for artist in plotter.ax.area_artists:
                    try: artist.remove()
                    except: pass
                plotter.ax.area_artists = []

            if hasattr(plotter.ax, 'pixel_marker'):
                for marker in plotter.ax.pixel_marker:
                    marker.remove()
                plotter.ax.pixel_marker = []

            plotter.update_pixel_marker(x, y)
            
            if target == "left": self.slice1_canvas.draw()
            else: self.slice2_canvas.draw()

    def _on_slice_click(self, event, target):
        if event.inaxes is None or self.active_shape != "*": return

        ax = self.slice1_plotter.ax if target == "left" else self.slice2_plotter.ax
        if event.inaxes != ax: return

        x, y = int(round(event.xdata)), int(round(event.ydata))
        self._handle_pixel_click(x, y, target)

    def _open_colormap_dialog(self, target):
        dialog = ColorMapDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            cmap = dialog.selected_cmap
            if target == 'left':
                self.slice1_cmap = cmap
                self.slice1_plotter.set_cmap(cmap)
                if self.image_left is not None: self.slice1_canvas.draw()
            else:
                self.slice2_cmap = cmap
                self.slice2_plotter.set_cmap(cmap)
                if self.image_right is not None: self.slice2_canvas.draw()

    def _change_scale(self, target):
        scale_type = self.scale_combo_left.currentText() if target == 'left' else self.scale_combo_right.currentText()
        if target == 'left':
            self.scale_left = scale_type
            if self.image_left is not None: self.display_image('left')
        else:
            self.scale_right = scale_type
            if self.image_right is not None: self.display_image('right')

    def _prompt_and_clear_contours(self):
        """Prompt user to select which slice's contours to clear."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Clear Contours")
        msg.setText("Which slice's contours would you like to completely remove?")
        btn_left = msg.addButton("Left", QMessageBox.ActionRole)
        btn_right = msg.addButton("Right", QMessageBox.ActionRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        
        msg.exec_()
        
        if msg.clickedButton() == btn_left:
            self.clear_contours("left")
        elif msg.clickedButton() == btn_right:
            self.clear_contours("right")

    def clear_contours(self, target):
        """Removes all stacked contours from a specific slice."""
        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
        
        # Clear the stacked contour list
        if hasattr(plotter, 'contours'):
            for contour_set in plotter.contours:
                if hasattr(contour_set, 'collections'):
                    for c in contour_set.collections:
                        try: c.remove()
                        except: pass
                else:
                    try: contour_set.remove()
                    except: pass
            plotter.contours = []
            
        # Clear the legacy reference
        if hasattr(plotter, 'current_contour') and plotter.current_contour:
            if hasattr(plotter.current_contour, 'collections'):
                for c in plotter.current_contour.collections:
                    try: c.remove()
                    except: pass
            else:
                try: plotter.current_contour.remove()
                except: pass
            plotter.current_contour = None
            
        if target == "left":
            self.slice1_canvas.draw()
            self.last_contour_left_img = None
            self.last_contour_left_level = None
        else:
            self.slice2_canvas.draw()
            self.last_contour_right_img = None
            self.last_contour_right_level = None

    def _wcs_reproject(self, src_img, src_wcs, dst_wcs, dst_shape):
        """
        Reproject src_img from src_wcs onto dst_wcs pixel grid.
        Pure numpy/scipy — no external reproject package needed.
        Uses bicubic interpolation with NaN-aware masking.
        """
        from scipy.ndimage import map_coordinates

        ny, nx = dst_shape

        # Build full output pixel coordinate grid
        y_g, x_g = np.mgrid[0:ny, 0:nx].astype(np.float64)

        # Map each output pixel centre → sky → input pixel
        sky      = dst_wcs.pixel_to_world(x_g, y_g)
        x_src, y_src = src_wcs.world_to_pixel(sky)

        # Clamp out-of-range coords to -1 so they land outside source bounds
        x_src = np.where(np.isfinite(x_src), x_src, -1.0)
        y_src = np.where(np.isfinite(y_src), y_src, -1.0)

        coords = np.array([y_src.ravel(), x_src.ravel()])

        # Bicubic interpolation on NaN-safe source
        # (replace NaN with 0 so scipy doesn't propagate NaN through the kernel)
        src_finite = np.isfinite(src_img)
        src_safe   = np.where(src_finite, src_img, 0.0)

        result  = map_coordinates(src_safe,
                                coords, order=3, mode='constant', cval=0.0)
        # Interpolate the valid-pixel mask to detect NaN contamination
        weight  = map_coordinates(src_finite.astype(np.float64),
                                coords, order=1, mode='constant', cval=0.0)

        out    = result.reshape(dst_shape)
        weight = weight.reshape(dst_shape)
        x_src  = x_src.reshape(dst_shape)
        y_src  = y_src.reshape(dst_shape)

        # Mask output pixels that are:
        #  • outside source image bounds
        #  • or where ≥50% of the contributing source pixels were NaN
        src_ny, src_nx = src_img.shape
        valid = (
            (x_src >= 0) & (x_src <= src_nx - 1) &
            (y_src >= 0) & (y_src <= src_ny - 1) &
            (weight >= 0.5)
        )
        out[~valid] = np.nan

        return out

    def _prompt_and_superimpose(self):
        if self.image_left is None or self.image_right is None:
            QMessageBox.warning(self, "Missing Images",
                                "Load an image in BOTH panels before superimposing.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Superimpose / WCS Match")
        dialog.setMinimumWidth(380)
        layout = QVBoxLayout(dialog)

        # Background panel
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("Background (reference) image:"))
        bg_combo = QComboBox()
        bg_combo.addItems(["Left", "Right"])
        bg_layout.addWidget(bg_combo)
        layout.addLayout(bg_layout)

        # Overlay colormap
        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Overlay colormap:"))
        cmap_combo = QComboBox()
        cmap_combo.addItems(["viridis", "plasma", "hot", "cool",
                            "Blues", "Greens", "Reds", "magma"])
        cmap_layout.addWidget(cmap_combo)
        layout.addLayout(cmap_layout)

        # Alpha
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Overlay alpha (0.1 – 1.0):"))
        alpha_spin = QDoubleSpinBox()
        alpha_spin.setRange(0.1, 1.0)
        alpha_spin.setValue(0.5)
        alpha_spin.setSingleStep(0.1)
        alpha_layout.addWidget(alpha_spin)
        layout.addLayout(alpha_layout)

        layout.addWidget(QLabel(
            "<i>The overlay image is reprojected onto the background\n"
            "WCS grid and blended with the selected alpha.</i>"))

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Superimpose")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        if dialog.exec_() != QDialog.Accepted:
            return

        bg          = bg_combo.currentText().lower()      # 'left' or 'right'
        ov          = 'right' if bg == 'left' else 'left'
        alpha       = alpha_spin.value()
        overlay_cmap = cmap_combo.currentText()

        bg_img    = self.image_left   if bg == 'left' else self.image_right
        bg_header = self.header_left  if bg == 'left' else self.header_right
        bg_wcs    = self.wcs_left     if bg == 'left' else self.wcs_right
        ov_img    = self.image_left   if ov == 'left' else self.image_right
        ov_header = self.header_left  if ov == 'left' else self.header_right
        ov_wcs    = self.wcs_left     if ov == 'left' else self.wcs_right

        if bg_wcs is None or ov_wcs is None:
            QMessageBox.warning(self, "No WCS",
                                "Both images must have valid WCS to superimpose.")
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            reprojected = self._wcs_reproject(ov_img, ov_wcs, bg_wcs, bg_img.shape)

            # Plot onto background panel axes
            plotter = self.slice1_plotter if bg == 'left' else self.slice2_plotter
            canvas  = self.slice1_canvas  if bg == 'left' else self.slice2_canvas
            ax      = plotter.ax

            ov_finite = reprojected[np.isfinite(reprojected)]
            if ov_finite.size == 0:
                QMessageBox.warning(self, "No Overlap",
                                    "The two images have no overlapping sky area.")
                return

            from matplotlib.colors import Normalize
            ov_norm = Normalize(vmin=float(np.nanmin(ov_finite)),
                                vmax=float(np.nanmax(ov_finite)))
            ax.imshow(reprojected, origin='lower', cmap=overlay_cmap,
                    norm=ov_norm, alpha=alpha, aspect='auto',
                    interpolation='nearest')
            canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Superimpose Error",
                                f"Failed to superimpose:\n{str(e)}")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

        # Find peak pixel of background and reprojected overlay
        bg_peak   = np.unravel_index(np.nanargmax(bg_img),      bg_img.shape)
        ov_peak   = np.unravel_index(np.nanargmax(reprojected), reprojected.shape)

        bg_sky    = bg_wcs.pixel_to_world(bg_peak[1], bg_peak[0])
        ov_sky    = bg_wcs.pixel_to_world(ov_peak[1], ov_peak[0])

        sep = bg_sky.separation(ov_sky).to('arcsec')
        print(f"DEBUG superimpose: BG  peak → RA={bg_sky.ra.deg:.6f}, Dec={bg_sky.dec.deg:.6f}")
        print(f"DEBUG superimpose: OVL peak → RA={ov_sky.ra.deg:.6f}, Dec={ov_sky.dec.deg:.6f}")
        print(f"DEBUG superimpose: Peak separation = {sep:.4f}")