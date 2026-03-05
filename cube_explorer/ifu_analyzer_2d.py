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
from scipy.ndimage import rotate

# Suppress FITS warning messages about header formatting
warnings.filterwarnings('ignore', category=Warning, module='astropy.wcs.wcs')
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QMessageBox, QComboBox, QDialog, QDoubleSpinBox, QApplication, QTextEdit, QFrame
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
        self.scale_left = 'Linear'
        self.scale_right = 'Linear'
        self.slice1_cmap = 'magma'
        self.slice2_cmap = 'magma'
        
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

        self._init_plots()
        self._init_shape_controls()
        self._setup_main_layout()

    def _init_plots(self):
        self.slice1_fig = Figure(figsize=(5, 5), dpi=100)
        self.slice1_canvas = FigureCanvas(self.slice1_fig)
        self.slice1_plotter = SlicePlotter(self.slice1_fig)
        self.slice1_toolbar = NavigationToolbar(self.slice1_canvas, self)
        self.slice1_canvas.mpl_connect("button_press_event", lambda e: self._on_slice_click(e, "left"))
        
        self.slice2_fig = Figure(figsize=(5, 5), dpi=100)
        self.slice2_canvas = FigureCanvas(self.slice2_fig)
        self.slice2_plotter = SlicePlotter(self.slice2_fig)
        self.slice2_toolbar = NavigationToolbar(self.slice2_canvas, self)
        self.slice2_canvas.mpl_connect("button_press_event", lambda e: self._on_slice_click(e, "right"))

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
        
        self.save_contour_btn = QPushButton("Save Contour Mask")
        self.save_contour_btn.setStyleSheet(btn_style)
        self.save_contour_btn.clicked.connect(self._prompt_and_save_contour)

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
        """Dialog to fetch contour parameters and trigger the plot"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Plot Contour")
        layout = QVBoxLayout()
        
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Plot contour ON:"))
        target_combo = QComboBox()
        target_combo.addItems(["Left Slice", "Right Slice"])
        target_layout.addWidget(target_combo)
        layout.addLayout(target_layout)
        
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Generate contour FROM:"))
        source_combo = QComboBox()
        source_combo.addItems(["Same Slice (Self)", "Other Slice (Match WCS)"])
        source_layout.addWidget(source_combo)
        layout.addLayout(source_layout)
        
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Number of Levels:"))
        level_spin = QDoubleSpinBox()
        level_spin.setDecimals(0)
        level_spin.setRange(1, 20)
        level_spin.setValue(5)
        level_layout.addWidget(level_spin)
        layout.addLayout(level_layout)
        
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Contour Color:"))
        color_combo = QComboBox()
        color_combo.addItems(["white", "red", "green", "cyan", "magenta", "yellow", "black"])
        color_layout.addWidget(color_combo)
        layout.addLayout(color_layout)
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Plot")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            target = "left" if target_combo.currentIndex() == 0 else "right"
            source = "self" if source_combo.currentIndex() == 0 else "other"
            levels = int(level_spin.value())
            color = color_combo.currentText()
            self._apply_contour(target, source, levels, color)

    def _apply_contour(self, target, source, levels, color):
        target_plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter
        target_wcs = WCS(self.header_left) if target == 'left' else WCS(self.header_right)
        
        if source == 'self':
            source_img = self.image_left if target == 'left' else self.image_right
            source_wcs = target_wcs
        else:
            source_img = self.image_right if target == 'left' else self.image_left
            source_wcs = WCS(self.header_right) if target == 'left' else WCS(self.header_left)

        if source_img is None:
            QMessageBox.warning(self, "Missing Data", "The source image for the contour is not loaded.")
            return
            
        if source == 'other' and (self.header_left is None or self.header_right is None):
            QMessageBox.warning(self, "Missing Data", "Both images must be loaded to map WCS between them.")
            return

        ax = target_plotter.ax
        
        # Store contour source info for saving later
        if target == 'left':
            self.last_contour_left_img = source_img
            self.last_contour_left_wcs = source_wcs
        else:
            self.last_contour_right_img = source_img
            self.last_contour_right_wcs = source_wcs

        # Clear existing contours (compatible with Matplotlib 3.8+)
        if hasattr(target_plotter, 'current_contour') and target_plotter.current_contour:
            if hasattr(target_plotter.current_contour, 'collections'):
                for c in target_plotter.current_contour.collections:
                    try: c.remove()
                    except: pass
            else:
                try: target_plotter.current_contour.remove()
                except: pass
        
        # Ignore NaNs when determining contour boundaries
        valid_data = source_img[np.isfinite(source_img)]
        if valid_data.size == 0: return
        
        vmin, vmax = np.percentile(valid_data, [5, 95])
        level_vals = np.linspace(vmin, vmax, levels)
        
        kwargs = {'colors': color, 'levels': level_vals, 'alpha': 0.8, 'linewidths': 1.5}
        
        try:
            if source == 'other':
                transform = ax.get_transform(source_wcs)
                kwargs['transform'] = transform
                
            target_plotter.current_contour = ax.contour(source_img, **kwargs)
            
            if target == 'left':
                self.last_contour_left_level = level_vals[0]
            else:
                self.last_contour_right_level = level_vals[0]
                
            if target == 'left': self.slice1_canvas.draw()
            else: self.slice2_canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Contour Error", f"Could not map coordinate systems:\n{str(e)}")
            traceback.print_exc()

    def _prompt_and_save_contour(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Save Contour Mask")
        msg.setText("Which slice's active contour would you like to save as a FITS Mask?")
        btn_left = msg.addButton("Left", QMessageBox.ActionRole)
        btn_right = msg.addButton("Right", QMessageBox.ActionRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        
        msg.exec_()
        
        if msg.clickedButton() == btn_left:
            self._save_contour_mask("left")
        elif msg.clickedButton() == btn_right:
            self._save_contour_mask("right")

    def _save_contour_mask(self, target):
        img = getattr(self, f"last_contour_{target}_img", None)
        wcs = getattr(self, f"last_contour_{target}_wcs", None)
        level = getattr(self, f"last_contour_{target}_level", None)
        
        if img is None or level is None:
            QMessageBox.warning(self, "No Contour", f"No active contour found on the {target} slice.")
            return
            
        fname, _ = QFileDialog.getSaveFileName(
            self, f"Save Contour Mask ({target.capitalize()})", filter="FITS files (*.fits)"
        )
        if not fname: return
        if not fname.lower().endswith('.fits'): fname += '.fits'
        
        try:
            mask = np.where(img >= level, 1, 0).astype(np.int16)
            header = wcs.to_header() if wcs else fits.Header()
            
            hdu = fits.PrimaryHDU(data=mask, header=header)
            hdu.writeto(fname, overwrite=True)
            QMessageBox.information(self, "Success", f"Contour binary mask successfully saved to:\n{fname}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save contour mask:\n{str(e)}")

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
        contour_layout.addWidget(self.save_contour_btn)

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

            ax_rect   = [0.10, 0.10, 0.72, 0.82]
            cbar_rect = [0.84, 0.10, 0.04, 0.82]

            try:
                wcs_in = WCS(header)
                if wcs_in.has_celestial:
                    wcs2d = wcs_in.celestial if wcs_in.pixel_n_dim > 2 else wcs_in
                    plotter.ax = plotter.fig.add_axes(ax_rect, projection=wcs2d)
                else:
                    plotter.ax = plotter.fig.add_axes(ax_rect)
            except Exception:
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
        image  = self.image_left   if target == 'left' else self.image_right
        header = self.header_left  if target == 'left' else self.header_right
        if image is None:
            return

        plotter    = self.slice1_plotter if target == 'left' else self.slice2_plotter
        stats_label = self.stats1_label   if target == 'left' else self.stats2_label
        cmap  = getattr(self, 'slice1_cmap' if target == 'left' else 'slice2_cmap', 'magma')
        scale = self.scale_left if target == 'left' else self.scale_right

        ax = plotter.ax
        ax.cla()

        finite = image[np.isfinite(image)]
        vmin   = float(np.nanmin(finite)) if finite.size else 0.0
        vmax   = float(np.nanmax(finite)) if finite.size else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0

        from matplotlib.colors import LogNorm, PowerNorm, Normalize
        if scale == 'Log':
            vmin = max(vmin, 1e-10)
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif scale == 'Sqrt':
            vmin = max(vmin, 0.0)
            norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        elif scale == 'Power 2':
            norm = PowerNorm(gamma=2.0, vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        im = ax.imshow(image, origin='lower', cmap=cmap, norm=norm,
                    interpolation='nearest', aspect='equal')
        ax.set_title(title, fontsize=9, pad=4)

        ax.set_clip_on(True)
        im.set_clip_on(True)
        im.set_clip_path(ax.patch)

        unit_str = ""
        if header is not None:
            if 'BUNIT' in header:
                unit_str = f" [{header['BUNIT']}]"
                
            try:
                wcs = WCS(header)
                if wcs.has_celestial:
                    ra  = ax.coords[0]
                    dec = ax.coords[1]
                    ra.set_axislabel('Right Ascension', minpad=0.5)
                    dec.set_axislabel('Declination',    minpad=0.5)
                    ra.set_ticklabel(rotation=45, pad=5, exclude_overlapping=True)
                    dec.set_ticklabel(exclude_overlapping=True)
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    ax.grid(color='white', alpha=0.1, linestyle='dotted')
            except Exception:
                pass

        if not hasattr(plotter, 'cbar_ax') or plotter.cbar_ax is None:
            plotter.cbar_ax = plotter.fig.add_axes([0.84, 0.10, 0.04, 0.82])

        plotter.cbar_ax.cla()
        plotter.cbar_ax.set_visible(True)

        cbar = plotter.fig.colorbar(im, cax=plotter.cbar_ax)
        cbar.ax.tick_params(labelsize=7)
        plotter.cbar = cbar

        mn   = np.nanmin(image)
        mx   = np.nanmax(image)
        mean = np.nanmean(image)
        stats_label.setText(
            f"Image Stats{unit_str}\n"
            f"min={mn:.3e}\n"
            f"max={mx:.3e}\n"
            f"mean={mean:.3e}"
        )

        if target == 'left':
            self.slice1_canvas.draw()
        else:
            self.slice2_canvas.draw()

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