"""
Main GUI class for the Cube Explorer application
"""

import gc
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Suppress FITS warning messages about header formatting
warnings.filterwarnings('ignore', category=Warning, module='astropy.wcs.wcs')
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .data_loader import CubeData
from .visualization import SpectrumPlotter, SlicePlotter
from .region_selector import RegionSelector

class CubeExplorer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cube Explorer")
        self.cube_data = CubeData()
        self.selected_file = None
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
        """Clear all plots, selections, and stats when loading a new cube"""
        # Clear slice displays
        if hasattr(self, 'slice1_plotter'):
            self.slice1_plotter.ax.clear()
            # Handle colorbar removal safely
            try:
                if hasattr(self.slice1_plotter, 'cbar') and self.slice1_plotter.cbar is not None:
                    self.slice1_plotter.cbar.remove()
                self.slice1_plotter.cbar = None
            except Exception:
                pass
            self.slice1_canvas.draw()
            
        if hasattr(self, 'slice2_plotter'):
            self.slice2_plotter.ax.clear()
            # Handle colorbar removal safely
            try:
                if hasattr(self.slice2_plotter, 'cbar') and self.slice2_plotter.cbar is not None:
                    self.slice2_plotter.cbar.remove()
                self.slice2_plotter.cbar = None
            except Exception:
                pass
            self.slice2_canvas.draw()
            
        # Clear spectrum display
        if hasattr(self, 'spec_plotter') and hasattr(self.spec_plotter, 'ax'):
            self.spec_plotter.ax.clear()
            self.spec_canvas.draw()
            
        # Reset all selections and markers
        self.last_selection_left = None
        self.last_selection_right = None
        self.last_idx_left = 0
        self.last_idx_right = 0
        
        # Clear region selectors
        if hasattr(self, 'region_selector_left'):
            self.region_selector_left.active_shape = None
        if hasattr(self, 'region_selector_right'):
            self.region_selector_right.active_shape = None
            
        # Reset stats labels
        self.stats1_label.setText("Slice 1 Stats:\n-")
        self.stats2_label.setText("Slice 2 Stats:\n-")
        
        # Uncheck all shape buttons
        for side in ["left", "right"]:
            for btn in self.shape_buttons[side].values():
                btn.setChecked(False)
                
        # Uncheck slice selection buttons
        self.set_left_btn.setChecked(False)
        self.set_right_btn.setChecked(False)
        self.active_target = None
        
        # Reset instrument label
        self.instrument_label.setText("Instrument: -")
        
        # Force garbage collection
        gc.collect()

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
        
        self.instrument_label = QLabel("Instrument: -")
        self.instrument_label.setStyleSheet("font-weight: bold;")

        self.controls_layout = QHBoxLayout()
        self.controls_layout.addWidget(self.open_btn)
        self.controls_layout.addWidget(self.load_btn)
        self.controls_layout.addWidget(self.instrument_label)
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
        self.spec_toolbar = NavigationToolbar(self.spec_canvas, self)
        self.spec_canvas.mpl_connect("button_press_event", self._on_spec_click)

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
        self.set_left_btn.setCheckable(True)
        self.set_right_btn.setCheckable(True)
        self.set_left_btn.clicked.connect(lambda: self._select_target("left"))
        self.set_right_btn.clicked.connect(lambda: self._select_target("right"))

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
        for shape in ["*", "▭", "◯", "⬭", "✏"]:
            left_sidebar.addWidget(self.shape_buttons["left"][shape])
        left_sidebar.addWidget(self.copy_btn_l)
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
        for shape in ["*", "▭", "◯", "⬭", "✏"]:
            right_sidebar.addWidget(self.shape_buttons["right"][shape])
        right_sidebar.addWidget(self.copy_btn_r)
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
        slice_select_layout.addStretch()
        main_layout.addLayout(slice_select_layout)

        # Add spectrum at bottom
        main_layout.addWidget(self.spec_canvas)
        main_layout.addWidget(self.spec_toolbar)

        self.setLayout(main_layout)

    def _init_state(self):
        """Initialize state variables"""
        self.active_shape = None
        self.active_target = None
        self.last_idx_left = 0
        self.last_idx_right = 0
        self.last_selection_left = None
        self.last_selection_right = None

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

    def _display_slice(self, idx, target):
        """Display a wavelength/velocity slice"""
        image = self.cube_data.get_slice(idx)
        
        # Select appropriate plotter and labels
        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
        stats_label = self.stats1_label if target == "left" else self.stats2_label
        
        # Create title with wavelength/velocity information
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
            
        # Update the plot
        plotter.plot_slice(image, title)

        # Force immediate redraw so UI updates without needing a manual resize
        if target == "left":
            self.slice1_canvas.draw()
        else:
            self.slice2_canvas.draw()

        # Update stats
        mn, mx, mean = np.nanmin(image), np.nanmax(image), np.nanmean(image)
        stats_label.setText(
            f"Intensity:\n"
            f"min={mn:.3e}\n"
            f"max={mx:.3e}\n"
            f"mean={mean:.3e}\n"
        )

        # Store the index
        if target == "left":
            self.last_idx_left = idx
        else:
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
            total_sum = np.nansum(selected_area)  # Add sum
            std = np.nanstd(selected_area)  # Add standard deviation
            
            stats_label = self.stats1_label if target == "left" else self.stats2_label
            current_text = stats_label.text().split("\n")
            
            # Keep the slice stats if they exist (they start with "Intensity:")
            slice_stats = [line for line in current_text if line.startswith("Intensity:")]
            
            # Create region stats with sum and std
            region_stats = [
                "Selected Area Stats:",
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
            if self.last_selection_left["shape"] == "*":
                x, y = self.last_selection_left["coords"]
                self._handle_pixel_click(x, y, "left")
            else:
                xmin, xmax, ymin, ymax = self.last_selection_left["coords"]
                mask = self.last_selection_left["mask"]
                self._recreate_region(xmin, xmax, ymin, ymax, mask, "left")
                
        elif self.active_target == "right" and self.last_selection_right:
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