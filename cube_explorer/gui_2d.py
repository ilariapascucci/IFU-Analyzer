"""
Dual 2D FITS Viewer - the ``Ifu_analyzer_2d_window`` QWidget plus its batch
resolution-matching dialog. The reprojection/convolution math lives in
``analyzer_2d.py`` (GUI-free); this module is Qt only.
"""

import os
import gc
import zipfile
import tempfile
import warnings
import traceback
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats  # used by: _apply_contour
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.coordinates import SkyCoord  # used by: _apply_loaded_contour
import astropy.units as u  # used by: _apply_loaded_contour
from scipy.ndimage import rotate  # used by: rotate_image

# Suppress FITS warning messages about header formatting
warnings.filterwarnings('ignore', category=Warning, module='astropy.wcs.wcs')
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QMessageBox, QComboBox, QDialog, QDoubleSpinBox, QApplication, QTextEdit,
    QInputDialog, QLineEdit, QListWidget, QListWidgetItem, QRadioButton, QButtonGroup
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize, PowerNorm  # used by: display_image
from PyQt5.QtCore import Qt
from .visualization import SlicePlotter
from .colormap_dialog import ColorMapDialog
from .scale_dialog import ScaleDialog
from .region_selector import RegionSelector
from .gui import CollapsibleButtonGroup
from .gui_region_selection import RegionSelectionMixin
from . import region_io
from .utils import detect_instrument_from_header
from .fits_export import (
    build_2d_export_header, build_resample_header, build_coverage_extension, write_fits_product,
)
from .analyzer_2d import (
    _get_pixel_size_arcsec,
    _make_2d_wcs,
    _is_per_pixel_flux_unit,
    _wcs_footprints_overlap,
    _reproject_wcs_to_wcs,
)
from .naman_convolve_reproject import (
    gaussian_convolve_match as _naman_convolve_match,
    reproject_flux_maps as _naman_reproject_flux_maps,
)


def _header_line_wavelength_um(header):
    """Read LINEWAVE, or LAMMIN/LAMMAX's midpoint, from a header -- no
    prompting. LINEWAVE (rotational-diagram Save Maps exports) is exact.
    LAMMIN/LAMMAX (Gaussian Map exports) bound one line's own fit window, so
    their midpoint is a good proxy. WAVEMIN/WAVEMAX (Flux Map exports) can
    span several blended lines, so those are never used here -- a midpoint
    there could silently name the wrong line.

    Args:
        header (astropy.io.fits.Header): This image's FITS header, or None.

    Returns:
        wavelength_um (float or None): Wavelength in microns, or None if
            neither header key is present or usable.
    """
    if header is not None and 'LINEWAVE' in header:
        try:
            return float(header['LINEWAVE'])
        except (TypeError, ValueError):
            pass
    if header is not None and 'LAMMIN' in header and 'LAMMAX' in header:
        try:
            return (float(header['LAMMIN']) + float(header['LAMMAX'])) / 2.0
        except (TypeError, ValueError):
            pass
    return None


def _get_line_wavelength_um(header, filename, parent):
    """Read this image's line wavelength via _header_line_wavelength_um;
    prompt the user for it if no known header key is present.

    Args:
        header (astropy.io.fits.Header): This image's FITS header.
        filename (str): Basename, shown in the prompt if no key is found.
        parent (QWidget): Parent widget for the prompt dialog.

    Returns:
        wavelength_um (float or None): Wavelength in microns, or None if no
            header key was usable and the user cancelled the prompt.
    """
    wl = _header_line_wavelength_um(header)
    if wl is not None:
        return wl
    value, ok = QInputDialog.getDouble(
        parent, "Line Wavelength Needed",
        f"'{filename}' has no LINEWAVE header key.\n\n"
        "Naman's algorithm needs this image's line wavelength (microns) "
        "to compute its PSF FWHM. Enter it manually:",
        5.0, 0.1, 30.0, 6)
    return float(value) if ok else None


def _non_miri_files(infos):
    """Return (filename, instrument) for every input that isn't MIRI --
    Naman's PSF formula has no confirmed fit for any other instrument yet.

    Args:
        infos (list): (filename, header) pairs to check.

    Returns:
        bad (list): (filename, detected instrument or 'unknown') pairs.
    """
    bad = []
    for filename, header in infos:
        inst = detect_instrument_from_header(header) if header is not None else None
        if inst != "MIRI":
            bad.append((filename, inst or "unknown"))
    return bad


def _run_naman_convolve_pair(conv_img, conv_header, conv_wl, conv_pix,
                              ref_img, ref_header, ref_wl):
    """Convolve conv_img to ref_wl's PSF (Naman's FWHM(\")=0.033*lambda+0.106
    formula) and reproject it onto ref_header's own native grid, using the
    same unmodified gaussian_convolve_match/reproject_flux_maps ported for
    the rotational-diagram Naman comparison.

    Args:
        conv_img (numpy.ndarray): Image to convolve+reproject (shorter wavelength).
        conv_header (astropy.io.fits.Header): conv_img's FITS header.
        conv_wl (float): conv_img's line wavelength (microns).
        conv_pix (float): conv_img's pixel scale (arcsec/pixel).
        ref_img (numpy.ndarray): Reference image (longer wavelength).
        ref_header (astropy.io.fits.Header): Reference image's FITS header.
        ref_wl (float): Reference image's line wavelength (microns).

    Returns:
        result (dict): 'image' (conv_img reprojected onto ref's grid),
            'footprint' (per-pixel coverage fraction, same grid),
            'sigma_arcsec' (kernel sigma used).

    Raises:
        ValueError: If ref_wl's PSF is not larger than conv_wl's (Naman's
            formula requires the reference to have the longer wavelength).
    """
    fwhm_in = 0.033 * conv_wl + 0.106
    fwhm_target = 0.033 * ref_wl + 0.106
    if fwhm_target <= fwhm_in:
        raise ValueError(
            f"Naman's algorithm requires the reference wavelength "
            f"({ref_wl:.4f} um) to have a larger PSF than the convolved "
            f"image's ({conv_wl:.4f} um) -- got FWHM {fwhm_target:.4f}\" "
            f"<= {fwhm_in:.4f}\".")

    conv_header = conv_header.copy()
    conv_header["NAXIS"] = 2
    conv_header["NAXIS1"] = conv_img.shape[1]
    conv_header["NAXIS2"] = conv_img.shape[0]
    ref_header = ref_header.copy()
    ref_header["NAXIS"] = 2
    ref_header["NAXIS1"] = ref_img.shape[1]
    ref_header["NAXIS2"] = ref_img.shape[0]

    convolved = _naman_convolve_match([conv_img], [conv_wl], [ref_wl], [conv_pix])
    reprojected, footprints = _naman_reproject_flux_maps(convolved, [conv_header], ref_header)

    sigma_in = fwhm_in / gaussian_sigma_to_fwhm
    sigma_target = fwhm_target / gaussian_sigma_to_fwhm
    sigma_kernel_arcsec = (sigma_target ** 2 - sigma_in ** 2) ** 0.5

    return {"image": reprojected[0], "footprint": footprints[0],
            "sigma_arcsec": sigma_kernel_arcsec}


# BATCH MATCH RESOLUTION DIALOG
# ==============================================================================

class FileListItem(QWidget):
    """Custom widget to display a filename and a remove button in the QListWidget"""
    def __init__(self, filepath, remove_callback, parent=None):
        """Build the filename label and remove button for this list row.

        Args:
            filepath (str): Path shown in the label and tooltip.
            remove_callback (callable): Called with this widget when
                the remove button is clicked.
            parent (QWidget): Parent widget, or None for a top-level widget.

        Returns:
            None
        """
        super().__init__(parent)
        self.filepath = filepath
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.lbl = QLabel(os.path.basename(filepath))
        self.lbl.setToolTip(filepath)
        
        self.remove_btn = QPushButton("✖")
        self.remove_btn.setFixedSize(24, 24)
        self.remove_btn.setStyleSheet("""
            QPushButton { color: red; font-weight: bold; border: none; } 
            QPushButton:hover { background-color: #ffe0e0; border-radius: 12px; }
        """)
        self.remove_btn.clicked.connect(lambda: remove_callback(self))
        
        layout.addWidget(self.lbl)
        layout.addStretch()
        layout.addWidget(self.remove_btn)

class BatchMatchResolutionDialog(QDialog):
    """Standalone dialog for processing multiple FITS files into a ZIP"""
    def __init__(self, parent=None):
        """Build the batch resolution-matching dialog's file list and buttons.

        Args:
            parent (QWidget): Parent widget, or None for a top-level dialog.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle("Batch Match Resolution & Reproject")
        self.setMinimumSize(500, 400)
        self.filepaths = []
        
        layout = QVBoxLayout(self)
        
        info_lbl = QLabel("Select multiple 2D FITS files. The app will find the worst resolution and largest\n"
                          "sky footprint among them, convolve all images to match, and package them in a ZIP.")
        layout.addWidget(info_lbl)
        
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("+ Add FITS Files")
        self.add_btn.clicked.connect(self.add_files)
        
        self.process_naman_btn = QPushButton("Process & Download ZIP")
        self.process_naman_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px;")
        self.process_naman_btn.setToolTip(
            "Convolve every file to the longest-wavelength file's PSF\n"
            "(Naman's FWHM(\") = 0.033*wavelength_um + 0.106), reprojected\n"
            "onto that file's own native grid. MIRI only."
        )
        self.process_naman_btn.clicked.connect(self.process_and_zip_naman)

        btn_layout.addWidget(self.add_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.process_naman_btn)
        
        layout.addLayout(btn_layout)
        
    def add_files(self):
        """Browse for FITS files and add any new ones to the list.

        Args:
            None

        Returns:
            None
        """
        files, _ = QFileDialog.getOpenFileNames(self, "Select 2D FITS Files", filter="FITS files (*.fits *.fz *.fit)")
        for f in files:
            if f not in self.filepaths:
                self.filepaths.append(f)
                
                item = QListWidgetItem(self.list_widget)
                item_widget = FileListItem(f, self.remove_file)
                item.setSizeHint(item_widget.sizeHint())
                self.list_widget.addItem(item)
                self.list_widget.setItemWidget(item, item_widget)
                
    def remove_file(self, item_widget):
        """Remove one file from the list and its filepaths.

        Args:
            item_widget (FileListItem): Row widget whose file to remove.

        Returns:
            None
        """
        self.filepaths.remove(item_widget.filepath)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if self.list_widget.itemWidget(item) == item_widget:
                self.list_widget.takeItem(i)
                break
                
    def process_and_zip_naman(self):
        """"Process & Download ZIP" button: convolve every file to the
        longest-wavelength file's PSF (FWHM(") = 0.033*wavelength_um + 0.106,
        MIRI only), reprojected onto that file's own native grid.

        Args:
            None

        Returns:
            None
        """
        if len(self.filepaths) < 2:
            QMessageBox.warning(
                self, "Need more files",
                "Please select at least 2 FITS files for batch processing.")
            return

        zip_path, _ = QFileDialog.getSaveFileName(
            self, "Save ZIP File", "Convolution_Images_Naman.zip", "ZIP Files (*.zip)")
        if not zip_path:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # ---- 1. Load every input ------------------------------------
            images_data = []
            for fp in self.filepaths:
                with fits.open(fp) as hdul:
                    data, header = None, None
                    for hdu in hdul:
                        if hdu.data is not None and getattr(hdu.data, 'size', 0) > 0:
                            sq = np.squeeze(hdu.data)
                            if sq.ndim == 2:
                                data = sq.astype(float)
                                header = hdu.header.copy()
                                break
                    if data is None:
                        raise ValueError(
                            f"No 2D image data in {os.path.basename(fp)}")
                    pix = _get_pixel_size_arcsec(header)
                    if pix is None:
                        raise ValueError(
                            f"Could not find pixel scale in "
                            f"{os.path.basename(fp)}")
                    wcs = _make_2d_wcs(header)
                    if wcs is None:
                        raise ValueError(
                            f"Could not build WCS for {os.path.basename(fp)}")
                    images_data.append({
                        'filename': os.path.basename(fp),
                        'data': data, 'header': header,
                        'pix_scale': pix, 'wcs': wcs, 'shape': data.shape,
                    })
                    disjoint_pairs = []
                    for i in range(len(images_data)):
                        for j in range(i + 1, len(images_data)):
                            a, b = images_data[i], images_data[j]
                            if not _wcs_footprints_overlap(
                                    a['wcs'], a['shape'], b['wcs'], b['shape']):
                                disjoint_pairs.append((a['filename'], b['filename']))

                    if disjoint_pairs:
                        QApplication.restoreOverrideCursor()
                        msg_lines = [
                            "Some of the selected images cover completely "
                            "different regions of the sky:\n"
                        ]
                        for fa, fb in disjoint_pairs[:10]:
                            msg_lines.append(f"   • {fa}  ↔  {fb}")
                        if len(disjoint_pairs) > 10:
                            msg_lines.append(
                                f"   …and {len(disjoint_pairs) - 10} more pair(s)")
                        msg_lines.append(
                            "\nBatch Convolution requires every input to share "
                            "sky coverage with the others, so they can be "
                            "matched onto a common WCS grid.\n\n"
                            "Please remove the off-field file(s) and try again.")
                        QMessageBox.warning(
                            self, "No Sky Overlap",
                            "\n".join(msg_lines))
                        return

            # ---- 2. MIRI-only check: no confirmed PSF formula otherwise ---
            bad = _non_miri_files([(info['filename'], info['header']) for info in images_data])
            if bad:
                QApplication.restoreOverrideCursor()
                detail = "\n".join(f"  - {fn}: {inst}" for fn, inst in bad)
                QMessageBox.warning(
                    self, "Unsupported Instrument",
                    "Batch currently supports MIRI only (no confirmed PSF "
                    f"formula for other instruments yet):\n{detail}")
                return

            # ---- 3. BUNIT check: Naman's reproject_flux_maps assumes -----
            #         flux-per-pixel input for every file.
            for info in images_data:
                if not _is_per_pixel_flux_unit(info['header'].get('BUNIT', '')):
                    QApplication.restoreOverrideCursor()
                    QMessageBox.warning(
                        self, "Unsupported BUNIT for Naman",
                        f"'{info['filename']}' has BUNIT "
                        f"'{info['header'].get('BUNIT', '')}', not a "
                        "flux-per-pixel unit.\n\n"
                        "Naman's algorithm assumes flux-per-pixel input for "
                        "every file -- use the regular Batch button for "
                        "surface-brightness images.")
                    return

            # ---- 4. Wavelength per file (LINEWAVE header, else prompt) ---
            QApplication.restoreOverrideCursor()
            for info in images_data:
                wl = _get_line_wavelength_um(info['header'], info['filename'], self)
                if wl is None:
                    return
                info['wavelength_um'] = wl
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # ---- 5. Longest wavelength = reference (largest Naman PSF) ---
            ref_info = max(images_data, key=lambda info: info['wavelength_um'])
            ref_header = ref_info['header'].copy()
            ref_header['NAXIS'] = 2
            ref_header['NAXIS1'] = ref_info['shape'][1]
            ref_header['NAXIS2'] = ref_info['shape'][0]

            # ---- 6. Convolve + reproject every other file onto the -------
            #         reference's own native grid.
            flux_audit = []
            with tempfile.TemporaryDirectory() as tmpdir:
                for info in images_data:
                    bunit = info['header'].get('BUNIT', '')
                    if info is ref_info:
                        result_img = np.asarray(info['data'], dtype=float).copy()
                        result_img[result_img == 0.0] = np.nan
                        coverage = np.where(np.isfinite(result_img), 1.0, 0.0)
                        sigma_arcsec = 0.0
                    else:
                        naman_out = _run_naman_convolve_pair(
                            info['data'], info['header'], info['wavelength_um'], info['pix_scale'],
                            ref_info['data'], ref_header, ref_info['wavelength_um'])
                        result_img = naman_out['image']
                        coverage = naman_out['footprint']
                        sigma_arcsec = naman_out['sigma_arcsec']

                    if not np.any(np.isfinite(result_img)):
                        QApplication.restoreOverrideCursor()
                        QMessageBox.warning(
                            self, "Empty Result",
                            f"Naman Convolve produced no valid pixels for "
                            f"{info['filename']}.")
                        return

                    out_header = ref_header.copy()
                    if bunit:
                        out_header['BUNIT'] = bunit
                    out_header['HISTORY'] = (
                        f"Naman Batch Convolution: sigma={sigma_arcsec:.4f}\"; "
                        f"reprojected onto {ref_info['filename']}'s native grid "
                        f"(ref wavelength {ref_info['wavelength_um']:.4f} um)"
                    )

                    out_name = f"matched_naman_{info['filename']}"
                    out_filepath = os.path.join(tmpdir, out_name)
                    cov_header = fits.Header()
                    cov_header['EXTNAME'] = 'COVERAGE'
                    cov_header['BUNIT'] = 'fraction'
                    cov_header['COMMENT'] = (
                        'Per-pixel input/output overlap fraction (0..1).')
                    fits.HDUList([
                        fits.PrimaryHDU(data=result_img, header=out_header),
                        fits.ImageHDU(data=np.asarray(coverage, dtype=np.float32),
                                      header=cov_header),
                    ]).writeto(out_filepath, overwrite=True)

                    in_total = float(np.nansum(info['data']))
                    out_total = float(np.nansum(result_img[np.isfinite(result_img)]))
                    flux_audit.append((info['filename'], in_total, out_total))

                # ---- 7. Pack everything into the ZIP ---------------------
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(tmpdir):
                        for file in files:
                            zipf.write(os.path.join(root, file), arcname=file)

            QApplication.restoreOverrideCursor()

            audit_lines = []
            for name, fi, fo in flux_audit:
                ratio = (fo / fi) if fi != 0 else float('nan')
                audit_lines.append(f"  {name}: ratio = {ratio:.6f}")

            final_msg = (
                f"Naman Batch Convolution complete!\n\n"
                f"All {len(images_data)} images reprojected onto "
                f"{ref_info['filename']}'s native grid "
                f"({ref_info['wavelength_um']:.4f} um reference, "
                f"{ref_info['shape'][1]} × {ref_info['shape'][0]} px):\n\n"
                f"Flux audit (output_total / input_total):\n"
                + "\n".join(audit_lines)
                + f"\n\nSaved to:\n{zip_path}")
            QMessageBox.information(self, "Success", final_msg)
            self.accept()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Processing Error", f"An error occurred:\n{e}")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

class Ifu_analyzer_2d_window(QWidget, RegionSelectionMixin):
    def __init__(self, parent=None):
        """Build the 2D dual-viewer window and initialize its state.

        Args:
            parent (QWidget): Parent widget, or None for a top-level window.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle("2D FITS Dual Viewer")

        self.setWindowFlags(Qt.Window)
        self.setMinimumSize(1200, 600)

        self.image_left = None
        self.header_left = None
        self.image_right = None
        self.header_right = None
        # Source file basename per panel, for Save Region's descriptive text.
        self.filename_left = None
        self.filename_right = None
        # Optional per-pixel coverage fraction (0-1) for reprojected images,
        # saved as a "COVERAGE" HDU; flux sums value*pixel_area*coverage.
        self.coverage_left = None
        self.coverage_right = None

        self._init_state()
        self._init_ui()

    def closeEvent(self, event):
        """Drop this window's collapsible groups and free its images on close.

        Args:
            event (QCloseEvent): Close event Qt sends to the widget.

        Returns:
            None
        """
        # Drop this window's groups from CollapsibleButtonGroup's class-level
        # registry so cross-window accordion logic never touches a deleted widget.
        for _g in (getattr(self, 'contour_group', None),
                   getattr(self, 'convolution_group', None)):
            try:
                if _g in CollapsibleButtonGroup._all_groups:
                    CollapsibleButtonGroup._all_groups.remove(_g)
            except Exception:
                pass

        self.image_left = None
        self.image_right = None
        self.base_image_left = None
        self.base_image_right = None
        gc.collect()
        event.accept()

    def _init_state(self):
        """Initialize state variables.

        Args:
            None

        Returns:
            None
        """
        self.active_shape = None
        self.active_target = None
        self.last_selection_left = None
        self.last_selection_right = None
        # Region-interaction state machine per side:
        # idle -> drawing -> placed -> editing (see _set_region_state).
        self.region_state = {"left": "idle", "right": "idle"}
        self.last_contour_right_wcs = None
        self.last_contour_right_level = None
        self.scale_left = 'Linear'
        self.scale_right = 'Linear'
        self.slice1_vmin = self.slice1_vmax = None  # None = auto
        self.slice2_vmin = self.slice2_vmax = None
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
        """Build every UI section in order.

        Args:
            None

        Returns:
            None
        """
        # Stats labels - WordWrap enabled to prevent text cutoff
        self.stats1_label = QLabel("Intensity:\n-")
        self.stats1_label.setMinimumWidth(150)
        self.stats1_label.setWordWrap(True)
        self.stats1_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        
        self.stats2_label = QLabel("Intensity:\n-")
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
        """Initialize matplotlib figures and canvases.

        Args:
            None

        Returns:
            None
        """
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

    def _open_colormap_dialog(self, target):
        """Opens the colormap dialog and applies the selection to the chosen panel.

        Args:
            target (str): Panel to recolor, 'left' or 'right'.

        Returns:
            None
        """
        # Open the custom dialog imported at the top of the file
        dialog = ColorMapDialog(self)
        
        if dialog.exec_():
            # Safely extract the chosen colormap string from the dialog
            # (Checking the most common variable names used in custom PyQt dialogs)
            selected_cmap = getattr(dialog, 'selected_cmap', None)
            if selected_cmap is None and hasattr(dialog, 'get_colormap'):
                selected_cmap = dialog.get_colormap()
            if selected_cmap is None:
                selected_cmap = getattr(dialog, 'cmap', 'magma') # fallback

            if selected_cmap:
                # Apply the colormap to the correct panel variable
                if target == 'left':
                    self.slice1_cmap = selected_cmap
                else:
                    self.slice2_cmap = selected_cmap
                    
                # Force the image to redraw with the new colors (if an image is loaded)
                if target == 'left' and self.image_left is not None:
                    self.display_image('left')
                elif target == 'right' and self.image_right is not None:
                    self.display_image('right')

    def copy_colormap(self, source, target):
        """Copy the source panel's colormap onto the target panel (matches 3D).

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
            if self.image_left is not None:
                self.display_image('left')
        else:
            self.slice2_cmap = src_cmap
            if self.image_right is not None:
                self.display_image('right')

    def _open_scale_dialog(self, target):
        """Open the scale-type + vmin/vmax range dialog for a panel (matches 3D).

        Args:
            target (str): Panel to configure, 'left' or 'right'.

        Returns:
            None
        """
        image = self.image_left if target == 'left' else self.image_right
        if image is None:
            QMessageBox.warning(self, "No Image Loaded", "Load an image into this panel first.")
            return

        finite = image[np.isfinite(image)]
        auto_vmin = float(np.nanmin(finite)) if finite.size else 0.0
        auto_vmax = float(np.nanmax(finite)) if finite.size else 1.0

        current_scale = self.scale_left if target == 'left' else self.scale_right
        current_vmin = self.slice1_vmin if target == 'left' else self.slice2_vmin
        current_vmax = self.slice1_vmax if target == 'left' else self.slice2_vmax

        dialog = ScaleDialog(current_scale, auto_vmin, auto_vmax,
                             vmin=current_vmin, vmax=current_vmax, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return

        scale_type = dialog.result_scale
        vmin, vmax = dialog.result_vmin, dialog.result_vmax
        range_desc = "Auto" if vmin is None and vmax is None else f"{vmin:.3e} to {vmax:.3e}"
        btn = self.scale_btn_left if target == 'left' else self.scale_btn_right
        btn.setToolTip(f"Intensity scaling for {target} slice ({scale_type}, {range_desc})")

        if target == 'left':
            self.scale_left = scale_type
            self.slice1_vmin, self.slice1_vmax = vmin, vmax
        else:
            self.scale_right = scale_type
            self.slice2_vmin, self.slice2_vmax = vmin, vmax

        self.display_image(target)

    def copy_scale(self, source, target):
        """Copy one panel's scale type and vmin/vmax range onto the other (matches 3D).

        Args:
            source (str): Panel to copy from, 'left' or 'right'.
            target (str): Panel to copy to, 'left' or 'right'.

        Returns:
            None
        """
        if source == 'left':
            scale_type = self.scale_left
            vmin, vmax = self.slice1_vmin, self.slice1_vmax
        else:
            scale_type = self.scale_right
            vmin, vmax = self.slice2_vmin, self.slice2_vmax

        range_desc = "Auto" if vmin is None and vmax is None else f"{vmin:.3e} to {vmax:.3e}"
        btn = self.scale_btn_left if target == 'left' else self.scale_btn_right
        btn.setToolTip(f"Intensity scaling for {target} slice ({scale_type}, {range_desc})")

        if target == 'left':
            self.scale_left = scale_type
            self.slice1_vmin, self.slice1_vmax = vmin, vmax
            if self.image_left is not None:
                self.display_image('left')
        else:
            self.scale_right = scale_type
            self.slice2_vmin, self.slice2_vmax = vmin, vmax
            if self.image_right is not None:
                self.display_image('right')

    def _init_shape_controls(self):
        """Initialize shape selection buttons.

        Args:
            None

        Returns:
            None
        """
        def make_shape_button(symbol, target, callback=None):
            """Build one square icon button wired to a shape or callback.

            Args:
                symbol (str): Unicode glyph shown as the button label.
                target (str): Side the button applies to ('left' or 'right').
                callback (callable): Click handler; defaults to _set_shape.

            Returns:
                btn (QPushButton): The configured button.
            """
            btn = QPushButton(symbol)
            btn.setCheckable(True)
            btn.setFixedSize(35, 35)
            btn.setStyleSheet("QPushButton { font-size: 18pt; }")
            if callback is None:
                btn.clicked.connect(lambda _, s=symbol: self._set_shape(s, target=target))
            else:
                btn.clicked.connect(callback)
            return btn

        shapes = ["*", "▭", "○", "⬭", "✏"]
        self.shape_buttons = {
            "left": {shape: make_shape_button(shape, "left") for shape in shapes},
            "right": {shape: make_shape_button(shape, "right") for shape in shapes}
        }

        self.color_btn_left = QPushButton("🌈")
        self.color_btn_left.setFixedSize(35, 35)
        self.color_btn_left.clicked.connect(lambda: self._open_colormap_dialog('left'))

        self.color_btn_right = QPushButton("🌈")
        self.color_btn_right.setFixedSize(35, 35)
        self.color_btn_right.clicked.connect(lambda: self._open_colormap_dialog('right'))

        # Scale selection buttons - square icon buttons that open the scale/
        # range dialog, matching the 3D window's own scale buttons.
        def make_scale_button(target):
            """Build a square button that opens the scale/range dialog.

            Args:
                target (str): Panel the button applies to ('left' or 'right').

            Returns:
                btn (QPushButton): The configured button.
            """
            btn = QPushButton("🎚")
            btn.setFixedSize(35, 35)
            btn.setStyleSheet("QPushButton { font-size: 18pt; }")
            btn.clicked.connect(lambda: self._open_scale_dialog(target))
            btn.setToolTip(f"Intensity scaling for {target} slice (Linear, Auto range)")
            return btn

        self.scale_btn_left = make_scale_button('left')
        self.scale_btn_right = make_scale_button('right')

        # Copy button sits on its own side's sidebar and pulls the OTHER
        # panel's selection onto this one (matches the 3D window convention).
        self.copy_btn_l = QPushButton("⎘")
        self.copy_btn_l.setFixedSize(35, 35)
        self.copy_btn_l.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.copy_btn_l.setToolTip("Copy selection from right slice to left")
        self.copy_btn_l.clicked.connect(lambda: self.copy_selection(source="right", target="left"))

        self.copy_btn_r = QPushButton("⎘")
        self.copy_btn_r.setFixedSize(35, 35)
        self.copy_btn_r.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.copy_btn_r.setToolTip("Copy selection from left slice to right")
        self.copy_btn_r.clicked.connect(lambda: self.copy_selection(source="left", target="right"))

        # Copy the OTHER panel's colormap onto this one (matches the 3D window).
        self.copy_color_btn_l = QPushButton("⎘🌈")
        self.copy_color_btn_l.setFixedSize(35, 35)
        self.copy_color_btn_l.setToolTip("Copy right slice colormap to left slice")
        self.copy_color_btn_l.clicked.connect(lambda: self.copy_colormap('right', 'left'))

        self.copy_color_btn_r = QPushButton("⎘🌈")
        self.copy_color_btn_r.setFixedSize(35, 35)
        self.copy_color_btn_r.setToolTip("Copy left slice colormap to right slice")
        self.copy_color_btn_r.clicked.connect(lambda: self.copy_colormap('left', 'right'))

        # Copy the OTHER panel's scale type and vmin/vmax range onto this one
        # (matches the 3D window).
        self.copy_scale_btn_l = QPushButton("⎘🎚")
        self.copy_scale_btn_l.setFixedSize(35, 35)
        self.copy_scale_btn_l.setToolTip("Copy right slice scale/range to left slice")
        self.copy_scale_btn_l.clicked.connect(lambda: self.copy_scale('right', 'left'))

        self.copy_scale_btn_r = QPushButton("⎘🎚")
        self.copy_scale_btn_r.setFixedSize(35, 35)
        self.copy_scale_btn_r.setToolTip("Copy left slice scale/range to right slice")
        self.copy_scale_btn_r.clicked.connect(lambda: self.copy_scale('left', 'right'))

        self.erase_btn_l = QPushButton("✖")
        self.erase_btn_l.setFixedSize(35, 35)
        self.erase_btn_l.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.erase_btn_l.clicked.connect(lambda: self.clear_region("left"))

        self.erase_btn_r = QPushButton("✖")
        self.erase_btn_r.setFixedSize(35, 35)
        self.erase_btn_r.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.erase_btn_r.clicked.connect(lambda: self.clear_region("right"))

        # Edit Shape: toggles drag/resize handles on a placed selection.
        # Enabled only in the "placed" state (and not for pixel selections).
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

        # Region save/import actions live in the "Region" collapsible group
        # (_setup_main_layout) -- each asks which panel (left/right) to use,
        # same as the 3D window's own Region group.
        self.region_save_slice_masked_btn = QPushButton("Save Slice (Masked)")
        self.region_save_slice_masked_btn.setToolTip(
            "Save one panel's image as a 2D FITS, with everything outside its region set to NaN")
        self.region_save_slice_masked_btn.clicked.connect(self._region_save_slice_masked)
        self.region_save_slice_masked_btn.setEnabled(False)

        self.region_save_wcs_btn = QPushButton("Save Region")
        self.region_save_wcs_btn.setToolTip(
            "Save one panel's region as RA/Dec sky coordinates, so it can be reconstructed on a different image")
        self.region_save_wcs_btn.clicked.connect(self._region_save_wcs_text)
        self.region_save_wcs_btn.setEnabled(False)

        self.region_import_wcs_btn = QPushButton("Import Region")
        self.region_import_wcs_btn.setToolTip(
            "Load a region previously saved as WCS text onto one panel, at the same sky position")
        self.region_import_wcs_btn.clicked.connect(self._region_import_wcs_text)
        self.region_import_wcs_btn.setEnabled(False)

        # Bottom panel controls: plain default styling, matching the 3D
        # window and the Open/Header/Save buttons above.

        # ----- ROTATION CONTROLS (per-panel; no "which slice?" prompt) -----
        self.rot_ccw_radio = QRadioButton("Counter-Clockwise")
        self.rot_cw_radio = QRadioButton("Clockwise")
        self.rot_ccw_radio.setChecked(True)
        self.rot_dir_group = QButtonGroup(self)
        self.rot_dir_group.addButton(self.rot_ccw_radio)
        self.rot_dir_group.addButton(self.rot_cw_radio)

        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(0.0, 360.0)
        self.angle_spin.setValue(90.0)
        self.angle_spin.setSuffix(" °")
        self.angle_spin.setMaximumWidth(90)

        self.rotate_left_btn = QPushButton("Rotate Left")
        self.rotate_left_btn.clicked.connect(lambda: self.rotate_image("left"))

        self.rotate_right_btn = QPushButton("Rotate Right")
        self.rotate_right_btn.clicked.connect(lambda: self.rotate_image("right"))

        # ----- CONTOUR CONTROLS (grouped under the "Contour" collapsible) -----
        self.contour_btn = QPushButton("Plot")
        self.contour_btn.clicked.connect(self._prompt_and_contour)

        self.superimpose_btn = QPushButton("Superimpose")
        self.superimpose_btn.clicked.connect(self._prompt_and_superimpose)

        self.save_contour_btn = QPushButton("Save")
        self.save_contour_btn.setToolTip("Save the active contour's coordinates to a text/CSV file")
        self.save_contour_btn.clicked.connect(self._prompt_and_save_contour)

        self.load_contour_btn = QPushButton("Load")
        self.load_contour_btn.setToolTip("Overlay a contour saved from this or another image")
        self.load_contour_btn.clicked.connect(self._prompt_and_load_contour)

        self.clear_contour_btn = QPushButton("Clear")
        self.clear_contour_btn.setToolTip("Remove all contours from a panel")
        self.clear_contour_btn.clicked.connect(self._prompt_and_clear_contours)

        # ----- CONVOLUTION CONTROLS (grouped under the "Convolution" collapsible) -----
        self.batch_match_btn = QPushButton("Batch")
        self.batch_match_btn.setToolTip(
            "Select multiple 2D FITS files. All are convolved to the\n"
            "longest-wavelength file's PSF and reprojected onto its native\n"
            "grid. Result is downloaded as a ZIP. MIRI only."
        )
        self.batch_match_btn.clicked.connect(self._open_batch_match_dialog)

        self.match_res_naman_btn = QPushButton("Pair")
        self.match_res_naman_btn.setToolTip(
            "Convolve the shorter-wavelength panel to match the longer-\n"
            "wavelength panel's PSF (FWHM(\") = 0.033*wavelength_um + 0.106),\n"
            "reprojected onto its own native grid. MIRI only."
        )
        self.match_res_naman_btn.clicked.connect(self.prompt_and_match_resolution_naman)

    def copy_selection(self, source, target):
        """Copy a region selection onto the other panel, WCS-aware: corners
        reproject through both panels' WCS (rectangles become polygons
        under rotation), else falls back to a plain pixel-index copy.

        Args:
            source (str): Panel to copy from, 'left' or 'right'.
            target (str): Panel to copy to, 'left' or 'right'.

        Returns:
            None
        """
        src_sel = self.last_selection_right if source == "right" else self.last_selection_left
        if not src_sel:
            return

        src_wcs = self.wcs_left if source == "left" else self.wcs_right
        tgt_wcs = self.wcs_left if target == "left" else self.wcs_right
        tgt_img = self.image_left if target == "left" else self.image_right
        selector = self.region_selector_left if target == "left" else self.region_selector_right
        shape = src_sel["shape"]

        wcs_ok = (
            tgt_img is not None
            and src_wcs is not None and getattr(src_wcs, "has_celestial", False)
            and tgt_wcs is not None and getattr(tgt_wcs, "has_celestial", False)
        )

        def _to_target_px(xs, ys):
            """Reproject source-panel pixel coordinates onto the target panel.

            Args:
                xs (numpy.ndarray): Source-panel x pixel coordinates.
                ys (numpy.ndarray): Source-panel y pixel coordinates.

            Returns:
                coords (tuple): (tx, ty) target-panel pixel coordinates.
            """
            sw = src_wcs.celestial if src_wcs.pixel_n_dim > 2 else src_wcs
            tw = tgt_wcs.celestial if tgt_wcs.pixel_n_dim > 2 else tgt_wcs
            sky = sw.pixel_to_world(np.asarray(xs, float), np.asarray(ys, float))
            tx, ty = tw.world_to_pixel(sky)
            return np.asarray(tx, float), np.asarray(ty, float)

        # ---- pixel ("*") ----
        if shape == "*":
            x, y = src_sel["coords"]
            if wcs_ok:
                try:
                    tx, ty = _to_target_px([x], [y])
                    if np.isfinite(tx[0]) and np.isfinite(ty[0]):
                        x, y = float(tx[0]), float(ty[0])
                except Exception:
                    pass
            self._handle_pixel_click(int(round(x)), int(round(y)), target)
            return

        # ---- lasso (freehand shape) ----
        if shape == "✏":
            patch = src_sel.get("patch")
            if patch is None or not hasattr(patch, "lasso_verts"):
                return
            verts = np.asarray(patch.lasso_verts, dtype=float)
            if wcs_ok:
                try:
                    tx, ty = _to_target_px(verts[:, 0], verts[:, 1])
                    if np.all(np.isfinite(tx)) and np.all(np.isfinite(ty)):
                        ny, nx = tgt_img.shape
                        verts = np.column_stack(
                            [np.clip(tx, 0, nx - 1), np.clip(ty, 0, ny - 1)])
                except Exception:
                    pass
            selector.active_shape = "✏"
            try:
                selector.on_lasso_select(verts)
            except Exception:
                pass
            return

        # ---- rectangle / circle / ellipse ----
        xmin, xmax, ymin, ymax = (float(c) for c in src_sel["coords"])
        if wcs_ok:
            try:
                cx = [xmin, xmax, xmax, xmin]
                cy = [ymin, ymin, ymax, ymax]
                tx, ty = _to_target_px(cx, cy)
                if not (np.all(np.isfinite(tx)) and np.all(np.isfinite(ty))):
                    raise ValueError("non-finite reprojected corners")
                ny, nx = tgt_img.shape
                axis_aligned = (
                    abs(tx[0] - tx[3]) < 0.5 and abs(tx[1] - tx[2]) < 0.5 and
                    abs(ty[0] - ty[1]) < 0.5 and abs(ty[2] - ty[3]) < 0.5
                )
                if shape == "▭" and not axis_aligned:
                    verts = np.column_stack(
                        [np.clip(tx, 0, nx - 1), np.clip(ty, 0, ny - 1)])
                    selector.active_shape = "✏"
                    try:
                        selector.on_lasso_select(verts)
                    except Exception:
                        pass
                    return
                # round (not truncate) so a copy between identical/near-identical
                # grids is exact instead of drifting a pixel on the TAN round-trip
                xmin = max(0, min(nx - 1, int(round(float(np.min(tx))))))
                xmax = max(0, min(nx - 1, int(round(float(np.max(tx))))))
                ymin = max(0, min(ny - 1, int(round(float(np.min(ty))))))
                ymax = max(0, min(ny - 1, int(round(float(np.max(ty))))))
                if xmax - xmin < 1 or ymax - ymin < 1:
                    QMessageBox.information(
                        self, "Copy Region",
                        "The source selection does not overlap the other "
                        "panel's field of view.")
                    return
            except Exception:
                xmin, xmax, ymin, ymax = (float(c) for c in src_sel["coords"])

        selector.active_shape = shape
        try:
            selector.create_region((int(round(xmin)), int(round(xmax)),
                                    int(round(ymin)), int(round(ymax))))
        except Exception:
            pass

    def _open_batch_match_dialog(self):
        """Open the batch resolution-matching dialog.

        Args:
            None

        Returns:
            None
        """
        dialog = BatchMatchResolutionDialog(self)
        dialog.exec_()

    def _prompt_and_contour(self):
        """Ask for target/source/sigma-levels/color, then plot a WCS contour.

        Args:
            None

        Returns:
            None
        """
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

        # 3. Background Std - LOCAL variable, not self.sigma_input
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Background Std (blank = Astropy auto):"))
        sigma_input = QLineEdit()          # ← local, not self.
        sigma_input.setPlaceholderText("Auto")
        sigma_input.setMinimumWidth(120)
        sigma_layout.addWidget(sigma_input)
        layout.addLayout(sigma_layout)

        # 4. Sigma levels - comma-separated multipliers
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
        """Compute background-relative sigma-level contours and plot them.

        Args:
            target (str): Panel to draw the contour on, 'left' or 'right'.
            source (str): 'self' (own image) or 'other' (the other panel).
            sigma_str (str): Background std typed by the user, or '' for
                the automatic sigma-clipped estimate.
            levels_str (str): Comma-separated sigma multipliers to contour.
            color (str): Matplotlib color name for the contour lines.

        Returns:
            None
        """
        target_plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter

        def get_2d_wcs(header):
            """Build a strictly-2D WCS from a FITS header, dropping extra axes.

            Args:
                header (astropy.io.fits.Header): FITS header, or None.

            Returns:
                wcs (astropy.wcs.WCS): The 2D WCS, or None if header is None.
            """
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

        # --- Parse sigma levels ---
        try:
            sigma_levels = [float(x.strip()) for x in levels_str.split(',') if x.strip()]
        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                                "Levels must be comma-separated numbers (e.g. 3, 5, 10).")
            return
        if not sigma_levels:
            QMessageBox.warning(self, "Invalid Input", "Enter at least one sigma level.")
            return

        # --- Calculate background stats ---
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

        # Build band-pairs: each level N maps to [N-0.05, N+0.05] * std + median.
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
            QMessageBox.critical(self, "Contour Error", f"Could not generate contours:\n{str(e)}")
            traceback.print_exc()

    def _contour_target(self, action):
        """Resolve which panel a contour action applies to. Auto-picks the
        panel with an active contour; prompts only when both have one, and
        warns when neither does.

        Args:
            action (str): Short verb used in the messages ('save', 'clear').

        Returns:
            side (str): 'left' or 'right', or None if no panel qualifies.
        """
        sides = [
            side for side, plotter in (("left", self.slice1_plotter),
                                       ("right", self.slice2_plotter))
            if getattr(plotter, "current_contour", None)
            or getattr(plotter, "contours", None)
        ]
        if not sides:
            QMessageBox.warning(self, "No Contour",
                                f"There is no active contour to {action}.")
            return None
        if len(sides) == 1:
            return sides[0]

        return self._ask_target_panel(
            "Select Panel", f"Both panels have contours. Which one would you like to {action}?")

    def _prompt_and_clear_contours(self):
        """Ask which panel to clear contours from, then clear them.

        Args:
            None

        Returns:
            None
        """
        target = self._contour_target("clear")
        if target:
            self.clear_contours(target)

    def clear_contours(self, target):
        """Remove every contour artist from a panel.

        Args:
            target (str): Panel to clear, 'left' or 'right'.

        Returns:
            None
        """
        plotter = self.slice1_plotter if target == "left" else self.slice2_plotter
        
        if hasattr(plotter, 'contours'):
            for contour_set in plotter.contours:
                if hasattr(contour_set, 'collections'):
                    for c in contour_set.collections:
                        try: c.remove()
                        except Exception: pass
                else:
                    try: contour_set.remove()
                    except Exception: pass
            plotter.contours = []
            
        if hasattr(plotter, 'current_contour') and plotter.current_contour:
            if hasattr(plotter.current_contour, 'collections'):
                for c in plotter.current_contour.collections:
                    try: c.remove()
                    except Exception: pass
            else:
                try: plotter.current_contour.remove()
                except Exception: pass
            plotter.current_contour = None
            
        if target == "left":
            self.slice1_canvas.draw()
            self.last_contour_left_img = None
            self.last_contour_left_level = None
        else:
            self.slice2_canvas.draw()
            self.last_contour_right_img = None
            self.last_contour_right_level = None

    def _prompt_and_save_contour(self):
        """Ask which panel to save a contour from, then save it.

        Args:
            None

        Returns:
            None
        """
        target = self._contour_target("save")
        if target:
            self._save_contour_txt(target)

    def _save_contour_txt(self, target):
        """Write a panel's active contour's coordinates to a text/CSV file.

        Args:
            target (str): Panel whose contour to save, 'left' or 'right'.

        Returns:
            None
        """
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
        """Prompt user for target slice, mask file, and color to overlay a saved contour.

        Args:
            None

        Returns:
            None
        """
        target = self._ask_target_panel(
            "Load Saved Contour", "Which slice would you like to plot the loaded contour ON?")
        if target is None:
            return

        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Contour Text File", filter="Text files (*.txt *.csv)"
        )
        if not fname:
            return
            
        # Ask for contour color
        colors = ["white", "red", "green", "cyan", "magenta", "yellow", "black"]
        color, ok = QInputDialog.getItem(
            self, "Contour Color", "Select contour line color:", colors, 0, False
        )
        
        if ok and color:
            self._apply_loaded_contour(target, fname, color)

    def _apply_loaded_contour(self, target, fname, color):
        """Loads a saved contour and plots it, reprojected onto this panel's own WCS.

        Args:
            target (str): Panel to plot on, 'left' or 'right'.
            fname (str): Path to the saved contour text/CSV file.
            color (str): Matplotlib color name for the contour lines.

        Returns:
            None
        """
        target_plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter
        target_img = self.image_left if target == 'left' else self.image_right
        target_wcs = self.wcs_left if target == 'left' else self.wcs_right

        if target_img is None:
            QMessageBox.warning(self, "No Target Image", f"Load an image in the {target} panel first.")
            return

        try:
            data = np.genfromtxt(fname, delimiter=',', comments='#')
            if data.ndim == 1: data = data.reshape(1, -1)

            x_pix_saved = data[:, 1]
            y_pix_saved = data[:, 2]

            # Prefer RA/Dec converted into this panel's pixel grid so the
            # contour aligns across different images; else fall back to raw pixels.
            use_sky = data.shape[1] >= 5 and target_wcs is not None
            if use_sky:
                ra_pts = data[:, 3]
                dec_pts = data[:, 4]
                use_sky = bool(np.all(np.isfinite(ra_pts)) and np.all(np.isfinite(dec_pts)))

            if use_sky:
                sky = SkyCoord(ra=ra_pts * u.deg, dec=dec_pts * u.deg, frame='icrs')
                x_pts, y_pts = target_wcs.world_to_pixel(sky)
                x_pts = np.asarray(x_pts, dtype=float)
                y_pts = np.asarray(y_pts, dtype=float)
            else:
                QMessageBox.warning(
                    self, "No WCS Alignment",
                    "This contour file has no RA/Dec, or this panel has no WCS, "
                    "so it will be plotted at its saved pixel coordinates.\n\n"
                    "It will only line up correctly if this panel uses the exact "
                    "same pixel grid as the panel it was saved from.")
                x_pts, y_pts = x_pix_saved, y_pix_saved

            dist = np.sqrt(np.diff(x_pts)**2 + np.diff(y_pts)**2)
            break_indices = np.where(dist > 5.0)[0] + 1 
            
            x_plot = np.insert(x_pts, break_indices, np.nan)
            y_plot = np.insert(y_pts, break_indices, np.nan)
            
            ax = target_plotter.ax

            if not hasattr(target_plotter, 'contours'):
                target_plotter.contours = []
            
            contour_line, = ax.plot(x_plot, y_plot, color=color, linewidth=1.5, alpha=0.8, zorder=10)
            
            class DummyContourSet:
                def __init__(self, line):
                    """Wrap a plain Line2D so it looks like a QuadContourSet.

                    Args:
                        line (matplotlib.lines.Line2D): Plotted contour line.

                    Returns:
                        None
                    """
                    self.collections = [line]
            
            target_plotter.contours.append(DummyContourSet(contour_line))
            target_plotter.current_contour = DummyContourSet(contour_line)
            
            if target == 'left': self.slice1_canvas.draw()
            else: self.slice2_canvas.draw()
            
            QMessageBox.information(self, "Loaded", f"Successfully loaded contour from {os.path.basename(fname)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to plot saved contour:\n{str(e)}")
            traceback.print_exc()

    def _prompt_and_superimpose(self):
        """Ask for background/overlay/colormap/alpha, then blend one panel's
        image, reprojected, over the other panel's background image.

        Args:
            None

        Returns:
            None
        """
        if self.image_left is None or self.image_right is None:
            QMessageBox.warning(self, "Missing Images", "Load an image in BOTH panels before superimposing.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Superimpose / WCS Match")
        dialog.setMinimumWidth(380)
        layout = QVBoxLayout(dialog)

        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("Background (reference) image:"))
        bg_combo = QComboBox()
        bg_combo.addItems(["Left", "Right"])
        bg_layout.addWidget(bg_combo)
        layout.addLayout(bg_layout)

        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Overlay colormap:"))
        cmap_combo = QComboBox()
        cmap_combo.addItems(["viridis", "plasma", "hot", "cool", "Blues", "Greens", "Reds", "magma"])
        cmap_layout.addWidget(cmap_combo)
        layout.addLayout(cmap_layout)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Overlay alpha (0.1 – 1.0):"))
        alpha_spin = QDoubleSpinBox()
        alpha_spin.setRange(0.1, 1.0)
        alpha_spin.setValue(0.5)
        alpha_spin.setSingleStep(0.1)
        alpha_layout.addWidget(alpha_spin)
        layout.addLayout(alpha_layout)

        layout.addWidget(QLabel("<i>The overlay image is reprojected onto the background\nWCS grid and blended with the selected alpha.</i>"))

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

        bg          = bg_combo.currentText().lower()      
        ov          = 'right' if bg == 'left' else 'left'
        alpha       = alpha_spin.value()
        overlay_cmap = cmap_combo.currentText()

        bg_img    = self.image_left   if bg == 'left' else self.image_right
        bg_wcs    = self.wcs_left     if bg == 'left' else self.wcs_right
        ov_img    = self.image_left   if ov == 'left' else self.image_right
        ov_wcs    = self.wcs_left     if ov == 'left' else self.wcs_right

        if bg_wcs is None or ov_wcs is None:
            QMessageBox.warning(self, "No WCS", "Both images must have valid WCS to superimpose.")
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            reprojected, footprint = _reproject_wcs_to_wcs(ov_img, ov_wcs, bg_wcs, bg_img.shape, order=1)

            plotter = self.slice1_plotter if bg == 'left' else self.slice2_plotter
            canvas  = self.slice1_canvas  if bg == 'left' else self.slice2_canvas
            ax      = plotter.ax

            if np.count_nonzero(footprint) == 0:
                QMessageBox.warning(self, "No Overlap", "The two images have no overlapping sky area.")
                return

            ov_finite = reprojected[np.isfinite(reprojected)]
            if ov_finite.size == 0:
                return

            ov_norm = Normalize(vmin=float(np.nanmin(ov_finite)), vmax=float(np.nanmax(ov_finite)))
            ax.imshow(reprojected, origin='lower', cmap=overlay_cmap,
                    norm=ov_norm, alpha=alpha, aspect='auto', interpolation='nearest')
            canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Superimpose Error", f"Failed to superimpose:\n{str(e)}")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

    def _setup_main_layout(self):
        """Set up the main application layout.

        Args:
            None

        Returns:
            None
        """
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        left_half = QHBoxLayout()
        
        self.open_left_btn = QPushButton("Open 2D FITS (Left)")
        self.open_left_btn.clicked.connect(lambda: self.load_2d_fits("left"))
        
        self.header_left_btn = QPushButton("Header")
        self.header_left_btn.clicked.connect(lambda: self.show_header("left"))

        self.save_left_btn = QPushButton("Save FITS")
        self.save_left_btn.clicked.connect(lambda: self.save_fits("left"))

        # Left Stats strictly bounded so they don't stretch Canvas space
        left_stats_layout = QVBoxLayout()
        left_stats_layout.addWidget(self.open_left_btn)
        left_stats_layout.addWidget(self.header_left_btn)
        left_stats_layout.addWidget(self.save_left_btn)
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
        left_sidebar.addWidget(self.scale_btn_left)
        for shape in ["*", "▭", "○", "⬭", "✏"]:
            left_sidebar.addWidget(self.shape_buttons["left"][shape])
        left_sidebar.addWidget(self.copy_btn_l)
        left_sidebar.addWidget(self.copy_color_btn_l)
        left_sidebar.addWidget(self.copy_scale_btn_l)
        left_sidebar.addWidget(self.edit_shape_btn_l)
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

        self.save_right_btn = QPushButton("Save FITS")
        self.save_right_btn.clicked.connect(lambda: self.save_fits("right"))

        # Right Stats strictly bounded
        right_stats_layout = QVBoxLayout()
        right_stats_layout.addWidget(self.open_right_btn)
        right_stats_layout.addWidget(self.header_right_btn)
        right_stats_layout.addWidget(self.save_right_btn)
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
        right_sidebar.addWidget(self.scale_btn_right)
        for shape in ["*", "▭", "○", "⬭", "✏"]:
            right_sidebar.addWidget(self.shape_buttons["right"][shape])
        right_sidebar.addWidget(self.copy_btn_r)
        right_sidebar.addWidget(self.copy_color_btn_r)
        right_sidebar.addWidget(self.copy_scale_btn_r)
        right_sidebar.addWidget(self.edit_shape_btn_r)
        right_sidebar.addWidget(self.erase_btn_r)
        right_sidebar.addStretch()

        right_sidebar_widget = QWidget()
        right_sidebar_widget.setLayout(right_sidebar)
        right_sidebar_widget.setFixedWidth(120)

        right_half.addWidget(right_stats_widget, 0)
        right_half.addWidget(right_slice_widget, 1)
        right_half.addWidget(right_sidebar_widget, 0)

        top_layout.addLayout(left_half, 1)
        top_layout.addLayout(right_half, 1)

        # ===== BOTTOM CONTROLS: two tiers packed right under the slice panels =====
        # Tier 1 (upstairs): the collapsible groups, nothing else.
        # Region save/import for a panel's currently loaded image -- no "Save
        # Cube (Masked)"/plain "Save Slice" here, unlike the 3D window: there
        # is no cube, and a loaded 2D image is already a standalone file.
        self.region_group = CollapsibleButtonGroup("Region")
        for _b in (self.region_save_slice_masked_btn, self.region_save_wcs_btn,
                   self.region_import_wcs_btn):
            self.region_group.add_button(_b)

        self.contour_group = CollapsibleButtonGroup("Contour")
        for _b in (self.contour_btn, self.save_contour_btn,
                   self.load_contour_btn, self.clear_contour_btn):
            self.contour_group.add_button(_b)

        self.convolution_group = CollapsibleButtonGroup("Convolution")
        for _b in (self.match_res_naman_btn, self.batch_match_btn):
            self.convolution_group.add_button(_b)

        tier1 = QHBoxLayout()
        tier1.setContentsMargins(0, 0, 0, 0)
        tier1.addWidget(self.region_group)
        tier1.addWidget(self.contour_group)
        tier1.addWidget(self.convolution_group)
        tier1.addStretch(1)

        # Tier 2 (downstairs): rotation cluster on the left, Superimpose on the right.
        rotate_widget = QWidget()
        rotate_row = QHBoxLayout(rotate_widget)
        rotate_row.setContentsMargins(0, 0, 0, 0)
        rotate_row.addWidget(self.rot_ccw_radio)
        rotate_row.addWidget(self.rot_cw_radio)
        rotate_row.addWidget(self.angle_spin)
        rotate_row.addWidget(self.rotate_left_btn)
        rotate_row.addWidget(self.rotate_right_btn)

        tier2 = QHBoxLayout()
        tier2.setContentsMargins(0, 0, 0, 0)
        tier2.addWidget(rotate_widget)
        tier2.addStretch(1)
        tier2.addWidget(self.superimpose_btn)

        main_layout.addLayout(top_layout, 3)
        main_layout.addLayout(tier1, 0)
        main_layout.addLayout(tier2, 0)
        main_layout.addStretch(2)  # empty space fills the rest (no spectrum here)

        self.setLayout(main_layout)

    def show_header(self, target):
        """Display the FITS header text for a panel.

        Args:
            target (str): Panel whose header to show, 'left' or 'right'.

        Returns:
            None
        """
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
        """Rebuild a panel's matplotlib axes from a FITS header's WCS.

        Args:
            target (str): Panel to rebuild, 'left' or 'right'.
            header (astropy.io.fits.Header): Header to build the WCS from.

        Returns:
            None
        """
        plotter = self.slice1_plotter if target == 'left' else self.slice2_plotter
        canvas  = self.slice1_canvas  if target == 'left' else self.slice2_canvas

        plotter.fig.clf()
        ax_rect   = [0.25, 0.20, 0.50, 0.66]
        cbar_rect = [0.80, 0.20, 0.03, 0.66]

        wcs_for_projection = None

        # --- Step 1: Build a clean, strictly-2D, SIP-free WCS ---
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
            print(f"DEBUG rebuild_axes [{target}]: WCS parsed OK -> {wcs_for_projection}")

        except Exception as e:
            print(f"DEBUG rebuild_axes [{target}]: WCS parse FAILED -> {e}")
            wcs_for_projection = None
            
        # Store so hover readout can use it
        if target == 'left':
            self.wcs_left  = wcs_for_projection
        else:
            self.wcs_right = wcs_for_projection

        # --- Step 2: Create axes (WCSAxes if possible, plain fallback) ---
        try:
            if wcs_for_projection is None:
                raise ValueError("No valid WCS available for projection")
            plotter.ax = plotter.fig.add_axes(ax_rect, projection=wcs_for_projection)
            print(f"DEBUG rebuild_axes [{target}]: WCSAxes created OK")
        except Exception as e:
            print(f"DEBUG rebuild_axes [{target}]: WCSAxes FAILED, using plain axes -> {e}")
            plotter.ax = plotter.fig.add_axes(ax_rect)

        plotter.cbar_ax = plotter.fig.add_axes(cbar_rect)
        plotter.cbar_ax.set_visible(False)

        # The figure was cleared above, so the old RegionSelector's patch
        # handlers would leak; disconnect it before rebinding a fresh selector.
        old_selector = self.region_selector_left if target == 'left' else self.region_selector_right
        if old_selector is not None and getattr(old_selector, 'interactive_patch', None) is not None:
            try:
                old_selector.interactive_patch.disconnect()
            except Exception:
                pass
            old_selector.interactive_patch = None

        if target == 'left':
            self.region_selector_left = RegionSelector(
                plotter.ax, canvas,
                lambda mask, patch, coords: self._handle_region_select(mask, patch, coords, 'left'))
        else:
            self.region_selector_right = RegionSelector(
                plotter.ax, canvas,
                lambda mask, patch, coords: self._handle_region_select(mask, patch, coords, 'right'))

        if target == 'left':
            self.last_selection_left = None
        else:
            self.last_selection_right = None
        if getattr(self, 'region_state', None) is not None:
            self._set_region_state(target, "idle")

    def load_2d_fits(self, target):
        """Open a file dialog and load a 2D FITS image into a panel.

        Args:
            target (str): Panel to load into, 'left' or 'right'.

        Returns:
            None
        """
        fname, _ = QFileDialog.getOpenFileName(
            self, f"Open 2D FITS ({target.capitalize()})", filter="FITS files (*.fits *.fz *.fit)")
        
        if not fname:
            return

        try:
            self.clear_region(target)
            
            with fits.open(fname) as hdul:
                data = None
                header = None
                coverage = None
                # Accurately find the correct 2D extension (fixes JWST empty primary HDUs)
                for hdu in hdul:
                    if hdu.data is not None and getattr(hdu.data, 'size', 0) > 0:
                        sq_data = np.squeeze(hdu.data)
                        if sq_data.ndim == 2:
                            data = sq_data
                            header = hdu.header.copy()
                            break

                # "COVERAGE" extension holds the per-pixel coverage fraction (0-1)
                # from the reproject pipeline, used to weight partial-overlap pixels.
                for hdu in hdul:
                    name = (getattr(hdu, 'name', '') or '').strip().upper()
                    if name == 'COVERAGE' and hdu.data is not None:
                        try:
                            cov_arr = np.squeeze(hdu.data).astype(float)
                            if data is not None and cov_arr.shape == data.shape:
                                coverage = cov_arr
                                break
                        except Exception:
                            pass
                
                if data is None:
                    # No HDU squeezed to 2D; distinguish a genuine multi-plane
                    # cube (reject with guidance) from a file with no image data.
                    cube_shape = None
                    for hdu in hdul:
                        if hdu.data is None or getattr(hdu.data, 'size', 0) == 0:
                            continue
                        sq = np.squeeze(hdu.data)
                        if sq.ndim > 2:
                            cube_shape = tuple(sq.shape)
                            break
                    if cube_shape is not None:
                        QMessageBox.warning(
                            self, "3D Cube Not Supported",
                            f"This file holds a {cube_shape} data cube, not a 2D image.\n\n"
                            "The 2D viewer needs a single plane. Extract or collapse a "
                            "slice in the main window first, then open that saved 2D FITS here.")
                    else:
                        QMessageBox.warning(
                            self, "Load Error", "No image data found in FITS file.")
                    return

                # data is the squeezed 2D array from the loop above (degenerate
                # axes accepted); true multi-plane cubes are rejected earlier.

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

                filename = os.path.basename(fname)
                if target == "left":
                    self.base_image_left = data.copy()
                    self.base_header_left = header.copy()
                    self.cum_angle_left = 0.0
                    self.image_left = data
                    self.header_left = header
                    self.coverage_left = coverage
                    self.filename_left = filename
                else:
                    self.base_image_right = data.copy()
                    self.base_header_right = header.copy()
                    self.cum_angle_right = 0.0
                    self.image_right = data
                    self.header_right = header
                    self.coverage_right = coverage
                    self.filename_right = filename

                self.rebuild_axes(target, header)
                self.display_image(target, title=filename)

                for _b in (self.region_save_slice_masked_btn, self.region_save_wcs_btn,
                           self.region_import_wcs_btn):
                    _b.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading FITS file:\n{str(e)}")
            traceback.print_exc()

    def rotate_image(self, target):
        """Rotate a panel's image by the spin box angle and update its WCS.

        Args:
            target (str): Panel to rotate, 'left' or 'right'.

        Returns:
            None
        """
        base_image = self.base_image_left if target == "left" else self.base_image_right
        base_header = self.base_header_left if target == "left" else self.base_header_right
        
        if base_image is None or base_header is None:
            QMessageBox.warning(self, "No Image", f"Please load an image into the {target} panel first.")
            return
            
        angle = self.angle_spin.value()
        if angle == 0.0:
            return
            
        if self.rot_cw_radio.isChecked():
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
                # Coverage map (if any) is no longer valid after rotation,
                # because rotate() reshapes the array; user can re-load to get it back.
                self.coverage_left = None
            else:
                self.image_right = array_out
                self.header_right = new_header
                self.coverage_right = None
                
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
        """Save a panel's image and header to a new FITS file.

        Args:
            target (str): Panel to save, 'left' or 'right'.

        Returns:
            None
        """
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
        """Render a panel's current image with its colormap and scale.

        Args:
            target (str): Panel to render, 'left' or 'right'.
            title (str): Plot title; defaults to no title.

        Returns:
            None
        """
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

        # 1. Calculate Intensity Normalization -- stored vmin/vmax (Scale
        # dialog) override auto, same resolution order as the 3D window.
        finite = image[np.isfinite(image)]
        auto_vmin = float(np.nanmin(finite)) if finite.size else 0.0
        auto_vmax = float(np.nanmax(finite)) if finite.size else 1.0
        stored_vmin = self.slice1_vmin if target == 'left' else self.slice2_vmin
        stored_vmax = self.slice1_vmax if target == 'left' else self.slice2_vmax
        vmin = auto_vmin if stored_vmin is None else stored_vmin
        vmax = auto_vmax if stored_vmax is None else stored_vmax
        if vmin == vmax:
            vmax = vmin + 1.0
        view_vmin, view_vmax = vmin, vmax  # as the user sees them, before any Log shift

        image_to_show = image
        if scale == 'Log':
            # Log needs v>0. Shift image, vmin, and vmax by one constant --
            # sized off the image's own true minimum, not vmin, so a vmin
            # narrowed above the true floor still lifts every real pixel
            # above zero instead of leaving some at/below zero for LogNorm
            # to mask instead of clamp to the floor color (see visualization.py).
            img_min = float(np.nanmin(finite)) if finite.size else 0.0
            if img_min <= 0:
                shift = -img_min + 1e-10
                image_to_show = image + shift
                vmin = vmin + shift
                vmax = vmax + shift
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif scale == 'Sqrt':
            vmin_sqrt = max(vmin, 0.0)
            norm = PowerNorm(gamma=0.5, vmin=vmin_sqrt, vmax=vmax)
        elif scale == 'Power (γ=2)':
            norm = PowerNorm(gamma=2.0, vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        # 2. Render the Image
        im = ax.imshow(image_to_show, origin='lower', cmap=cmap, norm=norm,
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

                    # Choose formatter precision based on FOV; Astropy auto-picks spacing.
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
                        # Sub-arcmin field needs sub-second RA precision.
                        ra.set_major_formatter('hh:mm:ss.ss')
                        dec.set_major_formatter('dd:mm:ss.s')

                    # No set_ticks() call: Astropy auto-locates ticks for the current FOV.

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

        # 5. Update Statistics Panel  (unit on its own line so the narrow
        # stats column doesn't clip it - matches the 3D window)
        mn   = np.nanmin(image)
        mx   = np.nanmax(image)
        mean = np.nanmean(image)
        total_sum = np.nansum(image)
        std  = np.nanstd(image)
        bunit = header['BUNIT'] if (header is not None and 'BUNIT' in header) else ''
        unit_line = f"[{bunit}]:\n" if bunit else ""
        stats_label.setText(
            f"Intensity\n{unit_line}"
            f"min={mn:.3e}\n"
            f"max={mx:.3e}\n"
            f"mean={mean:.3e}\n"
            f"sum={total_sum:.3e}\n"
            f"std={std:.3e}\n"
            f"\n"
            f"Scaling: {scale}\n"
            f"vmin={view_vmin:.3e}\n"
            f"vmax={view_vmax:.3e}"
        )

        # 6. Refresh Canvas
        if target == 'left':
            self.slice1_canvas.draw()
        else:
            self.slice2_canvas.draw()

    def _on_mouse_hover(self, event, target):
        """Handle dynamic updates of mouse coordinate position and value mapping.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Motion event.
            target (str): Panel hovered over, 'left' or 'right'.

        Returns:
            None
        """
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

    # ----- Region selection: the shared controller lives in RegionSelectionMixin.
    # ----- These hooks feed it this window's per-panel image + header WCS. ------
    def _region_image(self, target):
        """Return the 2D array currently shown in the given panel.

        Args:
            target (str): Panel to read from, 'left' or 'right'.

        Returns:
            image (numpy.ndarray): The displayed image, or None if unset.
        """
        return self.image_left if target == "left" else self.image_right

    def _region_flux_unit(self, target):
        """Return the flux-unit string for the given panel ('' when unknown).

        Args:
            target (str): Panel to read from, 'left' or 'right'.

        Returns:
            unit (str): Flux-unit label, or '' when unknown.
        """
        header = self.header_left if target == "left" else self.header_right
        if header is not None and "BUNIT" in header:
            return str(header["BUNIT"])
        return ""

    def _region_pixel_coords(self, target, x, y):
        """Convert a pixel position to sky coordinates for the given panel.

        Args:
            target (str): Panel the pixel belongs to, 'left' or 'right'.
            x (float): Pixel x coordinate.
            y (float): Pixel y coordinate.

        Returns:
            coords (tuple): (ra, dec) in degrees, or (None, None) if the
                panel has no celestial WCS.
        """
        wcs_obj = self.wcs_left if target == "left" else self.wcs_right
        if wcs_obj is None or not getattr(wcs_obj, "has_celestial", False):
            return None, None
        try:
            w2d = wcs_obj.celestial if wcs_obj.pixel_n_dim > 2 else wcs_obj
            sky = w2d.pixel_to_world(x, y)
            return float(sky.ra.deg), float(sky.dec.deg)
        except Exception:
            return None, None

    def _region_selection_to_mask(self, target, sel):
        """Build a full-image mask from a selection dict, the 2D-viewer
        analogue of the 3D window's own _selection_to_full_mask (there is no
        cube here, so the mask is sized to the panel's own loaded image).

        Args:
            target (str): Panel the selection belongs to, 'left' or 'right'.
            sel (dict): Selection dict from last_selection_left/right.

        Returns:
            mask (numpy.ndarray): Boolean mask, same shape as the panel's image.
        """
        if sel.get("shape") == "*":
            # Point selections store coords, not a mask -- build a single-pixel one.
            x, y = sel["coords"]
            x, y = int(round(x)), int(round(y))
            image = self._region_image(target)
            mask = np.zeros(image.shape, dtype=bool)
            mask[y, x] = True
            return mask
        return sel["mask"]

    def _region_save_slice_masked(self):
        """Save one panel's currently loaded image as a standalone 2D FITS,
        with pixels outside that panel's region set to NaN.

        Args:
            None

        Returns:
            None
        """
        target = self._ask_target_panel("Save Slice (Masked)", "Which panel's region would you like to save?")
        if target is None:
            return
        image = self._region_image(target)
        if image is None:
            QMessageBox.warning(self, "No Image Loaded", f"Load an image into the {target.capitalize()} panel first.")
            return
        sel = self.last_selection_left if target == "left" else self.last_selection_right
        if not sel:
            QMessageBox.warning(
                self, "Region Selection Required",
                f"Select a region or pixel on the {target.capitalize()} side first.")
            return
        mask = self._region_selection_to_mask(target, sel)
        header = self.header_left if target == "left" else self.header_right

        path, _ = QFileDialog.getSaveFileName(self, "Save Slice (Masked) as 2D FITS", "", "FITS files (*.fits)")
        if not path:
            return
        if not path.lower().endswith(".fits"):
            path += ".fits"

        masked = np.asarray(image, dtype=np.float32).copy()
        masked[~mask] = np.nan
        try:
            out_header = build_2d_export_header(header, masked.shape)
            write_fits_product(path, masked, out_header)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not export slice:\n{e}")
            return
        QMessageBox.information(self, "Export Complete", f"Masked slice saved to:\n{path}")

    def _region_save_wcs_text(self):
        """Save one panel's region as RA/Dec sky coordinates in a text
        file, so it can be placed on a different 2D image later.

        Args:
            None

        Returns:
            None
        """
        target = self._ask_target_panel("Save Region as WCS Text", "Which panel's region would you like to save?")
        if target is None:
            return
        sel = self.last_selection_left if target == "left" else self.last_selection_right
        if not sel:
            QMessageBox.warning(
                self, "Region Selection Required",
                f"Select a region or pixel on the {target.capitalize()} side first.")
            return

        header = self.header_left if target == "left" else self.header_right
        wcs2d = _make_2d_wcs(header)
        if wcs2d is None or not getattr(wcs2d, "has_celestial", False):
            QMessageBox.critical(
                self, "No WCS Available",
                "This image has no usable celestial WCS, so its region can't be saved as sky coordinates.")
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

        source_desc = getattr(self, f"filename_{target}", None) or ""
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
        using that panel's own image WCS to match the same sky position.

        Args:
            None

        Returns:
            None
        """
        target = self._ask_target_panel("Import Region", "Which panel should the imported region be placed on?")
        if target is None:
            return

        header = self.header_left if target == "left" else self.header_right
        wcs2d = _make_2d_wcs(header)
        if wcs2d is None or not getattr(wcs2d, "has_celestial", False):
            QMessageBox.critical(
                self, "No WCS Available",
                "This image has no usable celestial WCS, so a saved region can't be placed on it.")
            return

        image = self._region_image(target)
        if image is None:
            QMessageBox.warning(self, "Import Failed", "No image is currently loaded on that panel.")
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
        """Handle clicks on a slice display.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Click event.
            target (str): Panel clicked on, 'left' or 'right'.

        Returns:
            None
        """
        if event.inaxes is None or self.active_shape != "*": return

        ax = self.slice1_plotter.ax if target == "left" else self.slice2_plotter.ax
        if event.inaxes != ax: return

        x, y = int(round(event.xdata)), int(round(event.ydata))
        self._handle_pixel_click(x, y, target)

    def prompt_and_match_resolution_naman(self):
        """"Pair" button: convolve the shorter-wavelength panel to the
        longer-wavelength panel's PSF (FWHM(") = 0.033*wavelength_um + 0.106,
        MIRI only), reprojecting onto its own native grid -- no padded WCS.

        Args:
            None

        Returns:
            None
        """
        if self.image_left is None or self.image_right is None:
            QMessageBox.warning(self, "Missing Images",
                                "Load images in BOTH panels first.")
            return

        pix_l = _get_pixel_size_arcsec(self.header_left)
        pix_r = _get_pixel_size_arcsec(self.header_right)
        if pix_l is None or pix_r is None:
            QMessageBox.warning(
                self, "Missing Pixel Scale",
                "Could not determine pixel scale for one or both images.")
            return

        bad = _non_miri_files([(self.filename_left or "left panel", self.header_left),
                               (self.filename_right or "right panel", self.header_right)])
        if bad:
            detail = "\n".join(f"  - {fn}: {inst}" for fn, inst in bad)
            QMessageBox.warning(
                self, "Unsupported Instrument",
                "Pair currently supports MIRI only (no confirmed PSF formula "
                f"for other instruments yet):\n{detail}")
            return

        if not _is_per_pixel_flux_unit(self.header_left.get('BUNIT', '') if self.header_left else '') or \
           not _is_per_pixel_flux_unit(self.header_right.get('BUNIT', '') if self.header_right else ''):
            QMessageBox.warning(
                self, "Unsupported BUNIT for Naman",
                "Naman's algorithm assumes both images are flux-per-pixel.\n\n"
                "Use the regular Pair button for surface-brightness images.")
            return

        wl_left = _get_line_wavelength_um(
            self.header_left, self.filename_left or "left panel", self)
        if wl_left is None:
            return
        wl_right = _get_line_wavelength_um(
            self.header_right, self.filename_right or "right panel", self)
        if wl_right is None:
            return
        if wl_left == wl_right:
            QMessageBox.warning(
                self, "Naman Algorithm",
                "Both panels have the same wavelength -- nothing to convolve.")
            return

        # Naman's PSF grows with wavelength, so the longer-wavelength panel
        # is always the reference, regardless of pixel scale.
        if wl_left > wl_right:
            ref_side, conv_side = "left", "right"
            ref_wl, conv_wl = wl_left, wl_right
            ref_header, ref_img = self.header_left, self.image_left
            fine_header, fine_img, fine_pix = self.header_right, self.image_right, pix_r
        else:
            ref_side, conv_side = "right", "left"
            ref_wl, conv_wl = wl_right, wl_left
            ref_header, ref_img = self.header_right, self.image_right
            fine_header, fine_img, fine_pix = self.header_left, self.image_left, pix_l

        fine_wcs = _make_2d_wcs(fine_header)
        if fine_wcs is None:
            QMessageBox.warning(self, "WCS Error",
                                "Could not build a 2D WCS for the fine image.")
            return

        ref_wcs = _make_2d_wcs(ref_header)
        if ref_wcs is not None and ref_img is not None:
            if not _wcs_footprints_overlap(
                    fine_wcs, fine_img.shape, ref_wcs, ref_img.shape):
                QMessageBox.warning(
                    self, "No Sky Overlap",
                    "The two images cover completely different regions of the sky.\n\n"
                    "Pair requires the panels to share sky coverage so "
                    "the convolved image can be placed on a meaningful WCS grid.\n\n"
                    "Please load a matching pair of images of the same field.")
                return

        fwhm_in = 0.033 * conv_wl + 0.106
        fwhm_target = 0.033 * ref_wl + 0.106
        sigma_arcsec = ((fwhm_target / gaussian_sigma_to_fwhm) ** 2 - (fwhm_in / gaussian_sigma_to_fwhm) ** 2) ** 0.5

        bunit = fine_header.get('BUNIT', '') if fine_header is not None else ''

        info = (
            f"<b>Naman Convolve: {conv_side.capitalize()} ({conv_wl:.4f} &micro;m) &rarr; "
            f"{ref_side} ({ref_wl:.4f} &micro;m) PSF</b><br>"
            f"Naman FWHM({conv_wl:.4f} &micro;m) = {fwhm_in:.4f}\"<br>"
            f"Naman FWHM({ref_wl:.4f} &micro;m) = {fwhm_target:.4f}\"<br>"
            f"Smoothing &sigma; : {sigma_arcsec:.4f}\"<br>"
            f"Output grid    : {ref_side}'s own native grid "
            f"({ref_img.shape[1]} &times; {ref_img.shape[0]} px)<br>"
            f"BUNIT          : {bunit} &rarr; flux mode<br><br>"
            "Algorithm: Naman's gaussian_convolve_match/reproject_flux_maps "
            "(FWHM(\") = 0.033*wavelength_um + 0.106). Proceed?"
        )
        if QMessageBox.question(self, "Naman Convolve", info) != QMessageBox.Yes:
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            naman_out = _run_naman_convolve_pair(
                fine_img, fine_header, conv_wl, fine_pix,
                ref_img, ref_header, ref_wl)
            result_img = naman_out['image']
            coverage = naman_out['footprint']

            if not np.any(np.isfinite(result_img)):
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(
                    self, "Empty Result",
                    "Naman Convolve completed but the output has no valid pixels.")
                return

            out_header = build_resample_header(
                ref_header, result_img.shape, bunit=bunit or None,
                history=(f"Naman Convolve: sigma={sigma_arcsec:.4f}\"; "
                        f"reprojected onto {ref_side}'s native grid "
                        f"(ref wavelength {ref_wl:.4f} um)"),
            )

            fname, _ = QFileDialog.getSaveFileName(
                self, "Save Convolved FITS", filter="FITS (*.fits)")
            if fname:
                if not fname.lower().endswith(".fits"):
                    fname += ".fits"
                write_fits_product(fname, result_img, out_header,
                                   extensions=[build_coverage_extension(coverage)])

                in_total = float(np.nansum(fine_img))
                out_total = float(np.nansum(result_img[np.isfinite(result_img)]))
                ratio = (out_total / in_total) if in_total != 0 else float('nan')
                QApplication.restoreOverrideCursor()
                QMessageBox.information(
                    self, "Saved",
                    f"Saved to:\n{fname}\n\n"
                    f"Flux audit ({bunit}):\n"
                    f"  Input  total : {in_total:.4e}\n"
                    f"  Output total : {out_total:.4e}\n"
                    f"  Ratio out/in : {ratio:.6f}\n"
                    f"  (1.000000 means flux is exactly conserved.)"
                )
            else:
                QApplication.restoreOverrideCursor()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Naman Convolve Error",
                                 f"An error occurred:\n{e}")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

