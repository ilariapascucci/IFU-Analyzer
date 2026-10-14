"""
Worker and result dialog for the pixel-by-pixel rotational diagram map.

RotDiagMapWorker runs create_rotational_diagram_map off the GUI thread.
RotDiagMapResultsDialog shows the maps and re-fits a pixel's diagram on click.
"""

import html
import traceback
from types import SimpleNamespace

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QWidget, QFileDialog, QMessageBox, QTextEdit,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure, SubplotParams

from .rotational_diagram_map import (
    create_rotational_diagram_map, create_rotational_diagram_map_multi,
    create_rotational_diagram_map_mixed, save_rotational_diagram_map_fits,
    fit_pixel, fit_pixel_multi, compute_native_line_maps, combine_line_maps,
    fit_region_multi, fit_region_mixed,
    STATUS_OK, STATUS_NO_DATA, STATUS_TOO_FEW_LINES, STATUS_FIT_FAILED,
    NATIVE_PX_UL, NATIVE_PX_DETECTION,
)
from .rotational_diagram_h2_fitter import model_1T, model_2T
from .fits_export import build_line_map_header, write_fits_product
from .line_catalog import dedupe_colocated_lines as _dedupe_colocated
from . import region_io
from .region_selector import RegionSelector, mask_from_shape_coords
from .analyzer_2d import _make_2d_wcs
from .gui_region_selection import RegionSelectionMixin
from .visualization import SlicePlotter
from .colormap_dialog import ColorMapDialog
from .scale_dialog import ScaleDialog

# Display combo: label -> (1T/T1 key, T2 key or None, colour-bar label).
# 2T draws T1/T2 as two side-by-side panels; a None hot_key stays single.
_PARAMS = [
    ("Temperature", "T", "T2", "T (K)"),
    ("Temperature error", "T_err", "T2_err", "dT (K)"),
    ("Column density", "Ncol", "Ncol2", "N_u (cm⁻²)"),
    ("Column density error", "Ncol_err", "Ncol2_err", "dN_u (cm⁻²)"),
    ("Lines used", "n_lines", None, "detections"),
    ("Fit mask", "status", None, "1 = fitted"),
]


class RotDiagMapWorker(QThread):
    """
    Background thread that builds the rotational diagram map.

    Args:
        cube_or_list (CubeData): One cube, for the single-cube path.
            None when cubes_and_lines is given instead
        line_table (list): Line dicts for the fit. None when cubes_and_lines
            is given instead
        cubes_and_lines (list): (CubeData, line_dict) pairs, one per line, for
            two or more cubes -- always routes to the convolve+reproject path,
            even if the cubes already look aligned. None for the single-cube path
        kwargs (dict): Passed through to whichever create_rotational_diagram_map*
            function is used

    Returns:
        None
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, cube_or_list=None, line_table=None, cubes_and_lines=None,
                imported_lines=None, **kwargs):
        """Store the fit inputs for run() to consume on the worker thread.

        Args:
            cube_or_list (CubeData): One cube, for the single-cube path.
            line_table (list): Line dicts for the fit.
            cubes_and_lines (list): (CubeData, line_dict) pairs for two or more cubes.
            imported_lines (list): Line dicts with an 'imported_map' payload,
                or None/empty -- routes to create_rotational_diagram_map_mixed
                instead of _multi when non-empty.
            kwargs (dict): Passed through to whichever create_rotational_diagram_map* function is used.

        Returns:
            None
        """
        super().__init__()
        self.cube_or_list = cube_or_list
        self.line_table = line_table
        self.cubes_and_lines = cubes_and_lines
        self.imported_lines = imported_lines
        self.kwargs = kwargs
        self.cancelled = False

    def run(self):
        """
        Run the map and emit finished/error.

        Args:
            None

        Returns:
            None
        """
        try:
            if self.imported_lines:
                result = create_rotational_diagram_map_mixed(
                    self.cubes_and_lines or [], self.imported_lines,
                    progress_callback=self._emit_progress, cancel_flag=self,
                    **self.kwargs,
                )
            elif self.cubes_and_lines is not None:
                result = create_rotational_diagram_map_multi(
                    self.cubes_and_lines,
                    progress_callback=self._emit_progress, cancel_flag=self,
                    **self.kwargs,
                )
            else:
                result = create_rotational_diagram_map(
                    self.cube_or_list, self.line_table,
                    progress_callback=self._emit_progress, cancel_flag=self,
                    **self.kwargs,
                )
            if self.cancelled:
                self.error.emit("Rotational diagram map cancelled by user")
            elif result is None:
                self.error.emit("No cube data loaded")
            else:
                self.finished.emit(result)
        except Exception as exc:
            self.error.emit(f"{exc}\n{traceback.format_exc()}")

    def _emit_progress(self, current, total):
        """
        Forward progress to the GUI.

        Args:
            current (int): Pixels done
            total (int): Pixels total

        Returns:
            None
        """
        self.progress.emit(current, total)

    def request_cancel(self):
        """
        Ask the run to stop at the next row.

        Args:
            None

        Returns:
            None
        """
        self.cancelled = True


class CalculateNativeMapsWorker(QThread):
    """
    Background thread for the multi-cube "Calculate Maps" button: computes
    each line's own flux map at its cube's native resolution (stage 1 of the
    multi-cube map pipeline), without convolving or fitting a temperature.

    Args:
        cubes_and_lines (list): (CubeData, line_dict) pairs, one per input cube
        kwargs (dict): contam_lines/n_jobs/dedup_tol_um/region_masks/n_sigma,
            passed straight to compute_native_line_maps

    Returns:
        None
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, cubes_and_lines, **kwargs):
        """Store the inputs for run() to consume on the worker thread.

        Args:
            cubes_and_lines (list): (CubeData, line_dict) pairs.
            kwargs (dict): contam_lines/n_jobs/dedup_tol_um/region_masks/n_sigma.

        Returns:
            None
        """
        super().__init__()
        self.cubes_and_lines = cubes_and_lines
        self.kwargs = kwargs
        self.cancelled = False

    def run(self):
        """
        Run compute_native_line_maps and emit finished/error.

        Args:
            None

        Returns:
            None
        """
        try:
            result = compute_native_line_maps(
                self.cubes_and_lines, progress_callback=self._emit_progress,
                cancel_flag=self, **self.kwargs,
            )
            if self.cancelled:
                self.error.emit("Calculate Maps cancelled by user")
            else:
                self.finished.emit(result)
        except Exception as exc:
            self.error.emit(f"{exc}\n{traceback.format_exc()}")

    def _emit_progress(self, current, total):
        """
        Forward progress to the GUI.

        Args:
            current (int): Pixels done
            total (int): Pixels total

        Returns:
            None
        """
        self.progress.emit(current, total)

    def request_cancel(self):
        """
        Ask the run to stop at the next row.

        Args:
            None

        Returns:
            None
        """
        self.cancelled = True


class CombineMapsWorker(QThread):
    """
    Background thread for Run Map's fast path: recombines already-convolved
    per-line flux maps (Calculate Maps + Convolve Maps already ran) into a
    rotational diagram map, without repeating native extraction or convolution.

    Args:
        convolved_result (dict): Output of convolve_line_maps
        cubes_and_lines (list): (CubeData, line_dict) pairs, needed only to
            recompute region_fit (fit_region_multi needs the real cubes)
        region_masks (dict): {CubeData: boolean mask}, restricting
            region_fit's own summed spectrum the same way the imported
            region already restricts the cached native/convolved maps.
        kwargs (dict): molecule/isotopologue/opr/fit_opr/opr_grid/dist_pc/
            r_au/min_lines/fit2t, passed to fit_region_multi and combine_line_maps.
            An n_sigma entry is ignored: the convolved maps carry no threshold

    Returns:
        None
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, convolved_result, cubes_and_lines, region_masks=None,
                imported_lines=None, **kwargs):
        """Store the inputs for run() to consume on the worker thread.

        Args:
            convolved_result (dict): Output of convolve_line_maps.
            cubes_and_lines (list): (CubeData, line_dict) pairs.
            region_masks (dict): {CubeData: boolean mask}, or None.
            imported_lines (list): Line dicts with an 'imported_map' payload,
                or None/empty -- routes region_fit through fit_region_mixed
                instead of fit_region_multi when non-empty.
            kwargs (dict): molecule/isotopologue/opr/fit_opr/opr_grid/dist_pc/r_au/min_lines.

        Returns:
            None
        """
        super().__init__()
        self.convolved_result = convolved_result
        self.cubes_and_lines = cubes_and_lines
        self.region_masks = region_masks
        self.imported_lines = imported_lines
        self.kwargs = kwargs
        self.cancelled = False

    def run(self):
        """
        Recompute region_fit, run combine_line_maps, and emit finished/error.

        Args:
            None

        Returns:
            None
        """
        try:
            kept_pairs, _ = _dedupe_colocated(
                list(self.cubes_and_lines), 1e-5, line_of=lambda p: p[1])
            if self.imported_lines:
                region_fit = fit_region_mixed(
                    kept_pairs, self.imported_lines, molecule=self.kwargs["molecule"],
                    isotopologue=self.kwargs["isotopologue"], opr=self.kwargs["opr"],
                    fit_opr=self.kwargs["fit_opr"], opr_grid=self.kwargs["opr_grid"],
                    dist_pc=self.kwargs["dist_pc"], r_au=self.kwargs["r_au"],
                    min_lines=self.kwargs["min_lines"], region_masks=self.region_masks,
                    fit2t=self.kwargs.get("fit2t", False),
                )
            else:
                region_fit = fit_region_multi(
                    kept_pairs, molecule=self.kwargs["molecule"], isotopologue=self.kwargs["isotopologue"],
                    opr=self.kwargs["opr"], fit_opr=self.kwargs["fit_opr"], opr_grid=self.kwargs["opr_grid"],
                    dist_pc=self.kwargs["dist_pc"], r_au=self.kwargs["r_au"], min_lines=self.kwargs["min_lines"],
                    region_masks=self.region_masks, fit2t=self.kwargs.get("fit2t", False),
                )
            result = combine_line_maps(
                self.convolved_result, molecule=self.kwargs["molecule"], isotopologue=self.kwargs["isotopologue"],
                opr=self.kwargs["opr"], fit_opr=self.kwargs["fit_opr"], opr_grid=self.kwargs["opr_grid"],
                dist_pc=self.kwargs["dist_pc"], r_au=self.kwargs["r_au"], min_lines=self.kwargs["min_lines"],
                region_fit=region_fit, progress_callback=self._emit_progress, cancel_flag=self,
                fit2t=self.kwargs.get("fit2t", False),
            )
            if self.cancelled:
                self.error.emit("Rotational diagram map cancelled by user")
            else:
                self.finished.emit(result)
        except Exception as exc:
            self.error.emit(f"{exc}\n{traceback.format_exc()}")

    def _emit_progress(self, current, total):
        """
        Forward progress to the GUI.

        Args:
            current (int): Pixels done
            total (int): Pixels total

        Returns:
            None
        """
        self.progress.emit(current, total)

    def request_cancel(self):
        """
        Ask the run to stop at the next row.

        Args:
            None

        Returns:
            None
        """
        self.cancelled = True


def _opr_line(results):
    """
    One text line describing the OPR the fit used, the Run Fit wording.

    Args:
        results (dict): A rotational_diagram result dict

    Returns:
        line (str): "OPR = 3.00 (fixed)", "OPR = ... (fitted)", or "" for non-H2
    """
    if "OPR" not in results:
        return ""
    if "OPR_err" in results:
        return f"OPR = {results['OPR']:.3f} ± {results['OPR_err']:.3f} (fitted)\n"
    return f"OPR = {results['OPR']:.2f} (fixed)\n"


def _fit_block(results):
    """
    The "--- 1T FIT ---" or "--- 2T FIT ---" block, formatted like Run Fit's
    show_results -- whichever of '1T'/'2T' is present in `results`.

    Args:
        results (dict): A rotational_diagram result dict with a '1T' or '2T' entry

    Returns:
        text (str): The OPR line plus the fit's temperature(s) and column density(s)
    """
    text = _opr_line(results)
    if "1T" in results:
        r1 = results["1T"]
        text += (
            "\n--- 1T FIT ---\n"
            + f"T = {r1['T']:.1f} ± {r1['dT']:.1f} K\n"
            + f"Ncol = {r1['Ncol']:.2e} ± {r1['dNcol']:.2e} cm⁻²\n"
        )
    if "2T" in results:
        r2 = results["2T"]
        text += (
            "\n--- 2T FIT ---\n"
            + f"T1 = {r2['T1']:.1f} ± {r2.get('dT1', float('nan')):.1f} K\n"
            + f"T2 = {r2['T2']:.1f} ± {r2.get('dT2', float('nan')):.1f} K\n"
            + f"Ncol1 = {r2['Ncol1']:.2e} ± {r2.get('dNcol1', float('nan')):.2e} cm⁻²\n"
            + f"Ncol2 = {r2['Ncol2']:.2e} ± {r2.get('dNcol2', float('nan')):.2e} cm⁻²\n"
        )
    return text


def region_result_text(region):
    """
    Run-Fit-style block for the whole-cube summed spectrum.

    Args:
        region (dict): create_rotational_diagram_map's 'region_fit' entry, or None

    Returns:
        text (str): The header, 1T block, and spaxel count, or a short note
    """
    bar = "=" * 50
    head = (f"{bar}\nROTATIONAL DIAGRAM ANALYSIS RESULTS\n"
            f"(whole-cube summed spectrum -- same as Run Fit)\n{bar}\n\n")
    if not region or region.get("status") != STATUS_OK:
        reason = region.get("reason", "no data") if region else "not available"
        return head + f"No region fit: {reason}\n{bar}\n"
    tail = f"\nSpaxels summed: {region.get('n_pixels', 0)}"
    if region.get("n_cubes"):
        tail += f" (across {region['n_cubes']} cubes)"
    return head + _fit_block(region["results"]) + tail + f"\n{bar}\n"


def pixel_result_text(x, y, out):
    """
    Format one pixel's fit like the Run Fit results panel.

    Args:
        x (int): Spatial x index
        y (int): Spatial y index
        out (dict): fit_pixel output

    Returns:
        text (str): Multi-line summary
    """
    bar = "=" * 50
    head = f"{bar}\nROTATIONAL DIAGRAM ANALYSIS RESULTS -- PIXEL ({x}, {y})\n{bar}\n\n"
    if out.get("status") != STATUS_OK:
        return head + f"No fit: {out.get('reason', 'n/a')}\nLines used: {out.get('n_lines', 0)}\n{bar}\n"
    return head + _fit_block(out["results"]) + f"\nLines used: {out['n_lines']}\n{bar}\n"


def pixel_result_html(text, highlight=None):
    """
    Turn a pixel result text into HTML that bolds one 2T component's lines.
    The bold lines are the T and Ncol lines of the clicked map panel.

    Args:
        text (str): The text from pixel_result_text
        highlight (str): "T1" or "T2" to bold that component, or None for no bold

    Returns:
        html_text (str): HTML for a QTextEdit, with line breaks and spaces kept
    """
    bold_prefixes = ()
    if highlight in ("T1", "T2"):
        bold_prefixes = (f"{highlight} = ", f"Ncol{highlight[1]} = ")
    lines = []
    for line in text.splitlines():
        escaped = html.escape(line)
        lines.append(f"<b>{escaped}</b>" if line.startswith(bold_prefixes) else escaped)
    return '<div style="white-space: pre-wrap;">' + "\n".join(lines) + "</div>"


def _apply_scale_2d(data, mode):
    """
    Apply a linear/log/percentile display scale to a 2D array (NaNs preserved).
    Shared by every per-pixel map viewer in this module.

    Args:
        data (np.ndarray): The map values
        mode (str): "Linear", "Log", or "Percentile"

    Returns:
        scaled (np.ndarray): Log10, percentile-clipped, or unchanged
    """
    finite = data[np.isfinite(data)]
    if mode == "Log":
        out = np.full_like(data, np.nan)
        pos = np.isfinite(data) & (data > 0)
        out[pos] = np.log10(data[pos])
        return out
    if mode == "Percentile" and finite.size:
        lo, hi = np.percentile(finite, [2, 98])
        return np.clip(data, lo, hi)
    return data


def plot_rot_diagram(ax, plot_data, highlight=None):
    """
    Draw one rotational diagram on `ax`, the same style as show_results --
    one dashed 1T line, or both 2T components plus their combined curve.

    Args:
        ax (matplotlib.axes.Axes): Target axes
        plot_data (dict): The fitter's '_plot_data' (E_arr, y_arr, y_err,
            popt1, popt2)
        highlight (str): "T1" or "T2" to bold that component's line and
            legend entry -- which map panel a click came from, so it's
            unambiguous which temperature that panel was showing.

    Returns:
        None
    """
    e_arr = np.asarray(plot_data["E_arr"], dtype=float)
    y_arr = np.asarray(plot_data["y_arr"], dtype=float)
    y_err = np.asarray(plot_data["y_err"], dtype=float)
    popt1 = plot_data.get("popt1")
    popt2 = plot_data.get("popt2")

    ax.errorbar(e_arr, y_arr / np.log(10), yerr=y_err / np.log(10), fmt="o",
                color="black", markersize=6, capsize=4, zorder=3, label="Detection")
    if len(e_arr) > 1:
        e_fit = np.linspace(np.min(e_arr), np.max(e_arr), 200)
        if popt2 is not None:
            a1, t1, a2, t2 = popt2
            y_warm = a1 - e_fit / t1
            lw1 = 4.0 if highlight == "T1" else 2.5
            ax.plot(e_fit, y_warm / np.log(10), ":", linewidth=lw1, color="blue",
                    label=f"T1: {t1:.0f} K")
            y_hot = a2 - e_fit / t2
            lw2 = 4.0 if highlight == "T2" else 2.5
            ax.plot(e_fit, y_hot / np.log(10), ":", linewidth=lw2, color="orange",
                    label=f"T2: {t2:.0f} K")
            ax.plot(e_fit, model_2T(e_fit, *popt2) / np.log(10), "-", linewidth=3,
                    color="red", label="2-component model")
        elif popt1 is not None:
            ax.plot(e_fit, model_1T(e_fit, *popt1) / np.log(10), "--", linewidth=2, label="1T")
        ax.set_xlim(np.min(e_arr) - 500, np.max(e_arr) + 500)
    ax.set_xlabel("E_u (K)", fontsize=12, fontweight="bold")
    ax.set_ylabel("log(N_u / g_u)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    legend = ax.legend(fontsize=10, loc="upper right")
    if highlight in ("T1", "T2"):
        for text in legend.get_texts():
            if text.get_text().startswith(highlight + ":"):
                text.set_fontweight("bold")


class RotDiagMapResultsDialog(QDialog):
    """
    Map viewer for a rotational diagram map, with per-pixel diagram on click.

    Args:
        result (dict): Output of create_rotational_diagram_map
        cube_or_list (CubeData or list): Same cube(s) the map was built from
        line_table (list): Same line list, for re-fitting a clicked pixel
        fit_kwargs (dict): molecule/isotopologue/opr/... for fit_pixel
        parent (QWidget): Parent widget

    Returns:
        None
    """

    def __init__(self, result, cube_or_list, line_table, fit_kwargs, parent=None):
        """Store the map result and build the viewer UI.

        Args:
            result (dict): Output of create_rotational_diagram_map.
            cube_or_list (CubeData or list): Same cube(s) the map was built from.
            line_table (list): Same line list, for re-fitting a clicked pixel.
            fit_kwargs (dict): molecule/isotopologue/opr/... for fit_pixel.
            parent (QWidget): Parent widget.

        Returns:
            None
        """
        super().__init__(parent)
        self.result = result
        self.cube_or_list = cube_or_list
        self.line_table = line_table
        self.fit_kwargs = fit_kwargs
        self.fit2t = result.get("fit2t", False)
        self.pixel_select_enabled = True
        self.setWindowTitle(f"{result.get('molecule', 'H2')} Rotational Diagram Map")
        # Wider when 2T needs two side-by-side panels (T1/T2).
        if self.fit2t:
            self.resize(1550, 700)
        else:
            self.resize(1150, 680)
        self._setup_ui()

    def _setup_ui(self):
        """
        Build the map controls, canvas, and buttons.

        Args:
            None

        Returns:
            None
        """
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Display:"))
        self.param_combo = QComboBox()
        self.param_combo.addItems([p[0] for p in _PARAMS])
        self.param_combo.currentIndexChanged.connect(self._update_plot)
        ctrl.addWidget(self.param_combo)

        ctrl.addWidget(QLabel("  Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["inferno", "viridis", "plasma", "magma", "cividis", "coolwarm"])
        self.cmap_combo.currentIndexChanged.connect(self._update_plot)
        ctrl.addWidget(self.cmap_combo)

        # In 2T mode, T1 and T2 get their own independent scale -- e.g. a
        # faint T2 map in Log next to a well-behaved T1 map in Linear.
        if self.fit2t:
            ctrl.addWidget(QLabel("  Scale (T1):"))
            self.scale_combo_t1 = QComboBox()
            self.scale_combo_t1.addItems(["Linear", "Log", "Percentile"])
            self.scale_combo_t1.currentIndexChanged.connect(self._update_plot)
            ctrl.addWidget(self.scale_combo_t1)

            ctrl.addWidget(QLabel("  Scale (T2):"))
            self.scale_combo_t2 = QComboBox()
            self.scale_combo_t2.addItems(["Linear", "Log", "Percentile"])
            self.scale_combo_t2.currentIndexChanged.connect(self._update_plot)
            ctrl.addWidget(self.scale_combo_t2)
        else:
            ctrl.addWidget(QLabel("  Scale:"))
            self.scale_combo = QComboBox()
            self.scale_combo.addItems(["Linear", "Log", "Percentile"])
            self.scale_combo.currentIndexChanged.connect(self._update_plot)
            ctrl.addWidget(self.scale_combo)

        self.pixel_btn = QPushButton("✓ Pixel Select ON")
        self.pixel_btn.setCheckable(True)
        self.pixel_btn.setChecked(True)
        self.pixel_btn.setStyleSheet("background-color: #90EE90; font-weight: bold;")
        self.pixel_btn.clicked.connect(self._toggle_pixel_select)
        ctrl.addWidget(self.pixel_btn)
        ctrl.addWidget(QLabel("(click a pixel for its diagram; turn OFF to zoom/pan)"))
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.fig = Figure(figsize=(15, 5) if self.fit2t else (11, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        btns = QHBoxLayout()
        export_btn = QPushButton("Export as FITS")
        export_btn.clicked.connect(self._export_fits)
        btns.addWidget(export_btn)
        stats_btn = QPushButton("Show Statistics")
        stats_btn.clicked.connect(self._show_statistics)
        btns.addWidget(stats_btn)
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        self._update_plot()

    def _toggle_pixel_select(self):
        """
        Flip pixel-select mode and restyle the button like Flux Map.

        Args:
            None

        Returns:
            None
        """
        self.pixel_select_enabled = self.pixel_btn.isChecked()
        if self.pixel_select_enabled:
            self.pixel_btn.setText("✓ Pixel Select ON")
            self.pixel_btn.setStyleSheet("background-color: #90EE90; font-weight: bold;")
        else:
            self.pixel_btn.setText("✗ Pixel Select OFF")
            self.pixel_btn.setStyleSheet("background-color: #FFB6C6; font-weight: bold;")

    def _update_plot(self):
        """
        Redraw the selected map(s) with the chosen colormap and scale. In 2T
        mode, a T1/T2 parameter draws as two side-by-side panels instead
        of a combo entry each -- Lines used/Fit mask stay single always.

        Args:
            None

        Returns:
            None
        """
        label, key, hot_key, cbar_label = _PARAMS[self.param_combo.currentIndex()]
        self.fig.clear()
        # clear() drops the axes but not tight_layout()'s last spacing fix,
        # so repeated redraws keep tightening further -- reset it each time.
        self.fig.subplotpars = SubplotParams()

        if self.fit2t and hot_key is not None:
            ax_left = self.fig.add_subplot(121)
            ax_right = self.fig.add_subplot(122)
            self._draw_panel(ax_left, key, f"{label} (T1, lower T)", cbar_label, self.scale_combo_t1.currentText())
            self._draw_panel(ax_right, hot_key, f"{label} (T2, higher T)", cbar_label, self.scale_combo_t2.currentText())
            # Remembered so a click can tell which component's map it landed
            # on (see _on_canvas_click) -- only meaningful with two panels.
            self._ax_left, self._ax_right = ax_left, ax_right
        else:
            ax = self.fig.add_subplot(111)
            # Lines used/Fit mask stay single-panel even in 2T mode (no
            # hot_key) -- T1's own scale combo covers that one panel.
            scale_type = self.scale_combo_t1.currentText() if self.fit2t else self.scale_combo.currentText()
            self._draw_panel(ax, key, label, cbar_label, scale_type)
            self._ax_left, self._ax_right = None, None

        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_panel(self, ax, key, title, cbar_label, scale_type):
        """
        Draw one map parameter on `ax`, with its own colorbar.

        Args:
            ax (matplotlib.axes.Axes): Target axes
            key (str): Result dict key to plot ("status" draws as binary)
            title (str): Axes title (overridden for "status")
            cbar_label (str): Colorbar label
            scale_type (str): This panel's own "Linear"/"Log"/"Percentile"

        Returns:
            None
        """
        status = self.result["status"]
        if key == "status":
            data = (status == STATUS_OK).astype(float)
            cmap, title = "gray", "Fit mask  (white = fitted, black = no fit)"
        else:
            data = np.asarray(self.result[key], dtype=float)
            if key != "n_lines":
                data = np.where(status == STATUS_OK, data, np.nan)
            cmap = self.cmap_combo.currentText()
            data = self._apply_scale(data, scale_type)

        im = ax.imshow(data, origin="lower", cmap=cmap)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")
        self.fig.colorbar(im, ax=ax).set_label(cbar_label)

    def _apply_scale(self, data, scale_type):
        """
        Apply a scale type to a 2D array (NaNs preserved).

        Args:
            data (np.ndarray): The map values
            scale_type (str): "Linear", "Log", or "Percentile"

        Returns:
            scaled (np.ndarray): Log10, percentile-clipped, or unchanged
        """
        return _apply_scale_2d(data, scale_type)

    def _on_canvas_click(self, event):
        """
        Re-fit the clicked pixel and show its rotational diagram.

        Args:
            event (matplotlib.backend_bases.MouseEvent): The click

        Returns:
            None
        """
        # A modal dialog opened mid-press would swallow the matching
        # release, leaving the toolbar stuck mid-pan -- skip while active.
        if self.toolbar.mode or not self.pixel_select_enabled or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        ny, nx = self.result["status"].shape
        if 0 <= x < nx and 0 <= y < ny:
            if event.inaxes is self._ax_left:
                highlight = "T1"
            elif event.inaxes is self._ax_right:
                highlight = "T2"
            else:
                highlight = None
            self._show_pixel_diagram(x, y, highlight)

    def _show_pixel_diagram(self, x, y, highlight=None):
        """
        Pop a dialog with pixel (x, y)'s diagram and 1T result.

        Args:
            x (int): Spatial x index
            y (int): Spatial y index
            highlight (str): "T1"/"T2" to bold that component in the plot's
                legend (which map panel the click landed on), or None.

        Returns:
            None
        """
        if self.cube_or_list is not None:
            fit_fn = lambda: fit_pixel(self.cube_or_list, x, y, self.line_table, **self.fit_kwargs)
        elif "_flux_maps" in self.result:
            # Convolved/reprojected multi-cube map: re-fit from the already-
            # computed per-line flux maps instead of re-running convolution.
            multi_kwargs = {k: v for k, v in self.fit_kwargs.items() if k != "n_sigma"}
            fit_fn = lambda: fit_pixel_multi(self.result, x, y, **multi_kwargs)
        else:
            QMessageBox.information(
                self, "Not available for this map",
                "Per-pixel re-fit is not available for this map result (it "
                "predates caching the per-line flux maps) -- re-run Run Map, "
                "or use Show Statistics for the whole-map summed fit.")
            return
        try:
            out = fit_fn()
        except Exception as exc:
            QMessageBox.critical(self, "Pixel fit failed", str(exc))
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Pixel ({x}, {y}) rotational diagram")
        dlg.resize(1050, 520)
        lay = QHBoxLayout(dlg)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml(pixel_result_html(pixel_result_text(x, y, out), highlight))
        lay.addWidget(text, 1)

        if out["status"] == STATUS_OK and "_plot_data" in out["results"]:
            fig = Figure(figsize=(6.5, 5), dpi=100)
            plot_rot_diagram(fig.add_subplot(111), out["results"]["_plot_data"], highlight=highlight)
            fig.tight_layout()
            canvas = FigureCanvasQTAgg(fig)
            box = QWidget()
            box_lay = QVBoxLayout(box)
            box_lay.addWidget(NavigationToolbar2QT(canvas, dlg))
            box_lay.addWidget(canvas)
            lay.addWidget(box, 1)

        dlg.exec_()

    def _show_statistics(self):
        """
        Show temperature/column stats and the pixel fit breakdown.

        Args:
            None

        Returns:
            None
        """
        status = self.result["status"]
        ok = status == STATUS_OK
        total = status.size
        n_ok = int(ok.sum())
        min_lines = self.fit_kwargs.get("min_lines", self.result.get("min_lines", 3))
        t_cap = self.result.get("t_cap", 3000.0)

        # Region fit first, Run-Fit style, then the per-pixel spread.
        lines = region_result_text(self.result.get("region_fit")).splitlines() + [""]
        lines += ["PER-PIXEL SPREAD", "=" * 50, ""]

        def _block(name, arr, fmt):
            """Format one parameter's median/min/max/std over the fitted pixels.

            Args:
                name (str): Section heading to print.
                arr (ndarray): Per-pixel values for this parameter.
                fmt (str): Format spec applied to each printed number.

            Returns:
                lines (list): Text lines for this section, including a trailing blank line.
            """
            # Upcast from float32: Ncol's ~1e17-1e21 range overflows float32
            # during std's internal squaring, wrongly reporting inf.
            vals = arr[ok].astype(np.float64)
            vals = vals[np.isfinite(vals)]
            if not vals.size:
                return [f"{name}:", "  (no fitted pixels)", ""]
            return [
                f"{name}:",
                f"  Median:  {format(np.median(vals), fmt)}",
                f"  Min/Max: {format(np.min(vals), fmt)} / {format(np.max(vals), fmt)}",
                f"  Std:     {format(np.std(vals), fmt)}",
                "",
            ]

        t1_label = "TEMPERATURE T1 (K)" if self.fit2t else "TEMPERATURE (K)"
        ncol1_label = "COLUMN DENSITY N_u T1 (cm⁻²)" if self.fit2t else "COLUMN DENSITY N_u (cm⁻²)"
        lines += _block(t1_label, self.result["T"], ".1f")
        lines += _block(ncol1_label, self.result["Ncol"], ".3e")
        if self.fit2t:
            lines += _block("TEMPERATURE T2 (K)", self.result["T2"], ".1f")
            lines += _block("COLUMN DENSITY N_u T2 (cm⁻²)", self.result["Ncol2"], ".3e")
        lines += _block("LINES USED PER FITTED PIXEL", self.result["n_lines"].astype(float), ".1f")

        n_ceiling = int(np.sum(ok & (self.result["T"] >= 0.99 * t_cap)))
        if self.fit2t:
            n_ceiling += int(np.sum(ok & (self.result["T2"] >= 0.99 * t_cap)))
        lines += ["PIXEL STATUS:", f"  Fitted:     {n_ok:,} ({100 * n_ok / total:.1f}%)"]
        lines.append(f"    of which T pinned at the ceiling (≥ {t_cap:.0f} K): {n_ceiling:,}")
        lines.append(f"  Not fitted: {total - n_ok:,}")
        lines.append(f"    no data: {int(np.sum(status == STATUS_NO_DATA)):,}")
        lines.append(f"    too few lines (< {min_lines} detections): {int(np.sum(status == STATUS_TOO_FEW_LINES)):,}")
        lines.append(f"    fit failed: {int(np.sum(status == STATUS_FIT_FAILED)):,}")

        dlg = QDialog(self)
        dlg.setWindowTitle("Rotational Diagram Map Statistics")
        dlg.resize(520, 520)
        v = QVBoxLayout(dlg)
        body = QLabel("\n".join(lines))
        body.setStyleSheet("font-family: monospace; white-space: pre;")
        body.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        v.addWidget(body)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        v.addWidget(close_btn)
        dlg.exec_()

    def _export_fits(self):
        """
        Save the six maps to a multi-extension FITS file.

        Args:
            None

        Returns:
            None
        """
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Rotational Diagram Map",
            f"{self.result.get('molecule', 'H2')}_rotdiag_map.fits",
            "FITS files (*.fits);;All files (*.*)")
        if not path:
            return
        try:
            save_rotational_diagram_map_fits(self.result, path)
            QMessageBox.information(self, "Saved", f"Wrote:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))


class SelectCubeRegionDialog(QDialog, RegionSelectionMixin):
    """
    Draw a region on a line's Calculate Maps flux map, reusing
    RegionSelectionMixin/SlicePlotter -- the same engine the main window's
    own panels run on. Upper-limit pixels show their fitted flux.

    Args:
        entries (list): One dict per cube-fit line: 'label' (str), 'file' (str,
            the source cube's file name), 'wavelength'
            (float, um), 'header' (astropy.io.fits.Header), 'fit_flux',
            'status', 'ceiling' (2D arrays: fitted flux incl. upper limits,
            NATIVE_PX_* codes, n-sigma flux ceiling).
        n_sigma (float): Detection threshold the maps were calculated with.
        parent (QWidget): Parent widget.

    Returns:
        None
    """

    _SHAPES = ["*", "▭", "○", "⬭", "✏"]
    _SHAPE_TOOLTIPS = {
        "*": "Point selection", "▭": "Rectangle selection", "○": "Circle selection",
        "⬭": "Ellipse selection", "✏": "Freehand drawing",
    }

    def __init__(self, entries, n_sigma, parent=None):
        """Store the line entries and build the drawing UI.

        Args:
            entries (list): Per-line dicts, see the class docstring.
            n_sigma (float): Detection threshold the maps were calculated with.
            parent (QWidget): Parent widget.

        Returns:
            None
        """
        super().__init__(parent)
        self.entries = entries
        self.n_sigma = n_sigma
        self.result_region = None
        self._region_sky = None
        self.current_entry = None

        # RegionSelectionMixin's required state.
        self.region_state = {"left": "idle", "right": "idle"}
        self.active_shape = None
        self.active_target = None
        self.last_selection_left = None
        self.last_selection_right = None
        self.scale_left = "Linear"
        self.slice1_vmin = None
        self.slice1_vmax = None

        self.setWindowTitle("Select Region")
        self.resize(900, 650)
        self._setup_ui()

    def _setup_ui(self):
        """Build the line picker, the mixin-driven map panel (stats, hover,
        shape tools, Edit Shape, Colormap, Scale), and Accept/Cancel.

        Args:
            None

        Returns:
            None
        """
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Line:"))
        self.line_combo = QComboBox()
        for entry in self.entries:
            self.line_combo.addItem(entry["label"])
        top.addWidget(self.line_combo)
        top.addStretch()
        layout.addLayout(top)

        body = QHBoxLayout()

        stats_col = QVBoxLayout()
        self.file_label = QLabel("File: -")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("font-weight: bold;")
        stats_col.addWidget(self.file_label)
        stats_col.addSpacing(12)
        self.stats1_label = QLabel(self._region_stats_placeholder("left"))
        self.stats1_label.setWordWrap(True)
        self.hover1_label = QLabel("Cursor Info:\n-")
        self.hover1_label.setWordWrap(True)
        stats_col.addWidget(self.stats1_label)
        stats_col.addSpacing(12)
        stats_col.addWidget(self.hover1_label)
        stats_col.addStretch()
        body.addLayout(stats_col, 1)

        slice_col = QVBoxLayout()
        self.fig1 = Figure(figsize=(6, 5.5))
        self.slice1_canvas = FigureCanvasQTAgg(self.fig1)
        self.slice1_plotter = SlicePlotter(self.fig1)
        slice_col.addWidget(self.slice1_canvas)
        slice_col.addWidget(NavigationToolbar2QT(self.slice1_canvas, self))
        self.slice1_canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.slice1_canvas.mpl_connect("motion_notify_event", self._on_canvas_hover)
        self.region_selector_left = RegionSelector(
            self.slice1_plotter.ax, self.slice1_canvas,
            lambda mask, patch, coords: self._handle_region_select(mask, patch, coords, "left"))
        body.addLayout(slice_col, 4)

        sidebar = self._build_sidebar()
        body.addLayout(sidebar, 0)

        layout.addLayout(body)

        btns = QHBoxLayout()
        btns.addStretch()
        self.accept_btn = QPushButton("Accept")
        self.accept_btn.setEnabled(False)
        self.accept_btn.clicked.connect(self._accept_region)
        btns.addWidget(self.accept_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

        self._build_right_stand_in()
        self.line_combo.currentIndexChanged.connect(self._on_line_changed)
        self._on_line_changed()

    def _build_sidebar(self):
        """Build the left sidebar exactly like gui.py's own
        _init_shape_controls left column: Colormap, Scale, the 5 shape
        tools, Copy (disabled), Edit Shape, Erase.

        Args:
            None

        Returns:
            sidebar (QVBoxLayout): The populated sidebar layout.
        """
        sidebar = QVBoxLayout()

        self.color_btn_left = QPushButton("🌈")
        self.color_btn_left.setFixedSize(35, 35)
        self.color_btn_left.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.color_btn_left.setToolTip("Change color scale")
        self.color_btn_left.clicked.connect(self._open_colormap_dialog)
        sidebar.addWidget(self.color_btn_left)

        self.scale_btn_left = QPushButton("🎚")
        self.scale_btn_left.setFixedSize(35, 35)
        self.scale_btn_left.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.scale_btn_left.setToolTip("Intensity scaling (Linear, Auto range)")
        self.scale_btn_left.clicked.connect(self._open_scale_dialog)
        sidebar.addWidget(self.scale_btn_left)

        self.shape_buttons = {"left": {}, "right": {}}
        for shape in self._SHAPES:
            btn = QPushButton(shape)
            btn.setCheckable(True)
            btn.setFixedSize(35, 35)
            btn.setStyleSheet("QPushButton { font-size: 18pt; }")
            btn.setToolTip(self._SHAPE_TOOLTIPS[shape])
            btn.clicked.connect(lambda _checked, s=shape: self._set_shape(s, "left"))
            self.shape_buttons["left"][shape] = btn
            sidebar.addWidget(btn)
            self.shape_buttons["right"][shape] = QPushButton(shape)  # never shown

        self.copy_menu_btn_l = QPushButton("⎘")
        self.copy_menu_btn_l.setFixedSize(35, 35)
        self.copy_menu_btn_l.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.copy_menu_btn_l.setToolTip("Copy -- not applicable with a single panel")
        self.copy_menu_btn_l.setEnabled(False)
        sidebar.addWidget(self.copy_menu_btn_l)

        self.edit_shape_btn_l = QPushButton("✎")
        self.edit_shape_btn_l.setCheckable(True)
        self.edit_shape_btn_l.setFixedSize(35, 35)
        self.edit_shape_btn_l.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.edit_shape_btn_l.setToolTip("Toggle Edit Shape (drag/resize)")
        self.edit_shape_btn_l.setEnabled(False)
        self.edit_shape_btn_l.clicked.connect(lambda: self._toggle_edit_shape("left"))
        sidebar.addWidget(self.edit_shape_btn_l)
        self.edit_shape_btn_r = QPushButton("✎")  # never shown

        self.erase_btn_l = QPushButton("✖")
        self.erase_btn_l.setFixedSize(35, 35)
        self.erase_btn_l.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.erase_btn_l.setToolTip("Clear selection")
        self.erase_btn_l.clicked.connect(lambda: self.clear_region("left"))
        sidebar.addWidget(self.erase_btn_l)

        sidebar.addStretch()
        return sidebar

    def _build_right_stand_in(self):
        """Build the inert "right" panel RegionSelectionMixin's dual-panel
        contract requires -- never added to a layout, never shown, never
        wired to a click/hover handler, so it's never a real `target`.

        Args:
            None

        Returns:
            None
        """
        self.fig2 = Figure()
        self.slice2_canvas = FigureCanvasQTAgg(self.fig2)
        self.slice2_plotter = SlicePlotter(self.fig2)
        self.stats2_label = QLabel()
        self.region_selector_right = RegionSelector(
            self.slice2_plotter.ax, self.slice2_canvas, lambda *args: None)

    # ------------------------------------------------------- mixin hooks
    def _region_image(self, target):
        """2D array shown in `target` -- the current line's fitted flux map
        (upper limits included) for "left"; always None for "right" (never
        shown). NaN where no fit succeeded.

        Args:
            target (str): 'left' or 'right'.

        Returns:
            image (numpy.ndarray or None)
        """
        if target != "left" or self.current_entry is None:
            return None
        return np.asarray(self.current_entry["fit_flux"], dtype=float)

    def _region_flux_unit(self, target):
        """Unit of the Calculate Maps flux: always CGS integrated flux.

        Args:
            target (str): 'left' or 'right'.

        Returns:
            unit (str)
        """
        if target != "left" or self.current_entry is None:
            return ""
        return "erg/s/cm2"

    def _map_stats_text(self):
        """Whole-map stats in the same layout as the 3D window's slice stats: the
        data's own min/max/mean/sum/std (upper-limit pixels included), then the
        color range the plot uses.

        Args:
            None

        Returns:
            text (str): Stats label text, or the blank placeholder if no map is shown.
        """
        image = self._region_image("left")
        if image is None:
            return "Intensity:\n-"
        mn, mx, mean = np.nanmin(image), np.nanmax(image), np.nanmean(image)
        total_sum, std = np.nansum(image), np.nanstd(image)
        view_vmin = mn if self.slice1_vmin is None else self.slice1_vmin
        view_vmax = mx if self.slice1_vmax is None else self.slice1_vmax
        return (
            f"Intensity (data)\n[{self._region_flux_unit('left')}]:\n"
            f"min={mn:.3e}\n"
            f"max={mx:.3e}\n"
            f"mean={mean:.3e}\n"
            f"sum={total_sum:.3e}\n"
            f"std={std:.3e}\n"
            f"\n"
            f"Scaling: {self.scale_left}\n"
            f"vmin={view_vmin:.3e}\n"
            f"vmax={view_vmax:.3e}\n"
        )

    def _region_stats_placeholder(self, target):
        """Text the stats label reverts to when a selection is cleared: the
        whole-map stats, like the 3D window restores its whole-slice stats.

        Args:
            target (str): 'left' or 'right'.

        Returns:
            text (str): Stats label text.
        """
        return self._map_stats_text() if target == "left" else "Intensity:\n-"

    def _region_slice_stat_keep(self, line):
        """Keep the whole-map Scaling/vmin/vmax lines above region stats.

        Args:
            line (str): One line of the current stats label text.

        Returns:
            keep (bool): True for a Scaling/vmin/vmax line.
        """
        return line.startswith(("Scaling:", "vmin=", "vmax="))

    def _region_pixel_coords(self, target, x, y):
        """(ra_deg, dec_deg) for pixel (x, y) via the current line's cube WCS.

        Args:
            target (str): 'left' or 'right'.
            x (float): Pixel x coordinate.
            y (float): Pixel y coordinate.

        Returns:
            coords (tuple): (ra_deg, dec_deg), or (None, None).
        """
        if target != "left" or self.current_entry is None:
            return None, None
        wcs2d = _make_2d_wcs(self.current_entry["header"])
        if wcs2d is None or not getattr(wcs2d, "has_celestial", False):
            return None, None
        try:
            sky = wcs2d.pixel_to_world(x, y)
            return float(sky.ra.deg), float(sky.dec.deg)
        except Exception:
            return None, None

    def _pixel_status_lines(self, ix, iy):
        """Detection / upper-limit / no-fit lines for one pixel of the current
        line, with the n-sigma ceiling for an upper limit.

        Args:
            ix (int): Pixel x coordinate, already inside the image.
            iy (int): Pixel y coordinate, already inside the image.

        Returns:
            lines (list): Text lines to show.
        """
        code = int(self.current_entry["status"][iy, ix])
        if code == NATIVE_PX_DETECTION:
            return ["Detection"]
        if code == NATIVE_PX_UL:
            ceiling = float(self.current_entry["ceiling"][iy, ix])
            return [f"Upper limit ({self.n_sigma:g}σ)", f"Ceiling={ceiling:.3e}"]
        return ["No fit"]

    def _region_pixel_extra_lines(self, target, x, y):
        """Wavelength and fit status of the clicked pixel.

        Args:
            target (str): 'left' or 'right'.
            x (float): Pixel x coordinate.
            y (float): Pixel y coordinate.

        Returns:
            lines (list)
        """
        if target != "left" or self.current_entry is None:
            return []
        lines = [f"Wavelength: {self.current_entry['wavelength']:.4f} um"]
        return lines + self._pixel_status_lines(int(round(x)), int(round(y)))

    def _region_on_change(self, target, kind):
        """Keep self._region_sky (the sky-coordinate source of truth handed
        back on Accept) in sync with the mixin's own last_selection_left.

        Args:
            target (str): 'left' or 'right'.
            kind (str): 'select', 'pixel', or 'clear'.

        Returns:
            None
        """
        if target != "left":
            return
        if kind == "clear":
            self._region_sky = None
            self.accept_btn.setEnabled(False)
            return
        sel = self.last_selection_left
        if not sel or self.current_entry is None:
            return
        wcs2d = _make_2d_wcs(self.current_entry["header"])
        verts = None
        if sel["shape"] == "✏":
            verts = getattr(sel.get("patch"), "lasso_verts", None)
        try:
            self._region_sky = region_io.region_shape_to_sky(
                sel["shape"], sel["coords"], wcs2d, lasso_verts=verts)
            self.accept_btn.setEnabled(True)
        except Exception as exc:
            QMessageBox.warning(self, "Select Region", str(exc))

    # ------------------------------------------------------- colormap/scale
    def _open_colormap_dialog(self):
        """Open the shared colormap picker and apply the choice.

        Args:
            None

        Returns:
            None
        """
        dialog = ColorMapDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_cmap:
            self.slice1_plotter.set_cmap(dialog.selected_cmap)

    def _open_scale_dialog(self):
        """Open the shared scale/range picker and redraw with the choice.

        Args:
            None

        Returns:
            None
        """
        image = self._region_image("left")
        if image is None:
            return
        auto_vmin, auto_vmax = float(np.nanmin(image)), float(np.nanmax(image))
        dialog = ScaleDialog(self.scale_left, auto_vmin, auto_vmax,
                             vmin=self.slice1_vmin, vmax=self.slice1_vmax, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        self.scale_left = dialog.result_scale
        self.slice1_vmin, self.slice1_vmax = dialog.result_vmin, dialog.result_vmax
        self._draw_map()

    def _draw_map(self):
        """Plot the current line's fitted flux map, upper-limit pixels included.

        Args:
            None

        Returns:
            None
        """
        entry = self.current_entry
        self.slice1_plotter.plot_slice(
            self._region_image("left"), title=f"{entry['label']}  ({self.n_sigma:g}σ)",
            flux_unit=self._region_flux_unit("left"), scale=self.scale_left,
            vmin=self.slice1_vmin, vmax=self.slice1_vmax)
        self.slice1_canvas.draw_idle()

        self.stats1_label.setText(self._map_stats_text())
        range_desc = ("Auto" if self.slice1_vmin is None and self.slice1_vmax is None
                      else f"{self.slice1_vmin:.3e} to {self.slice1_vmax:.3e}")
        self.scale_btn_left.setToolTip(f"Intensity scaling ({self.scale_left}, {range_desc})")

    # ------------------------------------------------------- click/hover
    def _on_canvas_click(self, event):
        """Point ('*') selection: RegionSelector treats this as a no-op, so
        a plain click handler routes it to the mixin's own
        _handle_pixel_click, same division of labor as _on_slice_click.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Click event.

        Returns:
            None
        """
        if (event.inaxes is None or self.active_shape != "*"
                or event.inaxes is not self.slice1_plotter.ax):
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        self._handle_pixel_click(x, y, "left")

    def _on_canvas_hover(self, event):
        """Live cursor readout, mirroring gui.py's own _on_slice_hover.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Motion event.

        Returns:
            None
        """
        if (event.inaxes is not self.slice1_plotter.ax or event.xdata is None
                or event.ydata is None or self.current_entry is None):
            self.hover1_label.setText("Cursor Info:\n-")
            return
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        image = self._region_image("left")
        if image is None or not (0 <= iy < image.shape[0] and 0 <= ix < image.shape[1]):
            self.hover1_label.setText("Cursor Info:\n-")
            return
        text = [f"Pixel: (x={ix}, y={iy})"]
        if np.isfinite(image[iy, ix]):
            text.append(f"Flux={image[iy, ix]:.3e}")
            text.append(f"[{self._region_flux_unit('left')}]")
        text += self._pixel_status_lines(ix, iy)
        self.hover1_label.setText("\n".join(text))

    # ------------------------------------------------------- line switch
    def _on_line_changed(self):
        """Swap to the combo's current line's flux map, and re-place any
        active selection there via the mixin's own
        _recreate_region/_handle_pixel_click -- not a bespoke preview.

        Args:
            None

        Returns:
            None
        """
        idx = self.line_combo.currentIndex()
        if idx < 0:
            return
        entry = self.entries[idx]
        old_region_sky = self._region_sky

        if self.region_selector_left.interactive_patch is not None:
            self.region_selector_left.interactive_patch.disconnect()
            self.region_selector_left.interactive_patch = None
        self.region_selector_left.activate_selector(None)
        for btn in self.shape_buttons["left"].values():
            btn.setChecked(False)
        self.active_shape = None
        self.active_target = None

        self.current_entry = entry
        self.file_label.setText(f"File: {entry['file']}")
        # Another line's flux is on another scale, so a manual range from the last one is void.
        self.slice1_vmin = None
        self.slice1_vmax = None
        image = self._region_image("left")
        self._draw_map()

        self.last_selection_left = None
        self.stats1_label.setText(self._region_stats_placeholder("left"))
        self._set_region_state("left", "idle")

        if old_region_sky is None:
            self.accept_btn.setEnabled(False)
            return
        try:
            ny, nx = image.shape
            wcs2d = _make_2d_wcs(entry["header"])
            placed = region_io.sky_to_region_shape(
                old_region_sky["shape"], old_region_sky["sky_points"],
                old_region_sky["sky_center"], old_region_sky["sky_axis_x"],
                old_region_sky["sky_axis_y"], wcs2d, target_shape=(ny, nx))
            shape = placed["shape"]
            if shape == "*":
                x, y = placed["coords"]
                self._handle_pixel_click(int(round(x)), int(round(y)), "left")
                return
            if shape == "✏":
                verts = placed["verts"]
                xmin, ymin = verts.min(axis=0)
                xmax, ymax = verts.max(axis=0)
                patch_stub = SimpleNamespace(lasso_verts=verts)
            else:
                xmin, xmax, ymin, ymax = placed["coords"]
                patch_stub = None
            mask = mask_from_shape_coords(shape, placed.get("coords"), placed.get("verts"), (ny, nx))
            self.last_selection_left = {
                "shape": shape, "coords": (xmin, xmax, ymin, ymax),
                "mask": mask, "label": "Left Selection", "patch": patch_stub}
            self._recreate_region(int(xmin), int(xmax), int(ymin), int(ymax), mask, "left")
        except Exception:
            # No real sky overlap with this cube -- leave no stale patch.
            # The authoritative error still surfaces from
            # _rot_map_region_mask_for_cube at Calculate Fluxes time.
            pass

    def _accept_region(self):
        """Hand back the active sky region and close.

        Args:
            None

        Returns:
            None
        """
        self.result_region = self._region_sky
        self.accept()


class LineFluxMapDialog(QDialog):
    """
    Single-line flux (AUC) map viewer: one image plus a colormap/scale
    control and FITS export. The Run Map analogue of Run Fit's per-line
    "View" dialog (show_line_fit), but for the per-pixel map that line's
    own collective-Gaussian fit produced instead of one spectrum.

    Args:
        wavelength (float): This line's catalog wavelength (um)
        flux_map (np.ndarray): 2D flux map, CGS (erg/s/cm^2), NaN where that
            line's fit didn't succeed at a pixel
        source_header (astropy.io.fits.Header): Cube header (single-cube map)
            or dst_wcs.to_header() (multi-cube map), for the export WCS. May
            be None if neither is available.
        molecule (str): Species name, for the window title/export filename
        parent (QWidget): Parent widget

    Returns:
        None
    """

    def __init__(self, wavelength, flux_map, source_header, molecule="H2", parent=None,
                error_map=None, identity_header=None, src_filename=None):
        """Store the map and build the viewer UI.

        Args:
            wavelength (float): This line's catalog wavelength (um).
            flux_map (np.ndarray): 2D flux map, CGS (erg/s/cm^2).
            source_header (astropy.io.fits.Header): Header for the export WCS, or None.
            molecule (str): Species name.
            parent (QWidget): Parent widget.
            error_map (np.ndarray): Per-pixel error, for export's ERR
                extension -- None if unavailable (e.g. the 2D viewer).
            identity_header: Real cube header, for export's INSTRUME/TARGNAME/
                etc when source_header is a bare common-grid WCS -- None
                trusts source_header as-is.
            src_filename (str): Originating cube's filename, for export's
                SRCFILE -- None omits it.

        Returns:
            None
        """
        super().__init__(parent)
        self.wavelength = wavelength
        self.flux_map = flux_map
        self.source_header = source_header
        self.molecule = molecule
        self.error_map = error_map
        self.identity_header = identity_header
        self.src_filename = src_filename
        self.setWindowTitle(f"{molecule} Flux Map @ {wavelength:.6f} μm")
        self.resize(750, 620)
        self._setup_ui()

    def _setup_ui(self):
        """
        Build the colormap/scale controls, canvas, and buttons.

        Args:
            None

        Returns:
            None
        """
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["inferno", "viridis", "plasma", "magma", "cividis", "coolwarm"])
        self.cmap_combo.currentIndexChanged.connect(self._update_plot)
        ctrl.addWidget(self.cmap_combo)

        ctrl.addWidget(QLabel("  Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Linear", "Log", "Percentile"])
        self.scale_combo.currentIndexChanged.connect(self._update_plot)
        ctrl.addWidget(self.scale_combo)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.fig = Figure(figsize=(7, 5.5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

        btns = QHBoxLayout()
        export_btn = QPushButton("Export as FITS")
        export_btn.clicked.connect(self._export_fits)
        btns.addWidget(export_btn)
        stats_btn = QPushButton("Show Statistics")
        stats_btn.clicked.connect(self._show_statistics)
        btns.addWidget(stats_btn)
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        self._update_plot()

    def _update_plot(self):
        """
        Redraw the map with the chosen colormap and scale.

        Args:
            None

        Returns:
            None
        """
        data = _apply_scale_2d(np.asarray(self.flux_map, dtype=float), self.scale_combo.currentText())
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        im = ax.imshow(data, origin="lower", cmap=self.cmap_combo.currentText())
        ax.set_title(f"Flux @ {self.wavelength:.6f} um", fontweight="bold")
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")
        self.fig.colorbar(im, ax=ax).set_label("Flux (erg/s/cm2)")
        self.fig.tight_layout()
        self.canvas.draw()

    def _show_statistics(self):
        """
        Show whole-map flux statistics -- same Count/Sum/Mean/Std/Min-Max
        convention as the main window's Flux Map "Show Statistics", with
        Count also showing what fraction of the map has a value.

        Args:
            None

        Returns:
            None
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Statistics")
        dlg.resize(420, 260)
        layout = QVBoxLayout(dlg)

        data = np.asarray(self.flux_map, dtype=float)
        finite = data[np.isfinite(data)]
        pct = 100 * finite.size / data.size if data.size else 0.0
        stats = f"{self.molecule.upper()} FLUX @ {self.wavelength:.6f} um\n" + "=" * 45 + "\n\n"
        if finite.size > 0:
            stats += f"FLUX (erg/s/cm2):\n"
            stats += f"  Count: {finite.size:,}/{data.size:,} ({pct:.1f}%)\n"
            stats += f"  Sum:   {np.sum(finite):.4e}\n"
            stats += f"  Mean:  {np.mean(finite):.4e}\n"
            stats += f"  Std:   {np.std(finite):.4e}\n"
            stats += f"  Min/Max: {np.min(finite):.4e} / {np.max(finite):.4e}\n"
        else:
            stats += "No valid pixels.\n"

        label = QLabel(stats)
        label.setStyleSheet("font-family: monospace; white-space: pre;")
        label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        layout.addWidget(label)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec_()

    def _export_fits(self):
        """
        Save this line's flux map as a 2D FITS file.

        Args:
            None

        Returns:
            None
        """
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Line Flux Map",
            f"{self.molecule}_flux_{self.wavelength:.4f}um.fits",
            "FITS files (*.fits);;All files (*.*)")
        if not path:
            return
        try:
            header, extensions = build_line_map_header(
                self.source_header, self.flux_map.shape, wavelength=self.wavelength,
                molecule=self.molecule, flux_map=self.flux_map, error_map=self.error_map,
                src_filename=self.src_filename, fit_method="Multi-Gaussian fit",
                identity_header=self.identity_header, extra={"BUNIT": "erg/s/cm2"},
            )
            write_fits_product(path, np.asarray(self.flux_map, dtype=np.float32), header,
                               extensions=extensions)
            QMessageBox.information(self, "Saved", f"Wrote:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
