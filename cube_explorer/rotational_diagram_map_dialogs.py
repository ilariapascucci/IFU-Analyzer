"""
Worker and result dialog for the pixel-by-pixel rotational diagram map.

RotDiagMapWorker runs create_rotational_diagram_map off the GUI thread.
RotDiagMapResultsDialog shows the maps and re-fits a pixel's diagram on click.
"""

import traceback

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QWidget, QFileDialog, QMessageBox, QTextEdit,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from .rotational_diagram_map import (
    create_rotational_diagram_map, create_rotational_diagram_map_multi,
    save_rotational_diagram_map_fits, fit_pixel, fit_pixel_multi,
    compute_native_line_maps, combine_line_maps, fit_region_multi,
    STATUS_OK, STATUS_NO_DATA, STATUS_TOO_FEW_LINES, STATUS_FIT_FAILED,
)
from .rotational_diagram_h2_fitter import model_1T
from .fits_export import build_2d_export_header, write_fits_product
from .line_catalog import dedupe_colocated_lines as _dedupe_colocated

# Display combo: label -> (result key, colour-bar label). "Fit mask" is binary.
_PARAMS = [
    ("Temperature", "T", "T (K)"),
    ("Temperature error", "T_err", "dT (K)"),
    ("Column density", "Ncol", "N_u (cm⁻²)"),
    ("Column density error", "Ncol_err", "dN_u (cm⁻²)"),
    ("Lines used", "n_lines", "detections"),
    ("Fit mask", "status", "1 = fitted"),
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

    def __init__(self, cube_or_list=None, line_table=None, cubes_and_lines=None, **kwargs):
        """Store the fit inputs for run() to consume on the worker thread.

        Args:
            cube_or_list (CubeData): One cube, for the single-cube path.
            line_table (list): Line dicts for the fit.
            cubes_and_lines (list): (CubeData, line_dict) pairs for two or more cubes.
            kwargs (dict): Passed through to whichever create_rotational_diagram_map* function is used.

        Returns:
            None
        """
        super().__init__()
        self.cube_or_list = cube_or_list
        self.line_table = line_table
        self.cubes_and_lines = cubes_and_lines
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
            if self.cubes_and_lines is not None:
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
        kwargs (dict): molecule/isotopologue/opr/fit_opr/opr_grid/dist_pc/
            r_au/min_lines, passed to fit_region_multi and combine_line_maps.
            An n_sigma entry is ignored: the convolved maps carry no threshold

    Returns:
        None
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, convolved_result, cubes_and_lines, **kwargs):
        """Store the inputs for run() to consume on the worker thread.

        Args:
            convolved_result (dict): Output of convolve_line_maps.
            cubes_and_lines (list): (CubeData, line_dict) pairs.
            kwargs (dict): molecule/isotopologue/opr/fit_opr/opr_grid/dist_pc/r_au/min_lines.

        Returns:
            None
        """
        super().__init__()
        self.convolved_result = convolved_result
        self.cubes_and_lines = cubes_and_lines
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
            region_fit = fit_region_multi(
                kept_pairs, molecule=self.kwargs["molecule"], isotopologue=self.kwargs["isotopologue"],
                opr=self.kwargs["opr"], fit_opr=self.kwargs["fit_opr"], opr_grid=self.kwargs["opr_grid"],
                dist_pc=self.kwargs["dist_pc"], r_au=self.kwargs["r_au"], min_lines=self.kwargs["min_lines"],
            )
            result = combine_line_maps(
                self.convolved_result, molecule=self.kwargs["molecule"], isotopologue=self.kwargs["isotopologue"],
                opr=self.kwargs["opr"], fit_opr=self.kwargs["fit_opr"], opr_grid=self.kwargs["opr_grid"],
                dist_pc=self.kwargs["dist_pc"], r_au=self.kwargs["r_au"], min_lines=self.kwargs["min_lines"],
                region_fit=region_fit, progress_callback=self._emit_progress, cancel_flag=self,
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


def _one_t_block(results):
    """
    The "--- 1T FIT ---" block, formatted like Run Fit's show_results.

    Args:
        results (dict): A rotational_diagram result dict with a '1T' entry

    Returns:
        text (str): The OPR line plus the 1T temperature and column density
    """
    r1 = results["1T"]
    return (
        _opr_line(results)
        + "\n--- 1T FIT ---\n"
        + f"T = {r1['T']:.1f} ± {r1['dT']:.1f} K\n"
        + f"Ncol = {r1['Ncol']:.2e} ± {r1['dNcol']:.2e} cm⁻²\n"
    )


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
    return head + _one_t_block(region["results"]) + tail + f"\n{bar}\n"


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
    return head + _one_t_block(out["results"]) + f"\nLines used: {out['n_lines']}\n{bar}\n"


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


def plot_rot_diagram(ax, plot_data):
    """
    Draw one rotational diagram on `ax`, the same style as show_results.

    Args:
        ax (matplotlib.axes.Axes): Target axes
        plot_data (dict): The fitter's '_plot_data' (E_arr, y_arr, y_err, popt1)

    Returns:
        None
    """
    e_arr = np.asarray(plot_data["E_arr"], dtype=float)
    y_arr = np.asarray(plot_data["y_arr"], dtype=float)
    y_err = np.asarray(plot_data["y_err"], dtype=float)
    popt1 = plot_data.get("popt1")

    ax.errorbar(e_arr, y_arr / np.log(10), yerr=y_err / np.log(10), fmt="o",
                color="black", markersize=6, capsize=4, zorder=3, label="Detection")
    if popt1 is not None and len(e_arr) > 1:
        e_fit = np.linspace(np.min(e_arr), np.max(e_arr), 200)
        ax.plot(e_fit, model_1T(e_fit, *popt1) / np.log(10), "--", linewidth=2, label="1T")
        ax.set_xlim(np.min(e_arr) - 500, np.max(e_arr) + 500)
    ax.set_xlabel("E_u (K)", fontsize=12, fontweight="bold")
    ax.set_ylabel("log(N_u / g_u)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="upper right")


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
        self.pixel_select_enabled = True
        self.setWindowTitle(f"{result.get('molecule', 'H2')} Rotational Diagram Map")
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

        self.fig = Figure(figsize=(11, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
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
        Redraw the selected map with the chosen colormap and scale.

        Args:
            None

        Returns:
            None
        """
        label, key, cbar_label = _PARAMS[self.param_combo.currentIndex()]
        status = self.result["status"]

        if key == "status":
            data = (status == STATUS_OK).astype(float)
            cmap, title = "gray", "Fit mask  (white = fitted, black = no fit)"
        else:
            data = np.asarray(self.result[key], dtype=float)
            if key in ("T", "T_err", "Ncol", "Ncol_err"):
                data = np.where(status == STATUS_OK, data, np.nan)
            cmap, title = self.cmap_combo.currentText(), label
            data = self._apply_scale(data)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        im = ax.imshow(data, origin="lower", cmap=cmap)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")
        self.fig.colorbar(im, ax=ax).set_label(cbar_label)
        self.fig.tight_layout()
        self.canvas.draw()

    def _apply_scale(self, data):
        """
        Apply the Scale combo to a 2D array (NaNs preserved).

        Args:
            data (np.ndarray): The map values

        Returns:
            scaled (np.ndarray): Log10, percentile-clipped, or unchanged
        """
        return _apply_scale_2d(data, self.scale_combo.currentText())

    def _on_canvas_click(self, event):
        """
        Re-fit the clicked pixel and show its rotational diagram.

        Args:
            event (matplotlib.backend_bases.MouseEvent): The click

        Returns:
            None
        """
        if not self.pixel_select_enabled or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        ny, nx = self.result["status"].shape
        if 0 <= x < nx and 0 <= y < ny:
            self._show_pixel_diagram(x, y)

    def _show_pixel_diagram(self, x, y):
        """
        Pop a dialog with pixel (x, y)'s diagram and 1T result.

        Args:
            x (int): Spatial x index
            y (int): Spatial y index

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
        text.setText(pixel_result_text(x, y, out))
        lay.addWidget(text, 1)

        if out["status"] == STATUS_OK and "_plot_data" in out["results"]:
            fig = Figure(figsize=(6.5, 5), dpi=100)
            plot_rot_diagram(fig.add_subplot(111), out["results"]["_plot_data"])
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

        lines += _block("TEMPERATURE (K)", self.result["T"], ".1f")
        lines += _block("COLUMN DENSITY N_u (cm⁻²)", self.result["Ncol"], ".3e")
        lines += _block("LINES USED PER FITTED PIXEL", self.result["n_lines"].astype(float), ".1f")

        n_ceiling = int(np.sum(ok & (self.result["T"] >= 0.99 * t_cap)))
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

    def __init__(self, wavelength, flux_map, source_header, molecule="H2", parent=None):
        """Store the map and build the viewer UI.

        Args:
            wavelength (float): This line's catalog wavelength (um).
            flux_map (np.ndarray): 2D flux map, CGS (erg/s/cm^2).
            source_header (astropy.io.fits.Header): Header for the export WCS, or None.
            molecule (str): Species name.
            parent (QWidget): Parent widget.

        Returns:
            None
        """
        super().__init__(parent)
        self.wavelength = wavelength
        self.flux_map = flux_map
        self.source_header = source_header
        self.molecule = molecule
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
            header = build_2d_export_header(
                self.source_header, self.flux_map.shape,
                extra={
                    "BUNIT": "erg/s/cm2",
                    "LINEWAVE": (float(self.wavelength), "Line center wavelength (um)"),
                    "MOLECULE": self.molecule,
                },
            )
            write_fits_product(path, np.asarray(self.flux_map, dtype=np.float32), header)
            QMessageBox.information(self, "Saved", f"Wrote:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
