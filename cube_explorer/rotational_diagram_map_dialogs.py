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
    create_rotational_diagram_map, save_rotational_diagram_map_fits, fit_pixel,
    STATUS_OK, STATUS_NO_DATA, STATUS_TOO_FEW_LINES, STATUS_FIT_FAILED,
    STATUS_UNCONSTRAINED,
)
from .rotational_diagram_h2_fitter import model_1T

STATUS_LABELS = {
    STATUS_OK: "fitted",
    STATUS_NO_DATA: "no data",
    STATUS_TOO_FEW_LINES: "too few lines",
    STATUS_FIT_FAILED: "fit failed",
    STATUS_UNCONSTRAINED: "unconstrained",
}

# param combo label -> (result key, colour-bar label)
_PARAMS = [
    ("Temperature", "T", "T (K)"),
    ("Temperature error", "T_err", "dT (K)"),
    ("Column density", "Ncol", "N_u (cm^-2)"),
    ("Column density error", "Ncol_err", "dN_u (cm^-2)"),
    ("Lines used", "n_lines", "detections"),
    ("Status", "status", "status code"),
]


class RotDiagMapWorker(QThread):
    """
    Background thread that builds the rotational diagram map.

    Args:
        cube_or_list (CubeData or list): One cube, or co-aligned cubes to stitch
        line_table (list): Line dicts for the fit
        kwargs (dict): Passed through to create_rotational_diagram_map

    Returns:
        None
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, cube_or_list, line_table, **kwargs):
        super().__init__()
        self.cube_or_list = cube_or_list
        self.line_table = line_table
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

    y_log10 = y_arr / np.log(10)
    yerr_log10 = y_err / np.log(10)
    ax.errorbar(e_arr, y_log10, yerr=yerr_log10, fmt="o", color="black",
                markersize=6, capsize=4, zorder=3, label="Detection")

    if popt1 is not None and len(e_arr) > 1:
        e_fit = np.linspace(np.min(e_arr), np.max(e_arr), 200)
        ax.plot(e_fit, model_1T(e_fit, *popt1) / np.log(10), "--", linewidth=2,
                label="1T")
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
        Build the info line, map controls, plot canvas, and buttons.

        Args:
            None

        Returns:
            None
        """
        layout = QVBoxLayout(self)

        status = self.result["status"]
        counts = {code: int(np.sum(status == code)) for code in STATUS_LABELS}
        t_ok = self.result["T"][status == STATUS_OK]
        rng = (f"{np.nanmin(t_ok):.0f}-{np.nanmax(t_ok):.0f} K"
               if t_ok.size else "n/a")
        breakdown = "  ".join(f"{STATUS_LABELS[c]}={counts[c]}" for c in STATUS_LABELS)
        r_au = self.result.get("r_au")
        info = QLabel(
            f"Fitted {counts[STATUS_OK]}/{status.size} px   |   T {rng}   |   "
            f"R = {'total-molecules mode' if r_au is None else f'{r_au:.2f} AU'}   |   {breakdown}"
        )
        info.setStyleSheet("background-color: #f0f0f0; padding: 6px; border-radius: 3px;")
        info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(info)

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

        self.pixel_btn = QPushButton("Pixel Select: ON")
        self.pixel_btn.setCheckable(True)
        self.pixel_btn.setChecked(True)
        self.pixel_btn.clicked.connect(self._toggle_pixel_select)
        ctrl.addWidget(self.pixel_btn)
        ctrl.addWidget(QLabel("(click a pixel for its diagram; turn off to zoom/pan)"))
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
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        self._update_plot()

    def _toggle_pixel_select(self):
        """
        Flip pixel-select mode and relabel the button.

        Args:
            None

        Returns:
            None
        """
        self.pixel_select_enabled = self.pixel_btn.isChecked()
        self.pixel_btn.setText(
            "Pixel Select: ON" if self.pixel_select_enabled else "Pixel Select: OFF")

    def _update_plot(self):
        """
        Redraw the selected map.

        Args:
            None

        Returns:
            None
        """
        _label, key, cbar_label = _PARAMS[self.param_combo.currentIndex()]
        data = np.asarray(self.result[key], dtype=float)
        if key in ("T", "T_err", "Ncol", "Ncol_err"):
            data = np.where(self.result["status"] == STATUS_OK, data, np.nan)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        im = ax.imshow(data, origin="lower", cmap=self.cmap_combo.currentText())
        ax.set_title(_label, fontweight="bold")
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")
        self.fig.colorbar(im, ax=ax).set_label(cbar_label)
        self.fig.tight_layout()
        self.canvas.draw()

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
        if not (0 <= x < nx and 0 <= y < ny):
            return
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
        try:
            out = fit_pixel(self.cube_or_list, x, y, self.line_table, **self.fit_kwargs)
        except Exception as exc:
            QMessageBox.critical(self, "Pixel fit failed", str(exc))
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Pixel ({x}, {y}) rotational diagram")
        dlg.resize(900, 520)
        lay = QHBoxLayout(dlg)

        text = QTextEdit()
        text.setReadOnly(True)
        if out["status"] == STATUS_OK:
            r1 = out["results"]["1T"]
            text.setText(
                f"Pixel ({x}, {y})\n"
                f"lines used : {out['n_lines']}\n"
                f"T          : {r1['T']:.1f} +/- {r1['dT']:.1f} K\n"
                f"Ncol       : {r1['Ncol']:.3e} +/- {r1['dNcol']:.3e} cm^-2\n"
                f"red chi2   : {r1.get('red_chi2', float('nan')):.2f}\n"
                f"OPR        : {out['results'].get('OPR', 'n/a')}"
            )
        else:
            text.setText(f"Pixel ({x}, {y})\nno fit: {out.get('reason', 'n/a')}\n"
                         f"lines used: {out.get('n_lines', 0)}")
        lay.addWidget(text, 1)

        if out["status"] == STATUS_OK and "_plot_data" in out["results"]:
            fig = Figure(figsize=(5.5, 4.5), dpi=100)
            plot_rot_diagram(fig.add_subplot(111), out["results"]["_plot_data"])
            canvas = FigureCanvasQTAgg(fig)
            box = QWidget()
            box_lay = QVBoxLayout(box)
            box_lay.addWidget(NavigationToolbar2QT(canvas, dlg))
            box_lay.addWidget(canvas)
            lay.addWidget(box, 1)

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
