"""
Superimposition
Windows that draw contours and colormap images over one base image. Batch Superimposition
reads 2D FITS files that the user picks in the window. Pair Superimposition uses the two
images of the viewer panels and draws on the panel of the base row. Each table row is one
layer, and row 1 is the base image.
"""

import os
from dataclasses import dataclass, field, replace

import matplotlib
import numpy as np
from astropy.io import fits
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QDoubleSpinBox, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
)
from reproject import reproject_exact

from .file_dialogs import open_file_names
from .gui_sizing import fit_to_screen
from .icon_loader import set_button_icon
from .scale_dialog import ScaleDialog, ScaleSettings, build_norm
from .unit_converter import clean_flux_unit, sky_wcs_problem
from .utils import contour_levels, levels_inside_data, make_2d_wcs, rotate_image_with_wcs, select_2d_image
from .visualization import apply_wcs_grid

MODE_CONTOUR = "Contour"
MODE_IMAGE = "Image"
BASE_CMAP = "magma"
SUPERIMPOSITION_CMAP = "hot"
CONTOUR_COLORS = ["red", "white", "gray", "black", "cyan", "yellow", "lime", "magenta", "orange", "blue", "green"]
LINE_STYLES = ["solid", "dashed", "dotted", "dashdot"]
COLORMAPS = [
    "magma", "inferno", "plasma", "viridis", "cividis", "gray", "cubehelix",
    "turbo", "twilight", "twilight_shifted", "Spectral", "RdBu", "coolwarm",
    "hot", "cool", "Blues", "Greens", "Reds",
    "Pure Red", "Pure Green", "Pure Blue",
]
# Black to the full color: (0, 0, 0) to (255, 0, 0), (0, 255, 0) or (0, 0, 255).
# The ColorBrewer maps Reds, Greens and Blues go from near white to a dark shade instead.
for _name, _color in (("Pure Red", (1, 0, 0)), ("Pure Green", (0, 1, 0)), ("Pure Blue", (0, 0, 1))):
    if _name not in matplotlib.colormaps:
        matplotlib.colormaps.register(LinearSegmentedColormap.from_list(_name, [(0, 0, 0), _color]))
CONTOUR_ALPHA = 0.8
COLUMNS = ["#", "Mode", "Source", "Color", "Line Style", "Thickness", "Levels (σ)", "Sigma",
           "Colormap", "Scale", "Alpha", "Rotation", "View", ""]
COL_NUM, COL_MODE, COL_SOURCE, COL_COLOR, COL_STYLE, COL_THICK, COL_LEVELS, COL_SIGMA, \
    COL_CMAP, COL_SCALE, COL_ALPHA, COL_ROTATION, COL_VIEW, COL_DELETE = range(len(COLUMNS))


@dataclass
class SuperimpositionRow:
    """The settings of one superimposed layer.

    Args:
        mode (str): 'Contour' or 'Image'.
        source (str): Path of the FITS file that supplies the data. In Pair Superimposition,
            'left' or 'right' for the panel that supplies the data.
        color (str): Contour line color. Used by Contour only.
        line_style (str): Contour line style. Used by Contour only.
        thickness (float): Contour line width. Used by Contour only.
        levels (str): Comma-separated sigma levels. Used by Contour only.
        sigma (str): Sigma typed by the user, or '' for the automatic value.
        cmap (str): Matplotlib colormap name. Used by Image only.
        scale (ScaleSettings): Scale and color range. Used by Image only.
        alpha (float): Image opacity from 0.1 to 1.0. Used by Image only.
        contour_alpha (float): Contour line opacity from 0.1 to 1.0. Used by Contour only.
        rotation (float): Rotation of the image and its WCS together, in degrees. A positive
            value turns the image clockwise as drawn. It changes the figure only for row 1
            and for a row that View shows alone.
    """

    mode: str = MODE_CONTOUR
    source: str = ""
    color: str = "red"
    line_style: str = "solid"
    thickness: float = 1.5
    levels: str = "3, 5, 10"
    sigma: str = ""
    cmap: str = SUPERIMPOSITION_CMAP
    scale: ScaleSettings = field(default_factory=ScaleSettings)
    alpha: float = 0.5
    contour_alpha: float = CONTOUR_ALPHA
    rotation: float = 0.0


def load_superimposition_image(path):
    """Read the first 2D image of a FITS file, with its header and sky WCS.

    The file must hold an image with two axes and a sky WCS. A cube cannot be used here.
    The image is read into memory, so the file is closed when this function returns.

    Args:
        path (str): Path of the FITS file.

    Returns:
        image (dict): The keys image (numpy.ndarray), wcs (astropy.wcs.WCS),
            header (astropy.io.fits.Header) and filename (str, without the folder).

    Raises:
        ValueError: The file has no 2D image, or the image has no sky WCS.
        OSError: The file cannot be read as FITS.
    """
    name = os.path.basename(path)
    with fits.open(path) as hdul:
        try:
            found = select_2d_image(hdul)
        except ValueError:
            raise ValueError(f"{name} has no 2D image.")
        data = found["image"].astype(float)
        header = found["header"]
    problem = sky_wcs_problem(header)
    if problem is not None:
        raise ValueError(f"{name} has no sky WCS. {problem}")
    wcs = make_2d_wcs(header)
    if wcs is None or not wcs.has_celestial:
        raise ValueError(f"{name} has no sky WCS.")
    return {"image": data, "wcs": wcs, "header": header, "filename": name}


def _base_row(rows):
    """Return the first row, which is the base image, after a check of the rows.

    Args:
        rows (list): SuperimpositionRow objects, from the bottom layer to the top layer.

    Returns:
        base_row (SuperimpositionRow): The first row.

    Raises:
        ValueError: There is no row, or the first row is not an Image row.
    """
    if not rows:
        raise ValueError("There is no row to draw.")
    if rows[0].mode != MODE_IMAGE:
        raise ValueError("Row 1 is the base image. It must be an Image row.")
    return rows[0]


def rotated_source(source, angle):
    """Turn an image and its WCS together, like the Rotate button of the 2D viewer.

    The array grows to hold the turned image, and the new pixels are NaN. The WCS changes
    in the same way, so the sky position of every pixel stays the same. The header is
    copied without change, so only its keywords that are not WCS keywords are valid.

    Args:
        source (dict): Image data with the keys image, wcs, header and filename, as
            load_superimposition_image returns it.
        angle (float): Rotation in degrees. A positive value turns the image clockwise as drawn.

    Returns:
        rotated (dict): The same keys as source. It is source itself when angle is 0.
    """
    if not angle:
        return source
    wcs_in = source["wcs"].celestial if source["wcs"].pixel_n_dim > 2 else source["wcs"]
    array_out, wcs_out = rotate_image_with_wcs(source["image"], wcs_in, angle)
    return {**source, "image": array_out, "wcs": wcs_out}


def _draw_contour_row(ax, source, row, label, **kwargs):
    """Draw the contour lines of one row on axes. A level outside the data has no line.

    The levels come from contour_levels. A level that is not between the lowest and the
    highest pixel is skipped, and the returned note says so.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        source (dict): Image data with the keys image, wcs, header and filename.
        row (SuperimpositionRow): Contour row with the line settings.
        label (str): Name of the row in error messages and notes, for example 'Row 2'.
        **kwargs: More arguments for ax.contour, for example the transform.

    Returns:
        note (str): Message about skipped levels, or None when no level was skipped.
        artist (matplotlib.contour.QuadContourSet): The contour set, or None when no line was drawn.

    Raises:
        ValueError: The levels or the sigma of the row are wrong, or the image has no finite pixel.
    """
    try:
        levels = contour_levels(source["image"], row.levels, row.sigma)
    except ValueError as error:
        raise ValueError(f"{label}: {error}")
    inside, skipped = levels_inside_data(source["image"], levels)
    note = None
    if skipped:
        note =(f"{label}: no contour drawn at {skipped}. "
                "The level is outside the data range of the source image.")
    if not inside:
        return note, None
    artist = ax.contour(source["image"], levels=inside, colors=[row.color],
                        linestyles=row.line_style, linewidths=row.thickness,
                        alpha=row.contour_alpha, **kwargs)
    return note, artist


def draw_layers(ax, images, rows):
    """Draw every row after the first one on axes that already show the base image.

    Contour rows are drawn through the WCS of their source. Image rows are reprojected
    onto the grid of the base image. The first row is not drawn here. The rotation of
    row 1 turns the base grid. The rotation of any other row is not used, and a note says so.

    Args:
        ax (matplotlib.axes.Axes): WCS axes that show the base image in pixel coordinates.
        images (dict): Image data, keyed by the source of a row. Each value is a dict with
            the keys image, wcs, header and filename, as load_superimposition_image returns it.
        rows (list): SuperimpositionRow objects, from the base layer to the top layer.

    Returns:
        notes (list): Messages about rows that drew nothing, as text.
        artists (list): The images and contour sets that were drawn.

    Raises:
        ValueError: There is no row, the first row is not an Image row, a row has wrong
            settings, or an Image row has no sky area in common with the base.
    """
    base_row = _base_row(rows)
    base_panel = rotated_source(images[base_row.source], base_row.rotation)
    notes, artists = [], []
    for number, row in enumerate(rows[1:], start=2):
        source = images[row.source]
        # The base array is used as it is only when row 1 is not rotated.
        is_base = row.source == base_row.source and not base_row.rotation
        if row.rotation:
            notes.append(f"Row {number}: the rotation is not used when the row is drawn over the base image. "
                         "Rotate row 1 to turn the figure.")

        if row.mode == MODE_CONTOUR:
            kwargs = {} if is_base else {"transform": ax.get_transform(source["wcs"])}
            note, artist = _draw_contour_row(ax, source, row, f"Row {number}", **kwargs)
            if note:
                notes.append(note)
            if artist is not None:
                artists.append(artist)
        else:
            if is_base:
                layer = source["image"]
            else:
                layer, footprint = reproject_exact(
                    (source["image"], source["wcs"]), base_panel["wcs"], shape_out=base_panel["image"].shape)
                if np.count_nonzero(footprint) == 0:
                    raise ValueError(f"Row {number}: the source image has no sky area in common with the base image.")
            if not np.isfinite(layer).any():
                raise ValueError(f"Row {number}: the source image has no finite pixel.")
            layer_norm, layer_shown = build_norm(layer, row.scale)
            artists.append(ax.imshow(layer_shown, origin="lower", cmap=row.cmap, norm=layer_norm,
                                     alpha=row.alpha, interpolation="nearest", aspect="equal"))
    return notes, artists


def _draw_contour_alone(fig, images, row):
    """Draw the contour lines of one row on an empty frame. The frame is the grid of its own file.

    No other image is drawn, so the figure shows only the file of the row, turned by the
    rotation of the row. The levels come from the turned image.

    Args:
        fig (matplotlib.figure.Figure): Figure to draw on. It is cleared first.
        images (dict): Image data, keyed by the source of a row, as draw_superimposition reads it.
        row (SuperimpositionRow): Contour row to draw.

    Returns:
        notes (list): Message about skipped levels as text, or an empty list.

    Raises:
        ValueError: The levels or the sigma of the row are wrong, or the image has no finite pixel.
    """
    source = rotated_source(images[row.source], row.rotation)
    ny, nx = source["image"].shape
    fig.clear()
    ax = fig.add_subplot(111, projection=source["wcs"])
    apply_wcs_grid(ax, source["image"].shape, grid=False)
    note, _ = _draw_contour_row(ax, source, row, "This row")
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_aspect("equal")
    ax.set_title(source["filename"], fontsize=9)
    return [note] if note else []


def draw_superimposition(fig, images, rows):
    """Draw the rows on a new figure. The first row is the base image, and each next row goes on top.

    The base row sets the sky grid of the figure, with its own colormap, scale and rotation.
    The other rows are drawn by draw_layers. One Contour row alone is a special case. It is
    drawn without a base image, on the grid of its own file. The figure has no grid lines.

    Args:
        fig (matplotlib.figure.Figure): Figure to draw on. It is cleared first.
        images (dict): Image data, keyed by the source of a row. Each value is a dict with
            the keys image, wcs, header and filename, as load_superimposition_image returns it.
        rows (list): SuperimpositionRow objects, from the bottom layer to the top layer.

    Returns:
        notes (list): Messages about rows that drew nothing, as text.

    Raises:
        ValueError: There is no row, the first row is not an Image row (unless it is the
            only row), a row has wrong settings, or an Image row has no sky area in common
            with the base.
    """
    if len(rows) == 1 and rows[0].mode == MODE_CONTOUR:
        return _draw_contour_alone(fig, images, rows[0])
    base_row = _base_row(rows)
    base_panel = rotated_source(images[base_row.source], base_row.rotation)

    fig.clear()
    ax = fig.add_subplot(111, projection=base_panel["wcs"])
    norm, shown = build_norm(base_panel["image"], base_row.scale)
    base_im = ax.imshow(shown, origin="lower", cmap=base_row.cmap, norm=norm,
                        interpolation="nearest", aspect="equal")
    apply_wcs_grid(ax, base_panel["image"].shape, grid=False)
    cbar = fig.colorbar(base_im, ax=ax, fraction=0.046, pad=0.04)
    header = base_panel["header"]
    unit_text = clean_flux_unit(header["BUNIT"]) if header is not None and "BUNIT" in header else ""
    cbar.set_label(f"Flux / Intensity\n[{unit_text}]" if unit_text else "Flux / Intensity")

    notes, _ = draw_layers(ax, images, rows)
    ax.set_title(base_panel["filename"] if len(rows) == 1 else f"Base: {base_panel['filename']}", fontsize=9)
    return notes


class SuperimpositionResultWindow(QWidget):
    """Window that shows one superimposition figure. The toolbar saves it as an image."""

    def __init__(self, title, images, rows, parent=None):
        """Draw the rows and show them in a window with a toolbar.

        Args:
            title (str): Window title.
            images (dict): Image data, as draw_superimposition reads it.
            rows (list): SuperimpositionRow objects to draw. The first one is the base image.
            parent (QWidget): Parent widget, or None.

        Returns:
            None

        Raises:
            ValueError: The rows cannot be drawn. The message says why.
        """
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(title)
        self.fig = Figure(figsize=(7, 6))
        self.notes = draw_superimposition(self.fig, images, rows)
        canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.addWidget(NavigationToolbar(canvas, self))
        layout.addWidget(canvas)
        fit_to_screen(self, 760, 700)
        canvas.draw()


class SuperimpositionWindow(QDialog):
    """Batch Superimposition: a table of layers from the base image to the top layer, with files
    that the user picks, a View link per row, and Produce, which shows the result in a new window."""

    TITLE = "Batch Superimposition"
    INFO = ("+ Add FITS Files selects one or more 2D FITS files. Each file becomes an Image row. Row 1 is the base image. "
            "Each next row goes on top of the rows above it. Drag the = of a row, or type a new number in the # column, to change the order. "
            "Change the Mode of a row to Contour to draw lines at sigma levels of its file instead "
            "of a colormap. Alpha is the opacity of a layer. Rotation turns the image and its WCS together: "
            "in row 1 it turns the whole figure, and in other rows it changes only View. "
            "Double-click View to show one row alone. The trash button removes "
            "a row. Produce shows all rows together.")
    BASE_IS_PANEL = False

    def __init__(self, parent=None):
        """Build the window with an empty table. The window reads no image from the viewer.

        Args:
            parent (QWidget): Parent widget, or None.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle(self.TITLE)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.images = {}
        self.rows = []
        self.result_windows = []

        layout = QVBoxLayout(self)
        info = QLabel(self.INFO)
        info.setWordWrap(True)
        layout.addWidget(info)

        # Sorting stays off. The rows are layers, and their order is the drawing order.
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSortingEnabled(False)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(COL_SOURCE, QHeaderView.Stretch)
        # The same cell padding as the table of the rotational diagram window.
        self.table.setStyleSheet(
            "QTableWidget::item { padding: 2px 6px; } "
            "QHeaderView::section { padding: 2px 6px; }"
        )
        self.table.itemDoubleClicked.connect(self._on_item_double_clicked)
        # A cell widget takes the mouse, so a row moves by its number in the left header.
        row_header = self.table.verticalHeader()
        row_header.setSectionsMovable(True)
        row_header.setToolTip("Drag the = to move the row, or type a new number in the # column.")
        row_header.sectionMoved.connect(lambda *_: QTimer.singleShot(0, self._apply_row_order))
        layout.addWidget(self.table, 1)

        bottom = QHBoxLayout()
        self.add_row_btn = QPushButton("+ Add FITS Files")
        self.add_row_btn.setToolTip("Select one or more 2D FITS files. Each file becomes an Image row at the bottom.")
        self.add_row_btn.clicked.connect(self._add_row)
        bottom.addWidget(self.add_row_btn)
        bottom.addStretch(1)
        self.produce_btn = QPushButton("Produce")
        self.produce_btn.setToolTip("Draw all rows in one window, from the base image to the top row.")
        self.produce_btn.clicked.connect(self._produce)
        bottom.addWidget(self.produce_btn)
        layout.addLayout(bottom)

        fit_to_screen(self, 1350, 420)

    def _add_row(self):
        """Ask for FITS files and add one Image row at the bottom for each file.
        The same file can have more than one row. A file that cannot be read is reported
        by name, and the other files are still added.

        Args:
            None

        Returns:
            None
        """
        paths, _ = open_file_names(self, "Select 2D FITS Files",
                                                filter="FITS files (*.fits *.fz *.fit)")
        failed = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for path in paths:
                if path not in self.images:
                    try:
                        self.images[path] = load_superimposition_image(path)
                    except (ValueError, OSError) as error:
                        failed.append(f"{os.path.basename(path)}: {error}")
                        continue
                self.rows.append(SuperimpositionRow(mode=MODE_IMAGE, source=path,
                                                    cmap=SUPERIMPOSITION_CMAP if self.rows else BASE_CMAP))
        finally:
            QApplication.restoreOverrideCursor()
        self._rebuild_table()
        if failed:
            QMessageBox.warning(self, "Files Not Added", "These files were not added:\n" + "\n".join(failed))

    def _source_label(self, path):
        """Make the text that names a file in the Source list. It is the file name. If two loaded
        files have the same name, the name of the folder is added.

        Args:
            path (str): Path of a loaded file.

        Returns:
            label (str): The text for the list.
        """
        name = self.images[path]["filename"]
        if sum(1 for item in self.images.values() if item["filename"] == name) > 1:
            return f"{name} ({os.path.basename(os.path.dirname(path))})"
        return name

    def _apply_row_order(self):
        """Put the rows in the order that the left header shows after a drag. Then build the table
        again, so the # column shows the row numbers in order.

        Args:
            None

        Returns:
            None
        """
        header = self.table.verticalHeader()
        count = len(self.rows)
        order = [header.logicalIndex(position) for position in range(count)]
        if order == list(range(count)):
            return
        header.blockSignals(True)
        for position in range(count):
            header.moveSection(header.visualIndex(position), position)
        header.blockSignals(False)
        self._set_row_order([self.rows[index] for index in order])

    def _set_row_order(self, new_rows):
        """Use a new order of the rows, then build the table again. A Contour row cannot be row 1,
        so the old order stays when it would be.

        Args:
            new_rows (list): The same SuperimpositionRow objects in the new order.

        Returns:
            None
        """
        if new_rows[0].mode != MODE_IMAGE:
            QMessageBox.warning(self, "Base Image", "Row 1 is the base image. A Contour row cannot be the base.")
        else:
            self.rows = new_rows
        self._rebuild_table()

    def _move_row(self, row, position):
        """Move one row to a number that the user typed in the # column.

        Args:
            row (SuperimpositionRow): Row to move.
            position (int): New row number, from 1.

        Returns:
            None
        """
        if row not in self.rows or self.rows.index(row) == position - 1:
            return
        new_rows = [item for item in self.rows if item is not row]
        new_rows.insert(position - 1, row)
        self._set_row_order(new_rows)

    def _purge_unused_images(self):
        """Remove the files that no row uses from the memory.

        Args:
            None

        Returns:
            None
        """
        used = {row.source for row in self.rows}
        self.images = {path: item for path, item in self.images.items() if path in used}

    def _delete_row(self, row):
        """Remove one row. A file that no row uses any more leaves the memory.
        Row 1 is not removed when row 2 is a Contour row, because that row cannot be the base.

        Args:
            row (SuperimpositionRow): Row to remove.

        Returns:
            None
        """
        if self.rows.index(row) == 0 and len(self.rows) > 1 and self.rows[1].mode != MODE_IMAGE:
            QMessageBox.warning(self, "Base Image",
                                "Row 2 is a Contour row, and it cannot be the base image. "
                                "Change row 2 to Image, or remove it first.")
            return
        self.rows.remove(row)
        self._purge_unused_images()
        self._rebuild_table()

    def _rebuild_table(self):
        """Build every cell again from the row settings. A cell that a mode does not use shows a dash.

        Args:
            None

        Returns:
            None
        """
        self.table.clearContents()
        self.table.setRowCount(len(self.rows))
        self.table.setVerticalHeaderLabels(["="] * len(self.rows))
        for index, row in enumerate(self.rows):
            is_contour = row.mode == MODE_CONTOUR
            number = QSpinBox()
            number.setRange(1, len(self.rows))
            number.setValue(index + 1)
            number.setAlignment(Qt.AlignCenter)
            number.setButtonSymbols(QSpinBox.NoButtons)
            number.setMinimumWidth(36)
            # The row moves when the user presses Enter or leaves the box, not at each typed digit.
            number.setKeyboardTracking(False)
            number.setToolTip("Row number. Type a new number to move the row." if index
                              else "Row 1 is the base image. Type a new number to move the row.")
            number.valueChanged.connect(
                lambda value, r=row: QTimer.singleShot(0, lambda: self._move_row(r, value)))
            self.table.setCellWidget(index, COL_NUM, number)
            # The base image is always an Image row, so row 1 has no Contour choice.
            modes = [MODE_IMAGE] if index == 0 else [MODE_CONTOUR, MODE_IMAGE]
            self._set_combo(index, COL_MODE, modes, row.mode, self._on_mode_changed, row)
            source = QTableWidgetItem(self._source_label(row.source))
            source.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            source.setToolTip(row.source)
            self.table.setItem(index, COL_SOURCE, source)
            if is_contour:
                self._set_combo(index, COL_COLOR, CONTOUR_COLORS, row.color,
                                lambda v, r=row: setattr(r, "color", v))
                self._set_combo(index, COL_STYLE, LINE_STYLES, row.line_style,
                                lambda v, r=row: setattr(r, "line_style", v))
                thickness = QDoubleSpinBox()
                thickness.setRange(0.5, 6.0)
                thickness.setSingleStep(0.5)
                thickness.setValue(row.thickness)
                thickness.setAlignment(Qt.AlignCenter)
                thickness.valueChanged.connect(lambda v, r=row: setattr(r, "thickness", v))
                self.table.setCellWidget(index, COL_THICK, thickness)
                levels = QLineEdit(row.levels)
                levels.setAlignment(Qt.AlignCenter)
                levels.setToolTip("Separated by commas. A number N draws a line at the median + N × sigma. "
                                  "A number with %, for example 50%, draws a line at that percent of the peak.")
                levels.textChanged.connect(lambda v, r=row: setattr(r, "levels", v))
                self.table.setCellWidget(index, COL_LEVELS, levels)
                sigma = QLineEdit(row.sigma)
                sigma.setAlignment(Qt.AlignCenter)
                sigma.setPlaceholderText("auto")
                sigma.setToolTip("Sigma in the units of the source image. Empty means the scatter of the whole "
                                 "source image (sigma-clipped), which is not the noise of the image.")
                sigma.textChanged.connect(lambda v, r=row: setattr(r, "sigma", v))
                self.table.setCellWidget(index, COL_SIGMA, sigma)
            elif index > 0 or not self.BASE_IS_PANEL:
                self._set_combo(index, COL_CMAP, COLORMAPS, row.cmap, lambda v, r=row: setattr(r, "cmap", v))
                scale_btn = QPushButton(row.scale.scale_label())
                scale_btn.setToolTip("Change the scale and the color range.")
                scale_btn.clicked.connect(lambda _checked=False, r=row, b=scale_btn: self._edit_scale(r, b))
                self.table.setCellWidget(index, COL_SCALE, scale_btn)
            # The base row is opaque. Both modes have an opacity, and each mode keeps its own value.
            if index > 0:
                alpha_name = "contour_alpha" if is_contour else "alpha"
                alpha = QDoubleSpinBox()
                alpha.setRange(0.1, 1.0)
                alpha.setSingleStep(0.1)
                alpha.setValue(getattr(row, alpha_name))
                alpha.setAlignment(Qt.AlignCenter)
                alpha.setMinimumWidth(70)
                alpha.setToolTip("Opacity of this layer. 1.0 is fully opaque. A lower value shows the layers below.")
                alpha.valueChanged.connect(lambda v, r=row, n=alpha_name: setattr(r, n, v))
                self.table.setCellWidget(index, COL_ALPHA, alpha)
            # The base panel of Pair Superimposition has its own Rotate button in the viewer.
            if not (self.BASE_IS_PANEL and index == 0):
                rotation = QDoubleSpinBox()
                rotation.setRange(-360.0, 360.0)
                rotation.setSingleStep(5.0)
                rotation.setDecimals(1)
                rotation.setSuffix(" °")
                rotation.setValue(row.rotation)
                rotation.setAlignment(Qt.AlignCenter)
                rotation.setMinimumWidth(80)
                rotation.setToolTip("Turns the image and its WCS together. A positive value is clockwise. "
                                    "Rotation of row 1 turns the whole figure. For any other row, "
                                    "it only changes View, which shows that row alone.")
                rotation.valueChanged.connect(lambda v, r=row: setattr(r, "rotation", v))
                self.table.setCellWidget(index, COL_ROTATION, rotation)
            # The same blue underlined "View" cell as the rotational diagram table.
            view_item = QTableWidgetItem("View")
            view_item.setTextAlignment(Qt.AlignCenter)
            view_item.setForeground(QColor(0, 0, 238))
            font = view_item.font()
            font.setUnderline(True)
            view_item.setFont(font)
            view_item.setToolTip("Double-click to draw this row alone, without the other rows.")
            self.table.setItem(index, COL_VIEW, view_item)
            trash_btn = QPushButton()
            set_button_icon(trash_btn, "trash", size=18)
            trash_btn.setFlat(True)
            trash_btn.setToolTip("Remove this row.")
            trash_btn.clicked.connect(lambda _checked=False, r=row: self._delete_row(r))
            self.table.setCellWidget(index, COL_DELETE, trash_btn)
            # A cell that this row does not use gets a dash, so it does not look like a missing control.
            for column in range(COL_COLOR, COL_ROTATION + 1):
                if self.table.cellWidget(index, column) is None:
                    unused = QTableWidgetItem("-")
                    unused.setTextAlignment(Qt.AlignCenter)
                    unused.setFlags(Qt.NoItemFlags)
                    self.table.setItem(index, column, unused)
        self.table.resizeColumnsToContents()
        # A cell widget sits 6 px inside its cell on each side, so its column needs 12 px more than the widget.
        for column in range(len(COLUMNS)):
            if any(self.table.cellWidget(r, column) is not None for r in range(len(self.rows))):
                self.table.setColumnWidth(column, self.table.columnWidth(column) + 12)

    def _on_item_double_clicked(self, item):
        """Draw the row of a View cell after a double-click on it.

        Args:
            item (QTableWidgetItem): The cell that the user double-clicked.

        Returns:
            None
        """
        if item.column() == COL_VIEW and 0 <= item.row() < len(self.rows):
            self._view(self.rows[item.row()])

    def _set_combo(self, index, column, items, current, on_change, row=None):
        """Put a drop-down list in one cell.

        Args:
            index (int): Row number of the cell.
            column (int): Column number of the cell.
            items (list): Texts of the list.
            current (str): Text that is selected at first.
            on_change (callable): Called with the new text when the choice changes.
            row (SuperimpositionRow): Row that on_change also gets, or None when it needs no row.

        Returns:
            None
        """
        combo = QComboBox()
        combo.addItems(items)
        combo.setCurrentText(current)
        if row is None:
            combo.currentTextChanged.connect(on_change)
        else:
            combo.currentTextChanged.connect(lambda text, r=row: on_change(r, text))
        self.table.setCellWidget(index, column, combo)
        # The list of a drop-down is as wide as the drop-down, so a narrow one cuts the longest item.
        # The table gives the drop-down its font, so the width is measured after the drop-down is in the table.
        longest = max(combo.fontMetrics().horizontalAdvance(item) for item in items)
        combo.setMinimumWidth(longest + 40)

    def _on_mode_changed(self, row, mode):
        """Store a new mode and build the table again after the signal has ended.
        The drop-down list that sent the signal is deleted by the rebuild, so the rebuild waits.

        Args:
            row (SuperimpositionRow): Row whose mode changed.
            mode (str): The new mode.

        Returns:
            None
        """
        row.mode = mode
        QTimer.singleShot(0, self._rebuild_table)

    def _edit_scale(self, row, button):
        """Open the scale dialog for an Image row and store its result.

        Args:
            row (SuperimpositionRow): Row to change.
            button (QPushButton): Scale cell button. Its text shows the new scale.

        Returns:
            None
        """
        dialog = ScaleDialog(row.scale, self.images[row.source]["image"], parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        row.scale = dialog.result_settings
        button.setText(row.scale.scale_label())

    def _show_result(self, title, rows):
        """Draw rows in a new window. Show an error message if that fails.

        Args:
            title (str): Title of the new window.
            rows (list): SuperimpositionRow objects to draw. The first one is the base image.

        Returns:
            None
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            window = SuperimpositionResultWindow(title, self.images, rows)
        except ValueError as error:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Superimposition Error", str(error))
            return
        except Exception as error:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Superimposition Error", f"Could not draw the superimposition:\n{error}")
            raise
        QApplication.restoreOverrideCursor()
        self.result_windows.append(window)
        window.show()
        if window.notes:
            QMessageBox.information(self, "Superimposition Note", "\n".join(window.notes))

    def _view(self, row):
        """Show one row alone, in a new window. No other row and no base image is drawn.

        Args:
            row (SuperimpositionRow): Row to draw.

        Returns:
            None
        """
        self._show_result(f"Superimposition: row {self.rows.index(row) + 1}", [row])

    def _produce(self):
        """Show all rows, from the base image to the top row.

        Args:
            None

        Returns:
            None
        """
        if not self.rows:
            QMessageBox.information(self, "No Rows", "Add at least one row first.")
            return
        self._show_result("Superimposition: all rows", list(self.rows))


class PairSuperimpositionWindow(SuperimpositionWindow):
    """Pair Superimposition: the same table, but the sources are the two panels of the viewer,
    and Produce draws the rows on the panel of the base row."""

    TITLE = "Pair Superimposition"
    INFO = ("Row 1 is the base panel. Each next row goes on top of it. Source is the Left or the "
            "Right panel. Change the Mode of a row to Contour to draw lines instead of a colormap. Drag the = of a row, or type a new number in the # column, to change the order. "
            "Double-click View to show one row alone in a new window. Alpha is the opacity of a layer. "
            "Rotation of a row changes only View, because the base panel has its own Rotate button. "
            "Produce draws all rows on the base panel. "
            "Clear Superimposition removes them.")
    BASE_IS_PANEL = True

    def __init__(self, panels, draw_on_panels, parent=None):
        """Build the window with two rows: the left panel as the base, and the right panel on top.

        Args:
            panels (dict): The keys 'left' and 'right'. Each value is a dict with the keys
                image, wcs, header and filename of that panel.
            draw_on_panels (callable): Called with the images dict and the rows when the user
                clicks Produce. It draws on the panel of the base row, returns a list of
                notes, and raises ValueError for rows that cannot be drawn.
            parent (QWidget): Parent widget, or None.

        Returns:
            None
        """
        super().__init__(parent)
        self.images = dict(panels)
        self.draw_on_panels = draw_on_panels
        self.add_row_btn.setText("Add Row")
        self.add_row_btn.setToolTip("Add an Image row at the bottom for the Left or the Right panel.")
        self.produce_btn.setToolTip("Draw all rows on the panel of the base row.")
        self.rows = [SuperimpositionRow(mode=MODE_IMAGE, source="left", cmap=BASE_CMAP),
                     SuperimpositionRow(mode=MODE_IMAGE, source="right")]
        self._rebuild_table()

    def _source_label(self, path):
        """Make the text that names a panel in the Source list.

        Args:
            path (str): 'left' or 'right'.

        Returns:
            label (str): The panel name and the file name of its image.
        """
        return f"{path.capitalize()}: {self.images[path]['filename']}"

    def _add_row(self):
        """Ask for a panel and add an Image row at the bottom for it.
        The same panel can have more than one row, for example an Image row and a Contour row.

        Args:
            None

        Returns:
            None
        """
        names = [self._source_label(key) for key in ("left", "right")]
        choice, accepted = QInputDialog.getItem(self, "Add Row", "Panel for the new row:", names, 0, False)
        if not accepted:
            return
        key = "left" if choice == names[0] else "right"
        self.rows.append(SuperimpositionRow(mode=MODE_IMAGE, source=key))
        self._rebuild_table()

    def _purge_unused_images(self):
        """Keep both panels in the memory, also when no row uses one of them.

        Args:
            None

        Returns:
            None
        """

    def _produce(self):
        """Draw all rows on the panel of the base row. Show an error message if that fails.

        Args:
            None

        Returns:
            None
        """
        if not self.rows:
            QMessageBox.information(self, "No Rows", "Add at least one row first.")
            return
        rows = list(self.rows)
        notes = []
        if rows[0].rotation:
            rows[0] = replace(rows[0], rotation=0.0)
            notes.append("Row 1: the rotation is not used. Use the Rotate button of the panel.")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            notes += self.draw_on_panels(self.images, rows)
        except ValueError as error:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Superimposition Error", str(error))
            return
        except Exception as error:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Superimposition Error", f"Could not draw the superimposition:\n{error}")
            raise
        QApplication.restoreOverrideCursor()
        if notes:
            QMessageBox.information(self, "Superimposition Note", "\n".join(notes))
