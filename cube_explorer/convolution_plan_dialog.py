"""
Table dialog that lists how each image is blurred before images of different resolution are combined
"""

import os

from astropy.stats import gaussian_sigma_to_fwhm
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView,
)

from .gui_sizing import fit_to_screen, fit_dialog_to_table
from .utils import psf_fwhm_arcsec


def confirm_convolution_plan(parent, rows, header_text, footer_text, window_title, accept_label):
    """Show the convolution plan as a sortable table, and let the user accept or cancel.
    Run Map, Convolve Maps and the 2D viewer use it, so every plan looks the same.
    Each row says which image is blurred, and by how much, to reach the reference resolution.

    Args:
        parent (QWidget): Window that owns the dialog.
        rows (list): One dict per image, with 'wavelength' (um), 'instrument' (str), 'filename' (str),
            'native_pix_arcsec' (float), 'sigma_arcsec' (float) and 'sigma_pix' (float) of the kernel.
            The reference image has a kernel of 0.
        header_text (str): Line above the table.
        footer_text (str): Text below the table, with the question to the user.
        window_title (str): Dialog title.
        accept_label (str): Text on the accept button.

    Returns:
        proceed (bool): True if the user accepted.
    """
    rows = list(rows)

    dlg = QDialog(parent)
    dlg.setWindowTitle(window_title)
    fit_to_screen(dlg, 980, 480)
    lay = QVBoxLayout(dlg)

    header_label = QLabel(header_text)
    lay.addWidget(header_label)

    headers = ["Line (um)", "File", "Pixel Scale (\"/px)",
               "Native PSF (FWHM, \")", "PSF Matching Kernel (FWHM, \")"]
    table = QTableWidget(len(rows), len(headers), dlg)
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setSelectionMode(QTableWidget.NoSelection)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    # File names (JWST ones especially) can run long enough to blow the
    # dialog out to an unreasonable width -- cap this one column and
    # show the full name as a tooltip instead of forcing it to fit.
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
    table.setColumnWidth(1, 260)
    table.horizontalHeader().setStretchLastSection(True)

    def _native_psf_arcsec(c):
        """This line's own native angular resolution (its instrument's PSF
        formula), before any convolution -- the reference row's value is the
        common resolution every other row is being PSF-matched to."""
        return psf_fwhm_arcsec(c["wavelength"], c["instrument"])

    def _blur_text(c):
        """
        FWHM of the Gaussian kernel applied to raise this line's own
        native PSF up to the reference line's native PSF (0 for the
        reference line itself -- it needs no PSF matching).

        Args:
            c (dict): One convolution_plan()['per_cube'] entry.

        Returns:
            text (str): FWHM of the PSF-matching kernel, in arcsec and
                native pixels.
        """
        fwhm_arcsec = c["sigma_arcsec"] * gaussian_sigma_to_fwhm
        fwhm_pix = c["sigma_pix"] * gaussian_sigma_to_fwhm
        return f"{fwhm_arcsec:.4f}\" ({fwhm_pix:.2f} px)"

    def _fill():
        """
        Redraw every row of `rows` into `table`, in its current order.

        Args: None

        Returns:
            None
        """
        for r, c in enumerate(rows):
            fname = os.path.basename(c["filename"]) if c["filename"] else "cube"
            if c["sigma_pix"] <= 1e-6:
                fname += "  (reference)"
            texts = (f"{c['wavelength']:.4f}", fname,
                    f"{c['native_pix_arcsec']:.4f}",
                    f"{_native_psf_arcsec(c):.4f}\"", _blur_text(c))
            for col, text in enumerate(texts):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter if col != 1 else Qt.AlignVCenter | Qt.AlignLeft)
                if col == 1:
                    item.setToolTip(fname)
                table.setItem(r, col, item)

    _fill()

    sort_key = {
        0: lambda c: c["wavelength"],
        1: lambda c: (os.path.basename(c["filename"]) if c["filename"] else "cube").lower(),
        2: lambda c: c["native_pix_arcsec"],
        3: lambda c: _native_psf_arcsec(c),
        4: lambda c: c["sigma_pix"],
    }
    sort_state = {"column": None, "order": Qt.AscendingOrder}

    def _on_header_clicked(column):
        """
        Sort `rows` by the clicked column, toggling direction on a repeat click.

        Args:
            column (int): Table column index the user clicked.

        Returns:
            None
        """
        if sort_state["column"] == column and sort_state["order"] == Qt.AscendingOrder:
            sort_state["order"] = Qt.DescendingOrder
        else:
            sort_state["order"] = Qt.AscendingOrder
        sort_state["column"] = column
        rows.sort(key=sort_key[column], reverse=(sort_state["order"] == Qt.DescendingOrder))
        _fill()
        table.horizontalHeader().setSortIndicator(column, sort_state["order"])

    header = table.horizontalHeader()
    header.setSectionsClickable(True)
    header.setSortIndicatorShown(True)
    header.sectionClicked.connect(_on_header_clicked)
    lay.addWidget(table)

    fit_dialog_to_table(dlg, table)

    footer_label = QLabel(footer_text)
    footer_label.setWordWrap(True)
    lay.addWidget(footer_label)

    btns = QHBoxLayout()
    accept_btn = QPushButton(accept_label)
    cancel_btn = QPushButton("Cancel")
    accept_btn.clicked.connect(dlg.accept)
    cancel_btn.clicked.connect(dlg.reject)
    btns.addStretch()
    btns.addWidget(accept_btn)
    btns.addWidget(cancel_btn)
    lay.addLayout(btns)
    return dlg.exec_() == QDialog.Accepted
