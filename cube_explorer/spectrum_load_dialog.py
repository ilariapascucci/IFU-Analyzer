"""
Dialog that asks how to read a spectrum text file: which column is which, and the units.
"""

import os

from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from .spectrum_file import default_settings, normalize_wave_unit
from .utils import NIRSPEC_GRATINGS

_PREVIEW_ROWS = 8
_NONE_TEXT = "(none)"
_WAVE_CHOICES = ("um", "nm", "Å")
_FLUX_CHOICES = ("", "Jy", "mJy", "MJy/sr", "erg/s/cm2/um", "erg/s/cm2/Angstrom")
_INSTRUMENT_CHOICES = (_NONE_TEXT, "MIRI", "NIRSpec", "MUSE")


class SpectrumLoadDialog(QDialog):
    """Ask which column is the wavelength, the flux and the error, and what the units are.
    The boxes start with the values that the file gives. The preview shows the first rows of the file.

    Attributes:
        table (SpectrumTable): The table that the dialog is about
    """

    def __init__(self, table, parent=None):
        """Build the dialog.

        Args:
            table (SpectrumTable): The table that read_spectrum_table gives
            parent (QWidget): Parent window, or None

        Returns:
            None
        """
        super().__init__(parent)
        self.table = table
        start = default_settings(table)
        self.setWindowTitle("Load Spectrum")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        n_rows, n_columns = table.values.shape
        layout.addWidget(QLabel(f"{os.path.basename(table.path)}: {n_rows} rows, {n_columns} columns. "
                                "The first rows are shown below."))

        preview = QTableWidget(min(n_rows, _PREVIEW_ROWS), n_columns)
        preview.setHorizontalHeaderLabels(table.column_names)
        preview.setEditTriggers(QTableWidget.NoEditTriggers)
        for row in range(preview.rowCount()):
            for column in range(n_columns):
                preview.setItem(row, column, QTableWidgetItem(f"{table.values[row, column]:.6g}"))
        preview.setMaximumHeight(220)
        layout.addWidget(preview)

        form = QFormLayout()
        names = list(table.column_names)
        self.x_box = QComboBox()
        self.x_box.addItems(names)
        self.x_box.setCurrentIndex(start["x_index"])
        self.y_box = QComboBox()
        self.y_box.addItems(names)
        self.y_box.setCurrentIndex(start["y_index"])
        self.error_box = QComboBox()
        self.error_box.addItems([_NONE_TEXT] + names)
        self.error_box.setCurrentIndex(0 if start["error_index"] is None else start["error_index"] + 1)
        form.addRow("Wavelength column:", self.x_box)
        form.addRow("Flux column:", self.y_box)
        form.addRow("Flux error column:", self.error_box)

        self.wave_box = QComboBox()
        self.wave_box.addItems(_WAVE_CHOICES)
        try:
            unit = normalize_wave_unit(start["wave_unit"])[0]
        except ValueError:
            unit = "um"
        self.wave_box.setCurrentText(unit)
        self.wave_box.setToolTip("Nanometer values are converted to micron when the spectrum loads.")
        form.addRow("Wavelength unit:", self.wave_box)

        self.flux_box = QComboBox()
        self.flux_box.setEditable(True)
        self.flux_box.addItems(_FLUX_CHOICES)
        self.flux_box.setCurrentText(start["flux_unit"])
        self.flux_box.setToolTip("The unit of the flux. You can type another unit. Leave it empty if you do not know it.")
        form.addRow("Flux unit:", self.flux_box)

        self.medium_box = QComboBox()
        self.medium_box.addItems(("unknown", "vacuum", "air"))
        self.medium_box.setCurrentText(start["medium"] or "unknown")
        self.medium_box.setToolTip("Line lists give vacuum wavelengths. Air wavelengths are converted to vacuum for them.")
        form.addRow("Wavelengths are in:", self.medium_box)

        self.instrument_box = QComboBox()
        self.instrument_box.addItems(_INSTRUMENT_CHOICES)
        self.instrument_box.setCurrentText(start["instrument"] if start["instrument"] in _INSTRUMENT_CHOICES else _NONE_TEXT)
        self.instrument_box.setToolTip(
            "The instrument gives the resolving power of the Multi-Gauss fit. Without it, "
            "the fit uses the width of one global fit.")
        form.addRow("Instrument:", self.instrument_box)

        self.grating_box = QComboBox()
        self.grating_box.addItems(NIRSPEC_GRATINGS)
        if start["grating"] and start["grating"].upper() in NIRSPEC_GRATINGS:
            self.grating_box.setCurrentText(start["grating"].upper())
        self.grating_label = QLabel("NIRSpec grating:")
        form.addRow(self.grating_label, self.grating_box)
        self.instrument_box.currentTextChanged.connect(self._sync_grating)
        self._sync_grating()
        self._object = start["object"]

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _sync_grating(self, *_):
        """Show the grating box only when the instrument is NIRSpec.

        Args:
            None

        Returns:
            None
        """
        is_nirspec = self.instrument_box.currentText() == "NIRSpec"
        self.grating_box.setVisible(is_nirspec)
        self.grating_label.setVisible(is_nirspec)

    def settings(self):
        """Read the choices of the user.

        Args:
            None

        Returns:
            settings (dict): The same keys as default_settings gives, without 'complete'
        """
        instrument = self.instrument_box.currentText()
        return {
            "x_index": self.x_box.currentIndex(),
            "y_index": self.y_box.currentIndex(),
            "error_index": None if self.error_box.currentIndex() == 0 else self.error_box.currentIndex() - 1,
            "wave_unit": self.wave_box.currentText(),
            "flux_unit": self.flux_box.currentText().strip(),
            "medium": None if self.medium_box.currentText() == "unknown" else self.medium_box.currentText(),
            "instrument": None if instrument == _NONE_TEXT else instrument,
            "grating": self.grating_box.currentText() if instrument == "NIRSpec" else None,
            "object": self._object,
        }
