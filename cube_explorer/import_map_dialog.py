"""
Import Map Dialog

Prompts for an externally-produced 2D flux map (FITS) plus the species,
wavelength, and flat error percent needed to attach it to a Rotational
Diagram line-table row. HITRAN identity resolution happens in the caller
(gui_rotational_diagram.py's import_map_line), not here, so it can reuse the
existing QProgressDialog pattern import_csv_file already uses.
"""

import os

from astropy.io import fits
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QPushButton, QDoubleSpinBox, QFileDialog, QMessageBox,
)

_IDENTITY_KEYWORDS = ("MOLECULE", "LINEWAVE", "A_UL", "E_UPPER", "G_U",
                      "TRANSITN", "JLOWER", "BRANCH")


def _peek_header(path):
    """Best-effort peek at a FITS header for round-tripped identity keywords.
    Never raises -- this is a cosmetic shortcut, not validation; the real
    read happens on accept via build_imported_map_payload.

    Args:
        path (str): FITS file path.

    Returns:
        found (dict): Whichever of _IDENTITY_KEYWORDS are present.
    """
    try:
        with fits.open(path) as hdul:
            found = {}
            for hdu in hdul:
                for kw in _IDENTITY_KEYWORDS:
                    if kw not in found and kw in hdu.header:
                        found[kw] = hdu.header[kw]
            return found
    except Exception:
        return {}


class ImportMapDialog(QDialog):
    """Dialog to pick a flux-map FITS file and its species/wavelength/error."""

    def __init__(self, parent=None):
        """Build the Import Map prompt.

        Args:
            parent (QWidget): Parent widget, or None for a top-level dialog.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle("Import Map")
        self.setMinimumWidth(420)
        self._header_identity = None
        self._identity_species = None
        self._identity_wavelength = None

        layout = QVBoxLayout()
        info = QLabel(
            "Attach an already-reduced flux map (line-integrated, already "
            "convolved/resolution-matched elsewhere) to one line. It feeds "
            "Convolve Maps/Run Map only -- not Run Fit or Calculate Fluxes."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()

        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_btn)
        form.addRow("FITS file:", path_layout)

        self.species_edit = QLineEdit()
        self.species_edit.setPlaceholderText("e.g. H2")
        form.addRow("Species:", self.species_edit)

        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setRange(0.1, 1000.0)
        self.wavelength_spin.setDecimals(6)
        self.wavelength_spin.setValue(5.0)
        form.addRow("Wavelength (µm):", self.wavelength_spin)

        self.error_pct_spin = QDoubleSpinBox()
        self.error_pct_spin.setRange(0.0, 1000.0)
        self.error_pct_spin.setDecimals(1)
        self.error_pct_spin.setValue(20.0)
        self.error_pct_spin.setSuffix(" %")
        self.error_pct_spin.setToolTip(
            "Flat percent error: this map's error map is abs(flux) * pct/100, "
            "the same mechanism a hand-entered or CSV-imported flux uses."
        )
        form.addRow("Error %:", self.error_pct_spin)

        layout.addLayout(form)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #2e7d32;")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Import")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _browse(self):
        """Open a file dialog, fill in the FITS path, and pre-fill
        species/wavelength from the header if this is a round-tripped
        export (Save Maps or Gaussian Map's Area/AUC export).

        Args:
            None

        Returns:
            None
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Flux Map FITS", "", "FITS files (*.fits *.fz *.fit)")
        if not path:
            return
        self.path_edit.setText(path)

        found = _peek_header(path)
        if found.get("MOLECULE"):
            self.species_edit.setText(str(found["MOLECULE"]).strip().upper())
        if found.get("LINEWAVE") is not None:
            try:
                self.wavelength_spin.setValue(float(found["LINEWAVE"]))
            except (TypeError, ValueError):
                pass

        has_full_identity = all(k in found for k in ("A_UL", "E_UPPER", "G_U", "TRANSITN"))
        self._header_identity = found if has_full_identity else None
        # Tied to these exact values -- if the user edits species/wavelength
        # afterward, get_values() must stop returning this stale identity.
        self._identity_species = self.species_edit.text().strip().upper()
        self._identity_wavelength = self.wavelength_spin.value()
        if has_full_identity:
            self.status_label.setText(
                "Species, wavelength, and A_ul/E_u/g_u all found in this file's "
                "header -- no HITRAN lookup needed.")
            self.status_label.setVisible(True)
        elif found.get("MOLECULE") or found.get("LINEWAVE") is not None:
            self.status_label.setText(
                "Species/wavelength auto-filled from this file's header -- still editable.")
            self.status_label.setVisible(True)
        else:
            self.status_label.setVisible(False)

    def get_values(self):
        """Read the dialog's current fields.

        Args:
            None

        Returns:
            values (tuple): (fits_path, species, wavelength_um, error_pct,
                header_identity) -- header_identity is a dict with 'A_UL',
                'E_UPPER', 'G_U', 'TRANSITN' (plus 'JLOWER'/'BRANCH' when
                present) if the header already carried all four, else None.
        """
        species = self.species_edit.text().strip().upper()
        wavelength = self.wavelength_spin.value()
        identity = self._header_identity
        if identity is not None and (
                species != getattr(self, "_identity_species", None)
                or abs(wavelength - getattr(self, "_identity_wavelength", -1.0)) > 1e-9):
            identity = None
        return (self.path_edit.text().strip(), species, wavelength, self.error_pct_spin.value(), identity)

    def accept(self):
        """Validate required fields before closing.

        Args:
            None

        Returns:
            None
        """
        path, species, _wavelength, _error_pct, _identity = self.get_values()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Import Map", "Choose a valid FITS file.")
            return
        if not species:
            QMessageBox.warning(self, "Import Map", "Enter a species (e.g. H2).")
            return
        super().accept()
