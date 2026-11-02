#!/usr/bin/env python3

"""
IFU Cube Explorer

A graphical tool for visualizing and analyzing integral field unit (IFU) 
data cubes from various telescopes and instruments.

The application will open in a window sized 1100x800 pixels.
Click Open FITS to load a FITS data cube.
"""

import sys
import traceback

# The versions in requirements.txt are tested on this Python version only.
TESTED_PYTHONS = ((3, 13),)
TESTED_PYTHONS_TEXT = " or ".join(f"{major}.{minor}" for major, minor in TESTED_PYTHONS)


def warn_if_untested_python():
    """Print a warning when the Python version is not a tested one.

    Args:
        None

    Returns:
        None
    """
    if sys.version_info[:2] not in TESTED_PYTHONS:
        found = f"{sys.version_info[0]}.{sys.version_info[1]}"
        print(f"[WARNING] IFU Analyzer is tested on Python {TESTED_PYTHONS_TEXT}. "
              f"You have Python {found}.")
        print("          The libraries may not install, or results may differ. "
              "See the Installation page of the documentation.")


if __name__ == "__main__":
    warn_if_untested_python()

    # Spawn reruns this file per worker with __name__ != "__main__". Imports
    # here stay out of workers, instead of loading the GUI stack 8x over.
    try:
        from PyQt5.QtCore import Qt, QThread
        from PyQt5.QtWidgets import QApplication, QMessageBox
        from cube_explorer.gui import CubeExplorer
    except ImportError as exc:
        print(f"[ERROR] A required library is missing or has the wrong version: {exc}")
        print("        Install the libraries with: pip install -r requirements.txt")
        print(f"        These versions need Python {TESTED_PYTHONS_TEXT}.")
        sys.exit(1)

    # Scaling window
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    # Native styles give each OS different widget sizes. Fusion looks the same everywhere.
    app.setStyle("Fusion")

    def report_unhandled_exception(exc_type, exc_value, exc_traceback):
        """Show an error dialog for an exception no code caught, instead of letting PyQt5 abort.

        Args:
            exc_type (type): Exception class.
            exc_value (BaseException): Exception instance.
            exc_traceback (traceback): Traceback of the exception.

        Returns:
            None
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        # A dialog from a worker thread is not safe. Only the GUI thread shows one.
        if QThread.currentThread() is app.thread():
            QMessageBox.critical(
                None, "Unexpected error",
                f"{exc_type.__name__}: {exc_value}\n\nThe console has the full traceback.")

    sys.excepthook = report_unhandled_exception
    w = CubeExplorer()

    # Relative over absolute values
    screen_geometry = app.primaryScreen().availableGeometry()
    default_width = min(1100, int(screen_geometry.width() * 0.9))
    default_height = min(800, int(screen_geometry.height() * 0.9))
    w.resize(default_width, default_height)
    w.show()
    sys.exit(app.exec_())