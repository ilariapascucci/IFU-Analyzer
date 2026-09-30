#!/usr/bin/env python3

"""
IFU Cube Explorer

A graphical tool for visualizing and analyzing integral field unit (IFU) 
data cubes from various telescopes and instruments.

The application will open in a window sized 1100x800 pixels.
Use File->Open to load a FITS data cube.
"""

import sys

if __name__ == "__main__":
    # Spawn reruns this file per worker with __name__ != "__main__". Imports
    # here stay out of workers, instead of loading the GUI stack 8x over.
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    from cube_explorer.gui import CubeExplorer

    # Scaling window
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    w = CubeExplorer()

    # Relative over absolute values
    screen_geometry = app.primaryScreen().availableGeometry()
    default_width = min(1100, int(screen_geometry.width() * 0.9))
    default_height = min(800, int(screen_geometry.height() * 0.9))
    w.resize(default_width, default_height)
    w.show()
    sys.exit(app.exec_())