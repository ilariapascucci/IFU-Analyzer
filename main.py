#!/usr/bin/env python3

"""
IFU Cube Explorer

A graphical tool for visualizing and analyzing integral field unit (IFU) 
data cubes from various telescopes and instruments.

Features:
- Load and display 3D spectral data cubes
- Interactive wavelength/velocity slice viewing
- Integrated spectrum analysis
- Region and pixel selection tools
- Coordinate display and transformations
- Multi-instrument support (MUSE, ALMA, MIRI, NIRSpec)

Dependencies:
- PyQt5 for GUI
- matplotlib for plotting
- numpy for numerical operations
- astropy for FITS handling and WCS

Usage:
    python main.py

The application will open in a window sized 1100x800 pixels.
Use File->Open to load a FITS data cube.
"""

import sys
from PyQt5.QtWidgets import QApplication
from cube_explorer.gui import CubeExplorer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CubeExplorer()
    w.resize(1100, 800)
    w.show()
    sys.exit(app.exec_())