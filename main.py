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
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from cube_explorer.gui import CubeExplorer

if __name__ == "__main__":
    # Enable High-DPI scaling before the QApplication is constructed so window
    # and font metrics stay consistent across Retina/scaled displays.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    w = CubeExplorer()

    # Size relative to the actual screen instead of a fixed pixel size, so the
    # window fits smaller/laptop displays instead of forcing native widget
    # chrome (taller on macOS) to squeeze the plots.
    screen_geometry = app.primaryScreen().availableGeometry()
    default_width = min(1100, int(screen_geometry.width() * 0.9))
    default_height = min(800, int(screen_geometry.height() * 0.9))
    w.resize(default_width, default_height)
    w.show()
    sys.exit(app.exec_())

'''
Achievements:
Week 1.
- Create GUI template using PyQt.
- Develop a Cube class with all the necessary functions.
- Import FITS files and match them with predetermine telescope options.
- Create a spectrum given the cube.
- Show a specific slice by zooming and clicking on the spectrum.
- Display data of the specific slide (wavelength, intensity, RA, DEC).
- Implement process for different telescopes.

Week 2.
- Create GitHub repo to store weekly advancements and compare scripts.
- Move slice information to the left.
- RIGHT SIDE: Add second slice for comparison.
- Get instrument info from file and remove dropbox.
- Zoom in slices and click pixels.
- Get pixel info after click (RA/DEC, intensity, coordinates).

Week 3.
- Upgrade GUI placement and visualization.
- Update pixel RA/DEC using VAL, PIX, DELTA.
- Select region from different shapes (rectangle, circle, ellipse, point, custom).
- Upgrade slice comparison (copy settings)

Week 4 & 5.
- Display more cube information (e.g., pixel size and position angle in sky).
- README and additonal documentation (expected outputs and commands).
- Display pixel-specific data (unique RA/DEC?, pixel coordinate of reference point?)
- Reform file structure.
- Add header button.
- .lst files: given elemenet and micron region, select rows and plot vertical lines in wavelength
when clicked, show other info... -> list of elements option -> species, wavelength range, 1
- Info module: load cube, getting info out of cube, plot spectrum
- Tool module: zoom in slice, etc... will call info module
- create multiple small modules that'll call info module.
'''

'''
WEEK 6
- after selecting wavelengths and species, show a list of the possible
rows and make the user select the ones the user wants plotted.
- color
- option hitran (kinda)
- once you plot, hover on a line and identify the row/show in a box.
'''

'''
W7
- scaling
- refresh button
- lines when region
- hitran upgrade + merge
- fit gaussian and area under the curve W/m^2, erg/s/cm^2
- improve units
- troubleshooting
- gaussian fit in each pixel
'''

'''
WEEK 7 next steps
- normalizing by the median AKA the middle y is 1 (check picture)
- publish/make executable
- fits + lsd convert if not the same units (e.g. muse and micron)
- troubleshooting
- same order of magnitude (1 or 10)

WEEK 8-9
- cube of residuals (y-fit)/std
- two cases: 3x std 
- under 3: noise - create gaussian using new std
- over 3: it is a peak - done
- convert data to fits file -> intensity map and residual cube
- export ascii spectrum
- improve readme
'''

'''
Week 10
- export fits for all parameters
- have multiple programs simultaneously (new window)
'''

'''
next steps
- 3std and upper limits
- manually check colors in maps for differences
- continuum subtracted cube of the wavelength boundaries.
'''