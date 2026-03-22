=================
Command Reference
=================

Comprehensive listing of all commands accessible through menus and dialogs.

File Menu
=========

**Open FITS** (Ctrl+O)

Load a FITS data cube from disk.

Options:

- Select FITS file
- Choose instrument type
- Advanced: Specify subregion, HDU selection

**Save Project** (Ctrl+S)

Save your analysis session (basic functionality).

Options:

- Project metadata only
- Not typically used

**Export Spectrum** (Ctrl+E)

Export current spectrum for analysis.

Formats:

- ASCII tab-delimited (.txt)
- FITS binary table (.fits)
- Comma-separated values (.csv)

Options:

- Original or normalized
- Include error bars
- Custom column selection

**Export Results**

Export analysis results (Gaussian fits, maps, etc.).

Formats:

- FITS multi-extension
- ASCII table, CSV

Outputs:

- Parameter maps (amplitude, center, width, etc.)
- Success mask
- Quality metrics (χ², R²)

**Export Residuals**

Export residual cube from fitting.

Format:

- 3D FITS cube
- Same dimensions as input

**Recent Files**

Quick access to recently opened files.

List shows:

- Most recent first
- File paths and dates
- Click to re-open

**Exit** (Alt+F4)

Close application.

View Menu
=========

**Zoom In** (Ctrl++)

Magnify the intensity map.

Methods:

- Scroll wheel: Zoom
- Menu: Incremental zoom
- Type zoom level

**Zoom Out** (Ctrl+-)

Reduce magnification.

Methods:

- Scroll wheel
- Menu buttons
- Keyboard shortcut

**Fit to Window** (Ctrl+0)

Automatically scale view to fill window.

**Reset View**

Return to default zoom and pan.

**Coordinate System**

Select coordinate system for display.

Options:

- Wavelength (default)
- RA/DEC (if WCS available)

**Color Scale**

Change intensity scaling method.

Options:

- Linear
- Logarithmic
- Power Law
- Square Root

**Show Grid**

Toggle coordinate grid overlay.

Options:

- Grid color
- Grid spacing
- Grid labels

**Show Colorbar**

Toggle intensity scale display.

**Show Measurements**

Show/hide measurement overlays.

Tools Menu
==========

**Region Selection**

Tools for selecting spatial regions.

Sub-menu:

- Point Tool: Single pixels
- Rectangle Tool: Rectangular regions
- Circle Tool: Circular regions
- Ellipse Tool: Elliptical regions
- Polygon Tool: Custom shapes

Options:

- Feather edges
- Invert selection
- Border selection
- Union/intersection modes

**Spectral Analysis**

Analyze spectral properties.

Sub-menu:

- Feature Detection: Find emission/absorption lines
- Line Identification: Identify known lines
- Measure Equivalent Width
- Export Spectra

**Gaussian Fitting**

Fit Gaussian models to spectra.

Sub-menu:

- Fit Single: Interactive fitting
- Fit Cube: Batch processing
- View Results: Display parameter maps
- Fit Options: Configure parameters

**Continuum Subtraction**

Remove baseline from spectra.

Sub-menu:

- Simple Subtract: Built-in models
- Advanced Subtraction: Custom parameters
- Apply to All Pixels
- Automatic Detection

**CO Flux Extraction**

Extract carbon monoxide line fluxes.

Sub-menu:

- Simple Measurement: Single line
- Multi-Line Analysis: Multiple CO transitions
- Full Cube: Extract CO at every pixel
- Temperature Calculation

**Preferences**

Adjust application settings.

Categories:

- Display Options
- Keyboard Shortcuts
- Colormaps
- Performance
- Units
- Data Processing

**Normalize Display**

Choose normalization method.

Options:

- No normalization
- Per-pixel normalization
- Global normalization

Help Menu
=========

**Documentation**

Open this comprehensive manual.

**Keyboard Shortcuts**

Display keyboard shortcut reference.

**About IFU Analyzer**

Show version, copyright, credits.

**Check for Updates**

Look for newer versions online.

Toolbar Buttons
===============

============== ================================
Button         Command
============== ================================
Open FITS      File → Open FITS (Ctrl+O)
Save Project   File → Save (Ctrl+S)
Export         File → Export Spectrum (Ctrl+E)
Zoom In        View → Zoom In (Ctrl++)
Zoom Out       View → Zoom Out (Ctrl+-)
Reset Zoom     View → Fit to Window (Ctrl+0)
Select Region  Tools → Region Selection
Fit Single     Tools → Fit Single (F)
Fit Cube       Tools → Fit Cube (C)
Continuum      Tools → Continuum Subtraction
CO Flux        Tools → CO Flux Extraction
Colormap       Tools → Change Colormap
Color Scale    View → Color Scale
============== ================================

Right-Click Context Menus
==========================

**On Intensity Map**

- Copy Screen: Copy map to clipboard
- Save Image: Export map as image
- Rotate: Rotate displayed image
- Mirror: Mirror image
- Statistics: Show region statistics
- Clear Selection: Deselect current region
- Add Measurement: Annotate measurement
- Properties: Edit map properties

**On Spectrum**

- Export Spectrum: Save spectrum (Ctrl+E)
- Copy: Copy spectrum to clipboard
- Fit: Fit Gaussian to spectrum
- Zoom: Zoom to spectral region
- Statistics: Show statistics
- Clear: Reset spectrum display
- Options: Configure spectrum display
- Properties: Edit spectrum properties

**On Empty Area (Canvas)**

- Reset View: Return to default view
- Properties: Edit display properties
- Load Region: Load saved region
- Save Region: Save current region
- Clear All: Deselect and reset

Dialog Commands
===============

**Open FITS Dialog**

Buttons:

- Browse: Select file
- OK: Open selected file
- Cancel: Abort
- Advanced: More options

**Export Spectrum Dialog**

Options:

- Format: ASCII, FITS, CSV
- Normalized: Yes/No
- Include Errors: Yes/No
- Columns: Select specific columns
- Delimiter: Tab, comma, space

Buttons:

- Browse: Choose location
- Export: Save file
- Cancel: Abort

**Gaussian Fitting Dialog**

Configuration:

- Method: Interactive or automatic
- Wavelength range: Select lines to fit
- Initial guess: Provide starting values
- Constraints: Set parameter limits
- Continuum: Subtract baseline first
- Weighting: Use error bars

Buttons:

- Start: Begin fitting
- Options: Advanced settings
- Cancel: Abort

**Region Selection Dialog**

Tool Selection:

- Point, Rectangle, Circle, Ellipse, Polygon

Options:

- Mode: Replace, Add, Subtract, Intersect
- Feather: Soften edges
- Invert: Select everything except region
- Border: Select only edges

Buttons:

- OK: Apply selection
- Clear: Deselect
- Load: Load saved region
- Save: Save current region
- Cancel: Cancel operation

**Continuum Subtraction Dialog**

Model Selection:

- Linear
- Quadratic
- Cubic
- Spline
- Blackbody
- Power-Law

Parameters:

- Wavelength ranges for fitting
- Model order (if applicable)
- Fitting method

Buttons:

- Preview: Show result before applying
- Apply: Perform subtraction
- Cancel: Abort

**CO Flux Dialog**

Options:

- CO Band: Fundamental, first overtone, etc.
- Transition: J=1-0, J=2-1, etc.
- Wavelength: Auto-populated or manual
- Line width: Specify region size
- Options: Single vs. multi-component

Buttons:

- Measure: Extract flux
- Analyze All: Multiple transitions
- Cancel: Abort

Preferences Dialog
==================

**Display Settings**

- Window size behavior
- Font size
- Color scheme
- High contrast mode

**Keyboard**

- View/customize shortcuts
- Reset to defaults

**Colormaps**

- Select default colormap
- Load custom colormaps
- Colorbar options

**Performance**

- Enable decimation
- Processing threads
- Memory usage limits

**Units**

- Wavelength units (Angstrom, μm)
- Flux calibration constants
- Default display units

**Data Processing**

- Default fitting method
- Continuum model
- Outlier handling

Quick Commands
==============

Most common operations:

1. **Load data**: Ctrl+O
2. **Select region**: R
3. **Export spectrum**: Ctrl+E
4. **Fit single**: F
5. **Fit cube**: C
6. **Zoom**: Ctrl+± or scroll
7. **View results**: Shift+F
8. **Exit**: Ctrl+Q

Command Line Options
====================

To start IFU Analyzer with options:

.. code-block:: bash

   python main.py [options]

Available options:

--file FILENAME
  Open specified FITS file at startup

--region FILENAME
  Load specified region at startup

--readonly
  Open in read-only mode

--no-gui
  Batch processing mode (advanced)

--help
  Show command-line help

Example:

.. code-block:: bash

   python main.py --file mycube.fits --region myregion.reg

Next Steps
==========

- See :doc:`keyboard_shortcuts` for quick reference
- Explore menus and context menus for additional options
- Check :doc:`/user_guide/index` for tutorials
