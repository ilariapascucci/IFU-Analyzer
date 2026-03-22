============
GUI Overview
============

This section explains the major components of the IFU Analyzer interface and how to navigate them effectively.

Main Window Layout
==================

The IFU Analyzer window is organized into several key areas:

.. image:: /_static/layout.png
   :alt: GUI Layout

*Note: Screenshot showing main interface layout*

**1. Menu Bar** (Top)

- **File**: Open, save, export data
- **View**: Display options, coordinate systems
- **Tools**: Analysis functions, preferences
- **Help**: Documentation, keyboard shortcuts

**2. Toolbar** (Below menu)

Quick-access buttons for common functions:

============================== ========================================
Button                         Function
============================== ========================================
**Open FITS**                  Load a data cube (Ctrl+O)
**Save**                       Save project (Ctrl+S)
**Select Region**              Activate region selection tools
**Fit Single Pixel**           Fit Gaussian to current spectrum
**Fit Full Cube**              Fit Gaussians to all pixels
**Subtract Continuum**         Remove continuum baseline
**CO Flux Extraction**         Extract CO flux measurements
**Change Colormap**            Select color scheme
**Adjust Color Scale**         Set linear/log/sqrt scaling
**Zoom In/Out**               Adjust display magnification
**Reset View**                Restore default zoom
============================== ========================================

**3. Main Viewing Area** (Center)

Displays the currently selected wavelength/velocity slice:

- **Intensity Map**: Color-coded spatial data
- **Interactive**: Click to update spectrum
- **Zoomable**: Use scroll wheel or zoom buttons
- **Overlays**: Region selections, measurement tools

**4. Spectrum Panel** (Right side)

Shows 1D spectrum data:

- **Integrated Spectrum**: Sum of selected region
- **Peak Spectrum**: Maximum at each wavelength
- **Fitted Overlay**: Gaussian fits (when applied)
- **Residuals**: Difference from fit
- **Interactive**: Right-click for options

**5. Status Bar** (Bottom)

Displays useful information:

- Current wavelength/velocity
- Cursor position (RA, Dec or pixel coordinates)
- Intensity value under cursor
- Data cube dimensions
- Number of selected pixels

**6. Control Panels** (Left side)

Various controls for navigation and analysis:

- **Wavelength Slider**: Navigate through cube
- **Intensity Range**: Adjust vmin/vmax
- **Colormap Selection**: Choose color scheme
- **Statistics**: Min/max/mean of current view

Menu Reference
==============

File Menu
---------

**Open FITS** (Ctrl+O)
  Load a FITS data cube

**Save Project** (Ctrl+S)
  Save analysis state (not commonly used)

**Export Spectrum**
  Save current spectrum as ASCII or FITS

**Export Results**
  Save parameter maps (from Gaussian fitting)

**Export Residuals**
  Save residuals cube from fitting

**Recent Files**
  Quick access to recently loaded files

**Exit** (Alt+F4)
  Close the application

View Menu
---------

**Zoom In** (Ctrl++)
  Magnify the intensity map

**Zoom Out** (Ctrl+-)
  Reduce magnification

**Fit to Window**
  Auto-scale view to window size

**Coordinate System**
  - Wavelength (default)
  - RA/DEC (if WCS available)

**Color Scale**
  - Linear: Direct intensity mapping
  - Logarithmic: Emphasize faint features
  - Power Law: Non-linear scaling
  - Square Root: Intermediate option

**Show Grid**
  Overlay coordinate grid

**Show Colorbar**
  Display intensity scale

Tools Menu
----------

**Gaussian Fitting**
  - Fit Single: Interactive fitting
  - Fit Cube: Batch processing
  - View Residuals: Compare data to fits

**Continuum Subtraction**
  Remove baseline emission

**Region Statistics**
  Calculate statistics of selected regions

**CO Flux Extraction**
  Measure CO line fluxes (specialized)

**Preferences**
  Adjust display and analysis settings

**Normalize Display**
  Choose normalization method

Help Menu
---------

**Documentation**
  Open this documentation

**Keyboard Shortcuts**
  Display shortcut reference

**About**
  Version and copyright information

Keyboard Shortcuts
==================

=========================== ==================================
Shortcut                    Action
=========================== ==================================
**Ctrl+O**                  Open FITS file
**Ctrl+S**                  Save project
**Ctrl+E**                  Export spectrum
**Ctrl++** or **Scroll Up**  Zoom in
**Ctrl+-** or **Scroll Dn** Zoom out
**Ctrl+0**                  Reset zoom/fit to window
**Right Arrow**             Next wavelength channel
**Left Arrow**              Previous wavelength channel
**Up Arrow**                Increase wavelength step
**Down Arrow**              Decrease wavelength step
**R**                       Activate region selector
**F**                       Fit single pixel
**C**                       Fit full cube
**G**                       Toggle grid
**L**                       Linear color scale
**M**                       Logarithmic color scale
**Delete**                  Clear current selection
**Escape**                  Cancel current operation
=========================== ==================================

Context Menus
=============

**Right-click on Intensity Map**

- Rotate View
- Mirror Image
- Statistics
- Add Measurement
- Clear Selection

**Right-click on Spectrum**

- Fit Gaussian
- Set Baseline
- Zoom Spectrum
- Export Spectrum Data

Tips for Navigation
===================

**Quick Wavelength Jumping**

1. Click the wavelength display area
2. Type the desired wavelength
3. Press Enter

**Precise Pixel Selection**

1. Use **Select Region** → **Point Tool**
2. Click exact pixel location
3. Hold Shift to add multiple points

**Large Cube Navigation**

For very large cubes (>1 GB):

1. Use **Downsample** mode for initial exploration
2. Switch to full resolution for detailed analysis
3. Work with sub-regions to manage memory

**Comparing Multiple Wavelengths**

1. Load first wavelength
2. Note spectrum and features
3. Navigate to second wavelength
4. Compare intensity maps side-by-side

Customizing the Interface
==========================

**Window Size**

- Resize by dragging window edges
- Default size: 1100×800 pixels
- Recommended minimum: 800×600 pixels

**Panel Layout**

Some options can be reorganized:

- Drag panel headers to reposition
- Right-click panels for options
- Close unused panels for more space

**Color Scheme**

- Tools → Preferences → Display
- Choose from multiple UI themes
- Save preference for next session

**Toolbar**

- Right-click toolbar
- Select tools to show/hide
- Arrange button order (in some layouts)

Accessibility Features
======================

**High Contrast Mode**

- Tools → Preferences → Display → High Contrast

**Keyboard Navigation**

- Most functions accessible via keyboard
- Tab key moves between UI elements
- Arrow keys for fine adjustments

**Large Text**

- Tools → Preferences → Display → Font Size
- Increases all UI text size

Status Indicators
=================

The status bar shows various indicators:

**Color Status**

- Green: Normal operation
- Yellow: Warning (e.g., approaching memory limit)
- Red: Error or critical issue

**Processing Indicator**

- Spinning animation during analysis
- Progress bar for long operations
- Cancel button to stop processing

**Mode Indicator**

Shows current operational mode:

- Selection Mode
- Fitting Mode
- Measurement Mode
- Free Browse Mode

Getting Help
============

If you forget a button's function:

- **Hover**: Mouse over buttons to see tooltip
- **Menu**: Find functions in menus with descriptions
- **Keyboard**: Press F1 or ? for help
- **This Guide**: Full documentation available
