============
GUI Overview
============

This section explains the major components of the IFU Analyzer interface and how to navigate them effectively.

Main Window Layout
==================

The IFU Analyzer window is organized into several key areas:

.. image:: /_static/gui_overview1.png
   :alt: GUI Layout

*Note: Screenshot showing main interface layout*

**1. Menu Bar** (Top)

- **Open FITS**: Open FITS file
- **Header**: Display header(s) of FITS file
- **🔄**: Refresh program window
- **🗗**: Open new program window
- **2D**: Open 2D section of program

**2. Main Viewing Area** (Top Center)

Displays the currently selected wavelength left and right slices:

- **Intensity Map**: Color-coded flux data with statistics
- **Color & Scale**: Edit color scheme or scale
- **Region Selection**: Select a region of a slice for analysis

  - Pixel
  - Rectangle
  - Circle
  - Ellipse
  - Free-Form (polygon)
  - Copy the selection from the adjacent slice

**3. Spectrum Panel** (Bottom Center)

Shows the 1D spectrum, with its axes in flux and wavelength. The collapsible
button groups above the plot hold the analysis tools:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Group
     - What it does
   * - **Slice Selection**
     - Click a slice to set the left or right wavelength from that pixel
   * - **Gaussian Fit**
     - Fit lines, mark noise regions, and build full-cube parameter maps
   * - **Continuum Tools**
     - Fit and subtract the continuum, automatically or from custom anchor
       points
   * - **Rotational Diagram**
     - Fit H2 rotational temperatures from measured line fluxes
   * - **HITRAN**
     - Fetch molecular line lists with transition-level filtering
   * - **Shifting**
     - Apply a Doppler shift to the spectrum based on velocity

Getting Help
============

If you forget a button's function:

- **Hover**: Mouse over buttons to see tooltip
- **Toolbar Buttons**: Use buttons for common tasks
- **This Guide**: Full documentation available
