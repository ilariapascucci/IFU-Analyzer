============
GUI Overview
============

The main window is built around two mirrored slice panels, Left and Right, that drive one
shared spectrum panel below them -- setting a slice or selecting a region on either side
updates that side's own analysis and spectrum tab independently of the other.

Main Window Layout
==================

.. image:: /_static/gui_overview.png
   :alt: The main Cube Explorer window, showing the Left/Right slice panels, tool group toolbar, and spectrum panel with display controls

The window loaded above (a MIRI cube): Left and Right intensity maps at top, each with its
own running stats (min/max/mean/sum/std) and cursor readout; the Gaussian Fit / Continuum
Tools / Rotational Diagram / HITRAN tool groups in the blue bar; and the spectrum panel
with its Set Slice / tab / Normalize / Shift controls and plot at the bottom.

**1. Menu Bar** (Top)

- **Open FITS**: browse for and load a FITS cube or 2D image; the instrument is
  auto-detected from the header
- **Header**: view the FITS header of the loaded file(s)
- **🔄**: complete reset -- clears every plot, region selection, normalization, and
  Gaussian fit, then reloads the spectrum from the original data
- **🗗**: open a second, independent Cube Explorer window
- **2D**: open the :doc:`2D FITS Dual Viewer <2d_viewer>`

**2. Main Viewing Area** (Top Center)

Displays the currently selected wavelength left and right slices:

- **Intensity Map**: Color-coded flux data with statistics
- **🌈🎚 Color & Scale**: Edit color scheme or scale
- **Region Selection**: Select a region of a slice for analysis

  - **\*** Pixel
  - **▭** Rectangle
  - **○** Circle
  - **⬭** Ellipse
  - **✏** Free-Form (polygon)
  - **✎** Edit the placed region (drag/resize)
  - **⎘** Copy the selection from the adjacent slice
  - **⎘🌈** Copy the colormap from the adjacent slice

Saving a slice or a region is done from the **Region** tool group below, not
from a button here (:doc:`selecting_regions`).

**3. Spectrum Panel** (Bottom Center)

Shows the 1D spectrum, with its axes in flux and wavelength. Two rows of
controls sit above the plot: a row of collapsible analysis tool groups, and
a row of display/navigation controls that are always visible.

*Tool Groups (top row)*

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Group
     - What it does
   * - **Gaussian Fit**
     - Fit lines, mark noise regions, and build full-cube parameter maps
   * - **Continuum Tools**
     - Fit and subtract the continuum, automatically or from custom anchor
       points
   * - **Rotational Diagram**
     - Fit H2 rotational temperatures from measured line fluxes
   * - **HITRAN**
     - Fetch molecular line lists with transition-level filtering
   * - **Region**
     - Save a slice or a region, or import a region saved on another cube
       (:doc:`selecting_regions`)

*Display Controls (bottom row)*

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Control
     - What it does
   * - **Set Left Slice / Set Right Slice**
     - Click a slice to set the left or right wavelength from that pixel
   * - **Full / Left Slice / Right Slice tabs**
     - Switch which spectrum is displayed. Full always shows the whole
       cube's spectrum; Left Slice and Right Slice each show only that
       side's own selected region (nothing, if no region is selected there)
   * - **Show Left / Show Right**
     - While on the Full tab, overlay the Left and/or Right selection's own
       spectrum on top of the Full spectrum for comparison
   * - **Normalize**
     - Divide every displayed spectrum by the Full spectrum's own peak
   * - **Self-Normalize Left / Self-Normalize Right**
     - While Normalize is on, normalize that side to its own peak instead
       of Full's, useful for studying a region's line shape independent
       of its contribution to the total
   * - **Shift / Apply Shift / Cancel Preview / Revert Shift**
     - Preview a Doppler shift on the active tab's spectrum, then apply,
       cancel, or revert it

Getting Help
============

Every button has a tooltip -- hover over one if you're not sure what it does.

Next Steps
==========

Each tool group above has its own dedicated page with the full workflow:

1. :doc:`gaussian_fitting_1d` and :doc:`gaussian_fitting_cube` for the Gaussian Fit group
2. :doc:`continuum_subtraction` for the Continuum Tools group
3. :doc:`rotational_diagram` for the Rotational Diagram group
4. :doc:`flux_extraction` for the HITRAN group's fetch/plot workflow
5. :doc:`selecting_regions` and :doc:`spectrum_display` for the display
   controls row (Set Slice, Full/Left/Right tabs, Normalize, Shift)
