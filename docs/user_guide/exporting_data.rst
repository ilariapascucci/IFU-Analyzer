==============
Exporting Data
==============

Save your analysis results in various formats for further processing and publication.

Export Options Overview
=======================

IFU Analyzer provides several export capabilities:

============================== ==================== ================================================
What to Export                 Format               Purpose
============================== ==================== ================================================
**Spectrum**                   Text (.txt, .csv)    Further analysis
**Slice, masked slice, cube**  FITS                 One slice, or the cube outside a region set to NaN
**Region**                     Text                 The sky coordinates of a region, for reuse on
                                                    another cube (:doc:`selecting_regions`)
**Continuum cubes**            FITS                 The continuum and the continuum-subtracted cube
**Gaussian Fit Maps**          FITS                 Imaging analysis tools
**Flux Map**                   FITS                 The flux map and the detection map
**Rotational diagram**         CSV, FITS and PNG    Line fluxes, and the temperature maps
**2D viewer images**           FITS, text, ZIP      Rotated images, contours, and matched images
**Visualization**              PNG, JPEG, PDF       Slices and other 2D data
============================== ==================== ================================================

Exporting Spectra
=================

**Text Format**

Export the displayed spectrum as tab-delimited text:

1. Select the tab (Full, Left Region, or Right Region) that you want to save
2. Click **Save Spectrum**, under the spectrum plot
3. Specify a filename with a .txt or .csv extension
4. The file starts with header lines that give the instrument, the tab, and the units. Then
   the columns follow:

.. code-block:: text

   # Results Export from Cube Explorer
   # Instrument: MIRI
   # Spectrum tab: full
   # Type: full_results
   # Wavelength unit: um
   # Flux unit: MJy/sr
   #
   # Wavelength (um)    Original    Continuum    Subtracted
   4.500000e+00    1.234000e+01    nan    nan
   4.501200e+00    1.241000e+01    nan    nan
   ...

The **Continuum** and **Subtracted** columns hold values only after you fit a continuum for
that tab (:doc:`continuum_subtraction`). Without a continuum, they are NaN.

Exporting Slices, Regions, and Cubes
====================================

The **Export** and **Region** groups in the toolbar hold the buttons for slices, cubes, and
regions. Each button first asks which slice, Left or Right, to use.

- **Save Slice**: the current slice as a standalone 2D FITS file
- **Save Slice (Masked)**: the same, with everything outside the region set to NaN
- **Save Cube (Masked)**: the whole cube, with everything outside the region set to NaN
- **Save Region** and **Import Region**: a region as RA/Dec sky coordinates in a text file, and
  the way back (:doc:`selecting_regions`)

Exporting Continuum Cubes
=========================

**Save Continuum Cubes** (in the **Continuum Tools** group) writes two FITS cubes:
``<name>_continuum.fits`` and ``<name>_contsub.fits``
(:doc:`continuum_subtraction`).

Exporting Gaussian Fit Maps
============================

**Export Gaussian Fitting Results**

Save maps created from pixel-by-pixel cube fitting:

1. In the "Gaussian Parameter Maps" window, select the map in the **Parameter** dropdown
2. Click **Save as FITS File**
3. Specify filename

Each map is a single-slice FITS file. The **Residuals** map is saved as a 3D cube
(:doc:`gaussian_fitting_cube`).

Exporting Flux Maps and Rotational Diagrams
===========================================

- **Flux Map:** in the results window, click **Export as FITS** to save the map that is
  displayed (:doc:`flux_extraction`).
- **Rotational diagram:** **Save Fluxes** writes the line table as a CSV file, and
  **Import from CSV** reads it back. **Save Maps** writes the maps as FITS and PNG files in one
  ZIP file (:doc:`rotational_diagram`).

Exporting from the 2D Viewer
============================

- **Save FITS** (on each panel): the image, with its updated WCS header, for example after a rotation
- **Save** in the **Contour** group: the contours as a text or CSV file
- **Pair** and **Batch**: the matched images as FITS files, or as one ZIP file for **Batch**
  (:doc:`2d_viewer`)

**Image Export Formats**

Export intensity map as image:

1. Display desired map/wavelength
2. Click on the **Save** icon under every visualization
