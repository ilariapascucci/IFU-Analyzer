==============
Exporting Data
==============

Save your analysis results in various formats for further processing and publication.

Export Options Overview
=======================

IFU Analyzer provides several export capabilities:

========================== ==================== ================================================
What to Export             Format                Purpose
========================== ==================== ================================================
**Spectrum**               ASCII, FITS           Further analysis
**Gaussian Fit Maps**      FITS                  Imaging analysis tools
**Visualization**          PNG, JPEG, PDF        Slices and other 2D data
========================== ==================== ================================================

Exporting Spectra
=================

**ASCII Format**

Export spectrum as tab-delimited text:

1. Click on **Save Spectrum**
2. Specify filename with .txt or .csv extension
3. File created with header that includes instrument and units
4. File created with columns:

.. code-block:: text

   Wavelength(um)    Flux(mJy)
   4.5000            12.34
   4.5012            12.41
   ...

Exporting Gaussian Fit Maps
============================

**Export Gaussian Fitting Results**

Save maps created from pixel-by-pixel cube fitting:

1. Click on **Save as FITS file** for each parameter on its result window
2. Specify filename

**Image Export Formats**

Export intensity map as image:

1. Display desired map/wavelength
2. Click on the **Save** icon under every visualization
