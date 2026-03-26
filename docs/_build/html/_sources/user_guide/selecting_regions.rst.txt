================
Selecting Regions
================

Learn how to select areas of interest in your data cube for analysis.

Region Selection Tools
======================

IFU Analyzer provides five selection tools for different use cases:

======================== ==================================================================
Tool                     Best For
======================== ==================================================================
**Point Select**         Single pixels or precise locations
**Rectangle Select**     Rectangular areas, embedded sources
**Circle Select**        Star-like or circular features
**Ellipse Select**       Elongated structures, galaxies
**Polygon Select**       Irregular or complex shapes
**Free-Form Draw**       Precise manual outlines
======================== ==================================================================

Activating Region Selection
===========================

**Method 1: Toolbar**

1. Click figure button in toolbar
2. Tool is now active

Using Point Select Tool
=======================

Select individual point:

1. Click the **\*** from the region selector toolbar
2. Click a pixel location

Using Rectangle Select Tool
============================

Select rectangular regions:

1. Click the **▭** from the region selector toolbar
2. Hold upper-left corner
3. Drag to lower-right corner
4. Release mouse to finalize
5. Handles appear for resize/adjust

Adjustments:

- **Drag edge**: Resize in one dimension
- **Drag corner**: Resize in two dimensions
- **Drag interior**: Move entire rectangle
- **Delete**: Clear and start over

Using Circle Select Tool
========================

Select circular regions:

1. Click the **◯** from the region selector toolbar
2. Hold upper-left corner
3. Drag to lower-right corner
4. Release mouse to finalize
5. Handle appears for resize/adjust

Adjustments:

- **Drag edge**: Adjust radius
- **Drag center**: Move circle

Using Ellipse Select Tool
=========================

Select elliptical regions (for elongated features):

1. Click the **⬭** from the region selector toolbar
2. Hold upper-left corner
3. Drag to lower-right corner
4. Release mouse to finalize
5. Handle appears for resize/adjust

Adjustments:

- **Drag corners**: Adjust size
- **Drag center**: Move ellipse

Using Free-Form Select Tool
=========================

Select irregular or complex shapes:

1. Click the **✏** from the region selector toolbar
2. Draw a closed figure
3. Release mouse to finalize

Tips for Polygons:

- Make closed figures
- Make figures inside the boundaries

Statistics
====================

   - **Mean**: Central values
   - **Std**: Spread
   - **Min/Max**: Range
   - **Sum**: Total

Next Steps
==========

After selecting regions:

1. :doc:`gaussian_fitting` to analyze profiles
2. :doc:`continuum_subtraction` to remove baselines
3. :doc:`exporting_data` to save results
