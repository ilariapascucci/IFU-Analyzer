==============
Visualization
==============

IFU Analyzer provides powerful visualization tools to explore and display your data effectively. Learn how to customize the appearance and interact with your data.

Display Options
===============

**Intensity Map Display**

The main intensity map shows a 2D projection of your 3D data cube:

- Each pixel represents a spatial location (spaxel)
- Color shows intensity at the current wavelength
- Interactive: Click any location to update the spectrum by setting a left or right slice

**Spectrum Panel Display**

The spectrum panel shows 1D data:

- X-axis: Wavelength or frequency
- Y-axis: Intensity (flux)

Color Scales
============

Different scales emphasize different aspects of your data. The same slice
rendered with each of them:

.. image:: /_static/scale_comparison.png
   :alt: The same MIRI slice rendered with Linear, Log, Sqrt and Power scales

**Linear Scale** (Default)

- Direct mapping: intensity → color
- Best for: Data without extreme value ranges
- Fast computation

**Logarithmic Scale**

- Scale: intensity → log(intensity)
- Best for: Wide dynamic range
- Emphasizes: Faint features

**Square Root Scale**

- Scale: intensity → √intensity
- Best for: Moderate dynamic range
- Balanced detail emphasis

**Power Law Scale**

- Scale: intensity → intensity^2
- Best for: Custom emphasis

Colormaps
=========

Colormaps control how intensity is mapped to colors on slices. IFU Analyzer includes many options.

**Changing Colormaps**

1. Click **🌈** button in slice toolbar
2. Select colormap from list
3. Preview updates immediately

.. image:: /_static/colormap_dialog.png
   :alt: The Select Colormap dialog, showing the colormap list and preview bar

Zoom Controls
=============

**Zooming In**

1. Select magnifying glass icon under the figure.
2. Drag from corner to opposite corner.
3. Click on the Home icon to revert.

Exporting Visualizations
=========================

**Save Current View as Image**

1. Click on the Save icon under the figure.
2. Specify filename.
3. Save.