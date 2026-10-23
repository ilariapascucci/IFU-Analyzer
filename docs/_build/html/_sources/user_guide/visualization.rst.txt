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

Different scales emphasize different aspects of your data. Click the **Scale** icon
on a slice to open the "Scale and Color Range" dialog. There you choose the scale and
the color range. Each slice has its own settings. The same slice rendered with each
scale:

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

**Power Scale**

- Scale: normalized intensity raised to a power, called gamma
- You set the gamma value in the dialog. Gamma must be above 0
- A gamma below 1 brightens faint features. A gamma above 1 emphasizes bright features
- Best for: Custom emphasis

**Color Range**

The color range sets which intensity values map to the ends of the color bar. Choose one mode:

- **Absolute**: type a lower and an upper value. A blank box uses the minimum or the
  maximum of the image
- **Percentile**: use percentiles of the image pixels, for example to ignore a few very bright pixels

To use the same scale and range on the other slice, click the **Copy** icon and choose
**Copy Scale/Range**.

Colormaps
=========

Colormaps control how intensity is mapped to colors on slices. IFU Analyzer includes many options.

**Changing Colormaps**

1. Click the **Colormap** icon in the slice toolbar
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