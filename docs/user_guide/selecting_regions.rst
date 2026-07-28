=================
Selecting Regions
=================

Learn how to select areas of interest in your data cube for analysis.

Region Selection Tools
======================

IFU Analyzer provides five selection tools for different use cases:

======================== ==================================================================
Tool                     Best For
======================== ==================================================================
**\*** Point             Single pixels or precise locations
**▭** Rectangle          Rectangular areas, embedded sources
**◯** Circle             Circular features
**⬭** Ellipse            Elongated structures, galaxies
**✏** Freehand           Precise manual outlines
======================== ==================================================================

Three more buttons sit alongside them in each slice's toolbar:

======================== ==================================================================
Button                   Purpose
======================== ==================================================================
**✎** Edit Shape         Toggle drag/resize of the placed region
**✖** Clear              Remove the region and start over
**⎘** Copy               Copy the selection to the other slice
======================== ==================================================================

How Region Selection Works
==========================

Selection follows a strict order of operations, and each slice tracks its own
state independently. Only **one region per slice** can exist at a time.

1. **Choose a shape.** All five shape buttons are available. Click one to arm it.
2. **Draw the region.** Only the armed shape stays clickable -- click it again to
   cancel. The region is placed when you release the mouse.

.. image:: /_static/region_drawing.png
   :alt: Rectangle tool armed and highlighted, with no region drawn yet

3. **Region placed.** The shape buttons are now *disabled*. This is deliberate:
   you cannot start a second region while one exists. The stats panel switches
   to **Intensity (Area Selected)** for the region.

.. image:: /_static/region_placed.png
   :alt: A placed rectangular region, with all shape buttons greyed out

4. **Edit if needed.** Toggle **✎** to drag or resize. Toggle it off again to
   commit the change.
5. **Clear before drawing again.** Press **✖** to remove the region. The shape
   buttons become available again.

.. note::

   If the shape buttons look greyed out, a region is already placed on that
   slice. Press **✖** to clear it before selecting a new one.

Editing a Placed Region
=======================

**✎** is a toggle, and it only becomes available once a region has been placed.

- **Toggle on**: handles appear; drag to move or resize
- **Toggle off**: the edit is committed and handles are removed

.. image:: /_static/region_editing.png
   :alt: The same region in edit mode, showing drag handles at corners and edges

Editing is deliberately opt-in rather than always-on, so a region cannot be
nudged out of place by an accidental click while you are working with the
spectrum.

.. note::

   **✎** is not available for **Point** selections -- a single pixel has nothing
   to resize. Clear it and place a new point instead.

Using the Tools
===============

Point Select
------------

1. Click **\*** in the slice toolbar
2. Click a pixel location

Rectangle, Circle and Ellipse Select
------------------------------------

1. Click **▭**, **◯**, or **⬭** in the slice toolbar
2. Press and hold at one corner
3. Drag to the opposite corner
4. Release to place the region

To adjust it afterwards, toggle **✎** and drag the handles:

- **Drag edge**: resize in one dimension
- **Drag corner**: resize in two dimensions
- **Drag interior**: move the whole region

Freehand Select
---------------

1. Click **✏** in the slice toolbar
2. Draw a closed outline around the feature
3. Release to place the region

Tips for freehand:

- Make closed figures
- Keep the outline inside the slice boundaries

Statistics
==========

   - **Mean**: Central values
   - **Std**: Spread
   - **Min/Max**: Range
   - **Sum**: Total

Next Steps
==========

After selecting regions:

1. :doc:`gaussian_fitting_1d` to analyze profiles
2. :doc:`continuum_subtraction` to remove baselines
3. :doc:`exporting_data` to save results
