======================
Command-Line Interface
======================

``clip.py`` exposes the core IFU Analyzer pipeline as a standalone CLI —
useful for scripting, batch jobs, or headless server workflows where the
GUI is not available.

.. contents::
   :local:
   :depth: 2


Setup
=====

Run ``clip.py`` from the **project root**, one level above
``cube_explorer/``:

.. code-block:: bash

   python clip.py <command> [options]

.. note::

   ``clip.py`` must be placed **outside** the ``cube_explorer/`` folder so
   that the ``cube_explorer.data_loader`` import resolves correctly. If
   you see ``ImportError: Could not import CubeData``, move ``clip.py``
   one directory level up.


Commands
========

There are five CLI subcommands:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Command
     - Purpose
   * - :ref:`cli-auto-continuum`
     - Automatic continuum detection and export
   * - :ref:`cli-custom-continuum`
     - Continuum fit from user-supplied anchor points
   * - :ref:`cli-subtract-continuum`
     - Subtract a continuum (constant, fitted curve, or anchor points)
   * - :ref:`cli-2d-rotate`
     - Rotate a 2D FITS image with WCS update
   * - :ref:`cli-2d-contour`
     - Extract contour coordinates from a 2D FITS image


.. _cli-auto-continuum:

``auto-continuum`` — Automatic continuum detection and export
=============================================================

Automatically fits a continuum baseline to the cube's integrated spectrum
using median filtering, peak detection, and iterative sigma clipping,
then saves the fitted curve as a text file.

.. code-block:: bash

   python clip.py auto-continuum -i <cube.fits> [-p window sigma width distance exclusion iters smoothing]

**Flags**

.. list-table::
   :widths: 22 55 23
   :header-rows: 1

   * - Flag
     - Description
     - Default
   * - ``-i`` / ``--input``
     - Path to input FITS cube
     - *(required)*
   * - ``-p`` / ``--params``
     - Space-separated algorithm parameters (see below)
     - ``51 2.0 1 3 3 2 5``

**Parameters (in order)**

.. list-table::
   :widths: 8 14 60 18
   :header-rows: 1

   * - Position
     - Name
     - Meaning
     - Default
   * - 1
     - ``window``
     - Median-filter window size (channels)
     - ``51``
   * - 2
     - ``sigma``
     - Peak-detection threshold (× RMS)
     - ``2.0``
   * - 3
     - ``width``
     - Minimum peak width (channels)
     - ``1``
   * - 4
     - ``distance``
     - Minimum distance between peaks (channels)
     - ``3``
   * - 5
     - ``exclusion``
     - Channels to mask on each side of a detected peak
     - ``3``
   * - 6
     - ``iters``
     - Sigma-clipping iterations
     - ``2``
   * - 7
     - ``smoothing``
     - Spline smoothing factor
     - ``5``

**Algorithm**

1. Apply a median filter of width ``window`` as a first-pass continuum
   estimate.
2. Detect emission peaks above ``sigma × RMS`` using
   ``scipy.signal.find_peaks``.
3. Mask ``exclusion`` channels on each side of every detected peak.
4. Interpolate over masked regions, then re-apply the median filter.
5. Iteratively sigma-clip outliers for ``iters`` passes.
6. Fit a smoothing ``UnivariateSpline`` (k=3) to the cleaned continuum.
7. Prompt for an output filename → save ``<name>.txt`` with columns
   ``Wavelength  Continuum_Fit``.

**Output file format**

Tab-delimited two-column ASCII with a comment header:

.. code-block:: text

   # Type: fitted_continuum
   # Wavelength  Continuum_Fit
   3.0000e-06   1.2345e-02
   3.0010e-06   1.2347e-02
   ...

The ``# Type: fitted_continuum`` line lets :ref:`cli-subtract-continuum`
re-import the file later.

**Examples**

.. code-block:: bash

   # Default parameters
   python clip.py auto-continuum -i data/miri_cube.fits

   # Custom parameters: larger window, tighter sigma, more exclusion
   python clip.py auto-continuum -i data/miri_cube.fits -p 75 1.5 2 5 5 3 8


.. _cli-custom-continuum:

``custom-continuum`` — Continuum from user-supplied points
==========================================================

Fits a Cubic Spline through a set of user-defined wavelength/flux anchor
points and subtracts the result from the spectrum.

.. code-block:: bash

   python clip.py custom-continuum -i <cube.fits> -p <points.csv>

**Flags**

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Flag
     - Description
   * - ``-i`` / ``--input``
     - Path to input FITS cube
   * - ``-p`` / ``--points``
     - Path to CSV file containing anchor points

**Points file format**

Comma-separated, one point per line, no header:

.. code-block:: text

   3.5e-6,0.012
   4.0e-6,0.015
   4.8e-6,0.011

The spline is fit with ``bc_type='natural'`` and evaluated at every
wavelength channel.

**Output file format**

.. code-block:: text

   # Wavelength  Original_Flux  Continuum_Fit  Subtracted_Result
   3.0000e-06   1.4567e-02     1.2345e-02     2.2220e-03
   ...

**Example**

.. code-block:: bash

   python clip.py custom-continuum -i data/muse_cube.fits -p my_points.csv


.. _cli-subtract-continuum:

``subtract-continuum`` — Subtract a continuum from a spectrum
=============================================================

Subtracts either a constant offset, a previously exported fitted-continuum
file, or a raw-points file from the cube's spectrum.

.. code-block:: bash

   python clip.py subtract-continuum -i <cube.fits> -c <value_or_file>

**Flags**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Flag
     - Description
   * - ``-i`` / ``--input``
     - Path to input FITS cube
   * - ``-c`` / ``--continuum``
     - Numeric constant, path to a fitted-continuum ``.txt``, or path to
       a raw-points ``.txt``/``.csv``

**Three modes — auto-detected from the ``-c`` argument**

.. list-table::
   :widths: 22 33 45
   :header-rows: 1

   * - Mode
     - How detected
     - Behaviour
   * - **Constant**
     - ``-c`` is a plain number (e.g. ``0.05``)
     - Subtracts the constant from every channel
   * - **Fitted continuum file**
     - File contains a ``# Type: fitted_continuum`` header line
     - Interpolates the saved curve onto the current wavelength grid
       and subtracts
   * - **Raw points file**
     - File has no recognised header
     - Parses comma-separated ``(wavelength, flux)`` pairs, fits a Cubic
       Spline, then subtracts

**Output file format**

.. code-block:: text

   # Wavelength  Original_Flux  Continuum_Fit  Subtracted_Result

A continuum exported by :ref:`cli-auto-continuum` can be fed directly to
``subtract-continuum -c`` — the two commands are designed to pipe into
each other.

**Examples**

.. code-block:: bash

   # Constant baseline
   python clip.py subtract-continuum -i data/alma_cube.fits -c 0.03

   # Re-use a previously fitted continuum
   python clip.py subtract-continuum -i data/alma_cube.fits -c continuum_export.txt

   # Subtract using raw anchor points
   python clip.py subtract-continuum -i data/nirspec_cube.fits -c my_baseline_points.csv


.. _cli-2d-rotate:

``2d-rotate`` — Rotate a 2D FITS image
======================================

Rotates a 2D FITS image by an arbitrary angle and updates the WCS CD
matrix so RA/Dec coordinates remain accurate after rotation.

.. code-block:: bash

   python clip.py 2d-rotate -i <image.fits> -a <angle_degrees>

**Flags**

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Flag
     - Description
   * - ``-i`` / ``--input``
     - Path to input 2D FITS file
   * - ``-a`` / ``--angle``
     - Rotation angle in degrees (positive = counter-clockwise)

**What happens internally**

- ``scipy.ndimage.rotate`` with ``reshape=True`` and ``cval=NaN`` (empty
  border pixels are NaN, not zero).
- The CD matrix is updated:
  ``CD_new = CD_orig × R⁻¹``, where ``R`` is the 2D rotation matrix
  for ``-angle``.
- ``CRPIX`` is **not** changed — the reference pixel stays at the same
  sky coordinate.

**Output**

A new FITS file with the rotated image and an updated WCS header. The CLI
prompts for the output filename.

**Example**

.. code-block:: bash

   python clip.py 2d-rotate -i data/miri_intensity.fits -a 45
   # Prompted: Enter filename to save ROTATED FITS: rotated_45deg
   # → saves rotated_45deg.fits


.. _cli-2d-contour:

``2d-contour`` — Extract contour coordinates from a 2D FITS image
=================================================================

Generates contour lines at user-specified sigma levels, computes RA/Dec
for every contour vertex using the image WCS, and saves the result as a
text file.

.. code-block:: bash

   python clip.py 2d-contour -i <image.fits> -l <level1> [level2 ...]

**Flags**

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Flag
     - Description
   * - ``-i`` / ``--input``
     - Path to input 2D FITS file
   * - ``-l`` / ``--levels``
     - One or more sigma multipliers (space-separated)

**Level definition**

- Background median and RMS are estimated with Astropy
  ``sigma_clipped_stats`` (3σ clipping).
- Each level ``N`` draws a contour at ``median + N × RMS``.

**Output file format**

Comma-separated ASCII with a comment header:

.. code-block:: text

   # IFU Analyzer 2D - Contour Export
   # level, x_pixel, y_pixel, RA_deg, Dec_deg
   3.000000e+00, 42.1234, 18.7654, 67.90591, 18.20682
   ...

**Examples**

.. code-block:: bash

   # Single level
   python clip.py 2d-contour -i data/moment0.fits -l 3

   # Multiple levels
   python clip.py 2d-contour -i data/moment0.fits -l 3 5 10 20

   # High-significance contours only
   python clip.py 2d-contour -i data/miri_intensity.fits -l 5 10 30 50


Output Formats — quick reference
================================

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Command
     - Output columns
   * - ``auto-continuum``
     - ``Wavelength  Continuum_Fit``
   * - ``custom-continuum``
     - ``Wavelength  Original_Flux  Continuum_Fit  Subtracted_Result``
   * - ``subtract-continuum``
     - ``Wavelength  Original_Flux  Continuum_Fit  Subtracted_Result``
   * - ``2d-rotate``
     - FITS file with rotated image and updated WCS header
   * - ``2d-contour``
     - ``level, x_pixel, y_pixel, RA_deg, Dec_deg``

Files exported by the CLI are fully compatible with the GUI's
**subtract-continuum** import — a continuum file exported by
``auto-continuum`` can be fed directly to ``subtract-continuum -c``.
