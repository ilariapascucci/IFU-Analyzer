======================
Command-Line Interface
======================

``clip.py`` exposes the core IFU Analyzer pipeline as a standalone
command-line tool. It is useful for scripting, batch jobs, headless
servers, and Jupyter workflows where the GUI is not available.

The same code paths used by the GUI are exposed here, so the CLI and
the GUI are guaranteed to produce identical results given identical
inputs.


Setup
=====

Run ``clip.py`` from the project root, one level above
``cube_explorer/``:

.. code-block:: bash

   python clip.py <command> [options]

If you see ``ImportError: Could not import CubeData``, move ``clip.py``
one directory level up so it sits next to ``main.py``.


Commands
========

Five subcommands are available. Click any of them to jump to its full
reference page.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Command
     - Purpose
   * - :ref:`cli-auto-continuum`
     - Automatic continuum detection and export
   * - :ref:`cli-custom-continuum`
     - Continuum fit from user-supplied anchor points
   * - :ref:`cli-subtract-continuum`
     - Subtract a continuum (constant, fitted file, or anchor points)
   * - :ref:`cli-2d-rotate`
     - Rotate a 2D FITS image, updating the WCS
   * - :ref:`cli-2d-contour`
     - Extract contour coordinates from a 2D FITS image


.. _cli-auto-continuum:

auto-continuum
==============

.. code-block:: bash

   python clip.py auto-continuum -i <cube.fits> [-p window sigma width distance exclusion iters smoothing]

Automatically fit the continuum baseline of a cube's integrated spectrum
using median filtering, peak detection, iterative sigma clipping, and
spline smoothing. The fitted continuum is saved to a tab-delimited
ASCII file with header ``# Type: fitted_continuum``, ready to be fed
back into :ref:`cli-subtract-continuum`.

The algorithm runs in four steps:

1. **Median filtering** of width ``window`` produces a first-pass
   continuum estimate.
2. **Peak detection** finds emission/absorption features above
   ``sigma × RMS`` using ``scipy.signal.find_peaks`` with the given
   ``width`` and ``distance`` constraints.
3. **Iterative sigma clipping** removes outliers above the running
   continuum estimate for ``iters`` passes.
4. **Spline smoothing** fits a cubic ``UnivariateSpline`` to the
   cleaned continuum with smoothing parameter
   ``s = N / smoothing_factor``.

After fitting, you are prompted for an output filename. The file is
written as ``<name>.txt`` in the current directory.

:Parameters:

   **-i, --input** : str
      Path to the input FITS cube. Required.

   **-p, --params** : list of 7 numbers, optional
      Algorithm parameters in this exact order:
      ``window sigma width distance exclusion iters smoothing``.
      Defaults to ``51 2.0 1 3 3 2 5`` if omitted.

      * **window** (int, default 51) — Median-filter window size in channels.
        Larger values produce a smoother baseline but may miss fine
        structure. Must be odd. Typical range 21–101.

      * **sigma** (float, default 2.0) — Peak-detection threshold in
        units of the RMS. Lower values detect weaker lines. Use 1.5 for
        clean data, 2.0 for typical, 3.0 for noisy.

      * **width** (int, default 1) — Minimum width in channels for a
        feature to count as a peak. Filters out single-channel spikes.

      * **distance** (int, default 3) — Minimum separation in channels
        between detected peaks.

      * **exclusion** (int, default 3) — Channels masked on each side of
        every detected peak before the final fit. Use larger values for
        broad lines.

      * **iters** (int, default 2) — Number of sigma-clipping passes
        after the first fit.

      * **smoothing** (int, default 5) — Spline smoothing factor.
        Smoothness scales as ``s = N / smoothing_factor``. Lower →
        closer fit to the data; higher → smoother curve.

:Output file format:

   Tab-delimited ASCII with a comment header:

   .. code-block:: text

      # Export from IFU-Analyzer CLI
      # Instrument: MIRI
      # Type: fitted_continuum
      # Wavelength unit: micron
      # Flux unit: MJy/sr
      #
      # Wavelength (micron)   Continuum_Fit
      6.530400e+00            1.420169e+06
      6.531200e+00            1.448846e+06
      ...

   The ``# Type: fitted_continuum`` line lets
   :ref:`cli-subtract-continuum` auto-detect this format on re-import.

:Examples:

   Default parameters:

   .. code-block:: bash

      python clip.py auto-continuum -i data/miri_cube.fits

   Custom parameters — larger window, tighter sigma, more exclusion:

   .. code-block:: bash

      python clip.py auto-continuum -i data/miri_cube.fits -p 75 1.5 2 5 5 3 8


.. _cli-custom-continuum:

custom-continuum
================

.. code-block:: bash

   python clip.py custom-continuum -i <cube.fits> -p <points.csv>

Fit a cubic spline through user-supplied wavelength/flux anchor points,
then subtract the result from the spectrum. The output is a 4-column
ASCII file in ``# Type: full_results`` format — identical to the GUI's
``Save Results`` button.

:Parameters:

   **-i, --input** : str
      Path to the input FITS cube. Required.

   **-p, --points** : str
      Path to a CSV file of anchor points, one per line, comma-separated,
      no header. Required.

:Points file format:

   .. code-block:: text

      3.5e-6,0.012
      4.0e-6,0.015
      4.8e-6,0.011

   The spline is fit with ``bc_type='natural'`` and evaluated at every
   wavelength channel.

:Output file format:

   .. code-block:: text

      # Type: full_results
      # Wavelength (micron)   Original    Continuum    Subtracted
      6.530400e+00            1.420169e+06   1.450957e+06   -3.078756e+04
      ...

:Example:

   .. code-block:: bash

      python clip.py custom-continuum -i data/muse_cube.fits -p my_points.csv


.. _cli-subtract-continuum:

subtract-continuum
==================

.. code-block:: bash

   python clip.py subtract-continuum -i <cube.fits> -c <value_or_file>

Subtract a continuum from a cube's spectrum. The continuum source is
auto-detected from the ``-c`` argument:

* a plain number → constant offset
* a file with header ``# Type: fitted_continuum`` → 2-column file
  (auto-continuum output)
* a file with header ``# Type: full_results`` → 4-column file (GUI
  Save Results or CLI custom-continuum output)
* any other text file → parsed as raw anchor points (wavelength, flux),
  fit with a cubic spline

:Parameters:

   **-i, --input** : str
      Path to the input FITS cube. Required.

   **-c, --continuum** : str or float
      One of:

      * A numeric constant (e.g. ``0.05``). Subtracted from every channel.
      * Path to a ``fitted_continuum`` file. The 2-column data is
        interpolated onto the cube's wavelength grid and subtracted.
      * Path to a ``full_results`` file. The continuum column (3rd column)
        is extracted, interpolated, and subtracted. **Added in the
        2026-05 update** to support GUI ↔ CLI round-trips.
      * Path to a raw points file. Fit with a cubic spline and subtracted.

:Output file format:

   Same 4-column ``# Type: full_results`` format as
   :ref:`cli-custom-continuum`.

:Examples:

   Constant baseline:

   .. code-block:: bash

      python clip.py subtract-continuum -i data/alma_cube.fits -c 0.03

   Re-use a previously fitted continuum:

   .. code-block:: bash

      python clip.py subtract-continuum -i data/alma_cube.fits -c continuum_export.txt

   Re-use a GUI-saved results file (4-column format):

   .. code-block:: bash

      python clip.py subtract-continuum -i data/cube.fits -c gui_results.txt

   Subtract using raw anchor points:

   .. code-block:: bash

      python clip.py subtract-continuum -i data/nirspec_cube.fits -c my_baseline_points.csv


.. _cli-2d-rotate:

2d-rotate
=========

.. code-block:: bash

   python clip.py 2d-rotate -i <image.fits> -a <angle_degrees>

Rotate a 2D FITS image and update the WCS CD matrix so RA/Dec
coordinates remain accurate after rotation.

:Parameters:

   **-i, --input** : str
      Path to the input 2D FITS file. Required.

   **-a, --angle** : float
      Rotation angle in degrees, positive = counter-clockwise. Required.

:Notes:

   * ``scipy.ndimage.rotate`` is used with ``reshape=True`` and
     ``cval=NaN`` (empty border pixels are NaN, not zero).
   * The CD matrix update is ``CD_new = CD_orig × R⁻¹`` where ``R`` is
     the 2D rotation matrix for ``-angle``.
   * ``CRPIX`` is **not** changed — the reference pixel stays at the
     same sky coordinate.

:Output:

   A new FITS file with the rotated image and updated WCS header. You
   are prompted for the output filename.

:Example:

   .. code-block:: bash

      python clip.py 2d-rotate -i data/miri_intensity.fits -a 45


.. _cli-2d-contour:

2d-contour
==========

.. code-block:: bash

   python clip.py 2d-contour -i <image.fits> -l <level1> [level2 ...]

Generate contour lines at user-specified sigma levels and save every
vertex's pixel and sky coordinates to a text file.

:Parameters:

   **-i, --input** : str
      Path to the input 2D FITS file. Required.

   **-l, --levels** : list of float
      Sigma multipliers, space-separated. Each level ``N`` draws a
      contour at ``median + N × RMS`` where ``median`` and ``RMS`` are
      from ``astropy.stats.sigma_clipped_stats`` (3σ clipping).
      Required, at least one value.

:Output file format:

   Comma-separated ASCII:

   .. code-block:: text

      # IFU Analyzer 2D - Contour Export
      # level, x_pixel, y_pixel, RA_deg, Dec_deg
      3.000000e+00, 42.1234, 18.7654, 67.90591, 18.20682
      ...

   You are prompted for the output filename.

:Examples:

   Single level:

   .. code-block:: bash

      python clip.py 2d-contour -i data/moment0.fits -l 3

   Multiple levels:

   .. code-block:: bash

      python clip.py 2d-contour -i data/moment0.fits -l 3 5 10 20


Using the CLI from Jupyter
==========================

``clip.py`` exposes the same five commands as plain Python functions,
specifically for use from Jupyter notebooks or scripts. They take
``output_name`` as a keyword argument, skipping the interactive
filename prompt that the CLI uses.

.. code-block:: python

   from clip import (
       auto_continuum,
       custom_continuum,
       subtract_continuum,
       rotate_2d,
       contour_2d,
   )

   auto_continuum("mycube.fits", output_name="continuum_out")
   subtract_continuum("mycube.fits", "continuum_out.txt", output_name="subtracted")
   custom_continuum("mycube.fits", "points.csv", output_name="custom_sub")
   rotate_2d("image.fits", 45, output_name="rotated_45")
   contour_2d("image.fits", [3, 5, 10], output_name="contours")

All five wrappers accept ``output_name=None`` (the default), in which
case the function falls back to the same interactive ``input()`` prompt
the CLI uses.


Troubleshooting
===============

Jupyter notebook issues
-----------------------

**Cell hangs after running a CLI command with** ``!python clip.py ...``

   Jupyter does not forward keyboard input to ``!`` subprocesses, so
   when ``clip.py`` calls ``input()`` for the output filename, it sits
   waiting for input that will never arrive.

   Fix: pipe the filename in via ``echo``, or use the Python wrapper
   functions:

   .. code-block:: python

      # Option A — pipe the filename
      !echo my_output | python clip.py auto-continuum -i cube.fits

      # Option B — use the wrapper (recommended)
      from clip import auto_continuum
      auto_continuum("cube.fits", output_name="my_output")

**FileNotFoundError when loading the FITS file**

   Jupyter's working directory is wherever Jupyter was started, not
   where the FITS file lives. Either start Jupyter from the project
   folder, or change directory inside the notebook:

   .. code-block:: python

      import os
      os.chdir(r"C:\path\to\IFU-Analyzer")

   Then verify with ``os.path.exists("mycube.fits")``.

**ImportError: cannot import** ``CubeData``

   ``clip.py`` must be at the project root, one level above
   ``cube_explorer/``. The wrapper expects to be importable as
   ``from cube_explorer.data_loader import CubeData``.

   Fix: move ``clip.py`` up one level if it is currently inside
   ``cube_explorer/``.

**Filename contains a space or parenthesis**

   The shell splits on spaces, so a filename like
   ``Level3_ch1-long_s3d (1).fits`` is parsed as two arguments.

   Fix: quote the path, or rename the file:

   .. code-block:: bash

      python clip.py auto-continuum -i "Level3_ch1-long_s3d (1).fits"


CLI issues
----------

**"unrecognized arguments"**

   Almost always caused by an unquoted filename containing spaces. See
   the previous item.

**The continuum file from the GUI is not recognized**

   The CLI accepts both legacy ``fitted_continuum`` (2 columns) and
   the new ``full_results`` (4 columns) formats out of the box. If you
   still get a parse error, open the file in a text editor and confirm
   it has either ``# Type: fitted_continuum`` or ``# Type: full_results``
   in the first 15 lines.

**"--params expects 7 values"**

   ``auto-continuum``'s ``--params`` flag requires all seven values in
   order, even if you only want to change one. Use the default block
   ``51 2.0 1 3 3 2 5`` as a starting template and edit the values you
   need.

**Output file has fewer rows than the cube's wavelength channels**

   The continuum output (``fitted_continuum`` format) only writes rows
   where the fit was finite. NaN regions of the spectrum are skipped.
   The 4-column ``full_results`` output preserves every channel,
   writing ``nan`` where appropriate.

**Cube loads but the spectrum looks wrong**

   ``clip.py`` uses ``CubeData`` from ``cube_explorer.data_loader``,
   which computes ``cube.spectrum`` as the mean across all spatial
   pixels. If you need a region-restricted spectrum, use the GUI's
   region tools to define the region first, then save the spectrum
   from the GUI.