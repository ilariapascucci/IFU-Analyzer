=============
API Reference
=============

Complete reference for the public functions in ``cube_explorer.api``, the
GUI-free interface for scripts and notebooks.

Every entry on this page is generated from the function's own docstring, so it
always matches the code. For task-oriented walkthroughs, see the
:doc:`../user_guide/index`.

.. code-block:: python

    from cube_explorer.api import load_cube, fit_line

.. note::

   ``cube.wavelength`` is an astropy ``Quantity``, so it carries its unit.
   Use ``cube.wavelength.value`` when a function or comparison needs plain
   numbers.

Data Loading
============

.. autofunction:: cube_explorer.api.load_cube

Region Extraction
==================

.. autofunction:: cube_explorer.api.extract_pixel_spectrum

.. autofunction:: cube_explorer.api.extract_region_spectrum

Gaussian Line Fitting
=====================

.. autofunction:: cube_explorer.api.fit_line

Full-Cube Fitting
=================

.. autofunction:: cube_explorer.api.create_gaussian_map

.. autofunction:: cube_explorer.api.export_residual_cube

Continuum Fitting
=================

.. autofunction:: cube_explorer.api.fit_continuum

.. autofunction:: cube_explorer.api.measure_equivalent_width

.. autofunction:: cube_explorer.api.save_continuum_cubes

Flux Analysis
================

.. autofunction:: cube_explorer.api.fit_hitran_lines

.. autofunction:: cube_explorer.api.extract_flux

.. autofunction:: cube_explorer.api.create_flux_map

Rotational Diagram
==================

.. autofunction:: cube_explorer.api.rotational_diagram

Rotational Diagram Maps
=======================

These functions fit the rotational diagram at the pixels of one cube, or of several cubes
with different pixel scales (the ``_multi`` versions). Each function has a docstring with its
arguments. Use ``help(function_name)`` to read it. The user guide page
:doc:`../user_guide/rotational_diagram` has worked examples.

- ``fit_region``, ``fit_pixel``, ``create_rotational_diagram_map``: one cube
- ``fit_region_multi``, ``fit_pixel_multi``, ``create_rotational_diagram_map_multi``: several
  cubes at different pixel scales
- ``save_rotational_diagram_map_fits``: write a map result to a FITS file
- ``STATUS_OK``, ``STATUS_NO_DATA``, ``STATUS_TOO_FEW_LINES``, ``STATUS_FIT_FAILED``: the
  outcome of each fit
