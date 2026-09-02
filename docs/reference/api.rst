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

Flux Analysis
================

.. autofunction:: cube_explorer.api.fit_hitran_lines

.. autofunction:: cube_explorer.api.extract_flux

.. autofunction:: cube_explorer.api.create_flux_map

Rotational Diagram
==================

.. autofunction:: cube_explorer.api.rotational_diagram
