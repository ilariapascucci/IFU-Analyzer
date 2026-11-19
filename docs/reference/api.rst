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

.. note::

   The map functions (the Gaussian map, the Flux Map, the continuum cubes, and the
   rotational diagram maps) use several processes. On Windows and macOS, put the code
   of your script under ``if __name__ == "__main__":``.

Data Loading
============

.. autofunction:: cube_explorer.api.load_cube

.. autofunction:: cube_explorer.api.load_spectrum

.. autofunction:: cube_explorer.api.load_fit

Region Extraction
==================

.. autofunction:: cube_explorer.api.extract_pixel_spectrum

.. autofunction:: cube_explorer.api.extract_region_spectrum

Regions
=======

.. autofunction:: cube_explorer.api.make_region_mask

.. autofunction:: cube_explorer.api.save_region

.. autofunction:: cube_explorer.api.load_region

Gaussian Line Fitting
=====================

.. autofunction:: cube_explorer.api.fit_line

Full-Cube Fitting
=================

.. autofunction:: cube_explorer.api.create_gaussian_map

.. autofunction:: cube_explorer.api.create_velocity_map

.. autofunction:: cube_explorer.api.find_rest_lines

.. autofunction:: cube_explorer.api.export_residual_cube

.. autofunction:: cube_explorer.api.save_gaussian_map

.. autofunction:: cube_explorer.api.save_gaussian_maps

Continuum Fitting
=================

.. autofunction:: cube_explorer.api.fit_continuum

.. autofunction:: cube_explorer.api.measure_equivalent_width

.. autofunction:: cube_explorer.api.save_continuum_masks

.. autofunction:: cube_explorer.api.load_continuum_masks

.. autofunction:: cube_explorer.api.save_continuum_cubes

Flux Analysis
================

.. autofunction:: cube_explorer.api.fit_hitran_lines

.. autofunction:: cube_explorer.api.extract_flux

.. autofunction:: cube_explorer.api.create_flux_map

.. autofunction:: cube_explorer.api.save_flux_map

Rotational Diagram
==================

.. autofunction:: cube_explorer.api.rotational_diagram

.. autofunction:: cube_explorer.api.save_rotational_fit

.. autofunction:: cube_explorer.api.save_line_fluxes

.. autofunction:: cube_explorer.api.load_line_fluxes

Rotational Diagram Maps
=======================

These functions fit the rotational diagram at the pixels of one cube, or of several cubes
with different pixel scales (the ``_multi`` versions). The user guide page
:doc:`../user_guide/rotational_diagram` has worked examples.

Fits of one cube:

.. autofunction:: cube_explorer.api.fit_region

.. autofunction:: cube_explorer.api.fit_pixel

.. autofunction:: cube_explorer.api.create_rotational_diagram_map

Fits of several cubes at different pixel scales:

.. autofunction:: cube_explorer.api.fit_region_multi

.. autofunction:: cube_explorer.api.fit_pixel_multi

.. autofunction:: cube_explorer.api.create_rotational_diagram_map_multi

The steps of Run Map, one at a time. These are the "Calculate Maps", "Convolve Maps", and
"Run Map" buttons of the rotational diagram window:

.. autofunction:: cube_explorer.api.compute_native_line_maps

.. autofunction:: cube_explorer.api.convolution_plan

.. autofunction:: cube_explorer.api.convolve_line_maps

.. autofunction:: cube_explorer.api.combine_line_maps

Lines from flux maps that were made before (the "Load Maps" button), and runs that mix them
with lines from cubes:

.. autofunction:: cube_explorer.api.import_line_map

.. autofunction:: cube_explorer.api.fit_region_mixed

.. autofunction:: cube_explorer.api.create_rotational_diagram_map_mixed

Saving the maps:

.. autofunction:: cube_explorer.api.save_line_maps

.. autofunction:: cube_explorer.api.save_rotational_diagram_map_fits

The outcome of each fit is one of these values:

- ``STATUS_OK``
- ``STATUS_NO_DATA``
- ``STATUS_TOO_FEW_LINES``
- ``STATUS_FIT_FAILED``

Helpers
=======

.. autofunction:: cube_explorer.api.fetch_hitran_lines

.. autofunction:: cube_explorer.api.doppler_shift_factor
