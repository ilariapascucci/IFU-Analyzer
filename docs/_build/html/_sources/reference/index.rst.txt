=========
Reference
=========

Technical reference and API documentation for IFU Analyzer.

.. toctree::
   :maxdepth: 2

   file_formats
   keyboard_shortcuts
   command_reference
   glossary

API Documentation
=================

For Python developers using IFU Analyzer modules:

.. code-block:: python

   from cube_explorer import data_loader, gaussian_fitter
   
   # Load FITS file
   cube = data_loader.load_fits('mycube.fits')
   
   # Fit Gaussians
   results = gaussian_fitter.fit_cube(cube)

See module documentation for full API reference.

File Formats
============

See :doc:`file_formats` for detailed information about supported file formats.

Keyboard Shortcuts
==================

See :doc:`keyboard_shortcuts` for complete keyboard reference.

Commands and Features
=====================

See :doc:`command_reference` for comprehensive command listing.

Glossary
========

See :doc:`glossary` for definitions of technical terms.
