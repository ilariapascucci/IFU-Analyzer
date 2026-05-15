===
FAQ
===

Frequently asked questions and troubleshooting tips for IFU Analyzer.

.. contents::
   :local:
   :depth: 2


Getting Started
===============

**Q: I just installed IFU Analyzer. Where do I start?**

A: Follow these steps:

1. Read :doc:`quickstart` for a five-minute overview
2. Get some sample data (or find your own FITS cube)
3. Launch the application: ``python main.py``
4. **File → Open FITS** to load your first data cube
5. Explore the interface following the Quick Start guide

**Q: What FITS data can I use?**

A: Any standard FITS cube with:

- 2D or 3D data (images or cubes)
- Wavelength calibration in the FITS header
- A valid coordinate system (optional but useful for sky-aware tools)

Confirmed supported instruments: MUSE, ALMA, MIRI, NIRSpec. Generic 2D
FITS images are also supported in the 2D Dual Image Viewer.


Data & Files
============

**Q: Where do I find example data?**

A: IFU Analyzer doesn't include sample data. You'll need to:

- Use your own observational data
- Download public archive data
- Ensure it's in standard FITS format

**Q: How large can cubes be?**

A: Practically limited by your computer's RAM. Memory-mapped access is
enabled automatically for large files, so very large cubes can still be
opened — analysis steps will of course run slower on huge cubes.

**Q: I get an error opening a file. What's wrong?**

A: Try:

1. Verify the file is a valid FITS
2. Check that the FITS header parses correctly
3. Confirm required keywords are present (``INSTRUME``, ``TELESCOP`` help
   the instrument auto-detection; ``CTYPE1``/``CTYPE2`` and a valid CD
   matrix or ``CDELT`` are needed for sky-aware tools)

**Q: Why is cube fitting so slow?**

A: Large cubes are slow because:

- There are many pixels to process
- Each pixel requires non-linear curve fitting
- Solutions: use fewer cores, restrict to a smaller wavelength range, or
  fit a subset region first


Troubleshooting
===============

The following items cover the issues reported most often. Each maps to a
concrete fix.

Memory errors
-------------

- Close other memory-hungry applications.
- Try a smaller cube or a subset wavelength range.
- IFU Analyzer automatically uses memory mapping for large files; you
  don't need to enable it manually.

Display issues
--------------

- Make sure PyQt5 is properly installed (``pip install PyQt5``).
- Update graphics drivers if rendering is glitchy or windows flicker.
- Try a different color scale (Log, Power) for better contrast in
  faint-emission cubes.

File loading errors
-------------------

- Verify the file format matches a supported instrument or is a generic
  2D/3D FITS.
- Check the FITS header for the keywords used by instrument detection
  (``INSTRUME``, ``TELESCOP``).
- Ensure the file is not truncated or corrupted.
- 2D FITS images are supported natively in the 2D Viewer.

WCS grid not visible
--------------------

- Confirm the FITS header contains ``CTYPE1``/``CTYPE2`` with an RA/Dec
  projection (e.g. ``RA---TAN``, ``DEC--TAN``).
- Check the terminal output for ``DEBUG rebuild_axes`` messages
  confirming the WCS parse status.
- Non-celestial WCS automatically falls back to plain pixel axes.

Superimpose misalignment
------------------------

- Look in the terminal for ``DEBUG superimpose: Peak separation`` — it
  should be **< 0.5 px**.
- A large separation indicates a WCS header discrepancy between the two
  files (e.g. different ``EQUINOX``, mis-set ``CRPIX``, or a missing CD
  matrix).

Slides Convolute / Batch Convolution issues
-------------------------------------------

- If the output sum differs noticeably from the input, both files must
  have a valid ``CDELT`` or CD matrix in their headers — the pixel scale
  is required to compute the Gaussian kernel size.
- The flux audit (``Ratio out/in``) should always read ``1.000000``. Any
  large deviation indicates a corrupted WCS header.
- "Same resolution" is the expected message when both panels have
  identical pixel scales — no action needed.
- A "No Sky Overlap" warning means the loaded images cover disjoint sky
  regions; load images of the same field.

Gaussian fitting issues
-----------------------

- Ensure an adequate wavelength range is selected (>4 points).
- Check for positive peaks after baseline subtraction.
- ``x0 is infeasible`` errors are handled automatically with bounds
  clamping; they're informational.
- A low success rate usually indicates noise-dominated regions — try
  restricting the spatial area or improving the continuum subtraction
  first.

Performance issues
------------------

- Gaussian mapping is optimized to ~10–30 seconds for typical cubes.
- Large cubes (>100k pixels) may take 1–2 minutes.
- Background processing prevents the GUI from freezing during fits.
- Batch Convolution on many large images may take a few minutes per file
  (reprojection is pixel-exact, pure Python).

CLI (clip.py) issues
--------------------

- Run ``clip.py`` from the project root, **not** from inside
  ``cube_explorer/``.
- If you see ``ImportError: Could not import CubeData``, move
  ``clip.py`` one directory level up.
- The ``--params`` flag for ``auto-continuum`` expects all seven values
  in order: ``window sigma width distance exclusion iters smoothing``.


Still cannot find the answer?
=============================

1. Check the FAQ section or documentation guides above
2. Check related User Guide sections
3. Try the online documentation
4. File an issue on the project repository
5. Contact the development team
