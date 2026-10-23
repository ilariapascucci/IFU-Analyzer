===
FAQ
===

Frequently Asked Questions about IFU Analyzer.

Getting Started
===============

**Q: I just installed IFU Analyzer. Where do I start?**

A: Follow these steps:

1. See :doc:`installation` for setup
2. Install sample data (or find your own FITS cube)
3. Launch the application: ``python main.py``
4. Click **Open FITS** to load your first data cube
5. Explore the interface and refer to :doc:`user_guide/index` for tutorials

**Q: What FITS data can I use?**

A: Any standard FITS cube with:

- 2D or 3D data (images or cubes)
- Wavelength calibration (FITS header keywords)
- Valid coordinate system (optional but useful)

Confirmed supported instruments: MUSE, MIRI, NIRSpec.

Data & Files
============

**Q: Where do I find example data?**

A: IFU Analyzer doesn't include sample data. You'll need to:

- Use your own observational data
- Download public data
- Ensure it's in FITS format

**Q: How large can cubes be?**

A: Practically limited by your computer's RAM.

**Q: I get an error opening a file. What's wrong?**

A: Try:

1. Verify file is valid FITS
2. Check FITS header is valid

**Q: Why is cube fitting so slow?**

A: Large cubes are slow because:

- Many pixels to process
- Complex fitting at each pixel
- Solution: Use fewer cores, smaller wavelength range, or subset region

Still Cannot Find Answer?
===========================

1. Check the FAQ section or documentation guides
2. Check related user guide sections
3. Try online documentation
4. File an issue on project repository
5. Contact development team
