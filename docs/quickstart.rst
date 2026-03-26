===========
Quick Start
===========

Get up and running in 5 minutes.

Installation
============

.. code-block:: bash

   pip install -r requirements.txt
   python main.py

Done! The application opens.


Loading Your First Data File
=============================

1. Click **File → Open FITS**
2. Select a FITS file
3. Choose the instrument (MUSE, ALMA, MIRI, NIRSpec, or Generic)
4. Click **Open**

You'll see:
- **Intensity map** (center) - brightness at current wavelength
- **Spectrum** (right) - emission lines and features
- **Wavelength slider** (bottom) - navigate through the cube


Analyzing a Spectral Line
==========================

**Step 1: Navigate to your line**

Use the wavelength slider to find your emission line.

**Step 2: Select a region (optional)**

Click **Select Region** → choose a shape → draw on the intensity map.

This shows the spectrum just from that region.

**Step 3: Fit the line profile**

Right-click on the spectrum → **Fit Gaussian**

You'll see:
- Fitted curve overlaid on the data
- **Parameters**: Amplitude, Center, FWHM, Area, χ²

Click **Accept** to save.

**Step 4: Export your results**

File → **Export Spectrum** → choose format (ASCII or FITS) → save


Mapping Across the Cube
=======================

To fit the same line at every pixel in the cube:

1. Click **Fit Full Cube** (C key)
2. Set wavelength range for the line
3. Click **Run**
4. Wait for processing

Results appear as maps:
- **Amplitude map** - line brightness at each pixel
- **Center map** - wavelength shifts (velocity info)
- **FWHM map** - line width variations

Export any map via File → **Export**.


Common Tasks
============

**Navigate wavelengths**: Use slider or type wavelength in the field

**Change colors**: Click **Colormap** button

**Adjust brightness**: Use **vmin/vmax** sliders or click **Color Scale**

**Subtract continuum**: Click **Continuum** button, select regions, apply

**Measure CO flux**: Click **CO Flux** button


For More Information
====================

- Full feature guide: :doc:`user_guide/index`
- Advanced analysis: :doc:`features/index`
- Troubleshooting: :doc:`troubleshooting`
- FAQ: :doc:`faq`
