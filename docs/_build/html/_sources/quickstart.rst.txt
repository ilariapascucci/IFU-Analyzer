===========
Quick Start
===========

This quick start guide will get you analyzing IFU data in just a few minutes.

Installation (30 seconds)
=========================

1. Install dependencies:

.. code-block:: bash

   pip install numpy matplotlib astropy PyQt5 scipy

2. Run the application:

.. code-block:: bash

   python main.py

The IFU Analyzer window should open.

Your First Analysis (5 minutes)
===============================

1. **Load a FITS File**
   
   - Click **File → Open FITS**
   - Navigate to your FITS data cube
   - Select the instrument type (MUSE, ALMA, MIRI, NIRSpec, or Generic)
   - Click **Open**

2. **Explore the Data**
   
   - The **main view** shows an intensity map of the cube at a specific wavelength
   - The **spectrum panel** shows the integrated spectrum
   - Adjust the **wavelength slider** to move through the cube
   - Hover over the spectrum to see individual wavelengths

3. **Select a Region**
   
   - Click the **Select Region** button in the toolbar
   - Choose a selection tool (point, rectangle, circle, ellipse, or polygon)
   - Click and drag on the intensity map to select a region
   - The spectrum updates to show the averaged spectrum of that region

4. **Analyze Spectral Features**
   
   - Right-click on the spectrum → **Fit Gaussian**
   - The fit overlay appears on the spectrum showing the fitted profile
   - View fit parameters: amplitude, center, FWHM, area, χ²
   - Click **Accept** to save the fit

5. **Export Results**
   
   - Click **File → Export Spectrum**
   - Choose a format (ASCII or FITS)
   - Specify an output filename
   - Results are saved for further analysis

Basic Workflow Example
=====================

Let's analyze a CO emission line in a MUSE cube:

**Step 1: Load the data**

.. code-block:: text

   File → Open FITS → Select "muse_cube.fits" → Choose "MUSE" instrument

**Step 2: Navigate to the line**

- Use the wavelength slider to find the CO 2-1 transition around 6562.8 Å
- The intensity map should show a bright emission region

**Step 3: Select the emitting region**

- Select **Rectangle Tool** from the toolbar
- Draw a rectangle around the bright emission
- The spectrum panel shows the CO line clearly

**Step 4: Fit the line profile**

- Right-click on the spectrum → **Fit Gaussian**
- Observe:
  - **Amplitude**: Peak intensity of the line
  - **Center**: Exact wavelength of the emission
  - **FWHM**: Width of the line (related to velocity)
  - **Area**: Total flux in the line

**Step 5: Create a map**

- Click **Analysis → Fit Full Cube**
- Wait for the processing to complete
- The amplitude map shows the line strength at each pixel
- The center map shows velocity variations

**Step 6: Export**

- File → Export Spectrum to save the analyzed spectrum
- File → Export Results to save parameter maps as FITS

Key Toolbar Buttons
===================

=================== =====================================================
Button              Function
=================== =====================================================
**Open FITS**       Load a data cube
**Save**            Save analysis results
**Select Region**   Activate region selection tools
**Fit Single**      Fit a Gaussian to the currently displayed spectrum
**Fit Cube**        Fit Gaussians to every pixel in the cube
**Continuum**       Subtract the continuum baseline
**CO Flux**         Extract CO flux measurements
**Colormap**        Change the color scheme of the intensity map
**Color Scale**     Adjust linear/log/sqrt scaling
=================== =====================================================

Common Tasks
============

**Viewing Different Wavelengths**

- Use the **wavelength slider** in the bottom-left
- Or type a specific wavelength in the **wavelength input field**
- Press Enter to jump to that wavelength

**Changing Color Scale**

- Click **Color Scale** button
- Choose from: Linear, Logarithmic, Power, Square Root
- Adjust **vmin** and **vmax** to emphasize features

**Changing Colormap**

- Click **Colormap** button
- Select from available colormaps (Viridis, Hot, Cool, etc.)
- Preview updates in real-time

**Performing Baseline Subtraction**

- Click **Continuum** in the toolbar
- Choose the continuum subtraction method
- Specify the wavelength regions for fitting
- Click **Apply**

**Extracting Spectra from Multiple Regions**

1. Select your first region
2. Note the integrated spectrum
3. Select a different region
4. Compare the two spectra
5. Export both for comparison analysis

**Measuring Kinematics**

- Fit Gaussians across different spatial regions
- The **center map** shows velocity variations
- Use RA/DEC coordinates to identify structures
- Export the center map for kinematic analysis

Tips & Tricks
=============

1. **Smooth Your Display**: Use lower resolution for faster navigation with large cubes

2. **Use Velocity Coordinates**: If your data cube includes velocity information, switch to velocity axes for easier analysis

3. **Memory Management**: For very large cubes (>1 GB), work with sub-regions rather than the full cube

4. **Line Identification**: Check the integrated spectrum against known emission lines to confirm instrument calibration

5. **Quality Checks**: Always examine the χ² and R² values from fits to assess fit quality

6. **Batch Processing**: For analyzing multiple files, write a simple Python script using IFU Analyzer's analysis modules

Next Steps
==========

Now that you've completed your first analysis:

1. Read :doc:`user_guide/index` for detailed feature documentation
2. Explore :doc:`features/index` for advanced analysis techniques
3. Check :doc:`troubleshooting` if you encounter issues
4. Review the :ref:`genindex` for API reference

Need More Help?
===============

- **Installation problems?** → See :doc:`installation`
- **How do I...?** → Check :doc:`faq`
- **Something isn't working** → Go to :doc:`troubleshooting`
- **Looking for reference material** → See :doc:`reference/index`
