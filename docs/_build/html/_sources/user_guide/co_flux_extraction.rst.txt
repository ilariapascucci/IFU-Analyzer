==================
CO Flux Extraction
==================

Extract and measure carbon monoxide (CO) line fluxes from spectral data cubes. This specialized tool analyzes CO emission or absorption across multiple rotational transitions.

Starting CO Flux Extraction
===========================

**Access Tool**

1. Click **HITRAN** menu in toolbar
2. Input min/max wavelengths and click on **HITRAN** button
3. Choose molecule, isotopologue number, and vibrational level
4. (Optionally) Refresh HITRAN query

The spectrum now shows the lines given those parameters.
Hovering over a line shows its species, its wavelength, its transition, and its Einstein coefficients
You can remove a line by clicking on it in the spectrum or by clicking on the **SPECIES** button which renders a table with all the current species.

Load Contaminating Lines
========================

Additionally (and optionally), you can load contaminating lines to automatically remove unwanted lines.

1. Click on **Load Contaminating Lines** and load a .csv file with unwanted lines

IMPORTANT: Make sure this file has a column called 'wavelength'.

Multi-Gaussian Fit
=====================

Perform a Gaussian fit for each molecule line in the 1D spectrum.

1. Perform automatic continuum on the 1D spectrum by clicking on **Continuum Tools → Automatic Continuum → Continuum Line**
2. Click on **Multi-Gaussian Fit**

A Gaussian fit will be highlighted for each line, and the fit statistics will be shown.

**Analyze CO for All Pixels**

Extract CO flux maps for each pixel:

1. Complete Automatic Continuum
2. Click on **CO Flux Map**

**Progress Monitoring**

- Progress bar shows % complete

**Output Products**

Resulting maps:

- **CO Flux Map**: Map with integrated flux for each pixel
- **Detection Count**: Map with count for each pixel