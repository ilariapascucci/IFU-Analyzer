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


Using the CO Flux API
======================

Extract CO fluxes programmatically for single spectra or full data cubes.

**Extract CO Flux from a Single Spectrum**

.. code-block:: python

    from cube_explorer.api import load_cube, extract_co_flux, fit_continuum
    
    # Load your data
    cube = load_cube("data.fits")
    
    # Step 1: Fit and subtract continuum
    continuum = fit_continuum(cube.wavelength, cube.spectrum)
    line_spectrum = cube.spectrum - continuum
    
    # Step 2: Extract CO flux
    co_result = extract_co_flux(
        wavelength=cube.wavelength,
        spectrum=line_spectrum,
        co_catalog_path="path/to/CO_catalog.ecsv"
    )
    
    print(f"Total CO flux: {co_result['co_flux']:.3f} erg/s/cm²")
    print(f"Number of lines detected: {len(co_result['detected_lines'])}")
    print(f"Mean continuum level: {co_result['mean_continuum']:.3f}")

**Create CO Flux Maps**

Generate flux maps for the entire data cube:

.. code-block:: python

    from cube_explorer.api import load_cube, create_co_flux_map
    
    cube = load_cube("data.fits")
    
    # Create CO flux map for the entire cube
    # Uses HITRAN automatically by default, or provide a custom CO catalog
    co_flux_map = create_co_flux_map(
        cube,
        n_jobs=4  # Parallel jobs for faster processing
    )
    
    print(f"CO flux map shape: {co_flux_map['flux_map'].shape}")
    print(f"Mean CO flux: {co_flux_map['flux_map'][co_flux_map['flux_map'] > 0].mean():.3f} erg/s/cm²")
    print(f"Pixels with CO detection: {(co_flux_map['detection_map'] > 0).sum()}")

**Expected Catalog Format**

The CO catalog file should be in ECSV (Enhanced Character Separated Values) format with columns:

- ``wavelength``: Wavelength in micrometers
- ``species``: Molecule name (e.g., "CO")
- ``transition``: Transition description (e.g., "J=5-4")
- ``einstein_coeff``: Einstein A coefficient

**Tips for CO Flux Extraction**

- Always apply continuum subtraction before extracting CO flux
- Ensure your wavelength calibration is accurate (errors affect line identification)
- The algorithm automatically measures line widths and integrates flux
- For more control over individual line fits, use ``fit_hitran_lines()`` instead