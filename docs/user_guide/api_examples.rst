Using IFU Analyzer Without the GUI
===================================

This guide explains how to use IFU Analyzer as a Python library directly from your scripts or Jupyter notebooks.

Quick Start
-----------

The easiest way is to use the `example_basic_api.py` script as a template. It shows a complete workflow:

.. code-block:: bash

    python example_basic_api.py

This loads a FITS file, fetches HITRAN CO lines, fits them to the spectrum, and generates a plot.

Basic Workflow
--------------

Load a FITS Cube
~~~~~~~~~~~~~~~~

.. code-block:: python

    from cube_explorer.data_loader import CubeData
    
    cube = CubeData()
    cube.load_from_file("path/to/data.fits")
    
    # Access data
    wavelength = cube.wavelength
    spectrum = cube.spectrum
    full_cube = cube.cube  # Shape: (wavelength, x, y)
    
    print(f"Wavelength range: {wavelength.min():.4f} - {wavelength.max():.4f}")
    print(f"Cube shape: {full_cube.shape}")


Fetch HITRAN Molecular Lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from cube_explorer.line_catalog import fetch_hitran_data
    
    # Get CO v=1→0 lines in 4.7-4.8 µm range
    hitran_lines, _ = fetch_hitran_data(
        wavelength_min=4.7,
        wavelength_max=4.8,
        molecule='CO',
        isotopologue=1,        # Main isotope (12CO)
        v_upper=1,             # Upper vibrational level
        v_lower=0              # Lower vibrational level
    )
    
    print(f"Found {len(hitran_lines)} lines")
    for line in hitran_lines:
        print(f"  {line['wavelength']:.9f} um - {line['transition']}")


Fit Spectral Lines
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from cube_explorer.api import fit_hitran_lines

    # Fetches the same HITRAN lines internally and fits them all to the
    # spectrum in one call -- the hitran_lines list from the step above is
    # for inspection only and isn't needed here
    fit_results = fit_hitran_lines(
        wavelength, spectrum,
        molecule='CO',
        wavelength_min=4.7,
        wavelength_max=4.8,
        isotopologue=1,
        v_upper=1,              # Upper vibrational level
        v_lower=0,              # Lower vibrational level
        header=cube.header,     # Required whenever a line is found -- line width
                                 # comes from the real instrument resolution
        wave_unit=cube.wave_unit,
    )

    # Check which lines were successfully detected
    detected = [r for r in fit_results if r['success']]
    print(f"Successfully fit {len(detected)} lines")


Fit Continuum
~~~~~~~~~~~~~

The GUI's continuum fitting algorithm:

.. code-block:: python

    from scipy.ndimage import median_filter
    from scipy.interpolate import UnivariateSpline
    from scipy.signal import find_peaks
    import numpy as np
    
    def fit_continuum_gui(wavelength, flux):
        """Fit continuum using GUI parameters"""
        params = {
            'window_size': 51,
            'sigma_threshold': 2.0,
            'peak_exclusion': 3,
            'iterations': 2,
            'smoothing_factor': 5
        }
        
        # Smooth and find peaks (lines)
        smoothed = median_filter(flux, size=params['window_size'], mode='reflect')
        peaks, _ = find_peaks(
            flux,
            prominence=np.std(flux) * params['sigma_threshold'],
            distance=params['peak_exclusion']
        )
        
        # Exclude line pixels
        exclude = np.zeros(len(flux), dtype=bool)
        for peak in peaks:
            left = max(0, peak - params['peak_exclusion'])
            right = min(len(flux), peak + params['peak_exclusion'] + 1)
            exclude[left:right] = True
        
        # Fit spline through continuum
        valid = ~exclude
        spline = UnivariateSpline(
            wavelength[valid],
            smoothed[valid],
            s=len(wavelength) / params['smoothing_factor'],
            k=3
        )
        return spline(wavelength)
    
    continuum = fit_continuum_gui(wavelength, spectrum)
    line_spectrum = spectrum - continuum


Create a Plot
~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot spectrum
    ax.plot(wavelength, spectrum, 'b-', label='Spectrum', alpha=0.7)
    
    # Plot continuum
    ax.plot(wavelength, continuum, 'orange', linewidth=2.5, label='Continuum')
    
    # Mark detected lines
    for line in detected:
        ax.axvline(line['wavelength'], color='red', alpha=0.3, linestyle=':')
    
    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('Flux')
    ax.set_title(f'Spectrum: {len(detected)} CO lines detected')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spectrum_analysis.png', dpi=150)
    plt.show()


Complete Example
----------------

See `example_basic_api.py` for a full working example that includes:

1. Loading FITS data
2. Fetching HITRAN CO v=1→0 lines
3. Fitting lines to spectrum
4. Fitting continuum using GUI algorithm
5. Generating visualization with markers on detected CO lines

Run it:

.. code-block:: bash

    cd /path/to/IFU-Analyzer
    python example_basic_api.py

Output: `co_analysis_nirspec.png` showing spectrum + continuum + line markers


Common Issues
-------------

**Issue: Wavelength values are Quantity objects**

If wavelength has units (astropy Quantity), extract the numeric values:

.. code-block:: python

    if hasattr(wavelength, 'value'):
        wavelength = wavelength.value
    if hasattr(spectrum, 'value'):
        spectrum = spectrum.value


**Issue: CubeData not found**

Make sure you're in the IFU-Analyzer directory or add it to Python path:

.. code-block:: python

    import sys
    sys.path.insert(0, '/path/to/IFU-Analyzer')
    from cube_explorer.data_loader import CubeData


**Issue: HITRAN fetch fails**

The first fetch downloads data (~100MB). Subsequent calls use local cache.

For specific vibrational levels, use `v_upper` and `v_lower` parameters:

.. code-block:: python

    # Get v=1→0 lines only (not all CO lines)
    hitran_lines, _ = fetch_hitran_data(
        4.7, 4.8, 'CO',
        isotopologue=1,
        v_upper=1,      # Must match
        v_lower=0       # Must match
    )


Available Functions
-------------------

Core Modules
~~~~~~~~~~~~

**cube_explorer.data_loader**

* `CubeData.load_from_file(filepath)` - Load FITS cube with auto-instrument detection

**cube_explorer.line_catalog**

* `fetch_hitran_data(wavelength_min, wavelength_max, molecule, isotopologue=1, v_upper=None, v_lower=None)` - Get HITRAN lines with optional vibrational filtering

**cube_explorer.api**

* `fit_hitran_lines(wavelength, spectrum, molecule, wavelength_min=None, wavelength_max=None, isotopologue=1, v_upper=None, v_lower=None, header=None, wave_unit=None)` - Fetch HITRAN lines for a molecule and fit them all to the spectrum in one call

**cube_explorer.gaussian_fitter**

* `fit_gaussian_to_line(x1, x2, spectrum_x, spectrum_y, peak_x, noise_min=None, noise_max=None, header=None, wave_unit=None)` - Fit single line between x1/x2, with an explicit peak wavelength


For Detailed Documentation
---------------------------

Check module docstrings:

.. code-block:: python

    from cube_explorer.line_catalog import fetch_hitran_data
    help(fetch_hitran_data)
