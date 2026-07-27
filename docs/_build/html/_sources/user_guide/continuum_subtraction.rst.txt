=======================
Continuum Subtraction
=======================

The continuum is the underlying stellar/background emission that spans the entire spectrum
(as opposed to narrow spectral lines which are superimposed on top). Removing the continuum
is essential for accurate line flux measurements.

Automatic Continuum Subtraction
================================

IFU Analyzer includes an automatic continuum fitting algorithm that:

1. Detects emission/absorption peaks (spectral lines) in the spectrum
2. Fits a smooth curve through the remaining "line-free" pixels
3. Subtracts this fitted continuum from the spectrum

**Via the GUI:**

1. Click **Continuum Tools** in the toolbar
2. Select **Automatic Continuum**
3. Choose **Continuum Line** to fit the global continuum
4. The spectrum will update showing the continuum-subtracted result

**Using the Continuum API**

Fit and subtract continuum programmatically:

.. code-block:: python

    from cube_explorer.api import load_cube, fit_continuum
    
    # Load data
    cube = load_cube("data.fits")
    spectrum = cube.spectrum
    wavelength = cube.wavelength
    
    # Fit continuum using automatic peak detection
    continuum = fit_continuum(
        wavelength,
        spectrum,
        sigma_threshold=2.0,      # Peak detection sensitivity
        peak_exclusion=3,         # Pixels to exclude near peaks
        iterations=2              # Iterative refinement
    )
    
    # Subtract to get line-only spectrum
    line_spectrum = spectrum - continuum
    
    print(f"Original spectrum range: {spectrum.min():.3f} to {spectrum.max():.3f}")
    print(f"Line-only spectrum range: {line_spectrum.min():.3f} to {line_spectrum.max():.3f}")

**Continuum Parameters**

You can fine-tune the continuum fitting with these parameters:

.. code-block:: python

    continuum = fit_continuum(
        wavelength,
        spectrum,
        window_size=51,           # Window for median filter (must be odd)
        sigma_threshold=2.0,      # Prominence threshold in sigma
        peak_distance=3,          # Minimum pixels between peaks
        peak_exclusion=3,         # Pixels excluded on each side of peak
        iterations=2,             # Number of refinement iterations
        smoothing_factor=5        # Spline smoothing parameter
    )

Higher ``sigma_threshold`` values are less aggressive (detect fewer lines).
Lower ``sigma_threshold`` values are more aggressive (detect more lines).

**Tips for Better Continuum Fits**

- If too much continuum is being subtracted, increase ``sigma_threshold``
- If not enough continuum is being subtracted, decrease ``sigma_threshold``
- For noisy data, increase ``iterations`` for more refinement
- Adjust ``window_size`` based on the width of spectral features (wider features = larger window)
