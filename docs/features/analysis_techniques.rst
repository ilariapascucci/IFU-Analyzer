===================
Analysis Techniques
===================

Overview of advanced analysis methods in IFU Analyzer.

Spectroscopic Analysis
======================

See :doc:`/user_guide/spectral_analysis` for detailed spectral analysis techniques.

Gaussian Line Fitting
=====================

See :doc:`/user_guide/gaussian_fitting` for detailed Gaussian fitting procedures.

Continuum Subtraction
=====================

See :doc:`/user_guide/continuum_subtraction` for continuum removal and modeling.

CO Flux Extraction
==================

See :doc:`/user_guide/co_flux_extraction` for carbon monoxide analysis.

Region-Based Analysis
=====================

Analyzing properties of specific spatial regions:

1. Select region using region selection tools
2. Extract spectrum from that region
3. Perform analysis (fitting, measurements)
4. Results specific to that region

Advantages:

- Focus on objects of interest
- Reduce noise by combining pixels
- Easier to manage large cubes
- Extract specific source properties

Pixel-by-Pixel Analysis
=======================

Full cube analysis:

1. Perform operation on every pixel
2. Create parameter maps
3. Visualize spatial distributions

Examples:

- Full-cube Gaussian fitting
- Line width maps
- Velocity maps
- Quality assessments

When to use:

- Mapping spatial variations
- Studying extended structures
- Creating publication figures
- Large sample analysis

Kinematic Analysis
==================

Extract velocity information:

**From Line Centers**

1. Fit Gaussians to spectral lines
2. Center positions give velocities
3. Create velocity maps
4. Analyze bulk motion, rotation, outflows

**Velocity Dispersion**

1. Line width maps from Gaussian fitting
2. Sigma values → velocity dispersion
3. Indicates turbulence, rotation, temperature
4. Compare across spatial regions

**Position-Velocity Diagrams**

1. Extract 1D spectrum along spatial axis
2. Display as 2D: Position vs. Velocity
3. Reveals kinematic structures
4. Shows outflows, rotation, multiple components

Flux and Brightness Analysis
=============================

**Integrated Flux**

1. Integrate spectrum over line
2. Gives total energy in line
3. Related to source luminosity
4. Used for distance measurements

**Peak Brightness**

1. Maximum flux in line
2. Indicates intensity of source
3. May indicate temperature for thermal emission

**Surface Brightness**

1. Brightness per unit solid angle
2. = Flux / beam area
3. Important for extended sources

Diagnostic Diagrams
===================

**Line Ratio Diagnostics**

Using emission line ratios to determine physical properties:

1. Measure multiple emission lines
2. Calculate line ratios
3. Plot on diagnostic grid
4. Infer physical conditions:
   - Temperature
   - Density  
   - Ionization
   - Extinction

**BPT Diagrams (Optical)**

- Emission line diagnostic for ionization source
- Separates star-forming, Seyfert 1/2, LINER, AGN
- Uses [OIII]/Hβ vs. [NII]/Hα

**Infrared Diagnostics**

- Fine structure lines for gas properties
- [SII] ratios for density
- Dust properties from continuum

Profile Analysis
================

Study line profile shapes:

1. Compare observed profile to models
2. Gaussian: Symmetric single component
3. Lorentzian: Broader wings
4. Voigt: Combination of both
5. Asymmetric: Indicates multiple components or outflows

**Absorption vs. Emission**

- Positive profile: Emission (source brighter than background)
- Negative profile: Absorption (source dimmer than background)
- Can mix in complex profiles

**Multi-Component Decomposition**

When profiles show multiple components:

1. Identify distinct velocity components
2. Fit each separately
3. Extract parameters for each
4. Interpret physically

Extinction Determination
========================

Measure dust extinction using:

1. **Balmer decrement**: Hα/Hβ ratio
2. **Dust modeling**: Compare stellar templates to observations
3. **Reddening map**: Spatial variations in extinction

Use to create dereddened spectra or correct flux measurements.

Continuum Fitting for Physical Properties
==========================================

Fits to continuum reveal:

1. **Temperature**: From blackbody shape (if thermal)
2. **Power-law**: From synchrotron or AGN emission
3. **Dust**: From infrared continuum shape

More in :doc:`/user_guide/continuum_subtraction`.

Statistical Analysis
====================

Region-based statistics:

1. Mean, median, standard deviation
2. Min, max, percentiles
3. Skewness, kurtosis
4. Fit residuals distributions

Used for:

- Data quality assessment
- Outlier detection
- Comparing datasets

Time-Domain Analysis (If Applicable)
====================================

For time-variable sources:

1. Extract spectra at multiple epochs
2. Compare spatial/spectral changes
3. Analyze variability

Advanced Techniques
===================

**3D Spectral Analysis**

Beyond simple 2D + wavelength:

1. Use external Python libraries
2. Combine spatial and spectral information
3. Apply advanced machine learning if desired

**Population Synthesis**

Model star-forming populations:

1. Compare observations to templates
2. Estimate ages, masses, dust
3. Determine star formation history

**AGN/Stellar Decomposition**

Separate AGN from stellar emission:

1. Fit stellar templates
2. Subtract stellar component
3. Analyze AGN emission separately

**Nebular/ISM Modeling**

Model interstellar medium:

1. Use photoionization codes (e.g., Cloudy, MAPPINGS)
2. Interpret emission line ratios
3. Infer ISM properties

Next Steps
==========

For specific techniques, see related user guide sections:

- Spectral analysis: :doc:`/user_guide/spectral_analysis`
- Fitting: :doc:`/user_guide/gaussian_fitting`
- Continuum: :doc:`/user_guide/continuum_subtraction`
- CO analysis: :doc:`/user_guide/co_flux_extraction`
