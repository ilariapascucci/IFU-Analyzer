==================
CO Flux Extraction
==================

Extract and measure carbon monoxide (CO) line fluxes from spectral data cubes. This specialized tool analyzes CO emission or absorption across multiple rotational transitions.

Introduction to CO Spectroscopy
================================

CO molecules produce distinctive emission/absorption lines:

- **Rotational transitions**: Different J quantum number transitions
- **Multiple isotopologues**:
  - ¹²CO: Most abundant
  - ¹³CO: ~1/80 abundance of ¹²CO
  - C¹⁸O: ~1/500 of ¹²CO
- **Multiple vibrational bands**:
  - Fundamental (v=1→0) transitions
  - Overtone (v=2→1) transitions (rare, very high energy)

**Common CO Lines Observed**:

- **Near-infrared (JWST/NIRSpec)**: CO fundamental band near 4.7 μm
- **Mid-infrared (JWST/MIRI)**: Various rotational lines
- **Millimeter (ALMA)**: Rotational lines at low J values

Understanding CO Flux
=====================

**What is CO Flux?**

Integrated intensity of CO line:

.. math::

   F = \int I(\lambda) d\lambda

Units typically: Jy·km/s or K·km/s

**Why Measure CO?**

- **Molecular mass**: CO traces H₂ mass (CO-to-H₂ conversion factor)
- **Outflow kinematics**: Multiple lines show velocity structure
- **Excitation**: Line ratios determine temperature/density
- **Dynamics**: Kinematic maps show bulk motion

Starting CO Flux Extraction
===========================

**Access Tool**

1. Click **CO Flux** button in toolbar
2. Or: Tools → CO Flux Extraction
3. CO Flux dialog opens

**Simple CO Flux Measurement**

1. Dialog displays:
   
   - Select CO band (Fundamental, First Overtone, etc.)
   - Expected wavelength displayed
   - Fine-tune wavelength if needed

2. Define measurement wavelength range:
   
   - Click line on spectrum
   - Or type wavelength manually
   - Specify line width (usually 10-20 Å)

3. Click **Measure**:
   
   - Integrated flux calculated
   - Result displayed with uncertainty
   - Baseline uncertainty shown

4. Click **Accept** to record measurement

Multi-Line CO Analysis
======================

**Analyze Multiple CO Transitions**

Measure multiple CO lines simultaneously:

1. Tools → CO Flux → Multi-Line Analysis
2. Select CO transitions to analyze:
   - J=1-0, J=2-1, J=3-2, etc.
   - Or J'=2-1, J'=3-2 (first overtone)
3. Wavelengths automatically populated
4. Click **Analyze All**
5. Results table shows all line fluxes

**Temperature from Line Ratios**

Estimate excitation temperature:

1. After measuring multiple lines
2. Analysis → Excitation Temperature
3. Uses line flux ratio to determine T_ex
4. Assumes local thermodynamic equilibrium (LTE)
5. Formula:

.. math::

   T_{ex} = \frac{hc\Delta E}{k \ln(N_J / g_J N_{J'} / g_{J'})}

**Optical Depth Determination**

Estimate optical depth from line ratios:

1. Tools → CO Flux → Optical Depth
2. Measures relative line opacities
3. If τ << 1: Optically thin (free emission)
4. If τ >> 1: Optically thick (absorption)

Full-Cube CO Analysis
=====================

**Analyze CO for All Pixels**

Extract CO flux maps for entire cube:

1. Tools → CO Flux → Full Cube Analysis
2. Configuration:
   - Select CO band/transition
   - Wavelength range
   - Baseline method
   - Use multi-component option?
3. Click **Start**
4. Progress bar shows completion

**Output Products**

Resulting maps:

- **CO Flux Map**: J-integrated line magnitude
- **CO Velocity Map**: Line center (kinematics)
- **CO Width Map**: Line width (σ or FWHM)
- **CO Temperature Map**: Excitation temperature
- **Optical Depth Map**: If using LTE

Interpreting CO Results
=======================

**CO Flux Calibration**

Convert flux to physical quantities:

.. math::

   M_{H_2} = \alpha_{CO} \times I_{CO}

Where:

- α_CO: CO-to-H₂ conversion factor (~2×10²⁰ cm⁻² (K·km/s)⁻¹)
- I_CO: CO integrated intensity
- M_H₂: H₂ column density

**Line Opacity**

Assess whether line is optically thick or thin:

- **Wide, symmetric profile**: Usually optically thin
- **Narrow, peaked profile**: Often optically thick
- **Blue/red asymmetry**: May indicate optical depth effects

**Multiple Transitions**

Interpret J-ladder of CO:

================== ==================== =======
Transition         Wavelength (NearIR)  Excitation
================== ==================== =======
J=1-0 (fund)       4.732 μm             87 K*
J=2-1 (fund)       4.655 μm             174 K*
J=3-2 (fund)       4.586 μm             261 K*
J=1-0 (ovt)        2.323 μm             87 K*
================== ==================== =======

*Approximate excitation energy above ground state

**Interpreting Ratios**

Line ratios indicate:

- **J=2-1/J=1-0 >> 1**: High excitation, warm gas > 100 K
- **J=2-1/J=1-0 ≈ 1**: Moderate excitation, T ~ 50-100 K
- **J=2-1/J=1-0 << 1**: Low excitation (mostly J=1), cool gas

Continuum Subtraction for CO
============================

**Important: Remove Continuum First**

Most accurate CO measurements require continuum removal:

1. Tools → Continuum Subtraction
2. Define continuum regions away from CO lines
3. Subtract before CO flux extraction
4. More accurate flux measurements

**Without Continuum Subtraction**

CO flux includes background:

- Overestimates true line flux
- Affects temperature calculations
- Use only if feature clearly isolated

CO in Different Instruments
============================

**JWST/NIRSpec**

- Fundamental CO band available
- Multiple transitions detected
- High resolution spectra
- 1-5 μm range

**JWST/MIRI**

- Various rotational transitions
- 10-25 μm range
- Multiple species potentially identifiable
- Good for low-J transitions

**ALMA**

- Millimeter/submm CO lines
- Various rotational transitions
- High spatial and spectral resolution
- Excellent for kinematic studies

**MUSE**

- Limited to optical/near-IR
- Only if extended to NIR
- Usually not primary CO detection
- Use other instruments preferentially

Advanced CO Analysis
====================

**Non-LTE Analysis**

For optically thick or non-equilibrium gas:

1. Tools → CO Flux → Non-LTE Model
2. Requires collision partner densities
3. More complex but more accurate
4. Available if observational data supports

**Multiple Velocity Component Fitting**

When CO shows distinct velocity components:

1. Tools → CO Flux → Multi-Component
2. Number of components
3. Fit each component separately
4. Measure flux, velocity, width for each

Example:

- Component 1: Systemic (V = 0)
- Component 2: Outflow redshifted (V > 0)
- Component 3: Inflow blueshifted (V < 0)

**Outflow Properties from CO**

Extract outflow physical parameters:

1. Measure CO flux in extended wings
2. Determine outflow velocity (from line shift)
3. Estimate outflow mass: M_out = α_CO × Flux
4. Estimate outflow energy and momentum

CO Flux Exporting
=================

**Export CO Measurements**

Save individual CO flux measurements:

1. File → Export Results → CO Flux
2. Format options:
   - ASCII table (wavelength, flux, error)
   - FITS table (all parameters)
   - LaTeX table (for papers)
3. Specify filename

**Export CO Maps**

Save full-cube CO analysis results:

1. Analysis → Export CO Maps
2. Select maps to save:
   - Flux, velocity, width maps
   - Temperature, optical depth maps
3. FITS file with multiple extensions

**Create CO Diagnostic Diagram**

Generate diagnostic plots:

1. Analysis → CO Diagnostics → Temperature Ratio Plot
2. Plots J=2-1/J=1-0 ratio vs. other indicators
3. Compares to theoretical models
4. Can save as PNG/PDF

Data Quality Assessment
=======================

**Check CO Measurements**

Before trusting results:

- **S/N**: Signal-to-noise in CO lines > 3
- **Baseline**: Stability of continuum
- **Line shape**: Reasonable width for object
- **Comparison**: Against literature values if available

**Identifying Problems**

Issues that affect CO flux:

================ ===================================
Problem          Solution
================ ===================================
Wrong wavelength Verify against line catalog
Blended lines    Use higher resolution data
Contamination    Use non-CO line-free regions
Poor baseline    Better continuum subtraction
Noise            Average over larger regions
================ ===================================

Common CO Challenges
===================

**Issue: CO lines appear weak**

Possible causes:

- Lower sensitivity than expected (check instrument)
- Target has intrinsically weak CO (cool gas)
- CO lines blended or overlapped
- Solution: Integrate over larger area or use longer exposure

**Issue: Multiple peaks in CO line**

Indicates velocity components:

- Systemic component
- High-velocity gas
- Outflow/inflow
- Solution: Use multi-component fitting

**Issue: Flux varies dramatically spatial**

Suggests:

- Spatial structure (clumpy emission)
- Optical depth effects
- Calibration issues
- Solution: Check individual pixel spectra

**Issue: Unusual line widths**

Consider:

- Velocity dispersion (intrinsic)
- Beam smearing (instrumental)
- Multiple components
- Solution: Compare with other data

Best Practices for CO Analysis
==============================

1. **Always measure multiple transitions** if possible
2. **Check excitation** using line ratios
3. **Consider optical depth** using line opacity
4. **Compare with literature** for validation
5. **Document assumptions** (LTE, α_CO value)
6. **Keep original data** for reproducibility
7. **Include uncertainties** in all measurements
8. **Cross-check** with other mass tracers

References
==========

For detailed CO spectroscopy:

- RADEX radiative transfer models
- DESPOTIC non-LTE analysis tool
- Literature CO coefficients
- Instrument manuals for wavelength calibrations

Next Steps
==========

After extracting CO fluxes:

1. Compare with other tracers
2. Calculate gas masses
3. Analyze kinematics from CO velocity maps
4. Determine outflow properties
5. Export results for publication
6. Read :doc:`exporting_data`
