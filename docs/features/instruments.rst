==========
Instruments
==========

Detailed information about supported observatories and instruments.

MUSE (VLT)
==========

**Multi Unit Spectroscopic Explorer** (VLT, Paranal)

- **Wavelength Range**: 4750-9350 Å (optical)
- **Spectral Resolution**: R ~ 3000
- **Spatial Resolution**: 0.2 arcsec/pixel (nominal)
- **Field of View**: 1×1 arcmin (standard)
- **Data Format**: FITS cubes
- **Primary Science**: Integral field spectroscopy of galaxies, AGN, star-forming regions

MUSE supports:
- Both wide-field and narrow-band imaging
- Spectropolarimetry modes
- Multiple IFU modes

NIRSpec (JWST)
==============

**Near-Infrared Spectrograph** (JWST)

- **Wavelength Range**: 0.6-5.3 μm (near-infrared)
- **Spectral Resolution**: R ~ 1000-2700 (varies by grating)
- **Spatial Resolution**: ~0.1 arcsec/pixel
- **Integral Field Unit**: 3×3 arcsec field
- **Data Format**: FITS cubes, Stage 3 products
- **Primary Science**: High-redshift galaxies, star formation, exoplanet atmospheres

NIRSpec features:
- Multiple gratings for different wavelength ranges
- Multi-object spectroscopy capability
- Fixed slit and IFU modes

MIRI (JWST)
===========

**Mid-Infrared Instrument** (JWST)

- **Wavelength Range**: 5-27 μm (mid-infrared)
- **Spectral Resolution**: R ~ 1500-3500
- **Spatial Resolution**: 0.11 - 0.35 arcsec/pixel
- **Field of View**: Varies with channel
- **Channels**: Multiple spectroscopic channels
- **Data Format**: FITS cubes, Stage 3 products
- **Primary Science**: Dust emission, high-z galaxies, evolved stars

MIRI spectroscopy:
- Long wavelength channel (~ 10-27 μm)
- Short wavelength channel (~ 5-10 μm)
- High thermal background
- Improved sky subtraction required

ALMA
====

**Atacama Large Millimeter Array**

- **Wavelength Range**: Millimeter to submillimeter (0.3-10 mm)
- **Spectral Resolution**: Varies with band (can be very high)
- **Configuration**: Multiple antenna arrays
- **Data Format**: FITS format (similar to single-dish)
- **Primary Science**: Molecular gas, continuum emission, high-z objects

ALMA considerations:
- Interferometric data requires imaging/deconvolution
- Continuum and line data available
- Spectral lines resolved in velocity space
- Requires careful handling of synthesized beam

Generic/Custom Data
====================

For data from other instruments:

1. Must be in FITS format
2. Supports 2D and 3D data
3. WCS coordinates optional but recommended
4. Specify as "Generic" upon loading

Custom data requirements:
- Valid FITS structure
- Properly defined coordinate systems
- Wavelength calibration in header
- Flux calibration preferred

Instrument-Specific Tips
========================

**MUSE Data**

- Data already flux-calibrated
- Wavelength solution excellent
- Watch for sky contamination in outer wavelengths
- Use flux calibration files if available

**NIRSpec Data**

- Multiple gratings may need separate analysis
- Compare with photometry for flux checks
- Non-linear detector effects possible
- Use official pipeline data (Stage 3)

**MIRI Data**

- High thermal background requires careful masking
- Different orders may overlap
- Check for bad pixels
- Temperature variations affect flux

**ALMA Data**

- May need imaging before analysis
- Continuum and line data may be separate
- Velocity coordinates often more useful than wavelength
- Beam characteristics important

Cross-Instrument Analysis
=========================

**Combining Data from Multiple Instruments**

Multi-wavelength analysis:

1. Load first instrument's data
2. Analyze and save results
3. Load second instrument
4. Compare results between instruments
5. Create multi-wavelength composite

**Wavelength Coverage**

Typical multi-wavelength sequences:

- **Optical** (MUSE): 4750-9350 Å study nebular lines
- **NIR** (NIRSpec): 1-5 μm for molecular hydrogen, CO
- **MIR** (MIRI): 5-27 μm for dust, thermal emission
- **mm** (ALMA): High velocity resolution CO or continuum

**Flux Calibration Across Instruments**

When combining data:

- Verify all data flux-calibrated (typically Jy or erg/s/cm²/Å)
- Re-scale if necessary to common units
- Account for different beam sizes if needed
- Document flux scaling factors

**Coordinate Systems**

Ensure consistent coordinates:

- Use RA/DEC when available
- Account for astrometric uncertainties
- Match spatial pixels for similar sources
- Document coordinate reference frame

Choosing the Right Instrument
=============================

For different science goals:

**Emission Line Diagnostics**

→ Use MUSE (optical lines) or MIRI (mid-IR forbidden lines)

**Molecular Gas Properties**

→ Use NIRSpec (CO, H₂) or ALMA (CO, tracers)

**Kinematics and Dynamics**

→ Use ALMA (high resolution) or NIRSpec (multiple transitions)

**Continuum/Dust Properties**

→ Use MIRI (thermal dust) or ALMA (continuum)

**High Redshift Objects**

→ Use NIRSpec or MIRI (shifted into observed infrared)

Further Reading
===============

For detailed instrument documentation:

- MUSE: Official MUSE documentation (ESO)
- NIRSpec: JWST NIRSpec documentation (STScI)
- MIRI: JWST MIRI documentation (STScI)
- ALMA: ALMA documentation (ESO/NRAO/NAOJ)

See also: :doc:`/user_guide/loading_data` for instrument selection during file loading.
