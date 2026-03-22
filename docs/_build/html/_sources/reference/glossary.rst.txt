========
Glossary
========

Glossary of terms and definitions used in IFU Analyzer and IFU spectroscopy.

.. glossary::

   Amplitude
      In Gaussian fitting, the peak intensity of a fitted line profile. Units depend on data (Jy, erg/s/cm²/Å, K, etc.).

   Area
      Integrated flux of a spectral line (area under curve). Also called equivalent flux or integrated intensity.

   Baseline
      The underlying continuum level in a spectrum. Can be subtracted to isolate line features.

   Beam
      The effective resolution element of an instrument (in pixels or arcseconds).

   Blackbody
      Model of thermal emission from a hot object. Emission follows Planck distribution. Parameter: Temperature.

   BPT Diagram
      Emission line diagnostic using [OIII]/Hβ vs. [NII]/Hα to classify ionization source (star-forming, AGN, LINER).

   Centroid
      The center location of a feature; for Gaussians, corresponds to line center wavelength.

   Chi-Squared (χ²)
      Measure of goodness-of-fit between data and model. Lower values indicate better fits.

   Colormap
      Mapping from data values to colors for visualization.

   Coordinate Systems  
      Methods of specifying positions in data. IFU Analyzer supports wavelength and RA/DEC coordinates.

   Continuum
      Slowly-varying background emission in a spectrum, separate from line features.

   Cube
      Three-dimensional dataset with spatial (RA, Dec) and spectral (wavelength) dimensions.

   Declination (Dec)
      One of two celestial coordinates, analogous to latitude (-90° to +90°).

   Equivalent Width (EW)
      Measure of line strength; width × (continuum - line) / continuum. Units: Å or km/s.

   Excitation Temperature (T_ex)
      Temperature derived from emission line ratios, assuming local thermodynamic equilibrium.

   FITS
      Flexible Image Data System; standard format for astronomical data storage.

   FWHM
      Full Width at Half Maximum. For Gaussians: FWHM = 2.355 × σ. Measure of line width.

   Gaussian
      Mathematical function (bell curve) used to model spectral line profiles. Often appropriate for emission lines.

   Hα, Hβ
      Balmer series hydrogen emission lines frequently used in optical spectroscopy.

   HDU
      Header Data Unit. A section of a FITS file containing data and associated header keywords.

   Histogram
      Graph of data value distribution,used for understanding data range and setting display scales.

   IFU
      Integral Field Unit; instrument providing spatially-resolved spectroscopy over extended region.

   Integral Field Spectroscopy (IFS)
      Technique providing position (RA, Dec) and spectrum (wavelength) simultaneously across extended field.

   Intensity Map
      2D visualization of data at a specific wavelength or integrated over range.

   Ionization
      Process of removing electrons from atoms, usually via energetic photons or collisions.

   Kinematics
      Study of gas motion (velocity, dynamics). Revealed via line center shifts in spectral data.

   Line Fitting
      Process of modeling spectral line profiles with mathematical functions (usually Gaussians).

   Lorentzian
      Alternative line profile model with broader wings than Gaussian.

   LTE
      Local Thermodynamic Equilibrium. Assumption that matter is in thermal equilibrium locally.

   Magnitude
      Measure of brightness on logarithmic scale (brighter objects have lower magnitudes).

   MIRI
      Mid-Infrared Instrument on JWST. Covers 5-27 μm.

   MUSE
      Multi Unit Spectroscopic Explorer on VLT. Covers 4750-9350 Å in optical.

   NIRSpec
      Near-infrared Spectrograph on JWST. Covers 0.6-5.3 μm.

   Normalization
      Dividing data by a reference value (e.g., median, maximum) to remove intensity variations.

   Optical Depth (τ)
      Measure of how opaque a medium is to radiation. τ >> 1 is optically thick, τ << 1 is optically thin.

   Photometry
      Measurement of light intensity; brightness measurement

   Pixel
      Picture element; basic unit of digital images.

   Polarimetry
      Measurement of light polarization properties.

   Power-Law
      Model for data following power relationship: f(x) ∝ x^α.

   PSF
      Point Spread Function. How instrument responds to point source; sets resolution limit.

   RA
      Right Ascension; one of two celestial coordinates, analogous to longitude (0°-360° or 0h-24h).

   Redshift (z)
      Measure of how much light is shifted due to cosmic expansion. z = (λ_observed - λ_rest) / λ_rest.

   Residuals
      Difference between observed data and model fit. Used to assess fit quality.

   RGB Composite
      Color image made from three monochromatic images assigned to Red, Green, Blue channels.

   Robustness
      Resistance to outliers. Robust fitting downweights extreme values.

   Reddening
      Preferential absorption of short wavelengths by dust, making objects appear redder.

   Sigma (σ)
      In Gaussian profile, the standard deviation. Related to line width: FWHM = 2.355 × σ.

   Signal-to-Noise Ratio (S/N)
      Ratio of signal power to noise power. Higher is better. Used to assess data quality.

   Spectrograph
      Instrument that disperses light by wavelength to create spectra.

   Spectrum
      Graph of intensity vs. wavelength (or frequency, velocity).

   Spline
      Piecewise polynomial providing smooth curve through data points.

   Stokes Parameters
      Representation of light polarization state (I, Q, U, V components).

   Synchrotron
      Non-thermal radiation process from electrons accelerated by magnetic fields.

   Thermal Emission
      Radiation from hot matter; blackbody is special case.

   Velocity Dispersion
      Spread in velocities of gas; related to line width.

   Wavelength Calibration
      Process of determining relationship between pixel position and wavelength.

   WCS
      World Coordinate System. System for converting pixel coordinates to sky coordinates (RA/Dec).

   Yaw/Pitch/Roll
      Rotational angles for satellite orientation control.

Related Resources
==================

For more definitions:

- Astronomy glossaries online
- FITS standard documentation
- Instrument documentation from telescopes
- Spectroscopy textbooks
