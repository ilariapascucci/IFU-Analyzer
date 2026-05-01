"""
Rotational diagram fitting functions for H2 analysis
Implements single-temperature and two-temperature rotational diagram fits
"""

import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
import pymc as pm


# Constants
erg2ev = 6.24150647996e11  # erg -> eV
clight = 2.99792458e10  # cm/s
hplanck_eV = 4.135667696e-15  # eV*s
hplanck = hplanck_eV / erg2ev  # erg*s
pmass = 1.6726219e-24  # g
Msun = 1.989e33  # g
au = 1.496e13  # cm
pc = 3.086e18  # cm

# Molecular data for H2 (from HITRAN - these would be populated dynamically)
# For now, we'll accept these as inputs from the GUI
wave = {}  # micron
Aul = {}  # s^-1
Eu = {}  # K
gu = {}  # unitless


def get_spectral_resolution(wavelength, instrument, header=None):
    """
    Calculate spectral resolution R(λ) = λ/Δλ for different instruments.
    
    Args:
        wavelength: float or array, wavelength in microns
        instrument: str, instrument name ('NIRSPEC', 'MIRI')
        header: astropy.io.fits.Header, optional. Required for NIRSpec to get grating info
    
    Returns:
        float or array, spectral resolution R(λ)
    
    Formulas:
        MIRI: R(λ) = 4603 − 128 × λ
        NIRSpec (from grating):
            - G140H, G235H, G395H: R ~ 2700
            - G140M, G235M, G395M: R ~ 1000
            - PRISM: R ~ 100
    """
    instrument = instrument.upper()
    
    if instrument == 'MIRI':
        # MIRI spectral resolution formula
        return 4603 - 128 * wavelength
    
    elif instrument == 'NIRSPEC':
        if header is None:
            raise ValueError("NIRSpec requires FITS header for grating information")
        
        grating = header.get('GRATING', '').upper()
        
        # High resolution gratings: R ~ 2700
        if grating in ['G140H', 'G235H', 'G395H']:
            return 2700
        
        # Medium resolution gratings: R ~ 1000
        elif grating in ['G140M', 'G235M', 'G395M']:
            return 1000
        
        # Low resolution prism: R ~ 100
        elif grating == 'PRISM':
            return 100
        
        else:
            raise ValueError(f"Unknown NIRSpec grating: {grating}")
    
    else:
        raise ValueError(f"Instrument {instrument} not yet supported for spectral resolution")


def is_ortho(J):
    """Check if rotational level J is ortho (odd J)"""
    return (J % 2 == 1)


def compute_Nu(F, lam, A, R=None, dist=None):
    """
    Compute column density from flux.
    
    Args:
        F: Flux
        lam: Wavelength (micron)
        A: Einstein coefficient (s^-1)
        R: Emitting radius (cm), optional
        dist: Distance (cm), optional
    
    Returns:
        Column density Nu
    """
    lam_cm = lam * 1e-4  # micron to cm
    Nu_col = (4*np.pi * F * lam_cm) / (hplanck * clight * A)
    
    if R is not None and dist is not None:
        arcsec2 = np.pi*(R/dist)**2
        sr_arcsec2 = 4.25*(10**10)
        sr = arcsec2/sr_arcsec2
        Nu_total = Nu_col / sr
        return Nu_total
    else:
        return Nu_col


def apply_OPR(Nu, J, OPR):
    """Apply ortho-para ratio correction"""
    if OPR is None:
        return Nu

    f_ortho = OPR / (1 + OPR)
    f_para = 1 / (1 + OPR)

    return Nu / (f_ortho if is_ortho(J) else f_para)


def Q_Herbst(T):
    """Herbst partition function for H2"""
    return 0.0247 * T / (1 - np.exp(-6000/T))


def Q_Herbst_err(T, T_err):
    """Error on partition function"""
    a = 6000.0
    c = 0.0247
    ea = np.exp(-a/T)
    dQdT = c * (1/(1-ea) - (a*ea)/(T*(1-ea)**2))
    return np.abs(dQdT) * T_err


def model_1T(E, a, T):
    """Single temperature rotational diagram model"""
    return a - E / T


def model_2T(E, a1, T1, a2, T2):
    """Two temperature rotational diagram model"""
    return np.log(np.exp(a1 - E/T1) + np.exp(a2 - E/T2))


# ===========
# Upper limit
# ===========
def gaussian_flux_upper_limit(sigma_noise, wavelength, instrument=None, header=None, R=None):
    """
    Calculate upper limit flux from noise using spectral resolution.
    
    Args:
        sigma_noise: Noise standard deviation in MJy/sr (per-pixel)
        wavelength: Wavelength in microns
        instrument: str, instrument name ('NIRSPEC', 'MIRI'). If provided, calculates R automatically
        header: astropy.io.fits.Header, required for NIRSpec if instrument is specified
        R: float, spectral resolution R(λ) = λ/Δλ. Use if already calculated, overrides auto-calculation
    
    Returns:
        tuple: (F_ul, delta_lambda) - upper limit flux and wavelength resolution
    
    Formula:
        F_ul = 3 × σ_noise × λ/R
        Result is in [MJy/sr·μm]
    """
    # Calculate R if not provided
    if R is None:
        if instrument is None:
            raise ValueError("Must provide either R or instrument name")
        R = get_spectral_resolution(wavelength, instrument, header)
    
    delta_lambda = wavelength / R
    F_ul = 3.0 * sigma_noise * delta_lambda
    return F_ul, delta_lambda

def estimate_noise_local(wavelength_arr, flux_arr, line_wavelength, n_pixels=13):
    """
    Estimate per-pixel noise around a line using local continuum subtraction.

    Takes n_pixels on each side of the line center, fits a straight line
    through them, subtracts it, then returns np.nanstd of the residuals.

    Args:
        wavelength_arr: 1D array of wavelengths (microns)
        flux_arr:       1D array of flux values (MJy/sr)
        line_wavelength: Central wavelength of the line (microns)
        n_pixels:       Number of pixels to use on each side (default 13)

    Returns:
        sigma_noise: float, std dev of residuals in same units as flux_arr
                     None if not enough pixels
    """
    # Find the index closest to the line center
    center_idx = np.argmin(np.abs(wavelength_arr - line_wavelength))

    # Grab n_pixels on each side, excluding the center pixel itself
    left_idx  = np.arange(center_idx - n_pixels, center_idx)
    right_idx = np.arange(center_idx + 1, center_idx + n_pixels + 1)

    # Clip to valid array bounds
    left_idx  = left_idx[(left_idx >= 0) & (left_idx < len(wavelength_arr))]
    right_idx = right_idx[(right_idx >= 0) & (right_idx < len(wavelength_arr))]

    idx = np.concatenate([left_idx, right_idx])
    if len(idx) < 4:
        return None

    wave_region = wavelength_arr[idx]
    flux_region = flux_arr[idx]

    # Remove NaNs before fitting
    valid = np.isfinite(flux_region)
    if valid.sum() < 4:
        return None

    # Fit and subtract linear continuum
    coeffs    = np.polyfit(wave_region[valid], flux_region[valid], 1)
    residuals = flux_region - np.polyval(coeffs, wave_region)
    
    sigma_noise = float(np.nanstd(residuals))
    
    print(f"[DEBUG] estimate_noise_local:")
    print(f"[DEBUG]   center_idx={center_idx}, line_wavelength={line_wavelength:.6f}")
    print(f"[DEBUG]   left_idx range: {left_idx[0]}-{left_idx[-1]} ({len(left_idx)} pts)")
    print(f"[DEBUG]   right_idx range: {right_idx[0]}-{right_idx[-1]} ({len(right_idx)} pts)")
    print(f"[DEBUG]   wave_region: {wave_region[0]:.6f} - {wave_region[-1]:.6f}")
    print(f"[DEBUG]   flux_region min/max/mean: {np.nanmin(flux_region):.4e} / {np.nanmax(flux_region):.4e} / {np.nanmean(flux_region):.4e}")
    print(f"[DEBUG]   linear fit coeffs: slope={coeffs[0]:.6e}, intercept={coeffs[1]:.6e}")
    print(f"[DEBUG]   residuals min/max/mean: {np.nanmin(residuals):.4e} / {np.nanmax(residuals):.4e} / {np.nanmean(residuals):.4e}")
    print(f"[DEBUG]   sigma_noise = {sigma_noise:.6e}")

    return sigma_noise

def convert_flux_upper_limit_to_cgs(flux_ul_native, wavelength, header, flux_unit='MJy/sr'):
    """
    Convert upper limit flux from native units to CGS (erg/s/cm²).
    
    Designed specifically for upper limits computed as: 3σ_noise × √(2π) × σ_λ
    where the result is in [native_flux_unit × wavelength_unit].
    
    Args:
        flux_ul_native: Upper limit flux in native units × wavelength
                       (e.g., MJy/sr·μm for NIRSpec)
        wavelength: Line wavelength in microns
        header: FITS header containing CDELT1, CDELT2 (spatial pixel scales in degrees)
        flux_unit: Native flux unit ('MJy/sr' for NIRSpec/MIRI, etc.)
    
    Returns:
        float: Upper limit flux in erg/s/cm², or None if conversion fails
    
    Formula for MJy/sr·μm → erg/s/cm²:
        1. Convert [MJy/sr·μm] → [Jy·μm] using pixel solid angle:
           Ω = CDELT1 × CDELT2 × (π/180)² steradians
           flux_jy_um = flux_native × Ω × 1e-3
        2. Convert [Jy·μm] → [erg/s/cm²] using wavelength-to-frequency:
           flux_cgs = flux_jy_um × (c / λ²) × 1e-23
    """
    try:
        if header is None or flux_ul_native is None or wavelength is None:
            return None
        
        # Extract CDELT values (in degrees)
        cdelt1 = header.get('CDELT1', None)
        cdelt2 = header.get('CDELT2', None)
        
        if cdelt1 is None or cdelt2 is None:
            return None
        
        cdelt1_abs = abs(float(cdelt1))
        cdelt2_abs = abs(float(cdelt2))
        
        # Check if units are MJy/sr
        if 'MJY' not in flux_unit.upper() or 'SR' not in flux_unit.upper():
            return None
        
        # Step 1: Convert [MJy/sr·μm] → [Jy·μm]
        # Solid angle of one pixel in steradians
        steradian_per_degree2 = (np.pi / 180.0) ** 2
        omega_pixel_sr = cdelt1_abs * cdelt2_abs * steradian_per_degree2
        
        # Remove "/sr" and convert MJy → Jy
        flux_jy_um = flux_ul_native * omega_pixel_sr * 1e6
        
        # Step 2: Convert [Jy·μm] → [erg/s/cm²]
        # Using F_ν = F_λ × (c / λ²)
        c_micron_per_s = 2.99792458e14  # speed of light in μm/s
        flux_cgs = flux_jy_um * (c_micron_per_s / (wavelength ** 2)) * 1e-23
        
        return flux_cgs
    
    except Exception:
        return None


def flux_to_logNu_upper(F_ul, lam, Aul_val, gu_val, R=None, dist=None):
    lam_cm = lam * 1e-4
    Nu_ul = (4*np.pi * F_ul * lam_cm) / (hplanck * clight * Aul_val)

    if R is not None and dist is not None:
        arcsec2 = np.pi * (R / dist)**2
        sr_arcsec2 = 4.25*(10**10)
        Nu_ul = Nu_ul / (arcsec2 / sr_arcsec2)

    return np.log(Nu_ul / gu_val)


def rotational_diagram(fluxes, flux_errors, molecular_data=None, OPR=None, fit2T=False, R=None, dist=140*pc, upper_limits=None):
    """
    Perform rotational diagram analysis on H2 lines.
    
    Args:
        fluxes: Dictionary {J: flux_value} where J is the upper rotational level
        flux_errors: Dictionary {J: flux_error}
        molecular_data: Dictionary with 'wavelengths', 'Aul', 'Eu', 'gu' keys
                       Each value is a dict {J: value}
        OPR: Ortho-para ratio (None to skip correction)
        fit2T: Whether to perform 2-temperature fit
        R: Emitting radius (AU or cm).
        dist: Distance (pc or cm).
        upper_limits: Optional dict {J: True} for measurements marked as upper limits.
                      If present, PyMC censored regression is automatically used.
    
    Returns:
        Dictionary with fitting results
    
    Notes:
        - If no upper limits: uses fast curve_fit (scipy)
        - If upper limits present: automatically uses PyMC censored likelihood
    """
    
    global wave, Aul, Eu, gu

    if molecular_data is not None:
        wave = molecular_data.get('wavelengths', {})
        Aul = molecular_data.get('Aul', {})
        Eu = molecular_data.get('Eu', {})
        gu = molecular_data.get('gu', {})

    E_arr, y_arr, y_err = [], [], []
    is_ul_mask = []

    # =========================
    # Build dataset
    # =========================
    for J in fluxes:
        if J not in wave:
            continue

        F = fluxes[J]
        Nu = compute_Nu(F, wave[J], Aul[J], R, dist)
        y = np.log(Nu / gu[J])

        is_ul = upper_limits is not None and J in upper_limits
        
        # Skip detections without error data - no defaults allowed
        if (not is_ul) and (J not in flux_errors):
            print(f"J={J}: skipping (detection without error data)")
            continue

        print(f"J={J}, Eu={Eu[J]:.1f}, F={F:.4e}, is_ul={is_ul}")
        
        E_arr.append(Eu[J])
        y_arr.append(y)
        is_ul_mask.append(is_ul)

        # error handling
        if (not is_ul) and (J in flux_errors):
            dF = flux_errors[J]
            dNu = Nu * (dF / F)
            y_err.append(dNu / Nu)
        else:
            # upper limits: treated as constraints with 0 error
            y_err.append(0.0)

    E_arr = np.array(E_arr)
    y_arr = np.array(y_arr)
    y_err = np.array(y_err)
    is_ul_mask = np.array(is_ul_mask)

    if len(E_arr) < 2:
        raise ValueError("Need at least 2 points")

    # =========================
    # Decide method: auto-use PyMC if upper limits present
    # =========================
    has_upper_limits = np.any(is_ul_mask)
    use_pymc = has_upper_limits

    # ==========================================================
    # 1T FIT
    # ==========================================================
    if not use_pymc:

        print("\n--- 1T FIT (curve_fit) ---")

        sigma = y_err if np.any(y_err > 0) else None

        print(f"[DEBUG] y_err passed to curve_fit: {y_err}")

        popt1, pcov1 = curve_fit(
            model_1T,
            E_arr,
            y_arr,
            sigma=sigma,
            absolute_sigma=True,
            p0=[np.max(y_arr), 500]
        )

        a1, T1 = popt1
        da1, dT1 = np.sqrt(np.diag(pcov1))

    else:

        print("\n--- 1T FIT (CENSORED PyMC) ---")

        result = censored_1T(
            E_arr, y_arr, y_err, is_ul_mask
        )

        a1, T1 = result["a"], result["T"]
        da1, dT1 = result["da"], result["dT"]

    # =========================
    # Derived quantities
    # =========================
    Ncol1 = np.exp(a1)
    dNcol1 = Ncol1 * da1

    Ntot1 = Ncol1 * Q_Herbst(T1)
    dNtot1 = Ntot1 * np.sqrt((dNcol1/Ncol1)**2 + (Q_Herbst_err(T1, dT1)/Q_Herbst(T1))**2)

    results = {
        "1T": {
            "T": T1,
            "dT": dT1,
            "Ncol": Ntot1,
            "dNcol": dNtot1
        }
    }

    # ==========================================================
    # 2T FIT
    # ==========================================================
    if fit2T:

        if not use_pymc:

            print("\n--- 2T FIT (curve_fit) ---")

            popt2, pcov2 = curve_fit(
                model_2T,
                E_arr,
                y_arr,
                sigma=sigma,
                absolute_sigma=True,
                p0=[a1, T1, a1-5, T1*2],
                maxfev=10000
            )

            a1_2, T1_2, a2_2, T2_2 = popt2
            errs = np.sqrt(np.diag(pcov2))

        else:

            print("\n--- 2T FIT (CENSORED PyMC) ---")

            result = censored_2T(
                E_arr, y_arr, y_err, is_ul_mask,
                a1, T1
            )

            a1_2, T1_2, a2_2, T2_2 = result["popt"]
            errs = result["errs"]

        da1_2, dT1_2, da2_2, dT2_2 = errs

        N1 = np.exp(a1_2)
        N2 = np.exp(a2_2)

        N1_tot = N1 * Q_Herbst(T1_2)
        N2_tot = N2 * Q_Herbst(T2_2)

        dN1_tot = N1_tot * np.sqrt(
            (N1 * da1_2 / N1)**2 +
            (Q_Herbst_err(T1_2, dT1_2) / Q_Herbst(T1_2))**2
        )
        dN2_tot = N2_tot * np.sqrt(
            (N2 * da2_2 / N2)**2 +
            (Q_Herbst_err(T2_2, dT2_2) / Q_Herbst(T2_2))**2
        )

        results["2T"] = {
            "T1": T1_2,
            "dT1": dT1_2,
            "T2": T2_2,
            "dT2": dT2_2,
            "Ncol1": N1_tot,
            "dNcol1": dN1_tot,
            "Ncol2": N2_tot,
            "dNcol2": dN2_tot,
        }

        results["_plot_data"] = {
            "E_arr": E_arr,
            "y_arr": y_arr,
            "y_err": y_err,
            "is_ul_mask": is_ul_mask,
            "popt1": list(popt1) if not use_pymc else [a1, T1],
            "popt2": list(popt2) if not use_pymc else list(result["popt"]),
        }
    else:
        results["_plot_data"] = {
            "E_arr": E_arr,
            "y_arr": y_arr,
            "y_err": y_err,
            "is_ul_mask": is_ul_mask,
            "popt1": list(popt1) if not use_pymc else [a1, T1],
            "popt2": None,
        }

    return results


# ==========================================================
# CENSORED MODELS (ONLY USED IF ENABLED)
# ==========================================================
def censored_1T(E, y, yerr, is_ul):
    """Bayesian 1-temperature rotational diagram fit with censored upper limits.
    
    Uses PyMC to handle mixed detections and upper limits:
    - Detections: fit normally with likelihood
    - Upper limits: one-sided constraint (F < F_ul)
    """
    # mask = True for detections, False for upper limits
    mask = ~is_ul

    with pm.Model() as model:

        # === PRIORS ===
        # log(Ncol) ~ Normal(data mean, sigma=5 dex)
        a = pm.Normal("a", mu=np.mean(y[mask]), sigma=5)
        # Temperature range: 10 K to 5000 K (reasonable for H2)
        T = pm.Uniform("T", 10, 5000)

        # === LINEAR MODEL ===
        # rotational diagram: ln(N/g) = a - E_u/T
        mu = a - E / T

        # === LIKELIHOOD FOR DETECTIONS ===
        # Fit only the non-upper-limit points (mask=True)
        # Assume σ = 1.0 in log space for all detections
        pm.Normal("obs", mu=mu[mask], sigma=1.0, observed=y[mask])

        # === LIKELIHOOD FOR UPPER LIMITS ===
        # Censored: treat measured value as upper bound
        # y_ul < F_ul means ln(N/g)_meas is an upper limit on the true value
        if np.any(is_ul):
            pm.Censored(
                "ul",
                # Normal distribution centered at model prediction
                pm.Normal.dist(mu=mu[is_ul], sigma=1.0),
                # Upper censoring: observed value is upper bound
                upper=y[is_ul],
                observed=y[is_ul]
            )

        # === SAMPLING ===
        # Draw 500 posterior samples (+ 500 tuning steps) from joint posterior
        trace = pm.sample(500, tune=500, chains=1, cores=1, progressbar=False)

    # === RETURN POSTERIOR ESTIMATES ===
    return {
        "a": float(trace.posterior["a"].mean()),        # log(Ncol)
        "da": float(trace.posterior["a"].std()),        # uncertainty
        "T": float(trace.posterior["T"].mean()),        # temperature
        "dT": float(trace.posterior["T"].std())         # uncertainty
    }


def censored_2T(E, y, yerr, is_ul, a1_init, T1_init):
    """Bayesian 2-temperature rotational diagram fit with censored upper limits.
    
    Fits a mixture of two thermal components:
    ln(N/g) = log(exp(a1 - E/T1) + exp(a2 - E/T2))
    
    Args:
        E, y, yerr, is_ul: same as censored_1T
        a1_init, T1_init: initial guesses from 1T fit (for better prior)
    """
    # mask = True for detections, False for upper limits
    mask = ~is_ul

    with pm.Model() as model:

        # === PRIORS (2-component model) ===
        # Component 1: centered near the 1T result
        a1 = pm.Normal("a1", mu=a1_init, sigma=5)           # log(Ncol) warm component
        T1 = pm.Uniform("T1", 10, 5000)                     # temp warm component
        
        # Component 2: cooler component (a2 is lower, T2 is higher)
        a2 = pm.Normal("a2", mu=a1_init-5, sigma=5)         # log(Ncol) cool component
        T2 = pm.Uniform("T2", 10, 5000)                     # temp cool component

        # === TWO-TEMPERATURE MODEL ===
        # Combine two exponential terms in log space:
        # μ = ln(exp(a1 - E/T1) + exp(a2 - E/T2))
        # This is a mixture of two Boltzmann distributions
        mu = pm.math.log(
            pm.math.exp(a1 - E/T1) +
            pm.math.exp(a2 - E/T2)
        )

        # === LIKELIHOOD FOR DETECTIONS ===
        # Fit only detections; upper limits are censored
        pm.Normal("obs", mu=mu[mask], sigma=1.0, observed=y[mask])

        # === LIKELIHOOD FOR UPPER LIMITS ===
        # Same censoring approach: y_measured is upper bound on true value
        if np.any(is_ul):
            pm.Censored(
                "ul",
                pm.Normal.dist(mu=mu[is_ul], sigma=1.0),
                upper=y[is_ul],
                observed=y[is_ul]
            )

        # === SAMPLING ===
        # Draw posterior samples
        trace = pm.sample(500, tune=500, chains=1, cores=1, progressbar=False)

    # === RETURN POSTERIOR ESTIMATES ===
    return {
        # Best-fit parameters (posterior means)
        "popt": np.array([
            float(trace.posterior["a1"].mean()),     # log(Ncol1)
            float(trace.posterior["T1"].mean()),     # T_warm
            float(trace.posterior["a2"].mean()),     # log(Ncol2)
            float(trace.posterior["T2"].mean())      # T_cool
        ]),
        # Uncertainties (posterior stds)
        "errs": np.array([
            float(trace.posterior["a1"].std()),      # δlog(Ncol1)
            float(trace.posterior["T1"].std()),      # δT_warm
            float(trace.posterior["a2"].std()),      # δlog(Ncol2)
            float(trace.posterior["T2"].std())       # δT_cool
        ])
    }