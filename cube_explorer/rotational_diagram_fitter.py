"""
Rotational diagram fitting functions for H2 analysis
Implements single-temperature and two-temperature rotational diagram fits
"""

import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize



'''
CONSTANTS
'''

erg2ev = 6.24150647996e11  # erg -> eV
clight = 2.99792458e10  # cm/s
hplanck_eV = 4.135667696e-15  # eV*s
hplanck = hplanck_eV / erg2ev  # erg*s
pmass = 1.6726219e-24  # g
Msun = 1.989e33  # g
au = 1.496e13  # cm
pc = 3.086e18  # cm



'''
HITRAN DATA
'''

# Molecular data for H2 (from HITRAN - these would be populated dynamically)
# For now, we'll accept these as inputs from the GUI
wave = {}  # micron
Aul = {}  # s^-1
Eu = {}  # K
#gu = {}  # unitless



'''
FORMULAS
'''

def is_ortho(J):
    """Check if rotational level J is ortho (odd J)"""
    return (J % 2 == 1)

def get_Ju(SJ):
    """Upper rotational level for H2 S-branch line S(SJ)"""
    return SJ + 2

def get_gu(SJ):
    """Rotational-only degeneracy (2*Ju+1), nuclear spin excluded so apply_OPR can supply it"""
    Ju = get_Ju(SJ)
    #gs = 3 if is_ortho(Ju) else 1
    return (2*Ju + 1)

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

    if R is not None:
        arcsec2 = np.pi*((R/dist)**2)
        sr_arcsec2 = 4.25*(10**10)
        sr = arcsec2/sr_arcsec2
        Nu_total = Nu_col / sr
        return Nu_total
    else:
        return Nu_col*((dist*pc)**2)


def apply_OPR(Nu, J, OPR):
    """Apply ortho-para ratio correction"""
    if OPR is None:
        OPR = 3.0

    if isinstance(OPR, (int, float)):
        opr_val = OPR
    else:
        raise ValueError("OPR must be None or a float (use 'fit' in wrapper)")

    f_ortho = opr_val  # / (1 + opr_val)
    f_para  = 1.0  # / (1 + opr_val)

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




















'''
UPPER LIMITS
'''

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
        F_ul = 3 × σ_noise × Δλ
        where Δλ = λ/R (spectral resolution element)
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

    return sigma_noise


def convert_flux_upper_limit_to_cgs(flux_ul_native, wavelength, header, flux_unit='MJy/sr'):
    """
    Convert upper limit flux from native units to CGS (erg/s/cm²).
    
    Designed specifically for upper limits computed as: 3σ_noise × Δλ
    where Δλ = λ/R (spectral resolution element) and the result is in
    [native_flux_unit × wavelength_unit].
    
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
           flux_jy_um = flux_native × Ω × 1e6
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


def compute_line_flux(wavelength_arr, spectrum_arr, line_wavelength, header, instrument):
    """
    Fit a single line in an (already spatially-summed) spectrum and compute its
    integrated flux, flux uncertainty, and upper-limit status, all in CGS units.

    Args:
        wavelength_arr: 1D wavelength array (microns), increasing order
        spectrum_arr: 1D flux array (MJy/sr), matching wavelength_arr
        line_wavelength: Target line center wavelength (microns)
        header: FITS header (for spectral resolution + unit conversion)
        instrument: Instrument name ('MIRI', 'NIRSPEC', ...)

    Returns:
        dict with keys:
            success (bool), message (str or None),
            flux_cgs, flux_error_cgs, is_upper_limit, F_ul_cgs,
            sigma_noise, delta_lambda
        On failure (fit or unit-conversion failure), success=False, message
        explains why, and the numeric fields are absent.
    """
    from .gaussian_fitter import fit_gaussian_to_line, convert_area_to_cgs, estimate_flux_uncertainty_montecarlo

    wl = line_wavelength

    # Get spectral resolution and define a wavelength window (+/-2*delta_lambda) around the line
    R = get_spectral_resolution(wl, instrument=instrument, header=header)
    delta_lambda = wl / R
    window_width = delta_lambda * 2
    x1 = wl - window_width
    x2 = wl + window_width

    # Fit gaussian (native units, no header conversion)
    fit_result = fit_gaussian_to_line(
        x1=x1, x2=x2,
        spectrum_x=wavelength_arr,
        spectrum_y=spectrum_arr,
        peak_x=wl,
        header=None
    )

    if not fit_result.success:
        return {"success": False, "message": f"Gaussian fit failed: {fit_result.message}"}

    # Convert area to CGS
    flux_integrated = convert_area_to_cgs(
        fit_result.amplitude, fit_result.sigma, fit_result.center,
        header=header, wave_unit='UM', flux_unit_input='MJy/sr'
    )
    if flux_integrated is None:
        return {"success": False, "message": "Could not convert flux to CGS"}

    # Monte Carlo flux uncertainty (native units), converted to CGS using the same factor as flux
    flux_error_native = estimate_flux_uncertainty_montecarlo(x1, x2, wavelength_arr, spectrum_arr, wl)
    flux_error = convert_flux_upper_limit_to_cgs(flux_error_native, wl, header, flux_unit='MJy/sr')
    if flux_error is None:
        return {"success": False, "message": "Could not convert flux error to CGS"}

    # Auto-detect upper limit
    sigma_noise = estimate_noise_local(wavelength_arr, spectrum_arr, wl, n_pixels=13)
    if sigma_noise is None:
        sigma_noise = np.nanstd(spectrum_arr[len(spectrum_arr)//2:])

    F_ul_native, delta_lambda_ul = gaussian_flux_upper_limit(sigma_noise, wl, instrument=instrument, header=header)
    F_ul_cgs = convert_flux_upper_limit_to_cgs(F_ul_native, wl, header, flux_unit='MJy/sr')

    is_upper_limit = flux_integrated < F_ul_cgs

    print(f"[BATCH_FLUX] Line @ {wl:.6f} μm:")
    print(f"  flux = {flux_integrated:.4e} erg/s/cm²")
    print(f"  error = {flux_error:.4e} erg/s/cm²")
    print(f"  noise (native) = {sigma_noise:.4e} MJy/sr")
    print(f"  delta_lambda = {delta_lambda_ul:.4e} μm")
    print(f"  F_ul (CGS) = {F_ul_cgs:.4e} erg/s/cm²")
    print(f"  is_upper_limit = {is_upper_limit}")

    return {
        "success": True,
        "message": None,
        "flux_cgs": flux_integrated,
        "flux_error_cgs": flux_error,
        "is_upper_limit": is_upper_limit,
        "F_ul_cgs": F_ul_cgs,
        "sigma_noise": sigma_noise,
        "delta_lambda": delta_lambda_ul,
    }


# ==========================================================
# CONSTRAINED LEAST-SQUARES MODELS (used whenever upper limits are present)
# ==========================================================
def _chi2_1T(params, E, y, yerr):
    a, T = params
    return np.sum(((y - model_1T(E, a, T)) / yerr) ** 2)


def _ul_ceiling_1T(params, E_ul, y_ul):
    """>= 0 iff the model prediction stays at or below the upper limit."""
    a, T = params
    return y_ul - model_1T(E_ul, a, T)


def _chi2_2T(params, E, y, yerr):
    a1, T1, a2, T2 = params
    return np.sum(((y - model_2T(E, a1, T1, a2, T2)) / yerr) ** 2)


def _ul_ceiling_2T(params, E_ul, y_ul):
    a1, T1, a2, T2 = params
    return y_ul - model_2T(E_ul, a1, T1, a2, T2)


def constrained_1T(E, y, yerr, is_ul, n_boot=500, seed=0):
    """1-temperature rotational diagram fit that respects upper limits as
    one-sided ceiling constraints.

    Fits the detections with ordinary weighted least squares first. If that
    unconstrained fit already predicts values at or below every upper
    limit (the common case for a genuine non-detection), it's returned
    as-is - point estimate and covariance both come straight from
    curve_fit, identical to the no-upper-limits path. Only when the
    unconstrained fit would exceed an upper limit does this re-solve under
    an explicit "prediction <= F_ul" constraint (SLSQP), with uncertainties
    then estimated via a parametric bootstrap: resample each detection
    within its flux error, re-solve the same constrained problem, take the
    spread of solutions.
    """
    mask = ~is_ul
    E_d, y_d, yerr_d = E[mask], y[mask], yerr[mask]
    E_ul, y_ul = E[is_ul], y[is_ul]

    lo, hi = np.min(y_d), np.max(y_d) + 10
    p0 = [np.max(y_d), 500]

    popt0, pcov0 = curve_fit(model_1T, E_d, y_d, sigma=yerr_d, absolute_sigma=True,
                              p0=p0, bounds=([lo, 10], [hi, 3000]))

    if len(E_ul) == 0 or np.all(model_1T(E_ul, *popt0) <= y_ul):
        a, T = popt0
        da, dT = np.sqrt(np.diag(pcov0))
        print("  1T: no upper limit is actively binding - identical to the unconstrained fit")
        return {"a": float(a), "da": float(da), "T": float(T), "dT": float(dT),
                "n_active_constraints": 0, "used_bootstrap": False}

    bounds = [(lo, hi), (10, 3000)]
    constraints = [{"type": "ineq", "fun": _ul_ceiling_1T, "args": (E_ul, y_ul)}]

    res = minimize(_chi2_1T, x0=popt0, args=(E_d, y_d, yerr_d),
                    method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"Constrained 1T fit failed: {res.message}")
    a, T = res.x
    n_active = int(np.sum(model_1T(E_ul, a, T) >= y_ul - 1e-8))

    rng = np.random.default_rng(seed)
    boot_a, boot_T = [], []
    for _ in range(n_boot):
        y_d_boot = rng.normal(y_d, yerr_d)
        res_b = minimize(_chi2_1T, x0=[a, T], args=(E_d, y_d_boot, yerr_d),
                          method="SLSQP", bounds=bounds, constraints=constraints)
        if res_b.success:
            boot_a.append(res_b.x[0])
            boot_T.append(res_b.x[1])

    da = float(np.std(boot_a)) if boot_a else float("nan")
    dT = float(np.std(boot_T)) if boot_T else float("nan")

    print(f"  1T: {n_active} upper limit(s) actively constraining the fit")
    print(f"  Bootstrap uncertainty: {len(boot_a)}/{n_boot} resamples converged")

    return {
        "a": float(a), "da": da,
        "T": float(T), "dT": dT,
        "n_active_constraints": n_active,
        "used_bootstrap": True,
    }





def constrained_2T(E, y, yerr, is_ul, a1_init, T1_init, n_boot=500, seed=0):
    """2-temperature analogue of constrained_1T: fits a mixture of two
    thermal components (see model_2T) subject to the same one-sided
    upper-limit ceiling constraints, falling back to the plain
    unconstrained fit when no upper limit is actually violated.

    Args:
        E, y, yerr, is_ul: same as constrained_1T
        a1_init, T1_init: initial guess from the 1T fit (component 1 seed;
                           component 2 is seeded cooler/fainter from it)
    """
    mask = ~is_ul
    E_d, y_d, yerr_d = E[mask], y[mask], yerr[mask]
    E_ul, y_ul = E[is_ul], y[is_ul]

    lo, hi = np.min(y_d), np.max(y_d) + 10
    p0 = [a1_init, T1_init, a1_init - 5, T1_init * 2]

    popt0, pcov0 = curve_fit(model_2T, E_d, y_d, sigma=yerr_d, absolute_sigma=True,
                              p0=p0, bounds=([lo, 10, lo, 10], [hi, 3000, hi, 3000]),
                              maxfev=10000)

    if len(E_ul) == 0 or np.all(model_2T(E_ul, *popt0) <= y_ul):
        errs = np.sqrt(np.diag(pcov0))
        print("  2T: no upper limit is actively binding - identical to the unconstrained fit")
        return {"popt": popt0, "errs": errs, "n_active_constraints": 0, "used_bootstrap": False}

    bounds = [(lo, hi), (10, 3000), (lo, hi), (10, 3000)]
    constraints = [{"type": "ineq", "fun": _ul_ceiling_2T, "args": (E_ul, y_ul)}]

    res = minimize(_chi2_2T, x0=popt0, args=(E_d, y_d, yerr_d),
                    method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"Constrained 2T fit failed: {res.message}")
    popt = res.x
    n_active = int(np.sum(model_2T(E_ul, *popt) >= y_ul - 1e-8))

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        y_d_boot = rng.normal(y_d, yerr_d)
        res_b = minimize(_chi2_2T, x0=popt, args=(E_d, y_d_boot, yerr_d),
                          method="SLSQP", bounds=bounds, constraints=constraints)
        if res_b.success:
            boot.append(res_b.x)

    errs = np.std(np.array(boot), axis=0) if boot else np.full(4, np.nan)

    print(f"  2T: {n_active} upper limit(s) actively constraining the fit")
    print(f"  Bootstrap uncertainty: {len(boot)}/{n_boot} resamples converged")

    return {
        "popt": popt,
        "errs": errs,
        "n_active_constraints": n_active,
        "used_bootstrap": True,
    }
























'''
ROTATIONAL DIAGRAM
'''

## Partition Function
## Popovas & Jorgensen (2016): https://ui.adsabs.harvard.edu/abs/2016A%26A...595A.130P/abstract
def partitionH2(T, OPR='equilibrium'):
    T = np.maximum(T,10)
    # Adjusted from ranges given in paper (which overlap) to minimise jump
    loT = (T>=10)*(T<=237.5)
    miT = (T>=237.5)*(T<=1100)
    hiT = (T>=1100)*(T<=20000)
    extrapolate = (T>=20000)
    if OPR=='normal':
        loQ =  2.277668085144430     +  3.115475456535298e-04*T + -1.199204285095701e-05*T**2 +  1.348313176588763e-07*T**3 + -2.493525433376579e-10*T**4
        miQ =  1.919800108599199     +  1.746213811668277e-03*T +  7.693197369847348e-06*T**2 + -6.504303232753276e-09*T**3 +  2.132168610356154e-12*T**4
        hiQ = -1.142827289681385e-01 +  7.245357485089481e-03*T + -6.404547049175155e-07*T**2 +  3.110974834335349e-10*T**3 + -1.642844436813594e-14*T**4 + 2.790315674249059e-19*T**5
    elif OPR=='equilibrium':
        loQ =  2.673071615415136e-01 + -3.495586051757211e-03*T +  1.227901619954258e-04*T**2 + -5.776440695273789e-07*T**3 +  9.251224490175610e-10*T**4
        miQ =  1.410033600294133e-01 +  6.085738724141971e-03*T + -4.096994909866605e-07*T**2 +  4.220221708082499e-10*T**3 + -8.790594164685680e-14*T**4
        hiQ = -9.661842638994980e-01 +  7.302127874247883e-03*T + -6.760893004505151e-07*T**2 +  3.128741080316710e-10*T**3 + -1.645206030945910e-14*T**4 + 2.788597060472472e-19*T**5
    elif OPR=='ortho':
        loQ =  2.996425936476927     +  3.807856081411796e-04*T + -8.659737634951350e-06*T**2 +  5.779158662343921e-08*T**3 + -5.407749218414769e-11*T**4
        miQ =  2.891845114721733     + -1.243647561012797e-03*T +  1.312997140340565e-05*T**2 + -1.105110818469195e-08*T**3 +  3.562652399419052e-12*T**4
        hiQ =  2.231696541394133e-01 +  7.190413006059544e-03*T + -6.192076726407582e-07*T**2 +  3.096074912455146e-10*T**3 + -1.640036910200170e-14*T**4 + 2.790310846593324e-19*T**5
    elif OPR=='para':
        loQ =  9.994339093261912e-01 +  3.135419941617079e-04*T + -1.599408560500201e-05*T**2 +  2.032015813340736e-07*T**3 + -4.090066248229891e-10*T**4
        miQ =  3.312881928317837e-01 +  4.871438707876421e-03*T +  2.339907793941012e-06*T**2 + -2.214128676054798e-09*T**3 +  8.210809799374644e-13*T**4
        hiQ = -1.019053774741465e+00 +  7.360074236377774e-03*T + -6.935790834485644e-07*T**2 +  3.145274660505019e-10*T**3 + -1.646581413255403e-14*T**4 + 2.782419680521890e-19*T**5
    else:
        raise NotImplementedError("Option not recognised")
    return loQ*loT+miQ*miT+hiQ*hiT+hiQ*extrapolate, loQ, miQ, hiQ


def Q_total(T, OPR=3.0):
    Q_ortho, _, _, _ = partitionH2(T, OPR='ortho')
    Q_para, _, _, _ = partitionH2(T, OPR='para')

    f_ortho = OPR #/ (1.0 + OPR)
    f_para = 1.0 #/ (1.0 + OPR)

    return f_para * Q_para + f_ortho * Q_ortho


def Q_total_err(T, dT, OPR, dOPR=0.0):
    """
    Uncertainty on Q_total propagated from dT and dOPR.
    Uses finite differences.
    """
    eps_T = max(0.1, abs(T) * 1e-5)
    dQ_dT = (Q_total(T + eps_T, OPR) - Q_total(T - eps_T, OPR)) / (2 * eps_T)

    if dOPR > 0:
        eps_OPR = max(1e-5, abs(OPR) * 1e-5)
        dQ_dOPR = (Q_total(T, OPR + eps_OPR) - Q_total(T, OPR - eps_OPR)) / (2 * eps_OPR)
    else:
        dQ_dOPR = 0.0

    return np.sqrt((dQ_dT * dT)**2 + (dQ_dOPR * dOPR)**2)






def rotational_diagram(fluxes, flux_errors, molecular_data=None, OPR=None, fit_opr=False, opr_grid=None,
                        fit2T=False, R=None, dist=140*pc, upper_limits=None):
    """
    Perform rotational diagram analysis on H2 lines.

    Args:
        fluxes: Dictionary {J: flux_value} where J is the upper rotational level
        flux_errors: Dictionary {J: flux_error}
        molecular_data: Dictionary with 'wavelengths', 'Aul', 'Eu', 'gu' keys
                       Each value is a dict {J: value}
        OPR: Fixed ortho-para ratio. Used as-is when fit_opr=False (None defaults to
             3.0, see apply_OPR). Ignored (overwritten by the search result) when
             fit_opr=True.
        fit_opr: Whether to grid-search over OPR to minimize chi-squared. The search
                 always uses plain curve_fit on detections only (upper-limit points are
                 excluded from the search itself, since curve_fit has no notion of a
                 one-sided constraint) - the final fit below still uses the full data,
                 including upper limits via the constrained least-squares path when present.
        opr_grid: OPR values to search over. Defaults to np.linspace(0.1, 5.0, 200).
        fit2T: Whether to perform 2-temperature fit
        R: Emitting radius (AU or cm).
        dist: Distance (pc or cm).
        upper_limits: Optional dict {J: True} for measurements marked as upper limits.
                      If present, a deterministic constrained least-squares fit
                      (upper limits as one-sided ceilings, see constrained_1T/2T)
                      is automatically used for the final 1T/2T fit.

    Returns:
        Dictionary with fitting results

    Notes:
        - If no upper limits: uses fast curve_fit (scipy) for the final fit
        - If upper limits present: automatically uses constrained least-squares
          (upper limits as ceiling constraints) for the final fit - see constrained_1T/2T
        - fit_opr's grid search always uses curve_fit on detections only, regardless of upper_limits
    """

    global wave, Aul, Eu, gu

    if molecular_data is not None:
        wave = molecular_data.get('wavelengths', {})
        Aul = molecular_data.get('Aul', {})
        Eu = molecular_data.get('Eu', {})
        gu = molecular_data.get('gu', {})

    # ==========================================================
    # Build dataset for a given trial OPR
    # ==========================================================
    def _build_dataset(opr_value):
        """Compute excitation diagram arrays for a given OPR."""
        E_list, y_list, y_err_list, is_ul_list = [], [], [], []
        dropped = []  # J values excluded (detections with no error data)

        for J in sorted(fluxes.keys(), key=lambda j: wave.get(j, float('inf'))):
            if J not in wave:
                continue

            F = fluxes[J]
            Nu_no_opr = compute_Nu(F, wave[J], Aul[J], R, dist)
            Nu = apply_OPR(Nu_no_opr, J, opr_value)
            y = np.log(Nu / gu[J])

            is_ul = upper_limits is not None and J in upper_limits

            # Skip detections without error data - no defaults allowed
            if (not is_ul) and (J not in flux_errors):
                print(f"J={J}: skipping (detection without error data)")
                dropped.append(J)
                continue

            E_list.append(Eu[J])
            y_list.append(y)
            is_ul_list.append(is_ul)

            # error handling
            if (not is_ul) and (J in flux_errors):
                dF = flux_errors[J]
                dNu = Nu * (dF / F)
                y_err = dNu / Nu
                y_err_list.append(y_err)
            else:
                # upper limits: treated as constraints with 0 error
                y_err_list.append(0.0)

        return (np.array(E_list),
                np.array(y_list),
                np.array(y_err_list),
                np.array(is_ul_list, dtype=bool),
                dropped)

    # ==========================================================
    # Bounded curve_fit helpers (used for both the OPR grid search and the
    # final fit whenever the constrained least-squares path isn't required)
    # ==========================================================
    def _bounds_1T(y_arr):
        """Return bounds for the single-temperature model."""
        return ([np.min(y_arr), 10],
                [np.max(y_arr) + 10, 3000])

    def _bounds_2T(y_arr):
        """Return bounds for the two-temperature model."""
        lo = np.min(y_arr)
        hi = np.max(y_arr) + 10
        return ([lo, 10, lo, 10],
                [hi, 3000, hi, 3000])

    def _fit_1T_curvefit(E, y, y_err):
        """Perform a single-temperature fit and return (popt, pcov)."""
        sigma = y_err if np.any(y_err > 0) else None  # prevent all zeros array
        p0 = [np.max(y), 500]
        bounds = _bounds_1T(y)

        popt, pcov = curve_fit(
            model_1T,
            E,
            y,
            sigma=sigma,
            absolute_sigma=True,
            p0=p0,
            bounds=bounds
        )
        return popt, pcov

    def _fit_2T_curvefit(E, y, y_err, a_seed, T_seed):
        """Perform a two-temperature fit seeded from a 1T solution."""
        sigma = y_err if np.any(y_err > 0) else None
        p0 = [a_seed, T_seed, a_seed - 5, T_seed * 2]
        bounds = _bounds_2T(y)
        
        popt, pcov = curve_fit(
            model_2T,
            E,
            y,
            sigma=sigma,
            absolute_sigma=True,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        return popt, pcov

    def _compute_chi2(E_arr, y_arr, y_err, is_ul_mask, use_2T):
        """Return chi-squared for a quick fit (1T or 2T), fit on detections only.

        Upper-limit points are excluded here (curve_fit has no censored/one-sided
        likelihood) - they're only reintroduced in the final fit below, via the
        constrained least-squares path.
        """
        keep = ~is_ul_mask  # only consider non-upper limits here
        E_d, y_d, y_err_d = E_arr[keep], y_arr[keep], y_err[keep]

        n_params = 4 if use_2T else 2
        if len(E_d) <= n_params:
            # Not enough detections to constrain the model at this grid point -
            # avoid a degenerate zero-residual "perfect fit" from winning the search.
            return np.inf

        try:
            popt_1t, _ = _fit_1T_curvefit(E_d, y_d, y_err_d)
            if use_2T:
                popt, _ = _fit_2T_curvefit(E_d, y_d, y_err_d, popt_1t[0], popt_1t[1])
                model_vals = model_2T(E_d, *popt)
            else:
                model_vals = model_1T(E_d, *popt_1t)
        except Exception:
            return np.inf

        residuals = (y_d - model_vals) / np.maximum(y_err_d, 1e-10)
        return np.sum(residuals**2)

    # ==========================================================
    # OPR grid search (fast, detections-only, curve_fit)
    # ==========================================================
    best_opr = OPR if OPR is not None else 3.0
    best_opr_err = None
    best_opr_err_pos = None
    best_opr_err_neg = None
    opr_search_data = None

    if fit_opr:
        if opr_grid is None:
            opr_grid = np.linspace(0.1, 5.0, 200)

        chi2_list = []
        for opr_try in opr_grid:
            # consider upper limit as 4th param
            E_try, y_try, y_err_try, ul_try, _ = _build_dataset(opr_try)
            chi2_list.append(_compute_chi2(E_try, y_try, y_err_try, ul_try, use_2T=fit2T))
        
        chi2_arr = np.array(chi2_list)
        
        best_idx = np.argmin(chi2_arr)
        best_opr = opr_grid[best_idx]
        within_1sigma = np.where(chi2_arr < chi2_arr[best_idx] + 1)[0]
        err_opr_pos = opr_grid[within_1sigma[-1]] - best_opr
        err_opr_neg = best_opr - opr_grid[within_1sigma[0]]
        best_opr_err = 0.5 * (err_opr_pos + err_opr_neg)
        best_opr_err_pos = err_opr_pos
        best_opr_err_neg = err_opr_neg

        print("\n--- OPR FIT (curve_fit grid search, detections only) ---")
        print(f"Best-fit OPR = {best_opr:.3f}")
        print(f"OPR 1σ error: -{err_opr_neg:.4f} / +{err_opr_pos:.4f} (symm ≈ {best_opr_err:.4f})")

        opr_search_data = {"opr_grid": opr_grid, "chi2_list": chi2_arr}

    # ==========================================================
    # Build final dataset at best_opr (full data, including upper limits)
    # ==========================================================
    E_arr, y_arr, y_err, is_ul_mask, dropped_detections = _build_dataset(best_opr)





    if len(E_arr) < 2:
        raise ValueError("Need at least 2 points")

    # Check if we have enough detections to constrain the fit (at least 2 required)
    n_detections = np.sum(~is_ul_mask)
    has_upper_limits = np.any(is_ul_mask)

    if has_upper_limits and n_detections < 2:
        if n_detections == 0:
            raise ValueError(f"Cannot fit: all {len(E_arr)} measurements are upper limits. Need at least 2 detections to constrain temperature and column density.")
        else:
            raise ValueError(f"Need at least 2 detections for fitting with upper limits. You have {n_detections} detection(s) and {np.sum(is_ul_mask)} upper limit(s).")

    # Decide method: auto-use the constrained least-squares path if upper limits present
    use_constrained = has_upper_limits

    # ==========================================================
    # 1T FIT (final)
    # ==========================================================
    if not use_constrained:

        print("\n--- 1T FIT (curve_fit) ---")

        popt1, pcov1 = _fit_1T_curvefit(E_arr, y_arr, y_err)

        a1, T1 = popt1
        da1, dT1 = np.sqrt(np.diag(pcov1))

    else:

        print("\n--- 1T FIT (CONSTRAINED LEAST-SQUARES) ---")

        result = constrained_1T(
            E_arr, y_arr, y_err, is_ul_mask
        )

        a1, T1 = result["a"], result["T"]
        da1, dT1 = result["da"], result["dT"]
        constraint_diag_1T = {"n_active_constraints": result["n_active_constraints"],
                               "used_bootstrap": result["used_bootstrap"]}

    # =========================
    # Derived quantities
    # =========================
    Ncol1 = np.exp(a1)
    dNcol1 = Ncol1 * da1

    Ntot1 = Ncol1 * Q_Herbst(T1)
    dNtot1 = Ntot1 * np.sqrt((dNcol1/Ncol1)**2 + (Q_Herbst_err(T1, dT1)/Q_Herbst(T1))**2)

    # chi2 uses detections only - upper limits have no meaningful sigma to weight by
    # (y_err=0 is just a placeholder for them, not a real 1e-10 uncertainty)
    det_mask = ~is_ul_mask
    y_model1 = model_1T(E_arr[det_mask], a1, T1)
    chi2_1T = np.sum(((y_arr[det_mask] - y_model1) / np.maximum(y_err[det_mask], 1e-10))**2)
    dof_1T = max(np.sum(det_mask) - 2, 1)

    results = {
        "OPR": best_opr,
        "1T": {
            "T": T1,
            "dT": dT1,
            "Ncol": Ntot1,
            "dNcol": dNtot1,
            "chi2": chi2_1T,
            "red_chi2": chi2_1T / dof_1T,
        }
    }
    if use_constrained:
        results["1T"]["n_active_constraints"] = constraint_diag_1T["n_active_constraints"]
        results["1T"]["used_bootstrap"] = constraint_diag_1T["used_bootstrap"]
    if best_opr_err is not None:
        results["OPR_err"] = best_opr_err
        results["OPR_err_pos"] = best_opr_err_pos
        results["OPR_err_neg"] = best_opr_err_neg

    if dropped_detections:
        results["dropped_J"] = dropped_detections

    # ==========================================================
    # 2T FIT (final)
    # ==========================================================
    if fit2T:

        if not use_constrained:

            print("\n--- 2T FIT (curve_fit) ---")

            popt2, pcov2 = _fit_2T_curvefit(E_arr, y_arr, y_err, a1, T1)

            a1_2, T1_2, a2_2, T2_2 = popt2
            errs = np.sqrt(np.diag(pcov2))

        else:

            print("\n--- 2T FIT (CONSTRAINED LEAST-SQUARES) ---")

            result = constrained_2T(
                E_arr, y_arr, y_err, is_ul_mask,
                a1, T1
            )

            a1_2, T1_2, a2_2, T2_2 = result["popt"]
            errs = result["errs"]
            constraint_diag_2T = {"n_active_constraints": result["n_active_constraints"],
                                   "used_bootstrap": result["used_bootstrap"]}

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

        # chi2 uses detections only, same reasoning as the 1T fit above
        y_model2 = model_2T(E_arr[det_mask], a1_2, T1_2, a2_2, T2_2)
        chi2_2T = np.sum(((y_arr[det_mask] - y_model2) / np.maximum(y_err[det_mask], 1e-10))**2)
        dof_2T = max(np.sum(det_mask) - 4, 1)

        results["2T"] = {
            "T1": T1_2,
            "dT1": dT1_2,
            "T2": T2_2,
            "dT2": dT2_2,
            "Ncol1": N1_tot,
            "dNcol1": dN1_tot,
            "Ncol2": N2_tot,
            "dNcol2": dN2_tot,
            "chi2": chi2_2T,
            "red_chi2": chi2_2T / dof_2T,
        }
        if use_constrained:
            results["2T"]["n_active_constraints"] = constraint_diag_2T["n_active_constraints"]
            results["2T"]["used_bootstrap"] = constraint_diag_2T["used_bootstrap"]

        results["_plot_data"] = {
            "E_arr": E_arr,
            "y_arr": y_arr,
            "y_err": y_err,
            "is_ul_mask": is_ul_mask,
            "popt1": list(popt1) if not use_constrained else [a1, T1],
            "popt2": list(popt2) if not use_constrained else list(result["popt"]),
        }
    else:
        results["_plot_data"] = {
            "E_arr": E_arr,
            "y_arr": y_arr,
            "y_err": y_err,
            "is_ul_mask": is_ul_mask,
            "popt1": list(popt1) if not use_constrained else [a1, T1],
            "popt2": None,
        }

    if opr_search_data is not None:
        results["_plot_data"]["opr_grid"] = opr_search_data["opr_grid"]
        results["_plot_data"]["chi2_list"] = opr_search_data["chi2_list"]

    return results