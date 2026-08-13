"""
Rotational diagram fitting functions for H2 analysis
Implements single-temperature and two-temperature rotational diagram fits
"""

import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
import pymc as pm

from .unit_converter import integrated_flux_to_cgs



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


def convert_flux_upper_limit_to_cgs(flux_ul_native, wavelength, header, flux_unit=None):
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
        flux_unit: Native flux unit. Falls back to the header's BUNIT if None, so MUSE
                   cubes convert too rather than being assumed to be MJy/sr.

    Returns:
        float: Upper limit flux in erg/s/cm², or None if conversion fails

    The conversion itself lives in ``unit_converter.integrated_flux_to_cgs`` - the same
    one used for Gaussian areas, so upper limits and detections cannot drift apart.
    """
    try:
        if flux_unit is None and header is not None:
            flux_unit = header.get('BUNIT', None)

        return integrated_flux_to_cgs(
            flux_ul_native,
            flux_unit,
            wavelength=wavelength,
            header=header,
            wave_unit='um',
        )
    except Exception:
        return None


def _collective_amplitude_error(fit_result):
    """
    Closed-form (no Monte Carlo, no re-fitting) standard error on a fixed-center/
    fixed-sigma Gaussian amplitude, from the residuals fit_hitran_lines_collective
    already returned for this line.

    Why not Monte Carlo noise injection (re-fit with a nonlinear, free-center/
    free-sigma model many times and take the std dev of the results): that model
    is different from the one that actually produced this amplitude (fixed
    center/sigma, linear lstsq), so it would estimate the error of a different fit
    than the one being reported, and does so far more slowly. For a linear model
    the parameter covariance has a direct closed form -- no re-fitting needed.

    Amplitude was NOT estimated alone: fit_hitran_lines_collective's design matrix
    always appends a constant column ("a constant term for any small residual left
    after the global continuum subtraction"), even for a single requested line
    with no contaminants -- global continuum subtraction is one smooth curve fit
    across the whole spectrum, so it isn't guaranteed to land at exactly zero in
    the specific pixels under any one line. Without that column, a real leftover
    offset there has nowhere to go except into the amplitude itself, biasing it
    (verified numerically: an amplitude-only fit against data with a genuine small
    offset came out measurably too high, while the 2-parameter fit recovered the
    true amplitude and identified the offset separately). So the coefficients
    that actually came out of the real lstsq solve are [amplitude, local
    baseline], jointly and correlated -- estimating amplitude's error as if it
    were the only unknown ignores that correlation and understates it. This
    reconstructs the same 2-parameter design matrix and gets the standard OLS
    parameter covariance:

        Cov(beta) = noise_var * (X^T X)^-1,   X = [template, 1]

    amplitude's variance is the (0,0) entry. noise_var = sum(residuals^2)/dof,
    dof = n - 2 (two parameters: amplitude and the local constant).

    noise_var deliberately comes from this line's own *local* residuals rather
    than np.linalg.lstsq's own (currently unused) whole-fit SSE: that would be one
    noise number averaged over the entire joint fit -- every line, every
    contaminant, the whole padded window -- diluting exactly the kind of localized
    excess noise (blending, a poorly-tracked continuum near just this line) that
    should inflate THIS line's error specifically, not get smoothed away by
    cleaner data elsewhere in the spectrum.

    This is still a local approximation of the *full* joint covariance -- it
    doesn't account for correlation with *other* lines'/contaminants' templates
    when more than one shares the design matrix. fit_hitran_lines_collective
    already drops a contaminant's template entirely when it's near-degenerate with
    a requested line's (<2 sigma apart), which is exactly the case where that
    correlation would matter most, so what's left out here should be small.

    Returns the amplitude's 1-sigma uncertainty (native flux units), or None if
    there isn't enough local data to estimate it.
    """
    from .co_flux_extractor import gaussian

    x = fit_result.x_data
    if len(x) < 3:
        return None

    template = gaussian(x, 1.0, fit_result.center, fit_result.sigma)
    design = np.column_stack([template, np.ones_like(template)])

    dof = len(x) - design.shape[1]
    if dof < 1:
        return None
    noise_var = np.sum(fit_result.residuals ** 2) / dof

    try:
        cov = noise_var * np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return None

    return float(np.sqrt(cov[0, 0]))


def compute_line_fluxes_collective(wavelength_arr, spectrum_arr, line_wavelengths, header, instrument,
                                    contam_lines=None):
    """
    Fit every line in `line_wavelengths` simultaneously in one (already spatially-
    summed) spectrum, via fit_hitran_lines_collective's linear least-squares solve:
    fixed center + fixed instrument-resolution sigma, one design-matrix solve for
    every line's amplitude at once (as opposed to an independent per-line nonlinear
    curve_fit). For rotational-diagram lines this is usually the better fit: they're
    often weak/high-J and can sit close to other species' features, which an
    independent per-line fit has no way to keep from leaking in.

    Args:
        wavelength_arr, spectrum_arr: 1D arrays for the (spatially-summed) spectrum
        line_wavelengths: list of target line center wavelengths (microns)
        header: FITS header (for spectral resolution + unit conversion)
        instrument: Instrument name ('MIRI', 'NIRSPEC', ...)
        contam_lines: optional pd.DataFrame (same 'line_wave_um' format
            load_contaminating_lines_catalog() returns) -- non-target lines that get
            their own template column so they stop leaking into a requested line's
            amplitude, passed straight through to fit_hitran_lines_collective. None
            (default) skips this, same as if no catalog were loaded.

    Returns:
        list of dicts, same length/order as line_wavelengths. Each dict has keys
        success (bool), message (str or None), and on success also flux_cgs,
        flux_error_cgs, is_upper_limit, F_ul_cgs, sigma_noise, delta_lambda.
    """
    from .gaussian_fitter import get_wave_unit, convert_area_to_cgs
    from .co_flux_extractor import fit_auto_continuum, fit_hitran_lines_collective

    try:
        continuum = fit_auto_continuum(wavelength_arr, spectrum_arr)
    except Exception as e:
        return [{"success": False, "message": f"Continuum fit failed: {e}"} for _ in line_wavelengths]

    hitran_lines = [{'wavelength': wl} for wl in line_wavelengths]
    fit_pairs, _, velocity_shift_kms = fit_hitran_lines_collective(
        wavelength_arr, spectrum_arr, hitran_lines, continuum,
        header=header, wave_unit='um', contam_lines=contam_lines,
    )
    if velocity_shift_kms is not None:
        print(f"[BATCH_FLUX] Collective fit shared velocity shift: {velocity_shift_kms:.2f} km/s")

    wave_unit = get_wave_unit(header, 'um')

    results = []
    for line, fit_result in fit_pairs:
        wl = line['wavelength']

        # amp <= 0 ("No significant signal") is a real, expected outcome for a weak
        # or non-detected line, not a fit failure to abort on -- area is 0 in that
        # case, which flows straight into the upper-limit check below. Only bail out
        # here if there wasn't even enough spectral data to attempt a fit.
        if not fit_result.success and "Not enough spectral points" in (fit_result.message or ""):
            results.append({"success": False, "message": fit_result.message})
            continue

        # Already CGS (erg/s/cm^2): fit_hitran_lines_collective calls
        # convert_area_to_cgs internally when a header is given.
        flux_integrated = fit_result.area

        # Closed-form error consistent with the fixed-center/sigma model that
        # produced flux_integrated -- see _collective_amplitude_error's docstring
        # for why this replaces a Monte Carlo re-fit with a free-parameter model.
        amp_error = _collective_amplitude_error(fit_result)
        if amp_error is None:
            results.append({"success": False, "message": "Could not estimate flux error"})
            continue
        flux_error = convert_area_to_cgs(amp_error, fit_result.sigma, wl, header, wave_unit)
        if flux_error is None:
            results.append({"success": False, "message": "Could not convert flux error to CGS"})
            continue

        sigma_noise = estimate_noise_local(wavelength_arr, spectrum_arr, wl, n_pixels=13)
        if sigma_noise is None:
            sigma_noise = np.nanstd(spectrum_arr[len(spectrum_arr)//2:])

        F_ul_native, delta_lambda_ul = gaussian_flux_upper_limit(sigma_noise, wl, instrument=instrument, header=header)
        F_ul_cgs = convert_flux_upper_limit_to_cgs(F_ul_native, wl, header, flux_unit='MJy/sr')

        is_upper_limit = flux_integrated < F_ul_cgs

        print(f"[BATCH_FLUX] Line @ {wl:.6f} um (collective):")
        print(f"  flux = {flux_integrated:.4e} erg/s/cm^2")
        print(f"  error = {flux_error:.4e} erg/s/cm^2")
        print(f"  noise (native) = {sigma_noise:.4e} MJy/sr")
        print(f"  delta_lambda = {delta_lambda_ul:.4e} um")
        print(f"  F_ul (CGS) = {F_ul_cgs:.4e} erg/s/cm^2")
        print(f"  is_upper_limit = {is_upper_limit}")

        results.append({
            "success": True,
            "message": fit_result.message,
            "flux_cgs": flux_integrated,
            "flux_error_cgs": flux_error,
            "is_upper_limit": is_upper_limit,
            "F_ul_cgs": F_ul_cgs,
            "sigma_noise": sigma_noise,
            "delta_lambda": delta_lambda_ul,
        })

    return results


# ==========================================================
# CENSORED MODELS (ONLY USED IF ENABLED)
# ==========================================================
def _print_pymc_diagnostics(trace, var_names):
    """Print divergence count and r_hat for a PyMC trace; return them so
    callers can also check convergence programmatically instead of just
    trusting the posterior mean/std.
    """
    import arviz as az

    n_divergent = int(trace.sample_stats.diverging.sum())
    summary = az.summary(trace, var_names=var_names)
    max_rhat = float(summary["r_hat"].max())

    print(f"  Divergences: {n_divergent}")
    print(summary[["mean", "sd", "r_hat", "ess_bulk"]].to_string())
    if n_divergent > 0:
        print(f"  WARNING: {n_divergent} divergent transition(s) - posterior may be biased/unreliable")
    if max_rhat > 1.01:
        print(f"  WARNING: max r_hat = {max_rhat:.3f} (>1.01) - chains have not converged")

    return {"n_divergences": n_divergent, "max_r_hat": max_rhat}


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
        # 2 chains (sequential, cores=1) so r_hat/divergences are actually
        # checkable instead of trusting a single chain; target_accept raised
        # from PyMC's default 0.8 to reduce divergences in this a-vs-T ridge.
        trace = pm.sample(500, tune=500, chains=2, cores=1, target_accept=0.9, progressbar=False)

    diagnostics = _print_pymc_diagnostics(trace, ["a", "T"])

    # === RETURN POSTERIOR ESTIMATES ===
    return {
        "a": float(trace.posterior["a"].mean()),        # log(Ncol)
        "da": float(trace.posterior["a"].std()),        # uncertainty
        "T": float(trace.posterior["T"].mean()),        # temperature
        "dT": float(trace.posterior["T"].std()),        # uncertainty
        "n_divergences": diagnostics["n_divergences"],
        "max_r_hat": diagnostics["max_r_hat"],
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
        # 2 chains (sequential, cores=1) so r_hat/divergences are actually
        # checkable; target_accept raised to reduce divergences in this
        # 4-parameter censored mixture (more prone to them than 1T).
        trace = pm.sample(500, tune=500, chains=2, cores=1, target_accept=0.9, progressbar=False)

    diagnostics = _print_pymc_diagnostics(trace, ["a1", "T1", "a2", "T2"])

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
        ]),
        "n_divergences": diagnostics["n_divergences"],
        "max_r_hat": diagnostics["max_r_hat"],
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
                 including upper limits via the censored PyMC path when present.
        opr_grid: OPR values to search over. Defaults to np.linspace(0.1, 5.0, 200).
        fit2T: Whether to perform 2-temperature fit
        R: Emitting radius (AU or cm).
        dist: Distance (pc or cm).
        upper_limits: Optional dict {J: True} for measurements marked as upper limits.
                      If present, PyMC censored regression is automatically used for
                      the final 1T/2T fit.

    Returns:
        Dictionary with fitting results

    Notes:
        - If no upper limits: uses fast curve_fit (scipy) for the final fit
        - If upper limits present: automatically uses PyMC censored likelihood for the final fit
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
    # final fit whenever PyMC/censored regression isn't required)
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
        censored PyMC path.
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
        print(f"OPR 1-sigma error: -{err_opr_neg:.4f} / +{err_opr_pos:.4f} (symm ~= {best_opr_err:.4f})")

        opr_search_data = {"opr_grid": opr_grid, "chi2_list": chi2_arr}

    # ==========================================================
    # Build final dataset at best_opr (full data, including upper limits)
    # ==========================================================
    E_arr, y_arr, y_err, is_ul_mask, dropped_detections = _build_dataset(best_opr)





    # censor bayesian

    if len(E_arr) < 2:
        raise ValueError("Need at least 2 points")

    # Check if we have enough detections for PyMC (at least 2 detections required)
    n_detections = np.sum(~is_ul_mask)
    has_upper_limits = np.any(is_ul_mask)

    if has_upper_limits and n_detections < 2:
        if n_detections == 0:
            raise ValueError(f"Cannot fit: all {len(E_arr)} measurements are upper limits. Need at least 2 detections to constrain temperature and column density.")
        else:
            raise ValueError(f"Need at least 2 detections for fitting with upper limits. You have {n_detections} detection(s) and {np.sum(is_ul_mask)} upper limit(s).")

    # Decide method: auto-use PyMC if upper limits present
    use_pymc = has_upper_limits

    # ==========================================================
    # 1T FIT (final)
    # ==========================================================
    if not use_pymc:

        print("\n--- 1T FIT (curve_fit) ---")

        popt1, pcov1 = _fit_1T_curvefit(E_arr, y_arr, y_err)

        a1, T1 = popt1
        da1, dT1 = np.sqrt(np.diag(pcov1))

    else:

        print("\n--- 1T FIT (CENSORED PyMC) ---")

        result = censored_1T(
            E_arr, y_arr, y_err, is_ul_mask
        )

        a1, T1 = result["a"], result["T"]
        da1, dT1 = result["da"], result["dT"]
        pymc_diag_1T = {"n_divergences": result["n_divergences"], "max_r_hat": result["max_r_hat"]}

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
    if use_pymc:
        results["1T"]["n_divergences"] = pymc_diag_1T["n_divergences"]
        results["1T"]["max_r_hat"] = pymc_diag_1T["max_r_hat"]
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

        if not use_pymc:

            print("\n--- 2T FIT (curve_fit) ---")

            popt2, pcov2 = _fit_2T_curvefit(E_arr, y_arr, y_err, a1, T1)

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
            pymc_diag_2T = {"n_divergences": result["n_divergences"], "max_r_hat": result["max_r_hat"]}

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
        if use_pymc:
            results["2T"]["n_divergences"] = pymc_diag_2T["n_divergences"]
            results["2T"]["max_r_hat"] = pymc_diag_2T["max_r_hat"]

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

    if opr_search_data is not None:
        results["_plot_data"]["opr_grid"] = opr_search_data["opr_grid"]
        results["_plot_data"]["chi2_list"] = opr_search_data["chi2_list"]

    return results