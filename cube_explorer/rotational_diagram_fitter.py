"""
Rotational diagram fitting functions for H2 analysis
Implements single-temperature and two-temperature rotational diagram fits
"""

import numpy as np
from scipy.optimize import curve_fit
import pymc as pm
import nutpie
import arviz as az

from .unit_converter import integrated_flux_to_cgs
from .co_flux_extractor import fit_auto_continuum, fit_hitran_lines_collective

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

_PYMC_RANDOM_SEED = 42  # PyMC random seed for deterministic approach

# Systematic flux-calibration error floor (Pascucci): the statistical fit error alone
# understates the true flux uncertainty, since it says nothing about instrument
# flux-calibration uncertainty or continuum-estimation systematics -- both real, both
# roughly proportional to the flux itself rather than to how well-constrained the local
# fit happened to be. Added in quadrature: sqrt(statistical^2 + (SYSTEMATIC_FLUX_ERROR_FRACTION*flux)^2).
SYSTEMATIC_FLUX_ERROR_FRACTION = 0.05

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

def gaussian_flux_upper_limit(sigma_noise, sigma):
    """
    3-sigma upper-limit flux: the integrated area of an artificial Gaussian with
    amplitude = 3*sigma_noise and the given sigma -- i.e. exactly the Gaussian a marginal
    3-sigma detection would produce, using the same area formula
    (amplitude * sigma * sqrt(2*pi)) as every other line fit in this app. Not the old
    delta_lambda (= lambda/R, a FWHM-like width) formula, which used a different width
    convention than the area formula everything else here is built on.

    Args:
        sigma_noise: Noise standard deviation in MJy/sr (per-pixel), from the actual
            simultaneous chunk fit's own residuals (fit_glines' noise_std_chunk).
        sigma: The instrumental (spectral-resolution) sigma the line was actually fit
            with -- pass fit_result.catalog_sigma, not a value re-derived here, so this
            always matches the width fit_glines' own bounds used for this exact line.

    Returns:
        F_ul: upper limit flux [MJy/sr*um]
    """
    return 3.0 * sigma_noise * sigma * np.sqrt(2 * np.pi)


def compute_line_fluxes_collective(wavelength_arr, spectrum_arr, line_wavelengths, header,
                                    contam_lines=None):
    """
    Fit every line in `line_wavelengths` simultaneously in one spectrum (fit_hitran_lines_collective).
    For rotational-diagram lines this is usually the better fit.

    Args:
        wavelength_arr, spectrum_arr: 1D arrays for the (spatially-summed) spectrum
        line_wavelengths: list of target line center wavelengths (microns)
        header: FITS header (instrument is read from it -- see _line_sigma)
        contam_lines: optional pd.DataFrame
            None (default) skips this, same as if no catalog were loaded.

    Returns:
        list of dicts, same length/order as line_wavelengths. Each dict has keys
        success (bool), message (str or None), and on success also flux_cgs,
        flux_error_cgs (None when is_upper_limit), is_upper_limit, F_ul_cgs,
        sigma_noise, catalog_sigma, r_squared, and plot data (x_data, y_data,
        baseline, y_fit, center, fwhm, amplitude) for a per-line fit display.
    """
    try:
        continuum = fit_auto_continuum(wavelength_arr, spectrum_arr)
    except Exception as e:
        return [{"success": False, "message": f"Continuum fit failed: {e}"} for _ in line_wavelengths]

    hitran_lines = [{'wavelength': wl} for wl in line_wavelengths]
    try:
        fit_pairs, _, velocity_shift_kms = fit_hitran_lines_collective(
            wavelength_arr, spectrum_arr, hitran_lines, continuum,
            header=header, wave_unit='um', contam_lines=contam_lines,
        )
    except Exception as e:
        # e.g. _line_sigma raising because the header can't produce a real spectral
        # resolution -- surfaced per this function's own contract (a visible message per
        # line), not a silently-guessed sigma and not an uncaught crash either.
        return [{"success": False, "message": f"Collective fit failed: {e}"} for _ in line_wavelengths]
    if velocity_shift_kms is not None:
        print(f"[BATCH_FLUX] Collective fit shared velocity shift: {velocity_shift_kms:.2f} km/s")

    results = []
    for line, fit_result in fit_pairs:
        wl = line['wavelength']

        if not fit_result.success:
            results.append({"success": False, "message": fit_result.message})
            continue

        # noise_std/catalog_sigma come straight from fit_glines' own simultaneous fit (chunk
        # residual noise, and the instrumental sigma this line was actually fit with) -- both
        # None only if that chunk had too few data points to estimate noise from residuals at
        # all, which means we genuinely can't judge detection vs. upper limit here.
        if fit_result.noise_std is None or fit_result.catalog_sigma is None:
            results.append({"success": False, "message": "No noise estimate available for this line's chunk"})
            continue

        # Already CGS (erg/s/cm^2) -- the real fitted flux, from curve_fit's own amplitude/sigma
        flux_integrated = fit_result.area

        F_ul_native = gaussian_flux_upper_limit(fit_result.noise_std, fit_result.catalog_sigma)
        try:
            F_ul_cgs = integrated_flux_to_cgs(F_ul_native, 'MJy/sr', wavelength=wl, header=header, wave_unit='um')
        except Exception:
            F_ul_cgs = None
        if F_ul_cgs is None:
            results.append({"success": False, "message": "Could not convert upper-limit flux to CGS"})
            continue

        # A real detection: fitted flux clears the 3-sigma noise floor. An upper limit:
        # fitted flux doesn't. Either way, compute and keep the real error -- a caller
        # (the GUI) that lets the user manually override the upper-limit call needs a real
        # error already sitting there for that line, not nothing; is_upper_limit only
        # controls how this result gets *displayed/used* downstream, not whether the error
        # exists.
        is_upper_limit = flux_integrated < F_ul_cgs

        # Pascucci's error-propagation formula for area = amplitude*sigma*sqrt(2*pi),
        # assuming amplitude and sigma are uncorrelated:
        #   area_err = sqrt(2*pi * (sigma^2*amp_err^2 + amp^2*sigma_err^2))
        # Fitter-agnostic -- it only needs each parameter's own 1-sigma uncertainty from a
        # least-squares fit, which curve_fit's covariance gives exactly like MPFIT's perror
        # did for her. Scaled into CGS using this line's own already-computed
        # area/area_native ratio (its real per-line conversion factor) rather than
        # re-deriving the conversion separately.
        if fit_result.amplitude_error is None or fit_result.sigma_error is None or not fit_result.area_native:
            flux_error = None
        else:
            area_err_native = np.sqrt(2 * np.pi * (
                fit_result.sigma ** 2 * fit_result.amplitude_error ** 2 +
                fit_result.amplitude ** 2 * fit_result.sigma_error ** 2
            ))
            flux_error_statistical = area_err_native * (fit_result.area / fit_result.area_native)
            # Systematic flux-calibration/continuum floor, added in quadrature -- see
            # SYSTEMATIC_FLUX_ERROR_FRACTION's own comment for why this is needed on top of
            # the statistical error alone.
            flux_error = float(np.sqrt(
                flux_error_statistical ** 2 + (SYSTEMATIC_FLUX_ERROR_FRACTION * flux_integrated) ** 2
            ))

        print(f"[BATCH_FLUX] Line @ {wl:.6f} um (collective):")
        print(f"  flux = {flux_integrated:.4e} erg/s/cm^2")
        print(f"  error = {'N/A (upper limit or unavailable)' if flux_error is None else f'{flux_error:.4e} erg/s/cm^2'}")
        print(f"  noise (native) = {fit_result.noise_std:.4e} MJy/sr")
        print(f"  catalog sigma = {fit_result.catalog_sigma:.4e} um")
        print(f"  F_ul (CGS) = {F_ul_cgs:.4e} erg/s/cm^2")
        print(f"  is_upper_limit = {is_upper_limit}")

        results.append({
            "success": True,
            "message": fit_result.message,
            "flux_cgs": flux_integrated,
            "flux_error_cgs": flux_error,
            "is_upper_limit": is_upper_limit,
            "F_ul_cgs": F_ul_cgs,
            "sigma_noise": fit_result.noise_std,
            "catalog_sigma": fit_result.catalog_sigma,
            "r_squared": fit_result.r_squared,
            # Plot data for a per-line "view fit" display. x_data/y_data/y_fit are
            # already localized to this line's own +/-3*sigma window by fit_glines'
            # _local_fit_record (not the whole chunk/spectrum) -- padded, not the
            # entire bandpass. y_fit is the FULL local model (continuum + every
            # Gaussian in this line's window, not just this line's own) -- same
            # convention fit_glines itself uses, so a real blended neighbor shows up
            # honestly instead of being hidden. baseline is the continuum sampled at
            # those same x points, on the same (continuum-inclusive) scale as y_fit.
            "x_data": fit_result.x_data,
            "y_data": fit_result.y_data,
            "baseline": fit_result.baseline,
            "y_fit": fit_result.y_fit,
            "center": fit_result.center,
            "fwhm": fit_result.fwhm,
            "amplitude": fit_result.amplitude,
        })

    return results


# ==========================================================
# CENSORED MODELS (ONLY USED IF ENABLED)
# ==========================================================
def _print_pymc_diagnostics(trace, var_names):
    """Show PyMC stats in GUI"""

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

def _nutpie_progress_callback(label, progress_callback=None):
    """Enables progress bar.

    nutpie's own progress_callback shape -- a list of PyChainProgress (one per
    chain), each with finished_draws/total_draws/tuning -- rather than PyMC's
    default per-draw (trace, draw) callback. censored_1T/censored_2T call
    nutpie.compile_pymc_model()/nutpie.sample() directly instead of going through
    pm.sample(nuts_sampler='nutpie', callback=...): that wrapper silently drops
    the callback argument, which is why progress reporting went dark before this
    switch to nutpie's own API.
    """
    next_milestone = [0]

    def callback(chain_progresses):
        current = sum(cp.finished_draws for cp in chain_progresses)
        total = sum(cp.total_draws for cp in chain_progresses)
        if total <= 0:
            return
        if next_milestone[0] == 0:
            next_milestone[0] = max(1, total // 10)
        if current >= next_milestone[0] or current == total:
            phase = "tuning" if any(cp.tuning for cp in chain_progresses) else "sampling"
            print(f"[nutpie {label}] {phase}: {current}/{total} ({100*current//total}%)", flush=True)
            if progress_callback is not None:
                progress_callback(current, total)
            next_milestone[0] += max(1, total // 10)

    return callback


def censored_1T(E, y, yerr, is_ul, progress_callback=None):
    """Bayesian 1-temperature rotational diagram fit with censored upper limits.
    
    Uses PyMC to handle mixed detections and upper limits:
    - Detections: fit normally with likelihood
    - Upper limits: one-sided constraint (F < F_ul)
    """
    is_ul = np.asarray(is_ul, dtype=bool)
    # mask = True for detections, False for upper limits
    mask = ~is_ul

    # Fallback width for upper-limit points that don't have their own propagated
    # error (currently always, since _build_dataset places 0.0 there) -- the mean
    # error of the real detections, at least tied to this dataset's own scale
    # rather than an arbitrary constant.
    mean_yerr = np.mean(yerr[mask]) if np.any(mask) else 1.0

    with pm.Model() as model:

        # PRIORS: log(Ncol) ~ Normal(data mean, sigma=5 dex)
        a = pm.Normal("a", mu=np.mean(y[mask]), sigma=5)
        # Temperature range: matches curve_fit's own _bounds_1T ceiling (Pascucci's
        # original notebook bound), so the censored and uncensored fits search the
        # same space.
        T = pm.Uniform("T", 10, 3000)

        # LINEAR: rotational diagram: ln(N/g) = a - E_u/T
        mu = a - E / T

        # LIKELIHOOD: Fit only the non-upper-limit points.
        pm.Normal("obs", mu=mu[mask], sigma=yerr[mask], observed=y[mask])

        # LIKELIHOOD FOR UPPER LIMITS
        # Censored: treat measured value as upper bound. Width comes from this
        # line's own propagated error when available, else mean_yerr above --
        # not the old fixed sigma=1.0, which had no relation to this dataset's
        # actual scale in log(Nu/gu) space.
        if np.any(is_ul):
            ul_sigma = yerr[is_ul] if np.any(yerr[is_ul] > 0) else mean_yerr
            pm.Censored(
                "ul",
                pm.Normal.dist(mu=mu[is_ul], sigma=ul_sigma),
                # Upper censoring: observed value is upper bound
                lower=-np.inf,
                upper=y[is_ul],
                observed=y[is_ul]
            )

        # SAMPLING: nutpie's own compiled (numba) NUTS sampler instead of PyTensor's
        # own execution path -- there's no C++ compiler on this machine, so PyTensor
        # falls back to slow, uncompiled Python, and that scales especially badly
        # with the Censored likelihood's extra graph complexity. Measured ~4-5x
        # faster end-to-end on real fits here. cores=2 is safe with nutpie (chain
        # parallelism runs inside its own compiled runtime, not Python
        # multiprocessing), unlike the Windows process-spawn tax that made cores=2
        # a net loss for PyMC's own default sampler.
        if progress_callback is not None:
            # compile_pymc_model() has no sub-progress hook of its own and is
            # usually the dominant chunk of wall time -- (0, 0) tells the GUI to
            # show a busy/indeterminate bar rather than nothing at all.
            progress_callback(0, 0)
        compiled = nutpie.compile_pymc_model(model)
        trace = nutpie.sample(
            compiled, draws=1000, tune=1000, chains=4, cores=4, seed=_PYMC_RANDOM_SEED,
            target_accept=0.9, progress_bar=False,
            progress_callback=_nutpie_progress_callback("1T", progress_callback=progress_callback),
        )

    diagnostics = _print_pymc_diagnostics(trace, ["a", "T"])

    return {
        "a": float(trace.posterior["a"].mean()),        # log(Ncol)
        "da": float(trace.posterior["a"].std()),        # uncertainty
        "T": float(trace.posterior["T"].mean()),        # temperature
        "dT": float(trace.posterior["T"].std()),        # uncertainty
        "n_divergences": diagnostics["n_divergences"],
        "max_r_hat": diagnostics["max_r_hat"],
    }





def censored_2T(E, y, yerr, is_ul, a1_init, T1_init, progress_callback=None):
    """Bayesian 2-temperature rotational diagram fit with censored upper limits.
    
    Fits a mixture of two thermal components:
    ln(N/g) = log(exp(a1 - E/T1) + exp(a2 - E/T2))
    
    Args:
        E, y, yerr, is_ul: same as censored_1T
        a1_init, T1_init: initial guesses from 1T fit (for better prior)
    """
    is_ul = np.asarray(is_ul, dtype=bool)
    # mask = True for detections, False for upper limits
    mask = ~is_ul

    mean_yerr = np.mean(yerr[mask]) if np.any(mask) else 1.0

    with pm.Model() as model:

        # PRIORS:
        # Component 1 = the cooler component
        a1 = pm.Normal("a1", mu=a1_init, sigma=5)           # log(Ncol), cool component
        T1 = pm.Uniform("T1", 10, 3000)                     # temp, cool component (matches _bounds_2T)

        # Component 2 = the hotter component (T2>T1)
        # Avoid label switching
        a2 = pm.Normal("a2", mu=a1_init-5, sigma=5)         # log(Ncol), hot component
        T2 = pm.Uniform("T2", T1, 3000)                     # temp, hot component (> T1, matches _bounds_2T)

        # TWO-TEMPERATURE MODEL: μ = ln(exp(a1 - E/T1) + exp(a2 - E/T2))
        # Numerically-stable log-sum-exp (factor out the larger term before
        # exponentiating) -- identical result to the plain formula, just avoids
        # overflow/underflow for extreme E/T values.
        term1 = a1 - E / T1
        term2 = a2 - E / T2
        max_term = pm.math.maximum(term1, term2)
        mu = max_term + pm.math.log(pm.math.exp(term1 - max_term) + pm.math.exp(term2 - max_term))

        # LIKELIHOOD: Fit only detections
        pm.Normal("obs", mu=mu[mask], sigma=yerr[mask], observed=y[mask])

        # LIKELIHOOD FOR UPPER LIMITS
        # Censored: treat measured value as upper bound. Width comes from this
        # line's own propagated error when available, else mean_yerr above --
        # not the old fixed sigma=1.0.
        if np.any(is_ul):
            ul_sigma = yerr[is_ul] if np.any(yerr[is_ul] > 0) else mean_yerr
            pm.Censored(
                "ul",
                pm.Normal.dist(mu=mu[is_ul], sigma=ul_sigma),
                lower=-np.inf,
                upper=y[is_ul],
                observed=y[is_ul]
            )

        # SAMPLING: nutpie's own compiled sampler -- see censored_1T for why.
        if progress_callback is not None:
            progress_callback(0, 0)
        compiled = nutpie.compile_pymc_model(model)
        trace = nutpie.sample(
            compiled, draws=1000, tune=1000, chains=4, cores=4, seed=_PYMC_RANDOM_SEED,
            target_accept=0.9, progress_bar=False,
            progress_callback=_nutpie_progress_callback("2T", progress_callback=progress_callback),
        )

    diagnostics = _print_pymc_diagnostics(trace, ["a1", "T1", "a2", "T2"])

    return {
        # Best-fit parameters
        "popt": np.array([
            float(trace.posterior["a1"].mean()),     # log(Ncol1)
            float(trace.posterior["T1"].mean()),     # T_warm
            float(trace.posterior["a2"].mean()),     # log(Ncol2)
            float(trace.posterior["T2"].mean())      # T_cool
        ]),
        # Uncertainties
        "errs": np.array([
            float(trace.posterior["a1"].std()),      # dlog(Ncol1)
            float(trace.posterior["T1"].std()),      # dT_warm
            float(trace.posterior["a2"].std()),      # dlog(Ncol2)
            float(trace.posterior["T2"].std())       # dT_cool
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
                        fit1T=True, fit2T=False, R=None, dist=140*pc, upper_limits=None, progress_callback=None):
    """
    Perform rotational diagram analysis on H2 lines.

    Args:
        fluxes: Dictionary {J: flux_value} where J is the upper rotational level
        flux_errors: Dictionary {J: flux_error}
        molecular_data: Dictionary with 'wavelengths', 'Aul', 'Eu', 'gu' keys
                       Each value is a dict {J: value}
        progress_callback: GUI caller (e.g. a QThread worker) can drive a real
            progress bar instead of only having console prints.
        OPR: Fixed ortho-para ratio. Used as-is when fit_opr=False (None defaults to
             3.0, see apply_OPR). Ignored (overwritten by the search result) when
             fit_opr=True.
        fit_opr: Whether to grid-search over OPR to minimize chi-squared. The search
                 always uses plain curve_fit on detections only (upper-limit points are
                 excluded from the search itself, since curve_fit has no notion of a
                 one-sided constraint) - the final fit below still uses the full data,
                 including upper limits via the censored PyMC path when present.
        opr_grid: OPR values to search over. Defaults to np.linspace(0.1, 5.0, 200).
        fit1T: Whether to report a 1-temperature fit. The 1T fit is always computed the
               same way regardless of this flag (2T's own fit needs it as its (a1, T1)
               seed either way -- see censored_2T/_fit_2T_curvefit); fit1T only controls
               whether it's added to the returned results for reporting. At least one of
               fit1T/fit2T must be True.
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
        - If upper limits present: automatically uses PyMC censored likelihood for the
          final fit, but only for whichever of 1T/2T is actually being reported
        - fit_opr's grid search always uses curve_fit on detections only, regardless of upper_limits
    """
    if not fit1T and not fit2T:
        raise ValueError("At least one of fit1T/fit2T must be True -- nothing to fit.")

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

            # error handling: same dF/F propagation for detections and upper limits
            # alike whenever a real error is available for J -- an upper limit's own
            # propagated error becomes its censoring width in censored_1T/censored_2T
            # (ul_sigma), rather than every upper limit sharing one averaged fallback.
            # Falls back to 0.0 only when no error at all is available for J (inert as
            # a censoring width -- censored_1T/2T fall back to mean_yerr in that case).
            if J in flux_errors:
                dF = flux_errors[J]
                dNu = Nu * (dF / F)
                y_err = dNu / Nu
                y_err_list.append(y_err)
            else:
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

    # A detection's y_err is only ever real data
    det_y_err = y_err[~is_ul_mask]
    if np.any(det_y_err <= 0):
        bad_idx = np.where(~is_ul_mask)[0][det_y_err <= 0]
        bad_E = ", ".join(f"{E_arr[i]:.1f} K" for i in bad_idx)
        raise ValueError(
            f"Detection(s) at E_u = {bad_E} have a flux error of zero, which isn't "
            f"a valid measurement uncertainty. Enter a real error, mark the line as "
            f"an upper limit, or deselect it."
        )

    if has_upper_limits and n_detections < 2:
        if n_detections == 0:
            raise ValueError(f"Cannot fit: all {len(E_arr)} measurements are upper limits. Need at least 2 detections to constrain temperature and column density.")
        else:
            raise ValueError(f"Need at least 2 detections for fitting with upper limits. You have {n_detections} detection(s) and {np.sum(is_ul_mask)} upper limit(s).")

    # Decide method: auto-use PyMC if upper limits present
    use_pymc = has_upper_limits
    # Used below for both 1T's and 2T's own chi2 -- needed regardless of fit1T.
    det_mask = ~is_ul_mask

    # ==========================================================
    # 1T FIT (final)
    # ==========================================================
    # Always computed the same way as before (censored PyMC when upper limits are
    # present, plain curve_fit otherwise) regardless of fit1T -- 2T's own fit needs this
    # as its (a1, T1) seed either way (see censored_2T/_fit_2T_curvefit below), and this
    # is the same proven computation this app has always used for it. fit1T only
    # controls whether the result gets added to `results` for reporting below, not how
    # it's computed.
    if not use_pymc:

        print("\n--- 1T FIT (curve_fit) ---")

        popt1, pcov1 = _fit_1T_curvefit(E_arr, y_arr, y_err)

        a1, T1 = popt1
        da1, dT1 = np.sqrt(np.diag(pcov1))

    else:

        print("\n--- 1T FIT (CENSORED PyMC) ---")

        result = censored_1T(
            E_arr, y_arr, y_err, is_ul_mask,
            progress_callback=(lambda cur, tot: progress_callback("1T", cur, tot)) if progress_callback else None,
        )

        a1, T1 = result["a"], result["T"]
        da1, dT1 = result["da"], result["dT"]
        pymc_diag_1T = {"n_divergences": result["n_divergences"], "max_r_hat": result["max_r_hat"]}

    results = {"OPR": best_opr}
    if best_opr_err is not None:
        results["OPR_err"] = best_opr_err
        results["OPR_err_pos"] = best_opr_err_pos
        results["OPR_err_neg"] = best_opr_err_neg

    if dropped_detections:
        results["dropped_J"] = dropped_detections

    if fit1T:
        # =========================
        # Derived quantities
        # =========================
        Ncol1 = np.exp(a1)
        dNcol1 = Ncol1 * da1

        Ntot1 = Ncol1 * Q_Herbst(T1)
        dNtot1 = Ntot1 * np.sqrt((dNcol1/Ncol1)**2 + (Q_Herbst_err(T1, dT1)/Q_Herbst(T1))**2)

        # chi2 uses detections only - upper limits have no meaningful sigma to weight by
        # (y_err=0 is just a placeholder for them, not a real 1e-10 uncertainty)
        y_model1 = model_1T(E_arr[det_mask], a1, T1)
        chi2_1T = np.sum(((y_arr[det_mask] - y_model1) / np.maximum(y_err[det_mask], 1e-10))**2)
        dof_1T = max(np.sum(det_mask) - 2, 1)

        results["1T"] = {
            "T": T1,
            "dT": dT1,
            "Ncol": Ntot1,
            "dNcol": dNtot1,
            "chi2": chi2_1T,
            "red_chi2": chi2_1T / dof_1T,
        }
        if use_pymc:
            results["1T"]["n_divergences"] = pymc_diag_1T["n_divergences"]
            results["1T"]["max_r_hat"] = pymc_diag_1T["max_r_hat"]

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
                a1, T1,
                progress_callback=(lambda cur, tot: progress_callback("2T", cur, tot)) if progress_callback else None,
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