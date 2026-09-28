"""
H2 rotational diagram fits (1T/2T), with OPR (ortho-para ratio) support and
a hardcoded H2 partition function (Q_Herbst). For any other molecule, see
rotational_diagram_general_fitter.py instead.
"""

import numpy as np
from scipy.optimize import curve_fit
import pymc as pm
import nutpie
import arviz as az

from .unit_converter import integrated_flux_to_cgs
from .flux_map_extractor import fit_auto_continuum, fit_hitran_lines_collective
from .gaussian_fitter import gaussian_flux_upper_limit

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

# Steepness of the upper-limit barrier (upper_limit_1T/2T); no physical meaning.
# Empirically (K swept 1e-3 to 1e6): <~100 doesn't enforce it, >~1e5 breaks the sampler; 1e3 is mid-plateau.
UL_BARRIER_K = 1e3

# curve_fit upper bound on T (K), from rotational_diagram_OPRimplement.ipynb.
# Arbitrary, no theory behind the value; other modules import this rather than repeat it.
T_CAP = 5000

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
    """
    Check if rotational level J is ortho (odd J).

    Args:
        J (int): Rotational quantum number

    Returns:
        is_ortho (bool): True if J is odd
    """
    return (J % 2 == 1)

def get_Ju(SJ):
    """
    Upper rotational level for H2 S-branch line S(SJ).

    Args:
        SJ (int): S-branch line number

    Returns:
        Ju (int): Upper rotational quantum number
    """
    return SJ + 2

def get_gu(SJ):
    """
    Rotational-only degeneracy (2*Ju+1); nuclear spin excluded so apply_OPR can supply it.

    Args:
        SJ (int): S-branch line number

    Returns:
        gu (int): Rotational degeneracy of the upper level
    """
    Ju = get_Ju(SJ)
    return (2*Ju + 1)

def compute_Nu(F, lam, A, R=None, dist=None):
    """
    Compute column density from flux.

    Args:
        F (float): Flux
        lam (float): Wavelength (micron)
        A (float): Einstein coefficient (s^-1)
        R (float): Emitting radius (cm), optional
        dist (float): Distance (cm), optional

    Returns:
        Nu (float): Column density
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
    """
    Apply ortho-para ratio correction.

    Args:
        Nu (float): Column density before correction
        J (int): Rotational quantum number
        OPR (float): Ortho-para ratio; None defaults to 3.0

    Returns:
        Nu (float): Nu divided by OPR (ortho J) or 1.0 (para J)

    Raises:
        ValueError: if OPR is neither None nor a number
    """
    if OPR is None:
        OPR = 3.0

    if isinstance(OPR, (int, float)):
        opr_val = OPR
    else:
        raise ValueError("OPR must be None or a float (use 'fit' in wrapper)")

    f_ortho = opr_val
    f_para  = 1.0

    return Nu / (f_ortho if is_ortho(J) else f_para)


def Q_Herbst(T):
    """
    Herbst partition function for H2.

    Args:
        T (float): Temperature in K

    Returns:
        Q (float): Partition function value at T
    """
    return 0.0247 * T / (1 - np.exp(-6000/T))


def Q_Herbst_err(T, T_err):
    """
    Analytic error on Q_Herbst from a temperature uncertainty.

    Args:
        T (float): Temperature in K
        T_err (float): Uncertainty on T

    Returns:
        dQ (float): Propagated uncertainty on Q(T)
    """
    a = 6000.0
    c = 0.0247
    ea = np.exp(-a/T)
    dQdT = c * (1/(1-ea) - (a*ea)/(T*(1-ea)**2))
    return np.abs(dQdT) * T_err


def model_1T(E, a, T):
    """
    Single temperature rotational diagram model.

    Args:
        E (float or array): Upper-level energy (K)
        a (float): log(Ncol) intercept
        T (float): Temperature (K)

    Returns:
        y (float or array): a - E/T
    """
    return a - E / T


def model_2T(E, a1, T1, a2, T2):
    """
    Two temperature rotational diagram model.

    Args:
        E (float or array): Upper-level energy (K)
        a1 (float): log(Ncol) intercept, component 1
        T1 (float): Temperature (K), component 1
        a2 (float): log(Ncol) intercept, component 2
        T2 (float): Temperature (K), component 2

    Returns:
        y (float or array): log(exp(a1-E/T1) + exp(a2-E/T2))
    """
    return np.log(np.exp(a1 - E/T1) + np.exp(a2 - E/T2))


'''
UPPER LIMITS
'''

def compute_line_fluxes_collective(wavelength_arr, spectrum_arr, line_wavelengths, header,
                                    contam_lines=None, n_sigma=3.0):
    """
    Fit every line in `line_wavelengths` simultaneously in one spectrum
    (fit_hitran_lines_collective) -- usually the better fit for rotational-diagram lines.

    Args:
        wavelength_arr (array): Wavelength values for the (spatially-summed) spectrum
        spectrum_arr (array): Flux values for the spectrum
        line_wavelengths (list): Target line center wavelengths (microns)
        header (astropy.io.fits.Header): FITS header (instrument read from it)
        contam_lines (pd.DataFrame): Optional contaminating lines to fit alongside
        n_sigma (float): Detection threshold in noise sigmas: a line below n_sigma
            is an upper limit. 3.0 by default; Calculate Fluxes always uses 3.0

    Returns:
        results (list): One dict per line_wavelengths entry: success (bool), message (str),
            and on success also flux_cgs, flux_error_cgs, is_upper_limit, F_ul_cgs (the
            n_sigma ceiling), sigma_noise, catalog_sigma, r_squared, plus per-line plot data
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
        # resolution -- surfaced as a message per line, not an uncaught crash.
        return [{"success": False, "message": f"Collective fit failed: {e}"} for _ in line_wavelengths]
    if velocity_shift_kms is not None:
        print(f"[BATCH_FLUX] Collective fit shared velocity shift: {velocity_shift_kms:.2f} km/s")

    results = []
    for line, fit_result in fit_pairs:
        wl = line['wavelength']

        if not fit_result.success:
            results.append({"success": False, "message": fit_result.message})
            continue

        # noise_std/catalog_sigma come from fit_glines' own fit; both None only when that
        # chunk had too few points to estimate noise, so we can't judge detection vs upper limit.
        if fit_result.noise_std is None or fit_result.catalog_sigma is None:
            results.append({"success": False, "message": "No noise estimate available for this line's chunk"})
            continue

        # Already CGS (erg/s/cm^2) -- the real fitted flux, from curve_fit's own amplitude/sigma
        flux_integrated = fit_result.area

        F_ul_native = gaussian_flux_upper_limit(fit_result.noise_std, fit_result.catalog_sigma, n_sigma)
        try:
            F_ul_cgs = integrated_flux_to_cgs(F_ul_native, 'MJy/sr', wavelength=wl, header=header, wave_unit='um')
        except Exception:
            F_ul_cgs = None
        if F_ul_cgs is None:
            results.append({"success": False, "message": "Could not convert upper-limit flux to CGS"})
            continue

        # Detection if flux clears the n_sigma floor (a zero flux never does, even at n_sigma=0),
        # else upper limit -- either way keep the real error, since is_upper_limit only controls display.
        is_upper_limit = flux_integrated < F_ul_cgs or flux_integrated <= 0

        # fit_result.area_error already includes the systematic floor -- computed once in
        # flux_map_extractor._to_result, same number every consumer gets, not redone here.
        flux_error = fit_result.area_error

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
            # Plot data for a per-line view: x_data/y_fit span the whole shared fit chunk (not a
            # separate crop); y_fit is the full local model, so a blended neighbor shows up honestly.
            "x_data": fit_result.x_data,
            "y_data": fit_result.y_data,
            "baseline": fit_result.baseline,
            "y_fit": fit_result.y_fit,
            "y_fit_own": fit_result.y_fit_own,
            "center": fit_result.center,
            "sigma": fit_result.sigma,
            "fwhm": fit_result.fwhm,
            "amplitude": fit_result.amplitude,
        })

    return results


# Upper-limit models (only used if enabled)
def _print_pymc_diagnostics(trace, var_names):
    """
    Show PyMC convergence stats in the GUI/console.

    Args:
        trace (arviz.InferenceData): PyMC sampling trace
        var_names (list): Variable names to summarize

    Returns:
        diagnostics (dict): "n_divergences" (int), "max_r_hat" (float)
    """
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
    """
    Adapt nutpie's own progress_callback shape (list of PyChainProgress) to this
    app's (current, total) shape -- pm.sample(nuts_sampler='nutpie') silently
    drops the callback, so upper_limit_1T/2T call nutpie's API directly instead.

    Args:
        label (str): "1T" or "2T", used in the printed progress line
        progress_callback (callable): (current, total) -> None, or None to skip

    Returns:
        callback (callable): nutpie-compatible progress callback
    """
    next_milestone = [0]

    def callback(chain_progresses):
        """Print and forward progress every ~10% of total draws.

        Args:
            chain_progresses (list): nutpie PyChainProgress objects, one per chain.

        Returns:
            None
        """
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


def upper_limit_1T(E, y, yerr, is_ul, progress_callback=None, t_cap=T_CAP):
    """
    Bayesian 1-temperature rotational diagram fit with hard upper-limit bounds:
    detections get a normal likelihood, upper limits get a steep one-sided
    barrier (mu must stay at/below the limit) instead of a soft Censored one.

    Args:
        E (array): Upper-level energy (K)
        y (array): log(Nu/gu)
        yerr (array): Uncertainty on y
        is_ul (array): True where the point is an upper limit
        progress_callback (callable): (current, total) -> None, for a GUI progress bar
        t_cap (float): Upper bound on T for the sampler's prior

    Returns:
        result (dict): "a"/"da"/"T"/"dT" (posterior mean/std), plus PyMC diagnostics
            "n_divergences"/"max_r_hat"
    """
    is_ul = np.asarray(is_ul, dtype=bool)
    # mask = True for detections, False for upper limits
    mask = ~is_ul

    with pm.Model() as model:

        # PRIORS: log(Ncol) ~ Normal(data mean, sigma=5 dex)
        a = pm.Normal("a", mu=np.mean(y[mask]), sigma=5)
        # Temperature range: matches curve_fit's own _bounds_1T ceiling, so
        # this and the plain curve_fit path search the same space.
        T = pm.Uniform("T", 10, t_cap)

        # LINEAR: rotational diagram: ln(N/g) = a - E_u/T
        mu = a - E / T

        # LIKELIHOOD: Fit only the non-upper-limit points.
        pm.Normal("obs", mu=mu[mask], sigma=yerr[mask], observed=y[mask])

        # Zero cost while mu <= limit, quadratic cost past it -- unlike a Censored
        # likelihood's CDF tail (weakest right at the boundary), this penalizes it hardest there.
        if np.any(is_ul):
            excess = pm.math.maximum(0, mu[is_ul] - y[is_ul])
            pm.Potential("ul_barrier", -UL_BARRIER_K * pm.math.sum(excess**2))

        # nutpie's compiled NUTS sampler -- no C++ compiler here, so PyTensor's own path
        # falls back to slow Python; nutpie measured ~4-5x faster end-to-end.
        if progress_callback is not None:
            # compile_pymc_model() has no sub-progress hook and is usually the dominant
            # chunk of wall time -- (0, 0) tells the GUI to show a busy/indeterminate bar.
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


def upper_limit_2T(E, y, yerr, is_ul, a1_init, T1_init, progress_callback=None, t_cap=T_CAP):
    """
    Bayesian 2-temperature rotational diagram fit with hard upper-limit bounds:
    ln(N/g) = log(exp(a1 - E/T1) + exp(a2 - E/T2)), same barrier as upper_limit_1T.

    Args:
        E, y, yerr, is_ul (array): Same as upper_limit_1T
        a1_init, T1_init (float): 1T fit result, seeds the cool-component prior
        progress_callback (callable): (current, total) -> None, for a GUI progress bar
        t_cap (float): Upper bound on T1/T2 for the sampler's prior

    Returns:
        result (dict): "popt"/"errs" ([a1,T1,a2,T2] posterior mean/std), plus
            PyMC diagnostics "n_divergences"/"max_r_hat"
    """
    is_ul = np.asarray(is_ul, dtype=bool)
    # mask = True for detections, False for upper limits
    mask = ~is_ul

    with pm.Model() as model:

        # PRIORS:
        # Component 1 = the cooler component
        a1 = pm.Normal("a1", mu=a1_init, sigma=5)           # log(Ncol), cool component
        T1 = pm.Uniform("T1", 10, t_cap)                    # temp, cool component (matches _bounds_2T)

        # Component 2 = the hotter component (T2>T1)
        # Avoid label switching
        a2 = pm.Normal("a2", mu=a1_init-5, sigma=5)         # log(Ncol), hot component
        T2 = pm.Uniform("T2", T1, t_cap)                    # temp, hot component (> T1, matches _bounds_2T)

        # mu = ln(exp(a1-E/T1)+exp(a2-E/T2)), via numerically-stable log-sum-exp
        # (factor out the larger term first) -- same result, avoids overflow/underflow.
        term1 = a1 - E / T1
        term2 = a2 - E / T2
        max_term = pm.math.maximum(term1, term2)
        mu = max_term + pm.math.log(pm.math.exp(term1 - max_term) + pm.math.exp(term2 - max_term))

        # LIKELIHOOD: Fit only detections
        pm.Normal("obs", mu=mu[mask], sigma=yerr[mask], observed=y[mask])

        # Same steep one-sided penalty as upper_limit_1T -- zero cost while mu <= the
        # limit, quadratically expensive past it.
        if np.any(is_ul):
            excess = pm.math.maximum(0, mu[is_ul] - y[is_ul])
            pm.Potential("ul_barrier", -UL_BARRIER_K * pm.math.sum(excess**2))

        # SAMPLING: nutpie's own compiled sampler -- see upper_limit_1T for why.
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

def partitionH2(T, OPR='equilibrium'):
    """
    H2 partition function, Popovas & Jorgensen (2016). Unused (Q_Herbst is the
    active H2 partition function) -- kept for reference alongside Q_total below.
    https://ui.adsabs.harvard.edu/abs/2016A%26A...595A.130P/abstract

    Args:
        T (float or array): Temperature in K
        OPR (str): 'normal', 'equilibrium', 'ortho', or 'para'

    Returns:
        Q, loQ, miQ, hiQ (tuple): Combined Q(T) and its three piecewise polynomial terms

    Raises:
        NotImplementedError: if OPR isn't one of the four recognized options
    """
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
    """
    H2 partition function combining ortho/para components, weighted by OPR.
    Unused (partition_function/Q_Herbst are the active ones) -- kept for reference.

    Args:
        T (float or array): Temperature in K
        OPR (float): Ortho-para ratio weight

    Returns:
        Q (float or array): f_para*Q_para + OPR*Q_ortho
    """
    Q_ortho, _, _, _ = partitionH2(T, OPR='ortho')
    Q_para, _, _, _ = partitionH2(T, OPR='para')

    f_ortho = OPR
    f_para = 1.0

    return f_para * Q_para + f_ortho * Q_ortho


def Q_total_err(T, dT, OPR, dOPR=0.0):
    """
    Uncertainty on Q_total propagated from dT and dOPR, via finite differences.
    Unused (Q_total is unused) -- kept for reference.

    Args:
        T (float): Temperature in K
        dT (float): Uncertainty on T
        OPR (float): Ortho-para ratio
        dOPR (float): Uncertainty on OPR

    Returns:
        dQ (float): Propagated uncertainty on Q_total(T, OPR)
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
    Fit an H2 rotational diagram (1T and/or 2T) to a set of line fluxes.

    Args:
        fluxes (dict): {J: flux_value}, J is the upper rotational level
        flux_errors (dict): {J: flux_error}
        molecular_data (dict): 'wavelengths', 'Aul', 'Eu', 'gu' keys, each {J: value}
        OPR (float): Fixed ortho-para ratio. None defaults to 3.0
        fit_opr (bool): Grid-search OPR to minimize chi-squared
        opr_grid (array): OPR values to search. Defaults to np.linspace(0.1, 5.0, 200)
        fit1T (bool): Report the 1T fit (always computed regardless, as 2T's own seed)
        fit2T (bool): Perform the 2T fit
        R (float): Emitting radius (AU or cm)
        dist (float): Distance (pc or cm)
        upper_limits (dict): {J: True} for measurements marked as upper limits
        progress_callback (callable): (phase, current, total) -> None, for a GUI progress bar

    Returns:
        results (dict): Fit results, always including "OPR"/"OPR_err"/"OPR_err_pos"/"OPR_err_neg"

    Raises:
        ValueError: if neither fit1T nor fit2T is True
    """
    if not fit1T and not fit2T:
        raise ValueError("At least one of fit1T/fit2T must be True -- nothing to fit.")

    t_cap = T_CAP

    global wave, Aul, Eu, gu

    if molecular_data is not None:
        wave = molecular_data.get('wavelengths', {})
        Aul = molecular_data.get('Aul', {})
        Eu = molecular_data.get('Eu', {})
        gu = molecular_data.get('gu', {})

    # Build dataset for a given trial OPR
    def _build_dataset(opr_value):
        """
        Compute excitation diagram arrays for a given OPR.

        Args:
            opr_value (float): Ortho-para ratio to apply

        Returns:
            E, y, y_err, is_ul, dropped (tuple): Arrays (E in K, y=log(Nu/gu)) plus
                the list of J values dropped for missing error data
        """
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

            # Same dF/F propagation for detections and upper limits; an upper limit's own
            # error isn't used by the barrier fit, kept for display. Falls back to 0.0 if missing.
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

    # Bounded curve_fit helpers, used for the OPR grid search and the final
    # fit whenever PyMC/upper-limit handling isn't required.
    def _bounds_1T(y_arr):
        """
        Bounds for the single-temperature model.

        Args:
            y_arr (array): y values (log(Nu/gu)) in the current dataset

        Returns:
            bounds (tuple): (lower, upper) bound lists for curve_fit
        """
        return ([np.min(y_arr), 10],
                [np.max(y_arr) + 10, t_cap])

    def _bounds_2T(y_arr):
        """
        Bounds for the two-temperature model.

        Args:
            y_arr (array): y values (log(Nu/gu)) in the current dataset

        Returns:
            bounds (tuple): (lower, upper) bound lists for curve_fit
        """
        lo = np.min(y_arr)
        hi = np.max(y_arr) + 10
        return ([lo, 10, lo, 10],
                [hi, t_cap, hi, t_cap])

    def _fit_1T_curvefit(E, y, y_err):
        """
        Perform a single-temperature fit.

        Args:
            E, y, y_err (array): Rotational-diagram data

        Returns:
            popt, pcov (tuple): curve_fit's best-fit params and covariance
        """
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
        """
        Perform a two-temperature fit seeded from a 1T solution.

        Args:
            E, y, y_err (array): Rotational-diagram data
            a_seed, T_seed (float): 1T fit result, seeds component 1

        Returns:
            popt, pcov (tuple): curve_fit's best-fit params and covariance
        """
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
        """
        Chi-squared for a quick fit (1T or 2T) on detections only -- upper limits
        are reintroduced later via the PyMC barrier path, not here.

        Args:
            E_arr, y_arr, y_err (array): Rotational-diagram data
            is_ul_mask (array): True where the point is an upper limit
            use_2T (bool): Fit the two-temperature model instead of one

        Returns:
            chi2 (float): Sum of squared normalized residuals, or inf if underdetermined
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

    def _derive_column(a, da, T, dT):
        """
        Convert log-intercept to total column density with uncertainty.

        Args:
            a, da (float): log(Ncol) intercept and its uncertainty
            T, dT (float): Temperature (K) and its uncertainty

        Returns:
            Ntot, dNtot (tuple): Total column density and its uncertainty
        """
        Ncol = np.exp(a)
        dNcol = Ncol * da
        Ntot = Ncol * Q_Herbst(T)
        dNtot = Ntot * np.sqrt(
            (dNcol / Ncol) ** 2
            + (Q_Herbst_err(T, dT) / Q_Herbst(T)) ** 2)
        return Ntot, dNtot

    # OPR grid search (fast, detections-only, curve_fit)
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
        err_opr_pos = opr_grid[np.where(chi2_arr < chi2_arr[best_idx] + 1)[0][-1]] - best_opr
        err_opr_neg = best_opr - opr_grid[np.where(chi2_arr < chi2_arr[best_idx] + 1)[0][0]]
        best_opr_err = 0.5 * (err_opr_pos + err_opr_neg)
        best_opr_err_pos = err_opr_pos
        best_opr_err_neg = err_opr_neg

        print("\n--- OPR FIT (curve_fit grid search, detections only) ---")
        print(f"Best-fit OPR = {best_opr:.3f}")
        print(f"OPR 1-sigma error: -{err_opr_neg:.4f} / +{err_opr_pos:.4f} (symm ~= {best_opr_err:.4f})")

        opr_search_data = {"opr_grid": opr_grid, "chi2_list": chi2_arr}

    # Build final dataset at best_opr (full data, including upper limits)
    E_arr, y_arr, y_err, is_ul_mask, dropped_detections = _build_dataset(best_opr)

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

    # 1T fit (final): always computed as 2T's own seed regardless of fit1T,
    # which only controls whether it's reported below.
    if not use_pymc:

        print("\n--- 1T FIT (curve_fit) ---")

        popt1, pcov1 = _fit_1T_curvefit(E_arr, y_arr, y_err)

        a1, T1 = popt1
        da1, dT1 = np.sqrt(np.diag(pcov1))

    else:

        print("\n--- 1T FIT (PyMC, upper-limit barrier) ---")

        result = upper_limit_1T(
            E_arr, y_arr, y_err, is_ul_mask,
            progress_callback=(lambda cur, tot: progress_callback("1T", cur, tot)) if progress_callback else None,
            t_cap=t_cap,
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
        # Derived quantities
        Ntot1, dNtot1 = _derive_column(a1, da1, T1, dT1)

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

    # 2T fit (final)
    if fit2T:

        if not use_pymc:

            print("\n--- 2T FIT (curve_fit) ---")

            popt2, pcov2 = _fit_2T_curvefit(E_arr, y_arr, y_err, a1, T1)

            a1_2, T1_2, a2_2, T2_2 = popt2
            errs = np.sqrt(np.diag(pcov2))

        else:

            print("\n--- 2T FIT (PyMC, upper-limit barrier) ---")

            result = upper_limit_2T(
                E_arr, y_arr, y_err, is_ul_mask,
                a1, T1,
                progress_callback=(lambda cur, tot: progress_callback("2T", cur, tot)) if progress_callback else None,
                t_cap=t_cap,
            )

            a1_2, T1_2, a2_2, T2_2 = result["popt"]
            errs = result["errs"]
            pymc_diag_2T = {"n_divergences": result["n_divergences"], "max_r_hat": result["max_r_hat"]}

        da1_2, dT1_2, da2_2, dT2_2 = errs

        N1_tot, dN1_tot = _derive_column(a1_2, da1_2, T1_2, dT1_2)
        N2_tot, dN2_tot = _derive_column(a2_2, da2_2, T2_2, dT2_2)

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
