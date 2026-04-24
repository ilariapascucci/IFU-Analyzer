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

def gaussian_flux_upper_limit(sigma_noise, wavelength, R):
    A_ul = 3.0 * sigma_noise
    delta_lambda = wavelength / R
    sigma_lambda = delta_lambda / 2.355
    F_ul = A_ul * np.sqrt(2 * np.pi) * sigma_lambda
    return F_ul, sigma_lambda


def flux_to_logNu_upper(F_ul, lam, Aul_val, gu_val, R=None, dist=None):
    lam_cm = lam * 1e-4
    Nu_ul = (4*np.pi * F_ul * lam_cm) / (hplanck * clight * Aul_val)

    if R is not None and dist is not None:
        arcsec2 = np.pi * (R / dist)**2
        sr_arcsec2 = 4.25e10
        Nu_ul = Nu_ul / (arcsec2 / sr_arcsec2)

    return np.log(Nu_ul / gu_val)


def rotational_diagram(fluxes, flux_errors, molecular_data=None, OPR=None, fit2T=False, R=None, dist=140*pc, upper_limits=None, use_censored=False):
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
        upper_limits: Optional dict {J: True} for measurements marked as upper limits
    
    Returns:
        Dictionary with fitting results
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

        E_arr.append(Eu[J])
        y_arr.append(y)

        is_ul = upper_limits is not None and J in upper_limits
        is_ul_mask.append(is_ul)

        # error handling
        if (not is_ul) and (J in flux_errors):
            dF = flux_errors[J]
            dNu = Nu * (dF / F)
            y_err.append(dNu / Nu)
        else:
            y_err.append(0.0 if is_ul else 1.0)

    E_arr = np.array(E_arr)
    y_arr = np.array(y_arr)
    y_err = np.array(y_err)
    is_ul_mask = np.array(is_ul_mask)

    if len(E_arr) < 2:
        raise ValueError("Need at least 2 points")

    # =========================
    # Decide method
    # =========================
    has_upper_limits = np.any(is_ul_mask)
    use_pymc = use_censored and has_upper_limits

    # ==========================================================
    # 1T FIT
    # ==========================================================
    if not use_pymc:

        print("\n--- 1T FIT (curve_fit) ---")

        sigma = y_err if np.any(y_err > 0) else None

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
            "popt1": list(popt1) if not use_pymc else [a1, T1],
            "popt2": list(popt2) if not use_pymc else list(result["popt"]),
        }
    else:
        results["_plot_data"] = {
            "E_arr": E_arr,
            "y_arr": y_arr,
            "y_err": y_err,
            "popt1": list(popt1) if not use_pymc else [a1, T1],
            "popt2": None,
        }

    return results


# ==========================================================
# CENSORED MODELS (ONLY USED IF ENABLED)
# ==========================================================
def censored_1T(E, y, yerr, is_ul):

    mask = ~is_ul

    with pm.Model() as model:

        a = pm.Normal("a", mu=np.mean(y[mask]), sigma=5)
        T = pm.Uniform("T", 10, 5000)

        mu = a - E / T

        pm.Normal("obs", mu=mu[mask], sigma=1.0, observed=y[mask])

        if np.any(is_ul):
            pm.Censored(
                "ul",
                pm.Normal.dist(mu=mu[is_ul], sigma=1.0),
                upper=y[is_ul],
                observed=y[is_ul]
            )

        trace = pm.sample(500, tune=500, chains=1, cores=1, progressbar=False)

    return {
        "a": float(trace.posterior["a"].mean()),
        "da": float(trace.posterior["a"].std()),
        "T": float(trace.posterior["T"].mean()),
        "dT": float(trace.posterior["T"].std())
    }


def censored_2T(E, y, yerr, is_ul, a1_init, T1_init):

    mask = ~is_ul

    with pm.Model() as model:

        a1 = pm.Normal("a1", mu=a1_init, sigma=5)
        T1 = pm.Uniform("T1", 10, 5000)
        a2 = pm.Normal("a2", mu=a1_init-5, sigma=5)
        T2 = pm.Uniform("T2", 10, 5000)

        mu = pm.math.log(
            pm.math.exp(a1 - E/T1) +
            pm.math.exp(a2 - E/T2)
        )

        pm.Normal("obs", mu=mu[mask], sigma=1.0, observed=y[mask])

        if np.any(is_ul):
            pm.Censored(
                "ul",
                pm.Normal.dist(mu=mu[is_ul], sigma=1.0),
                upper=y[is_ul],
                observed=y[is_ul]
            )

        trace = pm.sample(500, tune=500, chains=1, cores=1, progressbar=False)

    return {
        "popt": np.array([
            float(trace.posterior["a1"].mean()),
            float(trace.posterior["T1"].mean()),
            float(trace.posterior["a2"].mean()),
            float(trace.posterior["T2"].mean())
        ]),
        "errs": np.array([
            float(trace.posterior["a1"].std()),
            float(trace.posterior["T1"].std()),
            float(trace.posterior["a2"].std()),
            float(trace.posterior["T2"].std())
        ])
    }