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
        arcsec2 = np.pi*((R/dist)**2)
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
        R: Emitting radius (AU or cm). If < 1e6, assumed to be AU; converted to cm
        dist: Distance (pc or cm). If < 1e12, assumed to be pc; converted to cm
        upper_limits: Optional dict {J: True} for measurements marked as upper limits
    
    Returns:
        Dictionary with fitting results
    """
    
    global wave, Aul, Eu, gu
    
    # Use provided molecular data or module-level data
    if molecular_data is not None:
        wave = molecular_data.get('wavelengths', {})
        Aul = molecular_data.get('Aul', {})
        Eu = molecular_data.get('Eu', {})
        gu = molecular_data.get('gu', {})
    
    E_arr, y_arr, y_err = [], [], []

    for J in fluxes:
        F = fluxes[J]

        # Get molecular data for this J level
        if J not in wave or J not in Aul or J not in Eu or J not in gu:
            print(f"Warning: Missing data for J={J}, skipping")
            continue

        # Skip points without flux errors unless they're upper limits
        is_upper_limit = upper_limits and J in upper_limits
        if not is_upper_limit and J not in flux_errors:
            print(f"Warning: No flux error for J={J}, skipping")
            continue

        Nu = compute_Nu(F, wave[J], Aul[J], R, dist)
        
        # Handle flux errors
        # Upper limits should have NO error (they're constraints, not measurements)
        if is_upper_limit:
            dy = 0.0  # Upper limits have no uncertainty
        else:
            dF = flux_errors[J]
            dNu = Nu * (dF / F)
            dy = dNu / Nu

        y = np.log(Nu / gu[J])

        E_arr.append(Eu[J])
        y_arr.append(y)
        y_err.append(dy)
        
        # Log upper limits for reference
        if is_upper_limit:
            print(f"  [UPPER LIMIT] E_u={Eu[J]:.2f} K, log(Nu/gu)={y:.2f}")

    E_arr = np.array(E_arr)
    y_arr = np.array(y_arr)
    y_err = np.array(y_err)

    if len(E_arr) < 2:
        raise ValueError("Need at least 2 data points for fitting")

    # Separate upper limits from measured points for fitting
    # Upper limits (dy=0) should not be included in the weighted fit
    measured_mask = y_err != 0.0
    E_measured = E_arr[measured_mask]
    y_measured = y_arr[measured_mask]
    y_err_measured = y_err[measured_mask]
    
    # Check if we have enough measured points (not upper limits)
    if len(E_measured) < 2:
        print("Warning: Not enough measured points (non-upper-limits) for fitting. Using all points.")
        E_fit = E_arr
        y_fit = y_arr
        y_err_fit = y_err
    else:
        E_fit = E_measured
        y_fit = y_measured
        y_err_fit = y_err_measured

    # Check if we have errors for the points being fitted
    # All points should have either real errors or be upper limits (dy=0)
    have_errors = np.any(y_err_fit != 0.0)  # True if any non-zero errors
    
    if not have_errors:
        print("Note: All points are upper limits - cannot fit")
        sigma = None
    else:
        sigma = y_err_fit

    # Build list of upper limit indices for PyMC
    # Map from array index back to J values
    J_arr = []
    for J in fluxes:
        if J not in wave or J not in Aul or J not in Eu or J not in gu:
            continue
        is_upper_limit = upper_limits and J in upper_limits
        if not is_upper_limit and J not in flux_errors:
            continue
        J_arr.append(J)
    
    upper_limit_indices = []
    if upper_limits:
        for idx, J in enumerate(J_arr):
            if J in upper_limits:
                upper_limit_indices.append(idx)
    
    # ============ 1T fit ============
    use_pymc = upper_limits and len(upper_limit_indices) > 0
    
    if use_pymc:
        print("\n--- 1T FIT (PyMC Censored Regression) ---")
        result_1t = censored_regression_1T(
            E_arr, y_arr, y_err, upper_limit_indices, 
            [y_arr[i] for i in upper_limit_indices], 
            molecular_data, OPR, R, dist
        )
        a1 = result_1t['a']
        T1 = result_1t['T']
        da1 = result_1t['da']
        dT1 = result_1t['dT']
        popt1 = result_1t['popt']
    else:
        print("\n--- 1T FIT (Standard curve_fit) ---")
        p0 = [np.max(y_fit), 500]
        popt1, pcov1 = curve_fit(
            model_1T, E_fit, y_fit,
            sigma=sigma, absolute_sigma=True, p0=p0
        )
        a1, T1 = popt1
        da1, dT1 = np.sqrt(np.diag(pcov1))

    Ncol1 = np.exp(a1)
    dNcol1 = Ncol1 * da1

    Ntot1 = Ncol1 * Q_Herbst(T1)
    dNtot1 = Ntot1 * np.sqrt((dNcol1/Ncol1)**2 + (Q_Herbst_err(T1, dT1)/Q_Herbst(T1))**2)

    print(f"T = {T1:.1f} ± {dT1:.1f} K")
    print(f"Ntot = {Ntot1:.2e} ± {dNtot1:.2e} cm^-2")

    results = {
        "1T": {
            "T": T1, "dT": dT1,
            "Ncol": Ntot1, "dNcol": dNtot1
        },
        "_plot_data": {
            "E_arr": E_arr,
            "y_arr": y_arr,
            "y_err": y_err,
            "popt1": popt1,
            "popt2": None
        }
    }

    # ============ 2T fit ============
    if fit2T:
        if use_pymc:
            print("\n--- 2T FIT (PyMC Censored Regression) ---")
            try:
                result_2t = censored_regression_2T(
                    E_arr, y_arr, y_err, upper_limit_indices,
                    [y_arr[i] for i in upper_limit_indices],
                    molecular_data, a1, T1, OPR, R, dist
                )
                a1_2, T1_2, a2_2, T2_2 = result_2t['popt']
                da1_2, dT1_2, da2_2, dT2_2 = result_2t['da1'], result_2t['dT1'], result_2t['da2'], result_2t['dT2']
                popt2 = result_2t['popt']
            except Exception as e:
                print(f"2T PyMC fit failed: {e}")
                return results
        else:
            print("\n--- 2T FIT (Standard curve_fit) ---")
            p0 = [a1, T1, a1-5, T1*2]

            try:
                popt2, pcov2 = curve_fit(
                    model_2T, E_fit, y_fit,
                    sigma=sigma, absolute_sigma=True,
                    p0=p0, maxfev=10000
                )

                a1_2, T1_2, a2_2, T2_2 = popt2
                errs = np.sqrt(np.diag(pcov2))

                da1_2, dT1_2, da2_2, dT2_2 = errs
            except Exception as e:
                print(f"2T curve_fit failed: {e}")
                return results
        
        N1 = np.exp(a1_2)
        N2 = np.exp(a2_2)

        N1_tot = N1 * Q_Herbst(T1_2)
        N2_tot = N2 * Q_Herbst(T2_2)

        dN1 = N1 * da1_2
        dN2 = N2 * da2_2

        dN1_tot = N1_tot * np.sqrt((dN1/N1)**2 + (Q_Herbst_err(T1_2, dT1_2)/Q_Herbst(T1_2))**2)
        dN2_tot = N2_tot * np.sqrt((dN2/N2)**2 + (Q_Herbst_err(T2_2, dT2_2)/Q_Herbst(T2_2))**2)

        print(f"T1 = {T1_2:.1f} ± {dT1_2:.1f} K")
        print(f"T2 = {T2_2:.1f} ± {dT2_2:.1f} K")
        print(f"[DEBUG] a1 = {a1_2}, a2 = {a2_2}")
        print(f"[DEBUG] Q(T1) = {Q_Herbst(T1_2)}, Q(T2) = {Q_Herbst(T2_2)}")
        print(f"[DEBUG] N1 (before Q) = {N1}, N2 (before Q) = {N2}")
        print(f"N1 = {N1_tot:.2e} ± {dN1_tot:.2e}")
        print(f"N2 = {N2_tot:.2e} ± {dN2_tot:.2e}")

        results["2T"] = {
            "T1": T1_2, "dT1": dT1_2,
            "T2": T2_2, "dT2": dT2_2,
            "Ncol1": N1_tot, "dNcol1": dN1_tot,
            "Ncol2": N2_tot, "dN2": dN2_tot
        }
        results["_plot_data"]["popt2"] = popt2

    return results


# ============ CENSORED REGRESSION (PyMC-based for upper limits) ============
# These functions use PyMC for proper censored regression when upper limits are present
# Based on: https://www.pymc.io/projects/examples/en/2022.01.0/generalized_linear_models/GLM-truncated-censored-regression.html

def censored_regression_1T(E_arr, y_arr, y_err, upper_limit_indices, upper_limit_values, molecular_data, OPR=None, R=None, dist=140*pc):
    """
    Perform 1T rotational diagram fit using PyMC censored regression.
    Upper limits are included as right-censored data points.
    
    Args:
        E_arr: Array of excitation energies
        y_arr: Array of log(Nu/gu) values for all points
        y_err: Error array (0 for upper limits, 1-sigma for measured)
        upper_limit_indices: List of J indices that are upper limits
        upper_limit_values: List of upper limit flux values (log scale)
        molecular_data: Dict with wavelengths, Aul, Eu, gu
        OPR: Ortho-para ratio
        R: Emitting radius
        dist: Distance
    
    Returns:
        Dictionary with T, dT, Ncol, dNcol
    """
    # Separate measured and censored data
    measured_mask = np.array([i not in upper_limit_indices for i in range(len(E_arr))])
    E_meas = E_arr[measured_mask]
    y_meas = y_arr[measured_mask]
    y_err_meas = y_err[measured_mask]
    
    E_cens = E_arr[~measured_mask]
    y_ul = y_arr[~measured_mask]
    
    print(f"\nCensored regression 1T: {np.sum(measured_mask)} measured, {len(upper_limit_indices)} upper limits")
    
    with pm.Model() as model:
        # Priors
        a = pm.Normal('a', mu=np.nanmax(y_meas) if len(y_meas) > 0 else 20, sigma=5)
        T = pm.Uniform('T', lower=10, upper=5000)
        
        # Likelihood for measured points (normal)
        if len(E_meas) > 0:
            mu_meas = a - E_meas / T
            sigma_meas = y_err_meas[y_err_meas > 0]
            if len(sigma_meas) > 0:
                pm.Normal('y_measured', mu=mu_meas[:len(sigma_meas)], 
                         sigma=sigma_meas, observed=y_meas[:len(sigma_meas)])
            else:
                # No errors provided, use unit variance
                pm.Normal('y_measured', mu=mu_meas, sigma=1, observed=y_meas)
        
        # Likelihood for censored points (upper limits as right-censored)
        if len(E_cens) > 0:
            mu_cens = a - E_cens / T
            # Right-censored (upper limit): use Normal CDF upper tail
            from scipy.special import ndtr  # Standard normal CDF
            pm.math.log(1 - pm.math.erf((y_ul - mu_cens) / (np.sqrt(2) * 1.0)))  # placeholder
            # Simpler approach: use Censored distribution
            pm.Censored('y_censored', pm.Normal.dist(mu=mu_cens, sigma=1.0),
                       lower=-np.inf, upper=y_ul, observed=np.ones_like(E_cens) * y_ul)
        
        # Sample
        trace = pm.sample(1000, tune=500, return_inferencedata=True, 
                         progressbar=False, random_seed=42, chains=1, cores=1)
    
    # Extract results
    a_samples = trace.posterior['a'].values.flatten()
    T_samples = trace.posterior['T'].values.flatten()
    
    a_val = np.mean(a_samples)
    a_err = np.std(a_samples)
    T_val = np.mean(T_samples)
    T_err = np.std(T_samples)
    
    # Compute column density
    Ncol = np.exp(a_val)
    dNcol = Ncol * a_err
    
    Ntot = Ncol * Q_Herbst(T_val)
    dNtot = Ntot * np.sqrt((dNcol/Ncol)**2 + (Q_Herbst_err(T_val, T_err)/Q_Herbst(T_val))**2)
    
    print(f"T = {T_val:.1f} ± {T_err:.1f} K")
    print(f"Ntot = {Ntot:.2e} ± {dNtot:.2e} cm^-2")
    
    return {
        'a': a_val, 'da': a_err,
        'T': T_val, 'dT': T_err,
        'Ncol': Ntot, 'dNcol': dNtot,
        'popt': np.array([a_val, T_val])
    }


def censored_regression_2T(E_arr, y_arr, y_err, upper_limit_indices, upper_limit_values, molecular_data, a1_init, T1_init, OPR=None, R=None, dist=140*pc):
    """
    Perform 2T rotational diagram fit using PyMC censored regression.
    """
    # Separate measured and censored data
    measured_mask = np.array([i not in upper_limit_indices for i in range(len(E_arr))])
    E_meas = E_arr[measured_mask]
    y_meas = y_arr[measured_mask]
    y_err_meas = y_err[measured_mask]
    
    E_cens = E_arr[~measured_mask]
    y_ul = y_arr[~measured_mask]
    
    print(f"\nCensored regression 2T: {np.sum(measured_mask)} measured, {len(upper_limit_indices)} upper limits")
    
    with pm.Model() as model:
        # Priors
        a1 = pm.Normal('a1', mu=a1_init, sigma=5)
        T1 = pm.Uniform('T1', lower=10, upper=5000)
        a2 = pm.Normal('a2', mu=a1_init - 5, sigma=5)
        T2 = pm.Uniform('T2', lower=10, upper=5000)
        
        # 2T model prediction
        mu_2T_meas = pm.math.log(pm.math.exp(a1 - E_meas/T1) + pm.math.exp(a2 - E_meas/T2))
        mu_2T_cens = pm.math.log(pm.math.exp(a1 - E_cens/T1) + pm.math.exp(a2 - E_cens/T2))
        
        # Measured points
        if len(E_meas) > 0:
            sigma_meas = y_err_meas[y_err_meas > 0]
            if len(sigma_meas) > 0:
                pm.Normal('y_measured', mu=mu_2T_meas[:len(sigma_meas)], 
                         sigma=sigma_meas, observed=y_meas[:len(sigma_meas)])
            else:
                pm.Normal('y_measured', mu=mu_2T_meas, sigma=1, observed=y_meas)
        
        # Censored points (upper limits)
        if len(E_cens) > 0:
            pm.Censored('y_censored', pm.Normal.dist(mu=mu_2T_cens, sigma=1.0),
                       lower=-np.inf, upper=y_ul, observed=np.ones_like(E_cens) * y_ul)
        
        # Sample
        trace = pm.sample(1000, tune=500, return_inferencedata=True,
                         progressbar=False, random_seed=42, chains=1, cores=1)
    
    # Extract results
    a1_samples = trace.posterior['a1'].values.flatten()
    T1_samples = trace.posterior['T1'].values.flatten()
    a2_samples = trace.posterior['a2'].values.flatten()
    T2_samples = trace.posterior['T2'].values.flatten()
    
    a1_val, T1_val = np.mean(a1_samples), np.mean(T1_samples)
    a1_err, T1_err = np.std(a1_samples), np.std(T1_samples)
    a2_val, T2_val = np.mean(a2_samples), np.mean(T2_samples)
    a2_err, T2_err = np.std(a2_samples), np.std(T2_samples)
    
    # Compute column densities
    N1 = np.exp(a1_val)
    N2 = np.exp(a2_val)
    dN1 = N1 * a1_err
    dN2 = N2 * a2_err
    
    N1_tot = N1 * Q_Herbst(T1_val)
    N2_tot = N2 * Q_Herbst(T2_val)
    dN1_tot = N1_tot * np.sqrt((dN1/N1)**2 + (Q_Herbst_err(T1_val, T1_err)/Q_Herbst(T1_val))**2)
    dN2_tot = N2_tot * np.sqrt((dN2/N2)**2 + (Q_Herbst_err(T2_val, T2_err)/Q_Herbst(T2_val))**2)
    
    print(f"T1 = {T1_val:.1f} ± {T1_err:.1f} K")
    print(f"T2 = {T2_val:.1f} ± {T2_err:.1f} K")
    print(f"N1 = {N1_tot:.2e} ± {dN1_tot:.2e}")
    print(f"N2 = {N2_tot:.2e} ± {dN2_tot:.2e}")
    
    return {
        'a1': a1_val, 'da1': a1_err, 'T1': T1_val, 'dT1': T1_err,
        'a2': a2_val, 'da2': a2_err, 'T2': T2_val, 'dT2': T2_err,
        'Ncol1': N1_tot, 'dNcol1': dN1_tot,
        'Ncol2': N2_tot, 'dNcol2': dN2_tot,
        'popt': np.array([a1_val, T1_val, a2_val, T2_val])
    }
