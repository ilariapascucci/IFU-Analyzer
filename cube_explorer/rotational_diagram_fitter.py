"""
Rotational diagram fitting functions for H2 analysis
Implements single-temperature and two-temperature rotational diagram fits
"""

import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit


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
        arcsec2 = (2*R/dist)**2
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


def rotational_diagram(fluxes, flux_errors, molecular_data=None, OPR=None, fit2T=False, R=None, dist=140*pc):
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
        dF = flux_errors[J]

        # Get molecular data for this J level
        if J not in wave or J not in Aul or J not in Eu or J not in gu:
            print(f"Warning: Missing data for J={J}, skipping")
            continue

        Nu = compute_Nu(F, wave[J], Aul[J], R, dist)
        dNu = Nu * (dF / F)

        y = np.log(Nu / gu[J])
        dy = dNu / Nu

        E_arr.append(Eu[J])
        y_arr.append(y)
        y_err.append(dy)

    E_arr = np.array(E_arr)
    y_arr = np.array(y_arr)
    y_err = np.array(y_err)

    if len(E_arr) < 2:
        raise ValueError("Need at least 2 data points for fitting")

    # ============ 1T fit ============
    p0 = [np.max(y_arr), 500]

    popt1, pcov1 = curve_fit(
        model_1T, E_arr, y_arr,
        sigma=y_err, absolute_sigma=True, p0=p0
    )

    a1, T1 = popt1
    da1, dT1 = np.sqrt(np.diag(pcov1))

    Ncol1 = np.exp(a1)
    dNcol1 = Ncol1 * da1

    Ntot1 = Ncol1 * Q_Herbst(T1)
    dNtot1 = Ntot1 * np.sqrt((dNcol1/Ncol1)**2 + (Q_Herbst_err(T1, dT1)/Q_Herbst(T1))**2)

    print("\n--- 1T FIT ---")
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
        p0 = [a1, T1, a1-5, T1*2]

        try:
            popt2, pcov2 = curve_fit(
                model_2T, E_arr, y_arr,
                sigma=y_err, absolute_sigma=True,
                p0=p0, maxfev=10000
            )

            a1_2, T1_2, a2_2, T2_2 = popt2
            errs = np.sqrt(np.diag(pcov2))

            da1_2, dT1_2, da2_2, dT2_2 = errs

            N1 = np.exp(a1_2)
            N2 = np.exp(a2_2)

            N1_tot = N1 * Q_Herbst(T1_2)
            N2_tot = N2 * Q_Herbst(T2_2)

            dN1 = N1 * da1_2
            dN2 = N2 * da2_2

            dN1_tot = N1_tot * np.sqrt((dN1/N1)**2 + (Q_Herbst_err(T1_2, dT1_2)/Q_Herbst(T1_2))**2)
            dN2_tot = N2_tot * np.sqrt((dN2/N2)**2 + (Q_Herbst_err(T2_2, dT2_2)/Q_Herbst(T2_2))**2)

            print("\n--- 2T FIT ---")
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
        except Exception as e:
            print(f"2T fit failed: {e}")

    return results
