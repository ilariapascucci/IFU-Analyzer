"""
Rotational diagram fitting for non-H2 molecules (e.g. CO).

No OPR (ortho-para ratio) here -- that's H2-only nuclear-spin-statistics physics,
see rotational_diagram_h2_fitter.py. Q(T) comes from HITRAN's own TIPS-2021 data
(via HAPI's partitionSum()) instead of a hardcoded per-molecule formula, so this
one function works for any molecule get_available_molecules() knows about.

Reuses the molecule-agnostic engine (upper_limit_1T/2T, model_1T/2T, compute_Nu,
gaussian_flux_upper_limit, compute_line_fluxes_collective) straight from
rotational_diagram_h2_fitter rather than duplicating it -- that logic doesn't
depend on OPR or on which Q(T) is used, so there's only one copy of it to keep
correct.
"""

import numpy as np
from scipy.optimize import curve_fit
from hapi import partitionSum

from .line_catalog import get_available_molecules
from .rotational_diagram_h2_fitter import (
    pc, model_1T, model_2T, compute_Nu, upper_limit_1T, upper_limit_2T,
    gaussian_flux_upper_limit, compute_line_fluxes_collective,
)

# Molecular data for the line set being fit (populated per-call from molecular_data)
wave = {}  # micron
Aul = {}  # s^-1
Eu = {}  # K
gu = {}  # unitless


def partition_function(T, molecule='CO', isotopologue=1):
    """
    Total internal partition function Q(T), from HITRAN's own TIPS-2021 data via
    HAPI's partitionSum() -- the same q<global_id>.txt tables published at
    hitran.org/data/Q/.

    Args:
        T (float): Temperature in K
        molecule (str): Species name, per get_available_molecules()
        isotopologue (int): HITRAN isotopologue number

    Returns:
        Q (float): Partition function value at T

    Raises:
        ValueError: if molecule isn't in get_available_molecules()
    """
    M = get_available_molecules().get(molecule)
    if M is None:
        raise ValueError(f"Unknown molecule '{molecule}' -- not in get_available_molecules().")
    return partitionSum(M, isotopologue, T)


def partition_function_err(T, T_err, molecule='CO', isotopologue=1):
    """
    Propagated error on partition_function() from a temperature uncertainty, via a
    central finite difference (HAPI's partitionSum() has no analytic derivative).

    Args:
        T (float): Temperature in K
        T_err (float): Uncertainty on T
        molecule (str): Species name, per get_available_molecules()
        isotopologue (int): HITRAN isotopologue number

    Returns:
        dQ (float): Propagated uncertainty on Q(T)
    """
    eps = max(0.1, abs(T) * 1e-5)
    dQdT = (partition_function(T + eps, molecule, isotopologue)
            - partition_function(T - eps, molecule, isotopologue)) / (2 * eps)
    return abs(dQdT) * T_err


def rotational_diagram(fluxes, flux_errors, molecular_data=None, fit1T=True, fit2T=False,
                        R=None, dist=140*pc, upper_limits=None, progress_callback=None,
                        molecule='CO', isotopologue=1):
    """
    Fit a rotational diagram (1T and/or 2T) to a set of line fluxes, for any
    non-H2 molecule. No OPR here -- see rotational_diagram_h2_fitter for H2.

    Args:
        fluxes (dict): {J: flux_value}, J is the upper rotational level
        flux_errors (dict): {J: flux_error}
        molecular_data (dict): 'wavelengths', 'Aul', 'Eu', 'gu' keys, each {J: value}
        fit1T (bool): Report the 1T fit (always computed regardless, as 2T's own seed)
        fit2T (bool): Perform the 2T fit
        R (float): Emitting radius (AU or cm)
        dist (float): Distance (pc or cm)
        upper_limits (dict): {J: True} for measurements marked as upper limits
        progress_callback (callable): (phase, current, total) -> None, for a GUI progress bar
        molecule (str): Species being fit, per get_available_molecules()
        isotopologue (int): HITRAN isotopologue number for `molecule`

    Returns:
        results (dict): Fit results ("1T" and/or "2T" keys, plus "_plot_data")

    Raises:
        ValueError: if neither fit1T nor fit2T is True
    """
    if not fit1T and not fit2T:
        raise ValueError("At least one of fit1T/fit2T must be True -- nothing to fit.")

    # Upper temperature bound for curve_fit -- arbitrary, but CO plausibly reaches
    # hotter rotational temperatures than H2 typically shows, so it gets more headroom.
    t_cap = 4000

    global wave, Aul, Eu, gu

    if molecular_data is not None:
        wave = molecular_data.get('wavelengths', {})
        Aul = molecular_data.get('Aul', {})
        Eu = molecular_data.get('Eu', {})
        gu = molecular_data.get('gu', {})

    # ==========================================================
    # Build dataset (no OPR to correct for here)
    # ==========================================================
    def _build_dataset():
        """
        Compute excitation diagram arrays.

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
            Nu = compute_Nu(F, wave[J], Aul[J], R, dist)
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

            # Same dF/F propagation for detections and upper limits -- see
            # rotational_diagram_h2_fitter's own _build_dataset for the full reasoning.
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
    # Bounded curve_fit helpers
    # ==========================================================
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
            model_1T, E, y, sigma=sigma, absolute_sigma=True, p0=p0, bounds=bounds
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
            model_2T, E, y, sigma=sigma, absolute_sigma=True, p0=p0, bounds=bounds, maxfev=10000
        )
        return popt, pcov

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
        Q = partition_function(T, molecule, isotopologue)
        Ntot = Ncol * Q
        dNtot = Ntot * np.sqrt(
            (dNcol / Ncol) ** 2
            + (partition_function_err(T, dT, molecule, isotopologue) / Q) ** 2)
        return Ntot, dNtot

    # ==========================================================
    # Build dataset (full data, including upper limits)
    # ==========================================================
    E_arr, y_arr, y_err, is_ul_mask, dropped_detections = _build_dataset()

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
    # 1T FIT (final) -- see rotational_diagram_h2_fitter for why this always
    # runs regardless of fit1T (2T's own fit needs it as a seed either way)
    # ==========================================================
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

    results = {}
    if dropped_detections:
        results["dropped_J"] = dropped_detections

    if fit1T:
        Ntot1, dNtot1 = _derive_column(a1, da1, T1, dT1)

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
            print("\n--- 2T FIT (PyMC, upper-limit barrier) ---")
            result = upper_limit_2T(
                E_arr, y_arr, y_err, is_ul_mask, a1, T1,
                progress_callback=(lambda cur, tot: progress_callback("2T", cur, tot)) if progress_callback else None,
                t_cap=t_cap,
            )
            a1_2, T1_2, a2_2, T2_2 = result["popt"]
            errs = result["errs"]
            pymc_diag_2T = {"n_divergences": result["n_divergences"], "max_r_hat": result["max_r_hat"]}

        da1_2, dT1_2, da2_2, dT2_2 = errs

        N1_tot, dN1_tot = _derive_column(a1_2, da1_2, T1_2, dT1_2)
        N2_tot, dN2_tot = _derive_column(a2_2, da2_2, T2_2, dT2_2)

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

    return results
