"""
Rotational diagram fitting for non-H2 molecules (e.g. CO). No OPR here --
that is H2-only physics (see rotational_diagram_h2_fitter.py). Q(T) comes
from HITRAN's TIPS data, so this works for any get_available_molecules() entry.
"""

from hapi import partitionSum

from .line_catalog import get_available_molecules
# The shared fit engine lives next to the models and PyMC code it calls.
# This module only supplies Q(T) for any HITRAN molecule.
from .rotational_diagram_h2_fitter import (
    pc, T_CAP, require_fit_selected, build_excitation_dataset, fit_excitation_diagram,
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
    require_fit_selected(fit1T, fit2T)

    global wave, Aul, Eu, gu

    if molecular_data is not None:
        wave = molecular_data.get('wavelengths', {})
        Aul = molecular_data.get('Aul', {})
        Eu = molecular_data.get('Eu', {})
        gu = molecular_data.get('gu', {})

    # Build dataset (no OPR to correct for here)
    E_arr, y_arr, y_err, is_ul_mask, dropped_detections, wave_arr = build_excitation_dataset(
        fluxes, flux_errors, wave, Aul, Eu, gu, upper_limits, R, dist,
    )

    return fit_excitation_diagram(
        E_arr, y_arr, y_err, is_ul_mask, dropped_detections, wave_arr, R, fit1T, fit2T,
        lambda T: partition_function(T, molecule, isotopologue),
        lambda T, dT: partition_function_err(T, dT, molecule, isotopologue),
        progress_callback=progress_callback, t_cap=T_CAP,
    )
