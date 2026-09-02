"""
HITRAN/Local Spectral Line Data Extractor
Extracts molecular spectroscopy data from HITRAN database or local line lists
"""

import re
import os
import numpy as np
import pandas as pd
import time
from astropy.units import Quantity
from astropy.io import ascii

# -------------------- Helper Functions --------------------
def _extract_wavelength_value(wavelength):
    """
    Extract numeric value from wavelength Quantity or return raw value.
    Allows division operations to work with both Quantity and raw arrays.
    """
    if isinstance(wavelength, Quantity):
        return wavelength.value
    return wavelength


_ISO_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_ISO_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def format_isotopologue_name(iso_name):
    """
    Convert a HITRAN/HAPI isotopologue formula (e.g. '(13C)(16O)') into a
    readable isotope-labeled formula with mass numbers as superscripts and
    atom counts as subscripts (e.g. '¹³C¹⁶O').
    """
    if not iso_name:
        return ""
    s = re.sub(
        r'\((\d+)([A-Za-z]+)\)',
        lambda m: m.group(1).translate(_ISO_SUPERSCRIPT) + m.group(2),
        iso_name,
    )
    s = re.sub(r'(?<=[A-Za-z])(\d+)', lambda m: m.group(1).translate(_ISO_SUBSCRIPT), s)
    return s


# -------------------- Species Normalization --------------------
def normalize_species_input(user_input):
    """
    Converts various species input formats into standardized bracketed notation.
    Handles space-separated (Fe II), concatenated (FeII), numeric (Fe2), and 
    already-bracketed formats.
    """
    if not user_input:
        return user_input
    
    original = user_input.strip()
    
    if original.startswith('[') and original.endswith(']'):
        return original
    
    species = original.upper()
    species = species.replace('[', '').replace(']', '').strip()
    
    molecules = ['H2', 'CO', 'CO2', 'H2O', 'CH4', 'NH3', 'O2', 'N2', 'NO', 'NO2', 
                 'SO2', 'O3', 'H2S', 'HCN', 'C2H2', 'C2H6', 'PH3', 'H2CO', 'N2O',
                 'HCL', 'HF', 'HBR', 'HI', 'OH', 'CLO', 'OCS', 'HOCL']
    
    if species in molecules:
        return original
    
    ionization_map = {
        '1': 'I', '2': 'II', '3': 'III', '4': 'IV', '5': 'V',
        '6': 'VI', '7': 'VII', '8': 'VIII', '9': 'IX', '10': 'X',
        'I': 'I', 'II': 'II', 'III': 'III', 'IV': 'IV', 'V': 'V',
        'VI': 'VI', 'VII': 'VII', 'VIII': 'VIII', 'IX': 'IX', 'X': 'X'
    }
    
    parts = species.split()
    if len(parts) == 2:
        element, ion_state = parts[0], parts[1]
        if ion_state in ionization_map or ion_state.isdigit():
            if ion_state.isdigit():
                ion_state = ionization_map.get(ion_state, ion_state)
            else:
                ion_state = ionization_map.get(ion_state, ion_state)
            return f"[{element.capitalize()} {ion_state}]"
    
    match = re.match(r'^([A-Z][A-Z])([IVX]+)$', species)
    if match:
        element, ion_state = match.groups()
        second_char = element[1]
        if (second_char not in ['I', 'V', 'X'] or len(ion_state) >= 2) and ion_state in ionization_map:
            return f"[{element.capitalize()} {ion_state}]"
    
    match = re.match(r'^([A-Z])([IVX]{2,})$', species)
    if match:
        element, ion_state = match.groups()
        if ion_state in ionization_map:
            return f"[{element} {ion_state}]"
    
    match = re.match(r'^([A-Z][A-Z]?)(\d+)$', species)
    if match:
        element, ion_num = match.groups()
        if not (len(element) == 1 and element + ion_num in molecules):
            ion_state = ionization_map.get(ion_num, ion_num)
            return f"[{element.capitalize()} {ion_state}]"
    
    return original

# -------------------- Local Line Filter --------------------
def filter_spectral_lines(file_path, wavelength_min, wavelength_max, species=None):
    """
    Parse a local spectral line list and return entries within wavelength range.
    If species is None or empty, return ALL lines in range.
    Otherwise perform case-insensitive exact or bracketed element matching.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return [], {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], {}
    
    data_rows = []
    all_species = {}
    
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                wavelength = float(parts[1])
                if wavelength_min <= wavelength <= wavelength_max:
                    bracketed_match = re.search(r'\[[\w\s]+\]', line)
                    if bracketed_match:
                        found_species = bracketed_match.group()
                    else:
                        if len(parts) > 3 and (parts[3].upper() in ['I','II','III','IV','V','VI','VII','VIII','IX','X']):
                            found_species = parts[2] + ' ' + parts[3]
                        else:
                            found_species = parts[2].strip()
                    
                    all_species[found_species] = all_species.get(found_species, 0) + 1

                    # If no species filter requested include everything
                    if species is None or str(species).strip() == "":
                        priority = 0
                        include = True
                    else:
                        species_upper = species.upper()
                        found_upper = found_species.upper()
                        exact_match = (found_upper == species_upper)
                        element_match = species_upper in found_upper and '[' in found_species
                        priority = 0 if exact_match else 1 if element_match else 2
                        include = (priority <= 1)

                    if include:
                        wavelength_val = _extract_wavelength_value(wavelength)
                        wavenumber = 1e4 / wavelength_val
                        model = parts[0]
                        transition = ""
                        a_coeff = 0.0
                        e_upper = 0.0
                        
                        try:
                            fields = line.split()
                            if len(fields) >= 6:
                                transition_parts = []
                                coeff_idx = -1
                                for i in range(3, len(fields)):
                                    if 'E' in fields[i] and any(c.isdigit() for c in fields[i]):
                                        try:
                                            a_coeff = float(fields[i])
                                            coeff_idx = i
                                            break
                                        except Exception:
                                            transition_parts.append(fields[i])
                                    else:
                                        transition_parts.append(fields[i])
                                transition = " ".join(transition_parts) if transition_parts else "Unknown"
                                if coeff_idx >= 0 and coeff_idx + 1 < len(fields):
                                    try:
                                        e_upper = float(fields[coeff_idx + 1])
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        data_rows.append({
                            'model': model,
                            'wavelength': wavelength,
                            'wavenumber': wavenumber,
                            'species': found_species,
                            'transition': transition,
                            'a_coeff': a_coeff,
                            'e_upper': e_upper,
                            'priority': priority,
                        })
            except (ValueError, IndexError):
                continue
    
    # Sort: if priority present use it, else all zeros
    data_rows = sorted(data_rows, key=lambda x: (x.get('priority', 0), x['wavelength']))
    for row in data_rows:
        if 'priority' in row:
            del row['priority']
    return data_rows, all_species

# -------------------- HITRAN Utilities --------------------
def get_available_molecules():
    return {
        'H2O': 1, 'CO2': 2, 'O3': 3, 'N2O': 4, 'CO': 5, 'CH4': 6, 'O2': 7, 'NO': 8,
        'SO2': 9, 'NO2': 10, 'NH3': 11, 'HNO3': 12, 'OH': 13, 'HF': 14, 'HCl': 15,
        'HBr': 16, 'HI': 17, 'ClO': 18, 'OCS': 19, 'H2CO': 20, 'HOCl': 21, 'N2': 22,
        'HCN': 23, 'CH3Cl': 24, 'H2O2': 25, 'C2H2': 26, 'C2H6': 27, 'PH3': 28,
        'COF2': 29, 'SF6': 30, 'H2S': 31, 'HCOOH': 32, 'HO2': 33, 'O': 34,
        'ClONO2': 35, 'NOp': 36, 'HOBr': 37, 'C2H4': 38, 'CH3OH': 39, 'CH3Br': 40,
        'CH3CN': 41, 'CF4': 42, 'C4H2': 43, 'HC3N': 44, 'H2': 45, 'CS': 46,
        'SO3': 47, 'C2N2': 48, 'COCl2': 49, 'SO': 50, 'CH3F': 51, 'GeH4': 52,
        'CS2': 53, 'CH3I': 54, 'NF3': 55,
    }

def convert_to_numpy(data):
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return np.array(data, dtype=float)
    if isinstance(data, (list, tuple)):
        try:
            return np.array(data, dtype=float)
        except Exception:
            return np.array(data)
    return np.array([data], dtype=float)

def delete_cache_properly(table_name):
    """Delete specific table cache files without removing the hitran_local directory"""
    try:
        from hapi import purgeCache
        purgeCache(table_name)
        print(f"Purged cache for {table_name}")
    except Exception as e:
        # Fallback: manually delete just the table files
        try:
            data_file = f'hitran_local/{table_name}.data'
            header_file = f'hitran_local/{table_name}.header'
            if os.path.exists(data_file):
                os.remove(data_file)
                print(f"Removed {data_file}")
            if os.path.exists(header_file):
                os.remove(header_file)
                print(f"Removed {header_file}")
        except Exception as e2:
            print(f"Could not delete cache files: {e2}")

def get_cache_age(table_name):
    try:
        data_file = f'hitran_local/{table_name}.data'
        if os.path.exists(data_file):
            mod_time = os.path.getmtime(data_file)
            age_seconds = time.time() - mod_time
            return age_seconds / (24 * 3600)
    except Exception:
        return None
    return None

def compute_hitran_table_name(wavelength_min, wavelength_max, molecule, isotopologue=1):
    """
    Compute the hitran_local/ cache key fetch_hitran_data() would use for this
    query, without importing hapi or touching the network.

    Args:
        wavelength_min, wavelength_max (float): Wavelength range in um
        molecule (str): Molecule name, per get_available_molecules()
        isotopologue (int): HITRAN isotopologue number

    Returns:
        table_name (str): Cache key, or None if molecule isn't recognized
    """
    M = get_available_molecules().get(molecule)
    if M is None:
        return None
    wavelength_min_val = _extract_wavelength_value(wavelength_min)
    wavelength_max_val = _extract_wavelength_value(wavelength_max)
    wavenum_min = 1e4 / wavelength_max_val
    wavenum_max = 1e4 / wavelength_min_val
    return f'{molecule}_{M}_{isotopologue}_{int(wavenum_min)}_{int(wavenum_max)}'


def find_containing_cached_table(wavelength_min, wavelength_max, molecule, isotopologue=1):
    """
    Find an already-cached table for this (molecule, isotopologue) whose
    wavenumber range fully contains the requested range, even when its own
    bounds don't match exactly -- lets a narrower query reuse a wider table
    that was already fetched instead of attempting a new exact-range download.

    Args:
        wavelength_min, wavelength_max (float): Wavelength range in um
        molecule (str): Molecule name, per get_available_molecules()
        isotopologue (int): HITRAN isotopologue number

    Returns:
        table_name (str): Name of the smallest containing cached table, or
            None if no cached table covers this range
    """
    M = get_available_molecules().get(molecule)
    if M is None or not os.path.isdir('hitran_local'):
        return None
    wavelength_min_val = _extract_wavelength_value(wavelength_min)
    wavelength_max_val = _extract_wavelength_value(wavelength_max)
    wavenum_min = 1e4 / wavelength_max_val
    wavenum_max = 1e4 / wavelength_min_val

    prefix = f'{molecule}_{M}_{isotopologue}_'
    best_name, best_span = None, None
    for fname in os.listdir('hitran_local'):
        if not (fname.startswith(prefix) and fname.endswith('.data')):
            continue
        try:
            lo_str, hi_str = fname[len(prefix):-len('.data')].split('_')
            lo, hi = int(lo_str), int(hi_str)
        except ValueError:
            continue
        if lo <= wavenum_min and hi >= wavenum_max:
            span = hi - lo
            if best_span is None or span < best_span:
                best_name, best_span = fname[:-len('.data')], span
    return best_name


def is_hitran_data_cached(wavelength_min, wavelength_max, molecule, isotopologue=1):
    """
    Check whether fetch_hitran_data() can already serve this query from
    hitran_local/ (an exact-range match or a wider containing table), without
    touching the network -- lets a caller offer "use the local copy" before
    defaulting to an online refresh.

    Args:
        wavelength_min, wavelength_max (float): Wavelength range in um
        molecule (str): Molecule name, per get_available_molecules()
        isotopologue (int): HITRAN isotopologue number

    Returns:
        cached (bool): True if a local .data file already covers this query
    """
    table_name = compute_hitran_table_name(wavelength_min, wavelength_max, molecule, isotopologue)
    if table_name is not None and os.path.exists(os.path.join('hitran_local', f'{table_name}.data')):
        return True
    return find_containing_cached_table(wavelength_min, wavelength_max, molecule, isotopologue) is not None


# Molecules ("common" as in: worth keeping pre-cached in hitran_local/ for offline
# use -- meetings/demos without internet, see warm_hitran_cache.py) -- not all 55
# HITRAN has (get_available_molecules()), just these four. But for each one, this
# is EVERY isotopologue HITRAN has for it (get_available_isotopologues() -- e.g.
# CO -> [1,2,3,4,5,6]), not a curated subset, so that fetch_hitran_data(...,
# isotopologue=None) ("search every isotopologue" -- see its docstring) can be
# served offline too for these molecules, not just isotopologue=1.
#
# Hardcoded rather than built by calling get_available_isotopologues() here, since
# that needs hapi imported, and every other hapi import in this module is lazy/
# inside a function -- importing it at module level would print HAPI's banner on
# every `import cube_explorer.line_catalog`, hapi installed or not needed.
#
# v_upper/v_lower aren't part of the cache key (HITRAN is cached by (molecule,
# isotopologue, wavelength range) only, then filtered by transition after loading),
# so they're documentation here, not a separate cache dimension. Full practical IR
# range (1-30um) rather than a narrow band per demo -- cheap enough in total that
# there's no per-demo range/isotopologue to guess right any more.
_ALL_ISOTOPOLOGUES_BY_MOLECULE = {
    "H2": [1, 2],
    "CO": [1, 2, 3, 4, 5, 6],
    # Far denser than the others (~108k lines for iso1 alone, ~17MB/~15s to fetch) --
    # don't assume every future molecule added here will stay this cheap; check
    # size/time before widening one that far.
    "H2O": [1, 2, 3, 4, 5, 6, 7],
    # OH iso2 (18O-H) and iso3 (16O-D) exist in HITRAN's isotopologue table but have
    # no actual line data above ~331 cm-1 (<=30.2um, i.e. entirely outside this app's
    # 1-30um range) -- confirmed against HITRAN directly, not a fetch bug. Querying
    # them here would just make every future warm-up "fail" for lines that were
    # never going to exist, so only the main isotopologue is worth pre-caching.
    "OH": [1],
}
COMMON_HITRAN_QUERIES = [
    {"name": f"{molecule} iso{isotopologue} 1-30um (all bands)", "molecule": molecule,
     "isotopologue": isotopologue, "wavelength_min": 1.0, "wavelength_max": 30.0,
     "v_upper": None, "v_lower": None}
    for molecule, isotopologues in _ALL_ISOTOPOLOGUES_BY_MOLECULE.items()
    for isotopologue in isotopologues
]


def get_available_isotopologues(molecule):
    """
    List the isotopologue numbers HITRAN has data for, for a given molecule
    (e.g. CO -> [1, 2, 3, 4, 5, 6]). Used to support isotopologue=None ("any
    isotopologue") in fetch_hitran_data(). Returns [] if HAPI isn't installed
    or the molecule isn't recognized.
    """
    try:
        from hapi import ISO
    except ImportError:
        return []
    M = get_available_molecules().get(molecule)
    if M is None:
        return []
    return sorted(i for (m, i) in ISO.keys() if m == M)


def fetch_hitran_data(wavelength_min, wavelength_max, molecule, isotopologue=1, refresh_cache=False, cache_days=30, v_upper=None, v_lower=None):
    """
    Retrieves molecular spectroscopic line data from the HITRAN database using HAPI.
    Implements smart caching to minimize downloads, converts wavelength ranges to
    wavenumbers, extracts quantum numbers for transition labeling, and calculates
    upper energy levels. Handles both cached and fresh data with user prompts for
    cache refresh based on age.

    Args:
        wavelength_min, wavelength_max: Wavelength range in µm
        molecule: Molecule name (e.g., 'CO')
        isotopologue: Isotopologue number (default 1), or None to fetch every
            isotopologue HITRAN has for this molecule in range (e.g. for a user
            who doesn't know/care which isotopologue a line belongs to) -- each
            result row is still individually tagged with its own 'isotopologue'
            / 'isotopologue_name', so they stay distinguishable afterward.
        refresh_cache: Force cache refresh
        cache_days: Cache age threshold
        v_upper: Filter by upper vibrational level (optional, e.g., 1 for v=1)
        v_lower: Filter by lower vibrational level (optional, e.g., 0 for v=0)
    """
    if isotopologue is None:
        iso_list = get_available_isotopologues(molecule)
        if not iso_list:
            print(f"Molecule '{molecule}' not found in HITRAN")
            return [], {}
        combined_rows = []
        errors = []
        for iso_num in iso_list:
            rows, meta = _fetch_hitran_data_single_iso(
                wavelength_min, wavelength_max, molecule, iso_num,
                refresh_cache, cache_days, v_upper, v_lower,
            )
            combined_rows.extend(rows)
            if meta.get('error'):
                errors.append(meta.get('message', 'unknown error'))
        combined_rows.sort(key=lambda r: r['wavelength'])
        # Only surface as a fetch failure if EVERY isotopologue failed -- some
        # succeeding means real (if incomplete) data came back.
        if not combined_rows and errors:
            return combined_rows, {'error': True, 'message': '; '.join(errors)}
        return combined_rows, {}

    return _fetch_hitran_data_single_iso(
        wavelength_min, wavelength_max, molecule, isotopologue,
        refresh_cache, cache_days, v_upper, v_lower,
    )


def _fetch_hitran_data_single_iso(wavelength_min, wavelength_max, molecule, isotopologue, refresh_cache=False, cache_days=30, v_upper=None, v_lower=None):
    try:
        from hapi import db_begin, fetch, getColumns, tableList, ISO
    except ImportError:
        print("Error: HAPI not installed!")
        print("Install with: pip install hitran-api")
        return [], {'error': True, 'message': 'HAPI (hitran-api) is not installed'}

    mol_numbers = get_available_molecules()
    if molecule not in mol_numbers:
        print(f"Molecule '{molecule}' not found in HITRAN")
        return [], {}

    M = mol_numbers[molecule]
    I = isotopologue
    # Isotopologue formula (e.g. '(13C)(16O)') for display in the GUI table/tooltips
    iso_entry = ISO.get((M, I))
    iso_name_pretty = format_isotopologue_name(iso_entry[1]) if iso_entry else None
    # Convert wavelength (μm) to wavenumber (cm⁻¹)
    wavelength_min_val = _extract_wavelength_value(wavelength_min)
    wavelength_max_val = _extract_wavelength_value(wavelength_max)
    wavenum_min = 1e4 / wavelength_max_val
    wavenum_max = 1e4 / wavelength_min_val
    table_name = compute_hitran_table_name(wavelength_min, wavelength_max, molecule, I)

    # A wider already-cached table covering this range serves it just as well as
    # an exact-range one -- reuse it instead of attempting a new (possibly
    # offline) download for the narrower range.
    if not refresh_cache:
        containing = find_containing_cached_table(wavelength_min, wavelength_max, molecule, I)
        if containing is not None:
            table_name = containing

    try:
        # Ensure hitran_local directory exists
        os.makedirs('hitran_local', exist_ok=True)
        db_begin('hitran_local')

        # Check for existing cached data
        try:
            existing_tables = tableList()
            data_cached = table_name in existing_tables
        except Exception:
            data_cached = False

        # Non-interactive cache policy: if cached and older than threshold, only refresh when requested
        if data_cached and not refresh_cache:
            cache_age = get_cache_age(table_name)
            if cache_age is not None:
                print(f"[CACHE] Age: {cache_age:.1f} days")
                if cache_age > cache_days:
                    print(f"[INFO] Cache is older than {cache_days} days (using cached unless refresh_cache=True)")

        if refresh_cache and data_cached:
            print(f"[INFO] Deleting old cache...")
            delete_cache_properly(table_name)
            data_cached = False

        if data_cached and not refresh_cache:
            print(f"[OK] Using cached {molecule} data")
        else:
            print(f"[INFO] Downloading {molecule} from HITRAN...")
            fetch(table_name, M, I, wavenum_min, wavenum_max)
            print(f"[OK] Download complete - cached locally")

        print(f"[INFO] Processing data...\n")
        data_rows = []

        # Attempt to retrieve quantum numbers for transition labeling
        try:
            nu, sw, A, gamma_air, gamma_self, elower, n_air, delta_air, Gp, Gpp, \
            global_upper_quanta, global_lower_quanta, local_upper_quanta, local_lower_quanta = getColumns(
                table_name, 
                ['nu', 'sw', 'A', 'gamma_air', 'gamma_self', 'elower', 'n_air', 
                 'delta_air', 'Gp', 'Gpp', 'global_upper_quanta', 'global_lower_quanta',
                 'local_upper_quanta', 'local_lower_quanta']
            )
        except Exception:
            # Fallback without quantum numbers
            nu, sw, A, gamma_air, gamma_self, elower, n_air, delta_air, Gp, Gpp = getColumns(
                table_name, 
                ['nu', 'sw', 'A', 'gamma_air', 'gamma_self', 'elower', 'n_air', 
                 'delta_air', 'Gp', 'Gpp']
            )
            global_upper_quanta = None
            global_lower_quanta = None
            local_upper_quanta = None
            local_lower_quanta = None

        nu = convert_to_numpy(nu)
        A = convert_to_numpy(A)
        elower = convert_to_numpy(elower)
        wavelength = 1e4 / nu  # Convert wavenumber to wavelength (already dimensionless)
        # Calculate upper energy level: E_upper = (E_lower + ν) × 1.44 [K]
        eupper = (elower + nu) * 1.44
        species_name = molecule

        for i in range(len(nu)):
            wl = wavelength[i]
            if wavelength_min <= wl <= wavelength_max:
                # Parse transition notation from quantum numbers
                j_lower = None
                branch = None
                if global_upper_quanta is not None and local_lower_quanta is not None:
                    try:
                        v_upper_str = str(global_upper_quanta[i]).strip()
                        v_lower_str = str(global_lower_quanta[i]).strip()
                        
                        # Extract numeric vibrational levels for filtering
                        try:
                            v_upper_num = int(re.search(r'\d+', v_upper_str).group())
                            v_lower_num = int(re.search(r'\d+', v_lower_str).group())
                        except Exception:
                            v_upper_num = None
                            v_lower_num = None
                        
                        # Apply vibrational level filter if specified
                        if v_upper is not None and v_upper_num != v_upper:
                            continue
                        if v_lower is not None and v_lower_num != v_lower:
                            continue

                        lower_str = str(local_lower_quanta[i]).strip()
                        j_match = re.search(r'(\d+)', lower_str)
                        branch_match = re.search(r'([A-Z])', lower_str)

                        if j_match and branch_match:
                            j_lower = int(j_match.group(1))
                            branch = branch_match.group(1)

                            # Apply selection rules for different branches
                            if branch == 'S':
                                j_upper = j_lower + 2
                            elif branch == 'R':
                                j_upper = j_lower + 1
                            elif branch == 'Q':
                                j_upper = j_lower
                            elif branch == 'P':
                                j_upper = j_lower - 1
                            elif branch == 'O':
                                j_upper = j_lower - 2
                            else:
                                j_upper = '?'

                            transition = f"v={v_upper_str}->{v_lower_str}, J={j_upper}->{j_lower}"
                        else:
                            transition = f"v={v_upper_str}->{v_lower_str}"
                    except Exception as ex:
                        transition = "N/A"
                else:
                    transition = "N/A"

                data_rows.append({
                    'model': 'HITRAN',
                    'wavelength': float(wl),
                    'wavenumber': float(nu[i]),
                    'species': species_name,
                    'transition': transition,
                    'a_coeff': float(A[i]),
                    'e_upper': float(eupper[i]),
                    'Gp': float(Gp[i]),
                    'j_lower': j_lower,
                    'branch': branch,
                    'isotopologue': I,
                    'isotopologue_name': iso_name_pretty,
                })

        print(f"[OK] Retrieved {len(data_rows)} lines\n")
        return data_rows, {}
    except Exception as e:
        # Distinguish this from a real "zero lines in range" result -- e.g. no
        # offline copy covers this range and the network fetch itself failed
        # (no internet) -- so callers don't report it as if HITRAN had nothing.
        print(f"[ERROR] {str(e)}")
        return [], {'error': True, 'message': str(e)}

def display_results(results):
    """
    Formats and prints spectral line data in a clean tabular layout. Displays
    wavelength, wavenumber, species, transition notation, Einstein A-coefficient,
    and upper energy level for each line, sorted by wavelength.
    """
    if not results:
        print("No entries found.")
        return
    
    print("="*115)
    print(f"{'MODEL':<10} | {'Wavelength [um]':<19} | {'Wavenumber [cm^-1]':<20} | {'L-species':<10} | {'Transition':<20} | {'A-coeff(1/sec)':<14} | {'E_upper [K]':<12}")
    print("="*115)
    
    # Sort by wavelength for organized output
    sorted_results = sorted(results, key=lambda x: x['wavelength'])
    for row in sorted_results:
        print(
            f"{row['model']:<10} | "
            f"{row['wavelength']:>19.9f} | "
            f"{row['wavenumber']:>20.4f} | "
            f"{row['species']:<10} | "
            f"{row['transition']:<20} | "
            f"{row['a_coeff']:>13.3e} | "
            f"{row['e_upper']:>11.2f}"
        )
    print("="*115)
    print(f"Total entries: {len(results)}\n")

def hitran_gui_interface(wavelength_min, wavelength_max, molecule, isotopologue=1, refresh_cache=False, v_upper=None, v_lower=None):
    """
    GUI-compatible function to fetch HITRAN data and return results for display.
    
    Args:
        wavelength_min, wavelength_max: Wavelength range in µm
        molecule: Molecule name
        isotopologue: Isotopologue number (default 1), or None for every
            isotopologue HITRAN has for this molecule
        refresh_cache: Force cache refresh
        v_upper: Optional upper vibrational level filter (None = include all)
        v_lower: Optional lower vibrational level filter (None = include all)

    Returns:
        results, meta (tuple): Line-data rows, and a dict with 'error'=True and
            a 'message' when the fetch itself failed (vs. a genuine zero-line
            result), so the GUI can tell "not in HITRAN" apart from "offline
            and no cache covers this range".
    """
    try:
        results, meta = fetch_hitran_data(
            wavelength_min, wavelength_max, molecule, isotopologue, refresh_cache,
            v_upper=v_upper, v_lower=v_lower
        )
        return results, meta
    except Exception as e:
        print(f"Error in HITRAN GUI interface: {e}")
        return [], {'error': True, 'message': str(e)}

# Example GUI integration point (to be connected to the main GUI application):
# Functions above are designed to be imported and used by the GUI; no CLI entry point here.

# -------------------- CO Catalog Loading --------------------
def load_CO_line_catalog(filepath):
    """
    Load CO rovibrational line catalog from ECSV format.
    
    Parameters
    ----------
    filepath : str
        Path to the ECSV file (e.g., 'data/CO/CO_comprehensive_linelist_with_g.ecsv')
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: trans_id, species, line_wave_um, a, transition, eupper(K), 
        highv, highj, lowv, lowj, gp, nu-err, nu-ref
    """
    try:
        if not os.path.exists(filepath):
            return None
        
        CO_lines = ascii.read(filepath)
        CO_lines = CO_lines.to_pandas()
        return CO_lines
    except Exception as e:
        return None

def load_contaminating_lines_catalog(filepath):
    """
    Load contaminating lines catalog from CSV format.

    'wavelength' and 'species' are both required (same as load_custom_line_catalog()
    -- every imported line needs to be identifiable, contaminant or not). Everything
    else is optional and carried through as-is if present.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (e.g., 'data/CO/ContaminatingLines.csv')

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: line_wave_um (renamed from 'wavelength'), species,
        and any other columns the CSV had. Rows missing wavelength or species are
        dropped individually (with a console note) rather than failing the whole
        file. None if the file is missing, empty, or every row is dropped.

    Raises
    ------
    ValueError
        If the CSV file does not contain a 'wavelength' or 'species' column.
    """
    try:
        if not os.path.exists(filepath):
            print(f"Error: File '{filepath}' not found.")
            return None

        print(f">>> Loading contaminating lines catalog from {filepath}...")
        contam_lines = pd.read_csv(filepath)

        # VALIDATE: 'wavelength' and 'species' columns MUST exist
        missing_cols = [c for c in ('wavelength', 'species') if c not in contam_lines.columns]
        if missing_cols:
            error_msg = (
                f"CSV file is missing required column(s): {', '.join(missing_cols)}.\n"
                f"Columns found: {', '.join(contam_lines.columns.tolist())}"
            )
            print(error_msg)
            raise ValueError(error_msg)

        # Drop rows missing either value -- unlike other optional columns, these two
        # aren't defaulted, since every imported line needs a real, distinguishable
        # identity rather than a generic placeholder.
        total = len(contam_lines)
        contam_lines = contam_lines[contam_lines['wavelength'].notna() & contam_lines['species'].notna()]
        contam_lines = contam_lines[contam_lines['species'].astype(str).str.strip() != '']
        skipped = total - len(contam_lines)
        if skipped:
            print(f"[WARN] Skipped {skipped} row(s) missing 'wavelength' or 'species'")

        if len(contam_lines) == 0:
            print("No valid rows (all missing 'wavelength' and/or 'species').")
            return None

        # Rename 'wavelength' column to 'line_wave_um' for consistency with CO catalog
        contam_lines = contam_lines.rename(columns={'wavelength': 'line_wave_um'}).reset_index(drop=True)
        print(f"Loaded {len(contam_lines)} contaminating lines")
        return contam_lines
    except ValueError:
        raise
    except Exception as e:
        print(f"Error loading contaminating lines catalog: {e}")
        return None


def load_custom_line_catalog(filepath):
    """
    Load a user-supplied, HITRAN-like line list from CSV -- for offline use (e.g. no
    internet access to query HITRAN directly), or to hand-pick lines HITRAN doesn't
    have. Populates the same row-dict shape fetch_hitran_data() returns, so these
    lines drop into the existing Lines table / plotting / multi-Gaussian fitting
    exactly like a live HITRAN fetch would.

    'wavelength' (µm) and 'species' are required. fit_hitran_lines_collective()
    only ever fixes each line's *center* from the catalog -- its width comes from
    the instrument's own spectral resolution, not a catalog value -- so wavelength
    is the only thing physically needed to fit these lines; species is required
    anyway so every imported line is actually identifiable in the table/tooltips/
    results, rather than a wall of indistinguishable "Unknown" entries. Everything
    past those two is optional and purely for display (species dialog table, hover
    tooltip, results text), each already tolerant of missing values via
    .get(key, default) at every call site:
        transition, a_coeff, e_upper, isotopologue, isotopologue_name,
        wavelength_unit (defaults to 'µm' if the column is absent).

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    list of dict
        One dict per row, tagged 'source': 'Custom' (so the species dialog's
        HITRAN-only filter and pink row-highlighting correctly treat these as
        non-HITRAN). None if the file is missing, empty, or lacks 'wavelength'/
        'species'. Rows missing either value are skipped individually (with a
        console note) rather than failing the whole file.

    Raises
    ------
    ValueError
        If the CSV file does not contain a 'wavelength' or 'species' column.
    """
    try:
        if not os.path.exists(filepath):
            print(f"Error: File '{filepath}' not found.")
            return None

        print(f">>> Loading custom line list from {filepath}...")
        df = pd.read_csv(filepath)

        missing_cols = [c for c in ('wavelength', 'species') if c not in df.columns]
        if missing_cols:
            error_msg = (
                f"CSV file is missing required column(s): {', '.join(missing_cols)}.\n"
                f"Columns found: {', '.join(df.columns.tolist())}"
            )
            print(error_msg)
            raise ValueError(error_msg)

        def _clean(value):
            return None if pd.isna(value) else value

        rows = []
        skipped = 0
        for _, row in df.iterrows():
            wl = _clean(row.get('wavelength'))
            species = _clean(row.get('species'))
            if wl is None or species is None or not str(species).strip():
                skipped += 1
                continue  # Missing center or identity -- not usable, and species
                          # is required so this isn't defaulted to "Unknown".
            entry = {
                'model': 'Custom',
                'source': 'Custom',
                'wavelength': float(wl),
                'wavelength_unit': str(_clean(row.get('wavelength_unit')) or 'µm'),
                'species': str(species).strip(),
            }
            transition = _clean(row.get('transition'))
            if transition is not None:
                entry['transition'] = str(transition)
            for float_col in ('a_coeff', 'e_upper'):
                value = _clean(row.get(float_col))
                if value is not None:
                    try:
                        entry[float_col] = float(value)
                    except (TypeError, ValueError):
                        pass
            iso_num = _clean(row.get('isotopologue'))
            if iso_num is not None:
                try:
                    entry['isotopologue'] = int(iso_num)
                except (TypeError, ValueError):
                    pass
            # Not run through format_isotopologue_name() -- that assumes HITRAN's
            # own '(13C)(16O)'-style bracket notation, and would misformat plain
            # user text (e.g. subscripting a mass number it mistakes for a count).
            # Shown exactly as given instead.
            iso_name = _clean(row.get('isotopologue_name'))
            if iso_name is not None:
                entry['isotopologue_name'] = str(iso_name)
            rows.append(entry)

        if skipped:
            print(f"[WARN] Skipped {skipped} row(s) missing 'wavelength' or 'species'")

        if not rows:
            print("No valid rows (all missing 'wavelength' and/or 'species').")
            return None

        print(f"[OK] Loaded {len(rows)} custom lines")
        return rows
    except ValueError:
        raise
    except Exception as e:
        print(f"Error loading custom line catalog: {e}")
        return None