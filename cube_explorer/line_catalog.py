"""
HITRAN/Local Spectral Line Data Extractor
Extracts molecular spectroscopy data from HITRAN database or local line lists
"""

import re
import os
import numpy as np
import pandas as pd
import time
from astropy.io import ascii
from .unit_converter import wavelength_to_doppler_velocity, extract_value_from_quantity, convert_wavelength

# -------------------- Helper Functions --------------------
# Shared by CSV import, Import Map, and Gaussian Map's export identity lookup.
# 50 km/s covers real source/systemic velocities the old flat 1e-4 um cutoff missed.
LINE_MATCH_VELOCITY_TOL_KMS = 50.0


def match_nearest_hitran_line(hitran_lines, wavelength, velocity_tol_kms=LINE_MATCH_VELOCITY_TOL_KMS):
    """Find the HITRAN line nearest to wavelength, within a velocity window.

    Args:
        hitran_lines (list): HITRAN line dicts, each with a 'wavelength' (um).
        wavelength (float): Wavelength to match, in micron.
        velocity_tol_kms (float): Max accepted Doppler-equivalent offset.

    Returns:
        best (dict or None): The nearest line within tolerance, or None.
    """
    best, best_diff = None, float('inf')
    for hl in hitran_lines:
        diff = abs(hl['wavelength'] - wavelength)
        v_kms = abs(wavelength_to_doppler_velocity(hl['wavelength'], wavelength, 'um'))
        if v_kms < velocity_tol_kms and diff < best_diff:
            best, best_diff = hl, diff
    return best


def dedupe_colocated_lines(items, tol_um, line_of=lambda item: item):
    """
    Drop the weaker line of each near-coincident pair (same transition
    tabulated twice in the catalog). tol_um is typically 1e-5, near-exact
    duplicates only, not physically blended-but-distinct lines.

    Args:
        items (list): Line dicts, or another item line_of can pull one from
        tol_um (float): Two lines are the same feature within this many microns
        line_of (callable): Returns a dict with 'wavelength' (and optionally
            'a_coeff') for one item; identity by default

    Returns:
        result (tuple): (kept, dropped) lists of the input items, kept sorted
            by wavelength
    """
    ordered = sorted(items, key=lambda it: line_of(it)["wavelength"])
    kept, dropped = [], []
    for item in ordered:
        if kept and abs(line_of(item)["wavelength"] - line_of(kept[-1])["wavelength"]) <= tol_um:
            weaker, stronger = sorted(
                (item, kept[-1]), key=lambda it: line_of(it).get("a_coeff", 0.0)
            )
            kept[-1] = stronger
            dropped.append(weaker)
        else:
            kept.append(item)
    return kept, dropped


_ISO_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_ISO_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def format_isotopologue_name(iso_name):
    """Convert a HITRAN/HAPI isotopologue formula (e.g. '(13C)(16O)') into a
    readable isotope-labeled formula with mass numbers as superscripts and
    atom counts as subscripts.

    Args:
        iso_name (str): Raw HITRAN/HAPI isotopologue formula, or falsy.

    Returns:
        name (str): Formatted formula, or '' if iso_name is falsy.
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


# -------------------- HITRAN Utilities --------------------

# Two HITRAN cache directories. hitran_local/ is the committed offline starter
# pack (see _ALL_ISOTOPOLOGUES_BY_MOLECULE). hitran_cache/ holds on-demand fetches.
BUNDLED_HITRAN_DIR = 'hitran_local'
USER_HITRAN_DIR = 'hitran_cache'
HITRAN_DIRS = (BUNDLED_HITRAN_DIR, USER_HITRAN_DIR)

_hitran_initialised = False
_hitran_write_dir = None  # directory HAPI writes a fresh fetch() into


def _hitran_init():
    """Prepare HAPI's local database without loading any table. db_begin()
    would parse every .data file in both cache directories; instead tables
    load one at a time in _load_one_hitran_table().

    Args:
        None

    Returns:
        None
    """
    global _hitran_initialised, _hitran_write_dir
    if _hitran_initialised:
        return
    import hapi
    for d in HITRAN_DIRS:
        os.makedirs(d, exist_ok=True)
    hapi.VARIABLES['BACKEND_DATABASE_NAME'] = USER_HITRAN_DIR
    _hitran_write_dir = USER_HITRAN_DIR
    _hitran_initialised = True


def _load_one_hitran_table(table_name):
    """Load a single cached table into HAPI's in-memory cache, from
    BUNDLED_HITRAN_DIR first then USER_HITRAN_DIR, restoring HAPI's write
    target afterward so a later fetch() still targets the on-demand cache.

    Args:
        table_name (str): HITRAN table name, without extension.

    Returns:
        loaded (bool): True if the table is now in HAPI's cache.
    """
    import hapi
    if table_name in hapi.LOCAL_TABLE_CACHE:
        return True
    saved = hapi.VARIABLES.get('BACKEND_DATABASE_NAME')
    try:
        for d in HITRAN_DIRS:
            if os.path.exists(os.path.join(d, f'{table_name}.data')):
                hapi.VARIABLES['BACKEND_DATABASE_NAME'] = d
                hapi.storage2cache(table_name)
                return table_name in hapi.LOCAL_TABLE_CACHE
        return False
    except Exception:
        return False
    finally:
        hapi.VARIABLES['BACKEND_DATABASE_NAME'] = saved


def _point_hitran_writes(dest):
    """
    Point HAPI's write target at `dest` so the next fetch() caches there.

    Args:
        dest (str): Cache directory for new downloads.

    Returns:
        None
    """
    global _hitran_write_dir
    if dest == _hitran_write_dir:
        return
    import hapi
    os.makedirs(dest, exist_ok=True)
    hapi.VARIABLES['BACKEND_DATABASE_NAME'] = dest
    _hitran_write_dir = dest


def get_available_molecules():
    """
    Map molecule name to its HITRAN molecule id, for every molecule with
    line-by-line data in this app's 1-30um range.

    Args:
        None

    Returns:
        molecules (dict): {name: hitran_id}, 52 entries.
    """
    # Excluded: SF6/CF4/NF3/ClONO2 (cross-sections, no line list), O (2 far-IR
    # lines), S2 (UV electronic); HOBr/C2N2/HONO have lines but all > 30um.
    return {
        'H2O': 1, 'CO2': 2, 'O3': 3, 'N2O': 4, 'CO': 5, 'CH4': 6, 'O2': 7, 'NO': 8,
        'SO2': 9, 'NO2': 10, 'NH3': 11, 'HNO3': 12, 'OH': 13, 'HF': 14, 'HCl': 15,
        'HBr': 16, 'HI': 17, 'ClO': 18, 'OCS': 19, 'H2CO': 20, 'HOCl': 21, 'N2': 22,
        'HCN': 23, 'CH3Cl': 24, 'H2O2': 25, 'C2H2': 26, 'C2H6': 27, 'PH3': 28,
        'COF2': 29, 'H2S': 31, 'HCOOH': 32, 'HO2': 33, 'NOp': 36, 'C2H4': 38,
        'CH3OH': 39, 'CH3Br': 40, 'CH3CN': 41, 'C4H2': 43, 'HC3N': 44, 'H2': 45,
        'CS': 46, 'SO3': 47, 'COCl2': 49, 'SO': 50, 'CH3F': 51, 'GeH4': 52,
        'CS2': 53, 'CH3I': 54,
        # 56-61 added to HITRAN post-2020; only fetchable with hitran-api >= 1.3.0.0
        'H3p': 56, 'CH3': 57, 'COFCl': 59, 'ClNO2': 61,
    }

def convert_to_numpy(data):
    """Convert a scalar, list, tuple, or array into a float numpy array.

    Args:
        data: Value to convert, or None.

    Returns:
        arr (numpy.ndarray): Float array, or None if data is None.
    """
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
    """
    Delete a cached table's files from the on-demand cache directory. This never
    touches the committed bundle (hitran_local/). Rebuild the bundle with
    warm_hitran_cache.py.

    Args:
        table_name (str): HITRAN table name, without extension.

    Returns:
        None
    """
    # hapi >= 1.3.0.0 replaced purgeCache with dropTable. dropTable only clears
    # the in-memory cache (on-disk delete is a TODO), so this also removes files.
    try:
        from hapi import dropTable
        dropTable(table_name)
    except Exception:
        pass
    try:
        removed = False
        for ext in ('data', 'header'):
            path = os.path.join(USER_HITRAN_DIR, f'{table_name}.{ext}')
            if os.path.exists(path):
                os.remove(path)
                removed = True
                print(f"Removed {path}")
        if not removed:
            print(f"No cache files found for {table_name} in {USER_HITRAN_DIR}/")
    except Exception as e:
        print(f"Could not delete cache files: {e}")

def get_cache_age(table_name):
    """
    Return the age in days of an on-demand cache table's .data file, or None.
    Bundled tables ship with the repository, so their mtime is not a real cache
    age. This function checks only USER_HITRAN_DIR.

    Args:
        table_name (str): HITRAN table name, without extension.

    Returns:
        age_days (float): File age in days, or None when the file is absent.
    """
    try:
        data_file = os.path.join(USER_HITRAN_DIR, f'{table_name}.data')
        if os.path.exists(data_file):
            return (time.time() - os.path.getmtime(data_file)) / (24 * 3600)
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
    wavelength_min_val = extract_value_from_quantity(wavelength_min)
    wavelength_max_val = extract_value_from_quantity(wavelength_max)
    wavenum_min = 1e4 / wavelength_max_val
    wavenum_max = 1e4 / wavelength_min_val
    return f'{molecule}_{M}_{isotopologue}_{int(wavenum_min)}_{int(wavenum_max)}'


def find_containing_cached_table(wavelength_min, wavelength_max, molecule, isotopologue=1):
    """Find an already-cached table for this (molecule, isotopologue) whose
    wavenumber range fully contains the requested range, so a narrower query
    can reuse a wider table instead of a new exact-range download.

    Args:
        wavelength_min, wavelength_max (float): Wavelength range in um
        molecule (str): Molecule name, per get_available_molecules()
        isotopologue (int): HITRAN isotopologue number

    Returns:
        table_name (str): Name of the smallest containing cached table, or
            None if no cached table covers this range
    """
    M = get_available_molecules().get(molecule)
    if M is None:
        return None
    wavelength_min_val = extract_value_from_quantity(wavelength_min)
    wavelength_max_val = extract_value_from_quantity(wavelength_max)
    wavenum_min = 1e4 / wavelength_max_val
    wavenum_max = 1e4 / wavelength_min_val

    prefix = f'{molecule}_{M}_{isotopologue}_'
    best_name, best_span = None, None
    for cache_dir in HITRAN_DIRS:
        if not os.path.isdir(cache_dir):
            continue
        for fname in os.listdir(cache_dir):
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


# The offline starter pack: molecules and isotopologues bundled in hitran_local/
# so core disk analysis works with no internet. warm_hitran_cache.py builds it.
_ALL_ISOTOPOLOGUES_BY_MOLECULE = {
    "H2O": [1, 2],      # H2(16O); H2(18O) -- water-isotope work in bright disks
    "CO2": [1, 2],      # 12CO2; 13CO2 (Grant+2023 GW Lup, CX Tau, MINDS)
    "CO": [1, 2, 3],    # 12CO / 13CO / C18O -- the 4.5-4.9um NIRSpec rovib set
    "OH": [1],          # isos 2 (18O-H), 3 (16O-D) have no lines <30um (checked vs HITRAN)
    "HCN": [1, 2],      # H(12C)(14N); H(13C)(14N) (MINDS)
    "C2H2": [1, 2],     # 12C2H2; (12C)(13C)H2 in carbon-rich disks (iso2 is ~0.5MB)
    "H2": [1, 2],       # H2; HD
}
COMMON_HITRAN_QUERIES = [
    {"name": f"{molecule} iso{isotopologue} 1-30um (all bands)", "molecule": molecule,
     "isotopologue": isotopologue, "wavelength_min": 1.0, "wavelength_max": 30.0,
     "v_upper": None, "v_lower": None}
    for molecule, isotopologues in _ALL_ISOTOPOLOGUES_BY_MOLECULE.items()
    for isotopologue in isotopologues
]


def get_available_isotopologues(molecule):
    """List the isotopologue numbers HITRAN has data for a given molecule
    (e.g. CO -> [1, 2, 3, 4, 5, 6]). Used to support isotopologue=None
    ("any isotopologue") in fetch_hitran_data().

    Args:
        molecule (str): Molecule name, per get_available_molecules().

    Returns:
        isotopologues (list): Sorted isotopologue numbers, or [] if HAPI
            isn't installed or the molecule isn't recognized.
    """
    try:
        from hapi import ISO
    except ImportError:
        return []
    M = get_available_molecules().get(molecule)
    if M is None:
        return []
    return sorted(i for (m, i) in ISO.keys() if m == M)


def fetch_hitran_data(wavelength_min, wavelength_max, molecule, isotopologue=1, refresh_cache=False, cache_days=30, v_upper=None, v_lower=None, write_dir=None):
    """Retrieve molecular spectroscopic line data from HITRAN via HAPI, with
    smart caching, wavelength-to-wavenumber conversion, quantum-number
    transition labeling, and upper energy level calculation.

    Args:
        wavelength_min, wavelength_max: Wavelength range in micron
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
        write_dir: Directory for a fresh download. Defaults to the on-demand
            cache (hitran_cache/); warm_hitran_cache.py passes hitran_local/.

    Returns:
        result (tuple): (rows, meta) -- list of line-data dicts, and a meta
            dict with 'error'/'message' set when the fetch failed.
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
                refresh_cache, cache_days, v_upper, v_lower, write_dir,
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
        refresh_cache, cache_days, v_upper, v_lower, write_dir,
    )


def _fetch_hitran_data_single_iso(wavelength_min, wavelength_max, molecule, isotopologue, refresh_cache=False, cache_days=30, v_upper=None, v_lower=None, write_dir=None):
    """Fetch (or reuse cached) HITRAN line data for one specific isotopologue.

    Args:
        wavelength_min, wavelength_max (float): Wavelength range in micron.
        molecule (str): Molecule name (e.g., 'CO').
        isotopologue (int): HITRAN isotopologue number.
        refresh_cache (bool): Force a fresh download instead of using the cache.
        cache_days (int): Cache age threshold in days.
        v_upper (int): Filter by upper vibrational level, or None.
        v_lower (int): Filter by lower vibrational level, or None.
        write_dir (str): Directory for a fresh download, or None for the
            default on-demand cache.

    Returns:
        result (tuple): (rows, meta) -- list of line-data dicts, and a meta
            dict with 'error'/'message' set when the fetch failed.
    """
    try:
        from hapi import fetch, getColumns, ISO
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
    # Convert wavelength (micron) to wavenumber (1/cm)
    wavelength_min_val = extract_value_from_quantity(wavelength_min)
    wavelength_max_val = extract_value_from_quantity(wavelength_max)
    wavenum_min = 1e4 / wavelength_max_val
    wavenum_max = 1e4 / wavelength_min_val
    table_name = compute_hitran_table_name(wavelength_min, wavelength_max, molecule, I)

    # A wider already-cached table covering this range serves just as well as
    # an exact-range one, so reuse it instead of a new narrower-range download.
    if not refresh_cache:
        containing = find_containing_cached_table(wavelength_min, wavelength_max, molecule, I)
        if containing is not None:
            table_name = containing

    try:
        _hitran_init()

        # Cheap filename check -- no table parsed yet.
        data_cached = any(
            os.path.exists(os.path.join(d, f'{table_name}.data')) for d in HITRAN_DIRS
        )

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
            if not _load_one_hitran_table(table_name):
                return [], {'error': True,
                            'message': f'cached table {table_name} could not be read'}
            print(f"[OK] Using cached {molecule} data")
        else:
            dest = write_dir or USER_HITRAN_DIR
            _point_hitran_writes(dest)
            print(f"[INFO] Downloading {molecule} from HITRAN...")
            fetch(table_name, M, I, wavenum_min, wavenum_max)
            print(f"[OK] Download complete - cached in {dest}/")

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
        # Calculate upper energy level: E_upper = (E_lower + nu) * 1.44 [K]
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

        # HAPI returns rows wavenumber-ascending (wavelength-descending) -- sort
        # ascending here to match the multi-isotopologue path above.
        data_rows.sort(key=lambda r: r['wavelength'])

        print(f"[OK] Retrieved {len(data_rows)} lines\n")
        return data_rows, {}
    except Exception as e:
        # Distinguish this from a real "zero lines in range" result, so callers
        # don't report a failed offline/network fetch as if HITRAN had nothing.
        print(f"[ERROR] {str(e)}")
        return [], {'error': True, 'message': str(e)}

def hitran_gui_interface(wavelength_min, wavelength_max, molecule, isotopologue=1, refresh_cache=False, v_upper=None, v_lower=None):
    """
    GUI-compatible function to fetch HITRAN data and return results for display.
    
    Args:
        wavelength_min, wavelength_max: Wavelength range in micron
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
    """Load CO rovibrational line catalog from ECSV format.

    Args:
        filepath (str): Path to the ECSV file.

    Returns:
        CO_lines (pandas.DataFrame): Columns trans_id, species, line_wave_um,
            a, transition, eupper(K), highv, highj, lowv, lowj, gp, nu-err,
            nu-ref; None if the file is missing or fails to load.
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
    """Load contaminating lines catalog from CSV format. 'wavelength' and
    'species' are both required (same as load_custom_line_catalog()); other
    columns are optional and carried through as-is if present.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        contam_lines (pandas.DataFrame): line_wave_um (renamed from
            'wavelength'), species, and any other CSV columns. Rows missing
            wavelength or species are dropped individually. None if the file
            is missing, empty, or every row is dropped.

    Raises:
        ValueError: if the CSV lacks a 'wavelength' or 'species' column.
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

        # Drop rows missing either value: unlike other columns, these two
        # aren't defaulted, since every line needs a real, distinguishable identity.
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
    """Load a user-supplied, HITRAN-like line list from CSV for offline use,
    populating the same row-dict shape fetch_hitran_data() returns so these
    lines drop into the existing Lines table/plotting/fitting unchanged.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        rows (list): One dict per row, tagged 'source': 'Custom'. None if
            the file is missing, empty, or lacks 'wavelength'/'species'.
            Rows missing either value are skipped individually.

    Raises:
        ValueError: if the CSV lacks a 'wavelength' or 'species' column.
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
            """Replace a pandas NaN with None; pass other values through.

            Args:
                value: Cell value read from the CSV.

            Returns:
                value: None if value is NaN, else value unchanged.
            """
            return None if pd.isna(value) else value

        rows = []
        skipped = 0
        # Only wavelength is physically needed (width comes from instrument
        # resolution); species is required anyway so every line is identifiable.
        for _, row in df.iterrows():
            wl = _clean(row.get('wavelength'))
            species = _clean(row.get('species'))
            if wl is None or species is None or not str(species).strip():
                skipped += 1
                continue
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
            # Skips format_isotopologue_name(): that assumes HITRAN's bracket
            # notation and would misformat plain user text. Shown as given.
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


# -------------------- Rest-Line Candidates (velocity map) --------------------
LST_CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lst_local', 'publis20240406.lst')

# HITRAN writes isotopologue numbers 10, 11 and 12 as one character.
_PAR_ISOTOPOLOGUE_CODES = {'0': 10, 'A': 11, 'B': 12}
_PAR_RECORD_LENGTH = 160


def _lst_records(path):
    """Read the records of an LST catalog file (.lst) one at a time.
    The header line and any line that is not a record are skipped.

    Args:
        path (str): Path to the .lst file.

    Returns:
        records (generator): One line dict per record, as _parse_lst_record gives it.
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            row = _parse_lst_record(line.rstrip('\r\n'))
            if row is not None:
                yield row


def available_rest_line_species():
    """List the molecules and atoms that have rest-line data in this app.

    Args:
        None

    Returns:
        result (tuple): (names, notes). names is a sorted list of species names.
            notes is a list of sentences that say which source is missing.
    """
    names = set(get_available_molecules())
    notes = []
    if not os.path.exists(LST_CATALOG_PATH):
        notes.append(f"The LST catalog is not at {LST_CATALOG_PATH}, so its lines are not available.")
    else:
        try:
            names.update(row['species'] for row in _lst_records(LST_CATALOG_PATH))
        except OSError as exc:
            notes.append(f"The LST catalog cannot be read: {exc}")
    return sorted(names, key=str.lower), notes


def _lst_lines(species, wavelength_min, wavelength_max):
    """Select the LST catalog lines of one species in a wavelength range.

    Args:
        species (str): Species name as the catalog writes it, for example 'H2' or 'He I'.
        wavelength_min (float): Lower end of the range, in um (vacuum).
        wavelength_max (float): Upper end of the range, in um (vacuum).

    Returns:
        rows (list): Line dicts in the fetch_hitran_data row format, tagged 'LST'.
            None if the catalog file does not exist.

    Raises:
        OSError: if the catalog file cannot be read.
    """
    if not os.path.exists(LST_CATALOG_PATH):
        return None
    rows = []
    for row in _lst_records(LST_CATALOG_PATH):
        if row['species'] == species and wavelength_min <= row['wavelength'] <= wavelength_max:
            row['model'] = 'LST'
            row['source'] = 'LST'
            rows.append(row)
    rows.sort(key=lambda r: r['wavelength'])
    return rows


def _check_window(wavelength_min, wavelength_max):
    """Check that a wavelength window is above zero and in order.

    Args:
        wavelength_min (float): Lower end of the window, in um.
        wavelength_max (float): Upper end of the window, in um.

    Returns:
        None

    Raises:
        ValueError: if the lower end is not above 0, or is not below the upper end.
    """
    if not 0 < wavelength_min < wavelength_max:
        raise ValueError(f"The wavelength window must have 0 < lower < upper, "
                         f"but it is {wavelength_min} to {wavelength_max} um.")


def _par_transition(record):
    """Write the quantum numbers of one .par record as upper -> lower text.

    Args:
        record (str): One 160-character HITRAN-format line.

    Returns:
        label (str): Global quanta, a bar, then local quanta. A blank field shows as '-'.
    """
    def field(start, stop):
        """Return one quanta field with its spaces collapsed.

        Args:
            start (int): First character of the field.
            stop (int): End of the field, not included.

        Returns:
            text (str): The field text, or '-' if the field is blank.
        """
        return ' '.join(record[start:stop].split()) or '-'

    return f"{field(67, 82)} -> {field(82, 97)} | {field(97, 112)} -> {field(112, 127)}"


def read_par_lines(path, wavelength_min, wavelength_max, molecule):
    """Read the lines of one molecule from a HITRAN-format .par file.
    HITEMP files use this format. The file is read one record at a time,
    so a file of many gigabytes does not fill the memory.

    Args:
        path (str): Path to the .par file.
        wavelength_min (float): Lower end of the range, in um (vacuum).
        wavelength_max (float): Upper end of the range, in um (vacuum).
        molecule (str): Molecule name, per get_available_molecules().

    Returns:
        rows (list): Line dicts in the fetch_hitran_data row format, sorted by wavelength.
            The source is 'File: <file name>'. The list is empty if no line is in range.

    Raises:
        ValueError: if the window is not 0 < lower < upper, the molecule has no HITRAN ID,
            the file is not in .par format, or the file holds other molecules only.
    """
    _check_window(wavelength_min, wavelength_max)
    molecule_id = get_available_molecules().get(molecule)
    if molecule_id is None:
        raise ValueError(f"{molecule} has no HITRAN molecule ID, so a .par file cannot hold its lines.")
    try:
        from hapi import ISO
    except ImportError:
        ISO = {}

    wavenumber_min = 1e4 / wavelength_max
    wavenumber_max = 1e4 / wavelength_min
    source = f"File: {os.path.basename(path)}"
    ids_found = set()
    n_records = 0
    n_bad = 0
    rows = []
    with open(path, 'r', encoding='ascii', errors='replace') as handle:
        for record in handle:
            record = record.rstrip('\r\n')
            try:
                if len(record) < _PAR_RECORD_LENGTH:
                    raise ValueError("record is too short")
                line_id = int(record[0:2])
                wavenumber = float(record[3:15])
            except ValueError:
                n_bad += 1
                continue
            n_records += 1
            ids_found.add(line_id)
            if line_id != molecule_id or not wavenumber_min <= wavenumber <= wavenumber_max:
                continue
            try:
                isotopologue = _PAR_ISOTOPOLOGUE_CODES.get(record[2])
                if isotopologue is None:
                    isotopologue = int(record[2])
                a_coeff = float(record[25:35])
                e_lower = float(record[45:55])
                g_text = record[146:153].strip()
            except ValueError:
                n_bad += 1
                continue
            iso_entry = ISO.get((molecule_id, isotopologue))
            rows.append({
                'model': 'File',
                'source': source,
                'wavelength': 1e4 / wavenumber,
                'wavenumber': wavenumber,
                'species': molecule,
                'transition': _par_transition(record),
                'a_coeff': a_coeff,
                'e_upper': (e_lower + wavenumber) * 1.44,
                'Gp': float(g_text) if g_text else None,
                'isotopologue': isotopologue,
                'isotopologue_name': format_isotopologue_name(iso_entry[1]) if iso_entry else None,
            })
    if n_records == 0:
        raise ValueError("The file has no 160-character HITRAN records, so it is not a .par file.")
    if molecule_id not in ids_found:
        found = ', '.join(str(i) for i in sorted(ids_found))
        raise ValueError(f"The file holds molecule ID {found}, but {molecule} has ID {molecule_id}.")
    if n_bad:
        print(f"[WARN] Skipped {n_bad} record(s) that could not be read in {os.path.basename(path)}")
    rows.sort(key=lambda r: r['wavelength'])
    return rows


def _read_csv_line_list(path, wavelength_min, wavelength_max, molecule):
    """Read the lines of one molecule from a user CSV, in a wavelength range.

    Args:
        path (str): Path to the CSV file. It needs 'wavelength' and 'species' columns.
        wavelength_min (float): Lower end of the range, in um (vacuum).
        wavelength_max (float): Upper end of the range, in um (vacuum).
        molecule (str): Species name to keep. The match ignores case.

    Returns:
        rows (list): Line dicts, wavelengths in um, source 'File: <file name>'.
            The list is empty if no line is in range.

    Raises:
        ValueError: if the CSV has no usable row, no row of this species, or an unknown unit.
    """
    rows = load_custom_line_catalog(path)
    if not rows:
        raise ValueError("The CSV has no usable row. Each row needs a 'wavelength' and a 'species'.")
    same_species = [r for r in rows if r['species'].lower() == molecule.lower()]
    if not same_species:
        found = ', '.join(sorted({r['species'] for r in rows}))
        raise ValueError(f"The CSV has no {molecule} line. It holds: {found}.")
    source = f"File: {os.path.basename(path)}"
    kept = []
    for row in same_species:
        row['wavelength'] = float(convert_wavelength(row['wavelength'], row['wavelength_unit'], 'um'))
        row['wavelength_unit'] = 'µm'
        row['source'] = source
        if wavelength_min <= row['wavelength'] <= wavelength_max:
            kept.append(row)
    kept.sort(key=lambda r: r['wavelength'])
    return kept


def _parse_lst_record(record):
    """Read one data line of an LST catalog file (.lst).
    The fixed columns are in the readme of the catalog. A ':' after the wavelength means it is uncertain.

    Args:
        record (str): One line of the file, without the line ending.

    Returns:
        row (dict): Line dict in the fetch_hitran_data row format, wavelength in um (vacuum).
            None if the line is a header or is not an LST record.
    """
    try:
        wavelength = float(record[7:24].replace(':', ' '))
    except ValueError:
        return None
    model = record[0:6].strip()
    species = record[25:34].strip()
    if not model.isalpha() or not species:
        return None
    row = {
        'model': 'File',
        'lst_model': model,
        'wavelength': wavelength,
        'wavelength_unit': 'µm',
        'species': species,
        'transition': record[34:74].strip() or None,
        'uncertain_wavelength': ':' in record[7:24],
    }
    for key, start, stop in (('a_coeff', 75, 84), ('e_upper', 85, 96)):
        text = record[start:stop].strip()
        if text:
            try:
                row[key] = float(text)
            except ValueError:
                return None
    return row


def read_lst_lines(path, wavelength_min, wavelength_max, species):
    """Read the lines of one species from an LST catalog file (.lst), in a wavelength range.

    Args:
        path (str): Path to the .lst file.
        wavelength_min (float): Lower end of the range, in um (vacuum).
        wavelength_max (float): Upper end of the range, in um (vacuum).
        species (str): Species name as the catalog writes it, for example 'H2'. The match ignores case.

    Returns:
        rows (list): Line dicts sorted by wavelength, source 'File: <file name>'.
            The list is empty if no line is in range.

    Raises:
        ValueError: if the file has no LST record, or no record of this species.
    """
    source = f"File: {os.path.basename(path)}"
    wanted = species.lower()
    species_found = set()
    rows = []
    for row in _lst_records(path):
        species_found.add(row['species'])
        if row['species'].lower() == wanted and wavelength_min <= row['wavelength'] <= wavelength_max:
            row['source'] = source
            rows.append(row)
    if not species_found:
        raise ValueError("The file has no LST records, so it is not an LST .lst file.")
    if wanted not in {s.lower() for s in species_found}:
        names = sorted(species_found)
        shown = ', '.join(names[:12]) + (f" and {len(names) - 12} more" if len(names) > 12 else '')
        raise ValueError(f"The file has no {species} line. It holds: {shown}.")
    rows.sort(key=lambda r: r['wavelength'])
    return rows


def load_line_list(path, wavelength_min, wavelength_max, molecule):
    """Read the lines of one molecule from a user file, in a wavelength range.
    The first line decides the format: a CSV header, a HITRAN .par record, or an LST .lst record.
    The file name and the extension do not matter.

    Args:
        path (str): Path to the line list file.
        wavelength_min (float): Lower end of the range, in um (vacuum).
        wavelength_max (float): Upper end of the range, in um (vacuum).
        molecule (str): Molecule or species name.

    Returns:
        rows (list): Line dicts sorted by wavelength. The list is empty if no line is in range.

    Raises:
        ValueError: if the window is not 0 < lower < upper, or the file is empty,
            has an unknown format, or lacks this molecule.
        OSError: if the file cannot be opened.
    """
    _check_window(wavelength_min, wavelength_max)
    first = ''
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as handle:
        for line in handle:
            if line.strip():
                first = line.rstrip('\r\n')
                break
    if not first:
        raise ValueError("The file is empty.")
    if first.startswith('|MODEL|'):
        return read_lst_lines(path, wavelength_min, wavelength_max, molecule)
    if ',' in first and 'wavelength' in first.lower():
        return _read_csv_line_list(path, wavelength_min, wavelength_max, molecule)
    try:
        is_par = len(first) >= _PAR_RECORD_LENGTH and int(first[0:2]) > 0 and float(first[3:15]) >= 0
    except ValueError:
        is_par = False
    if is_par:
        return read_par_lines(path, wavelength_min, wavelength_max, molecule)
    if _parse_lst_record(first) is not None:
        return read_lst_lines(path, wavelength_min, wavelength_max, molecule)
    raise ValueError("IFU Analyzer cannot read this file. Use a CSV with 'wavelength' and "
                     "'species' columns, a HITRAN-format .par file, or an LST .lst file.")


def collect_rest_line_candidates(molecule, wavelength_min, wavelength_max):
    """Collect the built-in lines of one species in a wavelength range.
    The sources are HITRAN (local pack, cache, or online) and the LST catalog.
    Every line stays in the list. Nothing is merged or hidden.

    Args:
        molecule (str): Molecule or atom name, per available_rest_line_species().
        wavelength_min (float): Lower end of the range, in um (vacuum).
        wavelength_max (float): Upper end of the range, in um (vacuum).

    Returns:
        result (tuple): (rows, notes). rows is a list of line dicts, each tagged
            with its 'source'. notes is a list of sentences that say which source failed.

    Raises:
        ValueError: if the window is not 0 < lower < upper.
    """
    _check_window(wavelength_min, wavelength_max)
    rows = []
    notes = []
    if molecule in get_available_molecules():
        hitran_rows, meta = hitran_gui_interface(wavelength_min, wavelength_max, molecule, None, False)
        for row in hitran_rows:
            row['source'] = 'HITRAN'
        rows.extend(hitran_rows)
        if meta.get('error'):
            notes.append(f"HITRAN gave no {molecule} data: {meta.get('message', 'unknown error')}")
    try:
        lst_rows = _lst_lines(molecule, wavelength_min, wavelength_max)
        if lst_rows is None:
            notes.append(f"The LST catalog is not at {LST_CATALOG_PATH}, so its lines are not listed.")
        else:
            rows.extend(lst_rows)
    except OSError as exc:
        notes.append(f"The LST catalog cannot be read: {exc}")
    rows.sort(key=lambda r: r['wavelength'])
    return rows, notes