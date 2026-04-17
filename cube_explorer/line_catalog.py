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
        print(f"❌ Error: File '{file_path}' not found.")
        return [], {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
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
                                        except:
                                            transition_parts.append(fields[i])
                                    else:
                                        transition_parts.append(fields[i])
                                transition = " ".join(transition_parts) if transition_parts else "Unknown"
                                if coeff_idx >= 0 and coeff_idx + 1 < len(fields):
                                    try:
                                        e_upper = float(fields[coeff_idx + 1])
                                    except:
                                        pass
                        except:
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
        'SO2': 9, 'NO2': 10, 'NH3': 11, 'HNO3': 12, 'OH': 13, 'HF': 14,
        'HI': 15, 'OCS': 16, 'H2CO': 17, 'N2': 18,
        'HCN': 19, 'H2O2': 20, 'C2H2': 21, 'C2H6': 22, 'PH3': 23,
        'COF2': 24, 'SF6': 25, 'H2S': 26, 'HCOOH': 27, 'HO2': 28, 'O': 29,
        'C2H4': 30, 'CH3OH': 31,
        'CH3CN': 32, 'CF4': 33, 'C4H2': 34, 'HC3N': 35, 'H2': 36, 'CS': 37,
        'SO3': 38, 'C2N2': 39, 'SO': 40, 'CH3F': 41,
        'CS2': 42, 'CH3I': 43, 'NF3': 44,
    }

def convert_to_numpy(data):
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return np.array(data, dtype=float)
    if isinstance(data, (list, tuple)):
        try:
            return np.array(data, dtype=float)
        except:
            return np.array(data)
    return np.array([data], dtype=float)

def delete_cache_properly(table_name):
    """Delete specific table cache files without removing the hitran_local directory"""
    try:
        from hapi import purgeCache
        purgeCache(table_name)
        print(f"✓ Purged cache for {table_name}")
    except Exception as e:
        # Fallback: manually delete just the table files
        try:
            data_file = f'hitran_local/{table_name}.data'
            header_file = f'hitran_local/{table_name}.header'
            if os.path.exists(data_file):
                os.remove(data_file)
                print(f"✓ Removed {data_file}")
            if os.path.exists(header_file):
                os.remove(header_file)
                print(f"✓ Removed {header_file}")
        except Exception as e2:
            print(f"⚠️ Could not delete cache files: {e2}")

def get_cache_age(table_name):
    try:
        data_file = f'hitran_local/{table_name}.data'
        if os.path.exists(data_file):
            mod_time = os.path.getmtime(data_file)
            age_seconds = time.time() - mod_time
            return age_seconds / (24 * 3600)
    except:
        return None
    return None

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
        isotopologue: Isotopologue number (default 1)
        refresh_cache: Force cache refresh
        cache_days: Cache age threshold
        v_upper: Filter by upper vibrational level (optional, e.g., 1 for v=1)
        v_lower: Filter by lower vibrational level (optional, e.g., 0 for v=0)
    """
    try:
        from hapi import db_begin, fetch, getColumns, tableList
    except ImportError:
        print("❌ Error: HAPI not installed!")
        print("📦 Install with: pip install hitran-api")
        return [], {}

    mol_numbers = get_available_molecules()
    if molecule not in mol_numbers:
        print(f"❌ Molecule '{molecule}' not found in HITRAN")
        return [], {}

    M = mol_numbers[molecule]
    I = isotopologue
    # Convert wavelength (μm) to wavenumber (cm⁻¹)
    wavelength_min_val = _extract_wavelength_value(wavelength_min)
    wavelength_max_val = _extract_wavelength_value(wavelength_max)
    wavenum_min = 1e4 / wavelength_max_val
    wavenum_max = 1e4 / wavelength_min_val
    table_name = f'{molecule}_{M}_{I}_{int(wavenum_min)}_{int(wavenum_max)}'

    try:
        # Ensure hitran_local directory exists
        os.makedirs('hitran_local', exist_ok=True)
        db_begin('hitran_local')

        # Check for existing cached data
        try:
            existing_tables = tableList()
            data_cached = table_name in existing_tables
        except:
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
        except:
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
                if global_upper_quanta is not None and local_lower_quanta is not None:
                    try:
                        v_upper_str = str(global_upper_quanta[i]).strip()
                        v_lower_str = str(global_lower_quanta[i]).strip()
                        
                        # Extract numeric vibrational levels for filtering
                        try:
                            v_upper_num = int(re.search(r'\d+', v_upper_str).group())
                            v_lower_num = int(re.search(r'\d+', v_lower_str).group())
                        except:
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
                })

        print(f"[OK] Retrieved {len(data_rows)} lines\n")
        return data_rows, {}
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return [], {}

def display_results(results):
    """
    Formats and prints spectral line data in a clean tabular layout. Displays
    wavelength, wavenumber, species, transition notation, Einstein A-coefficient,
    and upper energy level for each line, sorted by wavelength.
    """
    if not results:
        print("❌ No entries found.")
        return
    
    print("="*115)
    print(f"{'MODEL':<10} | {'Wavelength λ [μm]':<19} | {'Wavenumber ν [cm⁻¹]':<20} | {'L-species':<10} | {'Transition':<20} | {'A-coeff(1/sec)':<14} | {'E_upper [K]':<12}")
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
    print(f"✓ Total entries: {len(results)}\n")

def hitran_gui_interface(wavelength_min, wavelength_max, molecule, isotopologue=1, refresh_cache=False, v_upper=None, v_lower=None):
    """
    GUI-compatible function to fetch HITRAN data and return results for display.
    
    Args:
        wavelength_min, wavelength_max: Wavelength range in µm
        molecule: Molecule name
        isotopologue: Isotopologue number (default 1)
        refresh_cache: Force cache refresh
        v_upper: Optional upper vibrational level filter (None = include all)
        v_lower: Optional lower vibrational level filter (None = include all)
    """
    try:
        results, _ = fetch_hitran_data(
            wavelength_min, wavelength_max, molecule, isotopologue, refresh_cache,
            v_upper=v_upper, v_lower=v_lower
        )
        return results
    except Exception as e:
        print(f"❌ Error in HITRAN GUI interface: {e}")
        return []

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
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file (e.g., 'data/CO/ContaminatingLines.csv')
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: wavelength (required), species, and other optional columns.
        The 'wavelength' column is internally renamed to 'line_wave_um' for consistency.
        
    Raises
    ------
    ValueError
        If the CSV file does not contain a 'wavelength' column.
    """
    try:
        if not os.path.exists(filepath):
            print(f"❌ Error: File '{filepath}' not found.")
            return None
        
        print(f">>> Loading contaminating lines catalog from {filepath}...")
        contam_lines = pd.read_csv(filepath)
        
        # VALIDATE: 'wavelength' column MUST exist
        if 'wavelength' not in contam_lines.columns:
            error_msg = (
                f"❌ CSV file is missing required 'wavelength' column.\n"
                f"Columns found: {', '.join(contam_lines.columns.tolist())}"
            )
            print(error_msg)
            raise ValueError(error_msg)
        
        # Rename 'wavelength' column to 'line_wave_um' for consistency with CO catalog
        contam_lines = contam_lines.rename(columns={'wavelength': 'line_wave_um'})
        print(f"✓ Loaded {len(contam_lines)} contaminating lines")
        return contam_lines
    except Exception as e:
        print(f"❌ Error loading contaminating lines catalog: {e}")
        return None