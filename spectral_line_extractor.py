"""
HITRAN/Local Spectral Line Data Extractor
Extracts molecular spectroscopy data from HITRAN database or local line lists
"""


import re
import os
import numpy as np
import shutil
import time


def normalize_species_input(user_input):
    """
    Converts various species input formats into standardized bracketed notation.
    Handles space-separated (Fe II), concatenated (FeII), numeric (Fe2), and 
    already-bracketed formats. Distinguishes between molecular species and ionic 
    states using pattern matching with priority-based regex rules.
    """
    if not user_input:
        return user_input
    
    original = user_input.strip()
    
    # Preserve exact formatting for already-bracketed input
    if original.startswith('[') and original.endswith(']'):
        return original
    
    species = original.upper()
    species = species.replace('[', '').replace(']', '').strip()
    
    molecules = ['H2', 'CO', 'CO2', 'H2O', 'CH4', 'NH3', 'O2', 'N2', 'NO', 'NO2', 
                 'SO2', 'O3', 'H2S', 'HCN', 'C2H2', 'C2H6', 'PH3', 'H2CO', 'N2O',
                 'HCL', 'HF', 'HBR', 'HI', 'OH', 'CLO', 'OCS', 'HOCL']
    
    # Return molecules unchanged to prevent incorrect ion conversion
    if species in molecules:
        return original
    
    ionization_map = {
        '1': 'I', '2': 'II', '3': 'III', '4': 'IV', '5': 'V',
        '6': 'VI', '7': 'VII', '8': 'VIII', '9': 'IX', '10': 'X',
        'I': 'I', 'II': 'II', 'III': 'III', 'IV': 'IV', 'V': 'V',
        'VI': 'VI', 'VII': 'VII', 'VIII': 'VIII', 'IX': 'IX', 'X': 'X'
    }
    
    # Handle space-separated format first (highest priority)
    parts = species.split()
    if len(parts) == 2:
        element, ion_state = parts[0], parts[1]
        if ion_state in ionization_map or ion_state.isdigit():
            if ion_state.isdigit():
                ion_state = ionization_map.get(ion_state, ion_state)
            else:
                ion_state = ionization_map.get(ion_state, ion_state)
            return f"[{element.capitalize()} {ion_state}]"
    
    # Two-letter elements + roman numerals (e.g., TiII, NiII, SiIV)
    match = re.match(r'^([A-Z][A-Z])([IVX]+)$', species)
    if match:
        element, ion_state = match.groups()
        second_char = element[1]
        # Avoid false matches like OVI->Ov+I by checking second character
        if (second_char not in ['I', 'V', 'X'] or len(ion_state) >= 2) and ion_state in ionization_map:
            return f"[{element.capitalize()} {ion_state}]"
    
    # Single letter + multi-character roman numerals (e.g., OVI, CIV)
    match = re.match(r'^([A-Z])([IVX]{2,})$', species)
    if match:
        element, ion_state = match.groups()
        if ion_state in ionization_map:
            return f"[{element} {ion_state}]"
    
    # Numeric ionization notation (e.g., Fe2, Ca3)
    match = re.match(r'^([A-Z][A-Z]?)(\d+)$', species)
    if match:
        element, ion_num = match.groups()
        # Prevent molecules like H2, O2 from being converted to ions
        if not (len(element) == 1 and element + ion_num in molecules):
            ion_state = ionization_map.get(ion_num, ion_num)
            return f"[{element.capitalize()} {ion_state}]"
    
    return original


def filter_spectral_lines(file_path, wavelength_min, wavelength_max, species):
    """
    Parses a local spectral line list file and filters entries by wavelength range 
    and species match. Performs exact matching first, then partial element matching.
    Extracts transition data, A-coefficients, and upper energy levels from the file 
    format while tracking all available species in the specified range.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found in current directory.")
        return [], {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return [], {}
    data_rows = []
    all_species = {}
    
    # Skip header, process data lines
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                wavelength = float(parts[1])
                if wavelength_min <= wavelength <= wavelength_max:
                    # Extract species from bracketed or space-separated format
                    bracketed_match = re.search(r'\[[\w\s]+\]', line)
                    if bracketed_match:
                        found_species = bracketed_match.group()
                    else:
                        if len(parts) > 3 and (parts[3].upper() in ['I','II','III','IV','V','VI','VII','VIII','IX','X']):
                            found_species = parts[2] + ' ' + parts[3]
                        else:
                            found_species = parts[2].strip()
                    
                    if found_species not in all_species:
                        all_species[found_species] = 0
                    all_species[found_species] += 1
                    
                    # Priority matching: exact first, then partial element match
                    species_upper = species.upper()
                    found_upper = found_species.upper()
                    exact_match = (found_upper == species_upper)
                    element_match = species_upper in found_upper and '[' in found_species
                    if exact_match:
                        priority = 0
                    elif element_match:
                        priority = 1
                    else:
                        priority = 2
                    
                    if priority <= 1:
                        wavenumber = 1e4 / wavelength
                        model = parts[0]
                        transition = ""
                        a_coeff = 0.0
                        e_upper = 0.0
                        
                        # Parse transition info and A-coefficient from variable-width fields
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
    
    # Sort by priority (exact matches first) then wavelength
    data_rows = sorted(data_rows, key=lambda x: (x['priority'], x['wavelength']))
    for row in data_rows:
        del row['priority']
    return data_rows, all_species


def get_available_molecules():
    """
    Returns a dictionary mapping molecule names to their HITRAN database IDs.
    Includes all 55 molecules available in the HITRAN2020+ database for 
    spectroscopic line-by-line data retrieval.
    """
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
    """
    Ensures HAPI output is converted to numpy arrays for mathematical operations.
    Handles various input types including lists, tuples, and existing numpy arrays,
    converting them to float64 format for consistent numerical processing.
    """
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
    """
    Removes cached HITRAN data using either HAPI's purgeCache method or manual
    directory deletion as fallback. Ensures clean cache state when refreshing data
    from the remote HITRAN database.
    """
    try:
        from hapi import purgeCache
        print(f"✓ Using HAPI purgeCache to delete old data...")
        purgeCache(table_name)
    except:
        # Fallback to manual directory removal
        try:
            shutil.rmtree('hitran_local', ignore_errors=True)
            print(f"✓ Cache directory cleared")
        except Exception as e:
            print(f"⚠️  Warning: Could not delete cache: {e}")


def get_cache_age(table_name):
    """
    Calculates the age of cached HITRAN data files in days by checking file 
    modification timestamps. Used to determine if cached data should be refreshed
    based on user-defined cache expiration policies.
    """
    try:
        data_file = f'hitran_local/{table_name}.data'
        if os.path.exists(data_file):
            mod_time = os.path.getmtime(data_file)
            age_seconds = time.time() - mod_time
            age_days = age_seconds / (24 * 3600)
            return age_days
    except:
        return None
    return None


def fetch_hitran_data(wavelength_min, wavelength_max, molecule, isotopologue=1, refresh_cache=False, cache_days=30):
    """
    Retrieves molecular spectroscopic line data from the HITRAN database using HAPI.
    Implements smart caching to minimize downloads, converts wavelength ranges to 
    wavenumbers, extracts quantum numbers for transition labeling, and calculates 
    upper energy levels. Handles both cached and fresh data with user prompts for 
    cache refresh based on age.
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
    wavenum_min = 1e4 / wavelength_max 
    wavenum_max = 1e4 / wavelength_min
    table_name = f'{molecule}_{M}_{I}_{int(wavenum_min)}_{int(wavenum_max)}'
    
    try:
        db_begin('hitran_local')
        
        # Check for existing cached data
        try:
            existing_tables = tableList()
            data_cached = table_name in existing_tables
        except:
            data_cached = False
        
        if data_cached and not refresh_cache:
            cache_age = get_cache_age(table_name)
            if cache_age is not None:
                print(f"📅 Cache age: {cache_age:.1f} days")
                if cache_age > cache_days:
                    print(f"⚠️  Cache is older than {cache_days} days")
                    refresh = input("🔄 Download latest data? (y/n, default n): ").strip().lower() == 'y'
                    if refresh:
                        refresh_cache = True
        
        if refresh_cache and data_cached:
            print(f"🗑️  Deleting old cache...")
            delete_cache_properly(table_name)
            data_cached = False
        
        if data_cached and not refresh_cache:
            print(f"✓ Using cached {molecule} data")
        else:
            print(f"📥 Downloading {molecule} from HITRAN...")
            fetch(table_name, M, I, wavenum_min, wavenum_max)
            print(f"✓ Download complete - cached locally")
        
        print(f"⏳ Processing data...\n")
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
        wavelength = 1e4 / nu
        # Calculate upper energy level: E_upper = (E_lower + ν) × 1.44 [K]
        eupper = (elower + nu) * 1.44
        species_name = molecule
        
        for i in range(len(nu)):
            wl = wavelength[i]
            if wavelength_min <= wl <= wavelength_max:
                # Parse transition notation from quantum numbers
                if global_upper_quanta is not None and local_lower_quanta is not None:
                    try:
                        v_upper = str(global_upper_quanta[i]).strip()
                        v_lower = str(global_lower_quanta[i]).strip()
                        
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
                            
                            transition = f"v={v_upper}->{v_lower}, J={j_upper}->{j_lower}"
                        else:
                            transition = f"v={v_upper}->{v_lower}"
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
                })
        
        print(f"✓ Retrieved {len(data_rows)} lines\n")
        return data_rows, {}
    except Exception as e:
        print(f"❌ Error: {str(e)}")
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


def main():
    """
    Main command-line interface that orchestrates user interaction, data source
    selection (local file or HITRAN database), input validation, and result display.
    Handles both local spectral line list searches and remote HITRAN database queries
    with comprehensive error handling and user-friendly prompts.
    """
    print("\n" + "="*110)
    print(" " * 30 + "🔬 HITRAN/Local Spectral Line Data Extractor")
    print("="*110 + "\n")
    
    try:
        wavelength_min = float(input("📍 Enter minimum wavelength (μm): "))
        wavelength_max = float(input("📍 Enter maximum wavelength (μm): "))
        if wavelength_min >= wavelength_max:
            print("❌ Error: Minimum wavelength must be less than maximum")
            return
        
        print("\n📚 Choose data source:")
        print("  1️⃣  Local file")
        print("  2️⃣  HITRAN database")
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            # Local file search workflow
            file_name = input("\n📄 Enter file name (e.g., publis20240406.lst): ").strip()
            if not file_name:
                print("❌ File name cannot be empty")
                return
            
            species_input = input("🔍 Enter species (e.g., H2, Fe, He I, Fe II, [Fe II]): ").strip()
            if not species_input:
                print("❌ Species cannot be empty")
                return
            
            species = normalize_species_input(species_input)
            print(f"   → Searching for: {species}")
            print("\n⏳ Searching local file...")
            
            results, available_species = filter_spectral_lines(file_name, wavelength_min, wavelength_max, species)
            
            # Display available species if no match found
            if not results:
                print(f"\n⚠️  No matches found for '{species}'")
                if available_species:
                    print(f"\n📊 Available species in {wavelength_min:.2f}-{wavelength_max:.2f} μm range:")
                    print("─" * 80)
                    for sp, count in sorted(available_species.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {sp:<30} ({count} lines)")
                    print("─" * 80)
            else:
                unique_species = set([r['species'] for r in results])
                if len(unique_species) == 1 and list(unique_species)[0].upper() == species.upper():
                    print(f"✓ Found exact match for '{species}'")
                else:
                    print(f"✓ Found '{species}' in:")
                    for sp in sorted(unique_species):
                        print(f"   • {sp}")
            
            display_results(results)
            
        elif choice == '2':
            # HITRAN database search workflow
            mol_numbers = get_available_molecules()
            print("\n📊 Available molecules (55 total):")
            print("─" * 110)
            mol_list = sorted(mol_numbers.keys())
            for i in range(0, len(mol_list), 8):
                print("  " + "  ".join(f"{m:<10}" for m in mol_list[i:i+8]))
            print("─" * 110)
            
            molecule = input("\n🔬 Enter molecule (e.g., H2O, CO2, H2): ").strip().upper()
            if not molecule:
                print("❌ Molecule cannot be empty")
                return
            
            isotopologue_input = input("🔢 Enter isotopologue number (default 1): ").strip()
            isotopologue = int(isotopologue_input) if isotopologue_input else 1
            
            print("\n🔄 Cache options:")
            print("  1️⃣  Use cached data (fast)")
            print("  2️⃣  Refresh data (download latest)")
            cache_choice = input("\nEnter choice (default 1): ").strip() or "1"
            refresh = cache_choice == "2"
            
            results, _ = fetch_hitran_data(wavelength_min, wavelength_max, molecule, isotopologue, refresh_cache=refresh, cache_days=30)
            display_results(results)
            
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")
            
    except ValueError:
        print("❌ Error: Please enter valid numbers for wavelength")
    except KeyboardInterrupt:
        print("\n\n⛔ Program interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
