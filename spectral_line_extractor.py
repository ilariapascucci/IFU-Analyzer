"""
HITRAN/Local Spectral Line Data Extractor
Extracts molecular spectroscopy data from HITRAN database or local line lists
"""

import re
import os
import numpy as np
import shutil
import time

# Searches a local spectral line file for species matches (exact first, then partial), and returns result lines in range
def filter_spectral_lines(file_path, wavelength_min, wavelength_max, species):
    """Filter local file for entries with matching species and within wavelength range."""
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
    # Process each line in file (skip header)
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                wavelength = float(parts[1])
                if wavelength_min <= wavelength <= wavelength_max:
                    # Robustly extract species
                    bracketed_match = re.search(r'\[[\w\s]+\]', line)
                    if bracketed_match:
                        found_species = bracketed_match.group()
                    else:
                        if len(parts) > 3 and (parts[3].upper() in ['I','II','III','IV','V','VI','VII','VIII','IX','X']):
                            found_species = parts[2] + ' ' + parts[3]
                        else:
                            found_species = parts[2].strip()
                    # Track all found species
                    if found_species not in all_species:
                        all_species[found_species] = 0
                    all_species[found_species] += 1
                    # Determine match type
                    species_upper = species.upper()
                    found_upper = found_species.upper()
                    exact_match = (found_upper == species_upper)
                    element_match = species_upper in found_upper and '[' in found_species
                    if exact_match:
                        priority = 0  # exact first
                    elif element_match:
                        priority = 1  # partial after
                    else:
                        priority = 2
                    # Only include exact (0) or partial (1)
                    if priority <= 1:
                        wavenumber = 1e4 / wavelength
                        model = parts[0]
                        transition = ""
                        a_coeff = 0.0
                        e_upper = 0.0
                        # Robustly extract transition, A-coeff, e_upper
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
    # Sort for exact match priority, then by wavelength
    data_rows = sorted(data_rows, key=lambda x: (x['priority'], x['wavelength']))
    for row in data_rows:
        del row['priority']  # don't show priority in results
    return data_rows, all_species

# Returns a mapping of molecule names to HITRAN IDs
def get_available_molecules():
    """List available HITRAN molecular species."""
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

# Ensures HAPI output is a numpy array (for mathematical operations)
def convert_to_numpy(data):
    """Convert any data array to a Numpy array of floats."""
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

# Deletes cached HITRAN data
# (tries to use HAPI's purgeCache, else deletes cache folder manually)
def delete_cache_properly(table_name):
    """Remove .data/.header cache for a given table using recommended or fallback method."""
    try:
        from hapi import purgeCache
        print(f"✓ Using HAPI purgeCache to delete old data...")
        purgeCache(table_name)
    except:
        try:
            shutil.rmtree('hitran_local', ignore_errors=True)
            print(f"✓ Cache directory cleared")
        except Exception as e:
            print(f"⚠️  Warning: Could not delete cache: {e}")

# Checks how old the cache file is (in days)
def get_cache_age(table_name):
    """Return age of HITRAN cache file, in days."""
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

# Downloads (if needed) and returns HITRAN data for a molecule in a given wavelength range
def fetch_hitran_data(wavelength_min, wavelength_max, molecule, isotopologue=1, refresh_cache=False, cache_days=30):
    """Fetch molecular line data from HITRAN, using smart cache and data validation."""
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
    wavenum_min = 1e4 / wavelength_max 
    wavenum_max = 1e4 / wavelength_min
    table_name = f'{molecule}_{M}_{I}_{int(wavenum_min)}_{int(wavenum_max)}'
    try:
        db_begin('hitran_local')
        # Smart cache logic: only download if missing or cache is too old
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
        # If the user requested, nuke the old cache
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
        try:
            # Use full set of working columns if available
            nu, sw, A, gamma_air, gamma_self, elower, n_air, delta_air, Gp, Gpp, \
            global_upper_quanta, global_lower_quanta = getColumns(
                table_name, 
                ['nu', 'sw', 'A', 'gamma_air', 'gamma_self', 'elower', 'n_air', 
                 'delta_air', 'Gp', 'Gpp', 'global_upper_quanta', 'global_lower_quanta']
            )
        except:
            nu, sw, A, gamma_air, gamma_self, elower, n_air, delta_air, Gp, Gpp = getColumns(
                table_name, 
                ['nu', 'sw', 'A', 'gamma_air', 'gamma_self', 'elower', 'n_air', 
                 'delta_air', 'Gp', 'Gpp']
            )
            global_upper_quanta = None
            global_lower_quanta = None
            #local_upper quanta
            # local_lower_quanta
        nu = convert_to_numpy(nu)
        A = convert_to_numpy(A)
        elower = convert_to_numpy(elower)
        wavelength = 1e4 / nu
        eupper = (elower + nu) * 1.44
        data_rows = []
        species_name = molecule
        for i in range(len(nu)):
            wl = wavelength[i]
            if wavelength_min <= wl <= wavelength_max:
                if global_upper_quanta is not None:
                    upper_state = str(global_upper_quanta[i]).strip()
                    lower_state = str(global_lower_quanta[i]).strip()
                    transition = f"{upper_state}->{lower_state}"
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

# Nicely formats and prints out the data in table-layout
def display_results(results):
    """Display all result lines cleanly and with helpful columns."""
    if not results:
        print("❌ No entries found.")
        return
    print("="*115)
    print(f"{'MODEL':<10} | {'Wavelength λ [μm]':<19} | {'Wavenumber ν [cm⁻¹]':<20} | {'L-species':<10} | {'Transition':<20} | {'A-coeff(1/sec)':<14} | {'E_upper [K]':<12}")
    print("="*115)
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

# Top-level interface for all user-interaction and orchestration
def main():
    """Main command-line interface: gets user input, chooses and runs search, and displays."""
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
            file_name = input("\n📄 Enter file name (e.g., publis20240406.lst): ").strip()
            if not file_name:
                print("❌ File name cannot be empty")
                return
            species = input("🔍 Enter species (e.g., H2, Fe, He I, [Fe II]): ").strip()
            if not species:
                print("❌ Species cannot be empty")
                return
            print("\n⏳ Searching local file...")
            results, available_species = filter_spectral_lines(file_name, wavelength_min, wavelength_max, species)
            if not results:
                print(f"\n⚠️  No matches found for '{species}'")
                if available_species:
                    print(f"\n📊 Available species in {wavelength_min:.2f}-{wavelength_max:.2f} μm range:")
                    print("─" * 80)
                    for sp, count in sorted(all_species.items(), key=lambda x: x[1], reverse=True):
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
