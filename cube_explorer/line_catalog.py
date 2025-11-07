import re

def filter_spectral_lines(file_path, wavelength_min, wavelength_max, species=None):
    """
    Filter spectral lines in .lst file by wavelength and optional species.
    Returns:
        - matching_rows: list of line strings for the selected species
        - all_species: dict of all species found in the range
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data_rows = []
    all_species = {}
    
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                wavelength = float(parts[1])
                if wavelength_min <= wavelength <= wavelength_max:
                    line_text = line
                    bracketed_match = re.search(r'\[[\w\s]+\]', line_text)
                    if bracketed_match:
                        found_species = bracketed_match.group()
                    else:
                        if len(parts) > 3 and (parts[3].upper() in ['I','II','III','IV','V','VI','VII','VIII','IX','X']):
                            found_species = parts[2] + ' ' + parts[3]
                        else:
                            found_species = parts[2].strip()
                    
                    all_species[found_species] = all_species.get(found_species, 0) + 1
                    
                    if species is None or species == found_species:
                        data_rows.append({
                            "model": parts[0],
                            "lambda_micron": wavelength,
                            "species": found_species,
                            "transition": " ".join(parts[3:]),
                            "raw": line.strip()
                        })
            except (ValueError, IndexError):
                continue
    
    return data_rows, all_species