"""
Transition Regex Patterns for Molecules

Defines how each molecule's transitions are formatted in HITRAN data.
Used for parsing and filtering spectral lines by transition type.
"""

# Regex patterns for extracting transitions from HITRAN data
TRANSITION_PATTERNS = {
    'CO': {
        'regex': r'v=(\d+)->(\d+)',
        'example': 'v=1->0',
        'description': 'Vibrational transitions: upper->lower',
        'groups': ['v_upper', 'v_lower'],
    },
    'H2': {
        'regex': r'v=(\d+)->(\d+)',
        'example': 'v=6->3',
        'description': 'Vibrational transitions: upper->lower',
        'groups': ['v_upper', 'v_lower'],
    },
    'SO': {
        'regex': r'v=(V\d+(?:\+V\d+)*)?->GROUND',
        'example': 'v=V5->GROUND',
        'description': 'Vibrational transitions to ground state: single or combined vibrational modes',
        'groups': ['vibrational_modes'],
    },
    'OH': {
        'regex': r'([A-Z]\d+/\d+)\s+(\d+)->([A-Z]\d+/\d+)\s+(\d+)',
        'example': 'X1/2   3->X1/2   2',
        'description': 'Electronic state transitions: state + quantum numbers',
        'groups': ['upper_state', 'upper_quantum', 'lower_state', 'lower_quantum'],
    },
    'H2O': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=0 1 0->0 0 0',
        'description': 'Multi-quantum transitions: space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'C2H2': {
        'regex': r'v=([\d\s+]+?)\s+([ug])\s*->\s*([\d\s+]+?)\s+([ug])',
        'example': 'v=000 0 3 0 1  u->000 0 0 0 0+ g',
        'description': 'Rotational-vibrational transitions: quantum numbers + electronic symmetry (u/g)',
        'groups': ['upper_vibrational', 'upper_symmetry', 'lower_vibrational', 'lower_symmetry'],
    },
    'C2H4': {
        'regex': r'v=(V\d+(?:\+V\d+)*)?->GROUND',
        'example': 'v=V9->GROUND',
        'description': 'Vibrational transitions to ground state: single or combined vibrational modes (V# or V#+V#+...)',
        'groups': ['vibrational_modes'],
    },
    'C2H6': {
        'regex': r'v=(\d*V\d+(?:\+\d*V\d+)*)?->GROUND',
        'example': 'v=V7->GROUND',
        'description': 'Vibrational transitions to ground state: vibrational modes with optional coefficients (V#, 2V#, etc.)',
        'groups': ['vibrational_modes'],
    },
    'C2N2': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=0 0 0 0 7 1->0 0 0 0 6 2',
        'description': 'Multi-quantum transitions: multiple space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'C4H2': {
        'regex': r'v=([a-z0-9]*)\s+([ef])\s+([ug])->([a-z0-9]*)\s+([ef])\s+([ug])',
        'example': 'v=000100000 e u->000000000 e g',
        'description': 'Vibrational + electronic parity + symmetry transitions: optional quantum numbers, e/f parity, u/g symmetry',
        'groups': ['upper_vibrational', 'upper_parity', 'upper_symmetry', 'lower_vibrational', 'lower_parity', 'lower_symmetry'],
    },
    'CH3CN': {
        'regex': r'v=(\d*V\d+(?:\+\d*V\d+)*)->(GROUND|\d*V\d+(?:\+\d*V\d+)*)',
        'example': 'v=V4+V8->V8',
        'description': 'Vibrational transitions: modes with optional coefficients to GROUND or another vibrational state',
        'groups': ['upper_modes', 'lower_state'],
    },
    'CH3F': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=0 0 0 0 1 0->0 0 0 0 0 0',
        'description': 'Multi-quantum transitions: multiple space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'CH3I': {
        'regex': r'v=(\d+(?:\s+\d+)*)\s+(\d+[A-Z]+\d*)->(\d+(?:\s+\d+)*)\s+(\d+[A-Z]+\d*)',
        'example': 'v=0 0 1 0 1F2->0 0 0 0 1A1',
        'description': 'Vibrational + symmetry state transitions: multiple quantum numbers + state symbol (e.g., 1F2, 1A1)',
        'groups': ['upper_quantum', 'upper_state', 'lower_quantum', 'lower_state'],
    },
    'CH3OH': {
        'regex': r'v=(\d*V\d+(?:\+\d*V\d+)*)?->GROUND',
        'example': 'v=V9->GROUND',
        'description': 'Vibrational transitions to ground state: single or combined modes with optional coefficients (V#, 2V#+V#, etc.)',
        'groups': ['vibrational_modes'],
    },
    'CH4': {
        'regex': r'v=(\d+(?:\s+\d+)*\s+\d+[A-Z]+\d*)?->(\d+(?:\s+\d+)*\s+\d+[A-Z]+\d*)?',
        'example': 'v=0 0 0 2 1E->0 0 0 0 1A1',
        'description': 'Multi-quantum + symmetry state transitions: quantum numbers + state symbol, supports empty states',
        'groups': ['upper_state', 'lower_state'],
    },
    'CO2': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=6 0 0 15->0 0 0 01',
        'description': 'Multi-quantum transitions: space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'COF2': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=0 0 2 0->0 0 0 0',
        'description': 'Multi-quantum transitions: space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'CS': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=1000000 0 0 0->0000000 0 0 0',
        'description': 'Multi-quantum transitions: space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'CS2': {
        'regex': r'v=([a-zA-Z])\s+(\d+)->([A-Z])\s+(\d+)',
        'example': 'v=b      1->X      6',
        'description': 'Electronic state transitions: letter-based vibrational state + quantum numbers',
        'groups': ['upper_state', 'upper_quantum', 'lower_state', 'lower_quantum'],
    },
    'H2CO': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=1 2 2 0->1 0 0 0',
        'description': 'Multi-quantum transitions: space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'H2O2': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=0 2 2 0->0 1 1 0',
        'description': 'Multi-quantum transitions: space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'H2S': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=0 1 0 0 0 0->0 0 0 0 0 0',
        'description': 'Multi-quantum transitions: space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'HC3N': {
        'regex': r'v=(V\d+(?:\+V\d+)*)?->GROUND',
        'example': 'v=V6->GROUND',
        'description': 'Vibrational transitions to ground state: single or combined vibrational modes',
        'groups': ['vibrational_modes'],
    },
    'HCN': {
        'regex': r'v=(\d+)->(\d+)',
        'example': 'v=4->0',
        'description': 'Vibrational transitions: upper->lower',
        'groups': ['v_upper', 'v_lower'],
    },
    'HF': {
        'regex': r'v=(\d+)->(\d+)',
        'example': 'v=2->2',
        'description': 'Vibrational transitions: upper->lower',
        'groups': ['v_upper', 'v_lower'],
    },
    'HI': {
        'regex': r'v=(\d+)->(\d+)',
        'example': 'v=0->0',
        'description': 'Vibrational transitions: upper->lower',
        'groups': ['v_upper', 'v_lower'],
    },
    'HNO3': {
        'regex': r'v=(\d*V\d+(?:\+\d*V\d+)*)?->GROUND',
        'example': 'v=V5->GROUND',
        'description': 'Vibrational transitions to ground state: single or combined modes with optional coefficients',
        'groups': ['vibrational_modes'],
    },
    'HO2': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=0 1 0 0 0 0->0 0 0 0 0 0',
        'description': 'Multi-quantum transitions: space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'N2': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=1 2 2 0->1 0 0 0',
        'description': 'Multi-quantum transitions: space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'N2O': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=0 2 2 1->0 4 2 0',
        'description': 'Multi-quantum transitions: space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'O2': {
        'regex': r'v=([a-zA-Z])\s+(\d+)->([A-Z])\s+(\d+)',
        'example': 'v=a      1->X      0',
        'description': 'Electronic state transitions: letter-based vibrational state + quantum numbers',
        'groups': ['upper_state', 'upper_quantum', 'lower_state', 'lower_quantum'],
    },
    'O3': {
        'regex': r'v=(I{0,2})?\s+(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=II  2 3 3->0 0 0',
        'description': 'Multi-quantum transitions: optional Roman numeral prefix + space-separated quantum numbers',
        'groups': ['roman_prefix', 'upper_quantum', 'lower_quantum'],
    },
    'OCS': {
        'regex': r'v=(\d+)->(\d+)',
        'example': 'v=5->0',
        'description': 'Vibrational transitions: upper->lower',
        'groups': ['v_upper', 'v_lower'],
    },
    'PH3': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=1 1 1 3->0 0 0 0',
        'description': 'Multi-quantum transitions: space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'SF6': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=0 0 001 0 1->0 0 001 0 0',
        'description': 'Multi-quantum transitions: space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'SO2': {
        'regex': r'v=(\d+(?:\s+\d+)*)->(\d+(?:\s+\d+)*)',
        'example': 'v=2 1 1->0 0 0',
        'description': 'Multi-quantum transitions: space-separated quantum numbers for upper and lower states',
        'groups': ['upper_quantum', 'lower_quantum'],
    },
    'SO3': {
        'regex': r'v=(\d*V\d+(?:\+\d*V\d+)*)?->GROUND',
        'example': 'v=V9->GROUND',
        'description': 'Vibrational transitions to ground state: single or combined modes with optional coefficients (V#, 2V#+V#, etc.)',
        'groups': ['vibrational_modes'],
    },
}


def get_transition_pattern(molecule):
    """
    Get transition regex pattern for a molecule.
    
    Args:
        molecule (str): Molecule name (e.g., 'CO', 'H2', 'OH', 'SO', 'H2O')
    
    Returns:
        dict: Pattern info with 'regex', 'example', 'description', 'groups'
    """
    return TRANSITION_PATTERNS.get(molecule.upper())


def get_all_molecules():
    """Get list of all supported molecules."""
    return list(TRANSITION_PATTERNS.keys())


def get_pattern_info(molecule):
    """Get human-readable info about a molecule's transition format."""
    pattern = get_transition_pattern(molecule)
    if not pattern:
        return f"Molecule '{molecule}' not found"
    
    info = f"{molecule} transitions:\n"
    info += f"  Format: {pattern['description']}\n"
    info += f"  Example: {pattern['example']}\n"
    
    if pattern.get('regex'):
        info += f"  Regex: {pattern['regex']}\n"
    if pattern.get('match_type'):
        info += f"  Match type: {pattern['match_type']}\n"
    
    return info
