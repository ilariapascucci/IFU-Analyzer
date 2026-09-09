"""
Transition Patterns for Molecules

Describes how each molecule's transition string is formatted in HITRAN data, so
the GUI can filter lines by transition without any per-molecule code.

An entry has one of two match modes:
  match='regex'     -- 'regex' has named groups; a line matches when every field
                       the user filled equals its captured group.
  match='substring' -- a line matches when "<upper>-><lower>" is in the string.
'fields' drives the GUI prompts: name (regex group / 'upper'|'lower'), label,
type ('int' or 'str'), default. Entries without 'match'/'fields' are older,
unmigrated drafts (regex + 'groups'), not used by the filter yet.
"""

TRANSITION_PATTERNS = {
    'CO': {
        'match': 'regex',
        'regex': r'v=(?P<v_upper>\d+)->(?P<v_lower>\d+)',
        'example': 'v=1->0, J=56->57',
        'description': 'Vibrational band: upper v -> lower v. J is ignored.',
        'fields': [
            {'name': 'v_upper', 'label': 'Upper vibrational level (v)', 'type': 'int', 'default': 1},
            {'name': 'v_lower', 'label': 'Lower vibrational level (v)', 'type': 'int', 'default': 0},
        ],
    },
    'H2': {
        'match': 'regex',
        'regex': r'v=(?P<v_upper>\d+)->(?P<v_lower>\d+)',
        'example': 'v=1->0, J=18->20',
        'description': 'Vibrational band: upper v -> lower v. J is ignored.',
        'fields': [
            {'name': 'v_upper', 'label': 'Upper vibrational level (v)', 'type': 'int', 'default': 1},
            {'name': 'v_lower', 'label': 'Lower vibrational level (v)', 'type': 'int', 'default': 0},
        ],
    },
    'SO': {
        'regex': r'v=(V\d+(?:\+V\d+)*)?->GROUND',
        'example': 'v=V5->GROUND',
        'description': 'Vibrational transitions to ground state: single or combined vibrational modes',
        'groups': ['vibrational_modes'],
    },
    'OH': {
        'match': 'regex',
        'regex': r'v=X(?P<omega_upper>\d/\d)\s+(?P<v_upper>\d+)->X(?P<omega_lower>\d/\d)\s+(?P<v_lower>\d+)',
        'example': 'v=X3/2   8->X1/2   7, J=9->10',
        'description': 'X(2)Pi band: spin-orbit ladder Omega and v, upper -> lower. J is ignored.',
        'fields': [
            {'name': 'omega_upper', 'label': 'Upper spin-orbit ladder Omega (1/2 or 3/2)', 'type': 'str', 'default': '3/2'},
            {'name': 'v_upper', 'label': 'Upper vibrational level (v)', 'type': 'int', 'default': 1},
            {'name': 'omega_lower', 'label': 'Lower spin-orbit ladder Omega (1/2 or 3/2)', 'type': 'str', 'default': '3/2'},
            {'name': 'v_lower', 'label': 'Lower vibrational level (v)', 'type': 'int', 'default': 0},
        ],
    },
    'H2O': {
        'match': 'substring',
        'example': 'v=0 1 0->0 0 0',
        'description': 'Vibrational state string (v1 v2 v3), upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state (v1 v2 v3)', 'type': 'str', 'default': '0 1 0'},
            {'name': 'lower', 'label': 'Lower vib state (v1 v2 v3)', 'type': 'str', 'default': '0 0 0'},
        ],
    },
    'C2H2': {
        'match': 'substring',
        'example': 'v=000 2 0 2 0  g->000 0 1 0 1  u, J=44->45',
        'description': 'Vibrational state string (v1v2v3 v4 v5 l4 l5 + g/u), upper -> lower. Spacing and +/g/u are literal.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state (e.g. 000 0 1 0 1  u)', 'type': 'str', 'default': '000 0 1 0 1  u'},
            {'name': 'lower', 'label': 'Lower vib state (e.g. 000 0 0 0 0+ g)', 'type': 'str', 'default': '000 0 0 0 0+ g'},
        ],
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
        'match': 'substring',
        'example': 'v=0 1 1 01->0 0 0 01, J=14->13',
        'description': 'Vibrational state string (v1 v2 l2 v3 + Fermi rank), upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state (v1 v2 l2 v3r)', 'type': 'str', 'default': '0 1 1 01'},
            {'name': 'lower', 'label': 'Lower vib state (v1 v2 l2 v3r)', 'type': 'str', 'default': '0 0 0 01'},
        ],
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
        'match': 'substring',
        'example': 'v=1 0 0 0->0 1 1 1, J=30->29',
        'description': 'Vibrational state string (v1 v2 l2 v3), upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state (v1 v2 l2 v3)', 'type': 'str', 'default': '1 0 0 0'},
            {'name': 'lower', 'label': 'Lower vib state (v1 v2 l2 v3)', 'type': 'str', 'default': '0 0 0 0'},
        ],
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
    Return the transition pattern entry for a molecule, or None.

    Args:
        molecule (str): Molecule name, e.g. 'CO', 'OH', 'H2O'.

    Returns:
        pattern (dict): The entry (see the module docstring for its keys), or
            None if the molecule has no entry.
    """
    return TRANSITION_PATTERNS.get(molecule.upper())


def get_all_molecules():
    """
    List every molecule name that has a transition pattern entry.

    Args:
        None

    Returns:
        molecules (list): Molecule name strings, in dict order.
    """
    return list(TRANSITION_PATTERNS.keys())


def get_pattern_info(molecule):
    """
    Build a short human-readable summary of a molecule's transition format.

    Args:
        molecule (str): Molecule name, e.g. 'CO'.

    Returns:
        info (str): Multi-line summary, or a "not found" message.
    """
    pattern = get_transition_pattern(molecule)
    if not pattern:
        return f"Molecule '{molecule}' not found"

    info = f"{molecule} transitions:\n"
    info += f"  Format: {pattern['description']}\n"
    info += f"  Example: {pattern['example']}\n"
    info += f"  Match: {pattern.get('match', 'regex')}\n"
    if pattern.get('regex'):
        info += f"  Regex: {pattern['regex']}\n"
    return info
