"""
Transition Patterns for Molecules

Describes how each molecule's transition string is formatted in HITRAN data, so
the GUI can filter lines by transition without any per-molecule code.

An entry has one of two match modes:
  match='regex'     -- 'regex' has named groups; a line matches when every field
                       the user filled equals its captured group.
  match='substring' -- a line matches when "<upper>-><lower>" is in the string.
'fields' drives the GUI prompts: name (regex group / 'upper'|'lower'), label,
type ('int' or 'str'), default. Every default was verified to return real matches
against HITRAN data in 1-30um. Molecules with no lines in that range (SF6, CF4,
O, S2, ...) have no entry, so the GUI just does not offer the filter for them.
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
    'HCN': {
        'match': 'substring',
        'example': 'v=1 0 0 0->0 1 1 1, J=30->29',
        'description': 'Vibrational state string (v1 v2 l2 v3), upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state (v1 v2 l2 v3)', 'type': 'str', 'default': '1 0 0 0'},
            {'name': 'lower', 'label': 'Lower vib state (v1 v2 l2 v3)', 'type': 'str', 'default': '0 0 0 0'},
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
    'CO2': {
        'match': 'substring',
        'example': 'v=0 1 1 01->0 0 0 01, J=14->13',
        'description': 'Vibrational state string (v1 v2 l2 v3 + Fermi rank), upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state (v1 v2 l2 v3r)', 'type': 'str', 'default': '0 1 1 01'},
            {'name': 'lower', 'label': 'Lower vib state (v1 v2 l2 v3r)', 'type': 'str', 'default': '0 0 0 01'},
        ],
    },
    'CS': {
        'match': 'regex',
        'regex': r'v=(?P<v_upper>\d+)->(?P<v_lower>\d+)',
        'example': 'v=2->1, J=97->98',
        'description': 'Vibrational band: upper v -> lower v. J is ignored.',
        'fields': [
            {'name': 'v_upper', 'label': 'Upper vibrational level (v)', 'type': 'int', 'default': 1},
            {'name': 'v_lower', 'label': 'Lower vibrational level (v)', 'type': 'int', 'default': 0},
        ],
    },
    'OCS': {
        'match': 'substring',
        'example': 'v=1 0 0 0->0 1 1 0, J=69->69',
        'description': 'Vibrational state string (v1 v2 l2 v3), upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state (v1 v2 l2 v3)', 'type': 'str', 'default': '1 0 0 0'},
            {'name': 'lower', 'label': 'Lower vib state (v1 v2 l2 v3)', 'type': 'str', 'default': '0 0 0 0'},
        ],
    },
    'SO2': {
        'match': 'substring',
        'example': 'v=0 1 0->0 0 0',
        'description': 'Vibrational state string (v1 v2 v3), upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state (v1 v2 v3)', 'type': 'str', 'default': '0 1 0'},
            {'name': 'lower', 'label': 'Lower vib state (v1 v2 v3)', 'type': 'str', 'default': '0 0 0'},
        ],
    },
    'C2H6': {
        'match': 'substring',
        'example': 'v=V9->GROUND, J=?->45',
        'description': 'Vibrational-mode string (V#, 2V#, V#+V#), upper -> lower (lower is often GROUND).',
        'fields': [
            {'name': 'upper', 'label': 'Upper mode string (e.g. V9 or V9+V4)', 'type': 'str', 'default': 'V9'},
            {'name': 'lower', 'label': 'Lower mode string (e.g. GROUND)', 'type': 'str', 'default': 'GROUND'},
        ],
    },
    'C4H2': {
        'match': 'substring',
        'example': 'v=000000011 f g->000000001 f u, J=-1->0',
        'description': 'Vibrational state string (packed digits + e/f parity + g/u symmetry), upper -> lower. Spacing is literal.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state (e.g. 000000011 f g)', 'type': 'str', 'default': '000000011 f g'},
            {'name': 'lower', 'label': 'Lower vib state (e.g. 000000001 f u)', 'type': 'str', 'default': '000000001 f u'},
        ],
    },
    'HC3N': {
        'match': 'substring',
        'example': 'v=0001000 0 0 0->0000010 0 1 0, J=86->87',
        'description': 'Vibrational state string (packed 7-digit + l-quanta), upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state (e.g. 0001000 0 0 0)', 'type': 'str', 'default': '0001000 0 0 0'},
            {'name': 'lower', 'label': 'Lower vib state (e.g. 0000010 0 1 0)', 'type': 'str', 'default': '0000010 0 1 0'},
        ],
    },
    'CH4': {
        'match': 'substring',
        'example': 'v=0 0 0 1 1F2->0 0 0 0 1A1, J=?->12',
        'description': 'Vibrational state string (5 quanta + symmetry label), upper -> lower. About 35% of CH4 lines are unassigned and never match.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state (e.g. 0 0 0 1 1F2)', 'type': 'str', 'default': '0 0 0 1 1F2'},
            {'name': 'lower', 'label': 'Lower vib state (e.g. 0 0 0 0 1A1)', 'type': 'str', 'default': '0 0 0 0 1A1'},
        ],
    },
    'NH3': {
        'match': 'substring',
        'example': "v=0100 00 0 A1'->0000 00 0 A1', J=?->17",
        'description': 'Vibrational state string (packed quanta + A/E symmetry with prime marks), upper -> lower. Many NH3 lines are unassigned and never match.',
        'fields': [
            {'name': 'upper', 'label': "Upper vib state (e.g. 0100 00 0 A1')", 'type': 'str', 'default': "0100 00 0 A1'"},
            {'name': 'lower', 'label': "Lower vib state (e.g. 0000 00 0 A1')", 'type': 'str', 'default': "0000 00 0 A1'"},
        ],
    },
    'CH3': {
        'match': 'substring',
        'example': 'v=0 1 0   0  A2"->0 0 0   0  A1\', J=?->13',
        'description': 'Vibrational state string (4 quanta + symmetry, wide spacing), upper -> lower. Spacing is literal.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state (e.g. 0 1 0   0  A2")', 'type': 'str', 'default': '0 1 0   0  A2"'},
            {'name': 'lower', 'label': "Lower vib state (e.g. 0 0 0   0  A1')", 'type': 'str', 'default': "0 0 0   0  A1'"},
        ],
    },
    'H3p': {
        'match': 'substring',
        'example': 'v=0 1 1 E\'->0 0 0 E"',
        'description': 'Vibrational state string (v1 v2 l2 + E/A symmetry), upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': "Upper vib state (e.g. 0 1 1 E')", 'type': 'str', 'default': "0 1 1 E'"},
            {'name': 'lower', 'label': 'Lower vib state (e.g. 0 0 0 E")', 'type': 'str', 'default': '0 0 0 E"'},
        ],
    },
    'HF': {
        'match': 'regex',
        'regex': r'v=(?P<v_upper>\d+)->(?P<v_lower>\d+)',
        'example': 'v=10->10, J=13->12',
        'description': 'Vibrational band: upper v -> lower v. J is ignored.',
        'fields': [
            {'name': 'v_upper', 'label': 'Upper vibrational level (v)', 'type': 'int', 'default': 4},
            {'name': 'v_lower', 'label': 'Lower vibrational level (v)', 'type': 'int', 'default': 3},
        ],
    },
    'HCl': {
        'match': 'regex',
        'regex': r'v=(?P<v_upper>\d+)->(?P<v_lower>\d+)',
        'example': 'v=1->1, J=17->16',
        'description': 'Vibrational band: upper v -> lower v. J is ignored.',
        'fields': [
            {'name': 'v_upper', 'label': 'Upper vibrational level (v)', 'type': 'int', 'default': 13},
            {'name': 'v_lower', 'label': 'Lower vibrational level (v)', 'type': 'int', 'default': 12},
        ],
    },
    'HBr': {
        'match': 'regex',
        'regex': r'v=(?P<v_upper>\d+)->(?P<v_lower>\d+)',
        'example': 'v=7->7, J=27->26',
        'description': 'Vibrational band: upper v -> lower v. J is ignored.',
        'fields': [
            {'name': 'v_upper', 'label': 'Upper vibrational level (v)', 'type': 'int', 'default': 1},
            {'name': 'v_lower', 'label': 'Lower vibrational level (v)', 'type': 'int', 'default': 0},
        ],
    },
    'HI': {
        'match': 'regex',
        'regex': r'v=(?P<v_upper>\d+)->(?P<v_lower>\d+)',
        'example': 'v=5->5, J=33->32',
        'description': 'Vibrational band: upper v -> lower v. J is ignored.',
        'fields': [
            {'name': 'v_upper', 'label': 'Upper vibrational level (v)', 'type': 'int', 'default': 1},
            {'name': 'v_lower', 'label': 'Lower vibrational level (v)', 'type': 'int', 'default': 0},
        ],
    },
    'N2': {
        'match': 'regex',
        'regex': r'v=(?P<v_upper>\d+)->(?P<v_lower>\d+)',
        'example': 'v=0->0, J=43->41',
        'description': 'Vibrational band: upper v -> lower v. J is ignored.',
        'fields': [
            {'name': 'v_upper', 'label': 'Upper vibrational level (v)', 'type': 'int', 'default': 1},
            {'name': 'v_lower', 'label': 'Lower vibrational level (v)', 'type': 'int', 'default': 0},
        ],
    },
    'NOp': {
        'match': 'regex',
        'regex': r'v=(?P<v_upper>\d+)->(?P<v_lower>\d+)',
        'example': 'v=6->5, J=99->100',
        'description': 'Vibrational band: upper v -> lower v. J is ignored.',
        'fields': [
            {'name': 'v_upper', 'label': 'Upper vibrational level (v)', 'type': 'int', 'default': 6},
            {'name': 'v_lower', 'label': 'Lower vibrational level (v)', 'type': 'int', 'default': 5},
        ],
    },
    'NO': {
        'match': 'regex',
        'regex': r'v=X(?P<omega_upper>\d/\d)\s+(?P<v_upper>\d+)->X(?P<omega_lower>\d/\d)\s+(?P<v_lower>\d+)',
        'example': 'v=X3/2   6->X1/2   6, J=106->105',
        'description': 'X(2)Pi band: spin-orbit ladder Omega and v, upper -> lower. J is ignored.',
        'fields': [
            {'name': 'omega_upper', 'label': 'Upper spin-orbit ladder Omega (1/2 or 3/2)', 'type': 'str', 'default': '3/2'},
            {'name': 'v_upper', 'label': 'Upper vibrational level (v)', 'type': 'int', 'default': 6},
            {'name': 'omega_lower', 'label': 'Lower spin-orbit ladder Omega (1/2 or 3/2)', 'type': 'str', 'default': '1/2'},
            {'name': 'v_lower', 'label': 'Lower vibrational level (v)', 'type': 'int', 'default': 6},
        ],
    },
    'ClO': {
        'match': 'regex',
        'regex': r'v=X(?P<omega_upper>\d/\d)\s+(?P<v_upper>\d+)->X(?P<omega_lower>\d/\d)\s+(?P<v_lower>\d+)',
        'example': 'v=X3/2   1->X1/2   0, J=42->43',
        'description': 'X(2)Pi band: spin-orbit ladder Omega and v, upper -> lower. J is ignored.',
        'fields': [
            {'name': 'omega_upper', 'label': 'Upper spin-orbit ladder Omega (1/2 or 3/2)', 'type': 'str', 'default': '3/2'},
            {'name': 'v_upper', 'label': 'Upper vibrational level (v)', 'type': 'int', 'default': 1},
            {'name': 'omega_lower', 'label': 'Lower spin-orbit ladder Omega (1/2 or 3/2)', 'type': 'str', 'default': '1/2'},
            {'name': 'v_lower', 'label': 'Lower vibrational level (v)', 'type': 'int', 'default': 0},
        ],
    },
    'O2': {
        'match': 'regex',
        'regex': r'v=(?P<state_upper>[A-Za-z])\s+(?P<v_upper>\d+)->(?P<state_lower>[A-Za-z])\s+(?P<v_lower>\d+)',
        'example': 'v=X      1->X      0, J=37->39',
        'description': 'Electronic-state band: state letter and v, upper -> lower. J is ignored.',
        'fields': [
            {'name': 'state_upper', 'label': 'Upper electronic state (e.g. X, a, b)', 'type': 'str', 'default': 'X'},
            {'name': 'v_upper', 'label': 'Upper vibrational level (v)', 'type': 'int', 'default': 1},
            {'name': 'state_lower', 'label': 'Lower electronic state (e.g. X, a, b)', 'type': 'str', 'default': 'X'},
            {'name': 'v_lower', 'label': 'Lower vibrational level (v)', 'type': 'int', 'default': 0},
        ],
    },
    'SO': {
        'match': 'regex',
        'regex': r'v=(?P<state_upper>[A-Za-z])\s+(?P<v_upper>\d+)->(?P<state_lower>[A-Za-z])\s+(?P<v_lower>\d+)',
        'example': 'v=a      1->X      6, J=16->16',
        'description': 'Electronic-state band: state letter and v, upper -> lower. J is ignored.',
        'fields': [
            {'name': 'state_upper', 'label': 'Upper electronic state (e.g. X, a, b)', 'type': 'str', 'default': 'X'},
            {'name': 'v_upper', 'label': 'Upper vibrational level (v)', 'type': 'int', 'default': 6},
            {'name': 'state_lower', 'label': 'Lower electronic state (e.g. X, a, b)', 'type': 'str', 'default': 'X'},
            {'name': 'v_lower', 'label': 'Lower vibrational level (v)', 'type': 'int', 'default': 5},
        ],
    },
    'O3': {
        'match': 'substring',
        'example': 'v=0 0 1->0 0 0',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': '0 0 1'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': '0 0 0'},
        ],
    },
    'N2O': {
        'match': 'substring',
        'example': 'v=0 1 1 1->0 1 1 0, J=103->104',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': '0 1 1 1'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': '0 1 1 0'},
        ],
    },
    'NO2': {
        'match': 'substring',
        'example': 'v=0 1 0->0 0 0',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': '0 1 0'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': '0 0 0'},
        ],
    },
    'HNO3': {
        'match': 'substring',
        'example': 'v=V5+V6->V6',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': 'V5+V6'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': 'V6'},
        ],
    },
    'H2CO': {
        'match': 'substring',
        'example': 'v=0 0 0 0 0 2->0 0 0 0 0 0',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': '0 0 0 0 0 2'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': '0 0 0 0 0 0'},
        ],
    },
    'HOCl': {
        'match': 'substring',
        'example': 'v=1 0 0->0 0 0',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': '1 0 0'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': '0 0 0'},
        ],
    },
    'H2O2': {
        'match': 'substring',
        'example': 'v=0 0 022 0 0->0 0 004 0 0',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': '0 0 022 0 0'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': '0 0 004 0 0'},
        ],
    },
    'H2S': {
        'match': 'substring',
        'example': 'v=1 1 0->0 0 0',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': '1 1 0'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': '0 0 0'},
        ],
    },
    'HO2': {
        'match': 'substring',
        'example': 'v=0 0 0->0 0 0',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': '0 0 0'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': '0 0 0'},
        ],
    },
    'COF2': {
        'match': 'substring',
        'example': 'v=0 0 0 0 0 1->0 0 0 0 0 0',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': '0 0 0 0 0 1'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': '0 0 0 0 0 0'},
        ],
    },
    'COCl2': {
        'match': 'substring',
        'example': 'v=0 0 0 0 1 0->0 0 0 0 0 0',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': '0 0 0 0 1 0'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': '0 0 0 0 0 0'},
        ],
    },
    'CS2': {
        'match': 'substring',
        'example': 'v=0 1 1 1->0 1 1 0, J=199->200',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': '0 1 1 1'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': '0 1 1 0'},
        ],
    },
    'PH3': {
        'match': 'substring',
        'example': 'v=0 0 1 1->0 0 0 1, J=?->15',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': '0 0 1 1'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': '0 0 0 1'},
        ],
    },
    'SO3': {
        'match': 'substring',
        'example': "v=0 0 0 0 1 1E'->0 0 0 0 0 0A1', J=?->84",
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': "0 0 0 0 1 1E'"},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': "0 0 0 0 0 0A1'"},
        ],
    },
    'GeH4': {
        'match': 'substring',
        'example': 'v=0 0 1 0 1F2->0 0 0 0 1A1, J=?->30',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': '0 0 1 0 1F2'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': '0 0 0 0 1A1'},
        ],
    },
    'HCOOH': {
        'match': 'substring',
        'example': 'v=V7->GROUND',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': 'V7'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': 'GROUND'},
        ],
    },
    'C2H4': {
        'match': 'substring',
        'example': 'v=V7->GROUND',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': 'V7'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': 'GROUND'},
        ],
    },
    'CH3OH': {
        'match': 'substring',
        'example': 'v=V8->GROUND, J=?->35',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': 'V8'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': 'GROUND'},
        ],
    },
    'CH3Cl': {
        'match': 'substring',
        'example': 'v=V5+V6->GROUND, J=?->37',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': 'V5+V6'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': 'GROUND'},
        ],
    },
    'CH3Br': {
        'match': 'substring',
        'example': 'v=V6->GROUND, J=?->40',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': 'V6'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': 'GROUND'},
        ],
    },
    'CH3CN': {
        'match': 'substring',
        'example': 'v=2V8->V8, J=?->42',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': '2V8'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': 'V8'},
        ],
    },
    'CH3F': {
        'match': 'substring',
        'example': 'v=V6+V3->V6, J=?->36',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': 'V6+V3'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': 'V6'},
        ],
    },
    'CH3I': {
        'match': 'substring',
        'example': 'v=V4->GROUND, J=?->25',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': 'V4'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': 'GROUND'},
        ],
    },
    'COFCl': {
        'match': 'substring',
        'example': 'v=V2->GROUND',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': 'V2'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': 'GROUND'},
        ],
    },
    'ClNO2': {
        'match': 'substring',
        'example': 'v=V3->GROUND',
        'description': 'Vibrational state string, upper -> lower.',
        'fields': [
            {'name': 'upper', 'label': 'Upper vib state', 'type': 'str', 'default': 'V3'},
            {'name': 'lower', 'label': 'Lower vib state', 'type': 'str', 'default': 'GROUND'},
        ],
    },
}


_PATTERNS_CI = {name.upper(): entry for name, entry in TRANSITION_PATTERNS.items()}


def get_transition_pattern(molecule):
    """
    Return the transition pattern entry for a molecule, or None.

    Args:
        molecule (str): Molecule name, any casing (HCl, NOp, H3p all resolve).

    Returns:
        pattern (dict): The entry (see the module docstring for its keys), or
            None if the molecule has no entry.
    """
    return _PATTERNS_CI.get(molecule.upper())
