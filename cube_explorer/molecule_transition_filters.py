"""
Molecule Transition Filtering

Prompts the user for a transition and filters HITRAN line rows by it, driven
entirely by the entries in transition_patterns.py. The GUI calls
apply_transition_filter(); no per-molecule transition code lives in gui.py.
"""

import re
from PyQt5.QtWidgets import QInputDialog, QMessageBox, QLineEdit
from .transition_patterns import get_transition_pattern as _pattern_for


def _field_names(pattern_info):
    """
    List the field / capture-group names for a pattern entry, in order.

    Args:
        pattern_info (dict): One TRANSITION_PATTERNS entry.

    Returns:
        names (list): Field names from 'fields', or the legacy 'groups' list.
    """
    if pattern_info.get('fields'):
        return [f['name'] for f in pattern_info['fields']]
    return pattern_info.get('groups', [])


def get_user_transition_inputs(parent, molecule):
    """
    Prompt the user for one transition, using the molecule's 'fields' list.

    Args:
        parent (QWidget): Parent for the dialogs.
        molecule (str): Molecule name, per TRANSITION_PATTERNS.

    Returns:
        user_inputs (dict): {field_name: value}, or None if the user cancelled
            or the molecule has no pattern entry.
    """
    pattern_info = _pattern_for(molecule)
    if not pattern_info:
        QMessageBox.warning(parent, "Error", f"'{molecule}' has no transition pattern.")
        return None

    fields = pattern_info.get('fields')
    if not fields:
        # Legacy entry (regex + 'groups', no 'fields'): guess field type by name.
        fields = [
            {'name': g, 'label': g,
             'type': 'int' if ('quantum' in g.lower() or any(c.isdigit() for c in g)) else 'str',
             'default': 0 if ('quantum' in g.lower() or any(c.isdigit() for c in g)) else ''}
            for g in pattern_info.get('groups', [])
        ]

    title = f"{molecule.upper()} transition"
    user_inputs = {}
    for field in fields:
        prompt = f"{field['label']}\n(e.g. {pattern_info['example']})"
        if field.get('type') == 'int':
            value, ok = QInputDialog.getInt(parent, title, prompt, int(field.get('default', 0)), 0, 200, 1)
        else:
            value, ok = QInputDialog.getText(parent, title, prompt, QLineEdit.Normal, str(field.get('default', '')))
        if not ok:
            return None
        user_inputs[field['name']] = value
    return user_inputs


def filter_hitran_by_transition(results, molecule, user_inputs):
    """
    Keep only the HITRAN rows whose transition string matches user_inputs.

    Args:
        results (list): Line-data rows from fetch_hitran_data().
        molecule (str): Molecule name, per TRANSITION_PATTERNS.
        user_inputs (dict): {field_name: value} from get_user_transition_inputs().

    Returns:
        filtered (list): The rows that match. All rows if the molecule has no
            usable pattern.
    """
    pattern_info = _pattern_for(molecule)
    if not pattern_info:
        return results

    if pattern_info.get('match') == 'substring':
        needle = f"{user_inputs.get('upper', '')}->{user_inputs.get('lower', '')}"
        return [r for r in results if needle in (r.get('transition') or '')]

    regex = pattern_info.get('regex')
    if not regex:
        return results
    pattern = re.compile(regex)
    names = _field_names(pattern_info)

    filtered = []
    for row in results:
        transition = row.get('transition') or ''
        match = pattern.search(transition)
        if not match:
            continue
        by_name = match.groupdict()
        positional = match.groups()
        for i, name in enumerate(names):
            expected = user_inputs.get(name)
            if expected is None or expected == '':
                continue
            captured = by_name.get(name) if name in by_name else (positional[i] if i < len(positional) else None)
            if str(captured).strip() != str(expected).strip():
                break
        else:
            filtered.append(row)
    return filtered


def apply_transition_filter(parent, results, molecule):
    """
    Prompt for a transition, then return the rows that match it.

    Args:
        parent (QWidget): Parent for the dialogs.
        results (list): Line-data rows from fetch_hitran_data().
        molecule (str): Molecule name, per TRANSITION_PATTERNS.

    Returns:
        filtered (list): Matching rows, or None if the user cancelled or nothing
            matched (a dialog explains the empty case).
    """
    user_inputs = get_user_transition_inputs(parent, molecule)
    if user_inputs is None:
        return None

    filtered = filter_hitran_by_transition(results, molecule, user_inputs)
    if not filtered:
        QMessageBox.information(parent, "No Matches",
                               f"No {molecule} lines match that transition.")
        return None
    return filtered
