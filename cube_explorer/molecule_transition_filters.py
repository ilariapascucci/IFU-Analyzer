"""
Molecule Transition Filtering Module

Provides generic transition filtering for HITRAN data across different molecules.
Uses regex patterns from transition_patterns.py to filter lines.
GUI calls this - no transition logic should be in gui.py
"""

import re
from PyQt5.QtWidgets import QInputDialog, QMessageBox, QLineEdit
from .transition_patterns import TRANSITION_PATTERNS


def get_user_transition_inputs(parent, molecule):
    """
    Dynamically prompt user for transition filter inputs based on molecule pattern.
    
    Args:
        parent: Parent QWidget for dialogs
        molecule (str): Molecule name (e.g., 'CO', 'H2', 'OH', 'SO', 'H2O')
    
    Returns:
        dict or None: Dictionary of captured groups and values, or None if user cancelled
    """
    pattern_info = TRANSITION_PATTERNS.get(molecule.upper())
    if not pattern_info:
        QMessageBox.warning(parent, "Error", f"Molecule '{molecule}' not in transition patterns.")
        return None
    
    user_inputs = {}
    
    # For H2O (substring matching), we need direct user input
    if molecule.upper() == 'H2O':
        upper_text, ok = QInputDialog.getText(
            parent,
            "H₂O Upper Transition Numbers",
            "Enter upper state (v Ka Kc):\n(e.g., 0 3 0)",
            QLineEdit.Normal,
            "0 3 0"
        )
        if not ok or not upper_text:
            return None
        
        lower_text, ok = QInputDialog.getText(
            parent,
            "H₂O Lower Transition Numbers",
            "Enter lower state (v Ka Kc):\n(e.g., 0 1 0)",
            QLineEdit.Normal,
            "0 1 0"
        )
        if not ok or not lower_text:
            return None
        
        user_inputs['upper'] = upper_text
        user_inputs['lower'] = lower_text
        return user_inputs
    
    # For regex-based molecules, prompt for each capture group
    groups = pattern_info.get('groups', [])
    for i, group_name in enumerate(groups):
        # Determine field type from group name
        if 'quantum' in group_name.lower() or any(c.isdigit() for c in group_name):
            # Numeric field
            value, ok = QInputDialog.getInt(
                parent,
                f"{molecule.upper()} Transition Filter",
                f"Enter {group_name}:\n(e.g., {pattern_info['example'].split()[i] if i < len(pattern_info['example'].split()) else '0'})",
                0, 0, 100, 1
            )
            if not ok:
                return None
            user_inputs[group_name] = value
        else:
            # String field
            value, ok = QInputDialog.getText(
                parent,
                f"{molecule.upper()} Transition Filter",
                f"Enter {group_name}:\n(e.g., {pattern_info['example'].split()[i] if i < len(pattern_info['example'].split()) else 'X'})",
                QLineEdit.Normal,
                ""
            )
            if not ok or not value:
                return None
            user_inputs[group_name] = value
    
    return user_inputs


def filter_hitran_by_transition(results, molecule, user_inputs):
    """
    Filter HITRAN results by transition using regex patterns from transition_patterns.py
    
    Args:
        results (list): HITRAN line data from fetch_hitran_data
        molecule (str): Molecule name (e.g., 'CO', 'H2', 'OH', 'SO', 'H2O')
        user_inputs (dict): User-provided transition parameters
    
    Returns:
        list: Filtered HITRAN results matching the transition criteria
    """
    pattern_info = TRANSITION_PATTERNS.get(molecule.upper())
    if not pattern_info:
        return results
    
    filtered_results = []
    
    # Handle H2O substring matching
    if molecule.upper() == 'H2O':
        upper = user_inputs.get('upper', '')
        lower = user_inputs.get('lower', '')
        search_pattern = f"{upper}->{lower}"
        
        for line in results:
            transition = line.get('transition', '')
            if transition and search_pattern in transition:
                filtered_results.append(line)
        return filtered_results
    
    # Handle regex-based molecules
    regex = pattern_info.get('regex')
    groups = pattern_info.get('groups', [])
    
    if not regex:
        return results
    
    for line in results:
        transition = line.get('transition', '')
        if not transition or '->' not in transition:
            continue
        
        try:
            match = re.search(regex, transition)
            if match:
                # Extract captured groups from regex
                captured_groups = match.groups()
                
                # Check if all user inputs match the captured groups
                all_match = True
                for i, group_name in enumerate(groups):
                    if i < len(captured_groups):
                        captured_value = captured_groups[i]
                        expected_value = user_inputs.get(group_name)
                        
                        # Convert to same type for comparison
                        try:
                            if isinstance(expected_value, int):
                                captured_value = int(captured_value)
                            else:
                                captured_value = str(captured_value)
                        except:
                            pass
                        
                        if captured_value != expected_value:
                            all_match = False
                            break
                
                if all_match:
                    filtered_results.append(line)
        except (ValueError, AttributeError):
            pass
    
    return filtered_results


def apply_transition_filter(parent, results, molecule):
    """
    Main entry point: prompt user for transition inputs and filter results.
    
    Args:
        parent: Parent QWidget for dialogs
        results (list): HITRAN line data
        molecule (str): Molecule name
    
    Returns:
        list or None: Filtered results, or None if no matches or user cancelled
    """
    # Get user inputs
    user_inputs = get_user_transition_inputs(parent, molecule)
    if user_inputs is None:
        return None
    
    # Filter results
    filtered_results = filter_hitran_by_transition(results, molecule, user_inputs)
    
    # Show error if no matches
    if not filtered_results:
        QMessageBox.information(
            parent,
            "No Matches",
            f"No {molecule} lines found matching the specified transition."
        )
        return None
    
    return filtered_results
