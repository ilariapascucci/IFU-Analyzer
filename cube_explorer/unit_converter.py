"""
Unit Conversion Utilities

Handles conversion between different wavelength and flux units
commonly used in spectroscopic data.
"""

import numpy as np


def convert_wavelength(value, from_unit, to_unit):
    """
    Convert wavelength between different units.
    
    Args:
        value: Wavelength value or array to convert
        from_unit: Source unit (Å, µm, nm, etc.)
        to_unit: Target unit (Å, µm, nm, etc.)
    
    Returns:
        Converted wavelength value(s)
    
    Supported units:
        - Å (Angstrom)
        - µm (micron)
        - nm (nanometer)
        - m (meter)
    """
    # Normalize unit strings
    from_unit = _normalize_unit(from_unit)
    to_unit = _normalize_unit(to_unit)
    
    if from_unit == to_unit:
        return value
    
    # Convert to meters first (base unit)
    to_meters = {
        'angstrom': 1e-10,
        'nm': 1e-9,
        'micron': 1e-6,
        'um': 1e-6,
        'm': 1.0
    }
    
    if from_unit not in to_meters:
        raise ValueError(f"Unsupported source unit: {from_unit}")
    if to_unit not in to_meters:
        raise ValueError(f"Unsupported target unit: {to_unit}")
    
    # Convert: value -> meters -> target
    value_in_meters = value * to_meters[from_unit]
    converted = value_in_meters / to_meters[to_unit]
    
    return converted


def _normalize_unit(unit):
    """Normalize unit string to standard form."""
    unit = str(unit).lower().strip()
    
    # Map various representations to standard names
    unit_map = {
        'å': 'angstrom',
        'a': 'angstrom',
        'angstrom': 'angstrom',
        'angstroms': 'angstrom',
        'nm': 'nm',
        'nanometer': 'nm',
        'nanometers': 'nm',
        'µm': 'micron',
        'um': 'micron',
        'micron': 'micron',
        'microns': 'micron',
        'micrometer': 'micron',
        'micrometers': 'micron',
        'm': 'm',
        'meter': 'm',
        'meters': 'm'
    }
    
    return unit_map.get(unit, unit)


def get_conversion_factor(from_unit, to_unit):
    """
    Get the multiplicative conversion factor between units.
    
    Args:
        from_unit: Source unit
        to_unit: Target unit
    
    Returns:
        float: Multiplication factor (value_in_to_unit = value_in_from_unit * factor)
    """
    return convert_wavelength(1.0, from_unit, to_unit)


def convert_wavelength_array(wavelengths, from_unit, to_unit):
    """
    Convert an array of wavelengths.
    
    Args:
        wavelengths: numpy array or list of wavelength values
        from_unit: Source unit
        to_unit: Target unit
    
    Returns:
        numpy array of converted wavelengths
    """
    wavelengths = np.asarray(wavelengths)
    return convert_wavelength(wavelengths, from_unit, to_unit)


def are_units_compatible(unit1, unit2):
    """
    Check if two units are compatible (both wavelength or both velocity).
    
    Args:
        unit1: First unit
        unit2: Second unit
    
    Returns:
        bool: True if units can be converted between each other
    """
    wavelength_units = {'angstrom', 'nm', 'micron', 'um', 'm'}
    velocity_units = {'km/s', 'km s-1', 'm/s', 'm s-1'}
    
    norm1 = _normalize_unit(unit1)
    norm2 = _normalize_unit(unit2)
    
    # Check if both are wavelength units
    if norm1 in wavelength_units and norm2 in wavelength_units:
        return True
    
    # Check if both are velocity units
    # (Note: velocity conversion is more complex, requires rest wavelength)
    if norm1 in velocity_units and norm2 in velocity_units:
        return True
    
    return False


def format_wavelength_for_display(value, unit):
    """Format wavelength value for display with appropriate precision."""
    norm_unit = _normalize_unit(unit)
    
    if norm_unit == 'angstrom':
        return f"{value:.1f} Å"
    elif norm_unit in ['micron', 'um']:
        return f"{value:.4f} µm"
    elif norm_unit == 'nm':
        return f"{value:.2f} nm"
    elif norm_unit in ['km/s', 'km s-1']:
        return f"{value:.1f} km/s"
    else:
        return f"{value:.4f} {unit}"


def clean_flux_unit(unit_str):
    """
    Clean and format flux units for better readability.
    Converts verbose unit strings to compact symbol notation.
    
    Examples:
        '10**(-20)*erg/s/cm**2/Angstrom' -> '10⁻²⁰ erg s⁻¹ cm⁻² Å⁻¹'
        'Jy/beam' -> 'Jy beam⁻¹'
        'MJy/sr' -> 'MJy sr⁻¹'
    """
    if not unit_str:
        return 'flux units'
    
    # Handle common FITS unit formats
    replacements = [
        # Power notation
        ('**(-20)', '⁻²⁰'),
        ('**(-10)', '⁻¹⁰'),
        ('**(-2)', '⁻²'),
        ('**(-1)', '⁻¹'),
        ('**(2)', '²'),
        ('**(3)', '³'),
        ('**2', '²'),
        ('**3', '³'),
        
        # Common words to symbols
        ('Angstrom', 'Å'),
        ('angstrom', 'Å'),
        
        # Remove multiplication signs
        ('*', ' '),
        
        # Convert slashes to superscript -1
        # Do this carefully to avoid double conversion
    ]
    
    result = str(unit_str)
    for old, new in replacements:
        result = result.replace(old, new)
    
    # Handle division notation: convert /x to x⁻¹
    # Split by '/' and process
    parts = result.split('/')
    if len(parts) > 1:
        # First part stays as is
        formatted = parts[0]
        # Remaining parts get ⁻¹
        for part in parts[1:]:
            part = part.strip()
            if part:
                # Check if already has superscript
                if '⁻' not in part and '²' not in part and '³' not in part:
                    formatted += f" {part}⁻¹"
                else:
                    formatted += f" {part}"
        result = formatted
    
    # Clean up extra spaces
    result = ' '.join(result.split())
    
    return result


def clean_wave_unit(unit_str):
    """
    Clean and format wavelength/velocity units.
    
    Examples:
        'Angstrom' -> 'Å'
        'micron' -> 'µm'
    """
    if not unit_str:
        return ''
    
    unit_map = {
        'Angstrom': 'Å',
        'angstrom': 'Å',
        'micron': 'µm',
        'um': 'µm',
        'km/s': 'km/s',
        'km s-1': 'km/s',
        'nm': 'nm',
        'm/s': 'm/s'
    }
    
    return unit_map.get(unit_str.strip(), unit_str)
