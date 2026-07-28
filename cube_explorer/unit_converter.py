"""
Unit Conversion Utilities

Handles conversion between different wavelength and flux units
commonly used in spectroscopic data. Leverages astropy.units for accurate,
dimensionally-aware conversions while maintaining backward compatibility with
string-based units.
"""

import numpy as np
import astropy.units as u
from astropy import constants as const
from astropy.units import Quantity


def _string_to_astropy_unit(unit_str):
    """
    Convert a string unit representation to astropy unit object.
    
    Args:
        unit_str: String like 'Å', 'µm', 'nm', 'km/s', etc.
    
    Returns:
        astropy.units.Unit object, or None if unrecognized
    """
    norm_unit = _normalize_unit(unit_str)
    
    unit_map = {
        'angstrom': u.AA,
        'nm': u.nm,
        'micron': u.um,
        'um': u.um,
        'm': u.m,
        'km/s': u.km / u.s,
        'km s-1': u.km / u.s,
        'm/s': u.m / u.s,
        'm s-1': u.m / u.s,
    }
    
    return unit_map.get(norm_unit, None)


def convert_wavelength(value, from_unit, to_unit):
    """
    Convert wavelength between different units.
    
    Supports both raw numeric values and astropy Quantity objects.
    If a Quantity is provided, its intrinsic unit is used.
    
    Args:
        value: Wavelength value, array, or astropy Quantity to convert
        from_unit: Source unit (str or astropy Unit). Ignored if value is Quantity.
        to_unit: Target unit (str or astropy Unit)
    
    Returns:
        Converted wavelength value (same type as input: Quantity or float/array)
    
    Supported units:
        - Å (Angstrom)
        - µm (micron)
        - nm (nanometer)
        - m (meter)
        - km/s (km per second)
        - m/s (meters per second)
    
    Examples:
        >>> convert_wavelength(1000, 'Å', 'µm')
        0.1
        >>> from astropy.units import Quantity
        >>> q = Quantity(1000, 'AA')
        >>> convert_wavelength(q, None, 'µm')
        <Quantity 0.1 um>
    """
    # Handle Quantity objects
    if isinstance(value, Quantity):
        # Extract the unit from the Quantity
        from_unit_obj = value.unit
        to_unit_obj = _string_to_astropy_unit(to_unit)
        
        if to_unit_obj is None:
            # Fallback to manual conversion
            to_unit = _normalize_unit(to_unit)
            to_meters = {
                'angstrom': 1e-10,
                'nm': 1e-9,
                'micron': 1e-6,
                'um': 1e-6,
                'm': 1.0
            }
            if to_unit not in to_meters:
                raise ValueError(f"Unsupported target unit: {to_unit}")
            value_in_meters = value.to(u.m).value
            return value_in_meters / to_meters[to_unit]
        
        # Use astropy for conversion
        try:
            return value.to(to_unit_obj)
        except Exception:
            # If conversion fails, fall back to manual method
            pass
    
    # Manual conversion for raw values (backward compatibility)
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
    
    Supports both raw numeric arrays and astropy Quantity arrays.
    
    Args:
        wavelengths: numpy array, list, or astropy Quantity of wavelength values
        from_unit: Source unit (str or astropy Unit). Ignored if wavelengths is Quantity.
        to_unit: Target unit (str or astropy Unit)
    
    Returns:
        Converted wavelength array (same type as input)
    
    Examples:
        >>> import numpy as np
        >>> vals = np.array([1000, 2000, 3000])
        >>> convert_wavelength_array(vals, 'Å', 'µm')
        array([0.1, 0.2, 0.3])
        >>> from astropy.units import Quantity
        >>> q = Quantity([1000, 2000, 3000], 'AA')
        >>> convert_wavelength_array(q, None, 'µm')
        <Quantity [0.1, 0.2, 0.3] um>
    """
    # Use convert_wavelength which already handles Quantity objects
    return convert_wavelength(wavelengths, from_unit, to_unit)


def are_units_compatible(unit1, unit2):
    """
    Check if two units are compatible (both wavelength or both velocity).
    
    Supports both string units and astropy Unit/Quantity objects.
    
    Args:
        unit1: First unit (str, astropy Unit, or Quantity)
        unit2: Second unit (str, astropy Unit, or Quantity)
    
    Returns:
        bool: True if units can be converted between each other
    
    Examples:
        >>> are_units_compatible('Å', 'µm')
        True
        >>> are_units_compatible('km/s', 'Å')
        False
        >>> from astropy.units import Quantity
        >>> q = Quantity(1000, 'AA')
        >>> are_units_compatible(q, 'µm')
        True
    """
    # Extract unit from Quantity objects
    if isinstance(unit1, Quantity):
        unit1 = unit1.unit
    if isinstance(unit2, Quantity):
        unit2 = unit2.unit
    
    # Convert astropy Unit objects to strings
    if hasattr(unit1, 'to_string'):
        unit1 = unit1.to_string()
    if hasattr(unit2, 'to_string'):
        unit2 = unit2.to_string()
    
    wavelength_units = {'angstrom', 'nm', 'micron', 'um', 'm', 'aa'}
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


def format_wavelength_for_display(value, unit=None):
    """
    Format wavelength value for display with appropriate precision.
    
    Supports both raw numeric values and astropy Quantity objects.
    If a Quantity is provided, its unit is used automatically.
    
    Args:
        value: Wavelength value or astropy Quantity
        unit: Unit string (str or astropy Unit). Ignored if value is Quantity.
    
    Returns:
        str: Formatted string with value and unit
    
    Examples:
        >>> format_wavelength_for_display(1234.5678, 'Å')
        '1234.6 Å'
        >>> from astropy.units import Quantity
        >>> q = Quantity(1.234, 'um')
        >>> format_wavelength_for_display(q)
        '1.2340 µm'
    """
    # Handle Quantity objects
    if isinstance(value, Quantity):
        unit_str = value.unit.to_string()
        val = value.value
        # Convert unit string to format we recognize
        if 'AA' in unit_str or 'angstrom' in unit_str.lower():
            return f"{val:.1f} Å"
        elif 'um' in unit_str or 'micron' in unit_str.lower():
            return f"{val:.4f} µm"
        elif 'nm' in unit_str or 'nanometer' in unit_str.lower():
            return f"{val:.2f} nm"
        elif 'km' in unit_str and 's' in unit_str:
            return f"{val:.1f} km/s"
        elif 'm' in unit_str and 's' in unit_str:
            return f"{val:.2f} m/s"
        else:
            return f"{val:.4f} {unit_str}"
    
    # Handle raw numeric values
    if unit is None:
        return f"{value:.4f}"
    
    norm_unit = _normalize_unit(unit)
    
    if norm_unit == 'angstrom':
        return f"{value:.1f} Å"
    elif norm_unit in ['micron', 'um']:
        return f"{value:.4f} µm"
    elif norm_unit == 'nm':
        return f"{value:.2f} nm"
    elif norm_unit in ['km/s', 'km s-1']:
        return f"{value:.1f} km/s"
    elif norm_unit in ['m/s', 'm s-1']:
        return f"{value:.2f} m/s"
    else:
        return f"{value:.4f} {unit}"


CGS_FLUX = u.erg / u.s / u.cm ** 2


def pixel_solid_angle(header):
    """
    Solid angle subtended by one spatial pixel, in steradians.

    Args:
        header: FITS header with CDELT1 and CDELT2 in degrees

    Returns:
        astropy Quantity in steradians, or None if the header lacks the pixel scales
    """
    if header is None:
        return None
    cdelt1 = header.get('CDELT1', None)
    cdelt2 = header.get('CDELT2', None)
    if cdelt1 is None or cdelt2 is None:
        return None
    try:
        area_deg2 = abs(float(cdelt1)) * abs(float(cdelt2)) * u.deg ** 2
        return area_deg2.to(u.sr)
    except Exception:
        return None


def _parse_flux_unit(flux_unit):
    """Parse a BUNIT string into an astropy unit, or None. Handles MUSE's 10**(-20) prefix."""
    if not flux_unit:
        return None
    try:
        return u.Unit(str(flux_unit), parse_strict='silent')
    except Exception:
        return None


def _reduce_to_target(quantity, target, wavelength, wave_astropy_unit, header):
    """
    Apply instrument corrections until `quantity` matches `target`, or give up.

    Stages, checked for dimensional equivalence after each:

    1. **As-is** - flux densities already per unit wavelength (MUSE's erg/s/cm²/Å)
       reduce with no correction at all.
    2. **× pixel solid angle** - removes the per-steradian of a surface brightness.
    3. **× c/λ²** - converts a per-frequency quantity to per-wavelength.

    Because each stage is checked with ``is_equivalent``, units that cannot reach the
    target (Jy/beam, which needs a beam area rather than a pixel area) fall out on their
    own without a hand-maintained list of what is supported.
    """
    if quantity.unit.is_equivalent(target):
        return float(quantity.to(target).value)

    omega = pixel_solid_angle(header)
    if omega is None or wavelength is None or wave_astropy_unit is None:
        return None

    quantity = quantity * omega
    if quantity.unit.is_equivalent(target):
        return float(quantity.to(target).value)

    quantity = quantity * (const.c / (wavelength * wave_astropy_unit) ** 2)
    if quantity.unit.is_equivalent(target):
        return float(quantity.to(target).value)

    return None


def integrated_flux_to_cgs(value, flux_unit, wavelength=None, header=None, wave_unit=None):
    """
    Convert an integrated line flux to erg/s/cm².

    This is the single conversion used for Gaussian areas, CO fluxes and upper limits.
    ``value`` is the already-integrated quantity, carrying units of
    ``flux_unit × wave_unit`` (e.g. a Gaussian's ``A·σ·√(2π)``).

    Two instrument families reach erg/s/cm² by different routes, dispatched
    dimensionally rather than by name:

    - **erg/s/cm²/Å (MUSE)** - already a flux per unit wavelength, so the Å cancels and
      no correction is needed.
    - **MJy/sr (MIRI, NIRSpec)** - a surface brightness per unit frequency, so it needs
      the pixel solid angle and the ``c/λ²`` jacobian.

    Anything else (Jy/beam for ALMA) returns None, because a beam area is not a pixel
    area and the two are not interchangeable.

    Args:
        value: Integrated flux in ``flux_unit × wave_unit``
        flux_unit: Flux unit string, typically the BUNIT header keyword
        wavelength: Line centre, in ``wave_unit``. Required for MJy/sr, unused for MUSE.
        header: FITS header supplying CDELT1/CDELT2. Required for MJy/sr.
        wave_unit: Wavelength unit string, e.g. ``'Angstrom'`` or ``'um'``

    Returns:
        float: Flux in erg/s/cm², or None if the units cannot reach it

    Examples:
        >>> integrated_flux_to_cgs(1.0, 'erg/s/cm**2/Angstrom', wave_unit='Angstrom')
        1.0
        >>> integrated_flux_to_cgs(1.0, 'Jy/beam', 7.0, None, 'um') is None
        True
    """
    flux_astropy_unit = _parse_flux_unit(flux_unit)
    wave_astropy_unit = _string_to_astropy_unit(wave_unit)
    if value is None or flux_astropy_unit is None or wave_astropy_unit is None:
        return None

    quantity = value * flux_astropy_unit * wave_astropy_unit
    return _reduce_to_target(quantity, CGS_FLUX, wavelength, wave_astropy_unit, header)


def flux_density_to_cgs(value, flux_unit, wavelength=None, header=None, wave_unit=None):
    """
    Convert a flux density (a peak height, not an integral) to erg/s/cm² per wavelength.

    The counterpart to :func:`integrated_flux_to_cgs` for quantities that keep their
    spectral dependence, such as a Gaussian amplitude. Same dimensional dispatch, but the
    target retains a per-wavelength: erg/s/cm²/``wave_unit``.

    Args:
        value: Flux density in ``flux_unit``
        flux_unit: Flux unit string, typically the BUNIT header keyword
        wavelength: Line centre, in ``wave_unit``. Required for MJy/sr.
        header: FITS header supplying CDELT1/CDELT2. Required for MJy/sr.
        wave_unit: Wavelength unit string, e.g. ``'Angstrom'`` or ``'um'``

    Returns:
        float: Flux density in erg/s/cm² per ``wave_unit``, or None if unsupported
    """
    flux_astropy_unit = _parse_flux_unit(flux_unit)
    wave_astropy_unit = _string_to_astropy_unit(wave_unit)
    if value is None or flux_astropy_unit is None or wave_astropy_unit is None:
        return None

    quantity = value * flux_astropy_unit
    target = CGS_FLUX / wave_astropy_unit
    return _reduce_to_target(quantity, target, wavelength, wave_astropy_unit, header)


def clean_flux_unit(unit_str):
    """
    Clean and format flux units for better readability.
    Converts verbose unit strings to compact symbol notation.
    
    This function is designed for display purposes and works with string
    representations of flux units (typically from FITS headers).
    
    Args:
        unit_str: Flux unit string (e.g., from BUNIT header keyword)
    
    Returns:
        str: Cleaned unit string suitable for display
    
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
                # Everything after the first slash is a denominator and must end up
                # negative. A part that already carries '⁻' is inverted; one carrying a
                # positive exponent (e.g. 'cm²' from '/cm**2') needs that exponent
                # negated, not left alone.
                if '⁻' in part:
                    formatted += f" {part}"
                elif part.endswith('²'):
                    formatted += f" {part[:-1]}⁻²"
                elif part.endswith('³'):
                    formatted += f" {part[:-1]}⁻³"
                else:
                    formatted += f" {part}⁻¹"
        result = formatted
    
    # Clean up extra spaces
    result = ' '.join(result.split())
    
    return result


def clean_wave_unit(unit_str):
    """
    Clean and format wavelength/velocity units.
    
    This function is designed for display purposes and works with string
    representations of wavelength/velocity units. It does NOT handle
    astropy Quantity objects - use format_wavelength_for_display() instead.
    
    Args:
        unit_str: Wavelength unit string
    
    Returns:
        str: Cleaned unit string suitable for display
    
    Examples:
        'Angstrom' -> 'Å'
        'micron' -> 'µm'
        'km/s' -> 'km/s'
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
