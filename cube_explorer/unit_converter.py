"""
Unit Conversion Utilities

Handles conversion between different wavelength and flux units used in spectroscopic
data. Uses astropy.units for accurate, unit-aware conversions, and still accepts plain
unit strings for backward compatibility.
"""

import numpy as np
import astropy.units as u
from astropy import constants as const
from astropy.units import Quantity
from functools import lru_cache
from typing import Optional


def extract_units(data):
    """
    Extract the value and unit from an astropy Quantity, or from a plain array.

    Args:
        data (array or Quantity): Data, with or without units

    Returns:
        result (tuple): (value, unit_string). unit_string is None for a plain array
    """

    if hasattr(data, 'unit'):
        # It's an astropy Quantity
        return data.value, str(data.unit)
    else:
        # Plain numpy array or list
        return np.asarray(data), None


def extract_value_from_quantity(value):
    """
    Extract numeric value from a Quantity object, stripping units element-wise if needed.

    Args:
        value (Quantity): Scalar or array, or an array of individual Quantity elements

    Returns:
        value (number or array): Numeric value with units stripped
    """
    if isinstance(value, Quantity):
        return value.value
    elif isinstance(value, np.ndarray):
        # Check if array elements are Quantities
        if len(value) > 0 and isinstance(value.flat[0], Quantity):
            return np.array([v.value for v in value.flat]).reshape(value.shape)
        return value
    else:
        return value


# Called once per pixel in a full-cube fit.
@lru_cache(maxsize=64)
def _string_to_astropy_unit(unit_str):
    """
    Convert a string unit representation to an astropy unit object.

    Delegates entirely to astropy's own parser. This parser already handles every real
    unit spelling this project's FITS headers use, for example 'um' and 'Angstrom'.

    Args:
        unit_str (str): String like 'Angstrom', 'um', 'nm', 'km/s', etc.

    Returns:
        unit (astropy.units.Unit): Unit, or None
    """
    if not unit_str:
        return None
    try:
        return u.Unit(str(unit_str).strip())
    except Exception:
        return None


SPEED_OF_LIGHT_KMS = const.c.to_value(u.km / u.s)  # 299792.458


def convert_wavelength(value, from_unit, to_unit):
    """
    Convert wavelength between different units, via astropy's own unit math.

    Args:
        value (array/Quantity): Wavelength value to convert
        from_unit (str/Unit): Source unit. Ignored if value is Quantity
        to_unit (str/Unit): Target unit

    Returns:
        converted (array/Quantity): Converted wavelength value

    Raises:
        ValueError: if from_unit or to_unit is not a recognized unit
    """

    to_unit_obj = _string_to_astropy_unit(to_unit)
    if to_unit_obj is None:
        raise ValueError(f"Unsupported target unit: {to_unit}")

    if isinstance(value, Quantity):
        return value.to(to_unit_obj)

    from_unit_obj = _string_to_astropy_unit(from_unit)
    if from_unit_obj is None:
        raise ValueError(f"Unsupported source unit: {from_unit}")

    return (value * from_unit_obj).to(to_unit_obj).value


def wavelength_to_doppler_velocity(wavelength, reference, wave_unit):
    """
    Convert wavelengths to Doppler velocities in km/s relative to a reference wavelength.
    Uses the non-relativistic optical formula v = c * (wavelength - reference) / reference.
    A positive velocity means the line is redshifted (moving away).

    Args:
        wavelength (float/array): Line wavelength or wavelengths, in wave_unit
        reference (float): Zero-velocity wavelength, in wave_unit
        wave_unit (str): Unit of both wavelength and reference, for example 'um'

    Returns:
        velocity (float/array): Velocity in km/s, same shape as wavelength

    Raises:
        ValueError: if wave_unit is not a unit of length, or reference is not positive
    """
    unit_obj = _string_to_astropy_unit(wave_unit)
    if unit_obj is None or unit_obj.physical_type != 'length':
        raise ValueError(f"Velocity needs a wavelength axis, but the spectral unit is '{wave_unit}'")
    if not (reference > 0):
        raise ValueError(f"Reference wavelength must be positive, got {reference}")
    return SPEED_OF_LIGHT_KMS * (np.asarray(wavelength, dtype=float) - reference) / reference


def doppler_velocity_to_wavelength_factor(velocity):
    """
    Convert a Doppler velocity to the factor that moves observed wavelengths to rest.
    This is the inverse of wavelength_to_doppler_velocity: rest = observed / (1 + v / c).

    Args:
        velocity (float): Velocity in km/s. Positive means moving away (redshift)

    Returns:
        factor (float): Number to multiply observed wavelengths by
    """
    return 1 / (1 + velocity / SPEED_OF_LIGHT_KMS)


def _angstrom_per_unit(wave_unit):
    """
    Give the number of angstrom in one wave_unit.

    Args:
        wave_unit (str): Unit of length, for example 'um' or 'Angstrom'

    Returns:
        scale (float): Angstrom per wave_unit

    Raises:
        ValueError: if wave_unit is not a unit of length
    """
    unit_obj = _string_to_astropy_unit(wave_unit)
    if unit_obj is None or unit_obj.physical_type != 'length':
        raise ValueError(f"Air and vacuum wavelengths need a wavelength axis, but the spectral unit is '{wave_unit}'")
    return float(unit_obj.to(u.AA))


def _air_refraction_index(vacuum_angstrom):
    """
    Give the refractive index of air for a vacuum wavelength in angstrom.
    Uses the IAU standard formula (Morton 1991), valid above 2000 angstrom.

    Args:
        vacuum_angstrom (float/array): Vacuum wavelength or wavelengths, in angstrom

    Returns:
        index (float/array): Refractive index of air, same shape as the input
    """
    sigma_squared = (1e4 / vacuum_angstrom) ** 2
    return 1 + 6.4328e-5 + 2.94981e-2 / (146 - sigma_squared) + 2.5540e-4 / (41 - sigma_squared)


def vacuum_to_air_wavelength(wavelength, wave_unit):
    """
    Convert vacuum wavelengths to air wavelengths with the IAU standard formula.
    The result has the same shape and the same unit as the input.

    Args:
        wavelength (float/array): Vacuum wavelength or wavelengths, in wave_unit
        wave_unit (str): Unit of wavelength, for example 'um' or 'Angstrom'

    Returns:
        air_wavelength (float/array): Air wavelength or wavelengths, in wave_unit

    Raises:
        ValueError: if wave_unit is not a unit of length
    """
    vacuum = np.asarray(wavelength, dtype=float)
    return vacuum / _air_refraction_index(vacuum * _angstrom_per_unit(wave_unit))


def air_to_vacuum_wavelength(wavelength, wave_unit):
    """
    Convert air wavelengths to vacuum wavelengths with the IAU standard formula.
    This inverts vacuum_to_air_wavelength by fixed-point iteration.

    Args:
        wavelength (float/array): Air wavelength or wavelengths, in wave_unit
        wave_unit (str): Unit of wavelength, for example 'um' or 'Angstrom'

    Returns:
        vacuum_wavelength (float/array): Vacuum wavelength or wavelengths, in wave_unit

    Raises:
        ValueError: if wave_unit is not a unit of length
    """
    air = np.asarray(wavelength, dtype=float)
    scale = _angstrom_per_unit(wave_unit)
    vacuum = air
    for _ in range(3):
        vacuum = air * _air_refraction_index(vacuum * scale)
    return vacuum


def spectral_axis_medium(header):
    """
    Tell if a cube measures wavelengths in air or in vacuum.
    The FITS axis type decides: 'AWAV' means air and 'WAVE' means vacuum.

    Args:
        header (astropy.io.fits.Header): Cube header, or None

    Returns:
        medium (str): 'air' or 'vacuum', or None if no axis type states it
    """
    if header is None:
        return None
    for axis in range(1, 5):
        axis_type = str(header.get(f'CTYPE{axis}', '')).strip().upper()
        if axis_type.startswith('AWAV'):
            return 'air'
        if axis_type.startswith('WAVE'):
            return 'vacuum'
    return None


def cube_wavelength_to_vacuum_um(wavelength, wave_unit, medium):
    """
    Convert wavelengths as a cube measures them to vacuum wavelengths in um.
    Line lists give vacuum wavelengths in um, so this puts cube values on the same scale.

    Args:
        wavelength (float/array): Wavelength or wavelengths, in wave_unit
        wave_unit (str): Unit of wavelength, for example 'um' or 'Angstrom'
        medium (str): 'air' or 'vacuum', as spectral_axis_medium gives it

    Returns:
        vacuum_um (float/array): Vacuum wavelength or wavelengths, in um

    Raises:
        ValueError: if wave_unit is not a unit of length, or medium is not valid
    """
    if medium not in ('air', 'vacuum'):
        raise ValueError(f"Medium must be 'air' or 'vacuum', got {medium!r}")
    micron = convert_wavelength(wavelength, wave_unit, 'um')
    return air_to_vacuum_wavelength(micron, 'um') if medium == 'air' else np.asarray(micron, dtype=float)


def vacuum_um_to_cube_wavelength(vacuum_um, wave_unit, medium):
    """
    Convert vacuum wavelengths in um to wavelengths as a cube measures them.
    This inverts cube_wavelength_to_vacuum_um.

    Args:
        vacuum_um (float/array): Vacuum wavelength or wavelengths, in um
        wave_unit (str): Unit the cube uses, for example 'um' or 'Angstrom'
        medium (str): 'air' or 'vacuum', as spectral_axis_medium gives it

    Returns:
        wavelength (float/array): Wavelength or wavelengths, in wave_unit

    Raises:
        ValueError: if wave_unit is not a unit of length, or medium is not valid
    """
    if medium not in ('air', 'vacuum'):
        raise ValueError(f"Medium must be 'air' or 'vacuum', got {medium!r}")
    micron = vacuum_to_air_wavelength(vacuum_um, 'um') if medium == 'air' else np.asarray(vacuum_um, dtype=float)
    return convert_wavelength(micron, 'um', wave_unit)


def are_units_compatible(unit1, unit2):
    """
    Check if two units are compatible, for example both wavelength or both velocity.
    Uses astropy's own physical-type comparison, not a hardcoded unit list.

    Args:
        unit1 (str/Unit/Quantity): First unit
        unit2 (str/Unit/Quantity): Second unit

    Returns:
        result (bool): True if units can be converted between each other
    """

    # Extract unit from Quantity objects
    if isinstance(unit1, Quantity):
        unit1 = unit1.unit
    if isinstance(unit2, Quantity):
        unit2 = unit2.unit

    unit1_obj = _string_to_astropy_unit(str(unit1))
    unit2_obj = _string_to_astropy_unit(str(unit2))

    if unit1_obj is None or unit2_obj is None:
        return False

    return unit1_obj.physical_type == unit2_obj.physical_type


# Unit of CGS Flux
CGS_FLUX = u.erg / u.s / u.cm ** 2


def pixel_solid_angle(header):
    """
    Solid angle subtended by one spatial pixel, in steradians.

    Args:
        header (astropy.io.fits.Header): FITS header with CDELT1 and CDELT2 in degrees

    Returns:
        area (Quantity): Area of one pixel in steradians, or None if the header lacks the pixel scales
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


def _reduce_to_target(quantity, target, wavelength, wave_astropy_unit, header):
    """
    Apply instrument corrections until `quantity` matches `target`, or give up.

    Args:
        quantity (Quantity): Value to reduce, in its native units
        target (Unit): Unit `quantity` must reach
        wavelength (float): Line center wavelength, in `wave_astropy_unit`
        wave_astropy_unit (Unit): Unit of `wavelength`
        header (astropy.io.fits.Header): FITS header supplying CDELT1/CDELT2 for the pixel solid angle

    Returns:
        value (float): `quantity` converted to `target`, or None if no combination of corrections reaches it
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
    Convert an integrated line flux (for example, a Gaussian area) to erg/s/cm**2.

    Dispatches by unit, not instrument name. MUSE's erg/s/cm**2/Angstrom needs no
    correction. MIRI and NIRSpec's MJy/sr needs the pixel solid angle and the c/lambda**2 jacobian.

    Args:
        value (float): Integrated flux in flux_unit x wave_unit
        flux_unit (str): Flux unit string, typically the BUNIT header keyword
        wavelength (float): Line center, in wave_unit. Required for MJy/sr, not used for MUSE
        header (astropy.io.fits.Header): FITS header with CDELT1 and CDELT2. Required for MJy/sr
        wave_unit (str): Wavelength unit string, for example 'Angstrom' or 'um'

    Returns:
        flux (float): Flux in erg/s/cm**2, or None if the units cannot reach it
    """
    flux_astropy_unit = _string_to_astropy_unit(flux_unit)
    wave_astropy_unit = _string_to_astropy_unit(wave_unit)
    if value is None or flux_astropy_unit is None or wave_astropy_unit is None:
        return None

    quantity = value * flux_astropy_unit * wave_astropy_unit
    return _reduce_to_target(quantity, CGS_FLUX, wavelength, wave_astropy_unit, header)


def _resolve_flux_unit(flux_unit_input: Optional[str], header) -> Optional[str]:
    """
    Return the cube's flux unit, falling back to the header's BUNIT.

    Args:
        flux_unit_input (str): Caller-tracked flux unit, or None to use the header
        header (astropy.io.fits.Header): FITS header to fall back on

    Returns:
        flux_unit (str): Resolved flux unit string, or None if neither source has one
    """
    if flux_unit_input is not None:
        return flux_unit_input
    return header.get('BUNIT', None) if header is not None else None


def get_wave_unit(header, fallback: Optional[str] = None) -> Optional[str]:
    """
    Get the cube's wavelength unit from the header's CUNIT3 value, or use the fallback.
    Returns the raw string. The caller passes it to _string_to_astropy_unit, so this
    function does not need to normalize it.

    Args:
        header (astropy.io.fits.Header): FITS header to read CUNIT3 from
        fallback (str): Unit to return if the header has none

    Returns:
        unit (str): Wavelength unit string, or None if neither source has one
    """
    if header is not None:
        cunit3 = header.get("CUNIT3", None)
        if cunit3:
            return str(cunit3).strip()
    return fallback


def convert_area_to_cgs(amplitude_mjy_sr, sigma_val, center_val, header, wave_unit: Optional[str], flux_unit_input: Optional[str] = None):
    """
    Convert a fitted Gaussian area (A * sigma * sqrt(2*pi)) to erg/s/cm**2 via integrated_flux_to_cgs.

    Args:
        amplitude_mjy_sr (float): Peak amplitude in native flux units
        sigma_val (float): Gaussian sigma in wavelength units (um or Angstrom)
        center_val (float): Line center wavelength in wavelength units (um or Angstrom)
        header (astropy.io.fits.Header): FITS header to extract CDELT1/CDELT2 for the pixel solid angle
        wave_unit (str): Wavelength unit (Angstrom, um)
        flux_unit_input (str): Flux unit from header, for example 'MJy/sr'. Falls back to the header's BUNIT if None

    Returns:
        area (float): Area in erg/s/cm**2, or None if wave_unit isn't a wavelength unit or conversion fails
    """
    if (header is None or amplitude_mjy_sr is None or
        sigma_val is None or center_val is None or not np.isfinite(center_val)):
        return None

    try:
        wave_astropy_unit = _string_to_astropy_unit(wave_unit)
        if wave_astropy_unit is None or wave_astropy_unit.physical_type != 'length':
            return None  # A velocity axis, for example, needs a rest wavelength we do not have

        # Gaussian area in native units, A * sigma * sqrt(2*pi), carrying [flux_unit * wave_unit]
        area_native = amplitude_mjy_sr * sigma_val * np.sqrt(2 * np.pi)

        return integrated_flux_to_cgs(
            area_native,
            _resolve_flux_unit(flux_unit_input, header),
            wavelength=center_val,
            header=header,
            wave_unit=wave_unit,
        )
    except Exception:
        return None


def gaussian_map_unit_labels(flux_unit, wave_unit, header=None, wavelength=None):
    """
    Unit labels for fitted Gaussian parameter maps. Probes this by running the real
    conversion on a value of 1.0, so a label can never drift from the number it names.

    Args:
        flux_unit (str): Cube flux unit, typically BUNIT (for example, 'MJy/sr')
        wave_unit (str): Spectral axis unit (for example, 'um' or 'Angstrom')
        header (astropy.io.fits.Header): FITS header with CDELT1 and CDELT2. Required
            for MJy/sr, not used for MUSE
        wavelength (float): A representative wavelength in wave_unit (for example, the
            fit window's center)

    Returns:
        labels (dict): Has 'wave', 'amplitude', and 'area' keys. amplitude and area use
            CGS units when reachable, or the cube's own units otherwise
    """
    wave_astropy_unit = _string_to_astropy_unit(wave_unit)
    wave_label = wave_astropy_unit.to_string('unicode') if wave_astropy_unit is not None \
        else (str(wave_unit).strip() if wave_unit else '')

    # Probes with the real conversion. Whatever it can do, the maps already did.
    integrated = integrated_flux_to_cgs(1.0, flux_unit, wavelength, header, wave_unit)

    if integrated is not None and wave_astropy_unit is not None:
        return {
            'wave': wave_label,
            # An amplitude is a flux density, so it keeps a per-wavelength unit. An
            # area has integrated over the spectral axis, so it is a plain flux.
            'amplitude': (CGS_FLUX / wave_astropy_unit).to_string('unicode'),
            'area': CGS_FLUX.to_string('unicode'),
        }

    # Not convertible, for example Jy/beam, or a missing header. Reports the cube's own units.
    flux_astropy_unit = _string_to_astropy_unit(flux_unit)
    if flux_astropy_unit is None:
        native_amp = str(flux_unit).strip() if flux_unit else 'native'
        native_area = f'{native_amp}*{wave_label}' if wave_label else native_amp
    else:
        native_amp = flux_astropy_unit.to_string('unicode')
        native_area = (flux_astropy_unit * wave_astropy_unit).to_string('unicode') \
            if wave_astropy_unit is not None else native_amp

    return {'wave': wave_label, 'amplitude': native_amp, 'area': native_area}


def _clean_unit_str(unit_str, empty_fallback):
    """
    Shared body for clean_flux_unit and clean_wave_unit. Renders through astropy's own
    unicode formatter, and falls back to the raw string if astropy cannot parse it.

    Args:
        unit_str (str): Unit string to clean
        empty_fallback (str): Value to return if unit_str is empty/None

    Returns:
        cleaned (str): Cleaned unit string suitable for display
    """
    if not unit_str:
        return empty_fallback
    unit = _string_to_astropy_unit(unit_str)
    if unit is None:
        return str(unit_str).strip()
    return unit.to_string('unicode')


def clean_flux_unit(unit_str):
    """
    Convert a flux unit string (for example, from BUNIT) to compact unicode notation for display.

    Args:
        unit_str (str): Flux unit string, for example from the BUNIT header keyword

    Returns:
        cleaned (str): Cleaned unit string suitable for display
    """
    return _clean_unit_str(unit_str, 'flux units')


def clean_wave_unit(unit_str):
    """
    Convert a wavelength/velocity unit string to compact unicode notation for display.

    Args:
        unit_str (str): Wavelength unit string

    Returns:
        cleaned (str): Cleaned unit string suitable for display
    """
    return _clean_unit_str(unit_str, '')
