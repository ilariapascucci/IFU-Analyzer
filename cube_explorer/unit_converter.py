"""
Unit Conversion Utilities

Handles conversion between different wavelength and flux units used in spectroscopic
data. Uses astropy.units for accurate, unit-aware conversions, and still accepts plain
unit strings for backward compatibility.
"""

import numpy as np
import astropy.units as u
from astropy import constants as const
from astropy.io import fits
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
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


def um_to_wavenumber(wavelength_um):
    """
    Convert a wavelength in micron to a wavenumber in cm^-1, which is 1e4 / wavelength.
    The same operation converts a wavenumber in cm^-1 to a wavelength in micron.

    Args:
        wavelength_um (float/array): Wavelength or wavelengths in micron

    Returns:
        wavenumber (float/array): Wavenumber or wavenumbers in cm^-1, same shape as the input
    """
    return 1e4 / wavelength_um


def wavenumber_to_um(wavenumber):
    """
    Convert a wavenumber in cm^-1 to a wavelength in micron, which is 1e4 / wavenumber.

    Args:
        wavenumber (float/array): Wavenumber or wavenumbers in cm^-1

    Returns:
        wavelength_um (float/array): Wavelength or wavelengths in micron, same shape as the input
    """
    return 1e4 / wavenumber


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


def remove_systemic_velocity(velocity, systemic_velocity):
    """
    Give Doppler velocities relative to the systemic velocity of the source.
    The result is the same as shifting the spectrum by the systemic velocity and then
    measuring the line centers again, because wavelength_to_doppler_velocity is linear in wavelength.

    Args:
        velocity (float/array): Doppler velocity or velocities in km/s, from wavelength_to_doppler_velocity
        systemic_velocity (float): Systemic velocity in km/s. Positive means moving away (redshift)

    Returns:
        velocity (float/array): Velocity in km/s, same shape as velocity. NaN stays NaN

    Raises:
        ValueError: if systemic_velocity is not a finite number, or is not greater than -c
    """
    if not np.isfinite(systemic_velocity):
        raise ValueError(f"The systemic velocity must be a finite number, got {systemic_velocity}")
    if systemic_velocity <= -SPEED_OF_LIGHT_KMS:
        raise ValueError(f"The systemic velocity must be greater than -{SPEED_OF_LIGHT_KMS:.0f} km/s, got {systemic_velocity}")
    return (np.asarray(velocity, dtype=float) - systemic_velocity) / (1 + systemic_velocity / SPEED_OF_LIGHT_KMS)


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


def doppler_factor_to_velocity(factor):
    """
    Convert the factor that moves observed wavelengths to rest into a Doppler velocity.
    This is the inverse of doppler_velocity_to_wavelength_factor: v = c * (1 / factor - 1).

    Args:
        factor (float): Number that observed wavelengths are multiplied by

    Returns:
        velocity (float): Velocity in km/s. Positive means moving away (redshift)
    """
    return SPEED_OF_LIGHT_KMS * (1.0 / factor - 1.0)


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
# ASCII name of the CGS flux unit, for result dicts and FITS headers. Windows show cgs_flux_label().
CGS_FLUX_NAME = 'erg/s/cm2'


def cgs_flux_label():
    """
    Give the CGS flux unit as the text that every window and file shows to the user.

    Args:
        None

    Returns:
        label (str): The unit in unicode notation, for example "erg s⁻¹ cm⁻²"
    """
    return CGS_FLUX.to_string('unicode')


# Header keywords that set the area of one pixel. Equal values give an equal area.
_PIXEL_AREA_KEYS = ('PIXAR_SR', 'CDELT1', 'CDELT2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                    'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2')


@lru_cache(maxsize=32)
def _pixel_area_sr(values):
    """
    Work out the solid angle of one pixel from the header keywords that give it.
    The WCS gives the area from the pixel scale, the rotation and any CD matrix. The WCS always
    wins, because a header that was changed to another grid can keep an old PIXAR_SR.
    PIXAR_SR is used only if the header has no WCS pixel scale.

    Args:
        values (tuple): Header values in the order of _PIXEL_AREA_KEYS. None for a missing key.

    Returns:
        area (float): Solid angle of one pixel in steradians, or None if the header gives none.
    """
    keys = dict(zip(_PIXEL_AREA_KEYS, values))
    pixar = keys['PIXAR_SR']
    try:
        pixar = float(pixar) if pixar is not None and float(pixar) > 0 else None
    except (TypeError, ValueError):
        pixar = None

    wcs_area = None
    has_cd = any(keys[k] is not None for k in ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'))
    if has_cd or (keys['CDELT1'] is not None and keys['CDELT2'] is not None):
        minimal = fits.Header()
        minimal['NAXIS'] = 2
        minimal['CTYPE1'], minimal['CTYPE2'] = 'RA---TAN', 'DEC--TAN'
        minimal['CRPIX1'], minimal['CRPIX2'] = 1.0, 1.0
        minimal['CRVAL1'], minimal['CRVAL2'] = 0.0, 0.0
        for key, value in keys.items():
            if key != 'PIXAR_SR' and value is not None:
                minimal[key] = value
        try:
            wcs_area = float((proj_plane_pixel_area(WCS(minimal)) * u.deg ** 2).to(u.sr).value)
        except Exception:
            wcs_area = None

    return wcs_area if wcs_area is not None else pixar


def pixel_solid_angle(header):
    """
    Solid angle subtended by one spatial pixel, in steradians.

    Args:
        header (astropy.io.fits.Header): FITS header with the pixel scale: CDELT1 and CDELT2 in
            degrees, or a CD matrix in degrees. PIXAR_SR is used only if neither is present.

    Returns:
        area (Quantity): Area of one pixel in steradians, or None if the header lacks the pixel scales
    """
    if header is None:
        return None
    area = _pixel_area_sr(tuple(header.get(k) for k in _PIXEL_AREA_KEYS))
    return None if area is None else area * u.sr


def _reduce_to_target(quantity, target, wavelength, wave_astropy_unit, header):
    """
    Apply instrument corrections until `quantity` matches `target`, or give up.
    A flux density (for example Jy) needs only the c/lambda**2 factor. A surface brightness
    (for example MJy/sr) also needs the pixel solid angle.

    Args:
        quantity (Quantity): Value to reduce, in its native units
        target (Unit): Unit `quantity` must reach
        wavelength (float): Line center wavelength, in `wave_astropy_unit`
        wave_astropy_unit (Unit): Unit of `wavelength`
        header (astropy.io.fits.Header): FITS header supplying the pixel scale for the pixel solid angle, or None

    Returns:
        value (float): `quantity` converted to `target`, or None if no combination of corrections reaches it
    """
    if quantity.unit.is_equivalent(target):
        return float(quantity.to(target).value)

    if wavelength is None or wave_astropy_unit is None:
        return None

    # Flux density x wavelength -> flux. No pixel area is involved, so no header is needed.
    spectral = quantity * (const.c / (wavelength * wave_astropy_unit) ** 2)
    if spectral.unit.is_equivalent(target):
        return float(spectral.to(target).value)

    omega = pixel_solid_angle(header)
    if omega is None:
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
    correction. A flux density in Jy (for example an x1d spectrum) needs the c/lambda**2 jacobian.
    MIRI and NIRSpec's MJy/sr also needs the pixel solid angle.

    Args:
        value (float): Integrated flux in flux_unit x wave_unit
        flux_unit (str): Flux unit string, typically the BUNIT header keyword
        wavelength (float): Line center, in wave_unit. Required for Jy and MJy/sr, not used for MUSE
        header (astropy.io.fits.Header): FITS header with the pixel scale. Required for MJy/sr, not for Jy
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


def spectrum_flux_density_unit(flux_unit, header):
    """
    Tell how the pixel values of a cube become the values of a spectrum. A spectrum is a sum of
    pixels, so a surface brightness (for example MJy/sr) becomes a flux density (Jy) when each pixel
    is multiplied by its own solid angle. A unit that is not per steradian stays as it is.

    Args:
        flux_unit (str): Flux unit of the cube pixels, typically the BUNIT header keyword
        header (astropy.io.fits.Header): FITS header with the pixel scale

    Returns:
        result (tuple): (scale, unit). scale (float) multiplies the pixel values. unit (str) is the unit
            of the spectrum, 'Jy' for MJy/sr, or flux_unit unchanged. scale is 1.0 when the unit is unchanged.
    """
    astropy_unit = _string_to_astropy_unit(flux_unit)
    if astropy_unit is None or not (astropy_unit * u.sr).is_equivalent(u.Jy):
        return 1.0, flux_unit
    omega = pixel_solid_angle(header)
    if omega is None:
        print(f"[WARNING] The header has no pixel scale, so the spectra stay in {flux_unit}, a sum of pixels.")
        return 1.0, flux_unit
    return float(omega.value * (astropy_unit * u.sr).to(u.Jy)), 'Jy'


def is_per_solid_angle_unit(unit_text):
    """
    Tell whether a unit is a surface brightness, which is a unit per solid angle, for example MJy/sr,
    erg/s/cm2/sr or MJy/arcsec2. A map in such a unit gives the flux of a pixel when each value is
    multiplied by the solid angle of the pixel.

    Args:
        unit_text (str): Unit text, typically the BUNIT header keyword

    Returns:
        is_per_angle (bool): True if the unit has a solid angle in its denominator
    """
    unit = _string_to_astropy_unit(unit_text)
    if unit is None:
        return False
    # The steradian is rad**2 in astropy, so it shows as the radian after the unit is broken into base units.
    decomposed = unit.decompose()
    return dict(zip(decomposed.bases, decomposed.powers)).get(u.rad, 0) < 0


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


def gaussian_area(amplitude, sigma):
    """
    Area under a Gaussian line, amplitude * sigma * sqrt(2*pi). Every line fit in this program
    uses this one formula, including the upper-limit area of an artificial line.

    Args:
        amplitude (float): Peak amplitude, in any flux unit
        sigma (float): Gaussian sigma, in any wavelength unit

    Returns:
        area (float): Area, in the flux unit times the wavelength unit
    """
    return amplitude * sigma * np.sqrt(2 * np.pi)


def convert_area_to_cgs(amplitude_mjy_sr, sigma_val, center_val, header, wave_unit: Optional[str], flux_unit_input: Optional[str] = None):
    """
    Convert a fitted Gaussian area (A * sigma * sqrt(2*pi)) to erg/s/cm**2 via integrated_flux_to_cgs.

    Args:
        amplitude_mjy_sr (float): Peak amplitude in native flux units
        sigma_val (float): Gaussian sigma in wavelength units (um or Angstrom)
        center_val (float): Line center wavelength in wavelength units (um or Angstrom)
        header (astropy.io.fits.Header): FITS header to extract the pixel scale for the pixel solid angle
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
        area_native = gaussian_area(amplitude_mjy_sr, sigma_val)

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
        header (astropy.io.fits.Header): FITS header with the pixel scale. Required
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
            'area': cgs_flux_label(),
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
