"""
Write maps to FITS files, with no GUI.
The Gaussian map, the Flux Map and the rotational-diagram line maps each have a Save or Export
button in the GUI. The functions here write the same files with the same header keywords.
"""

import io
import os
import zipfile

import astropy.units as u
import numpy as np

from .fits_export import _ascii_safe, build_2d_export_header, build_line_map_header, write_fits_product
from .gaussian_fitter import fit_status_map
from .line_catalog import fetch_hitran_data, match_nearest_hitran_line
from .unit_converter import (
    CGS_FLUX_NAME, gaussian_map_unit_labels, remove_systemic_velocity, spectral_axis_medium,
    vacuum_um_to_cube_wavelength, wavelength_to_doppler_velocity,
)

# parameter -> (name in the GUI list, key in the maps dict, file name part)
GAUSSIAN_MAP_PARAMETERS = {
    "area": ("Area (AUC)", "area", "area_map"),
    "area_error": ("Area Error", "area_error", "area_error_map"),
    "amplitude": ("Amplitude", "amplitude", "amplitude_map"),
    "continuum": ("Continuum", "continuum", "continuum_map"),
    "center": ("Center", "center", "center_map"),
    "velocity": ("Velocity", None, "velocity_map"),
    "sigma": ("Sigma", "sigma", "sigma_map"),
    "fwhm": ("FWHM", "fwhm", "fwhm_map"),
    "chi2": ("chi2 (fit quality)", "r_squared", "chi2_map"),
    "fit_status": ("Fit Status", None, "fit_status"),
    "success_mask": ("Success Mask", "success", "success_mask"),
}

# The unit text of the maps that have no unit of a flux or a wavelength
_FIXED_UNITS = {
    "velocity": "km/s",
    "chi2": "dimensionless",
    "fit_status": "0=no data, 1=upper limit, 2=detection",
    "success_mask": "1=yes, 0=no",
}


def _gaussian_map_data(cube_data, maps, parameter, rest_wavelength, systemic_velocity):
    """
    Give the 2D array of one Gaussian map parameter.
    The Velocity map is built here from the line centers, as the GUI builds it.
    Fit Status and Success Mask are turned into numbers.

    Args:
        cube_data (CubeData): Loaded cube that the map was fitted on
        maps (dict): Result of create_gaussian_map()
        parameter (str): A key of GAUSSIAN_MAP_PARAMETERS
        rest_wavelength (float): Rest wavelength of the line in um (vacuum), for the Velocity map
        systemic_velocity (float): Systemic velocity in km/s, for the Velocity map

    Returns:
        data (numpy.ndarray): The 2D array

    Raises:
        ValueError: if the Velocity map has no rest wavelength, or the cube header does not say
            if its wavelengths are in air or in vacuum
    """
    if parameter == "velocity":
        if rest_wavelength is None:
            raise ValueError("The Velocity map needs rest_wavelength, in um (vacuum).")
        medium = spectral_axis_medium(cube_data.header)
        if medium is None:
            raise ValueError("The cube header does not say if its wavelengths are in air or in vacuum "
                             "(CTYPE3 must be WAVE or AWAV), so a rest wavelength cannot be compared.")
        rest = float(vacuum_um_to_cube_wavelength(rest_wavelength, cube_data.wave_unit, medium))
        velocity = wavelength_to_doppler_velocity(maps["center"], rest, cube_data.wave_unit)
        return remove_systemic_velocity(velocity, systemic_velocity)
    if parameter == "fit_status":
        return fit_status_map(maps)
    if parameter == "success_mask":
        return np.asarray(maps["success"]).astype(float)
    return maps[GAUSSIAN_MAP_PARAMETERS[parameter][1]]


def _gaussian_map_unit(cube_data, parameter, lam_min, lam_max):
    """
    Give the unit text that the GUI writes in BUNIT for one Gaussian map parameter.

    Args:
        cube_data (CubeData): Loaded cube that the map was fitted on
        parameter (str): A key of GAUSSIAN_MAP_PARAMETERS
        lam_min (float): Start of the fit window, in the wavelength unit of the cube
        lam_max (float): End of the fit window, in the wavelength unit of the cube

    Returns:
        unit (str): The unit text, with only ASCII characters
    """
    if parameter in _FIXED_UNITS:
        return _FIXED_UNITS[parameter]
    flux_unit = getattr(cube_data, "flux_unit", "") or "native"
    labels = gaussian_map_unit_labels(flux_unit, cube_data.wave_unit, header=cube_data.header,
                                      wavelength=(lam_min + lam_max) / 2.0)
    if parameter in ("area", "area_error"):
        return _ascii_safe(labels["area"])
    if parameter in ("amplitude", "continuum"):
        return _ascii_safe(flux_unit)
    return _ascii_safe(labels["wave"])


def save_gaussian_map(cube_data, maps, parameter, out_path, lam_min, lam_max, peak_wavelength=None,
                      rest_wavelength=None, rest_label=None, systemic_velocity=0.0, species=None):
    """
    Save one parameter of a Gaussian map to a FITS file.
    This is the file that the Export button of the Gaussian map window writes.
    The header has the fit window, the unit, and the keywords that each parameter needs.

    Args:
        cube_data (CubeData): Loaded cube that the map was fitted on
        maps (dict): Result of create_gaussian_map()
        parameter (str): One of "area", "area_error", "amplitude", "continuum", "center",
            "velocity", "sigma", "fwhm", "chi2", "fit_status", "success_mask"
        out_path (str): Path of the FITS file to write
        lam_min (float): Start of the fit window, in the wavelength unit of the cube
        lam_max (float): End of the fit window, in the wavelength unit of the cube
        peak_wavelength (float): Wavelength of the line peak. None uses the middle of the window.
        rest_wavelength (float): Rest wavelength in um (vacuum). The Velocity map needs it.
        rest_label (str): Short text that names the rest line, for the header. None writes the wavelength.
        systemic_velocity (float): Systemic velocity in km/s that the Velocity map is relative to
        species (str): Species of the line, for example "H2". For the Area map only. It adds the HITRAN
            constants of the nearest line to the header, so the rotational diagram can read the map.

    Returns:
        None

    Raises:
        ValueError: if the parameter is not known, or the Velocity map cannot be made
        OSError: if the file cannot be written
    """
    if parameter not in GAUSSIAN_MAP_PARAMETERS:
        raise ValueError(f"Unknown parameter '{parameter}'. Use one of: {', '.join(GAUSSIAN_MAP_PARAMETERS)}.")
    param_name = GAUSSIAN_MAP_PARAMETERS[parameter][0]
    data = _gaussian_map_data(cube_data, maps, parameter, rest_wavelength, systemic_velocity)
    lam_min, lam_max = min(lam_min, lam_max), max(lam_min, lam_max)
    bunit = _gaussian_map_unit(cube_data, parameter, lam_min, lam_max)

    source_header = cube_data.header
    object_label = (source_header.get("OBJECT", "Unknown") if source_header is not None else "Unknown") \
        + f" ({param_name})"
    line_wave = peak_wavelength if peak_wavelength is not None else (lam_min + lam_max) / 2.0
    extra = {
        "BUNIT": bunit,
        "OBJECT": object_label,
        "PARAM": param_name,
        "LAMMIN": (lam_min, "Minimum wavelength of fit"),
        "LAMMAX": (lam_max, "Maximum wavelength of fit"),
        "NSUCC": (int(np.sum(maps["success"])), "Number of successful fits"),
        "NTOTAL": (int(np.asarray(maps["success"]).size), "Total number of pixels"),
    }
    if parameter == "continuum":
        extra["CONTWAV"] = (float(maps["continuum_wavelength"]), "Wavelength of the continuum value")
        extra["CONTMETH"] = "Straight line between the fit window edges"
    if parameter == "velocity":
        medium = spectral_axis_medium(source_header)
        rest = float(vacuum_um_to_cube_wavelength(rest_wavelength, cube_data.wave_unit, medium))
        label = rest_label if rest_label is not None else f"{rest_wavelength:.6f} um vacuum"
        extra["VREFWAV"] = (rest, "Rest wavelength that sets zero velocity")
        extra["VREFUNIT"] = (_ascii_safe(str(cube_data.wave_unit)), "Unit of VREFWAV")
        extra["VREFMED"] = (medium, "Medium of VREFWAV: air or vacuum")
        extra["VREFLINE"] = (_ascii_safe(label)[:68], "Rest line")
        extra["VSYS"] = (float(systemic_velocity), "Systemic velocity removed from map (km/s)")

    is_area = parameter == "area"
    species_text = species.strip().upper() if (is_area and species) else ""
    if species_text:
        try:
            hitran_lines, _meta = fetch_hitran_data(
                line_wave - 0.01, line_wave + 0.01, molecule=species_text, isotopologue=1)
        except Exception:
            hitran_lines = []
        best_match = match_nearest_hitran_line(hitran_lines, line_wave)
        if best_match:
            extra["A_UL"] = (best_match.get("a_coeff", 0.0), "Einstein A coefficient (1/s)")
            extra["E_UPPER"] = (best_match.get("e_upper", 0.0), "Upper state energy (K)")
            extra["G_U"] = (best_match.get("Gp", 1.0), "Upper state statistical weight")
            extra["TRANSITN"] = best_match.get("transition", "User Data")
            if best_match.get("j_lower") is not None:
                extra["JLOWER"] = int(best_match["j_lower"])
            if best_match.get("branch"):
                extra["BRANCH"] = best_match["branch"]

    # The fit of the summed spectrum is a point fit. The rotational diagram can show it.
    global_fit = maps.get("global_fit") if is_area else None
    if global_fit:
        for key, name, comment in (
                ("area", "GFITAREA", "Global fit area (flux)"),
                ("area_error", "GFITAERR", "Global fit area error"),
                ("r_squared", "GFITR2", "Global fit R-squared"),
                ("amplitude", "GFITAMP", "Global fit amplitude"),
                ("center", "GFITCTR", "Global fit center (um)"),
                ("sigma", "GFITSIG", "Global fit sigma (um)")):
            if global_fit.get(key) is not None:
                extra[name] = (float(global_fit[key]), comment)

    err_map = maps.get("area_error") if is_area else None
    out_header, extensions = build_line_map_header(
        source_header, data.shape, wavelength=line_wave,
        molecule=species_text if species_text else None,
        flux_map=data if is_area else None, error_map=err_map,
        src_filename=(getattr(cube_data, "filename", None) or "UNKNOWN") if is_area else None,
        fit_method="Single-Gaussian fit" if is_area else None, extra=extra,
    )
    if global_fit and global_fit.get("x_data") is not None:
        extensions = list(extensions) + [
            ("SPECX", np.asarray(global_fit["x_data"], dtype=np.float32), None),
            ("SPECY", np.asarray(global_fit["y_data"], dtype=np.float32), None),
            ("SPECBASE", np.asarray(global_fit["baseline"], dtype=np.float32), None),
        ]
    instrument = out_header.get("INSTRUME", "").strip().upper() or cube_data.instrument or "UNKNOWN"
    telescope = out_header.get("TELESCOP", "").strip().upper() or "UNKNOWN"
    if instrument != "UNKNOWN":
        out_header["INSTRUME"] = instrument
    if telescope != "UNKNOWN":
        out_header["TELESCOP"] = telescope
    out_header["COMMENT"] = f"{param_name}: {bunit}"
    out_header["COMMENT"] = f"Gaussian fit parameter: {param_name}"
    write_fits_product(out_path, data, out_header, extensions=extensions or None)


def save_gaussian_maps(cube_data, maps, out_dir, lam_min, lam_max, parameters=None, peak_wavelength=None,
                       rest_wavelength=None, rest_label=None, systemic_velocity=0.0, species=None,
                       base_name=None):
    """
    Save several parameters of a Gaussian map, one FITS file for each.
    The file names are the ones that the Export button suggests: <cube name>_gaussian_<parameter>.fits.
    The Velocity map is left out when there is no rest_wavelength, unless you ask for it by name.

    Args:
        cube_data (CubeData): Loaded cube that the map was fitted on
        maps (dict): Result of create_gaussian_map()
        out_dir (str): Folder for the files. It is made if it does not exist.
        lam_min (float): Start of the fit window, in the wavelength unit of the cube
        lam_max (float): End of the fit window, in the wavelength unit of the cube
        parameters (list): Parameter names, see save_gaussian_map. None writes all of them.
        peak_wavelength (float): Wavelength of the line peak. None uses the middle of the window.
        rest_wavelength (float): Rest wavelength in um (vacuum), for the Velocity map
        rest_label (str): Short text that names the rest line, for the header
        systemic_velocity (float): Systemic velocity in km/s that the Velocity map is relative to
        species (str): Species of the line, for the Area map. See save_gaussian_map.
        base_name (str): First part of each file name. None uses the name of the cube file.

    Returns:
        paths (list): The paths of the files that were written

    Raises:
        ValueError: if a parameter is not known, or you ask for the Velocity map without rest_wavelength
        OSError: if a file cannot be written
    """
    if parameters is None:
        parameters = [p for p in GAUSSIAN_MAP_PARAMETERS if p != "velocity" or rest_wavelength is not None]
    if base_name is None:
        filename = getattr(cube_data, "filename", None)
        base_name = os.path.splitext(os.path.basename(filename))[0] if filename else "gaussian"
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for parameter in parameters:
        if parameter not in GAUSSIAN_MAP_PARAMETERS:
            raise ValueError(f"Unknown parameter '{parameter}'. Use one of: {', '.join(GAUSSIAN_MAP_PARAMETERS)}.")
        path = os.path.join(out_dir, f"{base_name}_gaussian_{GAUSSIAN_MAP_PARAMETERS[parameter][2]}.fits")
        save_gaussian_map(cube_data, maps, parameter, path, lam_min, lam_max, peak_wavelength=peak_wavelength,
                          rest_wavelength=rest_wavelength, rest_label=rest_label,
                          systemic_velocity=systemic_velocity, species=species)
        paths.append(path)
    return paths


def save_flux_map(result, out_path, species=None, cube_data=None):
    """
    Save a Flux Map result to a FITS file.
    This is the file that the Export button of the Flux Map window writes.
    The first image is the flux map. The extensions hold the detection counts and the contaminating lines.

    Args:
        result (dict): Result of create_flux_map()
        out_path (str): Path of the FITS file to write
        species (str): Species of the target lines, for example "CO". None writes "Target".
        cube_data (CubeData): Loaded cube. It gives the instrument name when the header has none.

    Returns:
        None

    Raises:
        OSError: if the file cannot be written
    """
    flux_map = np.asarray(result["flux_map"])
    header = result.get("header")
    shape = flux_map.shape
    species_label = species.strip() if species and species.strip() else "Target"

    flux_unit = result.get("flux_unit") or "MJy/sr um"
    if flux_unit != "MJy/sr um":
        unit_str = flux_unit
    else:
        bunit = header.get("BUNIT", "MJy/sr") if header else "MJy/sr"
        try:
            bunit = str(u.Unit(bunit))
        except (ValueError, TypeError):
            pass
        unit_str = f"{bunit} um"

    primary_extra = {
        "DESC": f"{species_label} rovibrational flux: Integrated area under target line gaussians",
    }
    wave_range = result.get("wave_range")
    if wave_range:
        primary_extra["WAVEMIN"] = (wave_range[0], "Min wavelength used (um)")
        primary_extra["WAVEMAX"] = (wave_range[1], "Max wavelength used (um)")
    primary_header, _ = build_line_map_header(
        header, shape, molecule=species_label, extra={**primary_extra, "BUNIT": unit_str})
    if "INSTRUME" not in primary_header and cube_data is not None and getattr(cube_data, "instrument", None):
        primary_header["INSTRUME"] = cube_data.instrument

    extensions = [(
        "DETECTION", np.asarray(result["detection_map"]).astype(np.int32),
        build_2d_export_header(header, shape, extra={
            "BUNIT": "count", "DESC": "Number of target lines detected per pixel"}),
    )]
    if result.get("contam_flux_map") is not None:
        extensions.append((
            "CONTAM_FLUX", np.asarray(result["contam_flux_map"]).astype(np.float32),
            build_2d_export_header(header, shape, extra={
                "BUNIT": unit_str,
                "DESC": "Contaminating-line flux: integrated area under non-target line gaussians"}),
        ))
    if result.get("contam_detection_map") is not None:
        extensions.append((
            "CONTAM_DETECTION", np.asarray(result["contam_detection_map"]).astype(np.int32),
            build_2d_export_header(header, shape, extra={
                "BUNIT": "count", "DESC": "Number of contaminating lines detected per pixel"}),
        ))
    write_fits_product(out_path, flux_map.astype(np.float32), primary_header,
                       extensions=extensions, output_verify="warn")


def _line_map_groups(result, molecule):
    """
    Collect the per-line flux maps of a rotational-diagram result, as the Save Maps button collects them.
    A result can hold the native maps (each line on its own grid), the convolved maps (all lines on
    one common grid), or both.

    Args:
        result (dict): Result of compute_native_line_maps, convolve_line_maps,
            create_rotational_diagram_map or create_rotational_diagram_map_multi
        molecule (str): Species name for the file names and headers. None uses result["molecule"], or "H2".

    Returns:
        groups (list): (folder, tag, entries) tuples. Each entry is a dict with wavelength, flux_map,
            flux_error_map, header, identity_header, source_filename and molecule.
    """
    molecule = molecule or result.get("molecule", "H2")
    kept_lines = result.get("kept_lines") or []
    groups = []

    native_maps = result.get("native_flux_maps")
    single_cube = native_maps is None and result.get("line_flux_maps") and result.get("dst_wcs") is None
    if single_cube:
        native_maps = result["line_flux_maps"]
    if kept_lines and native_maps:
        wcses = result.get("native_wcs")
        err_maps = (result.get("native_flux_error_maps") if not single_cube
                    else result.get("line_flux_error_maps")) or []
        entries = []
        for i, line in enumerate(kept_lines):
            if wcses is not None:
                header = wcses[i].to_header() if wcses[i] is not None else None
                identity_header = result.get("native_headers")[i] if result.get("native_headers") else None
                source_filename = result.get("cube_filenames")[i] if result.get("cube_filenames") else None
            else:
                header = result.get("header")
                identity_header = header
                source_filename = result.get("cube_filename")
            entries.append({"wavelength": line["wavelength"], "flux_map": native_maps[i],
                            "flux_error_map": err_maps[i] if i < len(err_maps) else None,
                            "header": header, "identity_header": identity_header,
                            "source_filename": source_filename, "molecule": molecule})
        groups.append(("Flux Maps", "flux", entries))

    convolved_maps = result.get("line_flux_maps")
    dst_wcs = result.get("dst_wcs")
    if kept_lines and convolved_maps and dst_wcs is not None:
        err_maps = result.get("line_flux_error_maps") or []
        # No cube owns the common grid, so the cube with the coarsest pixels gives the identity keywords.
        identity_header, identity_filename = None, None
        if result.get("native_headers") and result.get("native_pix_arcsec"):
            worst = int(np.argmax(result["native_pix_arcsec"]))
            identity_header = result["native_headers"][worst]
            if result.get("cube_filenames"):
                identity_filename = result["cube_filenames"][worst]
        source_header = dst_wcs.to_header()
        entries = [{"wavelength": line["wavelength"], "flux_map": convolved_maps[i],
                    "flux_error_map": err_maps[i] if i < len(err_maps) else None,
                    "header": source_header, "identity_header": identity_header,
                    "source_filename": identity_filename, "molecule": molecule}
                   for i, line in enumerate(kept_lines)]
        groups.append(("Convolved Maps", "convolved", entries))
    return groups


def _line_map_fits_bytes(entry):
    """
    Build the FITS file of one line map in memory, with the header of the Save Maps button.

    Args:
        entry (dict): One entry of _line_map_groups

    Returns:
        data (bytes): The raw bytes of the FITS file
    """
    flux_map = entry["flux_map"]
    header, extensions = build_line_map_header(
        entry.get("header"), flux_map.shape, wavelength=entry["wavelength"],
        molecule=entry["molecule"], flux_map=flux_map, error_map=entry.get("flux_error_map"),
        src_filename=entry.get("source_filename") or "UNKNOWN", fit_method="Multi-Gaussian fit",
        identity_header=entry.get("identity_header"),
        extra={"BUNIT": CGS_FLUX_NAME,
               "DESC": f"Flux map: {entry['molecule']} line at {entry['wavelength']:.4f} um"},
    )
    buffer = io.BytesIO()
    write_fits_product(buffer, np.asarray(flux_map, dtype=np.float32), header, extensions=extensions)
    return buffer.getvalue()


def save_line_maps(result, output_path, molecule=None):
    """
    Save the flux map of every line in a rotational-diagram result.
    This is the Save Maps button of the rotational-diagram window, without the PNG files.
    The native maps go in "Flux Maps/fits" and the convolved maps in "Convolved Maps/fits".
    Each file is named <species>_<flux or convolved>_<wavelength>um.fits.

    Args:
        result (dict): Result of compute_native_line_maps (native maps), convolve_line_maps
            (convolved maps), create_rotational_diagram_map (native maps), or
            create_rotational_diagram_map_multi (both)
        output_path (str): A folder, or a path that ends in .zip to write one zip file.
            The folder is made if it does not exist.
        molecule (str): Species name for the file names and headers. None uses the one in the result, or "H2".

    Returns:
        counts (dict): {"Flux Maps": number of native maps written, "Convolved Maps": number of
            convolved maps written}. A group that the result does not have is not in the dict.

    Raises:
        ValueError: if the result has no line maps
        OSError: if a file cannot be written
    """
    groups = _line_map_groups(result, molecule)
    if not groups:
        raise ValueError("The result has no per-line flux maps to save.")
    use_zip = str(output_path).lower().endswith(".zip")
    counts = {}
    archive = zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) if use_zip else None
    try:
        for folder, tag, entries in groups:
            for entry in sorted(entries, key=lambda e: e["wavelength"]):
                stem = f"{entry['molecule']}_{tag}_{entry['wavelength']:.4f}um"
                data = _line_map_fits_bytes(entry)
                relative = f"{folder}/fits/{stem}.fits"
                if use_zip:
                    archive.writestr(relative, data)
                else:
                    path = os.path.join(output_path, *relative.split("/"))
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(data)
            counts[folder] = len(entries)
    finally:
        if archive is not None:
            archive.close()
    return counts
