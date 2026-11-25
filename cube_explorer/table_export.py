"""
Regions, contours and rotational-diagram tables, read and written with no GUI.
The region files are the ones that "Save Region" and "Import Region" use in the GUI.
The contour files are the ones that "Save Contour" and "Load Contour" use in the 2D viewer.
The CSV files are the ones that "Save Fluxes", "Import from CSV" and "Save Fit" use in the
rotational-diagram window.
"""

import csv
import os
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from . import region_io
from .convolve_reproject import map_description
from .line_catalog import fetch_hitran_data, match_nearest_hitran_line
from .region_selector import mask_from_saved_region, mask_from_shape_coords
from .rotational_diagram_h2_fitter import model_1T, model_2T
from .utils import make_2d_wcs


def _celestial_wcs(cube_data):
    """
    Give the 2D celestial WCS of a cube.

    Args:
        cube_data (CubeData): Loaded cube

    Returns:
        wcs2d (astropy.wcs.WCS): The 2D celestial WCS

    Raises:
        ValueError: if the cube header has no usable celestial WCS
    """
    wcs2d = make_2d_wcs(cube_data.header)
    if wcs2d is None or not getattr(wcs2d, "has_celestial", False):
        name = getattr(cube_data, "filename", None) or "This cube"
        raise ValueError(f"{name} has no usable celestial WCS, so a region cannot be placed on it.")
    return wcs2d


def make_region_mask(shape, coords, target_shape, verts=None):
    """
    Build a boolean mask of a region from its shape and its pixel coordinates.
    The mask has True for the pixels that the region touches, the same as a region drawn in the GUI.

    Args:
        shape (str): "POINT", "RECTANGLE", "CIRCLE", "ELLIPSE" or "POLYGON"
        coords (tuple): (x, y) for "POINT". (xmin, xmax, ymin, ymax) in pixels for "RECTANGLE",
            "CIRCLE" and "ELLIPSE", where the circle and the ellipse fill that box. Not used for "POLYGON".
        target_shape (tuple): (height, width) of the image
        verts (array-like): (N, 2) polygon vertices as (x, y) pixels. Needed for "POLYGON".

    Returns:
        mask (numpy.ndarray): 2D boolean array of shape target_shape

    Raises:
        ValueError: if the shape is not known, or a polygon has no vertices
    """
    shape = str(shape).upper()
    if shape not in ("POINT", "RECTANGLE", "CIRCLE", "ELLIPSE", "POLYGON"):
        raise ValueError(f"Unknown region shape '{shape}'. Use POINT, RECTANGLE, CIRCLE, ELLIPSE or POLYGON.")
    if shape == "POLYGON" and verts is None:
        raise ValueError("A POLYGON region needs verts, the (x, y) pixel vertices.")
    return mask_from_shape_coords(shape, coords, verts, tuple(target_shape))


def save_region_file(path, shape, coords, cube_data, verts=None, source_desc=None):
    """
    Save a region to a text file, as RA and Dec sky coordinates.
    This is the file that Save Region writes in the GUI. Load it again with load_region_file,
    on this cube or on another image of the same field.

    Args:
        path (str): Path of the text file to write
        shape (str): "POINT", "RECTANGLE", "CIRCLE", "ELLIPSE" or "POLYGON"
        coords (tuple): Pixel coordinates, see make_region_mask
        cube_data (CubeData): Loaded cube. Its WCS turns the pixels into sky coordinates.
        verts (array-like): (N, 2) polygon vertices as (x, y) pixels. Needed for "POLYGON".
        source_desc (str): Note for the header of the file. None uses the name of the cube file.

    Returns:
        None

    Raises:
        ValueError: if the shape is not known, a polygon has no vertices, or the cube has no celestial WCS
        OSError: if the file cannot be written
    """
    shape = str(shape).upper()
    result = region_io.region_shape_to_sky(shape, coords, _celestial_wcs(cube_data),
                                           lasso_verts=None if verts is None else np.asarray(verts, dtype=float))
    if source_desc is None:
        source_desc = getattr(cube_data, "filename", None) or ""
    region_io.write_region_wcs_txt(
        path, shape, source_desc, result["pixel_points"], result["sky_points"],
        result.get("sky_center"), result.get("sky_axis_x"), result.get("sky_axis_y"))


def load_region_file(path, cube_data):
    """
    Read a region file and turn it into a boolean mask on the pixel grid of a cube.
    This is Import Region in the GUI. The file can come from any image of the same field,
    because it holds sky coordinates.

    Args:
        path (str): Path of a region file that Save Region wrote
        cube_data (CubeData): Loaded cube that the mask is for

    Returns:
        mask (numpy.ndarray): 2D boolean array, shape (ny, nx) of the cube. Use it as mask in
            extract_region_spectrum or fit_region, or in region_masks.

    Raises:
        ValueError: if the file is not a region file, the cube has no celestial WCS, or the region
            is outside the cube
        OSError: if the file cannot be opened
    """
    saved = region_io.read_region_wcs_txt(path)
    target_shape = (cube_data.cube.shape[1], cube_data.cube.shape[2])
    return mask_from_saved_region(saved, _celestial_wcs(cube_data), target_shape)


_FLUX_COLUMNS = ["species", "wavelength", "flux", "flux_error", "upper_limit"]


def save_line_fluxes_csv(path, lines):
    """
    Save the line fluxes to a CSV file with the columns species, wavelength, flux, flux_error, upper_limit.
    This is the file that Save Fluxes writes in the GUI. load_line_fluxes_csv reads it again.
    A line with no flux is not written.

    Args:
        path (str): Path of the CSV file to write
        lines (list): One dict for each line, with 'species', 'wavelength' (um), 'flux' and 'flux_error'
            (erg/s/cm2). A line with 'upper_limit' True is written as an upper limit. The 'flux_error'
            of an upper limit is written too.

    Returns:
        count (int): Number of lines written

    Raises:
        ValueError: if no line has a flux
        OSError: if the file cannot be written
    """
    rows = []
    for line in lines:
        if line.get("flux") is None:
            continue
        error = line.get("flux_error")
        rows.append({
            "species": line.get("species", "Unknown"),
            "wavelength": line["wavelength"],
            "flux": repr(float(line["flux"])),
            "flux_error": "" if error is None else repr(float(error)),
            "upper_limit": "1" if line.get("upper_limit") else "0",
        })
    if not rows:
        raise ValueError("No line has a flux, so there is nothing to save.")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FLUX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def load_line_fluxes_csv(path):
    """
    Read a CSV file of lines, and add the HITRAN constants that the rotational diagram needs.
    This is Import from CSV in the GUI. The file needs the columns species and wavelength (um).
    The columns flux, an error column (any name with "error") and upper_limit are optional.
    Each line is matched to the nearest HITRAN line. A line with no match has no A, E_u and g_u.

    Args:
        path (str): Path of the CSV file

    Returns:
        lines (list): One dict for each valid row, in the order of the file: 'species', 'wavelength',
            'hitran_matched', 'transition', and, when HITRAN matched, 'a_coeff', 'e_upper', 'Gp',
            'j_lower', 'branch'. Also 'flux', 'flux_error' and 'upper_limit' when the file gives them.
            It is the line_table of fit_region and create_rotational_diagram_map.

    Raises:
        ValueError: if the file is empty, has no species or wavelength column, has no valid row,
            or has an upper_limit value that is not 1, 0, true, false, yes or no
        OSError: if the file cannot be opened
    """
    csv_data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("The CSV file is empty.")
        columns = {col.strip().lower(): col.strip() for col in reader.fieldnames if col}
        if not {"species", "wavelength"}.issubset(columns):
            missing = {"species", "wavelength"} - set(columns)
            raise ValueError(f"The CSV file must have 'species' and 'wavelength' columns. Missing: {', '.join(sorted(missing))}")
        flux_col = columns.get("flux")
        error_col = next((actual for lower, actual in columns.items() if "error" in lower and lower != "wavelength"), None)
        upper_limit_col = columns.get("upper_limit")
        problems = []
        for row_dict in reader:
            number = reader.line_num
            try:
                row = {"species": (row_dict.get(columns["species"]) or "").strip().upper(),
                       "wavelength": float((row_dict.get(columns["wavelength"]) or "").strip())}
            except (ValueError, TypeError):
                problems.append(f"line {number}: the species or the wavelength is not valid, so the row was skipped")
                continue
            for key, column in (("flux", flux_col), ("flux_error", error_col)):
                text = (row_dict.get(column) or "").strip() if column else ""
                if text:
                    try:
                        row[key] = float(text)
                    except ValueError:
                        problems.append(f"line {number}: {key} '{text}' is not a number, so the row has no {key}")
            if upper_limit_col:
                token = (row_dict.get(upper_limit_col) or "").strip().lower()
                if token not in ("", "0", "false", "no", "1", "true", "yes"):
                    raise ValueError(f"Line {number}: upper_limit is '{token}'. Use 1, 0, true, false, yes or no.")
                row["upper_limit"] = token in ("1", "true", "yes")
            csv_data.append(row)
    if problems:
        shown = "; ".join(problems[:5]) + (" ..." if len(problems) > 5 else "")
        warnings.warn(f"{os.path.basename(path)}: {len(problems)} problem(s) in the CSV file. {shown}")
    if not csv_data:
        raise ValueError("The CSV file has no valid row.")

    wavelengths = [row["wavelength"] for row in csv_data]
    cache, failed = {}, {}
    for species in sorted({row["species"] for row in csv_data}):
        try:
            hitran_lines, meta = fetch_hitran_data(min(wavelengths) - 0.1, max(wavelengths) + 0.1,
                                                   molecule=species, isotopologue=1)
            cache[species] = hitran_lines
            if meta.get("error"):
                failed[species] = meta.get("message", "unknown error")
        except Exception as error:
            cache[species], failed[species] = [], str(error)

    lines = []
    for row in csv_data:
        best = match_nearest_hitran_line(cache.get(row["species"], []), row["wavelength"])
        if best:
            line = {"wavelength": row["wavelength"], "hitran_matched": True,
                    "a_coeff": best.get("a_coeff", 0.0), "e_upper": best.get("e_upper", 0.0),
                    "Gp": best.get("Gp", 1.0), "j_lower": best.get("j_lower"), "branch": best.get("branch"),
                    "transition": best.get("transition", "User Data"), "species": row["species"]}
        else:
            reason = f" ({failed[row['species']]})" if row["species"] in failed else ""
            warnings.warn(f"No HITRAN match for {row['species']} at {row['wavelength']:.6f} um{reason}. "
                          "A_ul, E_u and g_u are blank, so this line is not fitted until you add them.")
            line = {"wavelength": row["wavelength"], "hitran_matched": False, "transition": "NA",
                    "species": row["species"]}
        for key in ("flux", "flux_error", "upper_limit"):
            if key in row:
                line[key] = row[key]
        lines.append(line)
    return lines


def _fit_line_table(plot_data):
    """
    Give the fit line of a diagram on 200 energies, from the lowest to the highest E_u.
    A 1T fit has one model column. A 2T fit has the two components and the combined model.

    Args:
        plot_data (dict): The '_plot_data' of a fit (E_arr, popt1, popt2)

    Returns:
        table (tuple): (header, columns), or None if the diagram has no line to draw
    """
    e_arr = np.asarray(plot_data["E_arr"], dtype=float)
    popt1, popt2 = plot_data.get("popt1"), plot_data.get("popt2")
    if len(e_arr) < 2 or (popt1 is None and popt2 is None):
        return None
    ln10 = np.log(10)
    e_fit = np.linspace(np.min(e_arr), np.max(e_arr), 200)
    if popt2 is not None:
        a1, t1, a2, t2 = popt2
        return (["E_u_K", "log10_T1_component", "log10_T2_component", "log10_2T_model"],
                [e_fit, (a1 - e_fit / t1) / ln10, (a2 - e_fit / t2) / ln10, model_2T(e_fit, *popt2) / ln10])
    return ["E_u_K", "log10_1T_model"], [e_fit, model_1T(e_fit, *popt1) / ln10]


def save_rotational_fit_csv(path, results, header_text=None):
    """
    Save a rotational-diagram fit to a CSV file: one row for each point of the diagram, then the fit line.
    This is Save Fit in the rotational-diagram window. The lines that start with "#" at the top hold the
    results. In the GUI they are the text of the results window.

    Args:
        path (str): Path of the CSV file to write
        results (dict): The 'results' of fit_region or fit_pixel (or the result of rotational_diagram).
            It must have '_plot_data'.
        header_text (str): Text for the "#" lines at the top. None writes the single numbers of the
            "1T" and "2T" entries of results.

    Returns:
        None

    Raises:
        ValueError: if results has no '_plot_data'
        OSError: if the file cannot be written
    """
    if "_plot_data" not in results:
        raise ValueError("The fit has no '_plot_data', so there are no points to save.")
    plot_data = results["_plot_data"]
    if header_text is None:
        header_lines = ["Rotational diagram fit"]
        for key in ("1T", "2T"):
            for name, value in (results.get(key) or {}).items():
                if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                    header_lines.append(f"{key} {name} = {float(value):.6g}")
    else:
        header_lines = header_text.strip("\n").splitlines()

    ln10 = np.log(10)
    e_arr = np.asarray(plot_data["E_arr"], dtype=float)
    y_log = np.asarray(plot_data["y_arr"], dtype=float) / ln10
    y_err = np.asarray(plot_data["y_err"], dtype=float) / ln10
    is_ul = np.asarray(plot_data.get("is_ul_mask", np.zeros(len(e_arr), dtype=bool)), dtype=bool)
    wave_arr = plot_data.get("wave_arr")
    with open(path, "w", newline="", encoding="utf-8") as f:
        for line in header_lines:
            f.write(f"# {line}".rstrip() + "\n")
        writer = csv.writer(f)
        writer.writerow(["wavelength_um", "E_u_K", "log10_Nu_over_gu", "log10_Nu_over_gu_error", "upper_limit"])
        for i in range(len(e_arr)):
            writer.writerow([
                "" if wave_arr is None else f"{float(wave_arr[i]):.6f}",
                f"{e_arr[i]:.1f}", f"{y_log[i]:.4f}", "" if is_ul[i] else f"{y_err[i]:.4f}", int(is_ul[i])])
        table = _fit_line_table(plot_data)
        if table is not None:
            f.write("\n# Fit line, as plotted: 200 energies from the lowest to the highest E_u\n")
            writer.writerow(table[0])
            for row in zip(*table[1]):
                writer.writerow([f"{row[0]:.1f}"] + [f"{value:.4f}" for value in row[1:]])


def contour_segments(cset):
    """Read the lines of a matplotlib contour set, as the vertices of each level.

    The vertices of one level are in the order that matplotlib gives them, one line after the other.
    A level with no line is left out.

    Args:
        cset (matplotlib.contour.QuadContourSet): Contour set made by ax.contour.

    Returns:
        segments (list): One (level, vertices) tuple for each level that has a line. level is a float
            and vertices is an (N, 2) array of (x, y) pixel positions.
    """
    return [(float(level), np.concatenate(segs)) for level, segs in zip(cset.levels, cset.allsegs) if len(segs)]


def image_contour_segments(image, levels, source_wcs=None, target_wcs=None):
    """Find the contour lines of an image with no plot window.

    The lines are the ones that the 2D viewer draws, because the same matplotlib call finds them.
    Without WCS, the vertices are pixel positions in the image. With both WCS, the lines go through
    the sky onto the pixel grid of the target, as a contour from the other image does in the viewer.

    Args:
        image (numpy.ndarray): 2D array of pixel values.
        levels (list): Pixel values where the lines are drawn. Each one must be between the
            lowest and the highest finite pixel.
        source_wcs (astropy.wcs.WCS): 2D WCS of image, or None.
        target_wcs (astropy.wcs.WCS): 2D WCS of the image that the lines go on, or None.

    Returns:
        segments (list): The (level, vertices) tuples of contour_segments.
    """
    from matplotlib.figure import Figure
    if source_wcs is None or target_wcs is None:
        return contour_segments(Figure().add_subplot(111).contour(image, levels=levels))
    ax = Figure().add_subplot(111, projection=target_wcs)
    return contour_segments(ax.contour(image, levels=levels, transform=ax.get_transform(source_wcs)))


def save_contour_file(path, segments, wcs, image_name=None, header=None):
    """Save contour lines to a text file, as pixel positions and as RA and Dec in degrees.

    This is the file that Save Contour writes in the 2D viewer. The pixel positions are positions
    on the image that the lines are drawn on, so the wcs must be the WCS of that image.

    Args:
        path (str): Path of the text file to write.
        segments (list): (level, vertices) tuples, as contour_segments gives them.
        wcs (astropy.wcs.WCS): WCS of the image that the lines are drawn on, or None.
        image_name (str): Name of the image that the levels come from, for the header of the file.
            None leaves out the lines about that image.
        header (astropy.io.fits.Header): Header of the image that the levels come from, for the map
            name and the unit of the levels in the file header, or None.

    Returns:
        has_sky (bool): True if the file has RA and Dec. False if the columns are empty, because the
            WCS has no sky coordinates or cannot give them.

    Raises:
        OSError: The file cannot be written.
    """
    rows = ["# IFU Analyzer 2D - Contour Export\n"]
    if image_name:
        rows.append(f"# Image: {image_name}\n")
        description = map_description(header, True) if header is not None else ''
        if description:
            rows.append(f"# Map: {description}\n")
        unit = str(header.get('BUNIT', '')).strip() if header is not None else ''
        if unit:
            rows.append(f"# Level unit: {unit}\n")
    rows.append("# level, x_pixel, y_pixel, RA_deg, Dec_deg\n")

    has_sky = wcs is not None and wcs.has_celestial
    for level, vertices in segments:
        x_pix, y_pix = vertices[:, 0], vertices[:, 1]
        ra = dec = None
        if has_sky:
            try:
                sky = wcs.pixel_to_world(x_pix, y_pix)
                ra, dec = sky.ra.deg, sky.dec.deg
            except Exception:
                has_sky = False
        for index, (x, y) in enumerate(zip(x_pix, y_pix)):
            ra_str = dec_str = ''
            if ra is not None:
                ra_str = f"{ra[index]:.8f}"
                dec_str = f"{dec[index]:.8f}"
            rows.append(f"{level:.6e}, {x:.4f}, {y:.4f}, {ra_str}, {dec_str}\n")

    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(rows)
    return has_sky


def _read_contour_rows(path, wcs):
    """Read the rows of a contour file, with the positions in the pixel grid of another image.

    The RA and Dec of the file are turned into pixel positions with the WCS. A file with no RA
    and Dec, or no WCS, gives the saved pixel positions. The Load Contour of the 2D viewer and
    the API use this function.

    Args:
        path (str): Path of a file that save_contour_file wrote.
        wcs (astropy.wcs.WCS): WCS of the image that the lines go on, or None.

    Returns:
        rows (tuple): (levels, x_pts, y_pts, use_sky). The first three are arrays with one value for
            each row of the file. use_sky is True if the positions come from RA and Dec.

    Raises:
        ValueError: The file has no rows with a level, an x and a y position.
        OSError: The file cannot be read.
    """
    data = np.genfromtxt(path, delimiter=',', comments='#')
    if data.ndim == 1: data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError("This is not a contour file. It has no rows with a level, an x and a y position.")

    x_pix_saved = data[:, 1]
    y_pix_saved = data[:, 2]

    # Prefer RA/Dec converted into this panel's pixel grid so the
    # contour aligns across different images; else fall back to raw pixels.
    use_sky = data.shape[1] >= 5 and wcs is not None
    if use_sky:
        ra_pts = data[:, 3]
        dec_pts = data[:, 4]
        use_sky = bool(np.all(np.isfinite(ra_pts)) and np.all(np.isfinite(dec_pts)))

    if use_sky:
        sky = SkyCoord(ra=ra_pts * u.deg, dec=dec_pts * u.deg, frame='icrs')
        x_pts, y_pts = wcs.world_to_pixel(sky)
        x_pts = np.asarray(x_pts, dtype=float)
        y_pts = np.asarray(y_pts, dtype=float)
    else:
        x_pts, y_pts = x_pix_saved, y_pix_saved
    return data[:, 0], x_pts, y_pts, use_sky


def read_contour_file(path, wcs):
    """Read a contour file and give its lines in the pixel positions of another image.

    This is Load Contour in the 2D viewer. The RA and Dec of the file are turned into pixel
    positions with the WCS. A file with no RA and Dec, or no WCS, gives the saved pixel positions.
    A jump of more than 5 pixels between two points starts a new line.

    Args:
        path (str): Path of a file that save_contour_file wrote.
        wcs (astropy.wcs.WCS): WCS of the image that the lines go on, or None.

    Returns:
        x_plot (numpy.ndarray): x pixel positions, with NaN between the lines.
        y_plot (numpy.ndarray): y pixel positions, with NaN between the lines.
        used_sky (bool): True if the positions come from RA and Dec. False if they are the saved pixel
            positions, which are right only if the image has the same pixel grid as the image that
            the file came from.

    Raises:
        ValueError: The file has no rows with a level, an x and a y position.
        OSError: The file cannot be read.
    """
    segments, use_sky = read_contour_segments(path, wcs)
    x_plot, y_plot = contour_line_arrays(segments)
    return x_plot, y_plot, use_sky


def contour_line_arrays(segments):
    """Join the segments of a contour into one line for plotting.
    A jump of more than 5 pixels between two points starts a new line, and the arrays have NaN there.

    Args:
        segments (list): The (level, vertices) tuples of contour_segments, or of read_contour_segments.

    Returns:
        x_plot (numpy.ndarray): x pixel positions, with NaN between the lines.
        y_plot (numpy.ndarray): y pixel positions, with NaN between the lines.
    """
    vertices = np.concatenate([points for _, points in segments])
    x_pts, y_pts = vertices[:, 0], vertices[:, 1]
    dist = np.sqrt(np.diff(x_pts)**2 + np.diff(y_pts)**2)
    break_indices = np.where(dist > 5.0)[0] + 1

    x_plot = np.insert(x_pts, break_indices, np.nan)
    y_plot = np.insert(y_pts, break_indices, np.nan)
    return x_plot, y_plot


def read_contour_segments(path, wcs):
    """Read a contour file and give its lines with their levels, in the pixel positions of another image.
    The rows of one level that follow each other make one segment. This is what Save Contour needs
    to write a loaded contour again, and it is the same positions as read_contour_file gives.

    Args:
        path (str): Path of a file that save_contour_file wrote.
        wcs (astropy.wcs.WCS): WCS of the image that the lines go on, or None.

    Returns:
        segments (list): The (level, vertices) tuples of contour_segments. vertices is an (N, 2) array
            of (x, y) pixel positions.
        used_sky (bool): True if the positions come from RA and Dec. See read_contour_file.

    Raises:
        ValueError: The file has no rows with a level, an x and a y position.
        OSError: The file cannot be read.
    """
    levels, x_pts, y_pts, use_sky = _read_contour_rows(path, wcs)
    # A NaN level (no number in the file) is equal to the NaN level next to it.
    same_level = (levels[1:] == levels[:-1]) | (np.isnan(levels[1:]) & np.isnan(levels[:-1]))
    bounds = np.concatenate(([0], np.where(~same_level)[0] + 1, [len(levels)]))
    segments = [(float(levels[start]), np.column_stack((x_pts[start:stop], y_pts[start:stop])))
                for start, stop in zip(bounds[:-1], bounds[1:])]
    return segments, use_sky
