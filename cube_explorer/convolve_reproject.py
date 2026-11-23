"""
convolved_reproject.ipynb, ported for GUI use. Naman's version.
"""

import os
import tempfile
import zipfile

import numpy as np
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.io import fits
from astropy.stats import gaussian_sigma_to_fwhm
from reproject import reproject_exact

from .fits_export import build_coverage_extension, build_resample_header, write_fits_product
from .unit_converter import is_per_solid_angle_unit, pixel_solid_angle
from .utils import (
    detect_instrument_from_header, get_pixel_size_arcsec, is_per_pixel_flux_unit, make_2d_wcs,
    psf_fwhm_arcsec, wcs_footprints_overlap,
)


def gaussian_convolve_match(images,
                            wavelengths_um,
                            target_wavelength_um,
                            pixel_scales_arcsec,
                            instruments,
                            target_instrument):
    """
    Convolve a JWST/MIRI or NIRSpec line-flux map to the PSF of a longer wavelength.

    Parameters
    ----------
    images : list of 2D ndarray
        List of input images (e.g., W m^-2 per pixel).

    wavelengths_um : list of float
        List of wavelengths of the input images [microns].

    target_wavelength_um : float
        Wavelength whose PSF is to be matched [microns].
        Must be >= wavelength_um.

    pixel_scales_arcsec : list of float
        List of pixel sizes of the input images [arcsec/pixel].

    instruments : list of str
        Instrument of each input image ('MIRI' or 'NIRSpec').

    target_instrument : str
        Instrument of the image whose PSF is to be matched ('MIRI' or 'NIRSpec').

    Returns
    -------
    out : 2D ndarray
        PSF-matched image.

    Notes
    -----
    Assumes the PSF is Gaussian with the FWHM(") of psf_fwhm_arcsec:

        MIRI:    FWHM(") = 0.033 * lambda(um) + 0.106
        NIRSpec: FWHM(") = 0.1 * (0.031406393 * lambda^2 + 0.019719141 * lambda + 1.5167115)

    NaNs are interpolated during convolution and restored afterwards.
    Flux is preserved by using a normalized convolution kernel.
    """
    output_images = []

    for i in range(len(images)):
        image = np.asarray(images[i], dtype=float)
        image[np.where(image == 0.0)] = np.nan

        # PSF FWHM (arcsec)
        fwhm_in = psf_fwhm_arcsec(wavelengths_um[i], instruments[i])
        fwhm_target = psf_fwhm_arcsec(target_wavelength_um[0], target_instrument)

        if fwhm_target <= fwhm_in:
            print(f"Skipping convolution for {wavelengths_um[i]} um -> {target_wavelength_um[0]} um")
            continue

        # Convert FWHM -> sigma (arcsec)
        sigma_in = fwhm_in / gaussian_sigma_to_fwhm
        sigma_target = fwhm_target / gaussian_sigma_to_fwhm

        # Required convolution kernel
        sigma_kernel_arcsec = np.sqrt(sigma_target**2 - sigma_in**2)

        # Convert to pixels
        sigma_kernel_pix = sigma_kernel_arcsec / pixel_scales_arcsec[i]

        # Choose kernel size (~8 sigma, forced odd)
        size = max(5, int(np.ceil(8 * sigma_kernel_pix)))
        if size % 2 == 0:
            size += 1

        kernel = Gaussian2DKernel(
            sigma_kernel_pix,
            x_size=size,
            y_size=size
        )

        out = convolve_fft(
            image,
            kernel,
            nan_treatment='interpolate',
            preserve_nan=True,
            min_wt=0.5, #if more than 50% of the kernel is NaN, set output to NaN
            boundary='fill',
            fill_value=np.nan,
            normalize_kernel=True,
            allow_huge=True,
        )

        out[out < np.nanmin(image)] = np.nan

        output_images.append(np.asarray(out, dtype=float))

    return output_images


def reproject_flux_maps(images, input_headers, output_header):
    """
    Flux-conserving reprojection of multiple flux maps with NaN handling.

    Input maps are assumed to be flux per pixel (e.g. W m^-2 pixel^-1).
    """

    if len(images) != len(input_headers):
        raise ValueError(
            "images and input_headers must have the same length"
        )

    reprojected_images = []
    footprints = []

    # FIX (agreed with the algorithm's author): convert flux to surface brightness before
    # reprojecting, back to flux after -- see module docstring.
    omega_out = pixel_solid_angle(output_header)

    for image, header in zip(images, input_headers):

        # Valid pixels
        valid = np.isfinite(image)
        min_value = np.nanmin(image)

        # FIX: flux -> surface brightness, this image's own native pixel solid angle
        omega_in = pixel_solid_angle(header)
        image_sb = image / omega_in.value

        # Replace NaNs with zero for flux conservation
        image_filled = np.nan_to_num(image_sb, nan=0.0)

        # Reproject flux map
        image_out, footprint = reproject_exact(
            (image_filled, header),
            output_projection=output_header,
        )

        # Reproject validity mask
        valid_out, _ = reproject_exact(
            (valid.astype(float), header),
            output_projection=output_header,
        )

        # Pixels with no valid input contribution
        image_out[valid_out == 0] = np.nan

        #Very low-valued pixels are likely artifacts of the reprojection, so set them to NaN
        #if exclude_small:
        #    image_out[image_out < min_value] = np.nan

        # FIX: surface brightness -> flux, scaled by the actually-covered
        # fraction (footprint) -- not scaled up to the full pixel's own area.
        image_out = image_out * omega_out.value * footprint

        reprojected_images.append(image_out)
        footprints.append(footprint)

    return reprojected_images, footprints



def _require_all_supported(kept_lines, instruments):
    """Raise if any line isn't MIRI or NIRSpec -- Convolve Maps/Run Map have no
    confirmed PSF formula for any other instrument yet, so a silent wrong
    answer is worse than refusing.

    Args:
        kept_lines (list): Line dicts, each with a 'wavelength' key.
        instruments (list): Instrument name per kept_lines entry.

    Returns:
        None

    Raises:
        ValueError: naming every unsupported line's wavelength and instrument.
    """
    bad = [(l["wavelength"], inst) for l, inst in zip(kept_lines, instruments)
           if inst not in ("MIRI", "NIRSpec")]
    if bad:
        detail = ", ".join(f"{wl:.4f} um ({inst or 'unknown'})" for wl, inst in bad)
        raise ValueError(
            "Convolve Maps/Run Map currently support MIRI and NIRSpec only (no confirmed "
            f"PSF formula for other instruments yet): {detail}")


def convolution_plan(native_result):
    """Cheap, no-op preview of what convolve_reproject_maps will do to each
    line -- the same psf_fwhm_arcsec arithmetic, without running
    any convolution or reprojection. Shaped for the GUI's plan-confirmation table.

    Args:
        native_result (dict): Output of compute_native_line_maps.

    Returns:
        plan (dict): 'worst_pix_arcsec' (reference line's native pixel
            scale), 'dst_shape' (reference line's native shape), 'per_cube'
            (list of dicts: 'wavelength', 'instrument', 'filename',
            'native_pix_arcsec', 'sigma_arcsec', 'sigma_pix'), 'dropped_lines'.
    """
    kept_lines = native_result["kept_lines"]
    shapes = native_result["native_shapes"]
    scales = native_result["native_pix_arcsec"]
    filenames = native_result["cube_filenames"]
    instruments = native_result["native_instrument"]
    _require_all_supported(kept_lines, instruments)

    ref_idx = int(np.argmax([line["wavelength"] for line in kept_lines]))
    ref_wl = kept_lines[ref_idx]["wavelength"]
    fwhm_target = psf_fwhm_arcsec(ref_wl, instruments[ref_idx])

    per_cube = []
    for i, line in enumerate(kept_lines):
        wl = line["wavelength"]
        if i == ref_idx:
            per_cube.append({"wavelength": wl, "instrument": instruments[i], "filename": filenames[i],
                             "native_pix_arcsec": scales[i],
                             "sigma_arcsec": 0.0, "sigma_pix": 0.0})
            continue
        fwhm_in = psf_fwhm_arcsec(wl, instruments[i])
        sigma_in = fwhm_in / gaussian_sigma_to_fwhm
        sigma_target = fwhm_target / gaussian_sigma_to_fwhm
        sigma_kernel_arcsec = (max(0.0, sigma_target**2 - sigma_in**2)) ** 0.5
        sigma_kernel_pix = sigma_kernel_arcsec / scales[i] if scales[i] > 0 else 0.0
        per_cube.append({"wavelength": wl, "instrument": instruments[i], "filename": filenames[i],
                         "native_pix_arcsec": scales[i],
                         "sigma_arcsec": sigma_kernel_arcsec, "sigma_pix": sigma_kernel_pix})

    return {"worst_pix_arcsec": scales[ref_idx], "dst_shape": shapes[ref_idx],
            "per_cube": per_cube, "dropped_lines": native_result["dropped_lines"]}


def convolve_reproject_maps(native_result):
    """Run the notebook functions above on a compute_native_line_maps
    result: pick the longest-wavelength kept line as the reference (the
    notebook's own H2_S1 choice, generalized), convolve every other line to
    it with gaussian_convolve_match, reproject with reproject_flux_maps onto
    the reference's own native grid, and pass the reference line through
    with only the notebook's own 0->NaN cleanup (its last cell, no
    convolution/reprojection). Error maps are pushed through the identical
    unmodified functions a second time -- the notebook never had error maps
    to convolve, but combine_line_maps needs per-pixel flux uncertainty to
    fit at all, and this calls the same functions rather than writing new ones.

    Args:
        native_result (dict): Output of compute_native_line_maps.

    Returns:
        result (dict): Drop-in compatible with combine_line_maps's
            convolved_result argument -- 'kept_lines', 'line_flux_maps',
            'line_flux_error_maps', 'dst_wcs', 'dst_shape', 'worst_pix_arcsec'
            (the reference line's own native pixel scale), 'dropped_lines',
            native_flux_maps/native_flux_error_maps/native_wcs/native_shapes/
            native_pix_arcsec/cube_filenames (passed through from
            native_result unchanged), plus 'reference_wavelength' for display.
    """
    kept_lines = native_result["kept_lines"]
    flux_maps = native_result["native_flux_maps"]
    err_maps = native_result["native_flux_error_maps"]
    wcses = native_result["native_wcs"]
    scales = native_result["native_pix_arcsec"]
    instruments = native_result["native_instrument"]
    _require_all_supported(kept_lines, instruments)

    ref_idx = int(np.argmax([line["wavelength"] for line in kept_lines]))
    ref_shape = flux_maps[ref_idx].shape
    ref_header = wcses[ref_idx].to_header()
    ref_header["NAXIS"] = 2
    ref_header["NAXIS1"] = ref_shape[1]
    ref_header["NAXIS2"] = ref_shape[0]
    ref_wl = kept_lines[ref_idx]["wavelength"]

    other_idx = [i for i in range(len(kept_lines)) if i != ref_idx]

    def _run_one_array(maps):
        """Build the notebook's own list arguments (images_to_be_convolved-style
        lists, excluding the reference line, matching cell 6d57c517's own
        source-list construction) and call gaussian_convolve_match/
        reproject_flux_maps exactly as the notebook calls them."""
        images = [maps[i] for i in other_idx]
        wavelengths_um = [kept_lines[i]["wavelength"] for i in other_idx]
        pixel_scales_arcsec = [scales[i] for i in other_idx]
        input_instruments = [instruments[i] for i in other_idx]
        input_headers = []
        for i in other_idx:
            h = wcses[i].to_header()
            h["NAXIS"] = 2
            h["NAXIS1"] = maps[i].shape[1]
            h["NAXIS2"] = maps[i].shape[0]
            input_headers.append(h)

        convolved_images = gaussian_convolve_match(
            images, wavelengths_um, [ref_wl], pixel_scales_arcsec,
            input_instruments, instruments[ref_idx])
        if len(convolved_images) != len(other_idx):
            raise ValueError(
                "gaussian_convolve_match skipped a line (fwhm_target <= fwhm_in) -- "
                "the reference line must have the longest wavelength for every other "
                "line to need convolving to it.")
        reprojected_images, _ = reproject_flux_maps(convolved_images, input_headers, ref_header)

        out = [None] * len(kept_lines)
        for pos, i in enumerate(other_idx):
            out[i] = reprojected_images[pos]
        # The notebook's own reference-line treatment (last cell): 0->NaN only.
        ref_map = np.asarray(maps[ref_idx], dtype=float).copy()
        ref_map[ref_map == 0.0] = np.nan
        out[ref_idx] = ref_map
        return out

    out_flux = _run_one_array(flux_maps)
    out_err = _run_one_array(err_maps)

    return {
        "kept_lines": kept_lines, "dropped_lines": native_result["dropped_lines"],
        "line_flux_maps": out_flux, "line_flux_error_maps": out_err,
        "dst_wcs": make_2d_wcs(ref_header), "dst_shape": ref_shape,
        "worst_pix_arcsec": scales[ref_idx],
        "reference_wavelength": ref_wl, "reference_header": ref_header,
        "native_flux_maps": native_result["native_flux_maps"],
        "native_flux_error_maps": native_result["native_flux_error_maps"],
        "native_wcs": native_result["native_wcs"], "native_shapes": native_result["native_shapes"],
        "native_pix_arcsec": native_result["native_pix_arcsec"],
        "cube_filenames": native_result["cube_filenames"],
        # The real header of each cube, so that a saved map keeps INSTRUME, TARGNAME and the file name.
        "native_headers": native_result.get("native_headers"),
        "native_instrument": native_result.get("native_instrument"),
    }


# ================== 2D maps: convolve to the longest-wavelength map ==================
# The 2D viewer (Pair and Batch Convolution) and the API (convolve_maps) use these functions.


def header_line_wavelength_um(header):
    """Read LINEWAVE, or LAMMIN/LAMMAX's midpoint, from a header -- no
    prompting. LINEWAVE (rotational-diagram Save Maps exports) is exact.
    LAMMIN/LAMMAX (Gaussian Map exports) bound one line's own fit window, so
    their midpoint is a good proxy. WAVEMIN/WAVEMAX (Flux Map exports) can
    span several blended lines, so those are never used here -- a midpoint
    there could silently name the wrong line.

    Args:
        header (astropy.io.fits.Header): This image's FITS header, or None.

    Returns:
        wavelength_um (float or None): Wavelength in microns, or None if
            neither header key is present or usable.
    """
    if header is not None and 'LINEWAVE' in header:
        try:
            return float(header['LINEWAVE'])
        except (TypeError, ValueError):
            pass
    if header is not None and 'LAMMIN' in header and 'LAMMAX' in header:
        try:
            return (float(header['LAMMIN']) + float(header['LAMMAX'])) / 2.0
        except (TypeError, ValueError):
            pass
    return None


def unsupported_files(infos):
    """Return (filename, instrument) for every input that isn't MIRI or NIRSpec --
    the PSF formula has no confirmed fit for any other instrument yet.

    Args:
        infos (list): (filename, header) pairs to check.

    Returns:
        bad (list): (filename, detected instrument or 'unknown') pairs.
    """
    bad = []
    for filename, header in infos:
        inst = detect_instrument_from_header(header) if header is not None else None
        if inst not in ("MIRI", "NIRSpec"):
            bad.append((filename, inst or "unknown"))
    return bad


def map_description(header, is_primary):
    """Name what one HDU shows, from the keywords that IFU Analyzer exports write.
    DESC wins. A Gaussian map has PARAM, and its other HDUs add their own HDU name.
    If there is neither, the HDU name is used.

    Args:
        header (astropy.io.fits.Header): Header of one HDU.
        is_primary (bool): True if the HDU is the first HDU of the file.

    Returns:
        description (str): The description, or an empty string if the header has none.
    """
    extname = str(header.get('EXTNAME', '')).strip()
    desc = str(header.get('DESC', '')).strip()
    if desc:
        return desc
    param = str(header.get('PARAM', '')).strip()
    if param:
        return param if is_primary or not extname else f"{param}, {extname}"
    return '' if extname.upper() == 'PRIMARY' else extname


def convolved_description(source_header, ref_wavelength):
    """Build the DESC text of a convolved image.

    Args:
        source_header (astropy.io.fits.Header): Header of the image that was convolved.
        ref_wavelength (float): Wavelength of the reference image, in um.

    Returns:
        text (str): For example "Convolved to the PSF at 17.0348 um: Area (AUC)".
    """
    text = f"Convolved to the PSF at {ref_wavelength:.4f} um"
    source_desc = map_description(source_header, True)
    return f"{text}: {source_desc}" if source_desc else text


def convolve_map_to_reference(conv_img, conv_header, conv_wl, conv_pix,
                              ref_img, ref_header, ref_wl):
    """Convolve conv_img to ref_wl's PSF (psf_fwhm_arcsec formula of each
    image's instrument) and reproject it onto ref_header's own native grid, using the
    same unmodified gaussian_convolve_match/reproject_flux_maps ported for
    the rotational-diagram convolution.
    The unit of conv_img is a flux per pixel (for example erg/s/cm2) or a surface brightness
    (for example MJy/sr). A surface brightness becomes a flux per pixel before the engine runs
    and goes back to a surface brightness after it, so the result has the unit of conv_img.

    Args:
        conv_img (numpy.ndarray): Image to convolve+reproject (shorter wavelength).
        conv_header (astropy.io.fits.Header): conv_img's FITS header.
        conv_wl (float): conv_img's line wavelength (microns).
        conv_pix (float): conv_img's pixel scale (arcsec/pixel).
        ref_img (numpy.ndarray): Reference image (longer wavelength).
        ref_header (astropy.io.fits.Header): Reference image's FITS header.
        ref_wl (float): Reference image's line wavelength (microns).

    Returns:
        result (dict): 'image' (conv_img reprojected onto ref's grid),
            'footprint' (per-pixel coverage fraction, same grid),
            'sigma_arcsec' (kernel sigma used).

    Raises:
        ValueError: If ref_wl's PSF is not larger than conv_wl's (the
            formula requires the reference to have the longer wavelength), if the unit of
            conv_img is neither a flux per pixel nor a surface brightness, or if the header of
            a surface brightness has no pixel scale.
    """
    conv_instrument = detect_instrument_from_header(conv_header)
    ref_instrument = detect_instrument_from_header(ref_header)
    fwhm_in = psf_fwhm_arcsec(conv_wl, conv_instrument)
    fwhm_target = psf_fwhm_arcsec(ref_wl, ref_instrument)
    if fwhm_target <= fwhm_in:
        raise ValueError(
            f"The convolution requires the reference wavelength "
            f"({ref_wl:.4f} um) to have a larger PSF than the convolved "
            f"image's ({conv_wl:.4f} um) -- got FWHM {fwhm_target:.4f}\" "
            f"<= {fwhm_in:.4f}\".")

    bunit = conv_header.get("BUNIT", "")
    per_angle = is_per_solid_angle_unit(bunit)
    if not per_angle and not is_per_pixel_flux_unit(bunit):
        raise ValueError(
            f"The unit '{bunit}' is not a flux per pixel (for example erg/s/cm2) and not a "
            "surface brightness (for example MJy/sr), so the convolution cannot use it.")

    conv_header = conv_header.copy()
    conv_header["NAXIS"] = 2
    conv_header["NAXIS1"] = conv_img.shape[1]
    conv_header["NAXIS2"] = conv_img.shape[0]
    ref_header = ref_header.copy()
    ref_header["NAXIS"] = 2
    ref_header["NAXIS1"] = ref_img.shape[1]
    ref_header["NAXIS2"] = ref_img.shape[0]

    image_in = conv_img
    if per_angle:
        omega_in = pixel_solid_angle(conv_header)
        omega_out = pixel_solid_angle(ref_header)
        if omega_in is None or omega_out is None:
            raise ValueError("The header has no pixel scale, so the area of one pixel is not known.")
        image_in = np.asarray(conv_img, dtype=float) * omega_in.value

    convolved = gaussian_convolve_match([image_in], [conv_wl], [ref_wl], [conv_pix],
                                        [conv_instrument], ref_instrument)
    reprojected, footprints = reproject_flux_maps(convolved, [conv_header], ref_header)
    image_out, footprint = reprojected[0], footprints[0]
    if per_angle:
        # The engine gives the flux of the covered part of each pixel, so dividing by the covered
        # area gives the mean surface brightness back. A pixel with no coverage stays NaN.
        covered_sr = omega_out.value * footprint
        with np.errstate(divide="ignore", invalid="ignore"):
            image_out = np.where(covered_sr > 0, image_out / covered_sr, np.nan)

    sigma_in = fwhm_in / gaussian_sigma_to_fwhm
    sigma_target = fwhm_target / gaussian_sigma_to_fwhm
    sigma_kernel_arcsec = (sigma_target ** 2 - sigma_in ** 2) ** 0.5

    return {"image": image_out, "footprint": footprint,
            "sigma_arcsec": sigma_kernel_arcsec}


def load_map_for_convolution(path):
    """Read the first 2D image of a FITS file, with the header, the pixel scale and the WCS
    that the convolution of 2D maps needs.

    Args:
        path (str): Path of the FITS file.

    Returns:
        info (dict): 'filename', 'data' (2D float array), 'header', 'pix_scale' (arcsec/pixel),
            'wcs' (2D celestial WCS) and 'shape'.

    Raises:
        ValueError: If the file has no 2D image, no pixel scale, or no usable WCS.
    """
    with fits.open(path) as hdul:
        data, header = None, None
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, 'size', 0) > 0:
                sq = np.squeeze(hdu.data)
                if sq.ndim == 2:
                    data = sq.astype(float)
                    header = hdu.header.copy()
                    break
        if data is None:
            raise ValueError(
                f"No 2D image data in {os.path.basename(path)}")
        pix = get_pixel_size_arcsec(header)
        if pix is None:
            raise ValueError(
                f"Could not find pixel scale in "
                f"{os.path.basename(path)}")
        wcs = make_2d_wcs(header)
        if wcs is None:
            raise ValueError(
                f"Could not build WCS for {os.path.basename(path)}")
        return {
            'filename': os.path.basename(path),
            'data': data, 'header': header,
            'pix_scale': pix, 'wcs': wcs, 'shape': data.shape,
        }


def convolution_flux_totals(input_image, input_header, output_image, output_header):
    """Sum a map before and after a convolution, as a flux, for the check that flux is kept.
    A surface brightness is multiplied by the solid angle of one pixel first, so the sums are
    comparable.

    Args:
        input_image (numpy.ndarray): The map before the convolution.
        input_header (astropy.io.fits.Header): Header of the map before the convolution.
        output_image (numpy.ndarray): The map after the convolution, on the reference grid.
        output_header (astropy.io.fits.Header): Header of the reference grid.

    Returns:
        totals (tuple): (input_total, output_total).
    """
    input_total = float(np.nansum(input_image))
    output_total = float(np.nansum(output_image[np.isfinite(output_image)]))
    if is_per_solid_angle_unit(input_header.get('BUNIT', '')):
        omega_in = pixel_solid_angle(input_header)
        omega_out = pixel_solid_angle(output_header)
        if omega_in is not None and omega_out is not None:
            input_total *= omega_in.value
            output_total *= omega_out.value
    return input_total, output_total


def convolve_maps(paths, wavelengths_um=None):
    """Convolve 2D maps to the PSF of the map with the longest wavelength, and reproject every map
    onto the grid of that map. This is the Batch Convolution of the 2D viewer, without the windows.
    A map is a flux per pixel (for example erg/s/cm2) or a surface brightness (for example MJy/sr),
    and the result keeps the unit of each map. Only MIRI and NIRSpec maps work.

    Args:
        paths (list): Paths of the 2D FITS maps, at least two.
        wavelengths_um (list or dict): Line wavelength of each map, in um. A list has one value per
            path, in order, and None keeps the value of that header. A dict has a path or a file name
            as key. A value that you give replaces the header value. A map needs a wavelength from the
            header (LINEWAVE, or LAMMIN and LAMMAX) or from this argument.

    Returns:
        result (dict): 'reference' (file name of the reference map), 'reference_wavelength_um',
            'dst_shape' and 'maps'. 'maps' has one dict for each input, in the order of paths, with
            'filename', 'wavelength_um', 'is_reference', 'image', 'coverage' (overlap fraction of each
            pixel), 'header' (ready to save), 'sigma_arcsec' (kernel size, 0 for the reference),
            'input_total' and 'output_total' (flux sums, for a check that flux is kept).

    Raises:
        ValueError: If fewer than two maps are given, a file cannot be read, two maps share no sky
            coverage, a map is not MIRI or NIRSpec, a unit cannot be used, a wavelength is missing,
            or a convolution gives no valid pixel.
    """
    paths = [str(path) for path in paths]
    if len(paths) < 2:
        raise ValueError("Give at least two maps to convolve.")
    infos = [load_map_for_convolution(path) for path in paths]

    given = {}
    if isinstance(wavelengths_um, dict):
        by_name = {os.path.basename(str(key)): value for key, value in wavelengths_um.items()}
        given = {info['filename']: by_name.get(info['filename']) for info in infos}
    elif wavelengths_um is not None:
        if len(wavelengths_um) != len(paths):
            raise ValueError("wavelengths_um must have one value for each path.")
        given = {info['filename']: value for info, value in zip(infos, wavelengths_um)}

    disjoint = [(a['filename'], b['filename']) for i, a in enumerate(infos) for b in infos[i + 1:]
                if not wcs_footprints_overlap(a['wcs'], a['shape'], b['wcs'], b['shape'])]
    if disjoint:
        pairs = ", ".join(f"{first} and {second}" for first, second in disjoint[:10])
        raise ValueError(f"These maps cover different regions of the sky: {pairs}. "
                         "Every map must share sky coverage with the others.")
    bad = unsupported_files([(info['filename'], info['header']) for info in infos])
    if bad:
        raise ValueError("Only MIRI and NIRSpec maps work, because the PSF formula is known only for "
                         "them: " + ", ".join(f"{name} ({instrument})" for name, instrument in bad))
    for info in infos:
        bunit = info['header'].get('BUNIT', '')
        if not (is_per_pixel_flux_unit(bunit) or is_per_solid_angle_unit(bunit)):
            raise ValueError(f"'{info['filename']}' has the unit '{bunit}'. It is not a flux per pixel "
                             "(for example erg/s/cm2) and not a surface brightness (for example MJy/sr).")
    for info in infos:
        value = given.get(info['filename'])
        info['wavelength_um'] = float(value) if value is not None else header_line_wavelength_um(info['header'])
        if info['wavelength_um'] is None:
            raise ValueError(f"'{info['filename']}' has no LINEWAVE, LAMMIN or LAMMAX in its header. "
                             "Give its wavelength in wavelengths_um.")

    ref_info = max(infos, key=lambda info: info['wavelength_um'])
    ref_header = ref_info['header'].copy()
    ref_header['NAXIS'] = 2
    ref_header['NAXIS1'] = ref_info['shape'][1]
    ref_header['NAXIS2'] = ref_info['shape'][0]

    maps = []
    for info in infos:
        bunit = info['header'].get('BUNIT', '')
        if info is ref_info:
            result_img = np.asarray(info['data'], dtype=float).copy()
            result_img[result_img == 0.0] = np.nan
            coverage = np.where(np.isfinite(result_img), 1.0, 0.0)
            sigma_arcsec = 0.0
        else:
            convolve_out = convolve_map_to_reference(
                info['data'], info['header'], info['wavelength_um'], info['pix_scale'],
                ref_info['data'], ref_header, ref_info['wavelength_um'])
            result_img = convolve_out['image']
            coverage = convolve_out['footprint']
            sigma_arcsec = convolve_out['sigma_arcsec']
        if not np.any(np.isfinite(result_img)):
            raise ValueError(f"The convolution gave no valid pixels for {info['filename']}.")

        out_header = build_resample_header(
            ref_header, result_img.shape, bunit=bunit or None,
            history=(f"Batch Convolution: sigma={sigma_arcsec:.4f}\"; "
                     f"reprojected onto {ref_info['filename']}'s native grid "
                     f"(ref wavelength {ref_info['wavelength_um']:.4f} um)"),
            description=convolved_description(info['header'], ref_info['wavelength_um']),
            line_header=info['header'],
        )
        input_total, output_total = convolution_flux_totals(
            info['data'], info['header'], result_img, ref_header)
        maps.append({
            'filename': info['filename'], 'wavelength_um': info['wavelength_um'],
            'is_reference': info is ref_info, 'image': result_img, 'coverage': coverage,
            'header': out_header, 'sigma_arcsec': sigma_arcsec,
            'input_total': input_total, 'output_total': output_total,
        })
    return {'reference': ref_info['filename'], 'reference_wavelength_um': ref_info['wavelength_um'],
            'dst_shape': ref_info['shape'], 'maps': maps}


def save_convolved_maps(result, out_path):
    """Save the maps of convolve_maps as FITS files, one for each map, named matched_<file name>.
    Each file has the image, and the overlap fraction of each pixel in the extension COVERAGE.
    This is what the ZIP of the Batch Convolution of the 2D viewer holds.

    Args:
        result (dict): Result of convolve_maps.
        out_path (str): Path of a ZIP file (a path that ends with .zip), or of a folder. A folder
            is made if it does not exist.

    Returns:
        paths (list): The names of the files that were written. For a ZIP, the names inside it.
    """
    def write_all(folder):
        names = []
        for item in result['maps']:
            name = f"matched_{item['filename']}"
            write_fits_product(os.path.join(folder, name), item['image'], item['header'],
                               extensions=[build_coverage_extension(item['coverage'])])
            names.append(name)
        return names

    if str(out_path).lower().endswith('.zip'):
        with tempfile.TemporaryDirectory() as tmpdir:
            names = write_all(tmpdir)
            with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for name in names:
                    zipf.write(os.path.join(tmpdir, name), arcname=name)
        return names
    os.makedirs(out_path, exist_ok=True)
    return [os.path.join(out_path, name) for name in write_all(out_path)]
