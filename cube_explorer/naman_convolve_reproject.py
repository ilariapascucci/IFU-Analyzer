"""
convolved_reproject.ipynb, ported for GUI use. gaussian_convolve_match/save_reprojected_fits
are verbatim; reproject_flux_maps has one agreed fix (marked inline): convert flux to
surface brightness before reprojecting, back to flux after -- reproject_exact preserves brightness, not flux.
"""

import numpy as np
from astropy.convolution import convolve_fft, Gaussian2DKernel
from reproject import reproject_exact
from astropy.io import fits

from .unit_converter import pixel_solid_angle


def gaussian_convolve_match(images,
                            wavelengths_um,
                            target_wavelength_um,
                            pixel_scales_arcsec):
    """
    Convolve a JWST/MIRI line-flux map to the PSF of a longer wavelength.

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

    Returns
    -------
    out : 2D ndarray
        PSF-matched image.

    Notes
    -----
    Assumes the MIRI PSF is Gaussian with

        FWHM(") = 0.033 * lambda(um) + 0.106

    NaNs are interpolated during convolution and restored afterwards.
    Flux is preserved by using a normalized convolution kernel.
    """
    output_images = []

    for i in range(len(images)):
        image = np.asarray(images[i], dtype=float)
        image[np.where(image == 0.0)] = np.nan

        # MIRI PSF FWHM (arcsec)
        fwhm_in = 0.033 * wavelengths_um[i] + 0.106
        fwhm_target = 0.033 * target_wavelength_um[0] + 0.106

        if fwhm_target <= fwhm_in:
            print(f"Skipping convolution for {wavelengths_um[i]} um -> {target_wavelength_um[0]} um")
            continue

        # Convert FWHM -> sigma (arcsec)
        sigma_in = fwhm_in / 2.35482
        sigma_target = fwhm_target / 2.35482

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

    # FIX (agreed with Naman): convert flux to surface brightness before
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

        # FIX: surface brightness -> flux, the output grid's own pixel solid angle
        image_out = image_out * omega_out.value

        reprojected_images.append(image_out)
        footprints.append(footprint)

    return reprojected_images, footprints


def save_reprojected_fits(image, output_header, filename,
                          overwrite=True,
                          history=None):
    """
    Save a reprojected flux map as a FITS file with the correct WCS header.

    Parameters
    ----------
    image : 2D ndarray
        Reprojected image (e.g., W m^-2 pixel^-1).

    output_header : astropy.io.fits.Header
        Header defining the final WCS/grid.

    filename : str
        Output FITS filename.

    overwrite : bool, optional
        Overwrite existing file.

    history : str or list of str, optional
        Additional HISTORY entries.

    Returns
    -------
    None
    """

    # Make a copy so the original header is not modified
    header = output_header.copy()

    # Ensure dimensions match the saved array
    header["NAXIS"] = 2
    header["NAXIS1"] = image.shape[1]
    header["NAXIS2"] = image.shape[0]

    # Add optional provenance
    header.add_history(
        "Image PSF-matched and reprojected using reproject_exact."
    )

    if history is not None:
        if isinstance(history, str):
            history = [history]

        for h in history:
            header.add_history(h)

    # Write FITS
    hdu = fits.PrimaryHDU(
        data=np.asarray(image, dtype=float),
        header=header
    )

    hdu.writeto(
        filename,
        overwrite=overwrite
    )




from .analyzer_2d import _make_2d_wcs


def convolution_plan_naman(native_result):
    """Cheap, no-op preview of what convolve_reproject_naman will do to each
    line -- the same FWHM(") = 0.033*lambda+0.106 arithmetic
    gaussian_convolve_match uses, without running any convolution or
    reprojection. Same shape as rotational_diagram_map.convolution_plan(),
    so the GUI's existing plan-confirmation table works unmodified.

    Args:
        native_result (dict): Output of compute_native_line_maps.

    Returns:
        plan (dict): 'worst_pix_arcsec' (reference line's native pixel
            scale), 'dst_shape' (reference line's native shape), 'per_cube'
            (list of dicts: 'wavelength', 'filename', 'native_pix_arcsec',
            'sigma_arcsec', 'sigma_pix'), 'dropped_lines'.
    """
    kept_lines = native_result["kept_lines"]
    flux_maps = native_result["native_flux_maps"]
    scales = native_result["native_pix_arcsec"]
    filenames = native_result["cube_filenames"]

    ref_idx = int(np.argmax([line["wavelength"] for line in kept_lines]))
    ref_wl = kept_lines[ref_idx]["wavelength"]
    fwhm_target = 0.033 * ref_wl + 0.106

    per_cube = []
    for i, line in enumerate(kept_lines):
        wl = line["wavelength"]
        if i == ref_idx:
            per_cube.append({"wavelength": wl, "filename": filenames[i],
                             "native_pix_arcsec": scales[i],
                             "sigma_arcsec": 0.0, "sigma_pix": 0.0})
            continue
        fwhm_in = 0.033 * wl + 0.106
        sigma_in = fwhm_in / 2.35482
        sigma_target = fwhm_target / 2.35482
        sigma_kernel_arcsec = (max(0.0, sigma_target**2 - sigma_in**2)) ** 0.5
        sigma_kernel_pix = sigma_kernel_arcsec / scales[i] if scales[i] > 0 else 0.0
        per_cube.append({"wavelength": wl, "filename": filenames[i],
                         "native_pix_arcsec": scales[i],
                         "sigma_arcsec": sigma_kernel_arcsec, "sigma_pix": sigma_kernel_pix})

    return {"worst_pix_arcsec": scales[ref_idx], "dst_shape": flux_maps[ref_idx].shape,
            "per_cube": per_cube, "dropped_lines": native_result["dropped_lines"]}


def convolve_reproject_naman(native_result):
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
        input_headers = []
        for i in other_idx:
            h = wcses[i].to_header()
            h["NAXIS"] = 2
            h["NAXIS1"] = maps[i].shape[1]
            h["NAXIS2"] = maps[i].shape[0]
            input_headers.append(h)

        convolved_images = gaussian_convolve_match(
            images, wavelengths_um, [ref_wl], pixel_scales_arcsec)
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
        "dst_wcs": _make_2d_wcs(ref_header), "dst_shape": ref_shape,
        "worst_pix_arcsec": scales[ref_idx],
        "reference_wavelength": ref_wl, "reference_header": ref_header,
        "native_flux_maps": native_result["native_flux_maps"],
        "native_flux_error_maps": native_result["native_flux_error_maps"],
        "native_wcs": native_result["native_wcs"], "native_shapes": native_result["native_shapes"],
        "native_pix_arcsec": native_result["native_pix_arcsec"],
        "cube_filenames": native_result["cube_filenames"],
    }
