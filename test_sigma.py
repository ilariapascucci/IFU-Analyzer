import numpy as np
from astropy.io import fits

hdul = fits.open('data\\NIRSpec\\jw01621-o001_t001_nirspec_g140h-f100lp_s3d.fits')
err = hdul['ERR'].data  # shape (3915, 55, 53)

# Find index closest to 1.186 um
center_idx = 917  # from debug

# ERR values at the line wavelength, all spatial pixels
err_slice = err[center_idx, :, :]
print(f"ERR at 1.186 um: min={np.nanmin(err_slice):.4f}, max={np.nanmax(err_slice):.4f}, mean={np.nanmean(err_slice):.4f}, median={np.nanmedian(err_slice):.4f}")

# Center pixel specifically
print(f"ERR center pixel (27,26): {err[center_idx, 27, 26]:.4f} MJy/sr")

# Median over the 13+13 wavelength region around the line
err_center = err[:, 27, 26]  # center pixel ERR spectrum
region = np.concatenate([err_center[917-13:917], err_center[918:918+13]])
print(f"ERR 13+13 region: mean={np.nanmean(region):.4f}, std={np.nanstd(region):.4f}, median={np.nanmedian(region):.4f}")