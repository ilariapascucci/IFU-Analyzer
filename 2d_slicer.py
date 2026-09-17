import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import random
import os

def generate_clean_2d_slices(input_fits, num_slices=3, output_prefix="clean_slice"):
    """Save random spectral slices of a 3D FITS cube as standalone 2D FITS files.

    Args:
        input_fits (str): Path to the 3D FITS cube.
        num_slices (int): Number of random slices to extract.
        output_prefix (str): Filename prefix for the saved slices.

    Returns:
        None
    """
    print(f"Opening 3D cube: {input_fits}")
    
    with fits.open(input_fits) as hdul:
        # Find the extension with the 3D data (usually 'SCI' in JWST, or Primary)
        data = None
        header = None
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 3:
                data = hdu.data
                header = hdu.header
                break
                
        if data is None:
            print("Error: Could not find 3D image data in the FITS file.")
            return

        # Squeeze out any empty 4th dimensions (like Stokes) just in case
        data = np.squeeze(data)
        if data.ndim != 3:
            print(f"Error: Expected 3D data, got {data.ndim}D data.")
            return

        print(f"Found 3D data cube with shape: {data.shape}")
        
        # 1. Extract ONLY the 2D Celestial WCS (RA/Dec), stripping the spectral axis,
        # velocity aberration, and 3D rotation matrices (PC1_3, etc.) that crash 2D viewers.
        wcs3d = WCS(header)
        wcs2d = wcs3d.celestial 
        
        # 2. Generate a mathematically pure 2D header from the clean WCS
        clean_header = wcs2d.to_header()
        
        # 3. Copy over essential visual/physical keywords from the original header
        keys_to_keep = ['BUNIT', 'TELESCOP', 'INSTRUME', 'FILTER', 'GRATING', 'OBJECT', 'TARGNAME']
        for key in keys_to_keep:
            if key in header:
                clean_header[key] = header[key]
                
        # Force strict 2D definition
        clean_header['NAXIS'] = 2
        clean_header['WCSAXES'] = 2

        # 4. Pick random slices along the spectral axis (axis 0 in numpy)
        max_idx = data.shape[0] - 1
        random_indices = random.sample(range(0, max_idx), num_slices)
        
        print(f"Generating {num_slices} random slices at indices: {random_indices}")
        
        # 5. Extract and save the slices
        for i, idx in enumerate(random_indices):
            # Extract the 2D array
            slice_data = data[idx, :, :]
            
            # Add a comment to the header so you know where it came from
            slice_header = clean_header.copy()
            slice_header['SLICEIDX'] = (idx, 'Original 3D cube spectral index')
            
            # Create a new FITS file with just this 2D data and the pure header
            hdu_new = fits.PrimaryHDU(data=slice_data, header=slice_header)
            
            out_name = f"{output_prefix}_{i+1}_idx{idx}.fits"
            hdu_new.writeto(out_name, overwrite=True)
            print(f"  -> Saved: {out_name}")

if __name__ == "__main__":
    # --- Instructions ---
    # Replace 'your_jwst_cube.fits' with the path to your actual 3D IFU FITS file
    input_file = "Level3_ch2-long_s3d.fits"  
    
    if os.path.exists(input_file):
        generate_clean_2d_slices(input_file, num_slices=3)
    else:
        print(f"Please update the 'input_file' variable. Could not find: {input_file}")