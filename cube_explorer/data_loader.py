"""
Data loading and processing functions for different telescopes
"""

import tempfile
import gc
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from .utils import TELESCOPE_LOADERS, detect_instrument, infer_spec_axis, get_wavelength

class CubeData:
    """
    Loads and processes 3D spectral data cubes (MUSE, MIRI, NIRSpec) from FITS files.

    Supports memory-mapped access to large files and provides methods for spectral
    analysis and coordinate transformations.

    Attributes:
        cube (array): The 3D data cube, memory mapped when possible
        header (astropy.io.fits.Header): FITS header containing metadata
        wavelength (astropy.units.Quantity): Array of wavelength values with units
        wave_unit (str): String representation of wavelength/velocity unit (Angstrom, um)
        spec_axis (int): Index of the spectral axis (typically 0)
        spectrum (array): Full cube spectrum (sum over spatial dimensions)
        wcs (astropy.wcs.WCS): World Coordinate System for coordinate transforms
        instrument (str): Detected instrument name
        filename (str): FITS file name
        flux_unit (str): String representation of flux unit (erg/cm**2*s*Angstrom, MJy/sr)
    """


    def __init__(self):
        """
        Initialize an empty CubeData with all attributes set to None/empty.

        Args:
            None

        Returns:
            None
        """
        self.cube = None
        self.header = None
        self.wavelength = None
        self.wave_unit = ""
        self.spec_axis = 0
        self.spectrum = None
        self.wcs = None
        self.instrument = None
        self.filename = None
        self.flux_unit = ""


    def load_from_file(self, filename):
        """
        Load and process a FITS data cube from a file.

        Args:
            filename (str): Path to the FITS file to load

        Returns:
            success (bool): True if loading succeeded

        Raises:
            Exception: if the file cannot be loaded; wraps the underlying error message
        """

        try:
            # Store the filename for later use
            self.filename = filename
            
            # Open file with memory mapping
            self.hdul = fits.open(filename, memmap=True)
            
            # Detect instrument
            self.instrument = detect_instrument(self.hdul)

            if self.instrument is None:
                raise ValueError("Could not detect instrument from FITS headers")
            
            # Load data using appropriate loader for detected instrument
            self.header, data = TELESCOPE_LOADERS[self.instrument](self.hdul)

            # Drop any length-1 axes, for example a stray Stokes axis in the FITS data
            data = np.squeeze(data)

            if data.ndim != 3:
                raise ValueError(
                    f"Expected a 3D spectral cube, but got shape {data.shape}. "
                    "For 2D FITS images, for example parameter or moment maps, use the 2D viewer instead."
                )
                
            print(f"Data shape after processing: {data.shape}")  # Debug info

            # Convert to float32 to save memory, but keep as memmap
            if not isinstance(data, np.ndarray) or not data.flags.writeable:                
                # Create a temporary memmap file
                temp_filename = os.path.join(tempfile.gettempdir(), 'cube_data.npy')
                shape = data.shape
                
                # Create memory-mapped array
                self.cube = np.memmap(temp_filename, dtype='float32', mode='w+', 
                                    shape=shape)
                
                # Copy data in chunks to avoid memory issues
                chunk_size = 100  # Adjust based on available memory
                for i in range(0, shape[0], chunk_size):
                    end = min(i + chunk_size, shape[0])
                    self.cube[i:end] = data[i:end].astype('float32')
            else:
                # If data is already a writeable array, convert in place
                self.cube = data.astype('float32')

            self.wcs = WCS(self.header)
            
            # Set up spectral axis
            self.spec_axis = infer_spec_axis(self.header, self.cube.shape)
            if self.spec_axis != 0:
                # Create a view instead of copying for axis reordering
                self.cube = np.moveaxis(self.cube, self.spec_axis, 0)
                # After moving spectral axis to front, update spec_axis
                self.spec_axis = 0

            # Get wavelength information (returns Quantity with units, plus any
            # instrument-specific metadata to apply onto this instance)
            wave_info = get_wavelength(self.header, self.instrument, self.cube.shape)
            self.wavelength = wave_info.pop('wavelength')
            self.wave_unit = wave_info.pop('unit')
            for key, value in wave_info.items():
                setattr(self, key, value)

            # Handle wavelength array size mismatch
            if self.wavelength is not None and len(self.wavelength) != self.cube.shape[self.spec_axis]:
                self.wavelength = self.wavelength[:self.cube.shape[self.spec_axis]]

            # Compute spectrum
            self.spectrum = self._compute_spectrum()
            
            # Ensure wavelength increases left->right
            if self.wavelength is not None and self.wavelength[0] > self.wavelength[-1]:
                self.wavelength = self.wavelength[::-1]
                self.spectrum = self.spectrum[::-1]

            return True

        except Exception as e:
            raise Exception(f"Error reading FITS: {e}")


    def _compute_spectrum(self):
        """
        Compute the total spectrum by summing over the spatial dimensions.

        Args:
            None

        Returns:
            spectrum (array): 1D summed spectrum. All zeros if the computation fails
        """

        if self.cube is None:
            return None
            
        axes = tuple(i for i in range(self.cube.ndim) if i != self.spec_axis)
        
        # Calculate spectrum in chunks to avoid loading entire cube into memory
        chunk_size = 100  # Adjust based on available memory
        n_spectral = self.cube.shape[self.spec_axis]
        spectrum = np.zeros(n_spectral, dtype='float32')
        
        try:
            for i in range(0, n_spectral, chunk_size):
                end = min(i + chunk_size, n_spectral)
                if self.spec_axis == 0:
                    chunk = self.cube[i:end]
                else:
                    slicing = [slice(None)] * self.cube.ndim
                    slicing[self.spec_axis] = slice(i, end)
                    chunk = self.cube[tuple(slicing)]
                
                # Handle potential memory mapping issues
                chunk = np.array(chunk)  # Ensure we have a regular array
                spectrum[i:end] = np.nansum(chunk, axis=axes)
            
            # Verify the spectrum is valid
            if np.all(np.isnan(spectrum)):
                print("Warning: Spectrum contains all NaN values")
                return np.zeros_like(spectrum)
            
            return spectrum
            
        except Exception as e:
            print(f"Error computing spectrum: {e}")
            return np.zeros(n_spectral, dtype='float32')


    def get_slice(self, idx):
        """
        Extract a 2D slice from the data cube at a given spectral index.

        Args:
            idx (int): Index along the spectral axis to extract

        Returns:
            slice_data (numpy.ndarray): 2D array, dtype float32, or None if extraction fails
        """

        try:
            if self.spec_axis == 0:
                return np.array(self.cube[idx], dtype=np.float32)  # float32 for consistency
            else:
                slicing = [slice(None)] * self.cube.ndim
                slicing[self.spec_axis] = idx
                return np.array(self.cube[tuple(slicing)], dtype=np.float32)
        except Exception as e:
            print(f"Error getting slice: {e}")
            return None


    def get_coords(self, x, y, instrument=None):
        """
        Convert pixel coordinates to sky coordinates (RA/DEC).

        Args:
            x (float): X pixel coordinate (0-based)
            y (float): Y pixel coordinate (0-based)
            instrument (str, optional): Override detected instrument type

        Returns:
            coords (tuple): (ra, dec) in degrees, or (None, None) if the transform fails
        """
        instrument = instrument or self.instrument
        ra = dec = None

        try:
            if instrument == "MUSE":
                crpix1 = float(self.header["CRPIX1"])
                crpix2 = float(self.header["CRPIX2"])
                crval1 = float(self.header["CRVAL1"])
                crval2 = float(self.header["CRVAL2"])
                cd11 = float(self.header.get("CD1_1", 0.0))
                cd12 = float(self.header.get("CD1_2", 0.0))
                cd21 = float(self.header.get("CD2_1", 0.0))
                cd22 = float(self.header.get("CD2_2", 0.0))

                dx = x + 1 - crpix1
                dy = y + 1 - crpix2

                dra = cd11 * dx + cd12 * dy
                ddec = cd21 * dx + cd22 * dy

                ra = crval1 + dra
                dec = crval2 + ddec

            elif instrument in ["MIRI", "NIRSpec"]:
                if self.wcs is not None:
                    ra, dec = self.wcs.all_pix2world([[x, y]], 0)[0]
                else:
                    crpix1 = float(self.header.get("CRPIX1", 0.0))
                    crpix2 = float(self.header.get("CRPIX2", 0.0))
                    crval1 = float(self.header.get("CRVAL1", np.nan))
                    crval2 = float(self.header.get("CRVAL2", np.nan))
                    cdelt1 = float(self.header.get("CDELT1", 0.0))
                    cdelt2 = float(self.header.get("CDELT2", 0.0))
                    crota = float(self.header.get("CROTA2", 0.0))
                    theta = np.deg2rad(crota)

                    dx = x + 1 - crpix1
                    dy = y + 1 - crpix2

                    dra = dx * cdelt1 * np.cos(theta) - dy * cdelt2 * np.sin(theta)
                    ddec = dx * cdelt1 * np.sin(theta) + dy * cdelt2 * np.cos(theta)

                    ra = crval1 + dra
                    dec = crval2 + ddec

            elif self.wcs is not None:
                ra, dec = self.wcs.all_pix2world([[x, y]], 0)[0]

        except Exception as e:
            print(f"Coordinate conversion error: {e}")

        return ra, dec


    def compute_region_spectrum(self, mask):
        """
        Compute the mean spectrum over the pixels inside a mask.

        Args:
            mask (array): Boolean mask, same spatial size as the cube. True marks a
                pixel to include

        Returns:
            spectrum (array): 1D mean spectrum of the masked region. All zeros if the
                computation fails or the mask has no True pixels

        Raises:
            ValueError: if mask is not a 2D array
        """

        # Expect spectral axis to be first (0). If not, move it temporarily.
        if self.cube is None:
            return None

        # Ensure mask is boolean and 2D
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("Mask must be a 2D boolean array")

        n_spectral = self.cube.shape[0] if self.spec_axis == 0 else self.cube.shape[self.spec_axis]
        spectrum = np.zeros(n_spectral, dtype='float32')
        chunk_size = 100  # Adjust based on available memory

        # Count valid pixels in mask for mean calculation
        n_pixels = np.count_nonzero(mask)
        if n_pixels == 0:
            return np.zeros(n_spectral, dtype='float32')

        try:
            for i in range(0, n_spectral, chunk_size):
                end = min(i + chunk_size, n_spectral)
                if self.spec_axis == 0:
                    chunk = self.cube[i:end]  # shape (nspec_chunk, ny, nx)
                else:
                    slicing = [slice(None)] * self.cube.ndim
                    slicing[self.spec_axis] = slice(i, end)
                    chunk = self.cube[tuple(slicing)]
                    # Move spectral axis to front
                    chunk = np.moveaxis(chunk, self.spec_axis, 0)

                # Multiply by mask (broadcast mask to shape)
                # Ensure chunk is array (in case of memmap)
                chunk = np.array(chunk, dtype='float32')
                masked = chunk * mask[None, :, :]

                # Sum over spatial axes and compute mean
                sum_spec = np.nansum(masked, axis=(1, 2))
                spectrum[i:end] = sum_spec / float(n_pixels)

            return spectrum
        except Exception as e:
            print(f"Error computing region spectrum: {e}")
            return np.zeros(n_spectral, dtype='float32')


    def get_pixel_spectrum(self, x, y):
        """
        Extract the spectrum of a single pixel from the data cube.

        Args:
            x (int): X pixel coordinate (0-based)
            y (int): Y pixel coordinate (0-based)

        Returns:
            spectrum (array): 1D pixel spectrum, dtype float32, or None if extraction fails
        """

        if self.cube is None:
            return None

        try:
            if self.spec_axis == 0:
                # cube shape (nspec, ny, nx)
                return np.array(self.cube[:, int(y), int(x)], dtype='float32')
            else:
                # The spectral axis is not first here. Try the common (spec, y, x)
                # layout first, and fall back to moving the axis if that fails.
                try:
                    return np.array(self.cube[:, int(y), int(x)], dtype='float32')
                except Exception:
                    # Fallback: move the spectral axis to the front, then index.
                    tmp = np.moveaxis(self.cube, self.spec_axis, 0)
                    return np.array(tmp[:, int(y), int(x)], dtype='float32')
        except Exception as e:
            print(f"Error getting pixel spectrum: {e}")
            return None


    def free_memory(self):
        """
        Clean up resources and free memory: close the FITS file, delete the memmap
        cache file, and delete the large cube/spectrum arrays. Logs a warning and
        continues on error, instead of raising.

        Args:
            None

        Returns:
            None
        """
        
        try:
            # Close FITS file if open
            if hasattr(self, 'hdul'):
                self.hdul.close()
                del self.hdul

            # Clean up memmap file if it exists
            temp_filename = os.path.join(tempfile.gettempdir(), 'cube_data.npy')
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

            # Delete large arrays
            if hasattr(self, 'cube'):
                del self.cube
            if hasattr(self, 'spectrum'):
                del self.spectrum
                
        except (AttributeError, OSError) as e:
            print(f"Warning during cleanup: {e}")
            
        gc.collect()