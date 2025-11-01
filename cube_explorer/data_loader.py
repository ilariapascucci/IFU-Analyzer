"""
Data loading and processing functions for different telescopes
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from .utils import TELESCOPE_LOADERS, detect_instrument

class CubeData:
    """
    A class to handle loading and processing of 3D spectral data cubes from FITS files.
    
    This class supports various instruments including MUSE, ALMA, MIRI, and NIRSpec.
    It handles memory-mapped access to large data files and provides methods for
    spectral analysis and coordinate transformations.

    Attributes:
        cube (numpy.ndarray): The 3D data cube, memory mapped when possible
        header (astropy.io.fits.Header): FITS header containing metadata
        wavelength (numpy.ndarray): Array of wavelength/velocity values
        wave_unit (str): Unit of the wavelength/velocity axis (Å, µm, or km/s)
        spec_axis (int): Index of the spectral axis (typically 0)
        spectrum (numpy.ndarray): Full cube spectrum (sum over spatial dimensions)
        wcs (astropy.wcs.WCS): World Coordinate System for coordinate transforms
        instrument (str): Detected instrument name
    """
    def __init__(self):
        # Initialize all attributes as None/empty
        self.cube = None  # The main 3D data array
        self.header = None  # FITS header
        self.wavelength = None  # Spectral coordinate array
        self.wave_unit = ""  # Units for spectral axis
        self.spec_axis = 0  # Which axis contains spectral information
        self.spectrum = None  # Integrated spectrum
        self.wcs = None  # WCS object for coordinate transforms
        self.instrument = None  # Name of instrument

    def load_from_file(self, filename):
        """
        Load and process a FITS data cube from a file.

        This method:
        1. Opens the FITS file using memory mapping
        2. Detects the instrument type from headers
        3. Loads data using appropriate instrument-specific loader
        4. Converts data to float32 to save memory while preserving precision
        5. Sets up WCS and wavelength information
        6. Computes the integrated spectrum

        Args:
            filename (str): Path to the FITS file to load

        Returns:
            bool: True if loading was successful

        Raises:
            ValueError: If instrument cannot be detected or data shape is invalid
            Exception: If any other error occurs during loading

        Note:
            Uses memory mapping when possible to handle large files efficiently.
            Automatically orients spectral axis to index 0 for consistent processing.
        """
        try:
            # Open file with memory mapping
            self.hdul = fits.open(filename, memmap=True)
            
            # Detect instrument
            self.instrument = detect_instrument(self.hdul)
            if self.instrument is None:
                raise ValueError("Could not detect instrument from FITS headers")

            # Load data using appropriate loader
            self.header, data = TELESCOPE_LOADERS[self.instrument](self.hdul)
            
            # Process data - keep as memory mapped array when possible
            data = np.squeeze(data)
            if data.ndim != 3:
                raise ValueError(f"Expected 3D cube after squeezing, but got shape {data.shape}")

            # Convert to float32 to save memory, but keep as memmap
            if not isinstance(data, np.ndarray) or not data.flags.writeable:
                # If data is not a writeable array, create a new memmap file
                import tempfile
                import os
                
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
            self.spec_axis = self._infer_spec_axis()
            if self.spec_axis != 0:
                # Create a view instead of copying for axis reordering
                self.cube = np.moveaxis(self.cube, self.spec_axis, 0)
                # After moving spectral axis to front, update spec_axis
                self.spec_axis = 0

            # Get wavelength information
            self.wavelength, self.wave_unit = self._get_wavelength()
            
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

    def _infer_spec_axis(self):
        """
        Determine which axis contains spectral information based on FITS header.
        
        Uses NAXIS3 keyword to determine spectral axis placement. This is important
        because different instruments may store the spectral axis in different positions
        (e.g., first or last axis).

        Returns:
            int: Index of spectral axis (0 = first axis, 2 = last axis for 3D cube)
                 Returns 0 by default if cannot determine from header.

        Note:
            Convention: FITS axis j corresponds to numpy axis data.ndim - j
        """
        try:
            naxis3 = self.header.get('NAXIS3', None)
            if naxis3 is not None:
                if naxis3 == self.cube.shape[0]:
                    return 0
                elif naxis3 == self.cube.shape[-1]:
                    return self.cube.ndim - 1
        except Exception:
            pass
        return 0

    def _get_wavelength(self):
        """
        Calculate wavelength/velocity array from FITS header based on instrument type.

        Uses instrument-specific calculations:
        - MUSE: Wavelength in Angstroms (Å)
        - ALMA: Velocity in km/s 
        - MIRI/NIRSpec: Wavelength in microns (µm)

        Uses standard FITS keywords:
        - CRVAL3: Reference value
        - CRPIX3: Reference pixel
        - CDELT3/CD3_3: Step size/transformation matrix

        Returns:
            tuple: (wavelength_array, unit_string)
                wavelength_array: numpy array of wavelength/velocity values
                unit_string: "Å" for Angstroms, "µm" for microns, or "km/s" for velocity

        Note:
            Returns None, "" if wavelength calculation fails
        """
        if self.instrument == "MUSE":
            crval3 = self.header["CRVAL3"]
            crpix3 = self.header["CRPIX3"]
            cd33 = self.header["CD3_3"]
            n_slices = self.header["NAXIS3"]
            wavelengths = crval3 + (np.arange(1, n_slices + 1) - crpix3) * cd33
            return wavelengths, "Å"

        elif self.instrument == "ALMA":
            crval3 = self.header["CRVAL3"]
            cdelt3 = self.header["CDELT3"]
            crpix3 = self.header["CRPIX3"]
            n_slices = self.header["NAXIS3"]
            velocities = crval3 + (np.arange(1, n_slices + 1) - crpix3) * cdelt3
            velocities /= 1000.0  # convert to km/s
            return velocities, "km/s"

        elif self.instrument in ["MIRI", "NIRSpec"]:
            crval3 = self.header["CRVAL3"]
            cdelt3 = self.header["CDELT3"]
            crpix3 = self.header["CRPIX3"]
            n_slices = self.header["NAXIS3"]
            wavelengths = crval3 + (np.arange(1, n_slices + 1) - crpix3) * cdelt3
            return wavelengths, "µm"

    def _compute_spectrum(self):
        """
        Compute the total spectrum by summing over spatial dimensions.

        Processes data in chunks to handle large cubes efficiently:
        1. Divides spectral axis into manageable chunks
        2. Loads and processes each chunk separately
        3. Combines results into final spectrum

        Returns:
            numpy.ndarray: 1D array containing the summed spectrum
                          Returns zeros if computation fails

        Note:
            - Uses chunked processing to minimize memory usage
            - Handles memory-mapped arrays appropriately
            - Returns zeros if spectrum contains all NaN values
            - Automatically handles different spectral axis positions
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
        Extract a 2D slice from the data cube at a specific spectral index.

        Args:
            idx (int): Index along the spectral axis to extract

        Returns:
            numpy.ndarray: 2D array of the extracted slice, dtype=float32
                         None if extraction fails

        Note:
            - Handles different spectral axis positions automatically
            - Returns a copy of the data to avoid memory mapping issues
            - Converts to float32 for consistent numerical precision
        """
        try:
            if self.spec_axis == 0:
                return np.array(self.cube[idx], dtype=np.float32)
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

        Implements instrument-specific coordinate transformations:
        - MUSE: Uses CD matrix transformation
        - ALMA: Uses CDELT/CROTA-based transformation
        - MIRI/NIRSpec: Uses WCS or falls back to manual calculation
        - Other: Attempts WCS transformation if available

        Args:
            x (float): X pixel coordinate (0-based)
            y (float): Y pixel coordinate (0-based)
            instrument (str, optional): Override detected instrument type

        Returns:
            tuple: (ra, dec) in degrees
                  Both values None if transformation fails

        Note:
            - Uses appropriate transformation for each instrument
            - Handles rotation and scaling correctly
            - Falls back to WCS when available
            - Accounts for different coordinate conventions
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

            elif instrument == "ALMA":
                crval1 = float(self.header["CRVAL1"])
                crval2 = float(self.header["CRVAL2"])
                cdelt1 = float(self.header["CDELT1"])
                cdelt2 = float(self.header["CDELT2"])
                crpix1 = float(self.header["CRPIX1"])
                crpix2 = float(self.header["CRPIX2"])
                crota = float(self.header.get("CROTA2", 0.0))

                theta = np.deg2rad(crota)
                dx = x + 1 - crpix1
                dy = y + 1 - crpix2

                dra = dx * cdelt1 * np.cos(theta) - dy * cdelt2 * np.sin(theta)
                ddec = dx * cdelt1 * np.sin(theta) + dy * cdelt2 * np.cos(theta)

                ra = crval1 + dra
                dec = crval2 + ddec

            elif instrument in ["MIRI", "NIRSpec"]:
                if hasattr(self, "wcs"):
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

            elif hasattr(self, "wcs"):
                ra, dec = self.wcs.all_pix2world([[x, y]], 0)[0]

        except Exception as e:
            print(f"Coordinate conversion error: {e}")

        return ra, dec

    def compute_region_spectrum(self, mask):
        """
        Compute mean spectrum for pixels inside a given mask.
        
        Processes the cube in chunks to handle large datasets efficiently:
        1. Validates mask dimensions
        2. Counts valid pixels in mask
        3. Processes spectral chunks
        4. Computes mean spectrum across masked region

        Args:
            mask (numpy.ndarray): Boolean mask of same size as spatial dimensions
                                True values indicate pixels to include

        Returns:
            numpy.ndarray: 1D array containing mean spectrum of masked region
                         Returns zeros if computation fails or no valid pixels

        Raises:
            ValueError: If mask dimensions don't match data

        Note:
            - Uses chunked processing for memory efficiency
            - Handles memory mapping appropriately
            - Returns averaged spectrum (divide by number of pixels)
            - Works with any spectral axis position
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
        Extract spectrum for a single pixel from the data cube.

        This is a specialized and optimized version of region spectrum extraction
        for the common case of single pixel selection. It avoids the overhead
        of mask creation and region computation.

        Args:
            x (int): X pixel coordinate (0-based)
            y (int): Y pixel coordinate (0-based)

        Returns:
            numpy.ndarray: 1D array containing the pixel's spectrum in float32
                         None if extraction fails

        Note:
            - Faster than compute_region_spectrum for single pixels
            - Handles any spectral axis position
            - Returns a copy to avoid memory mapping issues
            - Uses fallback methods if simple indexing fails
        """
        if self.cube is None:
            return None

        try:
            if self.spec_axis == 0:
                # cube shape (nspec, ny, nx)
                return np.array(self.cube[:, int(y), int(x)], dtype='float32')
            else:
                # spectral axis not first; index appropriately
                slicing = [slice(None)] * self.cube.ndim
                slicing[1 if self.spec_axis != 1 else 0] = slice(None)
                # generic approach: take all along spectral axis
                idx = [slice(None)] * self.cube.ndim
                idx[self.spec_axis] = slice(None)
                idx[1 if self.spec_axis != 1 else 0] = int(y)
                idx[2 if self.spec_axis != 2 else 1] = int(x)
                # build a tuple with safe bounds: try common layout (spec,y,x)
                try:
                    return np.array(self.cube[:, int(y), int(x)], dtype='float32')
                except Exception:
                    # fallback: move spectral axis to front for indexing
                    tmp = np.moveaxis(self.cube, self.spec_axis, 0)
                    return np.array(tmp[:, int(y), int(x)], dtype='float32')
        except Exception as e:
            print(f"Error getting pixel spectrum: {e}")
            return None

    def free_memory(self):
        """
        Clean up resources and free memory.

        This method performs comprehensive cleanup:
        1. Closes open FITS files
        2. Deletes any temporary memory-mapped files
        3. Removes references to large data arrays
        4. Forces garbage collection

        Should be called when:
        - Loading a new file
        - Closing the application
        - Memory needs to be freed

        Note:
            - Safe to call multiple times
            - Handles missing attributes gracefully
            - Removes temporary files even if program crashes
        """
        import gc
        import os
        import tempfile
        
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