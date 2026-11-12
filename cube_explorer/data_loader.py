"""
Data loading and processing functions for different telescopes
"""

import tempfile
import gc
import os
import warnings
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from .utils import TELESCOPE_LOADERS, detect_instrument, infer_spec_axis, get_wavelength
from .fits_export import build_2d_export_header, write_fits_product
from .spectrum_file import prepare_spectrum, normalize_wave_unit

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
        spectrum_error (array): Flux error of each point of spectrum, or None. Only a spectrum file gives it.
        header_comments (list): The comment lines of the spectrum file, or None for a cube
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
        self.spectrum_error = None
        self.header_comments = None


    @property
    def is_spectrum_only(self):
        """
        Tell if the data is a single spectrum from a text file. It has no cube, no slices and no pixels.

        Args:
            None

        Returns:
            spectrum_only (bool): True if a spectrum is loaded and there is no cube
        """
        # free_memory deletes both attributes, so read them with getattr.
        return getattr(self, 'cube', None) is None and getattr(self, 'spectrum', None) is not None


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
            self.spectrum_error = None
            self.header_comments = None
            
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

            n_wave, n_y, n_x = self.cube.shape
            summary = (f"[OK] Loaded {os.path.basename(filename)}: {self.instrument}, {n_wave} wavelength points, "
                       f"{n_x} pixels wide by {n_y} pixels high")
            if self.wavelength is not None:
                wave_values = np.asarray(getattr(self.wavelength, 'value', self.wavelength), dtype=float)
                summary += f", {np.nanmin(wave_values):.4f} to {np.nanmax(wave_values):.4f} {self.wave_unit}"
            print(summary)

            return True

        except Exception as e:
            raise Exception(f"Error reading FITS: {e}")


    def load_from_arrays(self, wavelength, flux, wave_unit, flux_unit="", error=None, medium=None,
                         instrument=None, grating=None, object_name=None, filename=None):
        """
        Load one spectrum that is not from a cube, for example from a text file.
        The object then has no cube, no slices and no pixels. It has a small FITS header that
        holds the instrument, the grating, the wavelength medium and the flux unit, so the tools
        that read the header work on it in the same way as on a cube.

        Args:
            wavelength (array): Wavelengths in increasing order, in wave_unit
            flux (array): Flux at each wavelength, in flux_unit
            wave_unit (str): 'um' or 'Å'
            flux_unit (str): Flux unit, for example 'Jy' or 'MJy/sr'. An empty string means unknown.
            error (array): Flux error at each wavelength, in flux_unit, or None
            medium (str): 'air' or 'vacuum', or None if not known
            instrument (str): 'MIRI', 'NIRSpec' or 'MUSE', or None. Another name is kept as it is.
            grating (str): NIRSpec grating, for example 'G395H'. Needed for the resolving power of NIRSpec.
            object_name (str): Name of the object, or None
            filename (str): Path of the file that the spectrum came from, or None

        Returns:
            success (bool): True if loading succeeded

        Raises:
            ValueError: If wave_unit is not 'um' or 'Å', the arrays differ in length, or medium is not valid
        """
        wavelength = np.asarray(wavelength, dtype=float)
        flux = np.asarray(flux, dtype=float)
        if wave_unit not in ('um', 'Å'):
            raise ValueError(f"The wavelength unit must be 'um' or 'Å', got {wave_unit!r}.")
        if wavelength.shape != flux.shape or wavelength.ndim != 1:
            raise ValueError("The wavelength and flux must be 1D arrays of the same length.")
        if error is not None:
            error = np.asarray(error, dtype=float)
            if error.shape != flux.shape:
                raise ValueError("The error must have the same length as the flux.")
        if medium not in (None, 'air', 'vacuum'):
            raise ValueError(f"The medium must be 'air' or 'vacuum', got {medium!r}.")

        known = {'MIRI': 'MIRI', 'NIRSPEC': 'NIRSpec', 'MUSE': 'MUSE'}
        instrument = known.get(str(instrument).strip().upper(), instrument) if instrument else None

        header = fits.Header()
        if instrument:
            header['INSTRUME'] = str(instrument)
        if grating:
            header['GRATING'] = str(grating).strip().upper()
        if medium:
            header['CTYPE3'] = 'AWAV' if medium == 'air' else 'WAVE'
        if flux_unit:
            header['BUNIT'] = flux_unit
        if object_name:
            header['OBJECT'] = str(object_name)

        self.cube = None
        self.header = header
        self.wavelength = wavelength * (u.um if wave_unit == 'um' else u.AA)
        self.wave_unit = wave_unit
        self.spec_axis = 0
        self.spectrum = flux
        self.wcs = None
        self.instrument = instrument
        self.filename = filename
        self.flux_unit = flux_unit
        self.spectrum_error = error
        self.header_comments = None
        return True


    def load_from_table(self, table, settings):
        """
        Load the spectrum that a text file gives, after the columns and units are chosen.
        This removes the rows that cannot be used, sorts the rows by wavelength, and converts nm to um.

        Args:
            table (SpectrumTable): The table that spectrum_file.read_spectrum_table gives
            settings (dict): x_index, y_index, error_index, wave_unit, flux_unit, medium, instrument,
                grating and object, as spectrum_file.default_settings gives them

        Returns:
            notes (list): Sentences (str) that say what was changed in the data. Empty if nothing was changed.

        Raises:
            ValueError: If the columns, the units or the data are not valid
        """
        prepared = prepare_spectrum(table, settings["x_index"], settings["y_index"], settings["error_index"])
        unit, factor = normalize_wave_unit(settings["wave_unit"])
        notes = list(prepared["notes"])
        if factor != 1.0:
            notes.append(f"The wavelengths were converted from {settings['wave_unit']} to {unit}.")
        self.load_from_arrays(
            prepared["wavelength"] * factor, prepared["flux"], unit, settings["flux_unit"],
            error=prepared["error"], medium=settings["medium"], instrument=settings["instrument"],
            grating=settings["grating"], object_name=settings["object"], filename=table.path)
        self.header_comments = list(table.comments)
        wave = prepared["wavelength"] * factor
        print(f"[OK] Loaded {os.path.basename(table.path)}: spectrum file, {wave.size} wavelength points, "
              f"{wave.min():.4f} to {wave.max():.4f} {unit}")
        return notes


    def _compute_spectrum(self):
        """
        Compute the total spectrum by summing over the spatial dimensions.

        Args:
            None

        Returns:
            spectrum (array): 1D summed spectrum. All zeros if every value is NaN

        Raises:
            RuntimeError: if the sum over the spatial axes fails
        """

        if self.cube is None:
            return None
            
        axes = tuple(i for i in range(self.cube.ndim) if i != self.spec_axis)
        
        # Calculate spectrum in chunks to avoid loading entire cube into memory
        chunk_size = 100  # Adjust based on available memory
        n_spectral = self.cube.shape[self.spec_axis]
        spectrum = np.zeros(n_spectral, dtype='float32')
        has_data = False

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
                has_data = has_data or bool(np.isfinite(chunk).any())
                spectrum[i:end] = np.nansum(chunk, axis=axes)

            # nansum turns an all-NaN chunk into 0, so the sum cannot show missing data
            if not has_data:
                warnings.warn("The cube contains only NaN values. The summed spectrum is all zeros.")

            return spectrum
            
        except Exception as e:
            raise RuntimeError(f"Could not compute the total spectrum: {e}") from e


    def get_slice(self, idx):
        """
        Extract a 2D slice from the data cube at a given spectral index.

        Args:
            idx (int): Index along the spectral axis to extract

        Returns:
            slice_data (numpy.ndarray): 2D array, dtype float32

        Raises:
            ValueError: if no cube is loaded
            IndexError: if idx is outside the spectral axis
        """
        if self.cube is None:
            raise ValueError("No cube loaded")

        if self.spec_axis == 0:
            return np.array(self.cube[idx], dtype=np.float32)  # float32 for consistency
        else:
            slicing = [slice(None)] * self.cube.ndim
            slicing[self.spec_axis] = idx
            return np.array(self.cube[tuple(slicing)], dtype=np.float32)


    def compute_region_spectrum(self, mask):
        """
        Compute the mean spectrum over the pixels inside a mask.

        Args:
            mask (array): Boolean mask, same spatial size as the cube. True marks a
                pixel to include

        Returns:
            spectrum (array): 1D mean spectrum of the masked region. All zeros if the
                mask has no True pixels

        Raises:
            ValueError: if mask is not a 2D array
            RuntimeError: if the mean over the masked pixels fails
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
            raise RuntimeError(f"Could not compute the region spectrum: {e}") from e


    def export_slice_fits(self, idx, output_path, mask=None):
        """
        Write one 2D slice to a standalone FITS file (a plain PrimaryHDU).
        The header comes from a clean 2D WCS, plus a few descriptive
        keywords copied over from the cube's own header.

        Args:
            idx (int): Spectral index, as used by get_slice.
            output_path (str): Destination FITS path.
            mask (array, optional): Boolean mask, same shape as the slice;
                True keeps a pixel, False sets it to NaN. None exports the
                slice unmodified.

        Returns:
            None: Writes output_path as a side effect.

        Raises:
            ValueError: no cube is loaded, or mask's shape doesn't match the slice.
            IndexError: if idx is outside the spectral axis.
        """
        if self.cube is None:
            raise ValueError("No cube loaded")

        image = self.get_slice(idx)

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != image.shape:
                raise ValueError(f"Mask shape {mask.shape} does not match slice shape {image.shape}")
            image = image.copy()
            image[~mask] = np.nan

        # A plain PrimaryHDU: this file reloads through gui_2d.py's loader,
        # which scans every HDU.
        extra = {"DESC": "Slice of the cube"}
        if self.header is not None and "BUNIT" in self.header:
            extra["BUNIT"] = self.header["BUNIT"]
        if self.wavelength is not None and 0 <= idx < len(self.wavelength):
            wl_val = self.wavelength[idx]
            extra["DESC"] = f"Slice at {float(getattr(wl_val, 'value', wl_val)):.5f} {self.wave_unit or ''}".strip()
            # build_2d_export_header ASCII-sanitizes extra values/comments,
            # so wave_unit's "μm" doesn't need handling here too.
            extra["SLICEWL"] = (float(getattr(wl_val, "value", wl_val)), f"Wavelength of this slice ({self.wave_unit or ''})")

        out_header = build_2d_export_header(self.header, image.shape, extra=extra)
        write_fits_product(output_path, image.astype(np.float32), out_header)


    def get_pixel_spectrum(self, x, y):
        """
        Extract the spectrum of a single pixel from the data cube.

        Args:
            x (int): X pixel coordinate (0-based)
            y (int): Y pixel coordinate (0-based)

        Returns:
            spectrum (array): 1D pixel spectrum, dtype float32, or None if no cube is loaded

        Raises:
            IndexError: if (x, y) is outside the image
        """

        if self.cube is None:
            return None

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