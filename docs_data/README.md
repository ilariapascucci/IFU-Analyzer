# Sample data for the documentation

This folder holds the data files that the user guide uses. Use them to repeat the examples in the guide.

All cubes are JWST Level 3 data cubes. Open them with **Open FITS** in the main window, or with `load_cube()` in the API.

## Folders

| Folder | Contents |
|---|---|
| `cubes/` | The JWST data cubes, in one folder for each instrument and target |
| `lines/` | Line lists in `.csv` format |
| `fit_results/fits/` | Saved Gaussian fits (Save Fit) |
| `fit_results/parameters/` | The parameters of each fit as a table (Save Parameters) |
| `fit_results/` (the CSV files in the top folder) | The rotational diagram fits of the H2 lines |
| `maps/` | The CO flux map, a slice and a contour, and the Gaussian Area maps of the seven H2 lines (`area/`) with their convolved versions (`convolution/`) |
| `spectra/` | Saved spectra (`sample/`) and the continuum fits (`continuum_fit/`), and the files of one region (`region.txt`, `region.fits`) |

## Cubes

| File | Target | Use in the guide |
|---|---|---|
| `cubes/NIRSpec/HH30/Level3_g395h-f290lp_s3d.fits` | HH30, NIRSpec G395H | Multi-Gaussian fit and flux map of the 56 CO v=1->0 lines (4.40 to 4.79 µm) and the H2 contaminating lines |
| `cubes/MIRI/HH30/` (7 cubes) | HH30, MIRI | Gaussian fits and maps of the H2 lines, the convolution of the maps, and the H2 rotational diagram |

### HH30 MIRI cubes

Each cube contains one of the seven H2 v=0->0 S(1) to S(7) lines. The cubes have four different pixel grids, so they show the convolution step.

| Cube (`jw01282-c1024_t029_miri_…_s3d.fits`) | H2 line | Wavelength (µm) | Pixels (y × x) |
|---|---|---|---|
| `ch3-long` | S(1) | 17.034846 | 35 × 39 |
| `ch3-short` | S(2) | 12.278612 | 37 × 39 |
| `ch2-medium` | S(3) | 9.664911 | 31 × 37 |
| `ch2-short` | S(4) | 8.025041 | 31 × 37 |
| `ch1-long` | S(5) | 6.909509 | 33 × 39 |
| `ch1-medium` | S(6) | 6.108564 | 33 × 39 |
| `ch1-short` | S(7) | 5.511183 | 33 × 39 |

The `ch1-long` cube has one more use in the guide: continuum subtraction. The saved spectra in `spectra/` are all from this cube.

## Line lists

| File | Lines | Use in the guide |
|---|---|---|
| `lines/CO_lines.csv` | 56 CO lines | Multi-Gaussian fit and flux map |
| `lines/H2_ContaminatingLines.csv` | 10 H2 lines | Contaminating lines for the flux map |
| `lines/H2_lines_flux.csv` | 7 H2 lines with flux and flux error | Rotational diagram fit (1T and 2T) |
| `lines/H2_lines.csv` | 7 H2 lines, no flux | **Calculate Fluxes** and the upper-limit fit |
| `lines/H2_lines_fitOPR.csv` | 5 H2 lines with flux and a 75% flux error | **Fit OPR** |

## Fit results

The names of these files show the target, the instrument and the line. For example, `HH30_MIRI_H2_S5_…` is the H2 S(5) line in the HH30 MIRI cube.

### Fits (`fit_results/fits/`)

| File | Contents |
|---|---|
| `HH30_MIRI_H2_S1_gaussian_fit.txt` to `HH30_MIRI_H2_S7_gaussian_fit.txt` | One Gaussian fit for each of the seven H2 lines, on the spectrum of the entire cube |
| `HH30_MIRI_H2_S5_pixel_16_12_fit.txt` | The Gaussian fit of the H2 S(5) line at pixel x=16, y=12. This is a data export. **Load Fit** cannot load it. |
| `HH30_NIRSpec_CO_contam_multi_gaussian_fit.txt` | The multi-Gaussian fit of the 56 CO lines and the H2 contaminating lines, with the automatic continuum at its default settings |

To show a saved H2 fit on its spectrum, open the cube of that line (see the table above). Then use **Load Fit** in the Gaussian Fit menu. The fit and the Fit and Noise markers come back with no new fit.

To show the CO fit, open the NIRSpec cube. Then use **Load Fits** in the Lines & Fit menu.

### Parameters (`fit_results/parameters/`)

Each file is a tab-separated table of the parameters of one fit: center, amplitude, sigma, FWHM, area, area error and R². The multi-Gaussian file has one row for each line. The file name is the same as the file of the fit, with `_parameters` in place of `_fit`.

### Rotational diagram fits (`fit_results/`)

These files come from the rotational diagram window. Both use the fluxes in `lines/H2_lines_flux.csv`, an emitting radius of 17.5 au, a distance of 140 pc and a fixed OPR of 3.

| File | Contents |
|---|---|
| `H2_rotational_diagram_fit_given_flux.csv` | The 1T fit: T = 535.1 ± 4.8 K, Ncol = 6.22e21 ± 3.20e20 cm⁻² |
| `H2_rotational_diagram_fit_given_flux_2t.csv` | The 2T fit: T1 = 273.2 ± 9.6 K, T2 = 862.7 ± 34.1 K, N1 = 4.29e22 cm⁻², N2 = 4.57e20 cm⁻² |

## Maps

| File | Contents |
|---|---|
| `maps/slice.fits` | One image saved from the 2D viewer: the Area map of the H2 S(5) line (`ch1-long`) |
| `maps/contour.txt` | The contours of the same map at three levels. The header names the map and the unit of the levels. Load it with the contour loader of the 2D viewer. |
| `maps/CO_flux_map.fits` | The CO flux map of the NIRSpec cube. The extensions hold the number of detected lines for each pixel, and the flux and the number of detected lines of the contaminating H2 lines. |
| `maps/area/` | The Gaussian Area maps. There is one map for each H2 line. |
| `maps/convolution/` | The Area maps after the Batch convolution in the 2D viewer. The file names start with `matched_`. |

Each file name in `maps/area/` has the name of the cube, then `gaussian_area_map`, then the line (`H2_S1` for S(1), and so on to `H2_S7`). Open the maps in the 2D viewer.

In the Area maps, a pixel with an upper limit has no error value. The convolution does not know this, so it blends the upper-limit values into the pixels next to them. Do not compare the convolved maps with the maps of the rotational diagram window.

All the files in `maps/convolution/` are on the pixel grid of S(1) (`ch3-long`, 35 × 39 pixels) and have its resolution, because that line has the largest PSF. Each file keeps the header of its own line.

## Spectra

| File | Contents |
|---|---|
| `spectra/sample/HH30_MIRI_continuum.txt` | The spectrum of the entire `ch1-long` cube, with no continuum |
| `spectra/sample/HH30_MIRI_continuum_region.txt` | The spectrum of the region (the sum of the 106 spaxels in `spectra/region.txt`) |
| `spectra/sample/HH30_MIRI_continuum_region_norm.txt` | The same region spectrum, divided by its peak |
| `spectra/sample/HH30_MIRI_continuum_shiftfed.txt` | The spectrum of the entire cube, shifted by 500 km/s |
| `spectra/region.txt` | The region as a polygon of 228 points, in pixel and sky (RA/Dec) coordinates |
| `spectra/region.fits` | The Area map slice of S(5) with only the 106 pixels of the region (the other pixels are NaN) |
| `spectra/continuum_fit/HH30_MIRI_fit_auto_continuum.txt` | The same spectrum with the automatic continuum fit, at its default settings |
| `spectra/continuum_fit/HH30_MIRI_subtract_auto_continuum.txt` | The same spectrum after the subtraction of the automatic continuum |
| `spectra/continuum_fit/HH30_MIRI_points_custom_continuum.txt` | The 277 anchor points of the custom continuum (wavelength, flux). The fit is a natural cubic spline through these points. |
| `spectra/continuum_fit/HH30_MIRI_fit_custom_continuum.txt` | The same spectrum with the custom continuum fit |
| `spectra/continuum_fit/HH30_MIRI_subtract_custom_continuum.txt` | The same spectrum after the subtraction of the custom continuum |

The header of each file shows the continuum and its settings. Open a spectrum with **Load Spectrum**.

## Data in other folders

These folders are part of the repository. They are not repeated here.

- `hitran_local/`: HITRAN line lists for C2H2, CO, CO2, H2, H2O, HCN and OH (1 to 30 µm).
- `lst_local/`: the LST line catalog for the velocity map.
- `nirspec_disp/`: the NIRSpec resolving-power tables.

## Source

| Data | JWST program | Pipeline version |
|---|---|---|
| HH30, NIRSpec | 1621 | 1.13.4 |
| HH30, MIRI | 1282 | 1.19.1 |
