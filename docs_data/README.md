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
| `maps/area/` | Gaussian Area maps of the seven H2 lines |

## Cubes

| File | Target | Use in the guide |
|---|---|---|
| `cubes/NIRSpec/HH30/Level3_g395h-f290lp_s3d.fits` | HH30, NIRSpec G395H | Multi-Gaussian fit and flux map of the 56 CO v=1->0 lines (4.40 to 4.79 µm) and the H2 contaminating lines |
| `cubes/MIRI/HH30/` (7 cubes) | HH30, MIRI | Gaussian fits and maps of the H2 lines, and the H2 rotational diagram: fit, maps and convolution |

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

The `ch1-long` cube has one more use in the guide: continuum subtraction.

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
| `HH30_NIRSpec_CO_multi_gaussian_fit.txt` | The multi-Gaussian fit of the CO lines and the H2 contaminating lines |

To show a saved H2 fit on its spectrum, open the cube of that line (see the table above). Then use **Load Fit** in the Gaussian Fit menu. The fit and the Fit and Noise markers come back with no new fit.

### Parameters (`fit_results/parameters/`)

Each file is a tab-separated table of the parameters of one fit: center, amplitude, sigma, FWHM, area, area error and R². The file name is the same as the file of the fit, with `_parameters` in place of `_fit`.

## Maps

The Gaussian Area maps are in `maps/area/`. There is one map for each H2 line. Each file name has the name of the cube, then `gaussian_area_map`, then the line (`H2(1)` for S(1), and so on to `H2(7)`). Open the maps in the 2D viewer.

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
