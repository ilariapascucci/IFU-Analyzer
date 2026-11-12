# Sample data for the documentation

This folder holds the data files that the user guide uses. Use them to repeat the examples in the guide.

All cubes are JWST Level 3 data cubes. Open them with **Open FITS** in the main window, or with `load_cube()` in the API.

## Cubes

| File | Target | Use in the guide |
|---|---|---|
| `cubes/HH30_NIRSpec/Level3_g395h-f290lp_s3d.fits` | HH30, NIRSpec G395H | Multi-Gaussian fit and flux map of the 56 CO v=1->0 lines (4.40 to 4.79 µm) and the H2 contaminating lines |
| `cubes/HH30_MIRI/` (7 cubes) | HH30, MIRI | H2 rotational diagram: fit, maps and convolution |

### HH30 MIRI cubes

Each cube contains one of the seven H2 v=0->0 S(1) to S(7) lines. The cubes have three different pixel grids, so they show the convolution step.

| Cube (`jw01282-c1024_t029_miri_…_s3d.fits`) | H2 line (µm) | Pixels (y × x) |
|---|---|---|
| `ch1-short` | 5.511183 | 33 × 39 |
| `ch1-medium` | 6.108564 | 33 × 39 |
| `ch1-long` | 6.909509 | 33 × 39 |
| `ch2-short` | 8.025041 | 31 × 37 |
| `ch2-medium` | 9.664911 | 31 × 37 |
| `ch3-short` | 12.278612 | 37 × 39 |
| `ch3-long` | 17.034846 | 35 × 39 |

Two of these cubes have more uses in the guide:

- `ch1-long`: continuum subtraction, and the Gaussian fit of the first H2 line (6.909509 µm).
- `ch3-long`: the Gaussian fit of the second H2 line (17.034846 µm).

## Line lists

| File | Lines | Use in the guide |
|---|---|---|
| `lines/CO_lines.csv` | 56 CO lines | Multi-Gaussian fit and flux map |
| `lines/H2_ContaminatingLines.csv` | 10 H2 lines | Contaminating lines for the flux map |
| `lines/H2_lines_flux.csv` | 7 H2 lines with flux and flux error | Rotational diagram fit (1T and 2T) |
| `lines/H2_lines.csv` | 7 H2 lines, no flux | **Calculate Fluxes** and the upper-limit fit |
| `lines/H2_lines_fitOPR.csv` | 5 H2 lines with flux and a 75% flux error | **Fit OPR** |

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
