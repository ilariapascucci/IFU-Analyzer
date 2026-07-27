#!/usr/bin/env python
"""
Build the color-scale comparison figure for docs/user_guide/visualization.rst.

Renders the same cube slice under all four intensity scales as a single 2x2
figure, so the docs can show what "Log emphasises faint features" actually
means instead of only asserting it.

The normalization logic mirrors CubeCanvas.plot_slice() in
cube_explorer/visualization.py (see the `scale ==` branches around line 540).
It is duplicated here rather than imported because plot_slice() is a method on
a Qt canvas and instantiating it would require a running QApplication. If the
scaling behaviour in visualization.py changes, update this too.

Usage:
    python docs/make_scale_comparison.py

Output:
    docs/_static/scale_comparison.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, PowerNorm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUBE_PATH = os.path.join(REPO_ROOT, "data", "MIRI_CITau", "Level3_ch1-long_s3d.fits")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_static", "scale_comparison.png")

SCALES = ["Linear", "Log", "Sqrt", "Power (γ=2)"]
SLICE_INDEX = 0
CMAP = "magma"


def build_norm(image, scale):
    """
    Return (normalized_image, norm) for a scale.

    Mirrors the branches in visualization.py plot_slice(). Log and Sqrt shift
    non-positive data rather than clipping it, which is why the image itself
    can come back modified.
    """
    vmin, vmax = np.nanmin(image), np.nanmax(image)

    if scale == "Log":
        if vmin <= 0:
            image = image - vmin + 1e-10
            return image, LogNorm(vmin=np.nanmin(image), vmax=np.nanmax(image))
        return image, LogNorm(vmin=vmin, vmax=vmax)

    if scale == "Sqrt":
        if vmin < 0:
            image = image - vmin
            vmin, vmax = 0, np.nanmax(image)
        return image, PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

    if scale == "Power (γ=2)":
        return image, PowerNorm(gamma=2.0, vmin=vmin, vmax=vmax)

    return image, Normalize(vmin=vmin, vmax=vmax)


def main():
    if not os.path.exists(CUBE_PATH):
        print(f"[ERROR] sample cube not found: {CUBE_PATH}")
        print("        data/ is gitignored; this figure can only be rebuilt "
              "with the sample data present.")
        return 1

    sys.path.insert(0, REPO_ROOT)
    from cube_explorer.api import load_cube

    cube = load_cube(CUBE_PATH)
    image = cube.cube[SLICE_INDEX]
    wavelength = float(cube.wavelength[SLICE_INDEX].value
                       if hasattr(cube.wavelength, "value")
                       else cube.wavelength[SLICE_INDEX])

    print(f"[INFO] {cube.instrument} slice {SLICE_INDEX} @ {wavelength:.3f} um, "
          f"shape {image.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for ax, scale in zip(axes.ravel(), SCALES):
        scaled, norm = build_norm(image.copy(), scale)
        im = ax.imshow(scaled, origin="lower", interpolation="nearest",
                       cmap=CMAP, norm=norm)
        ax.set_title(scale, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"Flux / Intensity [{cube.flux_unit}]", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Intensity scales -- {cube.instrument} slice {SLICE_INDEX} "
        f"@ {wavelength:.3f} um",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130)
    print(f"[OK] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
