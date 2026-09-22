"""
Single-Pixel Fit Inspector for Gaussian Cube Fitting

Shows the fit stored in a Gaussian parameter map for one pixel: the spectrum,
the local baseline the fitter subtracted, the fitted Gaussian, and the
parameters/quality metrics recorded for that pixel.

The pixel is chosen by clicking it on the map (see GaussianMapResultsDialog).
This dialog only ever displays one pixel and never re-fits anything, so what it
shows is exactly what the maps contain.
"""

import numpy as np
from astropy.units import Quantity
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

from .gaussian_fitter import (
    gaussian, calculate_baseline, compute_line_free_noise_std, pixel_fit_status,
    STATUS_DETECTION, STATUS_UPPER_LIMIT,
)

# How much spectrum to show on either side of the fit window, as a fraction of the
# window width. Enough context to judge whether the endpoint baseline is sensible.
CONTEXT_PAD_FRACTION = 1.5


def _extract_wavelength_value(wavelength):
    """Extract numeric value from wavelength Quantity or return raw value.

    Args:
        wavelength (Quantity or ndarray): Wavelength array, with or without units.

    Returns:
        value (ndarray): Plain numeric wavelength array.
    """
    if isinstance(wavelength, Quantity):
        return wavelength.value
    return wavelength


class PixelFitDialog(QDialog):
    """
    Inspect the stored Gaussian fit for a single pixel of a Gaussian map.

    Args:
        parent: Parent widget
        cube: 3D data cube (n_wavelength, n_y, n_x)
        wavelength: 1D wavelength array (or Quantity) matching the cube
        lam_min, lam_max: The fit window (order does not matter)
        fit_results: The dict returned by fit_gaussians_per_pixel()
        row, col: Pixel to display (row = y index, col = x index)
        fac_above_stdres: Detection threshold in units of noise
        noise_min, noise_max: The line-free noise region used for the fit, if any
        flux_unit, wave_unit, area_unit: Display-only unit strings
    """

    def __init__(self, parent, cube, wavelength, lam_min, lam_max, fit_results,
                 row, col, fac_above_stdres=3.0, noise_min=None, noise_max=None,
                 flux_unit='', wave_unit='', area_unit=''):
        """Build the pixel inspector and render the stored fit immediately.

        Args:
            parent (QWidget): Parent widget.
            cube (ndarray): 3D data cube (n_wavelength, n_y, n_x).
            wavelength (ndarray or Quantity): 1D wavelength array matching the cube.
            lam_min (float): One edge of the fit window (order does not matter).
            lam_max (float): Other edge of the fit window.
            fit_results (dict): Result dict returned by fit_gaussians_per_pixel().
            row (int): Pixel row index (y).
            col (int): Pixel column index (x).
            fac_above_stdres (float): Detection threshold in units of noise.
            noise_min (float): Lower edge of the line-free noise region, or None.
            noise_max (float): Upper edge of the line-free noise region, or None.
            flux_unit (str): Display-only flux unit string.
            wave_unit (str): Display-only wavelength unit string.
            area_unit (str): Display-only area unit string.

        Returns:
            None
        """
        super().__init__(parent)
        self.cube = cube
        self.fit_results = fit_results
        self.row = int(row)
        self.col = int(col)
        self.fac_above_stdres = fac_above_stdres
        self.flux_unit = flux_unit or 'flux units'
        self.wave_unit = wave_unit or ''
        self.area_unit = area_unit or ''

        # Callers pass raw Left/Right click positions in either order; the fit
        # itself always used min/max.
        self.lam_min = min(lam_min, lam_max)
        self.lam_max = max(lam_min, lam_max)

        self.wavelength = _extract_wavelength_value(wavelength)
        self.wave_mask = (self.wavelength >= self.lam_min) & (self.wavelength <= self.lam_max)
        self.x_data = self.wavelength[self.wave_mask]

        # Noise region: recomputed here only to draw the detection band, using the same
        # estimator the fitter used, so the band matches the decision that was made.
        self.noise_std = None
        if noise_min is not None and noise_max is not None:
            nmin, nmax = min(noise_min, noise_max), max(noise_min, noise_max)
            noise_mask = (self.wavelength >= nmin) & (self.wavelength <= nmax)
            if np.count_nonzero(noise_mask) >= 2:
                try:
                    self.noise_std = compute_line_free_noise_std(
                        self.wavelength[noise_mask], cube[noise_mask, self.row, self.col])
                except Exception:
                    self.noise_std = None

        self.setWindowTitle(f"Pixel ({self.col}, {self.row}) Gaussian Fit")
        self.resize(1150, 640)

        self._setup_ui()
        self._plot_pixel()

    def _setup_ui(self):
        """Build the status label, plot canvas, and stats panel layout.

        Args:
            None

        Returns:
            None
        """
        layout = QVBoxLayout()

        self.status_label = QLabel()
        self.status_label.setStyleSheet(
            "background-color: #f0f0f0; padding: 6px; border-radius: 3px; font-weight: bold;")
        self.status_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        layout.addWidget(self.status_label)

        # Plot on the left, numbers on the right -- the stats are a tall narrow block,
        # so stacking them above a wide figure would waste most of the window.
        body = QHBoxLayout()

        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        plot_layout.addWidget(self.canvas)
        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)
        body.addWidget(plot_widget, stretch=3)

        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-family: Courier; font-size: 11px;")
        self.info_label.setAlignment(Qt.AlignTop)
        self.info_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        body.addWidget(self.info_label, stretch=1)

        layout.addLayout(body)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)

    def _plot_pixel(self):
        """Plot the stored spectrum, baseline, and fitted Gaussian for this pixel.

        Args:
            None

        Returns:
            None
        """
        row, col = self.row, self.col
        y_data = self.cube[self.wave_mask, row, col]

        amplitude = self.fit_results['amplitude'][row, col]
        center = self.fit_results['center'][row, col]
        sigma = self.fit_results['sigma'][row, col]
        area = self.fit_results['area'][row, col]
        chi2_red = self.fit_results['r_squared'][row, col]
        status = pixel_fit_status(self.fit_results, row, col)

        # Same endpoint-to-endpoint line the fitter subtracts -- local to the fit
        # window, not the continuum-subtraction tool, putting the Gaussian on a zero baseline.
        baseline = calculate_baseline(
            self.x_data[0], y_data[0], self.x_data[-1], y_data[-1], self.x_data, y_data)
        y_corrected = y_data - baseline

        # Only detections have stored parameters; every other outcome writes NaN.
        has_curve = (status == STATUS_DETECTION and
                     np.isfinite(amplitude) and np.isfinite(sigma) and np.isfinite(center))
        if has_curve:
            y_fit = gaussian(self.x_data, amplitude, center, sigma)
            residuals = y_corrected - y_fit
        else:
            y_fit = np.full_like(self.x_data, np.nan, dtype=float)
            residuals = np.full_like(self.x_data, np.nan, dtype=float)

        self.fig.clear()
        ax_data = self.fig.add_subplot(1, 1, 1)

        # Context: the surrounding spectrum, so the endpoint baseline can be judged
        # against what the line actually sits on.
        width = self.lam_max - self.lam_min
        pad = CONTEXT_PAD_FRACTION * width
        ctx_mask = ((self.wavelength >= self.lam_min - pad) &
                    (self.wavelength <= self.lam_max + pad))
        if np.count_nonzero(ctx_mask) > np.count_nonzero(self.wave_mask):
            ax_data.plot(self.wavelength[ctx_mask], self.cube[ctx_mask, row, col],
                         color='0.75', linewidth=1.0, zorder=1)
        ax_data.axvspan(self.lam_min, self.lam_max, color='0.92', zorder=0, label='Fit window')

        ax_data.plot(self.x_data, y_data, '-', color='k', linewidth=1.4,
                     label='Data', zorder=3)
        ax_data.plot(self.x_data, baseline, '--', color='tab:orange', linewidth=1.8,
                     label='Local baseline (window endpoints)', zorder=4)
        if has_curve:
            ax_data.plot(self.x_data, baseline + y_fit, '-', color='red', linewidth=2,
                         alpha=0.85, label='Baseline + Gaussian', zorder=5)
            ax_data.axvline(center, color='red', linestyle=':', linewidth=1.2, alpha=0.6)

        ax_data.set_xlabel(f'Wavelength [{self.wave_unit}]' if self.wave_unit else 'Wavelength')
        ax_data.set_ylabel(f'Flux [{self.flux_unit}]')
        ax_data.set_title(f'Pixel (x={col}, y={row})')
        ax_data.legend(loc='best', fontsize=8)
        ax_data.grid(True, alpha=0.3)
        ax_data.set_xlim(self.lam_min - pad, self.lam_max + pad)

        # Residual panel removed 2026-08-20: the old noise estimator read ~0.4x true sigma, so good
        # fits looked like 5-sigma failures. Fixed by compute_line_free_noise_std; the panel can return.

        self.fig.tight_layout()
        self.canvas.draw()

        self._update_text(status, amplitude, center, sigma, area, chi2_red,
                          y_data, y_corrected, residuals, has_curve)

    def _update_text(self, status, amplitude, center, sigma, area, chi2_red,
                     y_data, y_corrected, residuals, has_curve):
        """Render the status label and stats panel text for this pixel.

        Args:
            status (int): Pixel fit status code (detection, upper limit, or no data).
            amplitude (float): Fitted Gaussian amplitude, or NaN.
            center (float): Fitted Gaussian center wavelength, or NaN.
            sigma (float): Fitted Gaussian sigma, or NaN.
            area (float): Fitted or upper-limit line area.
            chi2_red (float): Reduced chi-squared stored for this pixel.
            y_data (ndarray): Raw flux values in the fit window.
            y_corrected (ndarray): Baseline-subtracted flux values.
            residuals (ndarray): Data minus model residuals, or NaN array.
            has_curve (bool): Whether a fitted curve exists to report against.

        Returns:
            None
        """
        if status == STATUS_DETECTION:
            status_text = "DETECTION - fitted, and passed the detection test"
        elif status == STATUS_UPPER_LIMIT:
            # Why it is a limit: a fit that converged on something too weak, or no fit
            # at all. Both store the same kind of number in 'area'.
            failed = self.fit_results.get('fit_failed')
            reason = ("the fit never ran here" if failed is not None and failed[self.row, self.col]
                      else "too weak to call a detection")
            status_text = (f"UPPER LIMIT - {reason}; "
                           f"area holds a {self.fac_above_stdres:g}σ limit")
        else:
            status_text = "NO DATA - the spectrum here is entirely NaN (outside the FOV)"
        self.status_label.setText(f"Pixel (x={self.col}, y={self.row})   |   {status_text}")

        def fmt(val, spec=".6e"):
            """Format a value, or return "NaN" if it is missing or non-finite.

            Args:
                val (float): Value to format.
                spec (str): Format spec string to apply.

            Returns:
                text (str): Formatted value, or "NaN".
            """
            if val is None or not np.isfinite(val):
                return "NaN"
            return format(val, spec)

        n_finite = int(np.count_nonzero(np.isfinite(y_data)))
        fwhm = 2.355 * sigma if np.isfinite(sigma) else np.nan
        lines = [
            "FIT PARAMETERS",
            "-" * 32,
            f" Center     {fmt(center, '.6f')} {self.wave_unit}",
            f" Amplitude  {fmt(amplitude)}",
            f"            {self.flux_unit}",
            f" Sigma      {fmt(sigma, '.6f')} {self.wave_unit}",
            f" FWHM       {fmt(fwhm, '.6f')} {self.wave_unit}",
            "",
            "INTEGRATED",
            "-" * 32,
            f" Area       {fmt(area)}",
            f"            {self.area_unit}",
        ]
        if status == STATUS_UPPER_LIMIT:
            lines.append(f"            ({self.fac_above_stdres:g}σ upper limit)")

        lines += [
            "",
            "FIT QUALITY",
            "-" * 32,
            f" chi2_red   {fmt(chi2_red, '.4f')}",
            f" Points     {n_finite} finite / {len(y_data)}",
        ]

        # R^2 (recomputed here) and reduced chi2 (from the map) answer different
        # questions, so both are shown rather than one standing in for the other.
        if has_curve:
            ss_res = np.nansum(residuals ** 2)
            ss_tot = np.nansum((y_corrected - np.nanmean(y_corrected)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            lines += [
                f" R^2        {fmt(r_squared, '.4f')}",
                f" Resid mean {fmt(np.nanmean(residuals))}",
                f" Resid std  {fmt(np.nanstd(residuals))}",
            ]

        noise_level = (self.fac_above_stdres * self.noise_std
                       if self.noise_std is not None else np.nan)
        lines += [
            "",
            "NOISE",
            "-" * 32,
            f" 1σ         {fmt(self.noise_std)}",
            f" {self.fac_above_stdres:g}σ level   {fmt(noise_level)}",
            "",
            "FIT WINDOW",
            "-" * 32,
            f" {self.lam_min:.6f}",
            f" {self.lam_max:.6f} {self.wave_unit}",
        ]

        self.info_label.setText("\n".join(lines))
