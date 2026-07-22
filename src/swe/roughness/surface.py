"""Surface roughness PSD from elevation data.

Computes one- and two-dimensional power spectral densities of
topographic profiles and grids, with spectral slope fitting
and dominant-wavelength extraction.

References
----------
Candela, T. et al. (2012). Roughness of fault surfaces over nine
    decades of length scales. J. Geophys. Res., 117, B08409.
Renard, F. et al. (2013). Constant dimensionality of fault roughness
    from the scale of micro-fractures to the scale of continents.
    Geophys. Res. Lett., 40, 83-87.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.signal import detrend as sp_detrend
from scipy.signal import welch
from scipy.stats import linregress

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from swe.topo.raster import ElevationGrid

logger = logging.getLogger(__name__)

__all__ = [
    "RoughnessPSDResult",
    "compute_roughness_psd_1d",
    "compute_roughness_psd_2d",
    "dominant_roughness_wavelength",
]


class RoughnessPSDResult(NamedTuple):
    """Result of roughness PSD computation.

    Attributes
    ----------
    wavelengths : ndarray
        Wavelengths (m), sorted from longest to shortest.
    wavenumbers : ndarray
        Spatial wavenumbers (1/m).
    psd : ndarray
        Power spectral density (m^3 for 1-D, m^4 for 2-D radial).
    psd_2d : ndarray or None
        Full 2-D PSD array if computed, else ``None``.
    dominant_wavelength : float
        Wavelength of peak PSD (m).
    spectral_slope : float
        Best-fit slope in log-log space (negative for typical surfaces).
    spectral_slope_std : float
        Standard error of the spectral slope.
    profile_direction : str or None
        Direction label (e.g. ``"along-strike"``) if specified,
        else ``None``.
    """

    wavelengths: NDArray[np.floating]
    wavenumbers: NDArray[np.floating]
    psd: NDArray[np.floating]
    psd_2d: NDArray[np.floating] | None
    dominant_wavelength: float
    spectral_slope: float
    spectral_slope_std: float
    profile_direction: str | None


def compute_roughness_psd_1d(
    x: NDArray[np.floating],
    z: NDArray[np.floating],
    *,
    detrend: bool = True,
    window: str = "hann",
    n_segments: int | None = None,
    profile_direction: str | None = None,
) -> RoughnessPSDResult:
    """Compute 1-D roughness PSD along a surface profile.

    Uses Welch's method to estimate the power spectral density
    of surface heights as a function of spatial frequency.

    Parameters
    ----------
    x : ndarray
        Along-profile distance coordinates (m), assumed uniformly spaced.
    z : ndarray
        Surface height values (m) at each *x* position.
    detrend : bool
        If ``True``, remove a linear trend before computing the PSD.
    window : str
        Window function for Welch's method (e.g. ``"hann"``, ``"boxcar"``).
    n_segments : int, optional
        Number of overlapping segments for Welch's method.  If ``None``,
        defaults to ``max(1, len(z) // 256)``.
    profile_direction : str, optional
        Label for the profile direction (e.g. ``"along-strike"``).

    Returns
    -------
    RoughnessPSDResult
    """
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if len(x) != len(z):
        raise ValueError(f"x and z must have the same length ({len(x)} vs {len(z)})")
    if len(z) < 4:
        raise ValueError("Need at least 4 points for PSD estimation")

    dx = np.median(np.diff(x))
    if dx <= 0:
        raise ValueError("x must be monotonically increasing")

    if detrend:
        z = sp_detrend(z, type="linear")

    # Welch PSD parameters
    fs = 1.0 / dx  # spatial sampling frequency (1/m)
    if n_segments is None:
        n_segments = max(1, len(z) // 256)
    nperseg = max(4, len(z) // n_segments)

    freqs, psd = welch(z, fs=fs, window=window, nperseg=nperseg)

    # Discard DC component
    valid = freqs > 0
    freqs = freqs[valid]
    psd = psd[valid]

    wavelengths = 1.0 / freqs
    wavenumbers = freqs

    # Fit spectral slope in log-log space
    log_k = np.log10(wavenumbers)
    log_psd = np.log10(psd)
    finite = np.isfinite(log_k) & np.isfinite(log_psd)
    if np.sum(finite) >= 2:
        slope_result = linregress(log_k[finite], log_psd[finite])
        spectral_slope = slope_result.slope
        spectral_slope_std = slope_result.stderr
    else:
        spectral_slope = np.nan
        spectral_slope_std = np.nan

    # Dominant wavelength
    peak_idx = np.argmax(psd)
    dominant_wl = float(wavelengths[peak_idx])

    logger.info(
        "1-D roughness PSD: %d points, dominant wavelength=%.3f m, spectral slope=%.2f",
        len(z),
        dominant_wl,
        spectral_slope,
    )

    return RoughnessPSDResult(
        wavelengths=wavelengths,
        wavenumbers=wavenumbers,
        psd=psd,
        psd_2d=None,
        dominant_wavelength=dominant_wl,
        spectral_slope=spectral_slope,
        spectral_slope_std=spectral_slope_std,
        profile_direction=profile_direction,
    )


def compute_roughness_psd_2d(
    elevation_grid: ElevationGrid,
    *,
    detrend: bool = True,
    window: str = "hann",
) -> RoughnessPSDResult:
    """Compute 2-D roughness PSD from a gridded elevation surface.

    Uses a 2-D FFT and radial averaging to produce a 1-D
    representation of the power spectral density.

    Parameters
    ----------
    elevation_grid : ElevationGrid
        Gridded elevation data (from ``swe.topo.grid_from_raster``).
    detrend : bool
        If ``True``, remove a planar trend before computing the PSD.
    window : str
        Window function applied to each dimension.

    Returns
    -------
    RoughnessPSDResult
    """
    from scipy.signal import get_window

    elev = np.asarray(elevation_grid.elevation, dtype=np.float64)
    ny, nx = elev.shape

    dx = np.median(np.diff(elevation_grid.x))
    dy = abs(np.median(np.diff(elevation_grid.y)))

    if detrend:
        # Remove planar trend: z = a*x + b*y + c
        yy, xx = np.meshgrid(
            np.arange(ny, dtype=np.float64),
            np.arange(nx, dtype=np.float64),
            indexing="ij",
        )
        A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(nx * ny)])
        coeffs, _, _, _ = np.linalg.lstsq(A, elev.ravel(), rcond=None)
        trend = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
        elev = elev - trend

    # Apply 2-D window
    win_y = get_window(window, ny)
    win_x = get_window(window, nx)
    win_2d = np.outer(win_y, win_x)
    elev_windowed = elev * win_2d

    # 2-D FFT
    fft_2d = np.fft.fft2(elev_windowed)
    psd_2d = np.abs(np.fft.fftshift(fft_2d)) ** 2 / (nx * ny)

    # Frequency axes
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
    k_radial = np.sqrt(kx_grid**2 + ky_grid**2)

    # Radial averaging
    k_max = min(kx.max(), ky.max())
    n_bins = min(nx, ny) // 2
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_centres = 0.5 * (k_edges[:-1] + k_edges[1:])
    psd_radial = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (k_radial >= k_edges[i]) & (k_radial < k_edges[i + 1])
        if np.any(mask):
            psd_radial[i] = np.mean(psd_2d[mask])

    # Remove zero-frequency bin
    valid = k_centres > 0
    k_centres = k_centres[valid]
    psd_radial = psd_radial[valid]

    wavelengths = 1.0 / k_centres
    wavenumbers = k_centres

    # Spectral slope
    log_k = np.log10(wavenumbers)
    log_psd = np.log10(psd_radial)
    finite = np.isfinite(log_k) & np.isfinite(log_psd) & (psd_radial > 0)
    if np.sum(finite) >= 2:
        slope_result = linregress(log_k[finite], log_psd[finite])
        spectral_slope = slope_result.slope
        spectral_slope_std = slope_result.stderr
    else:
        spectral_slope = np.nan
        spectral_slope_std = np.nan

    # Dominant wavelength
    valid_psd = psd_radial > 0
    if np.any(valid_psd):
        peak_idx = np.argmax(psd_radial[valid_psd])
        dominant_wl = float(wavelengths[valid_psd][peak_idx])
    else:
        dominant_wl = np.nan

    logger.info(
        "2-D roughness PSD: (%d x %d) grid, dominant wavelength=%.3f m",
        ny,
        nx,
        dominant_wl,
    )

    return RoughnessPSDResult(
        wavelengths=wavelengths,
        wavenumbers=wavenumbers,
        psd=psd_radial,
        psd_2d=psd_2d,
        dominant_wavelength=dominant_wl,
        spectral_slope=spectral_slope,
        spectral_slope_std=spectral_slope_std,
        profile_direction=None,
    )


def dominant_roughness_wavelength(
    psd_result: RoughnessPSDResult,
    *,
    min_wavelength: float = 0.01,
    max_wavelength: float = 100.0,
    method: str = "peak",
) -> float:
    """Extract dominant roughness wavelength from a PSD result.

    Parameters
    ----------
    psd_result : RoughnessPSDResult
        PSD result from :func:`compute_roughness_psd_1d` or
        :func:`compute_roughness_psd_2d`.
    min_wavelength : float
        Minimum wavelength to consider (m).
    max_wavelength : float
        Maximum wavelength to consider (m).
    method : str
        Extraction method: ``"peak"`` (argmax of PSD) or ``"break"``
        (piecewise linear fit to find spectral break).

    Returns
    -------
    float
        Dominant roughness wavelength (m).

    Raises
    ------
    ValueError
        If *method* is not ``"peak"`` or ``"break"``.
    """
    if method not in ("peak", "break"):
        raise ValueError(f"method must be 'peak' or 'break', got {method!r}")

    wl = psd_result.wavelengths
    psd = psd_result.psd

    # Filter to wavelength range
    mask = (wl >= min_wavelength) & (wl <= max_wavelength)
    wl_filt = wl[mask]
    psd_filt = psd[mask]

    if len(wl_filt) == 0:
        logger.warning("No wavelengths in range [%.3f, %.3f] m", min_wavelength, max_wavelength)
        return np.nan

    if method == "peak":
        idx = np.argmax(psd_filt)
        return float(wl_filt[idx])

    # method == "break": piecewise linear fit in log-log
    log_k = np.log10(1.0 / wl_filt)
    log_psd = np.log10(psd_filt)
    finite = np.isfinite(log_k) & np.isfinite(log_psd)
    log_k = log_k[finite]
    log_psd = log_psd[finite]

    if len(log_k) < 4:
        logger.warning("Too few points for piecewise fit; falling back to peak")
        idx = np.argmax(psd_filt)
        return float(wl_filt[idx])

    # Find break point that minimises total residual
    best_rss = np.inf
    best_break_k = log_k[len(log_k) // 2]

    for split in range(2, len(log_k) - 2):
        r1 = linregress(log_k[:split], log_psd[:split])
        r2 = linregress(log_k[split:], log_psd[split:])
        rss = np.sum((log_psd[:split] - (r1.slope * log_k[:split] + r1.intercept)) ** 2)
        rss += np.sum((log_psd[split:] - (r2.slope * log_k[split:] + r2.intercept)) ** 2)
        if rss < best_rss:
            best_rss = rss
            best_break_k = log_k[split]

    dominant_wl = float(10.0 ** (-best_break_k))
    logger.info("Spectral break at wavelength=%.3f m", dominant_wl)
    return dominant_wl
