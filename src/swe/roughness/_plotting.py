"""Internal plotting helpers for the roughness module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from swe.roughness.forward import ForwardModelResult
    from swe.roughness.frequency import (
        FrequencyTimeSeriesResult,
        ScalingLawFitResult,
    )
    from swe.roughness.spatial import SpatialCorrelationResult
    from swe.roughness.surface import RoughnessPSDResult


def plot_frequency_time_series(
    result: FrequencyTimeSeriesResult,
    *,
    slip_velocity_times: NDArray[np.floating] | None = None,
    slip_velocity_values: NDArray[np.floating] | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> Figure:
    """Plot frequency evolution with optional velocity overlay.

    Parameters
    ----------
    result : FrequencyTimeSeriesResult
        Binned frequency time series.
    slip_velocity_times : ndarray, optional
        Times for velocity overlay (matplotlib date numbers).
    slip_velocity_values : ndarray, optional
        Velocity values for overlay.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # Scatter individual measurements
    ax1.scatter(
        result.all_times,
        result.all_frequencies,
        s=4,
        alpha=0.3,
        color="0.6",
        label="Individual",
        zorder=1,
    )

    # Binned medians with error bars
    valid = result.count_per_bin > 0
    ax1.errorbar(
        result.times[valid],
        result.median_frequency[valid],
        yerr=result.std_frequency[valid],
        fmt="ko-",
        markersize=5,
        capsize=3,
        linewidth=1.5,
        label="Median",
        zorder=2,
    )

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("Instantaneous Frequency Evolution")
    ax1.grid(True, alpha=0.3)

    # Optional velocity overlay on secondary axis
    if slip_velocity_times is not None and slip_velocity_values is not None:
        ax2 = ax1.twinx()
        ax2.plot(
            slip_velocity_times,
            slip_velocity_values,
            "r-",
            linewidth=2,
            alpha=0.7,
            label="Slip velocity",
        )
        ax2.set_ylabel("Slip Velocity", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    return fig


def plot_roughness_psd(
    result: RoughnessPSDResult,
    *,
    show_slope_fit: bool = True,
    highlight_dominant: bool = True,
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot roughness PSD in log-log space.

    Parameters
    ----------
    result : RoughnessPSDResult
        PSD result.
    show_slope_fit : bool
        If ``True``, overlay the best-fit power-law line.
    highlight_dominant : bool
        If ``True``, annotate the dominant wavelength.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.loglog(result.wavenumbers, result.psd, "k-", linewidth=1.5, label="PSD")

    if show_slope_fit and not np.isnan(result.spectral_slope):
        log_k = np.log10(result.wavenumbers)
        log_psd = np.log10(result.psd)
        finite = np.isfinite(log_k) & np.isfinite(log_psd)
        if np.any(finite):
            from scipy.stats import linregress

            reg = linregress(log_k[finite], log_psd[finite])
            fit_psd = 10.0 ** (reg.slope * log_k + reg.intercept)
            ax.loglog(
                result.wavenumbers,
                fit_psd,
                "r--",
                linewidth=1.5,
                label=f"Slope = {result.spectral_slope:.2f}",
            )

    if highlight_dominant and not np.isnan(result.dominant_wavelength):
        dom_k = 1.0 / result.dominant_wavelength
        ax.axvline(
            dom_k,
            color="blue",
            linestyle="--",
            alpha=0.7,
            label=f"\u03bb = {result.dominant_wavelength:.3f} m",
        )

    ax.set_xlabel("Wavenumber (1/m)")
    ax.set_ylabel("PSD (m\u00b3)")
    direction = f" ({result.profile_direction})" if result.profile_direction else ""
    ax.set_title(f"Surface Roughness PSD{direction}")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    return fig


def plot_scaling_law_fit(
    result: ScalingLawFitResult,
    *,
    figsize: tuple[float, float] = (14, 6),
) -> Figure:
    """Plot scaling law fit results.

    Two panels: (1) time series comparison of observed and predicted
    frequency, (2) frequency vs velocity scatter with fit line.

    Parameters
    ----------
    result : ScalingLawFitResult
        Scaling law fit result.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(ncols=2, figsize=figsize)

    # Panel 1: Time series comparison
    n = len(result.observed_frequency)
    x_idx = np.arange(n)

    axes[0].plot(
        x_idx,
        result.observed_frequency,
        "ko-",
        markersize=5,
        label="Observed",
    )
    if len(result.predicted_frequency) == n:
        axes[0].plot(
            x_idx,
            result.predicted_frequency,
            "r^--",
            markersize=5,
            label="Predicted",
        )

    axes[0].set_xlabel("Bin Index")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Observed vs Predicted Frequency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: f vs V scatter
    if len(result.slip_velocity) == n:
        axes[1].scatter(
            result.slip_velocity,
            result.observed_frequency,
            c="steelblue",
            edgecolors="black",
            s=50,
            alpha=0.7,
            label="Data",
        )

        if not np.isnan(result.lambda_0):
            v_range = np.linspace(
                result.slip_velocity.min(),
                result.slip_velocity.max(),
                50,
            )
            f_model = v_range / result.lambda_0
            axes[1].plot(
                v_range,
                f_model,
                "r-",
                linewidth=2,
                label=f"\u03bb\u2080 = {result.lambda_0:.2e} m (R\u00b2 = {result.r_squared:.3f})",
            )

    axes[1].set_xlabel("Slip Velocity (m/s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title("Scaling Law: f = V / \u03bb\u2080")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_spatial_frequency_map(
    x_grid: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    freq_grid: NDArray[np.floating],
    *,
    station_coords: NDArray[np.floating] | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Plot spatial frequency map as a pseudocolour grid.

    Parameters
    ----------
    x_grid : ndarray
        1-D x coordinates.
    y_grid : ndarray
        1-D y coordinates.
    freq_grid : ndarray
        2-D frequency grid, shape ``(len(y_grid), len(x_grid))``.
    station_coords : ndarray, optional
        Station positions ``(N, 2)`` or ``(N, 3)`` to overlay.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    xx, yy = np.meshgrid(x_grid, y_grid)
    pcm = ax.pcolormesh(xx, yy, freq_grid, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label="Frequency (Hz)")

    if station_coords is not None:
        coords = np.asarray(station_coords)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            marker="^",
            c="red",
            s=80,
            edgecolors="black",
            zorder=5,
            label="Stations",
        )
        ax.legend()

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Spatial Frequency Map")
    ax.set_aspect("equal")

    return fig


def plot_frequency_roughness_correlation(
    result: SpatialCorrelationResult,
    *,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """Plot scatter of frequency vs local roughness with regression.

    Parameters
    ----------
    result : SpatialCorrelationResult
        Correlation result.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    valid = ~np.isnan(result.local_roughness) & ~np.isnan(result.event_frequency)

    if np.any(valid):
        ax.scatter(
            result.local_roughness[valid],
            result.event_frequency[valid],
            c="steelblue",
            edgecolors="black",
            s=40,
            alpha=0.7,
            label="Events",
        )

        # Regression line
        if not np.isnan(result.correlation_coefficient):
            pred_valid = result.predicted_frequency[valid]
            sort_idx = np.argsort(result.local_roughness[valid])
            ax.plot(
                result.local_roughness[valid][sort_idx],
                pred_valid[sort_idx],
                "r-",
                linewidth=2,
                label=(f"r = {result.correlation_coefficient:.3f} (p = {result.p_value:.4f})"),
            )

    ax.set_xlabel(f"Local Roughness ({result.roughness_metric})")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Frequency\u2013Roughness Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_forward_model_comparison(
    result: ForwardModelResult,
    *,
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot forward model stress spectrum and predicted frequency.

    Parameters
    ----------
    result : ForwardModelResult
        Forward model result.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    valid = result.stress_frequencies > 0
    if np.any(valid):
        ax.loglog(
            result.stress_frequencies[valid],
            result.stress_spectrum[valid],
            "k-",
            linewidth=1.5,
            label="Stress spectrum",
        )

    # Predicted frequency
    if not np.isnan(result.predicted_frequency):
        ax.axvline(
            result.predicted_frequency,
            color="blue",
            linestyle="--",
            alpha=0.7,
            label=f"Predicted: {result.predicted_frequency:.2f} Hz",
        )

    # Observed frequency
    if result.observed_frequency is not None:
        ax.axvline(
            result.observed_frequency,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Observed: {result.observed_frequency:.2f} Hz",
        )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Stress Perturbation Spectrum")
    ax.set_title(f"Forward Model ({result.model_type})")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    return fig
