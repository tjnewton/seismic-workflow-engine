"""Internal plotting helpers for the spectral module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from obspy.core.trace import Trace
    from obspy.core.utcdatetime import UTCDateTime

    from swe.spectral.coda import CodaQFrequencyResult, CodaQResult
    from swe.spectral.fitting import SpectralFitResult, StressDropResult
    from swe.spectral.ratios import HVSRResult, SSRResult


def plot_hvsr(
    result: HVSRResult,
    *,
    show_individual: bool = True,
    confidence_band: float = 1.0,
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot HVSR results.

    Parameters
    ----------
    result : HVSRResult
        HVSR computation result.
    show_individual : bool
        Whether to show individual window curves.
    confidence_band : float
        Number of standard deviations for the confidence band.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if show_individual:
        for i in range(result.n_windows):
            ax.plot(
                result.frequencies,
                result.hvsr_per_window[i],
                color="0.8",
                linewidth=0.5,
                alpha=0.5,
            )

    # Confidence band
    lower = result.hvsr_mean - confidence_band * result.hvsr_std
    upper = result.hvsr_mean + confidence_band * result.hvsr_std
    ax.fill_between(
        result.frequencies,
        np.maximum(lower, 0),
        upper,
        alpha=0.3,
        color="steelblue",
        label=f"\u00b1{confidence_band:.0f}\u03c3",
    )

    # Mean curve
    ax.plot(
        result.frequencies,
        result.hvsr_mean,
        "k-",
        linewidth=1.5,
        label="Mean HVSR",
    )

    # Peak annotation
    ax.axvline(
        result.peak_frequency,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Peak: {result.peak_frequency:.2f} Hz",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("H/V Ratio")
    ax.set_title(f"HVSR - {result.station} ({result.n_windows} windows)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_ssr(
    result: SSRResult,
    *,
    show_individual: bool = True,
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot Standard Spectral Ratio results.

    Parameters
    ----------
    result : SSRResult
        SSR computation result.
    show_individual : bool
        Whether to show individual event curves.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if show_individual:
        for i in range(result.n_events):
            ax.plot(
                result.frequencies,
                result.ssr_per_event[i],
                color="0.8",
                linewidth=0.5,
                alpha=0.5,
            )

    ax.plot(
        result.frequencies,
        result.ssr_mean,
        "k-",
        linewidth=1.5,
        label="Mean SSR",
    )

    # Confidence band
    if result.n_events > 1:
        lower = result.ssr_mean - result.ssr_std
        upper = result.ssr_mean + result.ssr_std
        ax.fill_between(
            result.frequencies,
            np.maximum(lower, 0),
            upper,
            alpha=0.3,
            color="steelblue",
            label="\u00b11\u03c3",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Spectral Ratio")
    ax.set_title(
        f"SSR - {result.target_station} / {result.reference_station} ({result.n_events} events)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    return fig


def plot_coda_decay(
    trace: Trace,
    result: CodaQResult,
    origin_time: UTCDateTime,
    *,
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Plot coda decay with fitted envelope.

    Two-panel figure: waveform with coda window, and log-envelope
    with linear fit.

    Parameters
    ----------
    trace : obspy.Trace
        Original waveform.
    result : CodaQResult
        Coda-Q estimation result.
    origin_time : UTCDateTime
        Event origin time.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.signal import hilbert as _hilbert

    sr = trace.stats.sampling_rate
    times = np.arange(len(trace.data)) / sr
    envelope = np.abs(_hilbert(trace.data.astype(np.float64)))

    fig, axes = plt.subplots(nrows=2, figsize=figsize, sharex=True)

    # Panel 1: Waveform with coda window
    axes[0].plot(times, trace.data, "k-", linewidth=0.4)
    axes[0].axvline(
        result.coda_start_time,
        color="green",
        linestyle="--",
        label="Coda start",
    )
    axes[0].axvline(
        result.coda_end_time,
        color="red",
        linestyle="--",
        label="Coda end",
    )
    axes[0].axvspan(
        result.coda_start_time,
        result.coda_end_time,
        alpha=0.1,
        color="yellow",
    )
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(
        f"Coda decay - {result.station} (Qc={result.qc:.0f}, f={result.central_frequency:.1f} Hz)"
    )
    axes[0].legend(loc="upper right")

    # Panel 2: Log-envelope with fit
    origin_rel = float(origin_time - trace.stats.starttime)
    t_lapse = times - origin_rel
    valid = (t_lapse > 0) & (envelope > 0)

    if np.any(valid):
        log_env_t = np.full_like(times, np.nan)
        log_env_t[valid] = np.log(envelope[valid] * t_lapse[valid])
        axes[1].plot(times, log_env_t, "k-", linewidth=0.4, label="Observed")

        # Overlay the fitted line in the coda window
        if not np.isnan(result.qc):
            coda_mask = (times >= result.coda_start_time) & (times <= result.coda_end_time) & valid
            if np.any(coda_mask):
                t_coda = t_lapse[coda_mask]
                slope = -np.pi * result.central_frequency / result.qc
                # Fit intercept from the data
                y_obs = np.log(envelope[coda_mask] * t_coda)
                intercept = np.mean(y_obs - slope * t_coda)
                y_fit = intercept + slope * t_coda
                axes[1].plot(
                    times[coda_mask],
                    y_fit,
                    "r-",
                    linewidth=2,
                    label=f"Fit (R\u00b2={result.fit_r_squared:.2f})",
                )

    axes[1].axvline(result.coda_start_time, color="green", linestyle="--")
    axes[1].axvline(result.coda_end_time, color="red", linestyle="--")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("ln(A(t) \u00b7 t)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    return fig


def plot_coda_q_vs_frequency(
    result: CodaQFrequencyResult,
    *,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """Plot Qc vs frequency with power-law overlay.

    Parameters
    ----------
    result : CodaQFrequencyResult
        Multi-band coda-Q result.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    valid = ~np.isnan(result.qc_values) & (result.qc_values > 0)

    if np.any(valid):
        ax.errorbar(
            result.frequencies[valid],
            result.qc_values[valid],
            yerr=result.qc_errors[valid],
            fmt="ko",
            markersize=8,
            capsize=4,
            label="Observed Qc",
        )

        # Power-law overlay
        if not np.isnan(result.q0):
            f_model = np.logspace(
                np.log10(result.frequencies[valid].min()),
                np.log10(result.frequencies[valid].max()),
                50,
            )
            q_model = result.q0 * f_model**result.eta
            ax.plot(
                f_model,
                q_model,
                "r-",
                linewidth=2,
                label=(f"Qc = {result.q0:.0f} \u00b7 f^{result.eta:.2f}"),
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Qc")
    ax.set_title(f"Coda-Q vs Frequency - {result.station}")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    return fig


def plot_brune_fit(
    result: SpectralFitResult,
    *,
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot observed and fitted Brune source spectrum.

    Parameters
    ----------
    result : SpectralFitResult
        Brune model fit result.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.loglog(
        result.frequencies,
        result.observed_spectrum,
        "k-",
        linewidth=0.8,
        alpha=0.7,
        label="Observed",
    )

    if not np.all(np.isnan(result.fitted_spectrum)):
        ax.loglog(
            result.frequencies,
            result.fitted_spectrum,
            "r-",
            linewidth=2,
            label="Brune fit",
        )

        if not np.isnan(result.corner_frequency):
            ax.axvline(
                result.corner_frequency,
                color="blue",
                linestyle="--",
                alpha=0.7,
                label=f"fc = {result.corner_frequency:.2f} Hz",
            )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Displacement Spectrum")
    ax.set_title(f"Brune Fit - {result.station} (RMS={result.residual_rms:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    return fig


def plot_stress_drop_summary(
    results: list[StressDropResult],
    *,
    figsize: tuple[float, float] = (14, 6),
) -> Figure:
    """Plot stress drop summary: histogram and magnitude scatter.

    Parameters
    ----------
    results : list of StressDropResult
        Stress drop results from multiple events.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(ncols=2, figsize=figsize)

    stress_drops = np.array([r.stress_drop_mpa for r in results])
    valid = ~np.isnan(stress_drops) & (stress_drops > 0)

    # Panel 1: Histogram
    if np.any(valid):
        axes[0].hist(
            np.log10(stress_drops[valid]),
            bins=20,
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
        )
        median_sd = np.median(stress_drops[valid])
        axes[0].axvline(
            np.log10(median_sd),
            color="red",
            linestyle="--",
            label=f"Median: {median_sd:.2f} MPa",
        )

    axes[0].set_xlabel("log\u2081\u2080(\u0394\u03c3) [MPa]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Stress Drop Distribution")
    axes[0].legend()

    # Panel 2: Stress drop vs magnitude
    moments = np.array([r.seismic_moment for r in results])
    if np.any(valid):
        # Convert moment to Mw for x-axis
        mw = (np.log10(moments[valid]) - 9.05) / 1.5
        axes[1].scatter(
            mw,
            stress_drops[valid],
            c="steelblue",
            edgecolors="black",
            alpha=0.7,
            s=40,
        )
        axes[1].set_yscale("log")

    axes[1].set_xlabel("Mw")
    axes[1].set_ylabel("\u0394\u03c3 (MPa)")
    axes[1].set_title("Stress Drop vs Magnitude")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
