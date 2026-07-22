"""Robust LFE source parameters: moment, duration, and spectral falloff.

For low-frequency earthquakes (LFEs) a Brune corner frequency is not a
defensible source parameter: the corner sits at or below the usable band, the
high-frequency falloff is steeper than omega-squared, and the moment is not
recoverable from the corner under self-similar scaling (Bostock et al. 2017;
Supino et al. 2020; Chestler & Creager 2017; Thomas & Beroza 2016).  The
defensible observables are the seismic moment from the low-frequency spectral
plateau, the source-time-function width, and the high-frequency falloff
exponent.  This module provides those.

References
----------
Aki, K. & Richards, P.G. (2002). Quantitative Seismology, 2nd ed.
Bostock, M.G., Thomas, A.M., Rubin, A.M., Christensen, N.I. (2017). On corner
    frequencies, attenuation, and low-frequency earthquakes. JGR 122, 543-557.
Hanks, T.C. & Kanamori, H. (1979). A moment magnitude scale. JGR 84, 2348-2350.
Supino, M., Poiata, N., Festa, G., et al. (2020). Self-similarity of low-
    frequency earthquakes. Sci. Rep. 10, 6523.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from obspy.core.trace import Trace

logger = logging.getLogger(__name__)

__all__ = [
    "FalloffResult",
    "MomentResult",
    "SourceDurationResult",
    "high_frequency_falloff",
    "moment_from_spectral_level",
    "source_time_function_width",
]

# RMS radiation-pattern coefficients averaged over the focal sphere
# (Aki & Richards 2002): <F^P> = sqrt(4/15), <F^S> = sqrt(2/5).  Using the
# average makes the moment estimate independent of the (here unknown) mechanism.
_RADIATION_RMS = {"P": np.sqrt(4.0 / 15.0), "S": np.sqrt(2.0 / 5.0)}


class MomentResult(NamedTuple):
    """Seismic moment from the low-frequency spectral plateau."""

    seismic_moment: float
    """Seismic moment M0 in N*m."""
    moment_magnitude: float
    """Moment magnitude Mw (Hanks & Kanamori 1979)."""
    omega_0: float
    """Low-frequency displacement spectral level (m*s)."""
    plateau_freq_min: float
    """Lower edge of the plateau band used (Hz)."""
    plateau_freq_max: float
    """Upper edge of the plateau band used (Hz)."""
    radiation_coefficient: float
    """RMS-averaged radiation coefficient applied."""
    station: str
    """Station identifier."""


class SourceDurationResult(NamedTuple):
    """Source-time-function width from envelope/pulse fitting."""

    duration_s: float
    """Estimated source duration (s); interpretation is method-dependent."""
    method: str
    """``'gaussian_fwhm'`` or ``'envelope_halfwidth'``."""
    fit_quality: float
    """For the Gaussian fit, the R^2 of the envelope fit; NaN otherwise."""
    station: str
    """Station identifier."""


class FalloffResult(NamedTuple):
    """High-frequency spectral falloff exponent."""

    gamma: float
    """High-frequency falloff exponent (amplitude ~ f**-gamma)."""
    gamma_std: float
    """1-sigma uncertainty on gamma from the log-log linear fit."""
    freq_min: float
    """Lower edge of the high-frequency band fitted (Hz)."""
    freq_max: float
    """Upper edge of the high-frequency band fitted (Hz)."""
    station: str
    """Station identifier."""


def _to_displacement(trace: Trace, *, water_level: float) -> Trace:
    """Return a copy of *trace* in displacement, deconvolving if possible.

    Mirrors the response handling of
    :func:`swe.spectral.fitting.compute_displacement_spectrum`: if response
    metadata is attached the instrument response is removed to displacement,
    otherwise the data is assumed to already be displacement.
    """
    tr = trace.copy()
    if hasattr(tr.stats, "response") and tr.stats.response is not None:
        try:
            tr.remove_response(output="DISP", water_level=water_level)
        except (ValueError, TypeError, OSError) as exc:
            logger.warning(
                "Failed to remove response for %s.%s.%s: %s; assuming data is already displacement",
                tr.stats.network,
                tr.stats.station,
                tr.stats.channel,
                exc,
            )
    else:
        logger.debug(
            "No response metadata for %s.%s.%s; assuming data is already displacement",
            tr.stats.network,
            tr.stats.station,
            tr.stats.channel,
        )
    return tr


def _displacement_spectral_density(
    trace: Trace, *, water_level: float, taper_fraction: float
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Displacement amplitude spectral density (units m*s) vs frequency.

    Distinct from :func:`compute_displacement_spectrum`, which returns
    ``|FFT|/npts``; the spectral *density* needed for the moment formula is
    ``|FFT| * dt``.  For a displacement pulse of area ``A`` the low-frequency
    level of this density equals ``A`` (verified in tests), which is the
    quantity ``Omega_0`` in the moment relation.
    """
    tr = _to_displacement(trace, water_level=water_level)
    tr.taper(max_percentage=taper_fraction, type="cosine")
    data = tr.data.astype(np.float64)
    sr = tr.stats.sampling_rate
    dt = 1.0 / sr
    spectrum = np.abs(np.fft.rfft(data)) * dt  # amplitude spectral density (m*s)
    freqs = np.fft.rfftfreq(len(data), d=dt)
    return freqs, spectrum


def moment_from_spectral_level(
    trace: Trace,
    *,
    distance_m: float,
    density_kg_m3: float = 2700.0,
    velocity_m_s: float = 3500.0,
    phase: str = "S",
    free_surface_factor: float = 1.0,
    radiation_coefficient: float | None = None,
    plateau_freq_min: float = 0.5,
    plateau_freq_max: float = 2.0,
    water_level: float = 10.0,
    taper_fraction: float = 0.05,
) -> MomentResult:
    """Estimate seismic moment from the low-frequency displacement plateau.

    Uses ``M0 = 4*pi*rho*c**3*R*Omega_0 / (U * F)``, where ``Omega_0`` is the
    long-period displacement spectral level, ``rho`` the source density, ``c``
    the wave speed (S by default), ``R`` the hypocentral distance, ``U`` an
    RMS-averaged radiation coefficient (mechanism-independent), and ``F`` a
    site/free-surface factor.  This avoids any corner-frequency or
    self-similarity assumption and is the defensible moment estimator for LFEs.

    The borehole default ``free_surface_factor=1.0`` reflects sensors emplaced
    below the surface (no free-surface doubling); use ``2.0`` for surface
    stations.

    Parameters
    ----------
    trace : obspy.Trace
        Waveform (response is removed to displacement if metadata is attached;
        otherwise the data is assumed to already be displacement).
    distance_m : float
        Hypocentral distance (m) for geometric-spreading correction.
    density_kg_m3 : float
        Source-region density (kg/m^3).
    velocity_m_s : float
        Wave speed at the source (m/s); use the S speed for ``phase='S'``.
    phase : str
        ``'P'`` or ``'S'`` (selects the default RMS radiation coefficient).
    free_surface_factor : float
        Site amplification factor F (1.0 borehole, ~2.0 free surface).
    radiation_coefficient : float, optional
        Override the RMS-averaged radiation coefficient U.
    plateau_freq_min, plateau_freq_max : float
        Band (Hz) over which the plateau level Omega_0 is averaged; choose a
        band below the apparent corner.
    water_level : float
        Water level (dB) for response removal.
    taper_fraction : float
        Taper fraction before the FFT.

    Returns
    -------
    MomentResult
    """
    if distance_m <= 0:
        raise ValueError(f"distance_m must be positive, got {distance_m}")
    if phase not in _RADIATION_RMS:
        raise ValueError(f"phase must be 'P' or 'S', got {phase!r}")
    if plateau_freq_min >= plateau_freq_max:
        raise ValueError("plateau_freq_min must be < plateau_freq_max")

    u = radiation_coefficient if radiation_coefficient is not None else _RADIATION_RMS[phase]

    freqs, density = _displacement_spectral_density(
        trace, water_level=water_level, taper_fraction=taper_fraction
    )
    mask = (freqs >= plateau_freq_min) & (freqs <= plateau_freq_max)
    if not np.any(mask):
        raise ValueError(
            f"No spectral samples in plateau band [{plateau_freq_min}, {plateau_freq_max}] Hz"
        )
    omega_0 = float(np.mean(density[mask]))

    m0 = 4.0 * np.pi * density_kg_m3 * velocity_m_s**3 * distance_m * omega_0
    m0 /= u * free_surface_factor
    # Hanks & Kanamori (1979), M0 in N*m: Mw = (2/3) (log10 M0 - 9.05)
    mw = (2.0 / 3.0) * (np.log10(m0) - 9.05) if m0 > 0 else float("nan")

    station = getattr(trace.stats, "station", "")
    return MomentResult(
        seismic_moment=float(m0),
        moment_magnitude=float(mw),
        omega_0=omega_0,
        plateau_freq_min=float(plateau_freq_min),
        plateau_freq_max=float(plateau_freq_max),
        radiation_coefficient=float(u),
        station=station,
    )


def _gaussian(t: NDArray[np.floating], amp: float, t0: float, sigma: float) -> NDArray[np.floating]:
    return np.asarray(amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2))


def source_time_function_width(
    trace: Trace,
    *,
    method: str = "gaussian_fwhm",
    signal_window: tuple[float, float] | None = None,
) -> SourceDurationResult:
    """Estimate source duration from the waveform envelope.

    This is an **exploratory, model-dependent** estimate, not a true
    empirical-Green's-function deconvolution: it measures the width of the
    (analytic-signal) envelope of the stacked waveform, which is contaminated by
    attenuation and propagation and therefore upper-bounds the true source
    duration.  Report it with that caveat.

    Parameters
    ----------
    trace : obspy.Trace
        Stacked LFE waveform.
    method : str
        ``'gaussian_fwhm'`` fits a Gaussian to the envelope and returns its FWHM;
        ``'envelope_halfwidth'`` returns the half-amplitude width of the envelope.
    signal_window : tuple of float, optional
        ``(start, end)`` in seconds from the trace start to restrict the
        measurement; defaults to the whole trace.

    Returns
    -------
    SourceDurationResult
    """
    from scipy.signal import hilbert

    if method not in ("gaussian_fwhm", "envelope_halfwidth"):
        raise ValueError(f"unknown method {method!r}")

    tr = trace.copy()
    sr = tr.stats.sampling_rate
    data = tr.data.astype(np.float64)
    if signal_window is not None:
        s = max(0, round(signal_window[0] * sr))
        e = min(len(data), round(signal_window[1] * sr))
        if e <= s:
            raise ValueError(f"signal_window {signal_window} is empty for this trace")
        data = data[s:e]
    if len(data) < 4:
        raise ValueError("trace too short for duration estimation")

    envelope = np.abs(hilbert(data))
    t = np.arange(len(envelope)) / sr
    station = getattr(trace.stats, "station", "")

    if method == "envelope_halfwidth":
        peak = float(np.max(envelope))
        if peak <= 0:
            return SourceDurationResult(float("nan"), method, float("nan"), station)
        above = np.where(envelope >= 0.5 * peak)[0]
        duration = float((above[-1] - above[0] + 1) / sr)
        return SourceDurationResult(duration, method, float("nan"), station)

    # gaussian_fwhm
    peak = float(np.max(envelope))
    t0_guess = float(t[np.argmax(envelope)])
    sigma_guess = max(len(envelope) / 6.0 / sr, 1.0 / sr)
    try:
        popt, _ = curve_fit(
            _gaussian,
            t,
            envelope,
            p0=[peak, t0_guess, sigma_guess],
            bounds=([0.0, t[0], 1.0 / sr], [np.inf, t[-1], t[-1] - t[0] + 1.0 / sr]),
            maxfev=10000,
        )
        sigma = abs(popt[2])
        fwhm = float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma)
        fitted = _gaussian(t, *popt)
        ss_res = float(np.sum((envelope - fitted) ** 2))
        ss_tot = float(np.sum((envelope - np.mean(envelope)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return SourceDurationResult(fwhm, method, r2, station)
    except (RuntimeError, ValueError) as exc:
        logger.warning("Gaussian envelope fit failed: %s", exc)
        return SourceDurationResult(float("nan"), method, float("nan"), station)


def high_frequency_falloff(
    trace: Trace,
    *,
    freq_min: float,
    freq_max: float,
    water_level: float = 10.0,
    taper_fraction: float = 0.05,
) -> FalloffResult:
    """Fit the high-frequency spectral falloff exponent gamma.

    Fits ``log10(amplitude) = a - gamma * log10(f)`` over ``[freq_min, freq_max]``
    (chosen above the apparent corner).  LFEs typically show ``gamma`` near 3,
    steeper than the omega-squared (gamma=2) of ordinary earthquakes (Supino et
    al. 2020); reported as a source-complexity measure, not a Brune parameter.

    Parameters
    ----------
    trace : obspy.Trace
        Waveform (deconvolved to displacement if response metadata is attached).
    freq_min, freq_max : float
        High-frequency band (Hz) for the log-log slope fit.
    water_level : float
        Water level (dB) for response removal.
    taper_fraction : float
        Taper fraction before the FFT.

    Returns
    -------
    FalloffResult
    """
    if freq_min >= freq_max:
        raise ValueError("freq_min must be < freq_max")

    freqs, density = _displacement_spectral_density(
        trace, water_level=water_level, taper_fraction=taper_fraction
    )
    mask = (freqs >= freq_min) & (freqs <= freq_max) & (density > 0)
    if np.count_nonzero(mask) < 3:
        raise ValueError(
            f"Need >=3 positive spectral samples in [{freq_min}, {freq_max}] Hz; "
            f"got {int(np.count_nonzero(mask))}"
        )

    log_f = np.log10(freqs[mask])
    log_a = np.log10(density[mask])
    # Linear fit log_a = intercept + slope * log_f; gamma = -slope.
    coeffs, cov = np.polyfit(log_f, log_a, deg=1, cov=True)
    slope = coeffs[0]
    slope_std = float(np.sqrt(cov[0, 0]))
    station = getattr(trace.stats, "station", "")
    return FalloffResult(
        gamma=float(-slope),
        gamma_std=slope_std,
        freq_min=float(freq_min),
        freq_max=float(freq_max),
        station=station,
    )
