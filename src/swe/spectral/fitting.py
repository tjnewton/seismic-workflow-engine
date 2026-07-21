"""Brune source model fitting and stress drop estimation.

Fits the Brune (1970) omega-squared source model to displacement
spectra and estimates stress drop from the corner frequency.

References
----------
Brune, J.N. (1970). Tectonic stress and the spectra of seismic shear
    waves from earthquakes. J. Geophys. Res., 75(26), 4997-5009.
Brune, J.N. (1971). Correction [to Brune 1970]. J. Geophys. Res.,
    76(20), 5002.
Madariaga, R. (1976). Dynamics of an expanding circular fault.
    Bull. Seismol. Soc. Am., 66(3), 639-666.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from obspy.core.trace import Trace

logger = logging.getLogger(__name__)

__all__ = [
    "SpectralFitResult",
    "StressDropResult",
    "brune_spectrum",
    "compute_displacement_spectrum",
    "compute_stress_drop",
    "fit_brune_model",
    "spectral_fit_batch",
]


class SpectralFitResult(NamedTuple):
    """Result from Brune source model fitting."""

    corner_frequency: float
    """Corner frequency in Hz."""
    long_period_level: float
    """Long-period spectral level (Omega_0)."""
    corner_frequency_std: float
    """Standard deviation of corner frequency estimate."""
    long_period_level_std: float
    """Standard deviation of long-period level estimate."""
    frequencies: NDArray[np.floating]
    """Frequency array (Hz)."""
    observed_spectrum: NDArray[np.floating]
    """Observed displacement spectrum amplitudes."""
    fitted_spectrum: NDArray[np.floating]
    """Best-fit Brune model spectrum amplitudes."""
    residual_rms: float
    """RMS residual between observed and fitted (log-space)."""
    n_gamma: float
    """High-frequency falloff exponent used in the fit."""
    station: str
    """Station identifier."""


class StressDropResult(NamedTuple):
    """Result from stress drop estimation."""

    stress_drop_mpa: float
    """Stress drop in MPa."""
    corner_frequency: float
    """Corner frequency in Hz."""
    seismic_moment: float
    """Seismic moment in N*m."""
    source_radius_m: float
    """Source radius in metres."""
    spectral_fit: SpectralFitResult
    """Underlying spectral fit result."""


def brune_spectrum(
    frequencies: NDArray[np.floating],
    *,
    omega_0: float,
    corner_frequency: float,
    n_gamma: float = 2.0,
) -> NDArray[np.floating]:
    """Compute the Brune (1970) analytical source spectrum.

    Parameters
    ----------
    frequencies : array-like
        Frequency array in Hz.
    omega_0 : float
        Long-period spectral level.
    corner_frequency : float
        Corner frequency in Hz.
    n_gamma : float
        High-frequency falloff exponent (default 2 for omega-squared).

    Returns
    -------
    numpy.ndarray
        Spectral amplitudes at the given frequencies.
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    return omega_0 / (1.0 + (freqs / corner_frequency) ** n_gamma)


def compute_displacement_spectrum(
    trace: Trace,
    *,
    water_level: float = 10.0,
    taper_fraction: float = 0.05,
    freq_min: float = 0.5,
    freq_max: float | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the displacement amplitude spectrum of a trace.

    Removes the instrument response if metadata is present, then
    computes the FFT to obtain the displacement spectrum.

    Parameters
    ----------
    trace : obspy.Trace
        Input waveform.
    water_level : float
        Water level (dB) for spectral division during response removal.
    taper_fraction : float
        Fraction of the trace to taper at each end before FFT.
    freq_min : float
        Minimum frequency (Hz) for the returned spectrum.
    freq_max : float, optional
        Maximum frequency (Hz). Defaults to the Nyquist frequency.

    Returns
    -------
    tuple of (frequencies, amplitudes)
        Frequency and amplitude arrays of the displacement spectrum.
    """
    tr = trace.copy()

    # Remove instrument response if available
    if hasattr(tr.stats, "response") and tr.stats.response is not None:
        try:
            tr.remove_response(output="DISP", water_level=water_level)
        except (ValueError, TypeError, OSError) as exc:
            logger.warning(
                "Failed to remove response for %s.%s.%s: %s; "
                "assuming data is already displacement",
                tr.stats.network,
                tr.stats.station,
                tr.stats.channel,
                exc,
            )
    else:
        logger.debug(
            "No response metadata for %s.%s.%s; "
            "assuming data is already displacement",
            tr.stats.network,
            tr.stats.station,
            tr.stats.channel,
        )

    # Taper and compute FFT
    tr.taper(max_percentage=taper_fraction, type="cosine")
    data = tr.data.astype(np.float64)
    npts = len(data)
    sr = tr.stats.sampling_rate

    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(npts, d=1.0 / sr)
    amplitudes = np.abs(spectrum) / npts

    # Apply frequency band limits
    nyquist = sr / 2.0
    f_max = freq_max if freq_max is not None else nyquist
    mask = (freqs >= freq_min) & (freqs <= f_max)

    return freqs[mask], amplitudes[mask]


def _brune_log(
    log_freqs: NDArray[np.floating],
    log_omega_0: float,
    log_fc: float,
    n_gamma: float,
) -> NDArray[np.floating]:
    """Brune spectrum in log-log space for curve fitting."""
    freqs = np.power(10.0, log_freqs)
    fc = np.power(10.0, log_fc)
    omega_0 = np.power(10.0, log_omega_0)
    return np.log10(omega_0 / (1.0 + (freqs / fc) ** n_gamma))


def fit_brune_model(
    trace: Trace,
    *,
    freq_min: float = 0.5,
    freq_max: float | None = None,
    n_gamma: float = 2.0,
    water_level: float = 10.0,
    taper_fraction: float = 0.05,
) -> SpectralFitResult:
    """Fit the Brune source model to a trace's displacement spectrum.

    Performs least-squares fitting in log-space using
    ``scipy.optimize.curve_fit``.

    Parameters
    ----------
    trace : obspy.Trace
        Input waveform.
    freq_min : float
        Minimum frequency (Hz) for fitting.
    freq_max : float, optional
        Maximum frequency (Hz) for fitting.
    n_gamma : float
        High-frequency falloff exponent.
    water_level : float
        Water level (dB) for response removal.
    taper_fraction : float
        Taper fraction before FFT.

    Returns
    -------
    SpectralFitResult
        Fit results including corner frequency, spectral level,
        uncertainties, and the observed/fitted spectra.
    """
    station = (
        f"{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}"
    )

    freqs, amps = compute_displacement_spectrum(
        trace,
        water_level=water_level,
        taper_fraction=taper_fraction,
        freq_min=freq_min,
        freq_max=freq_max,
    )

    nan_result = SpectralFitResult(
        corner_frequency=float("nan"),
        long_period_level=float("nan"),
        corner_frequency_std=float("nan"),
        long_period_level_std=float("nan"),
        frequencies=freqs,
        observed_spectrum=amps,
        fitted_spectrum=np.full_like(amps, float("nan")),
        residual_rms=float("nan"),
        n_gamma=n_gamma,
        station=station,
    )

    if len(freqs) < 3:
        logger.warning("Too few frequency points for Brune fit on %s", station)
        return nan_result

    # Remove zero amplitudes to avoid log issues
    valid = amps > 0
    if np.sum(valid) < 3:
        logger.warning("Too few non-zero amplitudes for Brune fit on %s", station)
        return nan_result

    log_freqs = np.log10(freqs[valid])
    log_amps = np.log10(amps[valid])

    # Constrain the corner frequency to the fitted band.  Without this bound the
    # least-squares fit can diverge to a physically meaningless corner frequency
    # (far above the data) when the spectrum has no clear corner - e.g. a weak or
    # narrow-band stack.  Bounding fc to the band makes a cornerless spectrum
    # resolve to the band edge instead of running away.
    fc_lo = float(freqs[valid].min())
    fc_hi = float(freq_max) if freq_max is not None else float(freqs[valid].max())
    if fc_hi <= fc_lo:
        fc_hi = fc_lo * 1.0001
    log_fc_lo, log_fc_hi = np.log10(fc_lo), np.log10(fc_hi)

    # Initial guesses
    log_omega_0_guess = float(np.max(log_amps))
    log_fc_guess = float(
        np.clip(np.log10(np.median(freqs[valid])), log_fc_lo, log_fc_hi)
    )

    def _model(log_f: NDArray, log_o0: float, log_fc: float) -> NDArray:
        return _brune_log(log_f, log_o0, log_fc, n_gamma)

    try:
        popt, pcov = curve_fit(
            _model,
            log_freqs,
            log_amps,
            p0=[log_omega_0_guess, log_fc_guess],
            bounds=([-np.inf, log_fc_lo], [np.inf, log_fc_hi]),
            maxfev=5000,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning("Brune fit failed for %s: %s", station, exc)
        return nan_result

    log_omega_0_fit, log_fc_fit = popt
    omega_0_fit = 10.0 ** log_omega_0_fit
    fc_fit = 10.0 ** log_fc_fit

    # Uncertainties from covariance
    perr = np.sqrt(np.diag(pcov))
    # Propagate log-space errors: sigma_x = x * ln(10) * sigma_log_x
    omega_0_std = omega_0_fit * np.log(10.0) * perr[0]
    fc_std = fc_fit * np.log(10.0) * perr[1]

    # Fitted spectrum (over full frequency range)
    fitted = brune_spectrum(
        freqs, omega_0=omega_0_fit, corner_frequency=fc_fit, n_gamma=n_gamma
    )

    # RMS residual in log-space
    log_fitted = np.log10(np.maximum(fitted[valid], 1e-30))
    residual_rms = float(np.sqrt(np.mean((log_amps - log_fitted) ** 2)))

    return SpectralFitResult(
        corner_frequency=float(fc_fit),
        long_period_level=float(omega_0_fit),
        corner_frequency_std=float(fc_std),
        long_period_level_std=float(omega_0_std),
        frequencies=freqs,
        observed_spectrum=amps,
        fitted_spectrum=fitted,
        residual_rms=residual_rms,
        n_gamma=n_gamma,
        station=station,
    )


def compute_stress_drop(
    spectral_fit: SpectralFitResult,
    *,
    seismic_moment: float | None = None,
    magnitude: float | None = None,
    velocity_m_s: float = 3500.0,
    source_model: str = "madariaga",
) -> StressDropResult:
    """Estimate stress drop from a spectral fit result.

    Computes the source radius from the corner frequency and derives
    the stress drop as ``(7/16) * M0 / r^3``.

    Parameters
    ----------
    spectral_fit : SpectralFitResult
        Result from :func:`fit_brune_model`.
    seismic_moment : float, optional
        Seismic moment in N*m. If not provided, computed from *magnitude*.
    magnitude : float, optional
        Moment magnitude (Mw). Used to compute seismic moment if
        *seismic_moment* is not given.
    velocity_m_s : float
        S-wave velocity (m/s) at the source, used for source radius.
    source_model : str
        Source model for the k-constant: ``"madariaga"`` (k=0.3724)
        or ``"brune"`` (k=0.3727).

    Returns
    -------
    StressDropResult
        Stress drop in MPa, corner frequency, seismic moment, and
        source radius.

    Raises
    ------
    ValueError
        If neither *seismic_moment* nor *magnitude* is provided, or
        if *source_model* is unknown.
    """
    if seismic_moment is None and magnitude is None:
        raise ValueError(
            "Either seismic_moment or magnitude must be provided"
        )

    if source_model == "madariaga":
        k = 0.3724
    elif source_model == "brune":
        k = 0.3727
    else:
        raise ValueError(
            f"Unknown source model {source_model!r}; "
            "use 'madariaga' or 'brune'"
        )

    fc = spectral_fit.corner_frequency

    if seismic_moment is None:
        # Hanks & Kanamori (1979): log10(M0) = 1.5 * Mw + 9.05
        seismic_moment = 10.0 ** (1.5 * magnitude + 9.05)

    # Source radius: r = k * v / fc
    source_radius = k * velocity_m_s / fc

    # Stress drop: delta_sigma = (7/16) * M0 / r^3
    stress_drop_pa = (7.0 / 16.0) * seismic_moment / source_radius**3
    stress_drop_mpa = stress_drop_pa / 1e6

    return StressDropResult(
        stress_drop_mpa=float(stress_drop_mpa),
        corner_frequency=float(fc),
        seismic_moment=float(seismic_moment),
        source_radius_m=float(source_radius),
        spectral_fit=spectral_fit,
    )


def spectral_fit_batch(
    traces: list[Trace],
    *,
    magnitudes: list[float] | None = None,
    seismic_moments: list[float] | None = None,
    freq_min: float = 0.5,
    freq_max: float | None = None,
    n_gamma: float = 2.0,
    water_level: float = 10.0,
    taper_fraction: float = 0.05,
    velocity_m_s: float = 3500.0,
    source_model: str = "madariaga",
) -> list[StressDropResult]:
    """Fit Brune models and compute stress drops for multiple traces.

    Parameters
    ----------
    traces : list of Trace
        Input waveforms.
    magnitudes : list of float, optional
        Moment magnitudes for each trace.
    seismic_moments : list of float, optional
        Seismic moments (N*m) for each trace.
    freq_min : float
        Minimum frequency (Hz) for fitting.
    freq_max : float, optional
        Maximum frequency (Hz) for fitting.
    n_gamma : float
        High-frequency falloff exponent.
    water_level : float
        Water level (dB) for response removal.
    taper_fraction : float
        Taper fraction before FFT.
    velocity_m_s : float
        S-wave velocity (m/s) at the source.
    source_model : str
        Source model (``"madariaga"`` or ``"brune"``).

    Returns
    -------
    list of StressDropResult
        Stress drop results for each trace.

    Raises
    ------
    ValueError
        If neither *magnitudes* nor *seismic_moments* is provided, or
        if list lengths do not match *traces*.
    """
    if magnitudes is None and seismic_moments is None:
        raise ValueError(
            "Either magnitudes or seismic_moments must be provided"
        )

    n = len(traces)
    if magnitudes is not None and len(magnitudes) != n:
        raise ValueError(
            f"magnitudes length ({len(magnitudes)}) must match "
            f"traces length ({n})"
        )
    if seismic_moments is not None and len(seismic_moments) != n:
        raise ValueError(
            f"seismic_moments length ({len(seismic_moments)}) must match "
            f"traces length ({n})"
        )

    results: list[StressDropResult] = []
    for i, trace in enumerate(traces):
        fit = fit_brune_model(
            trace,
            freq_min=freq_min,
            freq_max=freq_max,
            n_gamma=n_gamma,
            water_level=water_level,
            taper_fraction=taper_fraction,
        )
        m0 = seismic_moments[i] if seismic_moments is not None else None
        mag = magnitudes[i] if magnitudes is not None else None
        sd = compute_stress_drop(
            fit,
            seismic_moment=m0,
            magnitude=mag,
            velocity_m_s=velocity_m_s,
            source_model=source_model,
        )
        results.append(sd)

    return results
