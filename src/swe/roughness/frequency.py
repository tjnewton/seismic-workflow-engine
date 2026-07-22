"""Temporal frequency evolution and scaling law fitting.

Computes instantaneous-frequency time series from seismic arrivals
and fits the tribological scaling law f = V_slip / lambda_dominant.

References
----------
Bowden, F. P. & Tabor, D. (1950). The Friction and Lubrication of
    Solids. Oxford University Press.
Candela, T. et al. (2012). Roughness of fault surfaces over nine
    decades of length scales. J. Geophys. Res., 117, B08409.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from obspy.core.trace import Trace

logger = logging.getLogger(__name__)

__all__ = [
    "FrequencyTimeSeriesResult",
    "ScalingLawFitResult",
    "compute_frequency_time_series",
    "fit_scaling_law",
    "infer_roughness_wavelength",
]


class FrequencyTimeSeriesResult(NamedTuple):
    """Result of binned instantaneous-frequency time series.

    Attributes
    ----------
    times : ndarray
        Bin centre times as matplotlib date numbers.
    median_frequency : ndarray
        Median instantaneous frequency (Hz) per bin.
    std_frequency : ndarray
        Standard deviation of instantaneous frequency per bin.
    count_per_bin : ndarray
        Number of measurements per bin.
    bin_width_days : float
        Width of each time bin in days.
    all_frequencies : ndarray
        All individual frequency measurements (Hz).
    all_times : ndarray
        All individual measurement times as matplotlib date numbers.
    """

    times: NDArray[np.floating]
    median_frequency: NDArray[np.floating]
    std_frequency: NDArray[np.floating]
    count_per_bin: NDArray[np.integer]
    bin_width_days: float
    all_frequencies: NDArray[np.floating]
    all_times: NDArray[np.floating]


class ScalingLawFitResult(NamedTuple):
    """Result of fitting the scaling law f = V / lambda_0.

    Attributes
    ----------
    lambda_0 : float
        Best-fit dominant roughness wavelength (m).
    lambda_0_std : float
        Standard error of lambda_0 (m).
    r_squared : float
        Coefficient of determination.
    predicted_frequency : ndarray
        Predicted frequency from the fit (Hz).
    observed_frequency : ndarray
        Observed frequency values used in the fit (Hz).
    slip_velocity : ndarray
        Interpolated slip velocity values (m/s).
    residuals : ndarray
        Observed minus predicted frequency (Hz).
    """

    lambda_0: float
    lambda_0_std: float
    r_squared: float
    predicted_frequency: NDArray[np.floating]
    observed_frequency: NDArray[np.floating]
    slip_velocity: NDArray[np.floating]
    residuals: NDArray[np.floating]


def compute_frequency_time_series(
    traces: list[Trace],
    pick_times: list[float],
    *,
    halfwin: float = 0.06,
    bin_width_days: float = 1.0,
    min_frequency: float = 1.0,
    max_frequency: float = 100.0,
) -> FrequencyTimeSeriesResult:
    """Compute binned instantaneous-frequency time series from arrivals.

    For each trace/pick pair, the instantaneous frequency is estimated
    via the Hilbert transform in a window centred on the pick.  Results
    are then binned by time into daily (or custom) intervals.

    Parameters
    ----------
    traces : list of obspy.Trace
        Waveform traces, one per pick.
    pick_times : list of float
        Pick times as matplotlib date numbers, one per trace.
    halfwin : float
        Half-window length in seconds for instantaneous-frequency
        estimation (passed to ``swe.utils.signal.instantaneous_frequency``).
    bin_width_days : float
        Width of each time bin in days.
    min_frequency : float
        Discard frequency estimates below this value (Hz).
    max_frequency : float
        Discard frequency estimates above this value (Hz).

    Returns
    -------
    FrequencyTimeSeriesResult
    """
    from swe.utils.signal import instantaneous_frequency

    if len(traces) != len(pick_times):
        msg = f"Length mismatch: {len(traces)} traces vs {len(pick_times)} pick times"
        raise ValueError(msg)

    all_freqs = []
    all_times = []

    for trace, ptime in zip(traces, pick_times, strict=False):
        freq = instantaneous_frequency(trace, halfwin=halfwin)
        if np.isnan(freq):
            continue
        if freq < min_frequency or freq > max_frequency:
            continue
        all_freqs.append(freq)
        all_times.append(ptime)

    all_freqs_arr = np.asarray(all_freqs, dtype=np.float64)
    all_times_arr = np.asarray(all_times, dtype=np.float64)

    if len(all_freqs_arr) == 0:
        logger.warning("No valid frequency measurements after filtering")
        empty = np.array([], dtype=np.float64)
        return FrequencyTimeSeriesResult(
            times=empty,
            median_frequency=empty,
            std_frequency=empty,
            count_per_bin=np.array([], dtype=np.int64),
            bin_width_days=bin_width_days,
            all_frequencies=empty,
            all_times=empty,
        )

    # Bin by time
    t_min = all_times_arr.min()
    t_max = all_times_arr.max()
    bin_edges = np.arange(t_min, t_max + bin_width_days, bin_width_days)

    if len(bin_edges) < 2:
        bin_edges = np.array([t_min, t_min + bin_width_days])

    n_bins = len(bin_edges) - 1
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    medians = np.full(n_bins, np.nan)
    stds = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        mask = (all_times_arr >= bin_edges[i]) & (all_times_arr < bin_edges[i + 1])
        if i == n_bins - 1:
            mask |= all_times_arr == bin_edges[i + 1]
        n = mask.sum()
        counts[i] = n
        if n > 0:
            medians[i] = np.median(all_freqs_arr[mask])
            stds[i] = np.std(all_freqs_arr[mask]) if n > 1 else 0.0

    logger.info(
        "Computed frequency time series: %d measurements in %d bins",
        len(all_freqs_arr),
        n_bins,
    )

    return FrequencyTimeSeriesResult(
        times=bin_centres,
        median_frequency=medians,
        std_frequency=stds,
        count_per_bin=counts,
        bin_width_days=bin_width_days,
        all_frequencies=all_freqs_arr,
        all_times=all_times_arr,
    )


def fit_scaling_law(
    frequency_ts: FrequencyTimeSeriesResult,
    slip_velocity_times: NDArray[np.floating],
    slip_velocity_values: NDArray[np.floating],
    *,
    velocity_unit: str = "m/s",
) -> ScalingLawFitResult:
    """Fit the scaling law f = V / lambda_0 to frequency and velocity data.

    Interpolates the slip velocity time series to the frequency time
    bins, then fits a single parameter lambda_0 via least squares.

    Parameters
    ----------
    frequency_ts : FrequencyTimeSeriesResult
        Binned frequency time series.
    slip_velocity_times : ndarray
        Times of slip velocity measurements (matplotlib date numbers).
    slip_velocity_values : ndarray
        Slip velocity values.
    velocity_unit : str
        Unit of slip velocity.  Supported: ``"m/s"``, ``"m/day"``,
        ``"m/week"``.  Converted internally to m/s.

    Returns
    -------
    ScalingLawFitResult
    """
    unit_factors = {
        "m/s": 1.0,
        "m/day": 1.0 / 86400.0,
        "m/week": 1.0 / (7.0 * 86400.0),
    }
    if velocity_unit not in unit_factors:
        msg = f"Unknown velocity_unit {velocity_unit!r}. Supported: {list(unit_factors)}"
        raise ValueError(msg)

    factor = unit_factors[velocity_unit]
    vel_ms = np.asarray(slip_velocity_values, dtype=np.float64) * factor

    # Use bins with valid data
    valid = ~np.isnan(frequency_ts.median_frequency) & (frequency_ts.count_per_bin > 0)
    obs_times = frequency_ts.times[valid]
    obs_freq = frequency_ts.median_frequency[valid]

    if len(obs_times) < 2:
        logger.warning("Fewer than 2 valid bins; cannot fit scaling law")
        return ScalingLawFitResult(
            lambda_0=np.nan,
            lambda_0_std=np.nan,
            r_squared=np.nan,
            predicted_frequency=np.array([], dtype=np.float64),
            observed_frequency=obs_freq,
            slip_velocity=np.array([], dtype=np.float64),
            residuals=np.array([], dtype=np.float64),
        )

    # Interpolate velocity to frequency time bins
    interp_func = interp1d(
        np.asarray(slip_velocity_times, dtype=np.float64),
        vel_ms,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    vel_interp = interp_func(obs_times)

    # Fit f = V / lambda_0  ->  f = (1/lambda_0) * V
    def _model(v: NDArray, inv_lambda: float) -> NDArray:
        return inv_lambda * v

    try:
        popt, pcov = curve_fit(_model, vel_interp, obs_freq, p0=[1e6])
        inv_lambda = popt[0]
        inv_lambda_std = np.sqrt(pcov[0, 0])
        lambda_0 = 1.0 / inv_lambda if inv_lambda != 0 else np.nan
        lambda_0_std = inv_lambda_std / inv_lambda**2 if inv_lambda != 0 else np.nan
    except RuntimeError:
        logger.warning("Curve fit did not converge")
        return ScalingLawFitResult(
            lambda_0=np.nan,
            lambda_0_std=np.nan,
            r_squared=np.nan,
            predicted_frequency=np.array([], dtype=np.float64),
            observed_frequency=obs_freq,
            slip_velocity=vel_interp,
            residuals=np.array([], dtype=np.float64),
        )

    predicted = _model(vel_interp, inv_lambda)
    residuals = obs_freq - predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((obs_freq - np.mean(obs_freq)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    logger.info("Scaling law fit: lambda_0=%.3e m, R^2=%.4f", lambda_0, r_squared)

    return ScalingLawFitResult(
        lambda_0=lambda_0,
        lambda_0_std=lambda_0_std,
        r_squared=r_squared,
        predicted_frequency=predicted,
        observed_frequency=obs_freq,
        slip_velocity=vel_interp,
        residuals=residuals,
    )


def infer_roughness_wavelength(
    frequency: float | NDArray[np.floating],
    slip_velocity: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Infer roughness wavelength from frequency and slip velocity.

    Computes lambda = V / f with input validation.

    Parameters
    ----------
    frequency : float or ndarray
        Observed frequency (Hz).  Must be positive.
    slip_velocity : float or ndarray
        Slip velocity (m/s).  Must be non-negative.

    Returns
    -------
    float or ndarray
        Inferred roughness wavelength (m).

    Raises
    ------
    ValueError
        If *frequency* contains non-positive values or
        *slip_velocity* contains negative values.
    """
    freq = np.asarray(frequency, dtype=np.float64)
    vel = np.asarray(slip_velocity, dtype=np.float64)

    if np.any(freq <= 0):
        raise ValueError("frequency must be positive")
    if np.any(vel < 0):
        raise ValueError("slip_velocity must be non-negative")

    result = vel / freq

    if result.ndim == 0:
        return float(result)
    return result
