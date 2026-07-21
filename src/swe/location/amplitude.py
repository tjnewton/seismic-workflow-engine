"""Amplitude-based earthquake location via grid search.

Locates seismic events by fitting observed amplitude decay to a
point-source attenuation model over a 3-D spatial grid.  Also provides
a travel-time grid-search locator and a synthetic-event generator for
testing.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class LocationResult(NamedTuple):
    """Result from a grid-search location."""

    x: float
    """Easting (or x grid-coordinate) of the best-fit location."""
    y: float
    """Northing (or y grid-coordinate) of the best-fit location."""
    z: float
    """Depth (or z grid-coordinate) of the best-fit location."""
    misfit: float
    """Scalar misfit (residual) at the best-fit grid node."""
    uncertainty: object | None = None
    """Optional :class:`~swe.uncertainty.types.LocationUncertainty`."""


class SyntheticEventResult(NamedTuple):
    """Result from generating a synthetic seismic event."""

    travel_times: NDArray[np.floating]
    """Synthetic travel times in seconds."""
    amplitudes: NDArray[np.floating]
    """Synthetic observed amplitudes."""


class AmplitudeLocationResult(NamedTuple):
    """Extended result from amplitude-based location."""

    x: float
    """Easting of the best-fit location."""
    y: float
    """Northing of the best-fit location."""
    z: float
    """Depth of the best-fit location."""
    amplitude: float
    """Best-fit source amplitude."""
    misfit: float
    """Scalar misfit at the best-fit grid node."""
    predicted_amplitudes: NDArray[np.floating]
    """Predicted amplitudes at station locations for the best-fit source."""
    residual_grid: NDArray[np.floating]
    """Full 4-D misfit grid (nx, ny, nz, n_amplitudes)."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _default_attenuation_params() -> dict[str, float]:
    """Return default seismic amplitude attenuation parameters.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:

        - ``q_factor`` : quality factor *Q* (dimensionless).
        - ``frequency`` : dominant frequency in Hz.
        - ``velocity`` : shear-wave velocity in m/s.
        - ``geometric_spreading_exponent`` : exponent for geometric
          spreading (1.0 = body wave, 0.5 = surface wave).
    """
    return {
        "q_factor": 50.0,
        "frequency": 60.0,
        "velocity": 1020.0,
        "geometric_spreading_exponent": 1.0,
    }


def _attenuation_coefficient(
    frequency: float,
    q_factor: float,
    velocity: float,
) -> float:
    """Compute the anelastic attenuation coefficient *B*.

    Parameters
    ----------
    frequency : float
        Dominant frequency (Hz).
    q_factor : float
        Quality factor (dimensionless).
    velocity : float
        Seismic wave velocity (m/s).

    Returns
    -------
    float
        Attenuation coefficient B = pi * f / (Q * v)  in  1/m.
    """
    return np.pi * frequency / (q_factor * velocity)


def _compute_distance_grid(
    grid_x: NDArray[np.floating],
    grid_y: NDArray[np.floating],
    grid_z: NDArray[np.floating],
    station_x: float,
    station_y: float,
    station_z: float,
) -> NDArray[np.floating]:
    """Compute Euclidean distances from every grid node to a station.

    Parameters
    ----------
    grid_x, grid_y, grid_z : ndarray
        1-D arrays of grid coordinates along each axis.
    station_x, station_y, station_z : float
        Station coordinates.

    Returns
    -------
    ndarray
        3-D array of distances with shape ``(nx, ny, nz)``.
    """
    gx = grid_x[:, np.newaxis, np.newaxis]
    gy = grid_y[np.newaxis, :, np.newaxis]
    gz = grid_z[np.newaxis, np.newaxis, :]
    return np.sqrt(
        (gx - station_x) ** 2
        + (gy - station_y) ** 2
        + (gz - station_z) ** 2,
    )


def _predicted_amplitude(
    source_amplitude: float | NDArray[np.floating],
    distance: NDArray[np.floating],
    attenuation_coeff: float,
    spreading_exponent: float = 1.0,
) -> NDArray[np.floating]:
    """Point-source amplitude with geometric spreading and attenuation.

    A(r) = A0 * exp(-B * r) / r^n

    Parameters
    ----------
    source_amplitude : float or ndarray
        Source amplitude(s) A0.
    distance : ndarray
        Distance(s) in metres.
    attenuation_coeff : float
        Anelastic attenuation coefficient *B* (1/m).
    spreading_exponent : float
        Geometric spreading exponent *n* (default 1.0).

    Returns
    -------
    ndarray
        Predicted amplitude(s).
    """
    # Guard against division by zero at the source location and
    # overflow for very large attenuation_coeff * distance products.
    safe_dist = np.maximum(distance, np.finfo(float).tiny)
    exponent = np.clip(-attenuation_coeff * safe_dist, -700.0, 0.0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = source_amplitude * np.exp(exponent) / (
            safe_dist ** spreading_exponent
        )
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def locate_by_travel_time(
    observed_times: NDArray[np.floating],
    station_coords: NDArray[np.floating],
    velocity_model: float | NDArray[np.floating],
    *,
    grid_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    grid_spacing: tuple[float, float, float],
    reference_station_index: int = 0,
) -> LocationResult:
    """Locate an event by travel-time grid search.

    For each candidate source position on a 3-D grid, compute predicted
    travel times to every station (straight-ray, constant velocity) and
    minimise the L1 residual of differential times relative to a
    reference station.

    Parameters
    ----------
    observed_times : ndarray, shape (n_stations,)
        Observed arrival times (seconds) at each station.
    station_coords : ndarray, shape (n_stations, 3)
        Station coordinates ``[x, y, z]`` in the same units as the
        grid (typically metres).
    velocity_model : float or ndarray
        Constant velocity (m/s) for straight-ray approximation.  For
        3-D velocity models, use
        :func:`~swe.location.travel_time.locate_by_travel_time_grid`
        with a precomputed :class:`~swe.velocity.model.TravelTimeGrid`.
    grid_bounds : tuple of 3 (min, max) tuples
        ``((x_min, x_max), (y_min, y_max), (z_min, z_max))`` in
        metres.
    grid_spacing : tuple of 3 floats
        ``(dx, dy, dz)`` grid spacing in metres.
    reference_station_index : int
        Index of the reference station for differential times.

    Returns
    -------
    LocationResult
        Best-fit ``(x, y, z, residual)``.
    """
    n_stations = station_coords.shape[0]
    if observed_times.shape[0] != n_stations:
        msg = (
            f"observed_times length ({observed_times.shape[0]}) does not "
            f"match station_coords rows ({n_stations})"
        )
        raise ValueError(msg)

    # Build 1-D grid axes.
    grid_x = np.arange(grid_bounds[0][0], grid_bounds[0][1] + grid_spacing[0] / 2, grid_spacing[0])
    grid_y = np.arange(grid_bounds[1][0], grid_bounds[1][1] + grid_spacing[1] / 2, grid_spacing[1])
    grid_z = np.arange(grid_bounds[2][0], grid_bounds[2][1] + grid_spacing[2] / 2, grid_spacing[2])

    logger.info(
        "Travel-time grid search: %d x %d x %d nodes",
        len(grid_x), len(grid_y), len(grid_z),
    )

    # Observed differential times relative to reference station.
    obs_dt = observed_times - observed_times[reference_station_index]

    # Accumulate residuals over the grid.
    residuals = np.zeros((len(grid_x), len(grid_y), len(grid_z)))

    # Compute distances from every grid node to the reference station.
    ref_dist = _compute_distance_grid(
        grid_x, grid_y, grid_z,
        station_coords[reference_station_index, 0],
        station_coords[reference_station_index, 1],
        station_coords[reference_station_index, 2],
    )
    ref_tt = ref_dist / float(velocity_model)

    for ii in range(n_stations):
        if ii == reference_station_index:
            continue
        sta_dist = _compute_distance_grid(
            grid_x, grid_y, grid_z,
            station_coords[ii, 0],
            station_coords[ii, 1],
            station_coords[ii, 2],
        )
        sta_tt = sta_dist / float(velocity_model)
        pred_dt = sta_tt - ref_tt
        residuals += np.abs(obs_dt[ii] - pred_dt)

    # Find global minimum.
    min_idx = np.unravel_index(np.argmin(residuals), residuals.shape)
    best_x = grid_x[min_idx[0]]
    best_y = grid_y[min_idx[1]]
    best_z = grid_z[min_idx[2]]
    best_residual = float(residuals[min_idx])

    logger.info(
        "Best travel-time location: x=%.1f  y=%.1f  z=%.1f  residual=%.4f",
        best_x, best_y, best_z, best_residual,
    )

    return LocationResult(
        x=best_x,
        y=best_y,
        z=best_z,
        misfit=best_residual,
    )


def amplocate(
    observed_amplitudes: NDArray[np.floating],
    station_coords: NDArray[np.floating],
    *,
    source_amplitudes: NDArray[np.floating] | None = None,
    amplitude_corrections: NDArray[np.floating] | None = None,
    distance_grids: NDArray[np.floating] | None = None,
    attenuation_params: dict[str, float] | None = None,
    grid_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] | None = None,
    grid_spacing: tuple[float, float, float] | None = None,
    weight_power: int = 2,
) -> AmplitudeLocationResult:
    """Locate an event by amplitude-decay grid search.

    For each candidate source position, compute predicted amplitudes
    using a point-source attenuation model and minimise the weighted
    sum of squared amplitude residuals.

    Either provide ``distance_grids`` (pre-computed distance from every
    grid node to each station) **or** ``grid_bounds`` and
    ``grid_spacing`` (in which case Euclidean distances are computed
    on the fly).

    Parameters
    ----------
    observed_amplitudes : ndarray, shape (n_stations,)
        Observed amplitudes at each station.  Stations with no pick
        should be set to zero and will be excluded.
    station_coords : ndarray, shape (n_stations, 3)
        Station coordinates ``[x, y, z]`` in metres.
    source_amplitudes : ndarray, shape (n_A0,), optional
        Candidate source amplitudes to search over.  Defaults to
        ``np.logspace(-2, 2, 50)``.
    amplitude_corrections : ndarray, shape (n_stations,), optional
        Multiplicative site-amplification factors.  Observed
        amplitudes are divided by these before comparison.  Defaults
        to ones.
    distance_grids : ndarray, shape (n_stations, nx, ny, nz), optional
        Pre-computed distances from every grid node to each station.
        Mutually exclusive with ``grid_bounds`` / ``grid_spacing``.
    attenuation_params : dict, optional
        Override default attenuation parameters (see
        ``_default_attenuation_params``).
    grid_bounds : tuple, optional
        ``((x_min, x_max), (y_min, y_max), (z_min, z_max))``.
    grid_spacing : tuple, optional
        ``(dx, dy, dz)``.
    weight_power : int
        Power applied to distance-based weights.  0 = uniform,
        1 = linear, 2 = quadratic (default).

    Returns
    -------
    AmplitudeLocationResult
        Named tuple with best-fit location, amplitude, misfit,
        predicted amplitudes, and the full residual grid.
    """
    params = _default_attenuation_params()
    if attenuation_params is not None:
        params.update(attenuation_params)

    n_stations = station_coords.shape[0]
    if observed_amplitudes.shape[0] != n_stations:
        msg = (
            f"observed_amplitudes length ({observed_amplitudes.shape[0]}) "
            f"does not match station_coords rows ({n_stations})"
        )
        raise ValueError(msg)

    B = _attenuation_coefficient(
        params["frequency"], params["q_factor"], params["velocity"],
    )
    spreading = params["geometric_spreading_exponent"]

    # Amplitude corrections.
    if amplitude_corrections is None:
        amplitude_corrections = np.ones(n_stations)
    corrected_amps = observed_amplitudes / amplitude_corrections

    # Source amplitude search vector.
    if source_amplitudes is None:
        source_amplitudes = np.logspace(-2, 2, 50)
    n_a0 = len(source_amplitudes)

    # Distance grids: pre-computed or built on the fly.
    if distance_grids is not None:
        if distance_grids.shape[0] != n_stations:
            msg = (
                f"distance_grids first axis ({distance_grids.shape[0]}) "
                f"does not match n_stations ({n_stations})"
            )
            raise ValueError(msg)
        grid_shape = distance_grids.shape[1:]  # (nx, ny, nz)
        station_dists = distance_grids
        # Build dummy grid axes for result extraction.
        grid_x = np.arange(grid_shape[0], dtype=float)
        grid_y = np.arange(grid_shape[1], dtype=float)
        grid_z = np.arange(grid_shape[2], dtype=float)
    elif grid_bounds is not None and grid_spacing is not None:
        grid_x = np.arange(
            grid_bounds[0][0],
            grid_bounds[0][1] + grid_spacing[0] / 2,
            grid_spacing[0],
        )
        grid_y = np.arange(
            grid_bounds[1][0],
            grid_bounds[1][1] + grid_spacing[1] / 2,
            grid_spacing[1],
        )
        grid_z = np.arange(
            grid_bounds[2][0],
            grid_bounds[2][1] + grid_spacing[2] / 2,
            grid_spacing[2],
        )
        grid_shape = (len(grid_x), len(grid_y), len(grid_z))
        station_dists = np.empty((n_stations, *grid_shape))
        for ii in range(n_stations):
            station_dists[ii] = _compute_distance_grid(
                grid_x, grid_y, grid_z,
                station_coords[ii, 0],
                station_coords[ii, 1],
                station_coords[ii, 2],
            )
    else:
        msg = (
            "Either distance_grids or both grid_bounds and grid_spacing "
            "must be provided."
        )
        raise ValueError(msg)

    logger.info(
        "Amplitude grid search: grid %s, %d source amplitudes, %d stations",
        grid_shape, n_a0, n_stations,
    )

    # Compute weights: favour closer stations (using max distance per
    # station across the grid as a proxy).
    max_dists = np.array([station_dists[ii].max() for ii in range(n_stations)])
    if weight_power == 0:
        weights = np.ones(n_stations)
    else:
        weights = ((max_dists.max() - max_dists) / max_dists.max()) ** weight_power

    # Misfit accumulation: shape (nx, ny, nz, n_A0).
    residuals = np.zeros((*grid_shape, n_a0))

    # Mask stations with zero observed amplitude.
    active = corrected_amps != 0.0

    for ii in range(n_stations):
        if not active[ii]:
            continue
        r = station_dists[ii]  # (nx, ny, nz)
        for jj in range(n_a0):
            pred = _predicted_amplitude(
                source_amplitudes[jj], r, B, spreading,
            )
            with np.errstate(over="ignore"):
                diff_sq = np.nan_to_num(
                    (corrected_amps[ii] - pred) ** 2, nan=0.0, posinf=0.0,
                )
            residuals[..., jj] += diff_sq * weights[ii]

    # Find global minimum.
    min_idx = np.unravel_index(np.argmin(residuals), residuals.shape)
    best_x = float(grid_x[min_idx[0]])
    best_y = float(grid_y[min_idx[1]])
    best_z = float(grid_z[min_idx[2]])
    best_a0 = float(source_amplitudes[min_idx[3]])
    best_misfit = float(residuals[min_idx])

    # Predicted amplitudes at the best-fit source location.
    predicted = np.zeros(n_stations)
    for ii in range(n_stations):
        r_at_best = station_dists[ii][min_idx[0], min_idx[1], min_idx[2]]
        predicted[ii] = _predicted_amplitude(
            best_a0, np.array(r_at_best), B, spreading,
        )

    logger.info(
        "Best amplitude location: x=%.1f  y=%.1f  z=%.1f  A0=%.3g  "
        "misfit=%.4g",
        best_x, best_y, best_z, best_a0, best_misfit,
    )

    return AmplitudeLocationResult(
        x=best_x,
        y=best_y,
        z=best_z,
        amplitude=best_a0,
        misfit=best_misfit,
        predicted_amplitudes=predicted,
        residual_grid=residuals,
    )


def synthetic_event(
    true_location: tuple[float, float, float],
    station_coords: NDArray[np.floating],
    *,
    source_amplitude: float = 1.0,
    velocity: float = 1020.0,
    noise_level: float = 0.0,
    attenuation_params: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> SyntheticEventResult:
    """Generate synthetic travel times and amplitudes for testing.

    Computes straight-ray travel times and attenuated amplitudes from a
    point source to an array of stations, optionally adding Gaussian
    noise.

    Parameters
    ----------
    true_location : tuple of 3 floats
        Source coordinates ``(x, y, z)`` in metres.
    station_coords : ndarray, shape (n_stations, 3)
        Station coordinates ``[x, y, z]`` in metres.
    source_amplitude : float
        Source amplitude *A0*.
    velocity : float
        Constant wave velocity (m/s) for travel-time computation.
    noise_level : float
        Standard deviation of additive Gaussian noise applied to
        both travel times and amplitudes (relative to signal).
        0.0 = no noise.
    attenuation_params : dict, optional
        Override default attenuation parameters.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    travel_times : ndarray, shape (n_stations,)
        Synthetic travel times in seconds.
    amplitudes : ndarray, shape (n_stations,)
        Synthetic observed amplitudes.
    """
    if rng is None:
        rng = np.random.default_rng()

    params = _default_attenuation_params()
    if attenuation_params is not None:
        params.update(attenuation_params)

    B = _attenuation_coefficient(
        params["frequency"], params["q_factor"], params["velocity"],
    )
    spreading = params["geometric_spreading_exponent"]

    x0, y0, z0 = true_location
    n_stations = station_coords.shape[0]

    # Euclidean distances.
    dists = np.sqrt(
        (station_coords[:, 0] - x0) ** 2
        + (station_coords[:, 1] - y0) ** 2
        + (station_coords[:, 2] - z0) ** 2,
    )

    # Travel times.
    travel_times = dists / velocity

    # Amplitudes.
    amplitudes = _predicted_amplitude(source_amplitude, dists, B, spreading)

    # Optional noise.
    if noise_level > 0:
        tt_noise = rng.normal(0, noise_level * travel_times.mean(), n_stations)
        travel_times = travel_times + tt_noise
        amp_noise = rng.normal(0, noise_level * amplitudes.mean(), n_stations)
        amplitudes = np.maximum(amplitudes + amp_noise, 0.0)

    logger.debug(
        "Synthetic event at (%.1f, %.1f, %.1f): %d stations, "
        "noise_level=%.2f",
        x0, y0, z0, n_stations, noise_level,
    )

    return SyntheticEventResult(travel_times, amplitudes)


def locate_by_travel_time_with_uncertainty(
    observed_times: NDArray[np.floating],
    station_coords: NDArray[np.floating],
    velocity_model: float | NDArray[np.floating],
    *,
    grid_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    grid_spacing: tuple[float, float, float],
    reference_station_index: int = 0,
    pick_time_uncertainties: NDArray[np.floating] | None = None,
    n_monte_carlo: int = 100,
    random_state: int = 0,
) -> LocationResult:
    """Locate an event by travel-time grid search with optional Monte Carlo uncertainty.

    Wraps :func:`locate_by_travel_time` and, when *pick_time_uncertainties*
    is provided, runs Monte Carlo perturbation to estimate a
    :class:`~swe.uncertainty.types.LocationUncertainty`.

    Parameters
    ----------
    observed_times : ndarray, shape (n_stations,)
        Observed arrival times (seconds).
    station_coords : ndarray, shape (n_stations, 3)
        Station coordinates ``[x, y, z]`` in metres.
    velocity_model : float or ndarray
        Constant velocity (m/s).
    grid_bounds : tuple of 3 (min, max) tuples
        Grid bounds in metres.
    grid_spacing : tuple of 3 floats
        Grid spacing in metres.
    reference_station_index : int
        Index of the reference station for differential times.
    pick_time_uncertainties : ndarray, optional
        1-sigma pick-time uncertainties (seconds).  If provided,
        Monte Carlo uncertainty estimation is performed.
    n_monte_carlo : int
        Number of Monte Carlo realisations.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    LocationResult
        Best-fit location with ``uncertainty`` populated when
        *pick_time_uncertainties* is provided.
    """
    base_result = locate_by_travel_time(
        observed_times,
        station_coords,
        velocity_model,
        grid_bounds=grid_bounds,
        grid_spacing=grid_spacing,
        reference_station_index=reference_station_index,
    )

    if pick_time_uncertainties is None:
        return base_result

    from swe.uncertainty.propagation import monte_carlo_location_uncertainty

    loc_unc = monte_carlo_location_uncertainty(
        pick_times=observed_times,
        pick_uncertainties=np.asarray(pick_time_uncertainties, dtype=np.float64),
        locate_fn=locate_by_travel_time,
        locate_kwargs={
            "station_coords": station_coords,
            "velocity_model": velocity_model,
            "grid_bounds": grid_bounds,
            "grid_spacing": grid_spacing,
            "reference_station_index": reference_station_index,
        },
        n_samples=n_monte_carlo,
        random_state=random_state,
    )

    return LocationResult(
        x=base_result.x,
        y=base_result.y,
        z=base_result.z,
        misfit=base_result.misfit,
        uncertainty=loc_unc,
    )
