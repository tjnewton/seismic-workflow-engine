"""Tests for swe.location.travel_time."""

from __future__ import annotations

import numpy as np
import pytest

from swe.location.amplitude import LocationResult
from swe.location.travel_time import locate_by_travel_time_grid
from swe.velocity.model import TravelTimeGrid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_constant_velocity_ttg(
    velocity: float = 5000.0,
    grid_bounds=((0.0, 200.0), (0.0, 200.0), (0.0, 100.0)),
    grid_spacing=(10.0, 10.0, 10.0),
    station_coords=None,
):
    """Build a TravelTimeGrid using constant velocity (pure numpy)."""
    dx, dy, dz = grid_spacing
    gx = np.arange(grid_bounds[0][0], grid_bounds[0][1] + dx / 2, dx)
    gy = np.arange(grid_bounds[1][0], grid_bounds[1][1] + dy / 2, dy)
    gz = np.arange(grid_bounds[2][0], grid_bounds[2][1] + dz / 2, dz)

    if station_coords is None:
        station_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [200.0, 0.0, 0.0],
                [0.0, 200.0, 0.0],
                [200.0, 200.0, 0.0],
                [100.0, 100.0, 0.0],
                [50.0, 150.0, 0.0],
                [150.0, 50.0, 0.0],
            ]
        )

    nx, ny, nz = len(gx), len(gy), len(gz)
    n_stations = station_coords.shape[0]
    travel_times = np.empty((n_stations, nx, ny, nz))

    for idx in range(n_stations):
        sx, sy, sz = station_coords[idx]
        gxx, gyy, gzz = np.meshgrid(gx, gy, gz, indexing="ij")
        dist = np.sqrt((gxx - sx) ** 2 + (gyy - sy) ** 2 + (gzz - sz) ** 2)
        travel_times[idx] = dist / velocity

    return TravelTimeGrid(
        grid_x=gx,
        grid_y=gy,
        grid_z=gz,
        station_coords=station_coords,
        travel_times=travel_times,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLocateByTravelTimeGrid:
    def test_recovers_known_location(self):
        """Should recover a synthetic event at a grid node."""
        velocity = 5000.0
        true_loc = (100.0, 100.0, 50.0)
        ttg = _build_constant_velocity_ttg(velocity=velocity)

        # Synthesise observed times from the true location.
        stations = ttg.station_coords
        dists = np.sqrt(np.sum((stations - np.array(true_loc)) ** 2, axis=1))
        observed_times = dists / velocity

        result = locate_by_travel_time_grid(observed_times, ttg)

        assert isinstance(result, LocationResult)
        assert abs(result.x - true_loc[0]) <= 10.0
        assert abs(result.y - true_loc[1]) <= 10.0
        assert abs(result.z - true_loc[2]) <= 10.0

    def test_residual_near_zero_for_exact_match(self):
        """Residual should be near zero when source is on a grid node."""
        velocity = 5000.0
        true_loc = (100.0, 100.0, 50.0)
        ttg = _build_constant_velocity_ttg(velocity=velocity)

        stations = ttg.station_coords
        dists = np.sqrt(np.sum((stations - np.array(true_loc)) ** 2, axis=1))
        observed_times = dists / velocity

        result = locate_by_travel_time_grid(observed_times, ttg)
        assert result.misfit < 1e-6

    def test_residual_nonnegative(self):
        """Misfit should always be non-negative."""
        ttg = _build_constant_velocity_ttg()
        observed_times = np.linspace(0.01, 0.05, ttg.station_coords.shape[0])
        result = locate_by_travel_time_grid(observed_times, ttg)
        assert result.misfit >= 0.0

    def test_different_reference_station(self):
        """Should still recover the source with a different reference."""
        velocity = 5000.0
        true_loc = (80.0, 120.0, 30.0)
        ttg = _build_constant_velocity_ttg(velocity=velocity)

        stations = ttg.station_coords
        dists = np.sqrt(np.sum((stations - np.array(true_loc)) ** 2, axis=1))
        observed_times = dists / velocity

        result = locate_by_travel_time_grid(
            observed_times,
            ttg,
            reference_station_index=3,
        )
        assert abs(result.x - true_loc[0]) <= 10.0
        assert abs(result.y - true_loc[1]) <= 10.0
        assert abs(result.z - true_loc[2]) <= 10.0

    def test_rejects_mismatched_inputs(self):
        """Should raise ValueError when lengths don't match."""
        ttg = _build_constant_velocity_ttg()
        with pytest.raises(ValueError, match="does not match"):
            locate_by_travel_time_grid(
                np.array([0.01, 0.02]),  # wrong length
                ttg,
            )

    def test_output_type(self):
        """Result should be a LocationResult NamedTuple."""
        ttg = _build_constant_velocity_ttg()
        observed_times = np.ones(ttg.station_coords.shape[0]) * 0.03
        result = locate_by_travel_time_grid(observed_times, ttg)
        assert isinstance(result, LocationResult)
        assert hasattr(result, "x")
        assert hasattr(result, "y")
        assert hasattr(result, "z")
        assert hasattr(result, "misfit")
