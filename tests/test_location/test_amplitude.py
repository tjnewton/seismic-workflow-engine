"""Tests for swe.location.amplitude."""

from __future__ import annotations

import numpy as np
import pytest

from swe.location.amplitude import (
    AmplitudeLocationResult,
    LocationResult,
    _default_attenuation_params,
    amplocate,
    locate_by_travel_time,
    synthetic_event,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def station_coords() -> np.ndarray:
    """Seven stations with 3-D spread to constrain depth."""
    return np.array([
        [0.0, 0.0, 0.0],
        [200.0, 0.0, 0.0],
        [0.0, 200.0, 0.0],
        [-200.0, 0.0, 0.0],
        [0.0, -200.0, 0.0],
        [100.0, 100.0, -50.0],
        [-100.0, -100.0, 50.0],
    ])


@pytest.fixture()
def true_location() -> tuple[float, float, float]:
    """Known source location at a grid node."""
    return (50.0, 50.0, 0.0)


# ---------------------------------------------------------------------------
# Tests for _default_attenuation_params
# ---------------------------------------------------------------------------

class TestDefaultAttenuationParams:
    def test_returns_dict(self):
        params = _default_attenuation_params()
        assert isinstance(params, dict)

    def test_expected_keys(self):
        params = _default_attenuation_params()
        expected_keys = {
            "q_factor",
            "frequency",
            "velocity",
            "geometric_spreading_exponent",
        }
        assert set(params.keys()) == expected_keys

    def test_values_are_positive(self):
        params = _default_attenuation_params()
        for key, value in params.items():
            assert value > 0, f"{key} should be positive, got {value}"


# ---------------------------------------------------------------------------
# Tests for synthetic_event
# ---------------------------------------------------------------------------

class TestSyntheticEvent:
    def test_output_shapes(self, station_coords, true_location):
        tt, amps = synthetic_event(true_location, station_coords)
        n_stations = station_coords.shape[0]
        assert tt.shape == (n_stations,)
        assert amps.shape == (n_stations,)

    def test_travel_times_positive(self, station_coords, true_location):
        tt, _ = synthetic_event(true_location, station_coords)
        assert np.all(tt > 0)

    def test_amplitudes_positive(self, station_coords, true_location):
        _, amps = synthetic_event(true_location, station_coords)
        assert np.all(amps > 0)

    def test_noise_increases_variance(self, station_coords, true_location):
        rng = np.random.default_rng(42)
        _, amps_clean = synthetic_event(
            true_location, station_coords, noise_level=0.0,
        )
        _, amps_noisy = synthetic_event(
            true_location, station_coords, noise_level=0.1, rng=rng,
        )
        # The noisy amplitudes should differ from the clean ones.
        assert not np.allclose(amps_clean, amps_noisy)

    def test_deterministic_with_rng(self, station_coords, true_location):
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        tt1, a1 = synthetic_event(
            true_location, station_coords, noise_level=0.05, rng=rng1,
        )
        tt2, a2 = synthetic_event(
            true_location, station_coords, noise_level=0.05, rng=rng2,
        )
        np.testing.assert_array_equal(tt1, tt2)
        np.testing.assert_array_equal(a1, a2)


# ---------------------------------------------------------------------------
# Tests for amplocate
# ---------------------------------------------------------------------------

class TestAmplocate:
    def test_recovers_synthetic_location(self, station_coords, true_location):
        """amplocate should recover a synthetic source within one grid cell."""
        _, observed_amps = synthetic_event(
            true_location,
            station_coords,
            source_amplitude=5.0,
            noise_level=0.0,
        )
        result = amplocate(
            observed_amps,
            station_coords,
            source_amplitudes=np.array([4.0, 5.0, 6.0]),
            grid_bounds=((-100.0, 150.0), (-100.0, 150.0), (-80.0, 80.0)),
            grid_spacing=(10.0, 10.0, 10.0),
            weight_power=0,
        )
        assert isinstance(result, AmplitudeLocationResult)
        assert abs(result.x - true_location[0]) <= 10.0
        assert abs(result.y - true_location[1]) <= 10.0
        assert abs(result.z - true_location[2]) <= 10.0

    def test_predicted_amplitudes_shape(self, station_coords, true_location):
        _, observed_amps = synthetic_event(
            true_location, station_coords, source_amplitude=1.0,
        )
        result = amplocate(
            observed_amps,
            station_coords,
            grid_bounds=((-100.0, 150.0), (-100.0, 150.0), (-80.0, 80.0)),
            grid_spacing=(20.0, 20.0, 20.0),
        )
        assert result.predicted_amplitudes.shape == (station_coords.shape[0],)

    def test_misfit_value_is_nonnegative(self, station_coords, true_location):
        _, observed_amps = synthetic_event(
            true_location, station_coords, source_amplitude=1.0,
        )
        result = amplocate(
            observed_amps,
            station_coords,
            grid_bounds=((-100.0, 150.0), (-100.0, 150.0), (-80.0, 80.0)),
            grid_spacing=(20.0, 20.0, 20.0),
        )
        assert result.misfit >= 0.0

    def test_rejects_mismatched_inputs(self, station_coords):
        with pytest.raises(ValueError, match="does not match"):
            amplocate(
                np.array([1.0, 2.0]),  # wrong length
                station_coords,
                grid_bounds=((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)),
                grid_spacing=(5.0, 5.0, 5.0),
            )

    def test_requires_grid_or_distances(self, station_coords):
        with pytest.raises(ValueError, match="grid_bounds"):
            amplocate(np.ones(station_coords.shape[0]), station_coords)

    def test_with_precomputed_distance_grids(self, station_coords, true_location):
        """Test using pre-computed distance grids."""
        # Build a small grid.
        gx = np.arange(-100.0, 155.0, 20.0)
        gy = np.arange(-100.0, 155.0, 20.0)
        gz = np.arange(-80.0, 85.0, 20.0)

        n_stations = station_coords.shape[0]
        distance_grids = np.empty((n_stations, len(gx), len(gy), len(gz)))
        for ii in range(n_stations):
            dx = gx[:, None, None] - station_coords[ii, 0]
            dy = gy[None, :, None] - station_coords[ii, 1]
            dz = gz[None, None, :] - station_coords[ii, 2]
            distance_grids[ii] = np.sqrt(dx**2 + dy**2 + dz**2)

        _, observed_amps = synthetic_event(
            true_location, station_coords, source_amplitude=5.0,
        )

        result = amplocate(
            observed_amps,
            station_coords,
            source_amplitudes=np.array([4.0, 5.0, 6.0]),
            distance_grids=distance_grids,
        )
        assert isinstance(result, AmplitudeLocationResult)
        assert result.misfit >= 0.0


# ---------------------------------------------------------------------------
# Tests for locate_by_travel_time
# ---------------------------------------------------------------------------

class TestLocateByTravelTime:
    def test_recovers_synthetic_location(self, station_coords, true_location):
        """Grid-search should recover the source within one grid cell."""
        velocity = 1020.0
        tt, _ = synthetic_event(
            true_location,
            station_coords,
            velocity=velocity,
            noise_level=0.0,
        )
        result = locate_by_travel_time(
            tt,
            station_coords,
            velocity,
            grid_bounds=((-100.0, 150.0), (-100.0, 150.0), (-80.0, 80.0)),
            grid_spacing=(10.0, 10.0, 10.0),
        )
        assert isinstance(result, LocationResult)
        assert abs(result.x - true_location[0]) <= 10.0
        assert abs(result.y - true_location[1]) <= 10.0
        assert abs(result.z - true_location[2]) <= 10.0

    def test_residual_nonnegative(self, station_coords, true_location):
        velocity = 1020.0
        tt, _ = synthetic_event(
            true_location, station_coords, velocity=velocity,
        )
        result = locate_by_travel_time(
            tt,
            station_coords,
            velocity,
            grid_bounds=((-100.0, 150.0), (-100.0, 150.0), (-80.0, 80.0)),
            grid_spacing=(20.0, 20.0, 20.0),
        )
        assert result.misfit >= 0.0

    def test_rejects_mismatched_inputs(self, station_coords):
        with pytest.raises(ValueError, match="does not match"):
            locate_by_travel_time(
                np.array([0.1, 0.2]),  # wrong length
                station_coords,
                1000.0,
                grid_bounds=((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)),
                grid_spacing=(5.0, 5.0, 5.0),
            )

    def test_reference_station_index(self, station_coords, true_location):
        velocity = 1020.0
        tt, _ = synthetic_event(
            true_location, station_coords, velocity=velocity,
        )
        # Use a different reference station.
        result = locate_by_travel_time(
            tt,
            station_coords,
            velocity,
            grid_bounds=((-100.0, 150.0), (-100.0, 150.0), (-80.0, 80.0)),
            grid_spacing=(10.0, 10.0, 10.0),
            reference_station_index=2,
        )
        assert isinstance(result, LocationResult)
        assert abs(result.x - true_location[0]) <= 10.0
        assert abs(result.y - true_location[1]) <= 10.0
        assert abs(result.z - true_location[2]) <= 10.0
