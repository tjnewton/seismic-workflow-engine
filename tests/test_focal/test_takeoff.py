"""Tests for swe.focal.takeoff."""

from __future__ import annotations

import numpy as np
import pytest

from swe.focal.takeoff import (
    RayParameters,
    compute_takeoff_angles,
    compute_takeoff_angles_1d,
)


class TestComputeTakeoffAngles:
    """Tests for compute_takeoff_angles (homogeneous model)."""

    def test_returns_ray_parameters(self):
        source = {"x": 0.0, "y": 0.0, "z": 10.0}
        stations = {"STA1": [0.0, 0.0, 0.0]}
        result = compute_takeoff_angles(source, stations)
        assert isinstance(result, RayParameters)

    def test_station_directly_above(self):
        """Station directly above source -> takeoff angle = 0° (straight up)."""
        source = {"x": 0.0, "y": 0.0, "z": 10.0}
        stations = {"STA1": [0.0, 0.0, 0.0]}
        result = compute_takeoff_angles(source, stations)
        np.testing.assert_allclose(result.takeoff_angles[0], 0.0, atol=1e-10)

    def test_station_directly_below(self):
        """Station directly below source -> takeoff angle = 180° (straight down)."""
        source = {"x": 0.0, "y": 0.0, "z": 0.0}
        stations = {"STA1": [0.0, 0.0, 10.0]}
        result = compute_takeoff_angles(source, stations)
        np.testing.assert_allclose(result.takeoff_angles[0], 180.0, atol=1e-10)

    def test_station_horizontal(self):
        """Station at same depth, offset horizontally -> takeoff angle = 90°."""
        source = {"x": 0.0, "y": 0.0, "z": 5.0}
        stations = {"STA1": [10.0, 0.0, 5.0]}
        result = compute_takeoff_angles(source, stations)
        np.testing.assert_allclose(result.takeoff_angles[0], 90.0, atol=1e-10)

    def test_azimuth_north(self):
        """Station due North should have azimuth 0°."""
        source = {"x": 0.0, "y": 0.0, "z": 5.0}
        stations = {"STA1": [0.0, 10.0, 5.0]}
        result = compute_takeoff_angles(source, stations)
        np.testing.assert_allclose(result.azimuths[0], 0.0, atol=1e-10)

    def test_azimuth_east(self):
        """Station due East should have azimuth 90°."""
        source = {"x": 0.0, "y": 0.0, "z": 5.0}
        stations = {"STA1": [10.0, 0.0, 5.0]}
        result = compute_takeoff_angles(source, stations)
        np.testing.assert_allclose(result.azimuths[0], 90.0, atol=1e-10)

    def test_azimuth_south(self):
        """Station due South should have azimuth 180°."""
        source = {"x": 0.0, "y": 0.0, "z": 5.0}
        stations = {"STA1": [0.0, -10.0, 5.0]}
        result = compute_takeoff_angles(source, stations)
        np.testing.assert_allclose(result.azimuths[0], 180.0, atol=1e-10)

    def test_azimuth_west(self):
        """Station due West should have azimuth 270°."""
        source = {"x": 0.0, "y": 0.0, "z": 5.0}
        stations = {"STA1": [-10.0, 0.0, 5.0]}
        result = compute_takeoff_angles(source, stations)
        np.testing.assert_allclose(result.azimuths[0], 270.0, atol=1e-10)

    def test_distance_calculation(self):
        """Distance should be Euclidean."""
        source = {"x": 0.0, "y": 0.0, "z": 0.0}
        stations = {"STA1": [3.0, 4.0, 0.0]}
        result = compute_takeoff_angles(source, stations)
        np.testing.assert_allclose(result.distances[0], 5.0, atol=1e-10)

    def test_multiple_stations(self):
        """Should return results for all stations."""
        source = {"x": 0.0, "y": 0.0, "z": 10.0}
        stations = {
            "STA1": [0.0, 0.0, 0.0],
            "STA2": [10.0, 0.0, 0.0],
            "STA3": [0.0, 10.0, 0.0],
        }
        result = compute_takeoff_angles(source, stations)
        assert len(result.stations) == 3
        assert len(result.azimuths) == 3
        assert len(result.takeoff_angles) == 3
        assert len(result.distances) == 3

    def test_coincident_station_skipped(self):
        """Station at source location should be skipped."""
        source = {"x": 0.0, "y": 0.0, "z": 10.0}
        stations = {"STA1": [0.0, 0.0, 10.0]}
        result = compute_takeoff_angles(source, stations)
        assert len(result.stations) == 0

    def test_45_degree_takeoff(self):
        """Station with equal horizontal and vertical offset -> ~45° from vertical."""
        source = {"x": 0.0, "y": 0.0, "z": 10.0}
        stations = {"STA1": [10.0, 0.0, 0.0]}
        result = compute_takeoff_angles(source, stations)
        np.testing.assert_allclose(result.takeoff_angles[0], 45.0, atol=1e-10)


class TestComputeTakeoffAngles1d:
    """Tests for compute_takeoff_angles_1d (layered model)."""

    def test_returns_ray_parameters(self):
        source = {"x": 0.0, "y": 0.0, "z": 10.0}
        stations = {"STA1": [0.0, 0.0, 0.0]}
        profile = [[0.0, 5.0], [20.0, 6.5]]
        result = compute_takeoff_angles_1d(source, stations, velocity_profile=profile)
        assert isinstance(result, RayParameters)

    def test_homogeneous_matches_straight_ray(self):
        """Uniform velocity should give same result as homogeneous model."""
        source = {"x": 0.0, "y": 0.0, "z": 10.0}
        stations = {"STA1": [5.0, 0.0, 0.0]}
        profile = [[0.0, 6.0]]  # Single layer

        result_homo = compute_takeoff_angles(source, stations, velocity=6.0)
        result_1d = compute_takeoff_angles_1d(source, stations, velocity_profile=profile)

        np.testing.assert_allclose(
            result_1d.takeoff_angles[0],
            result_homo.takeoff_angles[0],
            atol=1.0,  # Allow 1° tolerance from bisection
        )

    def test_invalid_profile_shape(self):
        """Should raise ValueError for invalid profile shape."""
        source = {"x": 0.0, "y": 0.0, "z": 10.0}
        stations = {"STA1": [0.0, 0.0, 0.0]}
        with pytest.raises(ValueError, match="shape"):
            compute_takeoff_angles_1d(source, stations, velocity_profile=[[1.0, 2.0, 3.0]])

    def test_azimuths_match_homogeneous(self):
        """Azimuths should be identical regardless of velocity model."""
        source = {"x": 0.0, "y": 0.0, "z": 10.0}
        stations = {"STA1": [5.0, 5.0, 0.0]}
        profile = [[0.0, 4.0], [5.0, 6.0], [15.0, 8.0]]

        result_homo = compute_takeoff_angles(source, stations)
        result_1d = compute_takeoff_angles_1d(source, stations, velocity_profile=profile)

        np.testing.assert_allclose(
            result_1d.azimuths[0],
            result_homo.azimuths[0],
            atol=1e-10,
        )
