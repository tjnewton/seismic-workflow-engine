"""Tests for swe.uncertainty.magnitude."""

from __future__ import annotations

import numpy as np
import pytest

from swe.uncertainty.magnitude import propagate_location_to_magnitude
from swe.uncertainty.types import LocationUncertainty, MagnitudeUncertainty


class TestPropagateLocationToMagnitude:
    """Tests for propagate_location_to_magnitude."""

    def test_basic_output(self):
        lu = LocationUncertainty(horiz_uncertainty_m=500.0)
        result = propagate_location_to_magnitude(
            ml_std=0.12,
            ml_stderr=0.06,
            n_valid_stations=4,
            distances_km=np.array([10.0, 20.0, 30.0, 40.0]),
            location_uncertainty=lu,
        )
        assert isinstance(result, MagnitudeUncertainty)
        assert result.method == "analytical"
        assert result.n_valid_stations == 4
        assert result.station_std == 0.12
        assert result.station_stderr == 0.06

    def test_distance_contribution_positive(self):
        lu = LocationUncertainty(horiz_uncertainty_m=1000.0)
        result = propagate_location_to_magnitude(
            ml_std=0.12,
            ml_stderr=0.06,
            n_valid_stations=4,
            distances_km=np.array([10.0, 20.0, 30.0, 40.0]),
            location_uncertainty=lu,
        )
        assert result.distance_contribution > 0.0

    def test_zero_location_uncertainty(self):
        lu = LocationUncertainty(horiz_uncertainty_m=0.0)
        result = propagate_location_to_magnitude(
            ml_std=0.12,
            ml_stderr=0.06,
            n_valid_stations=4,
            distances_km=np.array([10.0, 20.0, 30.0, 40.0]),
            location_uncertainty=lu,
        )
        assert result.distance_contribution == 0.0
        assert result.mag_uncertainty == pytest.approx(0.06)

    def test_larger_location_uncertainty_gives_larger_mag_uncertainty(self):
        distances = np.array([15.0, 25.0, 35.0])

        lu_small = LocationUncertainty(horiz_uncertainty_m=100.0)
        result_small = propagate_location_to_magnitude(
            ml_std=0.1,
            ml_stderr=0.05,
            n_valid_stations=3,
            distances_km=distances,
            location_uncertainty=lu_small,
        )

        lu_large = LocationUncertainty(horiz_uncertainty_m=5000.0)
        result_large = propagate_location_to_magnitude(
            ml_std=0.1,
            ml_stderr=0.05,
            n_valid_stations=3,
            distances_km=distances,
            location_uncertainty=lu_large,
        )

        assert result_large.mag_uncertainty > result_small.mag_uncertainty
        assert result_large.distance_contribution > result_small.distance_contribution

    def test_combined_quadrature(self):
        lu = LocationUncertainty(horiz_uncertainty_m=1000.0)
        result = propagate_location_to_magnitude(
            ml_std=0.12,
            ml_stderr=0.06,
            n_valid_stations=4,
            distances_km=np.array([20.0]),
            location_uncertainty=lu,
        )
        expected = np.sqrt(0.06**2 + result.distance_contribution**2)
        assert result.mag_uncertainty == pytest.approx(expected, rel=1e-10)

    def test_zero_distance_stations_skipped(self):
        lu = LocationUncertainty(horiz_uncertainty_m=500.0)
        result = propagate_location_to_magnitude(
            ml_std=0.0,
            ml_stderr=0.0,
            n_valid_stations=1,
            distances_km=np.array([0.0, 20.0]),
            location_uncertainty=lu,
        )
        # Should not raise, zero-distance stations contribute zero
        assert result.distance_contribution >= 0.0

    def test_close_stations_more_sensitive(self):
        """Closer stations have steeper dML/dr, so should be more sensitive."""
        lu = LocationUncertainty(horiz_uncertainty_m=1000.0)
        result_close = propagate_location_to_magnitude(
            ml_std=0.1,
            ml_stderr=0.05,
            n_valid_stations=1,
            distances_km=np.array([5.0]),
            location_uncertainty=lu,
        )
        result_far = propagate_location_to_magnitude(
            ml_std=0.1,
            ml_stderr=0.05,
            n_valid_stations=1,
            distances_km=np.array([100.0]),
            location_uncertainty=lu,
        )
        assert result_close.distance_contribution > result_far.distance_contribution
