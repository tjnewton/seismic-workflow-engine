"""Tests for swe.uncertainty.conversions."""

from __future__ import annotations

import numpy as np
from obspy.core.event import QuantityError

from swe.uncertainty.conversions import (
    location_uncertainty_from_dict,
    location_uncertainty_to_dict,
    location_uncertainty_to_quantity_errors,
    magnitude_uncertainty_from_dict,
    magnitude_uncertainty_to_dict,
    quantity_error_to_uncertain_value,
    uncertain_value_from_dict,
    uncertain_value_to_dict,
)
from swe.uncertainty.types import (
    LocationUncertainty,
    MagnitudeUncertainty,
    UncertainValue,
)


class TestUncertainValueConversions:
    """Round-trip tests for UncertainValue."""

    def test_roundtrip_defaults(self):
        uv = UncertainValue(value=5.0)
        d = uncertain_value_to_dict(uv)
        restored = uncertain_value_from_dict(d)
        assert restored == uv

    def test_roundtrip_full(self):
        uv = UncertainValue(
            value=3.0,
            uncertainty=0.1,
            confidence_level=0.95,
            lower_bound=2.8,
            upper_bound=3.2,
            n_samples=100,
            method="bootstrap",
        )
        d = uncertain_value_to_dict(uv)
        restored = uncertain_value_from_dict(d)
        assert restored == uv

    def test_dict_is_json_safe(self):
        import json

        uv = UncertainValue(value=1.0, uncertainty=0.5)
        d = uncertain_value_to_dict(uv)
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


class TestLocationUncertaintyConversions:
    """Round-trip tests for LocationUncertainty."""

    def test_roundtrip_no_cov(self):
        lu = LocationUncertainty(
            horiz_uncertainty_m=50.0,
            depth_uncertainty_m=100.0,
            rms=0.15,
            n_samples=200,
            method="monte_carlo",
        )
        d = location_uncertainty_to_dict(lu)
        restored = location_uncertainty_from_dict(d)
        assert restored.horiz_uncertainty_m == lu.horiz_uncertainty_m
        assert restored.depth_uncertainty_m == lu.depth_uncertainty_m
        assert restored.rms == lu.rms
        assert restored.n_samples == lu.n_samples
        assert restored.method == lu.method
        assert restored.covariance_matrix is None

    def test_roundtrip_with_cov(self):
        cov = np.array([[100, 10, 5], [10, 200, 15], [5, 15, 300]], dtype=float)
        lu = LocationUncertainty(
            horiz_uncertainty_m=50.0,
            covariance_matrix=cov,
        )
        d = location_uncertainty_to_dict(lu)
        restored = location_uncertainty_from_dict(d)
        assert restored.horiz_uncertainty_m == 50.0
        np.testing.assert_array_almost_equal(restored.covariance_matrix, cov)

    def test_dict_is_json_safe(self):
        import json

        lu = LocationUncertainty(
            horiz_uncertainty_m=50.0,
            covariance_matrix=np.eye(3),
        )
        d = location_uncertainty_to_dict(lu)
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


class TestMagnitudeUncertaintyConversions:
    """Round-trip tests for MagnitudeUncertainty."""

    def test_roundtrip(self):
        mu = MagnitudeUncertainty(
            mag_uncertainty=0.15,
            station_std=0.12,
            station_stderr=0.06,
            distance_contribution=0.08,
            n_valid_stations=4,
            method="analytical",
        )
        d = magnitude_uncertainty_to_dict(mu)
        restored = magnitude_uncertainty_from_dict(d)
        assert restored == mu

    def test_dict_is_json_safe(self):
        import json

        mu = MagnitudeUncertainty(mag_uncertainty=0.1)
        d = magnitude_uncertainty_to_dict(mu)
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


class TestObspyInterop:
    """Tests for ObsPy conversion helpers."""

    def test_location_to_quantity_errors(self):
        lu = LocationUncertainty(
            horiz_uncertainty_m=1000.0,
            depth_uncertainty_m=500.0,
        )
        qe = location_uncertainty_to_quantity_errors(lu)
        assert "depth_errors" in qe
        assert "latitude_errors" in qe
        assert "longitude_errors" in qe
        assert qe["depth_errors"].uncertainty == 500.0
        expected_deg = 1000.0 / 111_320.0
        assert abs(qe["latitude_errors"].uncertainty - expected_deg) < 1e-10

    def test_quantity_error_to_uncertain_value(self):
        qe = QuantityError(uncertainty=0.5)
        uv = quantity_error_to_uncertain_value(qe, value=3.0)
        assert uv.value == 3.0
        assert uv.uncertainty == 0.5

    def test_quantity_error_none_uncertainty(self):
        qe = QuantityError()
        uv = quantity_error_to_uncertain_value(qe)
        assert uv.uncertainty == 0.0
