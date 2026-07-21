"""Tests for swe.uncertainty.types."""

from __future__ import annotations

import numpy as np
import pytest

from swe.uncertainty.types import (
    LocationUncertainty,
    MagnitudeUncertainty,
    UncertainValue,
)


class TestUncertainValue:
    """Tests for UncertainValue frozen dataclass."""

    def test_creation_defaults(self):
        uv = UncertainValue(value=2.5)
        assert uv.value == 2.5
        assert uv.uncertainty == 0.0
        assert uv.confidence_level == 0.68
        assert uv.lower_bound is None
        assert uv.upper_bound is None
        assert uv.n_samples == 0
        assert uv.method == ""

    def test_creation_full(self):
        uv = UncertainValue(
            value=3.0,
            uncertainty=0.1,
            confidence_level=0.95,
            lower_bound=2.8,
            upper_bound=3.2,
            n_samples=100,
            method="bootstrap",
        )
        assert uv.value == 3.0
        assert uv.uncertainty == 0.1
        assert uv.n_samples == 100

    def test_frozen(self):
        uv = UncertainValue(value=1.0)
        with pytest.raises(AttributeError):
            uv.value = 2.0  # type: ignore[misc]

    def test_equality(self):
        a = UncertainValue(value=1.0, uncertainty=0.1)
        b = UncertainValue(value=1.0, uncertainty=0.1)
        assert a == b

    def test_inequality(self):
        a = UncertainValue(value=1.0, uncertainty=0.1)
        b = UncertainValue(value=1.0, uncertainty=0.2)
        assert a != b


class TestLocationUncertainty:
    """Tests for LocationUncertainty frozen dataclass."""

    def test_creation_defaults(self):
        lu = LocationUncertainty()
        assert lu.horiz_uncertainty_m == 0.0
        assert lu.depth_uncertainty_m == 0.0
        assert lu.rms == 0.0
        assert lu.covariance_matrix is None
        assert lu.n_samples == 0
        assert lu.method == ""

    def test_creation_with_covariance(self):
        cov = np.eye(3) * 100.0
        lu = LocationUncertainty(
            horiz_uncertainty_m=50.0,
            depth_uncertainty_m=100.0,
            rms=0.15,
            covariance_matrix=cov,
            n_samples=200,
            method="monte_carlo",
        )
        assert lu.horiz_uncertainty_m == 50.0
        assert lu.n_samples == 200
        np.testing.assert_array_equal(lu.covariance_matrix, cov)

    def test_frozen(self):
        lu = LocationUncertainty(horiz_uncertainty_m=10.0)
        with pytest.raises(AttributeError):
            lu.horiz_uncertainty_m = 20.0  # type: ignore[misc]

    def test_equality_no_cov(self):
        a = LocationUncertainty(horiz_uncertainty_m=10.0, rms=0.1)
        b = LocationUncertainty(horiz_uncertainty_m=10.0, rms=0.1)
        assert a == b

    def test_equality_with_cov(self):
        cov = np.eye(3) * 50.0
        a = LocationUncertainty(horiz_uncertainty_m=10.0, covariance_matrix=cov)
        b = LocationUncertainty(horiz_uncertainty_m=10.0, covariance_matrix=cov.copy())
        assert a == b

    def test_inequality_cov_none_vs_array(self):
        cov = np.eye(3)
        a = LocationUncertainty(covariance_matrix=None)
        b = LocationUncertainty(covariance_matrix=cov)
        assert a != b

    def test_hashable(self):
        lu = LocationUncertainty(horiz_uncertainty_m=10.0)
        assert hash(lu) == hash(lu)
        s = {lu}
        assert lu in s


class TestMagnitudeUncertainty:
    """Tests for MagnitudeUncertainty frozen dataclass."""

    def test_creation_defaults(self):
        mu = MagnitudeUncertainty()
        assert mu.mag_uncertainty == 0.0
        assert mu.station_std == 0.0
        assert mu.station_stderr == 0.0
        assert mu.distance_contribution == 0.0
        assert mu.n_valid_stations == 0
        assert mu.method == ""

    def test_creation_full(self):
        mu = MagnitudeUncertainty(
            mag_uncertainty=0.15,
            station_std=0.12,
            station_stderr=0.06,
            distance_contribution=0.08,
            n_valid_stations=4,
            method="analytical",
        )
        assert mu.mag_uncertainty == 0.15
        assert mu.distance_contribution == 0.08

    def test_frozen(self):
        mu = MagnitudeUncertainty(mag_uncertainty=0.1)
        with pytest.raises(AttributeError):
            mu.mag_uncertainty = 0.2  # type: ignore[misc]

    def test_equality(self):
        a = MagnitudeUncertainty(mag_uncertainty=0.1, station_std=0.2)
        b = MagnitudeUncertainty(mag_uncertainty=0.1, station_std=0.2)
        assert a == b
