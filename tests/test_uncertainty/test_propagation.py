"""Tests for swe.uncertainty.propagation."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from swe.uncertainty.propagation import (
    hypocenters_to_location_uncertainty,
    monte_carlo_location_uncertainty,
)
from swe.uncertainty.types import LocationUncertainty


class _FakeLocation(NamedTuple):
    x: float
    y: float
    z: float
    misfit: float


class TestMonteCarloLocationUncertainty:
    """Tests for monte_carlo_location_uncertainty."""

    def _dummy_locator(self, **kwargs):
        """Simple locator: best location is mean of observed_times."""
        times = kwargs["observed_times"]
        return _FakeLocation(
            x=float(times[0]) * 100,
            y=float(times[1]) * 100,
            z=float(times[2]) * 50,
            misfit=0.0,
        )

    def test_basic_output(self):
        pick_times = np.array([1.0, 2.0, 3.0])
        pick_unc = np.array([0.01, 0.01, 0.01])
        result = monte_carlo_location_uncertainty(
            pick_times,
            pick_unc,
            self._dummy_locator,
            {},
            n_samples=50,
            random_state=42,
        )
        assert isinstance(result, LocationUncertainty)
        assert result.method == "monte_carlo"
        assert result.n_samples == 50

    def test_uncertainty_increases_with_pick_error(self):
        pick_times = np.array([1.0, 2.0, 3.0])

        small_unc = np.array([0.001, 0.001, 0.001])
        result_small = monte_carlo_location_uncertainty(
            pick_times, small_unc, self._dummy_locator, {},
            n_samples=200, random_state=0,
        )

        large_unc = np.array([0.1, 0.1, 0.1])
        result_large = monte_carlo_location_uncertainty(
            pick_times, large_unc, self._dummy_locator, {},
            n_samples=200, random_state=0,
        )

        assert result_large.horiz_uncertainty_m > result_small.horiz_uncertainty_m

    def test_covariance_matrix_shape(self):
        pick_times = np.array([1.0, 2.0, 3.0])
        pick_unc = np.array([0.01, 0.01, 0.01])
        result = monte_carlo_location_uncertainty(
            pick_times, pick_unc, self._dummy_locator, {},
            n_samples=50, random_state=0,
        )
        assert result.covariance_matrix is not None
        assert result.covariance_matrix.shape == (3, 3)

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            monte_carlo_location_uncertainty(
                np.array([1.0, 2.0]),
                np.array([0.01]),
                self._dummy_locator,
                {},
            )

    def test_all_relocations_fail(self):
        def _fail_locator(**kwargs):
            raise RuntimeError("fail")

        result = monte_carlo_location_uncertainty(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.01, 0.01, 0.01]),
            _fail_locator,
            {},
            n_samples=5,
        )
        assert result.n_samples == 0
        assert result.horiz_uncertainty_m == 0.0

    def test_deterministic_with_seed(self):
        pick_times = np.array([1.0, 2.0, 3.0])
        pick_unc = np.array([0.01, 0.01, 0.01])

        r1 = monte_carlo_location_uncertainty(
            pick_times, pick_unc, self._dummy_locator, {},
            n_samples=50, random_state=123,
        )
        r2 = monte_carlo_location_uncertainty(
            pick_times, pick_unc, self._dummy_locator, {},
            n_samples=50, random_state=123,
        )
        assert r1.horiz_uncertainty_m == r2.horiz_uncertainty_m


class TestHypocentersToLocationUncertainty:
    """Tests for hypocenters_to_location_uncertainty."""

    def test_empty_list(self):
        result = hypocenters_to_location_uncertainty([])
        assert result.method == "nll_scatter"
        assert result.n_samples == 0

    def test_single_hypocenter(self):
        from swe.location.nonlinloc import HypocenterResult

        hyp = HypocenterResult(
            x_km=10.0, y_km=20.0, z_m=5000.0,
            horiz_uncertainty_m=100.0,
            depth_uncertainty_m=200.0,
            rms=0.15,
        )
        result = hypocenters_to_location_uncertainty([hyp])
        assert result.horiz_uncertainty_m == 100.0
        assert result.depth_uncertainty_m == 200.0
        assert result.rms == 0.15
        assert result.n_samples == 1
        assert result.method == "nll_scatter"

    def test_multiple_hypocenters(self):
        from swe.location.nonlinloc import HypocenterResult

        hyps = [
            HypocenterResult(0, 0, 0, 100.0, depth_uncertainty_m=200.0, rms=0.1),
            HypocenterResult(1, 1, 100, 300.0, depth_uncertainty_m=400.0, rms=0.2),
        ]
        result = hypocenters_to_location_uncertainty(hyps)
        assert result.horiz_uncertainty_m == pytest.approx(200.0)
        assert result.depth_uncertainty_m == pytest.approx(300.0)
        assert result.rms == pytest.approx(0.15)
        assert result.n_samples == 2
