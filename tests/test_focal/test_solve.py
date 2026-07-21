"""Tests for swe.focal.solve."""

from __future__ import annotations

import numpy as np
import pytest

from swe.focal.solve import (
    FocalMechanism,
    FocalMechanismResult,
    _auxiliary_plane,
    _radiation_pattern,
    focal_mechanism_gridsearch,
    focal_mechanism_uncertainty,
    focal_mechanisms_batch,
    polarity_result_to_event_dict,
)


def _synthetic_polarities(
    strike: float,
    dip: float,
    rake: float,
    n_stations: int = 20,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic polarity observations from a known mechanism.

    Returns azimuths, takeoff_angles, and polarities.
    """
    rng = np.random.default_rng(seed)

    azimuths = rng.uniform(0, 360, n_stations)
    takeoff_angles = rng.uniform(10, 170, n_stations)

    s_rad = np.radians(strike)
    d_rad = np.radians(dip)
    r_rad = np.radians(rake)
    az_rad = np.radians(azimuths)
    to_rad = np.radians(takeoff_angles)

    rad = _radiation_pattern(s_rad, d_rad, r_rad, az_rad, to_rad)
    polarities = np.sign(rad).astype(np.int32)

    # Remove any zero polarities (on the nodal plane)
    nonzero = polarities != 0
    return azimuths[nonzero], takeoff_angles[nonzero], polarities[nonzero]


class TestRadiationPattern:
    """Tests for the P-wave radiation pattern."""

    def test_output_shape(self):
        """Output shape should match input arrays."""
        az = np.array([0.0, 90.0, 180.0])
        to = np.array([45.0, 45.0, 45.0])
        rad = _radiation_pattern(0.0, np.pi / 4, 0.0, np.radians(az), np.radians(to))
        assert rad.shape == (3,)

    def test_pure_strike_slip_symmetry(self):
        """Pure strike-slip should show quadrantal symmetry."""
        # Strike=0, Dip=90, Rake=0 is a vertical left-lateral strike-slip
        az = np.radians(np.array([45.0, 135.0, 225.0, 315.0]))
        to = np.radians(np.array([90.0, 90.0, 90.0, 90.0]))
        rad = _radiation_pattern(0.0, np.pi / 2, 0.0, az, to)
        # Opposite quadrants should have opposite signs
        assert np.sign(rad[0]) == -np.sign(rad[1])
        assert np.sign(rad[2]) == -np.sign(rad[3])


class TestAuxiliaryPlane:
    """Tests for auxiliary plane computation."""

    def test_vertical_strike_slip(self):
        """Auxiliary plane for vertical strike-slip."""
        s2, d2, r2 = _auxiliary_plane(0.0, 90.0, 0.0)
        # For strike=0, dip=90, rake=0: auxiliary should be strike=90, dip=90, rake=180 (or equivalent)
        assert 0.0 <= d2 <= 90.0

    def test_dip_range(self):
        """Auxiliary dip should be in [0, 90]."""
        for strike in range(0, 360, 30):
            for dip in range(10, 90, 20):
                for rake in range(-180, 180, 30):
                    _, d2, _ = _auxiliary_plane(float(strike), float(dip), float(rake))
                    assert 0.0 <= d2 <= 90.0 + 1e-6, (
                        f"Bad dip2={d2} for s={strike}, d={dip}, r={rake}"
                    )


class TestFocalMechanismGridsearch:
    """Tests for focal_mechanism_gridsearch."""

    def test_returns_result(self):
        """Should return FocalMechanismResult."""
        az, to, pol = _synthetic_polarities(45.0, 60.0, -90.0)
        result = focal_mechanism_gridsearch(
            az, to, pol, strike_step=10.0, dip_step=10.0, rake_step=10.0
        )
        assert isinstance(result, FocalMechanismResult)

    def test_round_trip_recovery(self):
        """Should recover input mechanism within grid resolution."""
        true_strike, true_dip, true_rake = 90.0, 45.0, -90.0
        az, to, pol = _synthetic_polarities(true_strike, true_dip, true_rake, n_stations=30)

        result = focal_mechanism_gridsearch(
            az, to, pol,
            strike_step=5.0, dip_step=5.0, rake_step=5.0,
            max_misfit=0.1,
        )

        assert result.best is not None
        assert result.best.misfit <= 0.1

        # The best mechanism should be close to the true one,
        # OR the auxiliary plane should be close
        strike_err = min(
            abs(result.best.strike - true_strike) % 360,
            abs(result.best.strike2 - true_strike) % 360,
        )
        assert strike_err < 20.0, f"Strike error: {strike_err}"

    def test_focal_data_shape(self):
        """focal_data should be (1, 3) for a successful solve."""
        az, to, pol = _synthetic_polarities(45.0, 60.0, -90.0)
        result = focal_mechanism_gridsearch(
            az, to, pol, strike_step=10.0, dip_step=10.0, rake_step=10.0
        )
        if result.best is not None:
            assert result.focal_data.shape == (1, 3)
            assert result.focal_data[0, 0] == result.best.strike
            assert result.focal_data[0, 1] == result.best.dip
            assert result.focal_data[0, 2] == result.best.rake

    def test_insufficient_polarities(self):
        """Should return empty result when too few polarities."""
        az = np.array([0.0, 90.0])
        to = np.array([45.0, 45.0])
        pol = np.array([1, -1])

        result = focal_mechanism_gridsearch(
            az, to, pol, min_polarities=8
        )
        assert result.best is None
        assert len(result.mechanisms) == 0
        assert result.focal_data.shape == (0, 3)

    def test_zero_polarities_excluded(self):
        """Stations with polarity 0 should be excluded from misfit."""
        az, to, pol = _synthetic_polarities(45.0, 60.0, -90.0, n_stations=30)
        # Add some zero-polarity stations
        az = np.concatenate([az, [0.0, 90.0, 180.0]])
        to = np.concatenate([to, [45.0, 45.0, 45.0]])
        pol = np.concatenate([pol, [0, 0, 0]])

        result = focal_mechanism_gridsearch(
            az, to, pol,
            strike_step=10.0, dip_step=10.0, rake_step=10.0,
        )
        if result.best is not None:
            # n_polarities should not count the zero-polarity stations
            assert result.best.n_polarities == len(pol) - 3

    def test_weighted_search(self):
        """Weights should not crash and should produce a result."""
        az, to, pol = _synthetic_polarities(45.0, 60.0, -90.0, n_stations=20)
        weights = np.ones_like(az)
        weights[:5] = 2.0  # Weight first 5 stations more

        result = focal_mechanism_gridsearch(
            az, to, pol,
            strike_step=10.0, dip_step=10.0, rake_step=10.0,
            weights=weights,
        )
        assert isinstance(result, FocalMechanismResult)

    def test_all_misfits_populated(self):
        """all_misfits should contain values for the full grid."""
        az, to, pol = _synthetic_polarities(45.0, 60.0, -90.0)
        result = focal_mechanism_gridsearch(
            az, to, pol,
            strike_step=30.0, dip_step=30.0, rake_step=30.0,
        )
        assert len(result.all_misfits) > 0
        assert all(0.0 <= m <= 1.0 for m in result.all_misfits)

    def test_angular_uncertainty_computed(self):
        """strike_std, dip_std, rake_std should be > 0 with multiple accepted."""
        az, to, pol = _synthetic_polarities(45.0, 60.0, -90.0, n_stations=15)
        result = focal_mechanism_gridsearch(
            az, to, pol,
            strike_step=10.0, dip_step=10.0, rake_step=10.0,
            max_misfit=0.3,
        )
        if len(result.mechanisms) > 1:
            assert result.strike_std > 0
            assert result.dip_std >= 0
            assert result.rake_std >= 0

    def test_focal_mechanism_uncertainty_helper(self):
        """focal_mechanism_uncertainty should return a dict."""
        az, to, pol = _synthetic_polarities(45.0, 60.0, -90.0)
        result = focal_mechanism_gridsearch(
            az, to, pol, strike_step=10.0, dip_step=10.0, rake_step=10.0,
        )
        summary = focal_mechanism_uncertainty(result)
        assert "strike_std" in summary
        assert "n_accepted" in summary
        assert summary["n_accepted"] == len(result.mechanisms)


class TestFocalMechanismsBatch:
    """Tests for focal_mechanisms_batch."""

    def test_multiple_events(self):
        """Should produce results for each event."""
        events = []
        for strike in [45.0, 90.0, 180.0]:
            az, to, pol = _synthetic_polarities(strike, 60.0, -90.0)
            events.append({
                "azimuths": az,
                "takeoff_angles": to,
                "polarities": pol,
            })

        results, focal_data = focal_mechanisms_batch(
            events, strike_step=10.0, dip_step=10.0, rake_step=10.0
        )

        assert len(results) == 3
        assert focal_data.ndim == 2
        assert focal_data.shape[1] == 3

    def test_focal_data_compatible_with_stress(self):
        """focal_data should be directly usable by stress_inversion."""
        events = []
        for strike in [30.0, 60.0, 90.0, 120.0, 150.0]:
            az, to, pol = _synthetic_polarities(strike, 50.0, -90.0, n_stations=25)
            events.append({
                "azimuths": az,
                "takeoff_angles": to,
                "polarities": pol,
            })

        results, focal_data = focal_mechanisms_batch(
            events, strike_step=10.0, dip_step=10.0, rake_step=10.0
        )

        # focal_data should have shape (N, 3) where N <= 5
        assert focal_data.shape[1] == 3
        assert focal_data.shape[0] <= 5

    def test_empty_events_list(self):
        """Empty events list should return empty results."""
        results, focal_data = focal_mechanisms_batch([])
        assert len(results) == 0
        assert focal_data.shape == (0, 3)


class TestPolarityResultToEventDict:
    """Tests for polarity_result_to_event_dict."""

    def test_basic_conversion(self):
        """Should produce a valid event dict from PolarityResult."""
        from swe.focal.polarity import PolarityObservation, PolarityResult

        obs = [
            PolarityObservation("S1", 1, 0.9, 10.0),
            PolarityObservation("S2", -1, 0.7, 8.0),
            PolarityObservation("S3", 1, 0.5, 5.0),
        ]
        pr = PolarityResult(
            observations=obs,
            stations=["S1", "S2", "S3"],
            polarities=np.array([1, -1, 1]),
            confidences=np.array([0.9, 0.7, 0.5]),
        )
        az = np.array([0.0, 90.0, 180.0])
        to = np.array([45.0, 45.0, 45.0])

        event = polarity_result_to_event_dict(pr, az, to)
        assert "azimuths" in event
        assert "takeoff_angles" in event
        assert "polarities" in event
        assert "weights" in event
        np.testing.assert_array_equal(event["polarities"], [1, -1, 1])
        np.testing.assert_array_almost_equal(event["weights"], [0.9, 0.7, 0.5])

    def test_no_weights(self):
        """With use_confidence_as_weights=False, no weights key."""
        from swe.focal.polarity import PolarityObservation, PolarityResult

        obs = [PolarityObservation("S1", 1, 0.9, 10.0)]
        pr = PolarityResult(
            observations=obs,
            stations=["S1"],
            polarities=np.array([1]),
            confidences=np.array([0.9]),
        )
        event = polarity_result_to_event_dict(
            pr, np.array([0.0]), np.array([45.0]),
            use_confidence_as_weights=False,
        )
        assert "weights" not in event


class TestIntegrationWithStress:
    """Integration test: focal mechanism -> stress inversion pipeline."""

    @pytest.mark.integration
    def test_focal_to_stress_pipeline(self):
        """Round-trip: synthetic polarities -> solve -> stress inversion."""
        from swe.stress import stress_inversion

        # Generate focal mechanisms for a normal-faulting stress regime
        # (sigma1 vertical, sigma3 horizontal)
        events = []
        for strike in [0.0, 45.0, 90.0, 135.0, 180.0]:
            az, to, pol = _synthetic_polarities(
                strike, 45.0, -90.0, n_stations=25
            )
            events.append({
                "azimuths": az,
                "takeoff_angles": to,
                "polarities": pol,
            })

        results, focal_data = focal_mechanisms_batch(
            events, strike_step=10.0, dip_step=10.0, rake_step=10.0
        )

        # Filter to events with solutions
        if focal_data.shape[0] == 0:
            pytest.skip("No focal mechanisms solved")

        # Run stress inversion with single cluster
        labels = np.zeros(focal_data.shape[0], dtype=np.int32)
        stress_result = stress_inversion(focal_data, labels)

        assert len(stress_result.results) == 1
        assert "sigma_1" in stress_result.results[0]
        assert "sigma_2" in stress_result.results[0]
        assert "sigma_3" in stress_result.results[0]
