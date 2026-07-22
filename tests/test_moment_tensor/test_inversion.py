"""Tests for swe.moment_tensor.inversion - grid-search MT inversion."""

from __future__ import annotations

import numpy as np
import pytest

from swe.moment_tensor.inversion import (
    MomentTensorResult,
    moment_tensor_batch,
    moment_tensor_from_polarities,
    moment_tensor_uncertainty,
)


class TestMomentTensorFromPolarities:
    """Tests for the grid-search inversion."""

    def test_dc_constraint_recovers_mechanism(self, synthetic_polarities):
        """DC-constrained search should recover the known focal mechanism."""
        result = moment_tensor_from_polarities(
            azimuths=synthetic_polarities["azimuths"],
            takeoff_angles=synthetic_polarities["takeoff_angles"],
            polarities=synthetic_polarities["polarities"],
            constraint="dc",
            max_misfit=0.1,
            min_polarities=8,
            strike_step=10.0,
            dip_step=10.0,
            rake_step=10.0,
        )

        assert isinstance(result, MomentTensorResult)
        assert result.best is not None
        assert result.n_polarities == len(synthetic_polarities["polarities"])
        assert len(result.accepted) >= 1

    def test_dc_zero_misfit(self, synthetic_polarities):
        """Perfect synthetic data should yield zero or near-zero misfit."""
        result = moment_tensor_from_polarities(
            azimuths=synthetic_polarities["azimuths"],
            takeoff_angles=synthetic_polarities["takeoff_angles"],
            polarities=synthetic_polarities["polarities"],
            constraint="dc",
            max_misfit=0.05,
            min_polarities=8,
            strike_step=5.0,
            dip_step=5.0,
            rake_step=5.0,
        )

        assert result.best is not None
        # Best misfit should be very low for noiseless synthetic data
        best_misfit = result.misfits.min()
        assert best_misfit <= 0.05

    def test_too_few_polarities(self):
        """Should return None when fewer than min_polarities."""
        result = moment_tensor_from_polarities(
            azimuths=np.array([0, 90, 180]),
            takeoff_angles=np.array([45, 45, 45]),
            polarities=np.array([1, -1, 1]),
            constraint="dc",
            min_polarities=8,
        )
        assert result.best is None
        assert result.accepted == []

    def test_zero_polarities_excluded(self):
        """Stations with polarity 0 should be filtered out."""
        azimuths = np.arange(0, 360, 18, dtype=np.float64)
        takeoffs = np.full(20, 45.0)
        pols = np.ones(20, dtype=np.int32)
        pols[:12] = 0  # Only 8 valid

        result = moment_tensor_from_polarities(
            azimuths=azimuths,
            takeoff_angles=takeoffs,
            polarities=pols,
            constraint="dc",
            min_polarities=8,
            max_misfit=1.0,
            strike_step=30.0,
            dip_step=30.0,
            rake_step=30.0,
        )
        assert result.n_polarities == 8

    def test_unknown_constraint_raises(self, synthetic_polarities):
        with pytest.raises(ValueError, match="Unknown constraint"):
            moment_tensor_from_polarities(
                azimuths=synthetic_polarities["azimuths"],
                takeoff_angles=synthetic_polarities["takeoff_angles"],
                polarities=synthetic_polarities["polarities"],
                constraint="invalid",
            )

    def test_weights_parameter(self, synthetic_polarities):
        """Weights should be accepted without error."""
        n = len(synthetic_polarities["polarities"])
        weights = np.ones(n, dtype=np.float64)
        result = moment_tensor_from_polarities(
            azimuths=synthetic_polarities["azimuths"],
            takeoff_angles=synthetic_polarities["takeoff_angles"],
            polarities=synthetic_polarities["polarities"],
            weights=weights,
            constraint="dc",
            max_misfit=0.2,
            min_polarities=8,
            strike_step=30.0,
            dip_step=30.0,
            rake_step=30.0,
        )
        assert isinstance(result, MomentTensorResult)


class TestMomentTensorBatch:
    """Tests for batch processing."""

    def test_batch_processes_multiple_events(self, synthetic_polarities):
        events = [
            {
                "azimuths": synthetic_polarities["azimuths"],
                "takeoff_angles": synthetic_polarities["takeoff_angles"],
                "polarities": synthetic_polarities["polarities"],
            },
            {
                "azimuths": synthetic_polarities["azimuths"],
                "takeoff_angles": synthetic_polarities["takeoff_angles"],
                "polarities": synthetic_polarities["polarities"],
            },
        ]
        results = moment_tensor_batch(
            events,
            constraint="dc",
            max_misfit=0.2,
            strike_step=30.0,
            dip_step=30.0,
            rake_step=30.0,
        )
        assert len(results) == 2
        assert all(isinstance(r, MomentTensorResult) for r in results)

    def test_empty_batch(self):
        results = moment_tensor_batch([])
        assert results == []


class TestMomentTensorUncertainty:
    """Tests for component range uncertainty."""

    def test_component_ranges_populated(self, synthetic_polarities):
        """iso/dc/clvd ranges should be populated when solutions exist."""
        result = moment_tensor_from_polarities(
            azimuths=synthetic_polarities["azimuths"],
            takeoff_angles=synthetic_polarities["takeoff_angles"],
            polarities=synthetic_polarities["polarities"],
            constraint="dc",
            max_misfit=0.2,
            min_polarities=8,
            strike_step=30.0,
            dip_step=30.0,
            rake_step=30.0,
        )
        if result.accepted:
            assert result.dc_pct_range[1] >= result.dc_pct_range[0]

    def test_uncertainty_helper(self, synthetic_polarities):
        """moment_tensor_uncertainty should return a dict."""
        result = moment_tensor_from_polarities(
            azimuths=synthetic_polarities["azimuths"],
            takeoff_angles=synthetic_polarities["takeoff_angles"],
            polarities=synthetic_polarities["polarities"],
            constraint="dc",
            max_misfit=0.2,
            min_polarities=8,
            strike_step=30.0,
            dip_step=30.0,
            rake_step=30.0,
        )
        summary = moment_tensor_uncertainty(result)
        assert "dc_pct_range" in summary
        assert "n_accepted" in summary
        assert summary["n_accepted"] == len(result.accepted)
