"""Tests for swe.moment_tensor.amplitude_inversion."""

from __future__ import annotations

import numpy as np
import pytest

from swe.moment_tensor.amplitude_inversion import (
    AmplitudeMTResult,
    moment_tensor_from_amplitude_ratios,
    moment_tensor_from_amplitudes,
)
from swe.moment_tensor.radiation import (
    radiation_pattern_p,
    radiation_pattern_sh,
    radiation_pattern_sv,
)
from swe.moment_tensor.tensor import strike_dip_rake_to_tensor


@pytest.fixture()
def synthetic_amplitudes():
    """Synthetic amplitude observations from a known focal mechanism.

    Strike=0, dip=45, rake=90 (pure thrust).
    """
    rng = np.random.default_rng(42)
    n_stations = 20
    azimuths = rng.uniform(0, 360, n_stations)
    takeoff_angles = rng.uniform(30, 150, n_stations)

    mt = strike_dip_rake_to_tensor(0.0, 45.0, 90.0, m0=1.0)

    amp_p = radiation_pattern_p(mt, azimuths, takeoff_angles)
    amp_sv = radiation_pattern_sv(mt, azimuths, takeoff_angles)
    amp_sh = radiation_pattern_sh(mt, azimuths, takeoff_angles)

    return {
        "azimuths": azimuths,
        "takeoff_angles": takeoff_angles,
        "amplitudes_p": amp_p,
        "amplitudes_sv": amp_sv,
        "amplitudes_sh": amp_sh,
        "true_mt": mt,
    }


class TestMomentTensorFromAmplitudes:
    def test_dc_constraint_finds_solution(self, synthetic_amplitudes):
        """DC-constrained search should find at least one solution."""
        result = moment_tensor_from_amplitudes(
            azimuths=synthetic_amplitudes["azimuths"],
            takeoff_angles=synthetic_amplitudes["takeoff_angles"],
            amplitudes_p=synthetic_amplitudes["amplitudes_p"],
            constraint="dc",
            max_misfit=0.5,
            min_observations=6,
            strike_step=30.0,
            dip_step=30.0,
            rake_step=30.0,
        )
        assert isinstance(result, AmplitudeMTResult)
        assert result.n_observations == 20
        assert result.best is not None
        assert len(result.accepted) >= 1

    def test_with_sv_amplitudes(self, synthetic_amplitudes):
        """Including SV amplitudes should still work."""
        result = moment_tensor_from_amplitudes(
            azimuths=synthetic_amplitudes["azimuths"],
            takeoff_angles=synthetic_amplitudes["takeoff_angles"],
            amplitudes_p=synthetic_amplitudes["amplitudes_p"],
            amplitudes_sv=synthetic_amplitudes["amplitudes_sv"],
            constraint="dc",
            max_misfit=1.0,
            strike_step=30.0,
            dip_step=30.0,
            rake_step=30.0,
        )
        assert isinstance(result, AmplitudeMTResult)

    def test_with_all_amplitudes(self, synthetic_amplitudes):
        """Including P, SV, and SH amplitudes."""
        result = moment_tensor_from_amplitudes(
            azimuths=synthetic_amplitudes["azimuths"],
            takeoff_angles=synthetic_amplitudes["takeoff_angles"],
            amplitudes_p=synthetic_amplitudes["amplitudes_p"],
            amplitudes_sv=synthetic_amplitudes["amplitudes_sv"],
            amplitudes_sh=synthetic_amplitudes["amplitudes_sh"],
            constraint="dc",
            max_misfit=1.0,
            strike_step=30.0,
            dip_step=30.0,
            rake_step=30.0,
        )
        assert isinstance(result, AmplitudeMTResult)

    def test_too_few_observations(self):
        """Should return None with too few observations."""
        result = moment_tensor_from_amplitudes(
            azimuths=np.array([0, 90, 180]),
            takeoff_angles=np.array([45, 45, 45]),
            amplitudes_p=np.array([1.0, -0.5, 0.3]),
            constraint="dc",
            min_observations=8,
        )
        assert result.best is None
        assert result.accepted == []

    def test_nan_amplitudes_excluded(self):
        """NaN amplitudes should be filtered out."""
        azimuths = np.arange(0, 360, 18, dtype=np.float64)
        takeoffs = np.full(20, 45.0)
        amps = np.ones(20)
        amps[:12] = np.nan  # Only 8 valid

        result = moment_tensor_from_amplitudes(
            azimuths=azimuths,
            takeoff_angles=takeoffs,
            amplitudes_p=amps,
            constraint="dc",
            min_observations=8,
            max_misfit=2.0,
            strike_step=30.0,
            dip_step=30.0,
            rake_step=30.0,
        )
        assert result.n_observations == 8

    def test_unknown_constraint_raises(self, synthetic_amplitudes):
        with pytest.raises(ValueError, match="Unknown constraint"):
            moment_tensor_from_amplitudes(
                azimuths=synthetic_amplitudes["azimuths"],
                takeoff_angles=synthetic_amplitudes["takeoff_angles"],
                amplitudes_p=synthetic_amplitudes["amplitudes_p"],
                constraint="invalid",
            )

    def test_variance_reduction(self, synthetic_amplitudes):
        """Best solution should have positive variance reduction."""
        result = moment_tensor_from_amplitudes(
            azimuths=synthetic_amplitudes["azimuths"],
            takeoff_angles=synthetic_amplitudes["takeoff_angles"],
            amplitudes_p=synthetic_amplitudes["amplitudes_p"],
            constraint="dc",
            max_misfit=0.5,
            strike_step=30.0,
            dip_step=30.0,
            rake_step=30.0,
        )
        if result.best is not None:
            assert result.variance_reduction >= 0.0

    def test_component_ranges(self, synthetic_amplitudes):
        """DC ranges should be populated when solutions exist."""
        result = moment_tensor_from_amplitudes(
            azimuths=synthetic_amplitudes["azimuths"],
            takeoff_angles=synthetic_amplitudes["takeoff_angles"],
            amplitudes_p=synthetic_amplitudes["amplitudes_p"],
            constraint="dc",
            max_misfit=0.5,
            strike_step=30.0,
            dip_step=30.0,
            rake_step=30.0,
        )
        if result.accepted:
            assert result.dc_pct_range[1] >= result.dc_pct_range[0]

    def test_weights_parameter(self, synthetic_amplitudes):
        """Weights should be accepted without error."""
        n = len(synthetic_amplitudes["amplitudes_p"])
        result = moment_tensor_from_amplitudes(
            azimuths=synthetic_amplitudes["azimuths"],
            takeoff_angles=synthetic_amplitudes["takeoff_angles"],
            amplitudes_p=synthetic_amplitudes["amplitudes_p"],
            weights=np.ones(n),
            constraint="dc",
            max_misfit=1.0,
            strike_step=30.0,
            dip_step=30.0,
            rake_step=30.0,
        )
        assert isinstance(result, AmplitudeMTResult)


class TestMomentTensorFromAmplitudeRatios:
    def test_dc_constraint_basic(self, synthetic_amplitudes):
        """Ratio-based inversion should produce results."""
        amp_p = synthetic_amplitudes["amplitudes_p"]
        amp_sv = synthetic_amplitudes["amplitudes_sv"]

        # Compute log10(SV/P) ratios
        safe_p = np.where(np.abs(amp_p) > 1e-30, amp_p, 1e-30)
        sv_p_ratios = np.log10(np.abs(amp_sv / safe_p) + 1e-30)

        result = moment_tensor_from_amplitude_ratios(
            azimuths=synthetic_amplitudes["azimuths"],
            takeoff_angles=synthetic_amplitudes["takeoff_angles"],
            sv_p_ratios=sv_p_ratios,
            constraint="dc",
            max_misfit=5.0,
            min_observations=6,
            strike_step=30.0,
            dip_step=30.0,
            rake_step=30.0,
        )
        assert isinstance(result, AmplitudeMTResult)
        assert result.n_observations == 20

    def test_too_few_observations(self):
        result = moment_tensor_from_amplitude_ratios(
            azimuths=np.array([0.0, 90.0]),
            takeoff_angles=np.array([45.0, 45.0]),
            sv_p_ratios=np.array([0.5, -0.3]),
            min_observations=6,
        )
        assert result.best is None

    def test_with_sh_ratios(self, synthetic_amplitudes):
        """Including SH/P ratios should work."""
        amp_p = synthetic_amplitudes["amplitudes_p"]
        amp_sv = synthetic_amplitudes["amplitudes_sv"]
        amp_sh = synthetic_amplitudes["amplitudes_sh"]

        safe_p = np.where(np.abs(amp_p) > 1e-30, amp_p, 1e-30)
        sv_p = np.log10(np.abs(amp_sv / safe_p) + 1e-30)
        sh_p = np.log10(np.abs(amp_sh / safe_p) + 1e-30)

        result = moment_tensor_from_amplitude_ratios(
            azimuths=synthetic_amplitudes["azimuths"],
            takeoff_angles=synthetic_amplitudes["takeoff_angles"],
            sv_p_ratios=sv_p,
            sh_p_ratios=sh_p,
            constraint="dc",
            max_misfit=10.0,
            strike_step=30.0,
            dip_step=30.0,
            rake_step=30.0,
        )
        assert isinstance(result, AmplitudeMTResult)

    def test_unknown_constraint_raises(self):
        with pytest.raises(ValueError, match="Unknown constraint"):
            moment_tensor_from_amplitude_ratios(
                azimuths=np.arange(0, 360, 18, dtype=np.float64),
                takeoff_angles=np.full(20, 45.0),
                sv_p_ratios=np.zeros(20),
                constraint="invalid",
            )
