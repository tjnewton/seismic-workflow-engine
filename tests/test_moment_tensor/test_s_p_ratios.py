"""Tests for swe.moment_tensor.moment_tensor_from_s_p_ratios."""

from __future__ import annotations

import numpy as np
import pytest

from swe.moment_tensor.amplitude_inversion import (
    AmplitudeMTResult,
    moment_tensor_from_s_p_ratios,
)
from swe.moment_tensor.radiation import (
    radiation_pattern_p,
    radiation_pattern_sh,
    radiation_pattern_sv,
)
from swe.moment_tensor.tensor import strike_dip_rake_to_tensor


def _synthetic_s_p(strike, dip, rake, *, n=24, seed=1):
    """Total S/P ratios for a known mechanism at a station distribution."""
    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 360, n)
    to = rng.uniform(20, 160, n)
    mt = strike_dip_rake_to_tensor(strike, dip, rake, m0=1.0)
    p = np.abs(radiation_pattern_p(mt, az, to))
    sv = radiation_pattern_sv(mt, az, to)
    sh = radiation_pattern_sh(mt, az, to)
    s = np.sqrt(sv**2 + sh**2)
    safe_p = np.where(p > 1e-6, p, 1e-6)
    return az, to, s / safe_p, mt


class TestMomentTensorFromSPRatios:
    def test_recovers_strike_slip_mechanism(self):
        """Synthetic recovery: a strike-slip input should be well fit."""
        az, to, ratios, true_mt = _synthetic_s_p(315.0, 80.0, 170.0)
        result = moment_tensor_from_s_p_ratios(
            az, to, ratios,
            constraint="dc", max_misfit=0.5, min_observations=6,
            strike_step=15.0, dip_step=15.0, rake_step=15.0,
        )
        assert isinstance(result, AmplitudeMTResult)
        assert result.best is not None
        # The best solution should reproduce the observed S/P pattern well.
        assert result.variance_reduction > 0.5
        # And its predicted ratios should correlate with the observed ones.
        p = np.abs(radiation_pattern_p(result.best, az, to))
        sv = radiation_pattern_sv(result.best, az, to)
        sh = radiation_pattern_sh(result.best, az, to)
        pred = np.sqrt(sv**2 + sh**2) / np.where(p > 1e-6, p, 1e-6)
        # log-space correlation (S/P spans orders of magnitude near nodes)
        finite = np.isfinite(np.log10(pred)) & np.isfinite(np.log10(ratios))
        corr = np.corrcoef(np.log10(pred[finite]), np.log10(ratios[finite]))[0, 1]
        assert corr > 0.7

    def test_too_few_observations_returns_none(self):
        az, to, ratios, _ = _synthetic_s_p(315.0, 80.0, 170.0, n=4)
        result = moment_tensor_from_s_p_ratios(
            az, to, ratios, min_observations=6, strike_step=30, dip_step=30, rake_step=30
        )
        assert result.best is None
        assert result.n_observations == 4

    def test_drops_nonpositive_and_nonfinite(self):
        az, to, ratios, _ = _synthetic_s_p(315.0, 80.0, 170.0, n=10)
        ratios = ratios.copy()
        ratios[0] = np.nan
        ratios[1] = -1.0
        ratios[2] = 0.0
        result = moment_tensor_from_s_p_ratios(
            az, to, ratios, min_observations=6,
            strike_step=30, dip_step=30, rake_step=30,
        )
        assert result.n_observations == 7  # 10 - 3 dropped

    def test_invalid_constraint(self):
        az, to, ratios, _ = _synthetic_s_p(315.0, 80.0, 170.0)
        with pytest.raises(ValueError, match="Unknown constraint"):
            moment_tensor_from_s_p_ratios(az, to, ratios, constraint="bogus")

    def test_variance_reduction_uses_mean_removed_denominator(self):
        """VR is defined against variance about the mean (not raw mean-square)."""
        az, to, ratios, _ = _synthetic_s_p(315.0, 80.0, 170.0)
        result = moment_tensor_from_s_p_ratios(
            az, to, ratios, max_misfit=2.0,
            strike_step=20, dip_step=20, rake_step=20,
        )
        assert 0.0 <= result.variance_reduction <= 1.0
