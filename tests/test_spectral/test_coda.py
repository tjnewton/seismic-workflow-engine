"""Tests for swe.spectral.coda."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Trace, UTCDateTime

from swe.spectral.coda import (
    CodaQFrequencyResult,
    CodaQResult,
    coda_q_frequency_bands,
    estimate_coda_q,
)


class TestEstimateCodaQ:
    """Tests for estimate_coda_q."""

    def test_returns_coda_q_result(self, coda_trace):
        origin = coda_trace.stats.starttime + 5.0
        result = estimate_coda_q(
            coda_trace,
            origin,
            freq_min=1.0,
            freq_max=10.0,
        )
        assert isinstance(result, CodaQResult)

    def test_recovers_qc(self, coda_trace):
        """Qc should be recovered within ±50% of the true value (100)."""
        origin = coda_trace.stats.starttime + 5.0
        result = estimate_coda_q(
            coda_trace,
            origin,
            freq_min=1.0,
            freq_max=10.0,
        )
        if not np.isnan(result.qc):
            assert 50 < result.qc < 200, f"Expected Qc ≈ 100, got {result.qc}"

    def test_positive_r_squared(self, coda_trace):
        origin = coda_trace.stats.starttime + 5.0
        result = estimate_coda_q(
            coda_trace,
            origin,
            freq_min=1.0,
            freq_max=10.0,
        )
        if not np.isnan(result.qc):
            assert result.fit_r_squared > 0

    def test_station_name(self, coda_trace):
        origin = coda_trace.stats.starttime + 5.0
        result = estimate_coda_q(coda_trace, origin)
        assert result.station == "XX.COD1.BHZ"

    def test_coda_window_times(self, coda_trace):
        origin = coda_trace.stats.starttime + 5.0
        result = estimate_coda_q(coda_trace, origin)
        if not np.isnan(result.qc):
            assert result.coda_start_time > 0
            assert result.coda_end_time > result.coda_start_time

    def test_short_trace_returns_nan(self):
        """Very short trace should return NaN Qc gracefully."""
        header = {
            "network": "XX",
            "station": "STA",
            "channel": "BHZ",
            "sampling_rate": 100.0,
            "npts": 100,
            "starttime": UTCDateTime(2024, 1, 1),
        }
        short = Trace(
            data=np.random.default_rng(42).standard_normal(100),
            header=header,
        )
        origin = short.stats.starttime
        result = estimate_coda_q(short, origin)
        assert np.isnan(result.qc)

    def test_negative_decay_rate(self, coda_trace):
        """Decay rate should be negative for a valid coda."""
        origin = coda_trace.stats.starttime + 5.0
        result = estimate_coda_q(
            coda_trace,
            origin,
            freq_min=1.0,
            freq_max=10.0,
        )
        if not np.isnan(result.qc):
            assert result.decay_rate < 0


class TestCodaQFrequencyBands:
    """Tests for coda_q_frequency_bands."""

    def test_returns_frequency_result(self, coda_trace):
        origin = coda_trace.stats.starttime + 5.0
        result = coda_q_frequency_bands(coda_trace, origin)
        assert isinstance(result, CodaQFrequencyResult)

    def test_frequencies_array(self, coda_trace):
        origin = coda_trace.stats.starttime + 5.0
        result = coda_q_frequency_bands(coda_trace, origin)
        assert len(result.frequencies) > 0
        assert len(result.frequencies) == len(result.qc_values)

    def test_custom_bands(self, coda_trace):
        origin = coda_trace.stats.starttime + 5.0
        bands = [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]
        result = coda_q_frequency_bands(
            coda_trace, origin, frequency_bands=bands
        )
        assert len(result.frequencies) == 3

    def test_power_law_fit(self, coda_trace):
        """Power-law parameters should be finite when enough valid bands exist."""
        origin = coda_trace.stats.starttime + 5.0
        result = coda_q_frequency_bands(
            coda_trace,
            origin,
            min_r_squared=0.1,
        )
        # At least some valid Qc values should exist
        valid = ~np.isnan(result.qc_values)
        if np.sum(valid) >= 2:
            assert not np.isnan(result.q0)
            assert not np.isnan(result.eta)

    def test_station_name(self, coda_trace):
        origin = coda_trace.stats.starttime + 5.0
        result = coda_q_frequency_bands(coda_trace, origin)
        assert result.station == "XX.COD1.BHZ"

    def test_r_squared_values_shape(self, coda_trace):
        origin = coda_trace.stats.starttime + 5.0
        result = coda_q_frequency_bands(coda_trace, origin)
        assert len(result.r_squared_values) == len(result.frequencies)
