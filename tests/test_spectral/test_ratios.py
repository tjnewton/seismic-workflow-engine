"""Tests for swe.spectral.ratios."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from swe.spectral.ratios import (
    HVSRResult,
    SSRResult,
    compute_hvsr,
    compute_ssr,
)


class TestComputeHVSR:
    """Tests for compute_hvsr."""

    def test_returns_hvsr_result(self, three_component_stream):
        result = compute_hvsr(
            three_component_stream,
            window_length=20.0,
            freq_min=0.5,
            freq_max=40.0,
        )
        assert isinstance(result, HVSRResult)

    def test_peak_near_5hz(self, three_component_stream):
        """HVSR peak should be near the 5 Hz signal injected on horizontals."""
        result = compute_hvsr(
            three_component_stream,
            window_length=20.0,
            freq_min=1.0,
            freq_max=40.0,
        )
        assert 3.0 < result.peak_frequency < 8.0

    def test_peak_amplitude_above_one(self, three_component_stream):
        result = compute_hvsr(
            three_component_stream,
            window_length=20.0,
            freq_min=1.0,
            freq_max=40.0,
        )
        assert result.peak_amplitude > 1.0

    def test_n_windows_positive(self, three_component_stream):
        result = compute_hvsr(
            three_component_stream,
            window_length=20.0,
        )
        assert result.n_windows > 0

    def test_per_window_shape(self, three_component_stream):
        result = compute_hvsr(
            three_component_stream,
            window_length=20.0,
        )
        assert result.hvsr_per_window.shape[0] == result.n_windows
        assert result.hvsr_per_window.shape[1] == len(result.frequencies)

    def test_station_name(self, three_component_stream):
        result = compute_hvsr(
            three_component_stream,
            window_length=20.0,
        )
        assert result.station == "XX.STA1"

    def test_missing_vertical_raises(self):
        """Stream without vertical component should raise ValueError."""
        st = Stream()
        rng = np.random.default_rng(42)
        for ch in ("BHN", "BHE"):
            header = {
                "network": "XX",
                "station": "STA",
                "channel": ch,
                "sampling_rate": 100.0,
                "npts": 5000,
                "starttime": UTCDateTime(2024, 1, 1),
            }
            st += Trace(data=rng.standard_normal(5000), header=header)
        with pytest.raises(ValueError, match="vertical"):
            compute_hvsr(st)

    def test_missing_horizontals_raises(self):
        """Stream with < 2 horizontal components should raise ValueError."""
        st = Stream()
        rng = np.random.default_rng(42)
        header = {
            "network": "XX",
            "station": "STA",
            "channel": "BHZ",
            "sampling_rate": 100.0,
            "npts": 5000,
            "starttime": UTCDateTime(2024, 1, 1),
        }
        st += Trace(data=rng.standard_normal(5000), header=header)
        with pytest.raises(ValueError, match="horizontal"):
            compute_hvsr(st)

    def test_short_trace_raises(self, three_component_stream):
        """Window longer than trace should raise ValueError."""
        with pytest.raises(ValueError, match="too short"):
            compute_hvsr(
                three_component_stream,
                window_length=500.0,
            )

    def test_quadratic_mean_merge(self, three_component_stream):
        result = compute_hvsr(
            three_component_stream,
            window_length=20.0,
            merge_method="quadratic_mean",
        )
        assert isinstance(result, HVSRResult)
        assert result.n_windows > 0


class TestComputeSSR:
    """Tests for compute_ssr."""

    def test_returns_ssr_result(self, event_trace_pair):
        target, reference = event_trace_pair
        result = compute_ssr(
            [target],
            [reference],
            freq_min=0.5,
            freq_max=40.0,
        )
        assert isinstance(result, SSRResult)

    def test_amplification_detected(self, event_trace_pair):
        """SSR should show amplification where signal was injected."""
        target, reference = event_trace_pair
        result = compute_ssr(
            [target],
            [reference],
            freq_min=1.0,
            freq_max=40.0,
        )
        # Find ratio near 4 Hz
        idx = np.argmin(np.abs(result.frequencies - 4.0))
        assert result.ssr_mean[idx] > 1.5

    def test_multi_event(self, event_trace_pair):
        target, reference = event_trace_pair
        result = compute_ssr(
            [target, target],
            [reference, reference],
        )
        assert result.n_events == 2
        assert result.ssr_per_event.shape[0] == 2

    def test_station_names(self, event_trace_pair):
        target, reference = event_trace_pair
        result = compute_ssr([target], [reference])
        assert result.target_station == "XX.TGT1"
        assert result.reference_station == "XX.REF1"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            compute_ssr([], [])

    def test_mismatched_lengths_raises(self, event_trace_pair):
        target, reference = event_trace_pair
        with pytest.raises(ValueError, match="same length"):
            compute_ssr([target, target], [reference])
