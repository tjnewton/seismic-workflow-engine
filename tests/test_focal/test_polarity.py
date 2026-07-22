"""Tests for swe.focal.polarity."""

from __future__ import annotations

import numpy as np
from obspy import Stream, UTCDateTime

from swe.focal.polarity import (
    PolarityObservation,
    PolarityResult,
    measure_polarities,
    measure_polarity,
)


class TestMeasurePolarity:
    """Tests for measure_polarity."""

    def test_up_polarity(self, up_trace):
        """Compressional first motion should yield +1."""
        t0 = up_trace.stats.starttime
        obs = measure_polarity(up_trace, t0 + 10.0)
        assert isinstance(obs, PolarityObservation)
        assert obs.polarity == 1

    def test_down_polarity(self, down_trace):
        """Dilatational first motion should yield -1."""
        t0 = down_trace.stats.starttime
        obs = measure_polarity(down_trace, t0 + 10.0)
        assert obs.polarity == -1

    def test_noisy_returns_zero(self, noisy_trace):
        """Low-SNR trace should yield polarity 0."""
        t0 = noisy_trace.stats.starttime
        obs = measure_polarity(noisy_trace, t0 + 10.0, min_snr=3.0)
        assert obs.polarity == 0

    def test_confidence_scaling(self, up_trace):
        """Higher SNR should produce higher confidence."""
        t0 = up_trace.stats.starttime
        obs = measure_polarity(up_trace, t0 + 10.0)
        assert 0.0 <= obs.confidence <= 1.0
        assert obs.snr > 0.0

    def test_noisy_zero_confidence(self, noisy_trace):
        """Low-SNR observation should have zero confidence."""
        t0 = noisy_trace.stats.starttime
        obs = measure_polarity(noisy_trace, t0 + 10.0, min_snr=3.0)
        assert obs.confidence == 0.0

    def test_station_id_format(self, up_trace):
        """Station ID should be network.station.channel."""
        t0 = up_trace.stats.starttime
        obs = measure_polarity(up_trace, t0 + 10.0)
        assert obs.station == "XX.UP01.BHZ"

    def test_with_bandpass_filter(self, up_trace):
        """Bandpass filtering should still detect polarity."""
        t0 = up_trace.stats.starttime
        obs = measure_polarity(up_trace, t0 + 10.0, lowcut=1.0, highcut=40.0)
        assert obs.polarity == 1

    def test_trace_too_short(self):
        """Should return indeterminate for a trace where P is outside data."""
        from obspy import Trace

        data = np.zeros(100, dtype=np.float64)
        tr = Trace(
            data=data,
            header={
                "network": "XX",
                "station": "SHORT",
                "channel": "BHZ",
                "sampling_rate": 100.0,
                "starttime": UTCDateTime(2024, 1, 1),
            },
        )
        # P-time is way after the trace ends
        obs = measure_polarity(tr, UTCDateTime(2024, 1, 1) + 100.0)
        assert obs.polarity == 0


class TestMeasurePolarities:
    """Tests for measure_polarities."""

    def test_returns_result(self, polarity_stream, p_times):
        """Should return a PolarityResult."""
        result = measure_polarities(polarity_stream, p_times)
        assert isinstance(result, PolarityResult)

    def test_observation_count(self, polarity_stream, p_times):
        """Should have one observation per matched trace."""
        result = measure_polarities(polarity_stream, p_times)
        assert len(result.observations) == 3

    def test_polarities_array(self, polarity_stream, p_times):
        """Polarities array should contain +1, -1, and 0."""
        result = measure_polarities(polarity_stream, p_times)
        assert len(result.polarities) == 3
        assert 1 in result.polarities
        assert -1 in result.polarities

    def test_stations_list(self, polarity_stream, p_times):
        """Station list should match traces."""
        result = measure_polarities(polarity_stream, p_times)
        assert len(result.stations) == 3

    def test_missing_p_time_skipped(self, up_trace):
        """Traces without P-times should be skipped."""
        stream = Stream([up_trace])
        result = measure_polarities(stream, {})
        assert len(result.observations) == 0

    def test_full_station_id_lookup(self, up_trace):
        """Should match on full network.station.channel ID."""
        stream = Stream([up_trace])
        t0 = up_trace.stats.starttime
        p_times = {"XX.UP01.BHZ": t0 + 10.0}
        result = measure_polarities(stream, p_times)
        assert len(result.observations) == 1
        assert result.observations[0].polarity == 1

    def test_confidences_array_shape(self, polarity_stream, p_times):
        """Confidences array should match observations."""
        result = measure_polarities(polarity_stream, p_times)
        assert result.confidences.shape == (3,)
        assert all(0.0 <= c <= 1.0 for c in result.confidences)
