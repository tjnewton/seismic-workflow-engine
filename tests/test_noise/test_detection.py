"""Tests for swe.noise.detection."""

from pathlib import Path

import numpy as np
import pytest
from obspy import Stream, Trace
from obspy.core.utcdatetime import UTCDateTime

from swe.noise.detection import (
    _default_station_parser,
    find_noise_windows,
)


class TestDefaultStationParser:
    def test_double_dot_format(self):
        """Double-dot format: NETWORK.STATION..CHANNEL.DATE.ms"""
        path = Path("5A.27..DP1.2018-03-13T00.00.00.ms")
        station, channel = _default_station_parser(path)
        assert station == "27"
        assert channel == "DP1"

    def test_double_dot_nested_path(self):
        """Double-dot format with a directory prefix."""
        path = Path("/data/seismic/UW.UGAP3..EHZ.2018-04-01.ms")
        station, channel = _default_station_parser(path)
        assert station == "UGAP3"
        assert channel == "EHZ"

    def test_simple_format(self):
        """Simple format: NETWORK.STATION.CHANNEL.DATE.ms (from swe.io)."""
        path = Path("CI.WBS.BHZ.2019-07-06.ms")
        station, channel = _default_station_parser(path)
        assert station == "WBS"
        assert channel == "BHZ"

    def test_simple_format_nested_path(self):
        """Simple format with a directory prefix."""
        path = Path("/data/ridgecrest/CI.JRC2.BHZ.2019-07-06.ms")
        station, channel = _default_station_parser(path)
        assert station == "JRC2"
        assert channel == "BHZ"


class TestFindNoiseWindows:
    def test_with_synthetic_mseed(self, tmp_path):
        """Create a synthetic MSEED file and detect noise windows."""
        # Create a trace with mostly quiet data and a few spikes
        sampling_rate = 100.0
        duration = 60  # seconds
        n_samples = int(sampling_rate * duration)

        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.001, n_samples).astype(np.float64)

        # Add a large spike at 30 seconds to create a non-noise region
        spike_start = int(30 * sampling_rate)
        data[spike_start : spike_start + 100] = 10.0

        tr = Trace(data=data)
        tr.stats.sampling_rate = sampling_rate
        tr.stats.network = "XX"
        tr.stats.station = "TEST"
        tr.stats.channel = "BHZ"
        tr.stats.starttime = UTCDateTime("2024-01-01T00:00:00")

        filepath = tmp_path / "XX.TEST..BHZ.2024-01-01.ms"
        st = Stream([tr])
        st.write(str(filepath), format="MSEED")

        # Detect noise windows
        start = UTCDateTime("2024-01-01T00:00:05")
        end = UTCDateTime("2024-01-01T00:00:55")
        # sta=10, lta=50, threshold=3, window_length=100
        params = (10, 50, 3.0, 100)

        windows, times = find_noise_windows(
            [str(filepath)],
            params,
            [(start, end)],
        )

        # Should find some noise windows (quiet periods)
        assert isinstance(windows, np.ndarray)
        assert windows.ndim == 2
        assert len(times) == windows.shape[0]

        # Each time entry should be a 4-tuple
        if len(times) > 0:
            assert len(times[0]) == 4
            assert times[0][2] == "TEST"
            assert times[0][3] == "BHZ"

    def test_nonexistent_file(self, tmp_path, caplog):
        """Non-existent files should be skipped with a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            windows, times = find_noise_windows(
                [str(tmp_path / "nonexistent.ms")],
                (10, 50, 3.0, 100),
                [
                    (
                        UTCDateTime("2024-01-01"),
                        UTCDateTime("2024-01-01T00:01:00"),
                    )
                ],
            )

        assert windows.shape[0] == 0
        assert len(times) == 0

    def test_response_dir_required(self, tmp_path):
        """Should raise ValueError when remove_response=True without dir."""
        # Create a minimal MSEED file so the loop body executes
        tr = Trace(data=np.zeros(1000, dtype=np.float64))
        tr.stats.sampling_rate = 100.0
        tr.stats.network = "XX"
        tr.stats.station = "TEST"
        tr.stats.channel = "BHZ"
        tr.stats.starttime = UTCDateTime("2024-01-01T00:00:00")
        filepath = tmp_path / "XX.TEST..BHZ.2024-01-01.ms"
        Stream([tr]).write(str(filepath), format="MSEED")

        with pytest.raises(ValueError, match="response_dir"):
            find_noise_windows(
                [str(filepath)],
                (10, 50, 3.0, 100),
                [(UTCDateTime("2024-01-01"), UTCDateTime("2024-01-01T00:00:09"))],
                remove_response=True,
            )
