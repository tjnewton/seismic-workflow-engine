"""Tests for swe.io.fdsn."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from obspy.clients.fdsn.header import FDSNException

from swe.io.fdsn import get_response_files, get_waveforms


def _make_day_stream(network, station, channel, date_str, sr=100.0, n=8640000):
    """Create a synthetic day-long Stream."""
    start = UTCDateTime(f"{date_str}T00:00:00.0Z")
    data = np.random.default_rng(42).normal(0, 1, n).astype(np.float32)
    tr = Trace(
        data=data,
        header={
            "network": network,
            "station": station,
            "channel": channel,
            "sampling_rate": sr,
            "starttime": start,
        },
    )
    return Stream([tr])


class TestGetWaveforms:
    @patch("swe.io.fdsn.Client")
    def test_writes_daily_file(self, MockClient, tmp_path):
        """Should write a daily MSEED file."""
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        # Return a short synthetic stream (1 day)
        start = UTCDateTime("2020-01-01T00:00:00.0Z")
        end = UTCDateTime("2020-01-01T23:59:59.9Z")
        data = np.zeros(100, dtype=np.float32)
        tr = Trace(
            data=data,
            header={
                "network": "XX",
                "station": "TEST",
                "channel": "BHZ",
                "sampling_rate": 100.0,
                "starttime": start,
            },
        )
        mock_client.get_waveforms.return_value = Stream([tr])

        result = get_waveforms(
            "XX",
            "TEST",
            "BHZ",
            start,
            end,
            output_dir=tmp_path,
            client="IRIS",
        )
        assert len(result) == 1
        assert result[0].exists()
        assert "XX.TEST.BHZ" in result[0].name

    @patch("swe.io.fdsn.Client")
    def test_handles_no_data(self, MockClient, tmp_path):
        """Should handle missing data gracefully."""
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.get_waveforms.side_effect = FDSNException("No data")

        start = UTCDateTime("2020-01-01T00:00:00.0Z")
        end = UTCDateTime("2020-01-01T23:59:59.9Z")

        result = get_waveforms(
            "XX",
            "TEST",
            "BHZ",
            start,
            end,
            output_dir=tmp_path,
        )
        assert result == []

    @patch("swe.io.fdsn.Client")
    def test_typeerror_propagates(self, MockClient, tmp_path):
        """TypeError should not be caught - it propagates."""
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.get_waveforms.side_effect = TypeError("bad arg")

        start = UTCDateTime("2020-01-01T00:00:00.0Z")
        end = UTCDateTime("2020-01-01T23:59:59.9Z")

        with pytest.raises(TypeError, match="bad arg"):
            get_waveforms(
                "XX",
                "TEST",
                "BHZ",
                start,
                end,
                output_dir=tmp_path,
            )


class TestGetResponseFiles:
    @patch("swe.io.fdsn.Client")
    def test_writes_stationxml(self, MockClient, tmp_path):
        """Should write StationXML response files."""
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        from obspy import Inventory

        mock_inv = MagicMock(spec=Inventory)
        mock_client.get_stations.return_value = mock_inv

        start = UTCDateTime("2020-01-01T00:00:00.0Z")
        end = UTCDateTime("2020-01-01T23:59:59.9Z")

        result = get_response_files(
            "XX",
            ["STA1", "STA2"],
            "BHZ",
            start,
            end,
            output_dir=tmp_path,
        )
        assert len(result) == 2
        assert mock_inv.write.call_count == 2

    @patch("swe.io.fdsn.Client")
    def test_handles_failure(self, MockClient, tmp_path):
        """Should handle failed response downloads."""
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.get_stations.side_effect = FDSNException("No response")

        start = UTCDateTime("2020-01-01T00:00:00.0Z")
        end = UTCDateTime("2020-01-01T23:59:59.9Z")

        result = get_response_files(
            "XX",
            ["FAIL"],
            "BHZ",
            start,
            end,
            output_dir=tmp_path,
        )
        assert result == []

    @patch("swe.io.fdsn.Client")
    def test_typeerror_propagates(self, MockClient, tmp_path):
        """TypeError should not be caught - it propagates."""
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.get_stations.side_effect = TypeError("bad arg")

        start = UTCDateTime("2020-01-01T00:00:00.0Z")
        end = UTCDateTime("2020-01-01T23:59:59.9Z")

        with pytest.raises(TypeError, match="bad arg"):
            get_response_files(
                "XX",
                ["FAIL"],
                "BHZ",
                start,
                end,
                output_dir=tmp_path,
            )
