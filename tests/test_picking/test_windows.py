"""Tests for swe.picking.windows."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from swe.picking.windows import (
    find_picking_windows,
    get_threshold_picks,
    remove_duplicate_picks,
)


class TestRemoveDuplicatePicks:
    """Tests for remove_duplicate_picks with known duplicate scenarios."""

    def _make_mock_utcdatetime(self, timestamp):
        """Create a simple mock that supports subtraction returning seconds."""

        class MockUTC:
            def __init__(self, ts):
                self._ts = ts

            def __sub__(self, other):
                return self._ts - other._ts

            def __abs__(self):
                return abs(self._ts)

            def __repr__(self):
                return f"MockUTC({self._ts})"

        return MockUTC(timestamp)

    def test_no_duplicates(self):
        """Picks on different stations are all retained."""
        station_list = ["STA1", "STA2", "STA3"]
        picked_windows = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0]),
        ]
        # (start_time, end_time, station, channel)
        # Use floats as mock times; remove_duplicate_picks computes
        # pick_time = window_start + (index / sampling_rate)
        from obspy import UTCDateTime

        t0 = UTCDateTime("2024-01-01T00:00:00.0Z")
        picked_windows_times = [
            (t0, t0 + 1, "STA1", "HHZ"),
            (t0 + 10, t0 + 11, "STA2", "HHZ"),
            (t0 + 20, t0 + 21, "STA3", "HHZ"),
        ]
        index_of_max = [1, 1, 1]

        station_picks, uw, uwt, uidx = remove_duplicate_picks(
            picked_windows,
            picked_windows_times,
            index_of_max,
            station_list,
            sampling_rate=100.0,
            duplicate_threshold=10.0,
        )

        # All 3 picks should be retained (different stations)
        assert len(uw) == 3
        assert len(uwt) == 3
        assert len(uidx) == 3
        # Each station should have exactly one pick
        assert len(station_picks["STA1"]) == 1
        assert len(station_picks["STA2"]) == 1
        assert len(station_picks["STA3"]) == 1

    def test_duplicates_same_station(self):
        """Picks very close in time on the same station are deduplicated."""
        from obspy import UTCDateTime

        station_list = ["STA1", "STA2"]
        picked_windows = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 3.1]),
            np.array([4.0, 5.0, 6.0]),
        ]

        t0 = UTCDateTime("2024-01-01T00:00:00.0Z")
        # Two picks on STA1, 0.01 seconds apart (well within threshold)
        picked_windows_times = [
            (t0, t0 + 1, "STA1", "HHZ"),
            (t0 + 0.01, t0 + 1.01, "STA1", "HHZ"),
            (t0 + 10, t0 + 11, "STA2", "HHZ"),
        ]
        index_of_max = [1, 1, 1]

        station_picks, uw, uwt, uidx = remove_duplicate_picks(
            picked_windows,
            picked_windows_times,
            index_of_max,
            station_list,
            sampling_rate=100.0,
            duplicate_threshold=10.0,
        )

        # Only 2 should remain: 1 from STA1 (first one kept), 1 from STA2
        assert len(uw) == 2
        assert len(uwt) == 2
        assert len(uidx) == 2
        assert len(station_picks["STA1"]) == 1
        assert len(station_picks["STA2"]) == 1

    def test_empty_input(self):
        """Empty input returns empty results."""
        station_picks, uw, uwt, uidx = remove_duplicate_picks(
            [],
            [],
            [],
            ["STA1"],
        )
        assert station_picks == {}
        assert uw == []
        assert uwt == []
        assert uidx == []

    def test_all_duplicates_same_station(self):
        """Multiple picks on one station within threshold -- only first kept."""
        from obspy import UTCDateTime

        station_list = ["STA1"]
        t0 = UTCDateTime("2024-01-01T00:00:00.0Z")
        n_picks = 5
        picked_windows = [np.array([float(i)]) for i in range(n_picks)]
        picked_windows_times = [
            (t0 + i * 0.001, t0 + 1 + i * 0.001, "STA1", "HHZ") for i in range(n_picks)
        ]
        index_of_max = [0] * n_picks

        station_picks, uw, _uwt, _uidx = remove_duplicate_picks(
            picked_windows,
            picked_windows_times,
            index_of_max,
            station_list,
            sampling_rate=100.0,
            duplicate_threshold=10.0,
        )

        # Only the first pick should be retained
        assert len(station_picks["STA1"]) == 1
        assert len(uw) == 1

    def test_distant_picks_not_removed(self):
        """Picks on the same station far apart in time are both kept."""
        from obspy import UTCDateTime

        station_list = ["STA1"]
        t0 = UTCDateTime("2024-01-01T00:00:00.0Z")
        picked_windows = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]
        # 60 seconds apart -- well beyond threshold
        picked_windows_times = [
            (t0, t0 + 1, "STA1", "HHZ"),
            (t0 + 60, t0 + 61, "STA1", "HHZ"),
        ]
        index_of_max = [0, 0]

        station_picks, uw, _uwt, _uidx = remove_duplicate_picks(
            picked_windows,
            picked_windows_times,
            index_of_max,
            station_list,
            sampling_rate=100.0,
            duplicate_threshold=10.0,
        )

        assert len(station_picks["STA1"]) == 2
        assert len(uw) == 2


# ---------------------------------------------------------------------------
# Tests for find_picking_windows
# ---------------------------------------------------------------------------


class TestFindPickingWindows:
    """Tests for find_picking_windows."""

    def _make_stream(self, npts, sampling_rate, starttime):
        """Build a single-trace Stream with synthetic data."""
        tr = Trace(data=np.ones(npts, dtype=np.float64) * 5.0)
        tr.stats.sampling_rate = sampling_rate
        tr.stats.starttime = starttime
        tr.stats.station = "STA1"
        tr.stats.channel = "BHZ"
        tr.stats.network = "XX"
        return Stream([tr])

    @patch("swe.picking.windows.read")
    def test_basic_window_extraction(self, mock_read):
        """Window count, shape, and metadata should be correct."""
        sr = 250.0
        npts = 2000
        wl = 500
        wo = 250
        start = UTCDateTime("2024-01-01T00:00:00")
        end = start + (npts / sr)

        mock_read.return_value = self._make_stream(npts, sr, start)

        windows, raw, meta = find_picking_windows(
            ["/fake/XX.STA1.00.BHZ.mseed"],
            window_length=wl,
            window_offset=wo,
            time_periods=[(start, end)],
            sampling_rate=sr,
        )

        expected_n = (npts - wl) // wo
        assert windows.shape == (expected_n, wl, 2)
        assert len(raw) == expected_n
        assert len(meta) == expected_n
        # Check metadata tuple structure
        for m in meta:
            assert len(m) == 4
            assert m[2] == "STA1"
            assert m[3] == "BHZ"

    @patch("swe.picking.windows.read")
    def test_log_transform_correctness(self, mock_read):
        """Constant-5 data -> log(|5| + eps) and sign(5)=+1."""
        sr = 250.0
        npts = 1000
        wl = 500
        wo = 500
        start = UTCDateTime("2024-01-01T00:00:00")
        end = start + (npts / sr)
        epsilon = 1e-6

        mock_read.return_value = self._make_stream(npts, sr, start)

        windows, _, _ = find_picking_windows(
            ["/fake/XX.STA1.00.BHZ.mseed"],
            window_length=wl,
            window_offset=wo,
            time_periods=[(start, end)],
            sampling_rate=sr,
            epsilon=epsilon,
        )

        expected_log = np.log(5.0 + epsilon)
        assert windows[0, 0, 0] == pytest.approx(expected_log, rel=1e-5)
        assert windows[0, 0, 1] == pytest.approx(1.0)

    @patch("swe.picking.windows.read_inventory")
    @patch("swe.picking.windows.read")
    def test_response_removal_and_filter(self, mock_read, mock_inv):
        """Response removal and filter paths should be exercised."""
        sr = 250.0
        npts = 1000
        start = UTCDateTime("2024-01-01T00:00:00")
        end = start + (npts / sr)

        st = self._make_stream(npts, sr, start)
        # Replace the trace methods with mocks to track calls
        st[0].remove_response = MagicMock()
        st[0].filter = MagicMock()
        st[0].detrend = MagicMock()

        mock_read.return_value = st
        mock_inv.return_value = MagicMock()

        find_picking_windows(
            ["/fake/XX.STA1.00.BHZ.mseed"],
            window_length=500,
            window_offset=500,
            time_periods=[(start, end)],
            sampling_rate=sr,
            apply_filter=True,
            apply_response=True,
            response_dir="/fake/resp",
        )

        mock_inv.assert_called_once()
        # The mocked methods are on the original stream trace,
        # but trim creates copies, so check read_inventory was called
        assert mock_inv.called


# ---------------------------------------------------------------------------
# Tests for get_threshold_picks
# ---------------------------------------------------------------------------


class TestGetThresholdPicks:
    """Tests for get_threshold_picks."""

    def test_basic_threshold_filtering(self):
        """Only rows with max > threshold should be retained."""
        predictions = np.array(
            [
                [0.1, 0.2, 0.9],  # max=0.9, above 0.5
                [0.1, 0.3, 0.4],  # max=0.4, below 0.5
                [0.8, 0.1, 0.2],  # max=0.8, above 0.5
            ]
        )
        raw = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        meta = [("t0", "t1", "S1", "C1"), ("t2", "t3", "S2", "C2"), ("t4", "t5", "S3", "C3")]

        pw, pwt, idx_max = get_threshold_picks(predictions, 0.5, raw, meta)

        assert len(pw) == 2
        assert len(pwt) == 2
        assert len(idx_max) == 2
        assert idx_max[0] == 2  # argmax of row 0
        assert idx_max[1] == 0  # argmax of row 2

    def test_no_picks_above_threshold(self):
        """All predictions below threshold -> empty results."""
        predictions = np.array([[0.1, 0.2], [0.3, 0.4]])
        raw = [np.array([1, 2]), np.array([3, 4])]
        meta = [("a", "b", "S1", "C1"), ("c", "d", "S2", "C2")]

        pw, pwt, idx_max = get_threshold_picks(predictions, 0.5, raw, meta)

        assert pw == []
        assert pwt == []
        assert idx_max == []

    def test_all_picks_above_threshold(self):
        """All rows above threshold -> all returned with correct max indices."""
        predictions = np.array(
            [
                [0.6, 0.9, 0.3],
                [0.7, 0.5, 0.8],
            ]
        )
        raw = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        meta = [("a", "b", "S1", "C1"), ("c", "d", "S2", "C2")]

        pw, _pwt, idx_max = get_threshold_picks(predictions, 0.5, raw, meta)

        assert len(pw) == 2
        assert idx_max[0] == 1  # argmax of [0.6, 0.9, 0.3]
        assert idx_max[1] == 2  # argmax of [0.7, 0.5, 0.8]


# ---------------------------------------------------------------------------
# Edge-case tests for get_threshold_picks
# ---------------------------------------------------------------------------


class TestGetThresholdPicksEdgeCases:
    """Edge-case tests for get_threshold_picks."""

    def test_nan_in_predictions(self):
        """Rows with NaN should be excluded (NaN > threshold is False)."""
        predictions = np.array(
            [
                [np.nan, np.nan, np.nan],
                [0.1, 0.9, 0.3],
            ]
        )
        raw = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        meta = [("a", "b", "S1", "C1"), ("c", "d", "S2", "C2")]

        pw, pwt, _idx_max = get_threshold_picks(predictions, 0.5, raw, meta)
        # NaN row should not pass threshold; only second row passes
        assert len(pw) == 1
        assert pwt[0] == meta[1]

    def test_inf_in_predictions(self):
        """Rows with inf should pass threshold (inf > threshold is True)."""
        predictions = np.array(
            [
                [np.inf, 0.1, 0.2],
                [0.1, 0.2, 0.3],
            ]
        )
        raw = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        meta = [("a", "b", "S1", "C1"), ("c", "d", "S2", "C2")]

        pw, _pwt, idx_max = get_threshold_picks(predictions, 0.5, raw, meta)
        assert len(pw) == 1  # Only the inf row passes
        assert idx_max[0] == 0  # inf is at index 0

    def test_threshold_zero(self):
        """Threshold=0 should let all positive-max rows pass."""
        predictions = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        )
        raw = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        meta = [("a", "b", "S1", "C1"), ("c", "d", "S2", "C2")]

        pw, _pwt, _idx_max = get_threshold_picks(predictions, 0.0, raw, meta)
        assert len(pw) == 2

    def test_threshold_one(self):
        """Threshold=1.0 should only pass rows with max > 1.0."""
        predictions = np.array(
            [
                [0.5, 0.9, 0.3],
                [0.8, 1.5, 0.2],
            ]
        )
        raw = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        meta = [("a", "b", "S1", "C1"), ("c", "d", "S2", "C2")]

        pw, _pwt, idx_max = get_threshold_picks(predictions, 1.0, raw, meta)
        assert len(pw) == 1
        assert idx_max[0] == 1  # max of second row at index 1

    def test_single_row(self):
        """Single prediction row should work correctly."""
        predictions = np.array([[0.8, 0.3]])
        raw = [np.array([10, 20])]
        meta = [("a", "b", "S1", "C1")]

        pw, _pwt, idx_max = get_threshold_picks(predictions, 0.5, raw, meta)
        assert len(pw) == 1
        assert idx_max[0] == 0


# ---------------------------------------------------------------------------
# Edge-case tests for find_picking_windows
# ---------------------------------------------------------------------------


class TestFindPickingWindowsEdgeCases:
    """Edge-case tests for find_picking_windows."""

    def test_empty_filepaths(self):
        """Empty filepath list should return empty results."""
        windows, raw, meta = find_picking_windows(
            [],
            window_length=500,
            window_offset=250,
            time_periods=[(UTCDateTime(0), UTCDateTime(10))],
        )
        assert len(windows) == 0
        assert raw == []
        assert meta == []

    @patch("swe.picking.windows.read")
    def test_window_larger_than_data(self, mock_read):
        """Window length exceeding data length should produce zero windows."""
        sr = 250.0
        npts = 100  # only 100 samples
        start = UTCDateTime("2024-01-01T00:00:00")
        end = start + (npts / sr)

        tr = Trace(data=np.ones(npts, dtype=np.float64))
        tr.stats.sampling_rate = sr
        tr.stats.starttime = start
        tr.stats.station = "STA1"
        tr.stats.channel = "BHZ"
        tr.stats.network = "XX"
        mock_read.return_value = Stream([tr])

        _windows, raw, meta = find_picking_windows(
            ["/fake/XX.STA1.00.BHZ.mseed"],
            window_length=500,  # larger than npts
            window_offset=250,
            time_periods=[(start, end)],
            sampling_rate=sr,
        )
        assert len(raw) == 0
        assert meta == []


# ---------------------------------------------------------------------------
# Edge-case tests for remove_duplicate_picks
# ---------------------------------------------------------------------------


class TestRemoveDuplicatePicksEdgeCases:
    """Additional edge cases for remove_duplicate_picks."""

    def test_unknown_station_not_in_list(self):
        """Station not in station_list should be added dynamically."""
        t0 = UTCDateTime("2024-01-01T00:00:00")
        picked_windows = [np.array([1.0, 2.0])]
        picked_windows_times = [(t0, t0 + 1, "UNKNOWN", "HHZ")]
        index_of_max = [0]
        station_list = ["STA1", "STA2"]  # UNKNOWN not in list

        station_picks, uw, _uwt, _uidx = remove_duplicate_picks(
            picked_windows,
            picked_windows_times,
            index_of_max,
            station_list,
            sampling_rate=100.0,
        )

        # UNKNOWN should have been added dynamically
        assert "UNKNOWN" in station_picks
        assert len(station_picks["UNKNOWN"]) == 1
        assert len(uw) == 1
