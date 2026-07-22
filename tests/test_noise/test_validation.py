"""Tests for swe.noise.validation."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from obspy.core.utcdatetime import UTCDateTime

from swe.noise.validation import explore_noise_windows, verify_noise_windows

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_MARKER_CONTENT = """\
# Snuffler Markers File Version 0.2
phase: 2024-01-01 00:00:05.0000  0 XX.STA1..BHZ    abc123hash= 2024-01-01   00:00:05.0000 P        None False
phase: 2024-01-01 00:00:25.0000  0 XX.STA2..BHZ    abc123hash= 2024-01-01   00:00:25.0000 P        None False
"""


def _make_windows_and_times(n=4):
    """Create synthetic noise windows and metadata."""
    rng = np.random.default_rng(42)
    windows = rng.standard_normal((n, 100))
    times = [
        (UTCDateTime("2024-01-01T00:00:00"), UTCDateTime("2024-01-01T00:00:10"), "STA1", "BHZ"),
        (UTCDateTime("2024-01-01T00:00:10"), UTCDateTime("2024-01-01T00:00:20"), "STA1", "BHZ"),
        (UTCDateTime("2024-01-01T00:00:20"), UTCDateTime("2024-01-01T00:00:30"), "STA2", "BHZ"),
        (UTCDateTime("2024-01-01T00:00:30"), UTCDateTime("2024-01-01T00:00:40"), "STA2", "BHZ"),
    ][:n]
    return windows, times


# ---------------------------------------------------------------------------
# verify_noise_windows tests
# ---------------------------------------------------------------------------


class TestVerifyNoiseWindows:
    def test_removes_windows_containing_picks(self, tmp_path):
        """Windows that contain a P-pick should be removed."""
        mrkr = tmp_path / "picks.mrkr"
        mrkr.write_text(SAMPLE_MARKER_CONTENT)

        windows, times = _make_windows_and_times(4)
        verified_w, verified_t = verify_noise_windows(windows, times, mrkr)

        # Pick at 00:00:05 hits window 0 (STA1, 0-10s)
        # Pick at 00:00:25 hits window 2 (STA2, 20-30s)
        assert len(verified_t) == 2
        assert verified_w.shape[0] == 2
        # Remaining windows are index 1 and 3
        assert verified_t[0][2] == "STA1"  # window 1 (10-20s)
        assert verified_t[1][2] == "STA2"  # window 3 (30-40s)

    def test_no_false_positives(self, tmp_path):
        """When no picks fall in windows, all windows are kept."""
        # Marker with a pick at a time that doesn't overlap any window
        mrkr = tmp_path / "picks.mrkr"
        mrkr.write_text(
            "phase: 2024-01-01 01:00:00.0000  0 XX.STA1..BHZ    "
            "abc123hash= 2024-01-01   01:00:00.0000 P        None False\n"
        )

        windows, times = _make_windows_and_times(4)
        verified_w, verified_t = verify_noise_windows(windows, times, mrkr)

        assert verified_w.shape[0] == 4
        assert len(verified_t) == 4

    def test_empty_marker_file(self, tmp_path):
        """Empty marker file means no windows removed."""
        mrkr = tmp_path / "empty.mrkr"
        mrkr.write_text("")

        windows, times = _make_windows_and_times(3)
        verified_w, verified_t = verify_noise_windows(windows, times, mrkr)

        assert verified_w.shape[0] == 3
        assert len(verified_t) == 3

    def test_preserves_window_data(self, tmp_path):
        """Remaining windows should have identical data to originals."""
        mrkr = tmp_path / "picks.mrkr"
        mrkr.write_text(SAMPLE_MARKER_CONTENT)

        windows, times = _make_windows_and_times(4)
        verified_w, _verified_t = verify_noise_windows(windows, times, mrkr)

        # Window 1 (index 1 in original) should now be first
        np.testing.assert_array_equal(verified_w[0], windows[1])

    def test_station_channel_must_match(self, tmp_path):
        """A pick on a different station should not remove the window."""
        mrkr = tmp_path / "picks.mrkr"
        # Pick on STA3 at time that overlaps STA1's window
        mrkr.write_text(
            "phase: 2024-01-01 00:00:05.0000  0 XX.STA3..BHZ    "
            "abc123hash= 2024-01-01   00:00:05.0000 P        None False\n"
        )

        windows, times = _make_windows_and_times(4)
        verified_w, _verified_t = verify_noise_windows(windows, times, mrkr)

        assert verified_w.shape[0] == 4


# ---------------------------------------------------------------------------
# explore_noise_windows tests
# ---------------------------------------------------------------------------


class TestExploreNoiseWindows:
    @patch("swe.noise.validation.plot_noise_windows")
    @patch("builtins.input", return_value="")
    def test_no_removal(self, mock_input, mock_plot):
        """Empty input means no windows removed."""
        windows, times = _make_windows_and_times(3)
        result_w, _result_t = explore_noise_windows(
            windows,
            times,
            batch_size=10,
        )
        # All windows processed, no removal
        assert result_w is None or result_w.shape[0] == 3

    @patch("swe.noise.validation.plot_noise_windows")
    @patch("builtins.input", return_value="0 2")
    def test_removal_by_index(self, mock_input, mock_plot):
        """User-specified indices should be removed."""
        windows, times = _make_windows_and_times(3)
        result_w, result_t = explore_noise_windows(
            windows,
            times,
            batch_size=10,
        )
        assert result_w is not None
        assert result_w.shape[0] == 1
        assert result_t is not None
        assert len(result_t) == 1

    @patch("swe.noise.validation.plot_noise_windows")
    @patch("builtins.input", return_value="quit")
    def test_quit_saves_progress(self, mock_input, mock_plot, tmp_path):
        """Typing 'quit' should save progress and return None."""
        windows, times = _make_windows_and_times(4)
        # Don't pass progress_file on first run - the function treats
        # a non-None value as "resume from existing file".  The default
        # save path is "explore_progress.json" in cwd.
        import os

        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            result_w, result_t = explore_noise_windows(
                windows,
                times,
                batch_size=2,
            )
        finally:
            os.chdir(original_dir)

        assert result_w is None
        assert result_t is None
        assert (tmp_path / "explore_progress.json").exists()

    @patch("swe.noise.validation.plot_noise_windows")
    @patch("builtins.input", return_value="1")
    def test_resume_from_progress(self, mock_input, mock_plot, tmp_path):
        """Should resume from a saved progress file."""
        import json

        progress = tmp_path / "progress.json"
        progress.write_text(
            json.dumps(
                {
                    "start_index": 2,
                    "removal_indices": [0],
                }
            )
        )

        windows, times = _make_windows_and_times(4)
        result_w, _result_t = explore_noise_windows(
            windows,
            times,
            progress_file=progress,
            batch_size=10,
        )
        # Removal indices: [0] from saved + [1] from user input = [0, 1]
        assert result_w is not None
        assert result_w.shape[0] == 2
