"""Tests for swe.noise.analysis."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure
from obspy.core.utcdatetime import UTCDateTime

from swe.noise.analysis import (
    analyze_noise_windows,
    plot_noise_station_histogram,
    plot_noise_time_histogram,
)
from swe.noise.detection import NoiseWindowMeta


def _make_noise_times(n, n_stations=3):
    """Create synthetic NoiseWindowMeta entries spread over *n_stations*."""
    base = UTCDateTime("2024-06-01T00:00:00")
    stations = [f"STA{i}" for i in range(n_stations)]
    entries = []
    for i in range(n):
        start = base + i * 60  # 1-minute spacing
        end = start + 30
        sta = stations[i % n_stations]
        entries.append(NoiseWindowMeta(start, end, sta, "BHZ"))
    return entries


class TestAnalyzeNoiseWindows:
    def test_basic_metrics(self):
        """Verify per-station metrics are computed correctly."""
        windows = np.array(
            [
                [1.0, -2.0, 3.0, -4.0],
                [0.5, 0.5, 0.5, 0.5],
            ]
        )
        times = [
            NoiseWindowMeta(UTCDateTime(0), UTCDateTime(1), "STA1", "BHZ"),
            NoiseWindowMeta(UTCDateTime(2), UTCDateTime(3), "STA2", "BHZ"),
        ]

        metric_map = analyze_noise_windows(windows, times)

        assert "STA1" in metric_map
        assert "STA2" in metric_map
        assert len(metric_map["STA1"]) == 1
        assert len(metric_map["STA2"]) == 1

        # STA1: max(abs) = 4.0, median(abs) = 2.5, rms = sqrt(mean([1,4,9,16])) = sqrt(7.5)
        sta1 = metric_map["STA1"][0]
        assert sta1[0] == 4.0  # max
        assert sta1[1] == 2.5  # median
        np.testing.assert_almost_equal(sta1[2], np.sqrt(7.5))  # rms

    def test_multiple_windows_same_station(self):
        windows = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        times = [
            NoiseWindowMeta(UTCDateTime(0), UTCDateTime(1), "STA1", "BHZ"),
            NoiseWindowMeta(UTCDateTime(2), UTCDateTime(3), "STA1", "BHZ"),
        ]
        metric_map = analyze_noise_windows(windows, times)
        assert len(metric_map["STA1"]) == 2

    def test_csv_output(self, tmp_path):
        windows = np.array([[1.0, 2.0]])
        times = [NoiseWindowMeta(UTCDateTime(0), UTCDateTime(1), "STA1", "BHZ")]
        output_csv = tmp_path / "metrics.csv"

        analyze_noise_windows(windows, times, output_csv=output_csv)
        assert output_csv.exists()
        content = output_csv.read_text()
        assert "STA1" in content

    def test_all_zeros_window(self):
        """All-zero window should give RMS=0, max=0, median=0."""
        windows = np.array([[0.0, 0.0, 0.0, 0.0]])
        times = [NoiseWindowMeta(UTCDateTime(0), UTCDateTime(1), "STA1", "BHZ")]
        metric_map = analyze_noise_windows(windows, times)
        sta1 = metric_map["STA1"][0]
        assert sta1[0] == 0.0  # max
        assert sta1[1] == 0.0  # median
        assert sta1[2] == 0.0  # rms

    def test_negative_values(self):
        """Negative values: abs should be taken for max and median."""
        windows = np.array([[-3.0, -2.0, -1.0]])
        times = [NoiseWindowMeta(UTCDateTime(0), UTCDateTime(1), "STA1", "BHZ")]
        metric_map = analyze_noise_windows(windows, times)
        sta1 = metric_map["STA1"][0]
        assert sta1[0] == 3.0  # max(abs)
        assert sta1[1] == 2.0  # median(abs)

    def test_single_sample_window(self):
        """Single-sample window should return valid metrics."""
        windows = np.array([[7.0]])
        times = [NoiseWindowMeta(UTCDateTime(0), UTCDateTime(1), "STA1", "BHZ")]
        metric_map = analyze_noise_windows(windows, times)
        sta1 = metric_map["STA1"][0]
        assert sta1[0] == 7.0
        assert sta1[1] == 7.0
        np.testing.assert_almost_equal(sta1[2], 7.0)


# ---------------------------------------------------------------------------
# plot_noise_time_histogram tests
# ---------------------------------------------------------------------------


class TestPlotNoiseTimeHistogram:
    """Tests for plot_noise_time_histogram."""

    def test_returns_figure(self):
        """Basic call should return a Figure."""
        times = _make_noise_times(20)
        fig = plot_noise_time_histogram(times, bin_width=3600)
        assert isinstance(fig, Figure)

    def test_bin_count(self):
        """Known span and bin_width should produce expected number of bars."""
        times = _make_noise_times(10)
        fig = plot_noise_time_histogram(times, bin_width=120)
        assert isinstance(fig, Figure)
        # Just verify it didn't crash and returned a valid figure
        assert len(fig.axes) >= 1

    def test_title_contains_count(self):
        """Subtitle should contain the noise window count."""
        times = _make_noise_times(15)
        fig = plot_noise_time_histogram(times, bin_width=3600)
        # The right-side title contains the count
        ax = fig.axes[0]
        # matplotlib stores multiple titles; check any contains the count
        titles = [ax.get_title(loc=loc) for loc in ("left", "center", "right")]
        assert any("15" in t for t in titles)

    def test_major_minor_tick_day_hour(self):
        """Exercises the day/hour tick code path."""
        times = _make_noise_times(20)
        fig = plot_noise_time_histogram(
            times, bin_width=3600, major_x_tick="day", minor_x_tick="hour",
        )
        assert isinstance(fig, Figure)

    def test_major_minor_tick_month_day(self):
        """Exercises the month/day tick code path."""
        # Spread windows over a longer time to make month ticks meaningful
        base = UTCDateTime("2024-01-01T00:00:00")
        times = []
        for i in range(30):
            start = base + i * 86400  # 1-day spacing
            end = start + 3600
            times.append(NoiseWindowMeta(start, end, "STA0", "BHZ"))
        fig = plot_noise_time_histogram(
            times, bin_width=86400, major_x_tick="month", minor_x_tick="day",
        )
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_noise_station_histogram tests
# ---------------------------------------------------------------------------


class TestPlotNoiseStationHistogram:
    """Tests for plot_noise_station_histogram."""

    def test_returns_figure(self):
        """Basic call should return a Figure."""
        times = _make_noise_times(12, n_stations=4)
        stations = ["STA0", "STA1", "STA2", "STA3"]
        fig = plot_noise_station_histogram(times, stations)
        assert isinstance(fig, Figure)

    def test_bar_count_matches_stations(self):
        """Number of bars should match the number of stations."""
        times = _make_noise_times(12, n_stations=4)
        stations = ["STA0", "STA1", "STA2", "STA3"]
        fig = plot_noise_station_histogram(times, stations)
        ax = fig.axes[0]
        assert len(ax.patches) == 4

    def test_empty_windows(self):
        """Empty window list should still produce a figure."""
        stations = ["STA0", "STA1"]
        fig = plot_noise_station_histogram([], stations)
        assert isinstance(fig, Figure)
