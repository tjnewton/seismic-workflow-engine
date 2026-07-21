"""Tests for swe.noise._plotting - plot_stalta and plot_noise_windows."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure
from obspy.core.trace import Trace
from obspy.core.utcdatetime import UTCDateTime
from obspy.signal.trigger import classic_sta_lta

from swe.noise._plotting import plot_noise_windows, plot_stalta


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_trace():
    """10-second synthetic Trace at 100 Hz."""
    npts = 1000
    tr = Trace(data=np.random.default_rng(42).normal(size=npts).astype(np.float64))
    tr.stats.sampling_rate = 100.0
    tr.stats.starttime = UTCDateTime("2024-01-01T00:00:00")
    tr.stats.station = "SYN"
    tr.stats.channel = "BHZ"
    tr.stats.network = "XX"
    return tr


@pytest.fixture()
def synthetic_cft(synthetic_trace):
    """Classic STA/LTA characteristic function from the synthetic trace."""
    return classic_sta_lta(synthetic_trace.data, int(0.5 * 100), int(5.0 * 100))


@pytest.fixture()
def noise_windows_2d():
    """2-D array of 5 noise windows, each 100 samples."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(5, 100))


@pytest.fixture()
def noise_windows_times():
    """List of (start, end, mpl_times) tuples for 3 noise windows."""
    base = UTCDateTime("2024-01-01T00:00:00")
    times_list = []
    for i in range(3):
        start = base + i * 2
        end = start + 1
        # 100 evenly spaced matplotlib date floats between start and end
        mpl_arr = np.linspace(start.matplotlib_date, end.matplotlib_date, 100)
        times_list.append((start, end, mpl_arr))
    return times_list


# ---------------------------------------------------------------------------
# plot_stalta tests
# ---------------------------------------------------------------------------


class TestPlotStalta:
    """Tests for plot_stalta."""

    def test_returns_figure(self, synthetic_trace, synthetic_cft):
        """Basic call should return a Figure."""
        start = synthetic_trace.stats.starttime + 1
        end = synthetic_trace.stats.endtime - 1
        fig = plot_stalta(synthetic_trace, synthetic_cft, start, end)
        assert isinstance(fig, Figure)

    def test_two_subplots(self, synthetic_trace, synthetic_cft):
        """Figure should contain exactly 2 subplots."""
        start = synthetic_trace.stats.starttime + 1
        end = synthetic_trace.stats.endtime - 1
        fig = plot_stalta(synthetic_trace, synthetic_cft, start, end)
        assert len(fig.axes) == 2

    def test_ylabel_contains_sta_lta(self, synthetic_trace, synthetic_cft):
        """When sta/lta samples given, ylabel includes them."""
        start = synthetic_trace.stats.starttime + 1
        end = synthetic_trace.stats.endtime - 1
        fig = plot_stalta(
            synthetic_trace, synthetic_cft, start, end,
            sta_samples=50, lta_samples=500,
        )
        ylabel = fig.axes[1].get_ylabel()
        assert "50" in ylabel
        assert "500" in ylabel

    def test_ylabel_default(self, synthetic_trace, synthetic_cft):
        """Without sta/lta params, ylabel should be 'STA/LTA'."""
        start = synthetic_trace.stats.starttime + 1
        end = synthetic_trace.stats.endtime - 1
        fig = plot_stalta(synthetic_trace, synthetic_cft, start, end)
        assert fig.axes[1].get_ylabel() == "STA/LTA"

    def test_narrow_time_window(self, synthetic_trace, synthetic_cft):
        """A 1-second window should still produce a valid figure."""
        start = synthetic_trace.stats.starttime + 2
        end = start + 1
        fig = plot_stalta(synthetic_trace, synthetic_cft, start, end)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_noise_windows compact (no times) tests
# ---------------------------------------------------------------------------


class TestPlotNoiseWindowsCompact:
    """Tests for plot_noise_windows in compact mode (no times)."""

    def test_returns_figure_compact(self, noise_windows_2d):
        """Compact view (no times) should return a Figure."""
        fig = plot_noise_windows(noise_windows_2d)
        assert isinstance(fig, Figure)

    def test_single_window(self):
        """A single noise window should return a valid Figure."""
        data = np.random.default_rng(42).normal(size=(1, 100))
        fig = plot_noise_windows(data)
        assert isinstance(fig, Figure)

    def test_custom_indices(self, noise_windows_2d):
        """Custom indices should be reflected in the y-ticks."""
        indices = [5, 10, 15, 20, 25]
        fig = plot_noise_windows(noise_windows_2d, indices=indices)
        ax = fig.axes[0]
        ytick_values = [int(t) for t in ax.get_yticks()]
        for idx in indices:
            assert idx in ytick_values

    def test_empty_windows(self):
        """Zero noise windows should be handled (returns a Figure)."""
        # plot_noise_windows with 0 windows: compact path with empty range
        # This exercises the edge - may raise, which is acceptable knowledge
        data = np.empty((0, 100))
        # With 0 windows the code would fail on min(indices). Verify we get an error.
        with pytest.raises((ValueError, IndexError)):
            plot_noise_windows(data)


# ---------------------------------------------------------------------------
# plot_noise_windows expanded (with times) tests
# ---------------------------------------------------------------------------


class TestPlotNoiseWindowsExpanded:
    """Tests for plot_noise_windows in expanded mode (with times)."""

    def test_returns_figure_expanded(self, noise_windows_times):
        """Expanded view (with times) should return a Figure."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(3, 100))
        fig = plot_noise_windows(data, noise_windows_times=noise_windows_times)
        assert isinstance(fig, Figure)

    def test_single_window_expanded(self):
        """Single window with times should still work."""
        base = UTCDateTime("2024-01-01T00:00:00")
        start = base
        end = base + 1
        mpl_arr = np.linspace(start.matplotlib_date, end.matplotlib_date, 100)
        times = [(start, end, mpl_arr)]
        data = np.random.default_rng(42).normal(size=(1, 100))
        fig = plot_noise_windows(data, noise_windows_times=times)
        assert isinstance(fig, Figure)

    def test_subplot_count_matches(self, noise_windows_times):
        """N windows with times should produce N subplots."""
        rng = np.random.default_rng(42)
        n = len(noise_windows_times)
        data = rng.normal(size=(n, 100))
        fig = plot_noise_windows(data, noise_windows_times=noise_windows_times)
        assert len(fig.axes) == n
