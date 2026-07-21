"""Tests for swe.viz.waveforms and swe.viz.statistics modules."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure
from obspy import Stream, Trace, UTCDateTime
from obspy.core.trace import Stats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trace(
    *,
    network: str = "XX",
    station: str = "STA1",
    channel: str = "BHZ",
    sampling_rate: float = 100.0,
    npts: int = 1000,
    starttime: UTCDateTime | None = None,
) -> Trace:
    """Return a synthetic Trace with random data."""
    if starttime is None:
        starttime = UTCDateTime(2024, 1, 1)
    data = np.random.default_rng(42).standard_normal(npts).astype(np.float64)
    header = {
        "network": network,
        "station": station,
        "channel": channel,
        "sampling_rate": sampling_rate,
        "npts": npts,
        "starttime": starttime,
    }
    return Trace(data=data, header=header)


def _make_stream(n_traces: int = 3, **kwargs) -> Stream:
    """Return a synthetic Stream with *n_traces* traces."""
    st = Stream()
    for i in range(n_traces):
        st += _make_trace(station=f"ST{i:02d}", **kwargs)
    return st


# ---------------------------------------------------------------------------
# Waveform tests
# ---------------------------------------------------------------------------

class TestPlotStream:
    def test_returns_figure(self):
        from swe.viz.waveforms import plot_stream

        st = _make_stream(2, npts=500)
        fig = plot_stream(st)
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_empty_stream(self):
        from swe.viz.waveforms import plot_stream

        fig = plot_stream(Stream())
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestPlotTimeSeriesAndSpectrogram:
    def test_returns_figure(self):
        from swe.viz.waveforms import plot_time_series_and_spectrogram

        st = _make_stream(2, npts=500)
        fig = plot_time_series_and_spectrogram(st, title="test")
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestPlotTimeSeries:
    def test_returns_figure(self):
        from swe.viz.waveforms import plot_time_series

        st = _make_stream(2, npts=500)
        fig = plot_time_series(st, title="test ts")
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestPlotStack:
    def test_returns_figure(self):
        from swe.viz.waveforms import plot_stack

        st = _make_stream(3, npts=400)
        fig = plot_stack(st, title="stack")
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestPlotStreamAbsolute:
    def test_returns_figure(self):
        from swe.viz.waveforms import plot_stream_absolute

        st = _make_stream(2, npts=500)
        fig = plot_stream_absolute(st, title="abs")
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestPlotEventPicks:
    def test_returns_figure(self):
        from swe.viz.waveforms import plot_event_picks

        st = _make_stream(3, npts=500)
        picks = [
            {"time": UTCDateTime(2024, 1, 1, 0, 0, 2)},
            {"time": UTCDateTime(2024, 1, 1, 0, 0, 3)},
            {"time": UTCDateTime(2024, 1, 1, 0, 0, 4)},
        ]
        fig = plot_event_picks(st, picks, title="picks")
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_empty_picks(self):
        from swe.viz.waveforms import plot_event_picks

        fig = plot_event_picks(Stream(), [])
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestPlotSpectrogram:
    def test_returns_figure(self):
        from swe.viz.waveforms import plot_spectrogram

        tr = _make_trace(npts=2000)
        fig = plot_spectrogram(tr)
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_with_freq_max(self):
        from swe.viz.waveforms import plot_spectrogram

        tr = _make_trace(npts=2000)
        fig = plot_spectrogram(tr, freq_max=20.0)
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestPlotSpectra:
    def test_returns_figure(self):
        from swe.viz.waveforms import plot_spectra

        tr = _make_trace(npts=2000)
        fig = plot_spectra(tr)
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_with_freq_max(self):
        from swe.viz.waveforms import plot_spectra

        tr = _make_trace(npts=2000)
        fig = plot_spectra(tr, freq_max=25.0)
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestPlotMultiSpectra:
    def test_returns_figure(self):
        from swe.viz.waveforms import plot_multi_spectra

        st = _make_stream(3, npts=1000)
        fig = plot_multi_spectra(st)
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_custom_labels(self):
        from swe.viz.waveforms import plot_multi_spectra

        st = _make_stream(2, npts=1000)
        fig = plot_multi_spectra(st, labels=["A", "B"])
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestPlotTemplateAndStack:
    def test_returns_figure(self):
        from swe.viz.waveforms import plot_template_and_stack

        tmpl = _make_stream(2, npts=400)
        stack = _make_stream(2, npts=400)
        fig = plot_template_and_stack(tmpl, stack, title="t vs s")
        assert isinstance(fig, Figure)
        plt_close(fig)


# ---------------------------------------------------------------------------
# Statistics tests
# ---------------------------------------------------------------------------

class TestPlotDistribution:
    def test_returns_figure(self):
        from swe.viz.statistics import plot_distribution

        values = np.random.default_rng(0).standard_normal(500)
        fig = plot_distribution(values, bins=50, title="test", xlabel="Hz")
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_default_params(self):
        from swe.viz.statistics import plot_distribution

        fig = plot_distribution([1, 2, 3, 4, 5])
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestPlotMomentDistribution:
    def test_returns_figure(self):
        from swe.viz.statistics import plot_moment_distribution

        mags = np.random.default_rng(1).uniform(-1, 3, size=200)
        fig = plot_moment_distribution(mags, bins=30)
        assert isinstance(fig, Figure)
        plt_close(fig)


class TestPlotDateDistribution:
    def test_returns_figure(self):
        from swe.viz.statistics import plot_date_distribution

        # Use matplotlib date floats
        base = 738886.0  # some arbitrary matplotlib date
        dates = base + np.random.default_rng(2).uniform(0, 30, size=100)
        fig = plot_date_distribution(dates, bins=30)
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_with_utcdatetime(self):
        from swe.viz.statistics import plot_date_distribution

        dates = [
            UTCDateTime(2024, 1, d) for d in range(1, 20)
        ]
        fig = plot_date_distribution(dates, bins=10, title="UTCDateTime test")
        assert isinstance(fig, Figure)
        plt_close(fig)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def plt_close(fig: Figure) -> None:
    """Close a figure to free memory."""
    import matplotlib.pyplot as plt
    plt.close(fig)
