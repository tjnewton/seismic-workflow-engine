"""Tests for swe.viz.maps."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from swe.viz.maps import (
    plot_cross_sections,
    plot_detection_timeline,
    plot_nll_scatter,
    plot_station_event_map,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def station_coords():
    return np.array(
        [
            [0.0, 0.0],
            [5.0, 0.0],
            [0.0, 5.0],
            [5.0, 5.0],
        ]
    )


@pytest.fixture()
def station_names():
    return ["STA1", "STA2", "STA3", "STA4"]


@pytest.fixture()
def event_coords_2d():
    rng = np.random.default_rng(42)
    return rng.uniform(0, 5, size=(30, 2))


@pytest.fixture()
def event_coords_3d():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 5, size=(30, 3))
    coords[:, 2] = rng.uniform(0, 15, size=30)  # depth
    return coords


@pytest.fixture()
def event_magnitudes():
    rng = np.random.default_rng(42)
    return rng.uniform(-1, 3, size=30)


@pytest.fixture()
def event_times():
    rng = np.random.default_rng(42)
    # Matplotlib date floats (days since epoch)
    return rng.uniform(738000, 738030, size=30)


@pytest.fixture()
def nll_pdfs():
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame(
        {
            "x": rng.normal(2.5, 0.5, n),
            "y": rng.normal(2.5, 0.5, n),
            "z": rng.normal(5.0, 1.0, n),
            "weight": rng.uniform(0, 1, n),
        }
    )


def _plt_close(fig: Figure) -> None:
    """Close a figure to free memory."""
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests: plot_station_event_map
# ---------------------------------------------------------------------------


class TestPlotStationEventMap:
    def test_returns_figure(self, station_coords, station_names):
        fig = plot_station_event_map(station_coords, station_names)
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_with_events_2d(
        self,
        station_coords,
        station_names,
        event_coords_2d,
    ):
        fig = plot_station_event_map(
            station_coords,
            station_names,
            event_coords=event_coords_2d,
        )
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_with_events_3d_depth_color(
        self,
        station_coords,
        station_names,
        event_coords_3d,
    ):
        fig = plot_station_event_map(
            station_coords,
            station_names,
            event_coords=event_coords_3d,
        )
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_with_magnitudes(
        self,
        station_coords,
        station_names,
        event_coords_3d,
        event_magnitudes,
    ):
        fig = plot_station_event_map(
            station_coords,
            station_names,
            event_coords=event_coords_3d,
            event_magnitudes=event_magnitudes,
        )
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_with_topo_grid(self, station_coords, station_names):
        x = np.linspace(0, 5, 50)
        y = np.linspace(0, 5, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y) * 100
        fig = plot_station_event_map(
            station_coords,
            station_names,
            topo_grid=(x, y, Z),
        )
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_with_title(self, station_coords, station_names):
        fig = plot_station_event_map(
            station_coords,
            station_names,
            title="Test Map",
        )
        assert isinstance(fig, Figure)
        _plt_close(fig)


# ---------------------------------------------------------------------------
# Tests: plot_cross_sections
# ---------------------------------------------------------------------------


class TestPlotCrossSections:
    def test_returns_figure(self, event_coords_3d):
        fig = plot_cross_sections(event_coords_3d)
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_with_stations(
        self,
        event_coords_3d,
        station_coords,
        station_names,
    ):
        fig = plot_cross_sections(
            event_coords_3d,
            station_coords=station_coords,
            station_names=station_names,
        )
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_with_color_values(self, event_coords_3d, event_magnitudes):
        fig = plot_cross_sections(
            event_coords_3d,
            color_values=event_magnitudes,
        )
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_with_magnitudes(self, event_coords_3d, event_magnitudes):
        fig = plot_cross_sections(
            event_coords_3d,
            magnitudes=event_magnitudes,
        )
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_2d_coords_raises(self, event_coords_2d):
        with pytest.raises(ValueError, match="at least 3 columns"):
            plot_cross_sections(event_coords_2d)

    def test_with_title(self, event_coords_3d):
        fig = plot_cross_sections(event_coords_3d, title="Cross Sections")
        assert isinstance(fig, Figure)
        _plt_close(fig)


# ---------------------------------------------------------------------------
# Tests: plot_detection_timeline
# ---------------------------------------------------------------------------


class TestPlotDetectionTimeline:
    def test_returns_figure(self, event_coords_2d, event_times):
        fig = plot_detection_timeline(event_coords_2d, event_times)
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_with_stations(
        self,
        event_coords_2d,
        event_times,
        station_coords,
        station_names,
    ):
        fig = plot_detection_timeline(
            event_coords_2d,
            event_times,
            station_coords=station_coords,
            station_names=station_names,
        )
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_with_magnitudes(
        self,
        event_coords_2d,
        event_times,
        event_magnitudes,
    ):
        fig = plot_detection_timeline(
            event_coords_2d,
            event_times,
            magnitudes=event_magnitudes,
        )
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_3d_coords(self, event_coords_3d, event_times):
        fig = plot_detection_timeline(event_coords_3d, event_times)
        assert isinstance(fig, Figure)
        _plt_close(fig)


# ---------------------------------------------------------------------------
# Tests: plot_nll_scatter
# ---------------------------------------------------------------------------


class TestPlotNLLScatter:
    def test_returns_figure(self, nll_pdfs):
        fig = plot_nll_scatter(nll_pdfs)
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_with_hypocenters(self, nll_pdfs):
        hyp = np.array([[2.5, 2.5, 5.0]])
        fig = plot_nll_scatter(nll_pdfs, hypocenters=hyp)
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_with_stations(self, nll_pdfs, station_coords, station_names):
        fig = plot_nll_scatter(
            nll_pdfs,
            station_coords_km=station_coords,
            station_names=station_names,
        )
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_without_weight_column(self):
        rng = np.random.default_rng(42)
        pdfs = pd.DataFrame(
            {
                "x": rng.normal(0, 1, 100),
                "y": rng.normal(0, 1, 100),
                "z": rng.normal(5, 1, 100),
            }
        )
        fig = plot_nll_scatter(pdfs)
        assert isinstance(fig, Figure)
        _plt_close(fig)

    def test_single_hypocenter_1d(self, nll_pdfs):
        """Single hypocenter as (3,) array should work."""
        hyp = np.array([2.5, 2.5, 5.0])
        fig = plot_nll_scatter(nll_pdfs, hypocenters=hyp)
        assert isinstance(fig, Figure)
        _plt_close(fig)
