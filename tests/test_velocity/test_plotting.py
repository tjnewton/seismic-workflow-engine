"""Tests for swe.velocity.plotting."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from swe.velocity.model import TravelTimeGrid, VelocityModel
from swe.velocity.plotting import (
    plot_travel_time_slice,
    plot_velocity_profile,
    plot_velocity_slice,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def velocity_model():
    """A small synthetic velocity model."""
    x = np.linspace(0, 100, 6)
    y = np.linspace(0, 100, 6)
    z = np.linspace(0, 50, 4)
    slowness = np.full((5, 5, 3), 1.0 / 5000.0)
    return VelocityModel(x=x, y=y, z=z, slowness=slowness, model_type="3d")


@pytest.fixture()
def travel_time_grid():
    """A small synthetic travel-time grid."""
    gx = np.linspace(0, 100, 6)
    gy = np.linspace(0, 100, 6)
    gz = np.linspace(0, 50, 4)
    stations = np.array([[10.0, 20.0, 0.0], [80.0, 70.0, 0.0]])
    rng = np.random.default_rng(42)
    tt = rng.random((2, 6, 6, 4)) * 0.1
    return TravelTimeGrid(
        grid_x=gx, grid_y=gy, grid_z=gz,
        station_coords=stations, travel_times=tt,
    )


# ---------------------------------------------------------------------------
# TestPlotVelocityProfile
# ---------------------------------------------------------------------------

class TestPlotVelocityProfile:
    def test_returns_figure(self, velocity_model):
        fig = plot_velocity_profile(velocity_model)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_custom_indices(self, velocity_model):
        fig = plot_velocity_profile(
            velocity_model, x_index=0, y_index=0,
        )
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_with_existing_axes(self, velocity_model):
        import matplotlib.pyplot as plt
        fig_ext, ax = plt.subplots()
        fig = plot_velocity_profile(velocity_model, ax=ax)
        assert fig is fig_ext
        plt.close(fig)


# ---------------------------------------------------------------------------
# TestPlotVelocitySlice
# ---------------------------------------------------------------------------

class TestPlotVelocitySlice:
    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_returns_figure_for_each_axis(self, velocity_model, axis):
        fig = plot_velocity_slice(velocity_model, axis=axis)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_custom_index(self, velocity_model):
        fig = plot_velocity_slice(velocity_model, axis="z", index=0)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_invalid_axis(self, velocity_model):
        with pytest.raises(ValueError, match="axis must be"):
            plot_velocity_slice(velocity_model, axis="w")


# ---------------------------------------------------------------------------
# TestPlotTravelTimeSlice
# ---------------------------------------------------------------------------

class TestPlotTravelTimeSlice:
    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_returns_figure_for_each_axis(self, travel_time_grid, axis):
        fig = plot_travel_time_slice(travel_time_grid, 0, axis=axis)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_second_station(self, travel_time_grid):
        fig = plot_travel_time_slice(travel_time_grid, 1)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_invalid_axis(self, travel_time_grid):
        with pytest.raises(ValueError, match="axis must be"):
            plot_travel_time_slice(travel_time_grid, 0, axis="w")
