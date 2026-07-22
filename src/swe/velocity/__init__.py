"""Velocity models and ray tracing."""

from __future__ import annotations

from swe.velocity.model import (
    TravelTimeGrid,
    VelocityModel,
    build_velocity_model_3d,
    load_travel_time_grid,
    read_velocity_model_1d,
    read_velocity_model_3d,
    save_travel_time_grid,
    save_velocity_model,
)
from swe.velocity.plotting import (
    plot_travel_time_slice,
    plot_velocity_profile,
    plot_velocity_slice,
)
from swe.velocity.raytrace import compute_travel_time_grid, compute_travel_times

__all__ = [
    # model types
    "TravelTimeGrid",
    "VelocityModel",
    # model I/O
    "build_velocity_model_3d",
    # ray tracing
    "compute_travel_time_grid",
    "compute_travel_times",
    "load_travel_time_grid",
    # plotting
    "plot_travel_time_slice",
    "plot_velocity_profile",
    "plot_velocity_slice",
    "read_velocity_model_1d",
    "read_velocity_model_3d",
    "save_travel_time_grid",
    "save_velocity_model",
]
