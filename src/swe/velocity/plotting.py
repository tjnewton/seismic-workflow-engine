"""Velocity model and travel-time visualisation.

Provides plotting functions for inspecting velocity models and
precomputed travel-time grids.
"""

from __future__ import annotations

import logging

from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_velocity_profile(
    model,
    *,
    x_index: int | None = None,
    y_index: int | None = None,
    ax=None,
    figsize: tuple[float, float] = (4, 6),
) -> Figure:
    """Plot a 1-D velocity-depth profile extracted from a model.

    Parameters
    ----------
    model : VelocityModel
        Velocity model (duck-typed - needs ``x``, ``y``, ``z``,
        ``slowness`` attributes).
    x_index : int, optional
        Index along the x-axis to extract the profile.  Defaults to
        the midpoint.
    y_index : int, optional
        Index along the y-axis.  Defaults to the midpoint.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  If ``None``, a new figure is created.
    figsize : tuple of 2 floats
        Figure size when creating a new figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the profile plot.
    """
    import matplotlib.pyplot as plt

    if x_index is None:
        x_index = model.slowness.shape[0] // 2
    if y_index is None:
        y_index = model.slowness.shape[1] // 2

    # Cell-centre depths and velocities.
    z_cell = 0.5 * (model.z[:-1] + model.z[1:])
    velocity = 1.0 / model.slowness[x_index, y_index, :]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(velocity, z_cell)
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Depth")
    ax.invert_yaxis()
    ax.set_title("Velocity profile")

    return fig


def plot_velocity_slice(
    model,
    *,
    axis: str = "z",
    index: int | None = None,
    cmap: str = "RdYlBu_r",
    ax=None,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """Plot a 2-D cross-section through a velocity model.

    Parameters
    ----------
    model : VelocityModel
        Velocity model (duck-typed).
    axis : str
        Axis perpendicular to the slice: ``"x"``, ``"y"``, or ``"z"``.
    index : int, optional
        Cell index along the slicing axis.  Defaults to the midpoint.
    cmap : str
        Matplotlib colourmap name.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple of 2 floats
        Figure size when creating a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    slowness = model.slowness

    if axis == "z":
        if index is None:
            index = slowness.shape[2] // 2
        data = 1.0 / slowness[:, :, index]
        x_cell = 0.5 * (model.x[:-1] + model.x[1:])
        y_cell = 0.5 * (model.y[:-1] + model.y[1:])
        xlabel, ylabel = "X", "Y"
    elif axis == "y":
        if index is None:
            index = slowness.shape[1] // 2
        data = 1.0 / slowness[:, index, :]
        x_cell = 0.5 * (model.x[:-1] + model.x[1:])
        y_cell = 0.5 * (model.z[:-1] + model.z[1:])
        xlabel, ylabel = "X", "Z"
    elif axis == "x":
        if index is None:
            index = slowness.shape[0] // 2
        data = 1.0 / slowness[index, :, :]
        x_cell = 0.5 * (model.y[:-1] + model.y[1:])
        y_cell = 0.5 * (model.z[:-1] + model.z[1:])
        xlabel, ylabel = "Y", "Z"
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    im = ax.pcolormesh(x_cell, y_cell, data.T, cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax, label="Velocity")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Velocity slice ({axis}={index})")

    return fig


def plot_travel_time_slice(
    ttg,
    station_index: int,
    *,
    axis: str = "z",
    index: int | None = None,
    cmap: str = "viridis",
    ax=None,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """Plot a 2-D travel-time field for one station.

    Parameters
    ----------
    ttg : TravelTimeGrid
        Precomputed travel-time grid (duck-typed).
    station_index : int
        Index of the station whose travel times to plot.
    axis : str
        Axis perpendicular to the slice: ``"x"``, ``"y"``, or ``"z"``.
    index : int, optional
        Node index along the slicing axis.  Defaults to the midpoint.
    cmap : str
        Matplotlib colourmap.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple of 2 floats
        Figure size when creating a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    tt = ttg.travel_times[station_index]

    if axis == "z":
        if index is None:
            index = tt.shape[2] // 2
        data = tt[:, :, index]
        x_coords, y_coords = ttg.grid_x, ttg.grid_y
        xlabel, ylabel = "X", "Y"
    elif axis == "y":
        if index is None:
            index = tt.shape[1] // 2
        data = tt[:, index, :]
        x_coords, y_coords = ttg.grid_x, ttg.grid_z
        xlabel, ylabel = "X", "Z"
    elif axis == "x":
        if index is None:
            index = tt.shape[0] // 2
        data = tt[index, :, :]
        x_coords, y_coords = ttg.grid_y, ttg.grid_z
        xlabel, ylabel = "Y", "Z"
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    im = ax.pcolormesh(x_coords, y_coords, data.T, cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax, label="Travel time (s)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Travel time - station {station_index} ({axis}={index})")

    return fig
