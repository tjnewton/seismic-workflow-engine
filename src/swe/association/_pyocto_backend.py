"""PyOcto association backend adapter.

Wraps the PyOcto ``OctoAssociator`` to conform to the SWE
``AssociationResult`` interface.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from obspy import UTCDateTime

from swe.association.clustering import AssociationResult

logger = logging.getLogger(__name__)


def _require_pyocto():
    """Import pyocto, raising a clear error if missing."""
    try:
        import pyocto
    except ImportError as exc:
        raise ImportError(
            "pyocto is required for method='pyocto'. "
            "Install it with: pip install seismic-workflow-engine[association]"
        ) from exc
    return pyocto


def _build_pyocto_velocity_model(pyocto, velocity_model: dict[str, float] | None):
    """Construct a PyOcto 1-D velocity model."""
    vm = velocity_model or {}
    vp = vm.get("vp", 6.0)
    vs = vm.get("vs", 3.43)
    # Simple homogeneous 1-D model
    return pyocto.associator.VelocityModel1D(
        layers=[(0, vp, vs)],
    )


def _build_pyocto_picks(
    station_picks: dict[str, list],
    pick_phases: dict[str, list[str]] | None,
    reference_time: UTCDateTime,
) -> list[dict]:
    """Convert SWE picks to PyOcto pick format.

    Returns a list of dicts with keys: station, phase, time (seconds from
    reference).
    """
    picks = []
    for sta, times in station_picks.items():
        phases = (pick_phases or {}).get(sta, ["P"] * len(times))
        for i, t in enumerate(times):
            phase = phases[i] if i < len(phases) else "P"
            picks.append(
                {
                    "station": sta,
                    "phase": phase,
                    "time": float(t - reference_time),
                }
            )
    return picks


def _build_pyocto_stations(
    location_dict: dict[str, list[float]],
) -> dict[str, tuple[float, float, float]]:
    """Convert station locations to PyOcto format (km coordinates)."""
    stations = {}
    for sta, coords in location_dict.items():
        # PyOcto expects local km coordinates; here we pass lon/lat/elev
        # and let the caller configure the coordinate system.
        stations[sta] = (coords[0], coords[1], coords[2] if len(coords) > 2 else 0.0)
    return stations


def associate_picks_pyocto(
    station_picks: dict[str, list],
    station_list: Sequence[str],
    location_dict: dict[str, list[float]],
    *,
    pick_phases: dict[str, list[str]] | None = None,
    velocity_model: dict[str, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    time_before: float = 1.0,
    n_picks: int = 5,
    n_p_picks: int = 2,
    n_s_picks: int = 2,
    n_p_and_s_picks: int = 0,
) -> AssociationResult:
    """Associate picks using PyOcto's OctoAssociator.

    Parameters
    ----------
    station_picks : dict
        ``{station: [UTCDateTime, ...]}`` picks per station.
    station_list : sequence of str
        Ordered station names.
    location_dict : dict
        ``{station: [lon, lat, elev]}`` coordinates.
    pick_phases : dict, optional
        ``{station: ["P", "S", ...]}`` per pick.
    velocity_model : dict, optional
        ``{"vp": float, "vs": float}``.  Defaults to vp=6.0, vs=3.43 km/s.
    xlim, ylim, zlim : tuple, optional
        Search bounds ``(min, max)`` in km.
    time_before : float
        Seconds before first pick to start search.
    n_picks, n_p_picks, n_s_picks, n_p_and_s_picks : int
        Minimum pick requirements.

    Returns
    -------
    AssociationResult
    """
    pyocto = _require_pyocto()

    station_distance = {str(sta): idx for idx, sta in enumerate(station_list)}
    reverse_station_distance = {idx: str(sta) for idx, sta in enumerate(station_list)}

    # Build velocity model
    vel_model = _build_pyocto_velocity_model(pyocto, velocity_model)

    # Determine reference time
    all_times = [t for times in station_picks.values() for t in times]
    if not all_times:
        return AssociationResult(
            station_distance, reverse_station_distance, [], [], np.array([], dtype=int)
        )
    reference_time = min(all_times)

    # Build picks and stations
    picks = _build_pyocto_picks(station_picks, pick_phases, reference_time)
    stations = _build_pyocto_stations(location_dict)

    # Configure associator
    xlim = xlim or (-50.0, 50.0)
    ylim = ylim or (-50.0, 50.0)
    zlim = zlim or (0.0, 40.0)

    associator = pyocto.OctoAssociator(
        velocity_model=vel_model,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        time_before=time_before,
        n_picks=n_picks,
        n_p_picks=n_p_picks,
        n_s_picks=n_s_picks,
        n_p_and_s_picks=n_p_and_s_picks,
    )

    events, assignments = associator.associate(picks, stations)
    logger.info("PyOcto found %d event(s)", len(events))

    # Convert assignments to labels
    pick_station_order: list[int] = []
    pick_times: list[float] = []
    labels_list: list[int] = []

    for sta, times in station_picks.items():
        for t in times:
            pick_station_order.append(station_distance.get(sta, 0))
            pick_times.append(t.matplotlib_date)
            labels_list.append(-1)

    # Map assignments back to labels
    if hasattr(assignments, "iterrows"):
        idx_col = "event_index" if "event_index" in assignments.columns else "event_idx"
        if idx_col in assignments.columns:
            for i, row in assignments.iterrows():
                if 0 <= i < len(labels_list):
                    labels_list[i] = int(row[idx_col])
    elif isinstance(assignments, (list, np.ndarray)):
        for i, label in enumerate(assignments):
            if 0 <= i < len(labels_list):
                labels_list[i] = int(label)

    return AssociationResult(
        station_distance,
        reverse_station_distance,
        pick_station_order,
        pick_times,
        np.array(labels_list, dtype=int),
    )
