"""GaMMA association backend adapter.

Wraps the GaMMA ``association()`` function to conform to the SWE
``AssociationResult`` interface.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from swe.association.clustering import AssociationResult

logger = logging.getLogger(__name__)


def _require_gamma():
    """Import gamma, raising a clear error if missing."""
    try:
        from gamma import utils as gamma_utils
    except ImportError as exc:
        raise ImportError(
            "gamma-seismology is required for method='gamma'. "
            "Install it with: pip install seismic-workflow-engine[association]"
        ) from exc
    return gamma_utils


def _build_gamma_picks(
    station_picks: dict[str, list],
    pick_phases: dict[str, list[str]] | None,
) -> pd.DataFrame:
    """Convert SWE picks to GaMMA picks DataFrame."""
    rows: list[dict] = []
    for sta, times in station_picks.items():
        phases = (pick_phases or {}).get(sta, ["P"] * len(times))
        for i, t in enumerate(times):
            phase = phases[i] if i < len(phases) else "P"
            rows.append(
                {
                    "id": sta,
                    "timestamp": t.datetime,
                    "type": phase,
                    "prob": 1.0,
                    "amp": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _build_gamma_stations(
    location_dict: dict[str, list[float]],
) -> pd.DataFrame:
    """Convert station dict to GaMMA stations DataFrame."""
    rows = []
    for sta, coords in location_dict.items():
        rows.append(
            {
                "id": sta,
                "longitude": coords[0],
                "latitude": coords[1],
                "elevation(m)": coords[2] if len(coords) > 2 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _build_gamma_config(
    location_dict: dict[str, list[float]],
    *,
    center: tuple[float, float] | None = None,
    use_amplitude: bool = False,
    min_picks_per_eq: int = 5,
    max_sigma11: float = 2.0,
) -> dict:
    """Construct GaMMA configuration dict."""
    if center is None:
        lons = [c[0] for c in location_dict.values()]
        lats = [c[1] for c in location_dict.values()]
        center = (float(np.mean(lons)), float(np.mean(lats)))

    return {
        "center": center,
        "xlim_degree": [center[0] - 2, center[0] + 2],
        "ylim_degree": [center[1] - 2, center[1] + 2],
        "zlim_km": [0, 40],
        "use_amplitude": use_amplitude,
        "min_picks_per_eq": min_picks_per_eq,
        "max_sigma11": max_sigma11,
        "method": "BGMM",
        "oversample_factor": 4,
        "dims": ["x(km)", "y(km)", "z(km)"],
        "covariance_prior": [5.0, 5.0],
    }


def associate_picks_gamma(
    station_picks: dict[str, list],
    station_list: Sequence[str],
    location_dict: dict[str, list[float]],
    *,
    pick_phases: dict[str, list[str]] | None = None,
    center: tuple[float, float] | None = None,
    use_amplitude: bool = False,
    min_picks_per_eq: int = 5,
    max_sigma11: float = 2.0,
) -> AssociationResult:
    """Associate picks using GaMMA's Bayesian Gaussian Mixture Model.

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
    center : tuple, optional
        ``(longitude, latitude)`` centre of the study area.
        Auto-inferred from stations if not provided.
    use_amplitude : bool
        Whether to use amplitude information in association.
    min_picks_per_eq : int
        Minimum picks per event.
    max_sigma11 : float
        Maximum covariance constraint.

    Returns
    -------
    AssociationResult
    """
    gamma_utils = _require_gamma()

    station_distance = {str(sta): idx for idx, sta in enumerate(station_list)}
    reverse_station_distance = {idx: str(sta) for idx, sta in enumerate(station_list)}

    if not any(station_picks.values()):
        return AssociationResult(
            station_distance, reverse_station_distance, [], [], np.array([], dtype=int)
        )

    # Build GaMMA inputs
    picks_df = _build_gamma_picks(station_picks, pick_phases)
    stations_df = _build_gamma_stations(location_dict)
    config = _build_gamma_config(
        location_dict,
        center=center,
        use_amplitude=use_amplitude,
        min_picks_per_eq=min_picks_per_eq,
        max_sigma11=max_sigma11,
    )

    # Run GaMMA association
    catalogs, assignments = gamma_utils.association(
        picks_df,
        stations_df,
        config,
    )
    logger.info("GaMMA found %d event(s)", len(catalogs))

    # Convert to AssociationResult format
    pick_station_order: list[int] = []
    pick_times: list[float] = []
    labels_list: list[int] = []

    for sta, times in station_picks.items():
        for t in times:
            pick_station_order.append(station_distance.get(sta, 0))
            pick_times.append(t.matplotlib_date)
            labels_list.append(-1)

    # Map assignments back
    if isinstance(assignments, pd.DataFrame) and "event_index" in assignments.columns:
        for i, row in assignments.iterrows():
            if 0 <= i < len(labels_list):
                labels_list[i] = int(row["event_index"])
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
