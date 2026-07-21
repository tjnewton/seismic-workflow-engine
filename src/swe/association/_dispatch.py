"""Unified dispatcher for phase-pick association backends.

Routes ``associate_picks()`` calls to DBSCAN (default), PyOcto, or GaMMA
based on the *method* parameter.  All backends return the same
:class:`~swe.association.clustering.AssociationResult` NamedTuple.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from swe.association.clustering import AssociationResult
from swe.association.clustering import associate_picks as _dbscan_associate
from swe.picks import SWEPick, to_station_picks

logger = logging.getLogger(__name__)

_SUPPORTED_METHODS = frozenset({"dbscan", "pyocto", "gamma"})


def associate_picks(
    station_picks: dict[str, list] | list[SWEPick],
    station_list: Sequence[str],
    location_dict: dict[str, list[float]],
    *,
    method: str = "dbscan",
    # DBSCAN parameters
    clustering_algo: str = "DBSCAN",
    scale_time: bool = False,
    scale_factor: float = 1.0,
    use_pca: bool = False,
    pca_components: int | None = None,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    duplicate_threshold: float = 10.0,
    temporal_threshold: float = 0.5,
    spatial_threshold: float = 5.0,
    min_picks_per_cluster: int = 4,
    sampling_rate: float = 250.0,
    # PyOcto parameters
    velocity_model: dict[str, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    time_before: float = 1.0,
    n_picks: int = 5,
    n_p_picks: int = 2,
    n_s_picks: int = 2,
    n_p_and_s_picks: int = 0,
    # GaMMA parameters
    center: tuple[float, float] | None = None,
    use_amplitude: bool = False,
    min_picks_per_eq: int = 5,
    max_sigma11: float = 2.0,
    # Shared
    pick_phases: dict[str, list[str]] | None = None,
) -> AssociationResult:
    """Associate phase picks into events using the specified method.

    Parameters
    ----------
    station_picks : dict
        ``{station: [UTCDateTime, ...]}`` of picks per station.
    station_list : sequence of str
        Ordered station names.
    location_dict : dict
        ``{station: [lon, lat, elev]}`` coordinates.
    method : str
        Association method: ``"dbscan"`` (default), ``"pyocto"``, or
        ``"gamma"``.
    pick_phases : dict, optional
        ``{station: ["P", "S", ...]}`` phase labels per pick, required
        for PyOcto and GaMMA backends. Must align with *station_picks*.

    Returns
    -------
    AssociationResult
        5-tuple compatible with all downstream consumers.

    Other Parameters
    ----------------
    See individual backend modules for method-specific parameters.
    """
    # Accept list[SWEPick] directly - convert to dict format
    if isinstance(station_picks, list):
        station_picks, derived_phases = to_station_picks(station_picks)
        if pick_phases is None:
            pick_phases = derived_phases

    method = method.lower()
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported association method: {method!r}. Supported: {sorted(_SUPPORTED_METHODS)}"
        )

    if method == "dbscan":
        return _dbscan_associate(
            station_picks,
            station_list,
            location_dict,
            clustering_algo=clustering_algo,
            scale_time=scale_time,
            scale_factor=scale_factor,
            use_pca=use_pca,
            pca_components=pca_components,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            duplicate_threshold=duplicate_threshold,
            temporal_threshold=temporal_threshold,
            spatial_threshold=spatial_threshold,
            min_picks_per_cluster=min_picks_per_cluster,
            sampling_rate=sampling_rate,
        )

    if method == "pyocto":
        from swe.association._pyocto_backend import associate_picks_pyocto

        return associate_picks_pyocto(
            station_picks,
            station_list,
            location_dict,
            pick_phases=pick_phases,
            velocity_model=velocity_model,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            time_before=time_before,
            n_picks=n_picks,
            n_p_picks=n_p_picks,
            n_s_picks=n_s_picks,
            n_p_and_s_picks=n_p_and_s_picks,
        )

    # method == "gamma"
    from swe.association._gamma_backend import associate_picks_gamma

    return associate_picks_gamma(
        station_picks,
        station_list,
        location_dict,
        pick_phases=pick_phases,
        center=center,
        use_amplitude=use_amplitude,
        min_picks_per_eq=min_picks_per_eq,
        max_sigma11=max_sigma11,
    )


# ---------------------------------------------------------------------------
# Shared conversion utilities
# ---------------------------------------------------------------------------


def picks_to_dataframe(
    station_picks: dict[str, list],
    *,
    pick_phases: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Convert ``{station: [UTCDateTime]}`` to a picks DataFrame.

    Parameters
    ----------
    station_picks : dict
        ``{station: [UTCDateTime, ...]}``
    pick_phases : dict, optional
        ``{station: ["P", "S", ...]}`` aligned with *station_picks*.
        Defaults to ``"P"`` for all picks.

    Returns
    -------
    pandas.DataFrame
        Columns: ``station``, ``time``, ``phase``.
    """
    rows: list[dict] = []
    for sta, times in station_picks.items():
        phases = (pick_phases or {}).get(sta, ["P"] * len(times))
        if pick_phases is not None and sta in pick_phases and len(phases) != len(times):
            raise ValueError(
                f"pick_phases[{sta!r}] has {len(phases)} entries but "
                f"station_picks[{sta!r}] has {len(times)} picks; "
                "these must have matching lengths"
            )
        for i, t in enumerate(times):
            phase = phases[i] if i < len(phases) else "P"
            rows.append(
                {
                    "station": sta,
                    "time": t.datetime if hasattr(t, "datetime") else t,
                    "timestamp": float(t),
                    "phase": phase,
                }
            )
    return pd.DataFrame(rows)


def stations_to_dataframe(
    location_dict: dict[str, list[float]],
) -> pd.DataFrame:
    """Convert ``{station: [lon, lat, elev]}`` to a stations DataFrame.

    Returns
    -------
    pandas.DataFrame
        Columns: ``station``, ``longitude``, ``latitude``, ``elevation``.
    """
    rows = [
        {
            "station": sta,
            "longitude": coords[0],
            "latitude": coords[1],
            "elevation": coords[2] if len(coords) > 2 else 0.0,
        }
        for sta, coords in location_dict.items()
    ]
    return pd.DataFrame(rows)


def labels_from_assignments(
    assignments: pd.DataFrame,
    all_picks: pd.DataFrame,
) -> np.ndarray:
    """Convert event-assignment DataFrame to a label array.

    Parameters
    ----------
    assignments : pandas.DataFrame
        Must have an ``event_index`` (or ``event_idx``) column and a row
        index aligning with *all_picks*.
    all_picks : pandas.DataFrame
        The full picks DataFrame used for association.

    Returns
    -------
    numpy.ndarray
        Label array of length ``len(all_picks)``.  ``-1`` = unassociated.
    """
    labels = np.full(len(all_picks), -1, dtype=int)
    idx_col = "event_index" if "event_index" in assignments.columns else "event_idx"
    if idx_col not in assignments.columns:
        return labels

    for i, row in assignments.iterrows():
        if 0 <= i < len(labels):
            labels[i] = int(row[idx_col])
    return labels
