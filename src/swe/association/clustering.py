"""DBSCAN-based phase-pick association and clustering.

Provides functions to group seismic phase picks into event clusters using
DBSCAN and a manual decision tree for post-processing (duplicate removal,
merge, noise reassignment).
"""

from __future__ import annotations

import base64
import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
from matplotlib.dates import num2date
from obspy import UTCDateTime
from sklearn.cluster import DBSCAN
from sklearn import decomposition

from swe.io.snuffler import write_events_to_markers

logger = logging.getLogger(__name__)


class AssociationResult(NamedTuple):
    """Result from phase-pick association."""

    station_distance: dict[str, int]
    """``{station: index}`` forward lookup."""
    reverse_station_distance: dict[int, str]
    """``{index: station}`` reverse lookup."""
    pick_station_order: list
    """Station-distance index for each pick."""
    pick_times: list
    """Matplotlib date number for each pick."""
    labels: np.ndarray
    """Final cluster labels (``-1`` = noise)."""
    cluster_quality: dict[int, dict] | None = None
    """Per-cluster quality metrics, or ``None`` if not computed."""


def build_pick_dict(
    pick_times: np.ndarray | list,
    pick_station_order: list | np.ndarray,
    reverse_station_distance: dict[int, str],
    *,
    duplicate_threshold: float = 10.0,
    sampling_rate: float = 250.0,
    station_list: Sequence[str] | None = None,
) -> dict[str, list]:
    """Build a dictionary of per-station pick times, excluding duplicates.

    Parameters
    ----------
    pick_times : array-like
        Matplotlib date numbers for each pick.
    pick_station_order : array-like
        Station-distance index for each pick (maps via
        *reverse_station_distance*).
    reverse_station_distance : dict
        ``{index: station_name}`` lookup.
    duplicate_threshold : float
        Minimum separation in samples for picks on the same station.
    sampling_rate : float
        Sampling rate (Hz), used with *duplicate_threshold*.
    station_list : sequence of str, optional
        Explicit list of station names.  If *None*, keys are derived from
        *reverse_station_distance*.

    Returns
    -------
    dict
        ``{station_name: [UTCDateTime, ...]}``
    """
    station_picks: dict[str, list] = {}
    if station_list is not None:
        for sta in station_list:
            station_picks[str(sta)] = []
    else:
        for sta in reverse_station_distance.values():
            station_picks[str(sta)] = []

    if len(pick_times) == 0:
        return station_picks

    for pick_index, station_idx in enumerate(pick_station_order):
        station = reverse_station_distance[station_idx]
        pick_utc = UTCDateTime(num2date(pick_times[pick_index]))

        if len(station_picks[station]) == 0:
            station_picks[station].append(pick_utc)
        else:
            duplicate_flag = False
            for existing_pick in station_picks[station]:
                if abs(existing_pick - pick_utc) <= (
                    duplicate_threshold / sampling_rate
                ):
                    duplicate_flag = True
            if not duplicate_flag:
                station_picks[station].append(pick_utc)

    return station_picks


def build_pick_data(
    station_picks: dict[str, list],
    location_dict: dict[str, list[float]],
    station_distance: dict[str, int],
) -> tuple[list[list[float]], list[int], list[float]]:
    """Build arrays of pick locations and times for clustering.

    Parameters
    ----------
    station_picks : dict
        ``{station: [UTCDateTime, ...]}`` from :func:`build_pick_dict`.
    location_dict : dict
        ``{station: [longitude, latitude, elevation]}`` coordinates.
    station_distance : dict
        ``{station: order_index}`` along the array.

    Returns
    -------
    pick_array : list of list
        ``[[lon, lat, elev, matplotlib_date], ...]`` for each pick.
    pick_station_order : list of int
        Station distance index for each pick.
    pick_times : list of float
        Matplotlib date number for each pick.
    """
    pick_array: list[list[float]] = []
    pick_station_order: list[int] = []
    pick_times: list[float] = []

    for station in station_picks:
        for pick_time in station_picks[station]:
            pick_loc_and_time = location_dict[station].copy()
            pick_loc_and_time.append(pick_time.matplotlib_date)
            pick_array.append(pick_loc_and_time)
            pick_station_order.append(station_distance[station])
            pick_times.append(pick_time.matplotlib_date)

    return pick_array, pick_station_order, pick_times


def cluster_picks(
    pick_array: list | np.ndarray,
    *,
    clustering_algo: str = "DBSCAN",
    scale_time: bool = False,
    scale_factor: float = 1.0,
    use_pca: bool = False,
    pca_components: int | None = None,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
) -> np.ndarray:
    """Cluster picks using DBSCAN on standardised pick coordinates.

    Parameters
    ----------
    pick_array : array-like
        ``(N, 4)`` array of ``[lon, lat, elev, time]`` for each pick.
    clustering_algo : str
        Clustering algorithm.  Currently only ``"DBSCAN"`` is supported.
    scale_time : bool
        If *True*, multiply the time column by *scale_factor* before
        clustering.
    scale_factor : float
        Multiplicative factor for the time dimension.
    use_pca : bool
        If *True*, apply PCA before clustering.
    pca_components : int, optional
        Number of PCA components.  If *None*, all components are kept.
    dbscan_eps : float
        DBSCAN *eps* parameter (neighbourhood radius).
    dbscan_min_samples : int
        DBSCAN *min_samples* parameter.

    Returns
    -------
    numpy.ndarray
        Cluster labels for each pick (``-1`` indicates noise).
    """
    pick_array = np.asarray(pick_array, dtype=float)
    pick_array = np.nan_to_num(pick_array)

    means = np.mean(pick_array, axis=0)
    stds = np.std(pick_array, axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero
    pick_array = (pick_array - means) / stds
    pick_array = np.nan_to_num(pick_array)

    if scale_time and pick_array.shape[1] > 3:
        pick_array[:, 3] *= scale_factor

    if use_pca:
        pca = decomposition.PCA(n_components=pca_components)
        pca.fit(pick_array)
        data_to_cluster = pca.transform(pick_array)
    else:
        data_to_cluster = pick_array

    if clustering_algo == "DBSCAN":
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(
            data_to_cluster
        )
        labels = db.labels_
    else:
        raise ValueError(f"Unsupported clustering algorithm: {clustering_algo}")

    return labels


def get_cluster_stats(
    labels: np.ndarray,
    pick_times: list | np.ndarray,
) -> tuple[set[int], dict[int, float]]:
    """Compute median time and label set for each cluster.

    Parameters
    ----------
    labels : numpy.ndarray
        Cluster labels from :func:`cluster_picks`.
    pick_times : array-like
        Matplotlib date numbers for each pick.

    Returns
    -------
    unique_labels : set of int
        Distinct cluster labels (including ``-1`` for noise).
    cluster_medians : dict
        ``{label: median_matplotlib_date}``.
    """
    unique_labels_set = set(labels)
    pick_times = np.array(pick_times)

    cluster_medians: dict[int, float] = {}
    for label in unique_labels_set:
        member_indices = np.where(labels == label)[0]
        cluster_medians[label] = float(np.median(pick_times[member_indices]))

    return unique_labels_set, cluster_medians


def compute_cluster_quality(
    labels: np.ndarray,
    pick_times: list | np.ndarray,
    pick_station_order: list | np.ndarray,
    reverse_station_distance: dict[int, str],
) -> dict[int, dict]:
    """Compute quality metrics for each cluster.

    Parameters
    ----------
    labels : numpy.ndarray
        Cluster labels (``-1`` = noise).
    pick_times : array-like
        Matplotlib date numbers.
    pick_station_order : array-like
        Station distance indices per pick.
    reverse_station_distance : dict
        ``{index: station_name}`` lookup.

    Returns
    -------
    dict
        ``{cluster_label: {"n_picks": int, "n_stations": int,
        "time_spread_s": float, "station_coverage": float}}``.
    """
    pick_times = np.asarray(pick_times)
    pick_station_order = np.asarray(pick_station_order)
    total_stations = len(reverse_station_distance)

    quality: dict[int, dict] = {}
    for label in np.unique(labels):
        if label == -1:
            continue
        mask = labels == label
        cluster_times = pick_times[mask]
        cluster_stations = pick_station_order[mask]

        n_picks = int(np.sum(mask))
        unique_stations = set(int(s) for s in cluster_stations)
        n_stations = len(unique_stations)

        if len(cluster_times) > 1:
            time_range = float(cluster_times.max() - cluster_times.min())
            time_spread_s = time_range * 86400.0  # mpl dates are in days
        else:
            time_spread_s = 0.0

        station_coverage = n_stations / total_stations if total_stations > 0 else 0.0

        quality[label] = {
            "n_picks": n_picks,
            "n_stations": n_stations,
            "time_spread_s": time_spread_s,
            "station_coverage": station_coverage,
        }

    return quality


def process_clusters(
    labels: np.ndarray,
    pick_times: list | np.ndarray,
    pick_station_order: list | np.ndarray,
    *,
    min_picks_per_cluster: int = 4,
    temporal_threshold: float = 0.5,
    spatial_threshold: float = 5.0,
    merge_clusters: bool = False,
    process_noise: bool = False,
) -> np.ndarray:
    """Post-process clusters by dissolving small ones, removing misfits,
    merging overlapping clusters, and reassigning noise picks.

    Parameters
    ----------
    labels : numpy.ndarray
        Cluster labels (modified in-place).
    pick_times : array-like
        Matplotlib date numbers.
    pick_station_order : array-like
        Station distance indices.
    min_picks_per_cluster : int
        Minimum number of picks to keep a cluster.
    temporal_threshold : float
        Maximum time (seconds) from cluster median for a pick to belong.
    spatial_threshold : float
        Maximum station-distance difference for a pick to belong.
    merge_clusters : bool
        If *True*, merge clusters whose medians are within thresholds.
    process_noise : bool
        If *True*, attempt to reassign noise picks to nearby clusters.

    Returns
    -------
    numpy.ndarray
        Updated cluster labels.
    """
    labels = np.array(labels, copy=True)
    pick_times = np.array(pick_times)
    pick_station_order = np.array(pick_station_order)

    unique_labels, cluster_medians = get_cluster_stats(labels, pick_times)

    for cluster in list(unique_labels):
        if cluster == -1:
            continue

        member_indices = np.where(labels == cluster)[0]

        if len(member_indices) < min_picks_per_cluster:
            labels[member_indices] = -1
            continue

        _, cluster_medians = get_cluster_stats(labels, pick_times)
        cluster_median = UTCDateTime(num2date(cluster_medians[cluster]))

        all_cluster_stations = list(pick_station_order[member_indices])

        # Remove duplicate picks on the same station
        duplicates = defaultdict(list)
        for dup_idx, dup_station in enumerate(all_cluster_stations):
            duplicates[dup_station].append(dup_idx)
        duplicates = {
            s: idxs for s, idxs in duplicates.items() if len(idxs) > 1
        }

        for dup_station in duplicates:
            median_distances = []
            for index_index in duplicates[dup_station]:
                pick_time = UTCDateTime(
                    num2date(pick_times[member_indices[index_index]])
                )
                dist = abs(pick_time - cluster_median)
                median_distances.append(dist)

            min_dist = min(median_distances)
            min_index = median_distances.index(min_dist)

            for place_in_list, index_index in enumerate(
                duplicates[dup_station]
            ):
                if place_in_list != min_index:
                    labels[member_indices[index_index]] = -1

        member_indices = np.where(labels == cluster)[0]
        if len(member_indices) < min_picks_per_cluster:
            labels[member_indices] = -1
            continue

        _, cluster_medians = get_cluster_stats(labels, pick_times)
        cluster_median = UTCDateTime(num2date(cluster_medians[cluster]))
        unique_cluster_stations = set(pick_station_order[member_indices])

        # Remove misfit picks
        for index, pick in enumerate(pick_times[member_indices]):
            cluster_stations = list(unique_cluster_stations.copy())
            pick_station = pick_station_order[member_indices][index]

            if pick_station in cluster_stations:
                cluster_stations.remove(pick_station)
            if len(cluster_stations) == 0:
                continue

            station_diffs = [
                abs(pick_station - cs) for cs in cluster_stations
            ]
            min_diff_idx = min(
                range(len(station_diffs)), key=station_diffs.__getitem__
            )

            pick_utc = UTCDateTime(num2date(pick))
            if (abs(pick_utc - cluster_median) > temporal_threshold) or (
                abs(pick_station - cluster_stations[min_diff_idx])
                > spatial_threshold
            ):
                labels[member_indices[index]] = -1

    # Merge overlapping clusters
    if merge_clusters:
        unique_labels, cluster_medians = get_cluster_stats(labels, pick_times)
        work_labels = unique_labels.copy()
        work_labels.discard(-1)
        merge_dict: dict[int, int] = {}

        for cluster in work_labels:
            current_median = UTCDateTime(
                num2date(cluster_medians[cluster])
            )
            for other_cluster in list(cluster_medians.keys()):
                if other_cluster == -1 or other_cluster == cluster:
                    continue
                if (
                    other_cluster in merge_dict
                    or cluster in merge_dict
                ):
                    continue
                other_median = UTCDateTime(
                    num2date(cluster_medians[other_cluster])
                )
                if abs(current_median - other_median) <= temporal_threshold:
                    c_members = np.where(labels == cluster)[0]
                    o_members = np.where(labels == other_cluster)[0]
                    c_stations = set(pick_station_order[c_members])
                    o_stations = set(pick_station_order[o_members])

                    if c_stations.isdisjoint(o_stations):
                        c_sta_list = list(c_stations)
                        o_sta_list = list(o_stations)
                        for ps in c_sta_list:
                            diffs = [abs(ps - os) for os in o_sta_list]
                            min_idx = min(
                                range(len(diffs)), key=diffs.__getitem__
                            )
                            if (
                                abs(ps - o_sta_list[min_idx])
                                <= spatial_threshold
                            ):
                                if (
                                    other_cluster not in merge_dict
                                    and cluster not in merge_dict
                                ):
                                    merge_dict[cluster] = other_cluster
                    else:
                        merge_dict[cluster] = other_cluster

        for src, dst in merge_dict.items():
            labels[np.where(labels == src)[0]] = dst

    # Reassign noise picks to nearby clusters
    if process_noise:
        noise_indices = np.where(labels == -1)[0]
        _, cluster_medians = get_cluster_stats(labels, pick_times)

        for index, pick_val in enumerate(pick_times[noise_indices]):
            pick_utc = UTCDateTime(num2date(pick_val))
            noise_station = pick_station_order[noise_indices[index]]

            for cluster_key in cluster_medians:
                if cluster_key == -1:
                    continue
                c_median = UTCDateTime(num2date(cluster_medians[cluster_key]))
                c_members = np.where(labels == cluster_key)[0]
                c_stations = list(set(pick_station_order[c_members]))

                if len(c_stations) == 0:
                    continue
                diffs = [abs(noise_station - cs) for cs in c_stations]
                min_idx = min(range(len(diffs)), key=diffs.__getitem__)

                if (abs(pick_utc - c_median) <= temporal_threshold) and (
                    abs(noise_station - c_stations[min_idx])
                    <= spatial_threshold
                ):
                    if noise_station not in c_stations:
                        labels[noise_indices[index]] = cluster_key
                    break

    return labels


def associate_picks(
    station_picks: dict[str, list],
    station_list: Sequence[str],
    location_dict: dict[str, list[float]],
    *,
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
) -> AssociationResult:
    """Associate phase picks into events via clustering and post-processing.

    Parameters
    ----------
    station_picks : dict
        ``{station: [UTCDateTime, ...]}`` of picks per station.
    station_list : sequence of str
        Ordered station names (defines distance proxy).
    location_dict : dict
        ``{station: [lon, lat, elev]}`` coordinates.
    clustering_algo : str
        Clustering algorithm (``"DBSCAN"``).
    scale_time : bool
        Scale time dimension before clustering.
    scale_factor : float
        Time scale factor.
    use_pca : bool
        Apply PCA.
    pca_components : int, optional
        Number of PCA components.
    dbscan_eps : float
        DBSCAN eps parameter.
    dbscan_min_samples : int
        DBSCAN min_samples parameter.
    duplicate_threshold : float
        Duplicate removal threshold (samples).
    temporal_threshold : float
        Temporal threshold (seconds) for cluster processing.
    spatial_threshold : float
        Spatial threshold (station index units).
    min_picks_per_cluster : int
        Minimum picks to form an event.
    sampling_rate : float
        Sampling rate (Hz).

    Returns
    -------
    station_distance : dict
        ``{station: index}``
    reverse_station_distance : dict
        ``{index: station}``
    pick_station_order : list
        Station indices for each pick.
    pick_times : list
        Matplotlib dates for each pick.
    labels : numpy.ndarray
        Final cluster labels.
    """
    station_distance = {
        str(sta): idx for idx, sta in enumerate(station_list)
    }
    reverse_station_distance = {
        idx: str(sta) for idx, sta in enumerate(station_list)
    }

    pick_array, pick_station_order, pick_times = build_pick_data(
        station_picks, location_dict, station_distance
    )

    if len(pick_array) == 0:
        return AssociationResult(
            station_distance,
            reverse_station_distance,
            [],
            [],
            np.array([], dtype=int),
        )

    labels = cluster_picks(
        pick_array,
        clustering_algo=clustering_algo,
        scale_time=scale_time,
        scale_factor=scale_factor,
        use_pca=use_pca,
        pca_components=pca_components,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )

    labels = process_clusters(
        labels,
        pick_times,
        pick_station_order,
        min_picks_per_cluster=min_picks_per_cluster,
        temporal_threshold=temporal_threshold,
        spatial_threshold=spatial_threshold,
        merge_clusters=True,
        process_noise=True,
    )

    labels = process_clusters(
        labels,
        pick_times,
        pick_station_order,
        min_picks_per_cluster=min_picks_per_cluster,
        temporal_threshold=temporal_threshold,
        spatial_threshold=spatial_threshold,
        merge_clusters=False,
        process_noise=False,
    )

    quality = compute_cluster_quality(
        labels, pick_times, pick_station_order, reverse_station_distance,
    )

    return AssociationResult(
        station_distance,
        reverse_station_distance,
        pick_station_order,
        pick_times,
        labels,
        cluster_quality=quality,
    )


def save_associations(
    event_filename: str | Path,
    labels: np.ndarray,
    pick_times: list | np.ndarray,
    pick_station_order: list | np.ndarray,
    reverse_station_distance: dict[int, str],
    *,
    channel: str = "DP1",
) -> None:
    """Save association results to a Snuffler marker file.

    Parameters
    ----------
    event_filename : str or Path
        Output path for the ``.mrkr`` marker file.
    labels : numpy.ndarray
        Cluster labels.
    pick_times : array-like
        Matplotlib date numbers.
    pick_station_order : array-like
        Station distance indices.
    reverse_station_distance : dict
        ``{index: station_name}`` lookup.
    """
    pick_times = np.array(pick_times)
    pick_station_order = np.array(pick_station_order)

    unique_labels, cluster_medians = get_cluster_stats(labels, pick_times)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    events: dict[str, list[dict]] = {}
    for cluster in unique_labels:
        event_time_dt = num2date(cluster_medians[cluster])
        event_time_str = event_time_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-1]
        event_hash = base64.urlsafe_b64encode(
            hashlib.sha1(event_time_str.encode("utf8")).digest()
        ).decode("ascii")

        member_indices = np.where(labels == cluster)[0]
        picks_list: list[dict] = []
        for index, pick_val in enumerate(pick_times[member_indices]):
            pick_station_hack = pick_station_order[member_indices][index]
            pick_station = reverse_station_distance[pick_station_hack]
            pick_channel = channel
            pick_utc = UTCDateTime(num2date(pick_val))
            picks_list.append(
                {"station": f"{pick_station}.{pick_channel}", "time": pick_utc}
            )

        events[event_hash] = picks_list

    station_order_list = [
        reverse_station_distance[i]
        for i in sorted(reverse_station_distance.keys())
    ]
    write_events_to_markers(
        events,
        event_filename,
        station_order=station_order_list,
    )
    logger.info("Saved associations to %s", event_filename)
