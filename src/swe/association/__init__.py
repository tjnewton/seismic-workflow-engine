"""Phase-pick association via DBSCAN clustering, PyOcto, and GaMMA."""

from __future__ import annotations

from swe.association._dispatch import associate_picks
from swe.association.clustering import (
    AssociationResult,
    build_pick_data,
    build_pick_dict,
    cluster_picks,
    compute_cluster_quality,
    get_cluster_stats,
    process_clusters,
    save_associations,
)
from swe.association.converters import association_to_catalog

__all__ = [
    "AssociationResult",
    "associate_picks",
    "association_to_catalog",
    "build_pick_data",
    "build_pick_dict",
    "cluster_picks",
    "compute_cluster_quality",
    "get_cluster_stats",
    "process_clusters",
    "save_associations",
]
