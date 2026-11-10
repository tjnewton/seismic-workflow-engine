"""Topographic and raster data processing, and terrain morphometry."""

from __future__ import annotations

from swe.topo.morphometry import (
    PlaneFit,
    curvature,
    faceted_template,
    grid_spacing,
    local_plane_fit,
    slope_aspect,
    template_correlation,
)
from swe.topo.pointcloud import PointCloud, load_point_cloud
from swe.topo.raster import (
    ElevationGrid,
    elevations_from_raster,
    grid_from_raster,
    terrain_correction,
)

__all__ = [
    "ElevationGrid",
    "PlaneFit",
    "PointCloud",
    "curvature",
    "elevations_from_raster",
    "faceted_template",
    "grid_from_raster",
    "grid_spacing",
    "load_point_cloud",
    "local_plane_fit",
    "slope_aspect",
    "template_correlation",
    "terrain_correction",
]
