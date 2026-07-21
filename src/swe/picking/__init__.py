"""Seismic phase picking: sliding-window extraction and deep-learning pickers."""

from __future__ import annotations

from swe.picks import SWEPick, from_phase_arrivals, to_station_picks
from swe.picking.deep_learning import PhaseArrival, pick_stream, predict_phase_arrivals
from swe.picking.windows import (
    DeduplicatedPickResult,
    PickingWindowResult,
    ThresholdPickResult,
    find_picking_windows,
    get_threshold_picks,
    remove_duplicate_picks,
)

__all__ = [
    "DeduplicatedPickResult",
    "PhaseArrival",
    "PickingWindowResult",
    "SWEPick",
    "ThresholdPickResult",
    "find_picking_windows",
    "from_phase_arrivals",
    "get_threshold_picks",
    "pick_stream",
    "predict_phase_arrivals",
    "remove_duplicate_picks",
    "to_station_picks",
]
