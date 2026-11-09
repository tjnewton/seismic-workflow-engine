"""Data I/O and format parsers."""

from __future__ import annotations

from swe.io.fdsn import (
    DEFAULT_EXCLUDED_NETWORKS,
    catalog_to_dataframe,
    get_events,
    get_response_files,
    get_stations_in_region,
    get_waveforms,
    inventory_to_dataframe,
    load_response,
)
from swe.io.quakeml import read_quakeml, write_quakeml
from swe.io.snuffler import (
    parse_all_markers,
    parse_phase_picks,
    party_to_markers,
    write_events_to_markers,
)

__all__ = [
    "DEFAULT_EXCLUDED_NETWORKS",
    "catalog_to_dataframe",
    "get_events",
    "get_response_files",
    "get_stations_in_region",
    "get_waveforms",
    "inventory_to_dataframe",
    "load_response",
    "parse_all_markers",
    "parse_phase_picks",
    "party_to_markers",
    "read_quakeml",
    "write_events_to_markers",
    "write_quakeml",
]
