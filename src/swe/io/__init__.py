"""Data I/O and format parsers."""

from __future__ import annotations

from swe.io.fdsn import get_response_files, get_waveforms
from swe.io.quakeml import read_quakeml, write_quakeml
from swe.io.snuffler import (
    parse_all_markers,
    parse_phase_picks,
    party_to_markers,
    write_events_to_markers,
)

__all__ = [
    "get_response_files",
    "get_waveforms",
    "parse_all_markers",
    "parse_phase_picks",
    "party_to_markers",
    "read_quakeml",
    "write_events_to_markers",
    "write_quakeml",
]
