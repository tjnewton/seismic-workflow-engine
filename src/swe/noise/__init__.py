"""Seismic noise window detection, validation, and analysis."""

from __future__ import annotations

from swe.noise.analysis import (
    analyze_noise_windows,
    plot_noise_station_histogram,
    plot_noise_time_histogram,
)
from swe.noise.detection import (
    NoiseWindowMeta,
    NoiseWindowResult,
    find_noise_windows,
)
from swe.noise.validation import explore_noise_windows, verify_noise_windows

__all__ = [
    "NoiseWindowMeta",
    "NoiseWindowResult",
    "analyze_noise_windows",
    "explore_noise_windows",
    "find_noise_windows",
    "plot_noise_station_histogram",
    "plot_noise_time_histogram",
    "verify_noise_windows",
]
