"""Visualization utilities for seismic data and analysis results."""

from __future__ import annotations

from swe.viz.statistics import (
    plot_date_distribution,
    plot_distribution,
    plot_moment_distribution,
)
from swe.viz.maps import (
    plot_cross_sections,
    plot_detection_timeline,
    plot_nll_scatter,
    plot_station_event_map,
)
from swe.viz.waveforms import (
    plot_event_picks,
    plot_multi_spectra,
    plot_party_detections,
    plot_spectra,
    plot_spectrogram,
    plot_stack,
    plot_stream,
    plot_stream_absolute,
    plot_template_and_stack,
    plot_time_series,
    plot_time_series_and_spectrogram,
)

__all__ = [
    # maps
    "plot_station_event_map",
    "plot_cross_sections",
    "plot_detection_timeline",
    "plot_nll_scatter",
    # waveforms
    "plot_stream",
    "plot_time_series_and_spectrogram",
    "plot_time_series",
    "plot_party_detections",
    "plot_stack",
    "plot_template_and_stack",
    "plot_stream_absolute",
    "plot_event_picks",
    "plot_spectrogram",
    "plot_spectra",
    "plot_multi_spectra",
    # statistics
    "plot_distribution",
    "plot_moment_distribution",
    "plot_date_distribution",
]
