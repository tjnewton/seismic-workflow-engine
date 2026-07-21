"""Matched-filter detection and waveform stacking."""

from __future__ import annotations

from swe.detection.catalog import (
    LFEResult,
    cull_party_dates,
    find_lfes,
    get_day_detections,
    get_threshold_detections,
    party_snrs,
    sort_party,
)
from swe.detection.evaluation import (
    MatchResult,
    dedup_times_by_interval,
    match_detections,
)
from swe.detection.matched_filter import (
    MarkerTemplateResult,
    cull_detections,
    detect_signals,
    detections_from_stacks,
    get_detections,
    make_templates,
    markers_to_template,
)
from swe.detection.stacking import (
    Stack,
    StackResult,
    median_stack,
    nth_root_stack,
    selective_stack,
    stack_template_detections,
    stack_waveforms,
    time_shift,
    trace_snr,
)

__all__ = [
    "LFEResult",
    "MarkerTemplateResult",
    "MatchResult",
    "Stack",
    "StackResult",
    "cull_detections",
    "cull_party_dates",
    "dedup_times_by_interval",
    "detect_signals",
    "detections_from_stacks",
    "find_lfes",
    "get_day_detections",
    "get_detections",
    "get_threshold_detections",
    "make_templates",
    "markers_to_template",
    "match_detections",
    "median_stack",
    "nth_root_stack",
    "party_snrs",
    "selective_stack",
    "sort_party",
    "stack_template_detections",
    "stack_waveforms",
    "time_shift",
    "trace_snr",
]
