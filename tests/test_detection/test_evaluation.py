"""Tests for swe.detection.evaluation (detection-to-catalog metrics)."""

from __future__ import annotations

import pytest
from obspy import UTCDateTime

from swe.detection.evaluation import (
    MatchResult,
    dedup_times_by_interval,
    match_detections,
)


def _t(offset: float) -> UTCDateTime:
    """Catalog/detection time at ``offset`` seconds past a fixed epoch."""
    return UTCDateTime("2015-02-17T00:00:00") + offset


class TestMatchDetections:
    def test_perfect_match(self):
        cat = [_t(0), _t(10), _t(20)]
        det = [_t(0.1), _t(9.9), _t(20.2)]
        r = match_detections(det, cat, tolerance=1.0)
        assert r.true_positives == 3
        assert r.false_positives == 0
        assert r.missed == 0
        assert r.precision == 1.0
        assert r.recall == 1.0

    def test_counts_sum_consistently(self):
        cat = [_t(0), _t(10), _t(20), _t(30)]
        det = [_t(0.1), _t(20.1), _t(100.0)]  # 2 TP, 1 FP, 2 missed
        r = match_detections(det, cat, tolerance=1.0)
        assert r.true_positives == 2
        assert r.false_positives == 1
        assert r.missed == 2
        # Consistency: tp + fp == n_detections; tp + missed == n_catalog
        assert r.true_positives + r.false_positives == r.n_detections
        assert r.true_positives + r.missed == r.n_catalog

    def test_no_double_counting_one_detection_two_catalog(self):
        """A single detection near two catalog events counts once, not twice."""
        cat = [_t(0.0), _t(0.4)]  # both within tolerance of one detection
        det = [_t(0.2)]
        r = match_detections(det, cat, tolerance=1.0)
        assert r.true_positives == 1  # not 2
        assert r.missed == 1
        assert r.false_positives == 0

    def test_no_double_counting_two_detections_one_catalog(self):
        cat = [_t(0.0)]
        det = [_t(-0.1), _t(0.2)]  # both near the one catalog event
        r = match_detections(det, cat, tolerance=1.0)
        assert r.true_positives == 1
        assert r.false_positives == 1  # the unmatched detection
        assert r.missed == 0

    def test_greedy_assigns_nearest(self):
        """The nearer detection should claim the catalog event."""
        cat = [_t(0.0)]
        det = [_t(0.5), _t(0.05)]
        r = match_detections(det, cat, tolerance=1.0)
        assert r.true_positives == 1
        # The matched detection is index 1 (the 0.05 s one), residual ~0.05
        (di, ci, resid), = r.matches
        assert di == 1
        assert ci == 0
        assert abs(resid - 0.05) < 1e-6

    def test_tolerance_excludes(self):
        cat = [_t(0.0)]
        det = [_t(5.0)]
        r = match_detections(det, cat, tolerance=1.0)
        assert r.true_positives == 0
        assert r.false_positives == 1
        assert r.missed == 1

    def test_empty_inputs(self):
        assert match_detections([], [], tolerance=1.0).recall == 1.0
        r = match_detections([], [_t(0)], tolerance=1.0)
        assert r.recall == 0.0 and r.missed == 1
        r = match_detections([_t(0)], [], tolerance=1.0)
        assert r.precision == 0.0 and r.false_positives == 1

    def test_invalid_tolerance(self):
        with pytest.raises(ValueError):
            match_detections([_t(0)], [_t(0)], tolerance=0.0)

    def test_returns_matchresult(self):
        r = match_detections([_t(0)], [_t(0)], tolerance=1.0)
        assert isinstance(r, MatchResult)
        assert r.residuals == [0.0]


class TestDedupTimesByInterval:
    def test_collapses_close_times(self):
        times = [_t(0), _t(2), _t(4), _t(20)]  # first three within 8 s
        kept = dedup_times_by_interval(times, min_interval=8.0)
        assert kept == [_t(0), _t(20)]

    def test_keeps_well_separated(self):
        times = [_t(0), _t(10), _t(20)]
        kept = dedup_times_by_interval(times, min_interval=8.0)
        assert kept == times

    def test_unsorted_input(self):
        times = [_t(20), _t(0), _t(4)]
        kept = dedup_times_by_interval(times, min_interval=8.0)
        assert kept == [_t(0), _t(20)]

    def test_boundary_exactly_interval_kept(self):
        times = [_t(0), _t(8)]  # exactly 8 s apart -> both kept (>=)
        kept = dedup_times_by_interval(times, min_interval=8.0)
        assert kept == times

    def test_empty(self):
        assert dedup_times_by_interval([], min_interval=8.0) == []

    def test_invalid_interval(self):
        with pytest.raises(ValueError):
            dedup_times_by_interval([_t(0)], min_interval=0.0)

    def test_resolvable_cap_matches_review_numbers(self):
        """Sanity: dense bursts within trig_int reduce the matchable count."""
        # 5 events, 4 of them within 8 s of the prior -> only 2 resolvable
        times = [_t(0), _t(3), _t(6), _t(9), _t(50)]
        kept = dedup_times_by_interval(times, min_interval=8.0)
        # 0 kept; 3 (<8 from 0) drop; 9 (>=8 from 0) kept; 50 kept
        assert kept == [_t(0), _t(9), _t(50)]
