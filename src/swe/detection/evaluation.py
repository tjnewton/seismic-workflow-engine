"""Evaluation of matched-filter detections against a reference catalog.

Provides consistent, one-to-one matching of detection times to catalog times
(so true positives, false positives, and missed events sum correctly) and a
helper to collapse a catalog to the subset that a given trigger interval can
actually resolve.  Hand-rolled ``any(abs(dt - ct) < tol)`` matching
double-counts (one detection can satisfy several nearby catalog entries and
vice versa); the routines here avoid that.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import NamedTuple

from obspy import UTCDateTime

logger = logging.getLogger(__name__)

__all__ = [
    "MatchResult",
    "dedup_times_by_interval",
    "match_detections",
]


class MatchResult(NamedTuple):
    """Result of a one-to-one detection-to-catalog match."""

    matches: list[tuple[int, int, float]]
    """``(detection_index, catalog_index, signed_residual_s)`` for each pair."""
    n_detections: int
    """Number of detection times supplied."""
    n_catalog: int
    """Number of catalog times supplied."""
    true_positives: int
    """Detections matched one-to-one to a catalog event (== ``len(matches)``)."""
    false_positives: int
    """Detections with no catalog match (``n_detections - true_positives``)."""
    missed: int
    """Catalog events with no detection (``n_catalog - true_positives``)."""

    @property
    def precision(self) -> float:
        """True positives / detections (1.0 when there are no detections)."""
        return self.true_positives / self.n_detections if self.n_detections else 1.0

    @property
    def recall(self) -> float:
        """True positives / catalog events (1.0 when the catalog is empty)."""
        return self.true_positives / self.n_catalog if self.n_catalog else 1.0

    @property
    def residuals(self) -> list[float]:
        """Signed detection-minus-catalog residuals (s) for matched pairs."""
        return [m[2] for m in self.matches]


def match_detections(
    detection_times: Sequence[UTCDateTime],
    catalog_times: Sequence[UTCDateTime],
    *,
    tolerance: float = 5.0,
) -> MatchResult:
    """Match detections to catalog events one-to-one by nearest time.

    Each detection matches at most one catalog event and each catalog event is
    matched by at most one detection.  Pairs are assigned greedily in order of
    increasing absolute time difference, which yields a stable, consistent set
    of true positives, false positives, and missed events.

    Parameters
    ----------
    detection_times : sequence of obspy.UTCDateTime
        Detection origin/trigger times.
    catalog_times : sequence of obspy.UTCDateTime
        Reference (ground-truth) catalog times.
    tolerance : float
        Maximum absolute time difference (seconds) for a valid match.

    Returns
    -------
    MatchResult
        Matched pairs and consistent true-positive / false-positive / missed
        counts.  ``precision`` and ``recall`` are exposed as properties.
    """
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")

    n_det = len(detection_times)
    n_cat = len(catalog_times)

    # Build all candidate pairs within tolerance, then assign greedily nearest.
    candidates: list[tuple[float, int, int, float]] = []
    for i, dt in enumerate(detection_times):
        for j, ct in enumerate(catalog_times):
            residual = float(dt - ct)
            if abs(residual) <= tolerance:
                candidates.append((abs(residual), i, j, residual))
    candidates.sort(key=lambda c: c[0])

    used_det: set[int] = set()
    used_cat: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for _, i, j, residual in candidates:
        if i in used_det or j in used_cat:
            continue
        used_det.add(i)
        used_cat.add(j)
        matches.append((i, j, residual))

    matches.sort(key=lambda m: m[0])
    tp = len(matches)
    return MatchResult(
        matches=matches,
        n_detections=n_det,
        n_catalog=n_cat,
        true_positives=tp,
        false_positives=n_det - tp,
        missed=n_cat - tp,
    )


def dedup_times_by_interval(
    times: Sequence[UTCDateTime],
    *,
    min_interval: float,
) -> list[UTCDateTime]:
    """Collapse times separated by less than ``min_interval`` seconds.

    A matched-filter detector with trigger interval ``trig_int`` cannot emit two
    detections within ``trig_int`` of each other, so catalog events closer than
    that are not independently resolvable.  Deduplicating the catalog with
    ``min_interval = trig_int`` gives the subset against which recall can be
    fairly measured.  Within each cluster the earliest time is kept.

    Parameters
    ----------
    times : sequence of obspy.UTCDateTime
        Times to deduplicate (order-independent; sorted internally).
    min_interval : float
        Minimum separation (seconds) between retained times.

    Returns
    -------
    list of obspy.UTCDateTime
        Retained times in chronological order.
    """
    if min_interval <= 0:
        raise ValueError(f"min_interval must be positive, got {min_interval}")
    if not times:
        return []

    ordered = sorted(times)
    kept: list[UTCDateTime] = [ordered[0]]
    for t in ordered[1:]:
        if (t - kept[-1]) >= min_interval:
            kept.append(t)
    return kept
