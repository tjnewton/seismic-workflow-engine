"""Shared fixtures for swe.detection tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime


def _make_mock_detection(
    detect_time,
    detect_val=0.5,
    threshold=8.0,
    threshold_input=1.0,
    threshold_type="MAD",
    template_name="test",
):
    """Create a mock EQcorrscan Detection."""
    det = MagicMock()
    det.detect_time = detect_time
    det.detect_val = detect_val
    det.threshold = threshold
    det.threshold_input = threshold_input
    det.threshold_type = threshold_type
    det.template_name = template_name
    return det


def _make_mock_family(template=None, detections=None, catalog=None):
    """Create a mock EQcorrscan Family."""
    family = MagicMock()
    family.template = template or MagicMock()
    family.detections = detections or []
    family.catalog = catalog or MagicMock()
    family.__len__ = lambda self: len(self.detections)
    return family


def _make_mock_party(detections=None, family=None):
    """Create a mock Party with a single Family."""
    if family is None:
        family = _make_mock_family(detections=detections or [])
    party = MagicMock()
    party.families = [family]
    return party


@pytest.fixture()
def synthetic_day_stream():
    """A 60-second synthetic Stream at 100 Hz on YG.STA1..BHZ."""
    npts = 6001
    tr = Trace(
        data=np.random.default_rng(42).standard_normal(npts).astype(np.float64),
    )
    tr.stats.network = "YG"
    tr.stats.station = "STA1"
    tr.stats.channel = "BHZ"
    tr.stats.sampling_rate = 100.0
    tr.stats.starttime = UTCDateTime("2020-01-15T00:00:00")
    return Stream([tr])
