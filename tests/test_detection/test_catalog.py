"""Tests for swe.detection.catalog."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from obspy import UTCDateTime

from swe.detection.catalog import (
    cull_party_dates,
    get_day_detections,
    get_threshold_detections,
)


def _make_mock_detection(detect_time, detect_val=0.5, threshold=8.0,
                          threshold_input=1.0, threshold_type="MAD",
                          template_name="test"):
    """Create a mock EQcorrscan Detection."""
    det = MagicMock()
    det.detect_time = detect_time
    det.detect_val = detect_val
    det.threshold = threshold
    det.threshold_input = threshold_input
    det.threshold_type = threshold_type
    det.template_name = template_name
    return det


def _make_mock_party(detections):
    """Create a mock Party with a single Family."""
    family = MagicMock()
    family.detections = detections
    family.template = MagicMock()
    party = MagicMock()
    party.families = [family]
    return party


class TestGetDayDetections:
    def test_filters_to_single_day(self):
        d1 = _make_mock_detection(UTCDateTime("2020-01-15T10:00:00"))
        d2 = _make_mock_detection(UTCDateTime("2020-01-15T22:00:00"))
        d3 = _make_mock_detection(UTCDateTime("2020-01-16T01:00:00"))
        party = _make_mock_party([d1, d2, d3])

        with patch("swe.detection.catalog.Party") as MockParty, \
             patch("swe.detection.catalog.Family") as MockFamily:
            MockFamily.return_value = MagicMock()
            MockParty.return_value = MagicMock()

            result = get_day_detections(UTCDateTime("2020-01-15"), party)
            # Check Family was called with 2 detections
            call_args = MockFamily.call_args
            assert len(call_args[0][1]) == 2

    def test_no_detections_on_day(self):
        d1 = _make_mock_detection(UTCDateTime("2020-01-15T10:00:00"))
        party = _make_mock_party([d1])

        with patch("swe.detection.catalog.Party") as MockParty, \
             patch("swe.detection.catalog.Family") as MockFamily:
            MockFamily.return_value = MagicMock()
            MockParty.return_value = MagicMock()

            result = get_day_detections(UTCDateTime("2020-02-01"), party)
            call_args = MockFamily.call_args
            assert len(call_args[0][1]) == 0


class TestCullPartyDates:
    def test_keeps_matching_dates(self):
        d1 = _make_mock_detection(UTCDateTime("2020-01-15T10:00:00"))
        d2 = _make_mock_detection(UTCDateTime("2020-01-16T10:00:00"))
        d3 = _make_mock_detection(UTCDateTime("2020-01-17T10:00:00"))
        party = _make_mock_party([d1, d2, d3])

        keep_dates = [
            UTCDateTime("2020-01-15"),
            UTCDateTime("2020-01-17"),
        ]

        with patch("swe.detection.catalog.Party") as MockParty, \
             patch("swe.detection.catalog.Family") as MockFamily:
            MockFamily.return_value = MagicMock()
            MockParty.return_value = MagicMock()

            result = cull_party_dates(keep_dates, party)
            call_args = MockFamily.call_args
            assert len(call_args[0][1]) == 2


class TestGetThresholdDetections:
    def test_filters_above_threshold(self):
        # detect_val=10.0, threshold=8.0, threshold_input=1.0
        # threshold_value = 8.0 / 1.0 * threshold_arg
        d1 = _make_mock_detection(
            UTCDateTime("2020-01-15T10:00:00"),
            detect_val=10.0, threshold=8.0, threshold_input=1.0,
        )
        d2 = _make_mock_detection(
            UTCDateTime("2020-01-15T11:00:00"),
            detect_val=5.0, threshold=8.0, threshold_input=1.0,
        )
        d3 = _make_mock_detection(
            UTCDateTime("2020-01-15T12:00:00"),
            detect_val=20.0, threshold=8.0, threshold_input=1.0,
        )
        party = _make_mock_party([d1, d2, d3])

        with patch("swe.detection.catalog.Party") as MockParty, \
             patch("swe.detection.catalog.Family") as MockFamily:
            MockFamily.return_value = MagicMock()
            MockParty.return_value = MagicMock()

            # threshold_value = 8.0 * 1.0 = 8.0
            # d1: |10.0| >= 8.0 -> keep
            # d2: |5.0| >= 8.0 -> reject
            # d3: |20.0| >= 8.0 -> keep
            result = get_threshold_detections(party, threshold=1.0)
            call_args = MockFamily.call_args
            assert len(call_args[0][1]) == 2

    def test_high_threshold_removes_all(self):
        d1 = _make_mock_detection(
            UTCDateTime("2020-01-15T10:00:00"),
            detect_val=5.0, threshold=8.0, threshold_input=1.0,
        )
        party = _make_mock_party([d1])

        with patch("swe.detection.catalog.Party") as MockParty, \
             patch("swe.detection.catalog.Family") as MockFamily:
            MockFamily.return_value = MagicMock()
            MockParty.return_value = MagicMock()

            # threshold_value = 8.0 * 100.0 = 800.0
            result = get_threshold_detections(party, threshold=100.0)
            call_args = MockFamily.call_args
            assert len(call_args[0][1]) == 0
