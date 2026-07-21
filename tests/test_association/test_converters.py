"""Tests for swe.association.converters."""

import numpy as np
import pytest
from obspy.core.event import Catalog

from swe.association.clustering import AssociationResult
from swe.association.converters import association_to_catalog


class TestAssociationToCatalog:
    """Tests for association_to_catalog()."""

    @staticmethod
    def _make_result(labels, n_stations=3):
        """Build a synthetic AssociationResult with matplotlib-date pick_times."""
        from matplotlib.dates import date2num
        from obspy import UTCDateTime

        base = UTCDateTime("2024-01-01T12:00:00")
        n_picks = len(labels)
        station_distance = {f"S{i}": i for i in range(n_stations)}
        reverse_station_distance = {i: f"S{i}" for i in range(n_stations)}
        pick_station_order = [i % n_stations for i in range(n_picks)]
        pick_times = [
            date2num((base + i * 0.1).datetime) for i in range(n_picks)
        ]
        return AssociationResult(
            station_distance=station_distance,
            reverse_station_distance=reverse_station_distance,
            pick_station_order=pick_station_order,
            pick_times=pick_times,
            labels=np.array(labels),
        )

    def test_returns_catalog(self):
        """Should return an ObsPy Catalog."""
        result = self._make_result([0, 0, 0, 1, 1, 1])
        catalog = association_to_catalog(result)
        assert isinstance(catalog, Catalog)

    def test_correct_event_count(self):
        """Two non-noise clusters should produce two events."""
        result = self._make_result([0, 0, 0, 1, 1, 1])
        catalog = association_to_catalog(result)
        assert len(catalog.events) == 2

    def test_noise_excluded(self):
        """Noise cluster (-1) should not appear as an event."""
        result = self._make_result([-1, -1, 0, 0, 0])
        catalog = association_to_catalog(result)
        assert len(catalog.events) == 1

    def test_all_noise(self):
        """All-noise labels should produce an empty catalog."""
        result = self._make_result([-1, -1, -1])
        catalog = association_to_catalog(result)
        assert len(catalog.events) == 0

    def test_pick_count_per_event(self):
        """Each event should have the correct number of picks."""
        result = self._make_result([0, 0, 0, 1, 1])
        catalog = association_to_catalog(result)
        pick_counts = sorted(len(e.picks) for e in catalog.events)
        assert pick_counts == [2, 3]

    def test_network_and_channel_propagated(self):
        """Network and channel should appear on every pick's WaveformStreamID."""
        result = self._make_result([0, 0, 0])
        catalog = association_to_catalog(result, network="CI", channel="BHZ")
        for event in catalog.events:
            for pick in event.picks:
                assert pick.waveform_id.network_code == "CI"
                assert pick.waveform_id.channel_code == "BHZ"

    def test_phase_hint_propagated(self):
        """Phase hint should appear on every pick."""
        result = self._make_result([0, 0, 0])
        catalog = association_to_catalog(result, phase_hint="S")
        for event in catalog.events:
            for pick in event.picks:
                assert pick.phase_hint == "S"

    def test_default_origin_coordinates(self):
        """Origin should use the default lat/lon/depth."""
        result = self._make_result([0, 0, 0])
        catalog = association_to_catalog(
            result,
            default_latitude=35.77,
            default_longitude=-117.60,
            default_depth=5000.0,
        )
        origin = catalog.events[0].origins[0]
        assert origin.latitude == pytest.approx(35.77)
        assert origin.longitude == pytest.approx(-117.60)
        assert origin.depth == pytest.approx(5000.0)

    def test_empty_result(self):
        """Empty labels should produce an empty catalog."""
        result = AssociationResult(
            station_distance={},
            reverse_station_distance={},
            pick_station_order=[],
            pick_times=[],
            labels=np.array([], dtype=int),
        )
        catalog = association_to_catalog(result)
        assert len(catalog.events) == 0
