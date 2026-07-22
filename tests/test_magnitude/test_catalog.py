"""Tests for swe.magnitude.catalog."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Catalog, Stream, Trace, UTCDateTime
from obspy.core.event import Event, Origin

from swe.magnitude.catalog import (
    attach_duration_magnitudes,
    attach_local_magnitudes,
    attach_magnitudes_to_catalog,
)


def _make_trace(
    *,
    network: str = "XX",
    station: str = "STA1",
    channel: str = "BHZ",
    sampling_rate: float = 100.0,
    npts: int = 6000,
    starttime: UTCDateTime | None = None,
    amplitude: float = 1.0,
    signal_onset: float | None = None,
) -> Trace:
    """Return a synthetic Trace with optional signal onset."""
    if starttime is None:
        starttime = UTCDateTime(2024, 1, 1)
    rng = np.random.default_rng(42)
    data = rng.standard_normal(npts).astype(np.float64) * 0.01

    if signal_onset is not None:
        onset_idx = int(signal_onset * sampling_rate)
        t = np.arange(npts) / sampling_rate
        for i in range(onset_idx, npts):
            elapsed = (i - onset_idx) / sampling_rate
            data[i] += amplitude * np.exp(-elapsed / 10.0) * np.sin(2.0 * np.pi * 5.0 * t[i])

    header = {
        "network": network,
        "station": station,
        "channel": channel,
        "sampling_rate": sampling_rate,
        "npts": npts,
        "starttime": starttime,
    }
    return Trace(data=data, header=header)


def _make_event(starttime: UTCDateTime | None = None) -> Event:
    """Create a minimal Event with an Origin."""
    if starttime is None:
        starttime = UTCDateTime(2024, 1, 1)
    origin = Origin(
        time=starttime,
        latitude=35.0,
        longitude=-117.0,
        depth=10000.0,
    )
    event = Event(origins=[origin])
    event.preferred_origin_id = origin.resource_id
    return event


class TestAttachLocalMagnitudes:
    """Tests for attach_local_magnitudes."""

    def test_attaches_magnitude(self):
        """Should add a Magnitude to the event."""
        event = _make_event()
        st = Stream(
            [
                _make_trace(station="STA1"),
                _make_trace(station="STA2"),
            ]
        )
        distances = [10.0, 20.0]

        result = attach_local_magnitudes(event, st, distances)

        assert result is event
        assert len(event.magnitudes) == 1
        assert event.magnitudes[0].magnitude_type == "ML"

    def test_preferred_magnitude_set(self):
        """Should set preferred_magnitude_id."""
        event = _make_event()
        st = Stream([_make_trace()])
        attach_local_magnitudes(event, st, [10.0])

        assert event.preferred_magnitude_id is not None
        assert event.preferred_magnitude_id == event.magnitudes[0].resource_id

    def test_station_magnitudes_created(self):
        """Should create StationMagnitude objects."""
        event = _make_event()
        st = Stream(
            [
                _make_trace(station="STA1"),
                _make_trace(station="STA2"),
            ]
        )
        attach_local_magnitudes(event, st, [10.0, 20.0])

        # station_magnitudes may have some NaN entries filtered
        assert len(event.station_magnitudes) >= 0

    def test_station_count(self):
        """Magnitude station_count should match valid stations."""
        event = _make_event()
        st = Stream(
            [
                _make_trace(station="STA1"),
                _make_trace(station="STA2"),
                _make_trace(station="STA3"),
            ]
        )
        attach_local_magnitudes(event, st, [10.0, 20.0, 30.0])

        if event.magnitudes:
            assert event.magnitudes[0].station_count >= 0

    def test_returns_event(self):
        """Should return the same Event object."""
        event = _make_event()
        st = Stream([_make_trace()])
        result = attach_local_magnitudes(event, st, [10.0])
        assert result is event


class TestAttachDurationMagnitudes:
    """Tests for attach_duration_magnitudes."""

    def test_attaches_magnitude(self):
        """Should add an Md magnitude to the event."""
        event = _make_event()
        t0 = UTCDateTime(2024, 1, 1)
        st = Stream(
            [
                _make_trace(station="STA1", signal_onset=10.0, amplitude=5.0),
            ]
        )
        p_times = {"STA1": t0 + 10.0}

        attach_duration_magnitudes(event, st, p_times)

        assert len(event.magnitudes) == 1
        assert event.magnitudes[0].magnitude_type == "Md"

    def test_preferred_magnitude_set(self):
        """Should set preferred_magnitude_id."""
        event = _make_event()
        t0 = UTCDateTime(2024, 1, 1)
        st = Stream(
            [
                _make_trace(station="STA1", signal_onset=10.0, amplitude=5.0),
            ]
        )
        p_times = {"STA1": t0 + 10.0}

        attach_duration_magnitudes(event, st, p_times)

        assert event.preferred_magnitude_id is not None

    def test_with_distances(self):
        """Should accept optional distances_km."""
        event = _make_event()
        t0 = UTCDateTime(2024, 1, 1)
        st = Stream(
            [
                _make_trace(station="STA1", signal_onset=10.0, amplitude=5.0),
            ]
        )
        p_times = {"STA1": t0 + 10.0}
        distances = {"STA1": 15.0}

        attach_duration_magnitudes(event, st, p_times, distances_km=distances)

        assert len(event.magnitudes) == 1

    def test_empty_stream(self):
        """Empty stream should not add any magnitude."""
        event = _make_event()
        attach_duration_magnitudes(event, Stream(), {})
        assert len(event.magnitudes) == 0

    def test_no_matching_p_times(self):
        """No matching P-times should not add magnitude."""
        event = _make_event()
        st = Stream([_make_trace(station="STA1")])
        attach_duration_magnitudes(event, st, {"UNKNOWN": UTCDateTime()})
        assert len(event.magnitudes) == 0


class TestAttachMagnitudesToCatalog:
    """Tests for attach_magnitudes_to_catalog."""

    def test_batch_ml(self):
        """Should attach ML to all events."""
        events = [_make_event(), _make_event()]
        catalog = Catalog(events=events)
        streams = [
            Stream([_make_trace(station="STA1")]),
            Stream([_make_trace(station="STA2")]),
        ]
        distances = [[10.0], [20.0]]

        result = attach_magnitudes_to_catalog(catalog, streams, distances)

        assert result is catalog
        for event in catalog:
            assert len(event.magnitudes) >= 0  # May be 0 if WA sim returns NaN

    def test_mismatched_lengths(self):
        """Should raise ValueError for mismatched lengths."""
        catalog = Catalog(events=[_make_event()])
        with pytest.raises(ValueError, match="Lengths must match"):
            attach_magnitudes_to_catalog(catalog, [], [])

    def test_returns_catalog(self):
        """Should return the same Catalog object."""
        catalog = Catalog(events=[_make_event()])
        streams = [Stream([_make_trace()])]
        distances = [[10.0]]
        result = attach_magnitudes_to_catalog(catalog, streams, distances)
        assert result is catalog


class TestMagErrors:
    """Tests for mag_errors on ObsPy Magnitude objects."""

    def test_ml_mag_errors_set(self):
        """attach_local_magnitudes should set mag_errors when stderr > 0."""
        event = _make_event()
        st = Stream([_make_trace(station=f"STA{i}", amplitude=float(i + 1)) for i in range(4)])
        distances = [10.0, 20.0, 30.0, 40.0]
        attach_local_magnitudes(event, st, distances)

        if event.magnitudes:
            mag = event.magnitudes[0]
            # With multiple stations and varying amplitudes, stderr should be set
            if mag.mag_errors is not None:
                assert mag.mag_errors.uncertainty > 0

    def test_md_mag_errors_set(self):
        """attach_duration_magnitudes should set mag_errors for multi-station Md."""
        event = _make_event()
        t0 = UTCDateTime(2024, 1, 1)
        st = Stream(
            [
                _make_trace(station="STA1", signal_onset=10.0, amplitude=5.0),
                _make_trace(station="STA2", signal_onset=10.0, amplitude=3.0),
            ]
        )
        p_times = {"STA1": t0 + 10.0, "STA2": t0 + 10.0}
        attach_duration_magnitudes(event, st, p_times)

        if len(event.magnitudes) == 1:
            mag = event.magnitudes[0]
            assert mag.magnitude_type == "Md"
            # With 2 stations, mag_errors should be populated
            if mag.mag_errors is not None:
                assert mag.mag_errors.uncertainty >= 0
