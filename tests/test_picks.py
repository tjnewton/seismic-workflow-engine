"""Tests for swe.picks - unified pick format and converters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from obspy import UTCDateTime
from obspy.core.event import Pick, WaveformStreamID

from swe.picks import (
    SWEPick,
    from_obspy_picks,
    from_phase_arrivals,
    picks_from_dataframe,
    picks_to_dataframe,
    to_obspy_picks,
    to_station_picks,
)


# ---------------------------------------------------------------------------
# SWEPick basics
# ---------------------------------------------------------------------------


class TestSWEPick:
    """Basic SWEPick construction and defaults."""

    def test_create_minimal(self):
        t = UTCDateTime("2024-01-01T00:00:10")
        p = SWEPick(station="STA1", phase="P", time=t, confidence=0.9)
        assert p.station == "STA1"
        assert p.phase == "P"
        assert p.time == t
        assert p.confidence == 0.9
        assert p.network == ""
        assert p.channel == ""
        assert p.method == ""

    def test_create_full(self):
        t = UTCDateTime("2024-01-01T00:00:10")
        p = SWEPick(
            station="STA1",
            phase="S",
            time=t,
            confidence=0.7,
            network="XX",
            channel="HHZ",
            method="PhaseNet",
        )
        assert p.network == "XX"
        assert p.method == "PhaseNet"

    def test_is_namedtuple(self):
        t = UTCDateTime("2024-01-01T00:00:10")
        p = SWEPick(station="STA1", phase="P", time=t, confidence=0.9)
        assert isinstance(p, tuple)
        assert len(p) == 8


# ---------------------------------------------------------------------------
# from_phase_arrivals
# ---------------------------------------------------------------------------


class TestFromPhaseArrivals:
    """Converting PhaseArrival -> SWEPick."""

    def test_basic_conversion(self):
        from swe.picking.deep_learning import PhaseArrival

        t = UTCDateTime("2024-01-01T00:00:10")
        arrivals = [
            PhaseArrival(station="STA1", phase="P", time=t, confidence=0.9),
            PhaseArrival(station="STA1", phase="S", time=t + 5, confidence=0.7),
        ]
        picks = from_phase_arrivals(arrivals)
        assert len(picks) == 2
        assert all(isinstance(p, SWEPick) for p in picks)
        assert picks[0].phase == "P"
        assert picks[1].phase == "S"
        assert picks[0].confidence == 0.9

    def test_empty_list(self):
        assert from_phase_arrivals([]) == []


# ---------------------------------------------------------------------------
# to_station_picks
# ---------------------------------------------------------------------------


class TestToStationPicks:
    """Converting SWEPick list -> dict format for association."""

    def test_basic_conversion(self):
        t0 = UTCDateTime("2024-01-01")
        picks = [
            SWEPick("STA1", "P", t0 + 10, 0.9),
            SWEPick("STA1", "S", t0 + 15, 0.7),
            SWEPick("STA2", "P", t0 + 11, 0.8),
        ]
        station_picks, pick_phases = to_station_picks(picks)

        assert set(station_picks.keys()) == {"STA1", "STA2"}
        assert len(station_picks["STA1"]) == 2
        assert len(station_picks["STA2"]) == 1
        assert pick_phases["STA1"] == ["P", "S"]
        assert pick_phases["STA2"] == ["P"]

    def test_sorted_by_time(self):
        t0 = UTCDateTime("2024-01-01")
        picks = [
            SWEPick("STA1", "S", t0 + 20, 0.7),
            SWEPick("STA1", "P", t0 + 10, 0.9),
        ]
        station_picks, pick_phases = to_station_picks(picks)
        assert station_picks["STA1"][0] < station_picks["STA1"][1]
        assert pick_phases["STA1"] == ["P", "S"]

    def test_empty_list(self):
        sp, pp = to_station_picks([])
        assert sp == {}
        assert pp == {}


# ---------------------------------------------------------------------------
# ObsPy round-trip
# ---------------------------------------------------------------------------


class TestObspyRoundTrip:
    """SWEPick ↔ ObsPy Pick conversion."""

    def test_to_obspy_picks(self):
        t = UTCDateTime("2024-01-01T00:00:10")
        picks = [
            SWEPick("STA1", "P", t, 0.9, network="XX", channel="HHZ"),
            SWEPick("STA2", "S", t + 5, 0.7),
        ]
        obspy_picks = to_obspy_picks(picks, network="YY")
        assert len(obspy_picks) == 2
        assert obspy_picks[0].phase_hint == "P"
        assert obspy_picks[0].waveform_id.network_code == "XX"
        # Fallback network used when pick has no network
        assert obspy_picks[1].waveform_id.network_code == "YY"

    def test_from_obspy_picks(self):
        t = UTCDateTime("2024-01-01T00:00:10")
        obspy_picks = [
            Pick(
                time=t,
                waveform_id=WaveformStreamID(
                    network_code="XX",
                    station_code="STA1",
                    channel_code="HHZ",
                ),
                phase_hint="P",
            ),
        ]
        picks = from_obspy_picks(obspy_picks)
        assert len(picks) == 1
        assert picks[0].station == "STA1"
        assert picks[0].phase == "P"
        assert picks[0].network == "XX"
        assert picks[0].time == t

    def test_round_trip(self):
        t = UTCDateTime("2024-01-01T00:00:10")
        original = [
            SWEPick("STA1", "P", t, 0.9, network="XX", channel="HHZ"),
        ]
        obspy = to_obspy_picks(original)
        roundtripped = from_obspy_picks(obspy)
        assert roundtripped[0].station == "STA1"
        assert roundtripped[0].phase == "P"
        assert roundtripped[0].time == t
        assert roundtripped[0].network == "XX"

    def test_confidence_to_time_error_round_trip(self):
        """Confidence ↔ time_errors round-trip via ObsPy."""
        t = UTCDateTime("2024-01-01T00:00:10")
        original = [
            SWEPick("STA1", "P", t, 0.8, network="XX", channel="HHZ"),
            SWEPick("STA2", "S", t + 5, 0.5),
        ]
        obspy_picks = to_obspy_picks(
            original, map_confidence_to_time_error=True, max_time_error=1.0
        )
        # Verify time_errors set
        assert obspy_picks[0].time_errors is not None
        assert obspy_picks[0].time_errors.uncertainty == pytest.approx(0.2)
        assert obspy_picks[1].time_errors.uncertainty == pytest.approx(0.5)

        # Round-trip back
        roundtripped = from_obspy_picks(obspy_picks)
        assert roundtripped[0].confidence == pytest.approx(0.8)
        assert roundtripped[1].confidence == pytest.approx(0.5)

    def test_from_obspy_reads_time_errors(self):
        """from_obspy_picks reads time_errors.uncertainty into confidence."""
        from obspy.core.event import QuantityError as QE

        t = UTCDateTime("2024-01-01T00:00:10")
        pick = Pick(
            time=t,
            waveform_id=WaveformStreamID(
                network_code="XX", station_code="STA1", channel_code="HHZ",
            ),
            phase_hint="P",
        )
        pick.time_errors = QE(uncertainty=0.3)
        result = from_obspy_picks([pick])
        assert result[0].confidence == pytest.approx(0.7)

    def test_from_obspy_no_time_errors_defaults_to_one(self):
        """Without time_errors, confidence defaults to 1.0."""
        t = UTCDateTime("2024-01-01T00:00:10")
        pick = Pick(
            time=t,
            waveform_id=WaveformStreamID(
                network_code="XX", station_code="STA1", channel_code="HHZ",
            ),
            phase_hint="P",
        )
        result = from_obspy_picks([pick])
        assert result[0].confidence == 1.0


# ---------------------------------------------------------------------------
# DataFrame round-trip
# ---------------------------------------------------------------------------


class TestDataFrameRoundTrip:
    """SWEPick ↔ DataFrame conversion."""

    def test_to_dataframe(self):
        t = UTCDateTime("2024-01-01T00:00:10")
        picks = [
            SWEPick("STA1", "P", t, 0.9, method="PhaseNet"),
            SWEPick("STA2", "S", t + 5, 0.7),
        ]
        df = picks_to_dataframe(picks)
        assert len(df) == 2
        assert set(df.columns) >= {"station", "phase", "time", "confidence", "method"}
        assert df["station"].tolist() == ["STA1", "STA2"]
        assert df["phase"].tolist() == ["P", "S"]

    def test_from_dataframe(self):
        t = UTCDateTime("2024-01-01T00:00:10")
        df = pd.DataFrame(
            {
                "station": ["STA1", "STA2"],
                "phase": ["P", "S"],
                "time": [t.datetime, (t + 5).datetime],
                "confidence": [0.9, 0.7],
            }
        )
        picks = picks_from_dataframe(df)
        assert len(picks) == 2
        assert picks[0].station == "STA1"
        assert picks[0].confidence == 0.9

    def test_round_trip(self):
        t = UTCDateTime("2024-01-01T00:00:10")
        original = [
            SWEPick("STA1", "P", t, 0.9, network="XX", channel="HHZ", method="M"),
        ]
        df = picks_to_dataframe(original)
        roundtripped = picks_from_dataframe(df)
        assert roundtripped[0].station == original[0].station
        assert roundtripped[0].phase == original[0].phase
        assert roundtripped[0].confidence == original[0].confidence
        assert roundtripped[0].method == original[0].method
