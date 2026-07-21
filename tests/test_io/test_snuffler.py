"""Tests for swe.io.snuffler."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from obspy import UTCDateTime

from swe.io.snuffler import (
    parse_phase_picks,
    parse_all_markers,
    party_to_markers,
    write_events_to_markers,
)


class TestParsePhasePicksMrkr:
    """Test parsing Snuffler marker files."""

    # Real Snuffler format: fields are whitespace-delimited
    # Fields: phase: DATE TIME 0 NET.STA.LOC.CHA HASH DATE TIME PHASE_HINT None False
    SAMPLE_MARKER_CONTENT = """\
# Snuffler Markers File Version 0.2
phase: 2018-03-13 01:32:14.1230  0 5A.27..DP1      abc123hash= 2018-03-13   01:32:14.0000 P        None False
phase: 2018-03-13 01:32:14.4560  0 5A.28..DP1      abc123hash= 2018-03-13   01:32:14.0000 P        None False
phase: 2018-03-13 01:32:14.7890  0 UW.UGAP3..EHZ   abc123hash= 2018-03-13   01:32:14.0000 S        None False
event: 2018-03-13 01:32:14.0000  0
"""

    def test_parse_p_picks(self, tmp_path):
        mrkr_file = tmp_path / "test.mrkr"
        mrkr_file.write_text(self.SAMPLE_MARKER_CONTENT)

        picks = parse_phase_picks(str(mrkr_file), phase_hint="P")
        assert len(picks) == 2
        assert picks[0]["station"] == "27"
        assert picks[0]["channel"] == "DP1"
        assert picks[0]["network"] == "5A"
        assert picks[1]["station"] == "28"

    def test_parse_s_picks(self, tmp_path):
        mrkr_file = tmp_path / "test.mrkr"
        mrkr_file.write_text(self.SAMPLE_MARKER_CONTENT)

        picks = parse_phase_picks(str(mrkr_file), phase_hint="S")
        assert len(picks) == 1
        assert picks[0]["station"] == "UGAP3"
        assert picks[0]["channel"] == "EHZ"

    def test_parse_all_markers(self, tmp_path):
        mrkr_file = tmp_path / "test.mrkr"
        mrkr_file.write_text(self.SAMPLE_MARKER_CONTENT)

        markers = parse_all_markers(str(mrkr_file))
        phase_markers = [m for m in markers if m["type"] == "phase" and "station" in m]
        assert len(phase_markers) == 3

    def test_empty_file(self, tmp_path):
        mrkr_file = tmp_path / "empty.mrkr"
        mrkr_file.write_text("")

        picks = parse_phase_picks(str(mrkr_file))
        assert picks == []


class TestWriteEventsToMarkers:
    """Test writing events to Snuffler marker files."""

    def test_write_and_read_roundtrip(self, tmp_path):
        """Written events should be readable as phase markers."""
        events = {
            "evt001": [
                {"station": "STA1.BHZ", "time": UTCDateTime("2020-01-01T10:00:00")},
                {"station": "STA2.BHZ", "time": UTCDateTime("2020-01-01T10:00:01")},
            ],
        }
        output = tmp_path / "roundtrip.mrkr"
        result_path = write_events_to_markers(events, output)
        assert result_path.exists()

        content = result_path.read_text()
        assert "event:" in content
        assert "phase:" in content
        lines = content.strip().split("\n")
        # 1 event line + 2 phase lines = 3 lines
        assert len(lines) == 3

    def test_station_order(self, tmp_path):
        """Station order should set network code to index."""
        events = {
            "evt002": [
                {"station": "AAA.BHZ", "time": UTCDateTime("2020-06-15T12:00:00")},
            ],
        }
        output = tmp_path / "ordered.mrkr"
        write_events_to_markers(
            events, output, station_order=["ZZZ", "AAA", "BBB"],
        )
        content = output.read_text()
        # AAA is index 1 in station_order, so network code should be "1"
        assert "1.AAA..BHZ" in content

    def test_empty_events(self, tmp_path):
        """Empty events dict should produce empty file."""
        output = tmp_path / "empty_out.mrkr"
        write_events_to_markers({}, output)
        # File may not exist or be empty
        if output.exists():
            assert output.read_text() == ""

    def test_overwrites_existing(self, tmp_path):
        """Should overwrite existing file, not append."""
        output = tmp_path / "overwrite.mrkr"
        output.write_text("# existing content\n")

        events = {
            "evt003": [
                {"station": "STA.CHA", "time": UTCDateTime("2020-01-01T00:00:00")},
            ],
        }
        write_events_to_markers(events, output)
        content = output.read_text()
        assert "# existing content" not in content
        assert "evt003" in content


def _make_pick(net, sta, cha, time, phase_hint="P"):
    """Create a mock ObsPy Pick-like object."""
    wid = SimpleNamespace(
        network_code=net,
        station_code=sta,
        channel_code=cha,
        location_code="",
    )
    return SimpleNamespace(
        waveform_id=wid,
        time=UTCDateTime(time),
        phase_hint=phase_hint,
    )


def _make_party(events_data):
    """Create a mock EQcorrscan Party from a list of event pick lists.

    Parameters
    ----------
    events_data : list of list of tuple
        Each inner list is an event; tuples are
        ``(network, station, channel, time_str, phase_hint)``.
    """
    from obspy import Catalog
    from obspy.core.event import Event, Pick, WaveformStreamID

    families = []
    for event_picks in events_data:
        cat = Catalog()
        ev = Event()
        for net, sta, cha, time_str, hint in event_picks:
            wid = WaveformStreamID(
                network_code=net, station_code=sta,
                channel_code=cha, location_code="",
            )
            ev.picks.append(Pick(
                waveform_id=wid,
                time=UTCDateTime(time_str),
                phase_hint=hint,
            ))
        cat.events.append(ev)
        families.append(SimpleNamespace(catalog=cat))
    return SimpleNamespace(families=families)


class TestPartyToMarkers:
    """Tests for party_to_markers()."""

    def test_two_events_with_p_and_s(self, tmp_path):
        """Party with 2 events should produce readable markers."""
        party = _make_party([
            [
                ("5A", "STA1", "BHZ", "2020-01-01T10:00:00", "P"),
                ("5A", "STA1", "BHZ", "2020-01-01T10:00:02", "S"),
            ],
            [
                ("UW", "STA2", "EHZ", "2020-01-01T11:00:00", "P"),
            ],
        ])
        output = tmp_path / "party.mrkr"
        result = party_to_markers(party, output)
        assert result.exists()

        content = result.read_text()
        assert content.count("event:") == 2
        assert content.count("phase:") == 3

        p_picks = parse_phase_picks(str(result), phase_hint="P")
        s_picks = parse_phase_picks(str(result), phase_hint="S")
        assert len(p_picks) == 2
        assert len(s_picks) == 1

    def test_empty_party(self, tmp_path):
        """Empty party should produce empty file."""
        party = SimpleNamespace(families=[])
        output = tmp_path / "empty_party.mrkr"
        result = party_to_markers(party, output)
        assert result.exists()
        assert result.read_text() == ""

    def test_roundtrip(self, tmp_path):
        """Picks written by party_to_markers should round-trip through parse_phase_picks."""
        party = _make_party([
            [
                ("XX", "ABC", "BHZ", "2021-06-15T08:30:00.1234", "P"),
                ("XX", "DEF", "BHN", "2021-06-15T08:30:01.5678", "P"),
            ],
        ])
        output = tmp_path / "roundtrip.mrkr"
        party_to_markers(party, output)

        picks = parse_phase_picks(str(output), phase_hint="P")
        assert len(picks) == 2
        stations = {p["station"] for p in picks}
        assert stations == {"ABC", "DEF"}

    def test_phase_hint_override(self, tmp_path):
        """phase_hint kwarg should override individual pick hints."""
        party = _make_party([
            [
                ("XX", "STA", "BHZ", "2020-01-01T00:00:00", "P"),
                ("XX", "STA", "BHZ", "2020-01-01T00:00:01", "S"),
            ],
        ])
        output = tmp_path / "override.mrkr"
        party_to_markers(party, output, phase_hint="P")

        p_picks = parse_phase_picks(str(output), phase_hint="P")
        s_picks = parse_phase_picks(str(output), phase_hint="S")
        assert len(p_picks) == 2
        assert len(s_picks) == 0

    def test_network_codes_preserved(self, tmp_path):
        """Network codes from picks should appear in marker lines."""
        party = _make_party([
            [("NZ", "WEL", "HHZ", "2022-03-01T12:00:00", "P")],
        ])
        output = tmp_path / "net.mrkr"
        party_to_markers(party, output)
        content = output.read_text()
        assert "NZ.WEL..HHZ" in content
