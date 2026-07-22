"""Tests for swe.detection.matched_filter."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from obspy.core.event import (
    Catalog,
    Event,
    Origin,
    Pick,
    WaveformStreamID,
)

from swe.detection.matched_filter import (
    _next_day,
    cull_detections,
    detect_signals,
    detections_from_stacks,
    get_detections,
    make_templates,
    markers_to_template,
)


# ---------------------------------------------------------------------------
# Local mock helpers (mirrors conftest.py but avoids cross-package imports)
# ---------------------------------------------------------------------------
def _make_mock_detection(
    detect_time,
    detect_val=0.5,
    threshold=8.0,
    threshold_input=1.0,
    threshold_type="MAD",
    template_name="test",
):
    det = MagicMock()
    det.detect_time = detect_time
    det.detect_val = detect_val
    det.threshold = threshold
    det.threshold_input = threshold_input
    det.threshold_type = threshold_type
    det.template_name = template_name
    return det


def _make_mock_family(template=None, detections=None, catalog=None):
    family = MagicMock()
    family.template = template or MagicMock()
    family.detections = detections or []
    family.catalog = catalog or MagicMock()
    family.__len__ = lambda self: len(self.detections)
    return family


def _make_mock_party(detections=None, family=None):
    if family is None:
        family = _make_mock_family(detections=detections or [])
    party = MagicMock()
    party.families = [family]
    return party


# ---------------------------------------------------------------------------
# TestMarkersToTemplate (existing, kept intact)
# ---------------------------------------------------------------------------
class TestMarkersToTemplate:
    @pytest.fixture()
    def marker_file(self, tmp_path):
        """Create a minimal Snuffler marker file with two S picks."""
        content = (
            "phase: 2020-01-01 00:00:10.0000  0 "
            "YG.STA1..BHZ    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 2020-01-01   "
            "00:00:10.0000 S        None False\n"
            "phase: 2020-01-01 00:00:12.5000  0 "
            "YG.STA2..BHZ    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 2020-01-01   "
            "00:00:12.5000 S        None False\n"
        )
        marker_path = tmp_path / "test.mrkr"
        marker_path.write_text(content)
        return marker_path

    def test_returns_four_tuple(self, marker_file):
        result = markers_to_template(marker_file, prepick_offset=2.0)
        assert len(result) == 4

    def test_station_dict_populated(self, marker_file):
        _, station_dict, _, _ = markers_to_template(marker_file, prepick_offset=2.0)
        assert "STA1" in station_dict
        assert "STA2" in station_dict
        assert station_dict["STA1"]["network"] == "YG"

    def test_pick_offsets(self, marker_file):
        _, _, pick_offset, _ = markers_to_template(marker_file, prepick_offset=2.0)
        assert pick_offset["STA1"] == 0.0
        assert abs(pick_offset["STA2"] - 2.5) < 0.01

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.mrkr"
        empty.write_text("")
        templates, sd, _po, _tw = markers_to_template(empty, prepick_offset=1.0)
        assert templates == []
        assert sd == {}

    def test_template_lines_format(self, marker_file):
        templates, _, _, _ = markers_to_template(marker_file, prepick_offset=2.0)
        assert templates[0].startswith("#")
        assert len(templates) == 3

    def test_borehole_channel_preserved(self, tmp_path):
        """Borehole channels (e.g. DP1) must be preserved, not coerced to '...Z'.

        Regression: the old code did ``channel[:-1] + "Z"`` which turned the
        Parkfield HRSN channel ``DP1`` into the non-existent ``DPZ``.
        """
        content = (
            "phase: 2020-01-01 00:00:10.0000  0 "
            "5A.27..DP1      xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 2020-01-01   "
            "00:00:10.0000 S        None False\n"
        )
        marker_path = tmp_path / "borehole.mrkr"
        marker_path.write_text(content)
        _, station_dict, _, _ = markers_to_template(marker_path, prepick_offset=2.0)
        assert station_dict["27"]["channel"] == "DP1"


# ---------------------------------------------------------------------------
# TestNextDay
# ---------------------------------------------------------------------------
class TestNextDay:
    def test_normal_day(self):
        result = _next_day(UTCDateTime("2020-01-15T12:34:56"))
        assert result == UTCDateTime("2020-01-16T00:00:00")

    def test_month_end(self):
        result = _next_day(UTCDateTime("2020-01-31T00:00:00"))
        assert result == UTCDateTime("2020-02-01T00:00:00")

    def test_year_end(self):
        result = _next_day(UTCDateTime("2020-12-31T23:59:59"))
        assert result == UTCDateTime("2021-01-01T00:00:00")


# ---------------------------------------------------------------------------
# Helpers for mock Tribe / Party
# ---------------------------------------------------------------------------


def _mock_tribe_with_detect(detect_return):
    """Return a mock Tribe whose .detect() returns *detect_return*."""
    tribe = MagicMock()
    tribe.detect.return_value = detect_return
    return tribe


def _empty_party_mock():
    """Party with one empty family (len == 0)."""
    family = MagicMock()
    family.__len__ = lambda self: 0
    party = MagicMock()
    party.families = [family]
    return party


def _nonempty_party_mock(n=1):
    """Party with one family containing *n* detections."""
    family = MagicMock()
    family.__len__ = lambda self: n
    family.append.return_value = family
    party = MagicMock()
    party.families = [family]
    party.write = MagicMock()
    return party


# ---------------------------------------------------------------------------
# TestMakeTemplates
# ---------------------------------------------------------------------------
class TestMakeTemplates:
    _MODULE = "swe.detection.matched_filter"

    def _catalog_with_picks(self, stations, station_dict):
        """Build a real Catalog with one event and one pick per station."""
        picks = []
        for sta in stations:
            info = station_dict.get(sta, {"network": "YG", "channel": "BHZ"})
            picks.append(
                Pick(
                    time=UTCDateTime("2020-01-01T00:00:10"),
                    waveform_id=WaveformStreamID(
                        network_code=info["network"],
                        station_code=sta,
                        channel_code=info["channel"],
                    ),
                )
            )
        event = Event(
            origins=[Origin(time=UTCDateTime("2020-01-01"), latitude=0, longitude=0, depth=0)],
            picks=picks,
        )
        return Catalog([event])

    @patch(f"{_MODULE}.Tribe")
    @patch(f"{_MODULE}.read")
    @patch(f"{_MODULE}.read_events")
    def test_happy_path_returns_tribe(self, mock_read_events, mock_read, MockTribe):
        station_dict = {"STA1": {"network": "YG", "channel": "BHZ"}}
        mock_read_events.return_value = self._catalog_with_picks(["STA1"], station_dict)
        mock_read.return_value = Stream([Trace(data=np.zeros(100))])
        tribe_inst = MagicMock()
        tribe_inst.construct.return_value = tribe_inst
        MockTribe.return_value = tribe_inst

        result = make_templates(
            ["# header\n", "STA1  0.000  1  P\n"],
            ["/fake/file.ms"],
            station_dict,
            template_length=16.0,
            template_prepick=2.0,
        )
        assert result is tribe_inst
        tribe_inst.construct.assert_called_once()

    @patch(f"{_MODULE}.Tribe")
    @patch(f"{_MODULE}.read")
    @patch(f"{_MODULE}.read_events")
    def test_three_component_true_expands_picks(self, mock_read_events, mock_read, MockTribe):
        station_dict = {"STA1": {"network": "YG", "channel": "BHZ"}}
        catalog = self._catalog_with_picks(["STA1"], station_dict)
        mock_read_events.return_value = catalog
        mock_read.return_value = Stream([Trace(data=np.zeros(100))])
        tribe_inst = MagicMock()
        tribe_inst.construct.return_value = tribe_inst
        MockTribe.return_value = tribe_inst

        make_templates(
            ["# header\n", "STA1  0.000  1  P\n"],
            ["/fake/file.ms"],
            station_dict,
            template_length=16.0,
            template_prepick=2.0,
            three_component=True,
        )
        # Original Z pick + N + E = 3 picks
        assert len(catalog.events[0].picks) == 3

    @patch(f"{_MODULE}.Tribe")
    @patch(f"{_MODULE}.read")
    @patch(f"{_MODULE}.read_events")
    def test_three_component_false_keeps_z_only(self, mock_read_events, mock_read, MockTribe):
        station_dict = {"STA1": {"network": "YG", "channel": "BHZ"}}
        catalog = self._catalog_with_picks(["STA1"], station_dict)
        mock_read_events.return_value = catalog
        mock_read.return_value = Stream([Trace(data=np.zeros(100))])
        tribe_inst = MagicMock()
        tribe_inst.construct.return_value = tribe_inst
        MockTribe.return_value = tribe_inst

        make_templates(
            ["# header\n", "STA1  0.000  1  P\n"],
            ["/fake/file.ms"],
            station_dict,
            template_length=16.0,
            template_prepick=2.0,
            three_component=False,
        )
        assert len(catalog.events[0].picks) == 1
        assert catalog.events[0].picks[0].waveform_id.channel_code.endswith("Z")

    @patch(f"{_MODULE}.Tribe")
    @patch(f"{_MODULE}.read")
    @patch(f"{_MODULE}.read_events")
    def test_station_not_in_dict_skipped(self, mock_read_events, mock_read, MockTribe):
        station_dict = {}  # STA1 not present
        catalog = self._catalog_with_picks(["STA1"], {"STA1": {"network": "YG", "channel": "BHZ"}})
        mock_read_events.return_value = catalog
        mock_read.return_value = Stream([Trace(data=np.zeros(100))])
        tribe_inst = MagicMock()
        tribe_inst.construct.return_value = tribe_inst
        MockTribe.return_value = tribe_inst

        result = make_templates(
            ["# header\n", "STA1  0.000  1  P\n"],
            ["/fake/file.ms"],
            station_dict,
            template_length=16.0,
            template_prepick=2.0,
        )
        # Pick should still have original network (not overwritten)
        assert result is tribe_inst

    @patch(f"{_MODULE}.Tribe")
    @patch(f"{_MODULE}.read")
    @patch(f"{_MODULE}.read_events")
    def test_empty_templates_list(self, mock_read_events, mock_read, MockTribe):
        mock_read_events.return_value = Catalog()
        mock_read.return_value = Stream()
        tribe_inst = MagicMock()
        tribe_inst.construct.return_value = tribe_inst
        MockTribe.return_value = tribe_inst

        result = make_templates(
            [],
            [],
            {},
            template_length=16.0,
            template_prepick=2.0,
        )
        assert result is tribe_inst

    @patch(f"{_MODULE}.Tribe")
    @patch(f"{_MODULE}.read")
    @patch(f"{_MODULE}.read_events")
    def test_borehole_channels_three_component(self, mock_read_events, mock_read, MockTribe):
        """1/2/3 borehole channels expand to their real siblings, not Z/N/E.

        Regression: ``SP1`` previously became the non-existent ``SPZ``/``SPN``/
        ``SPE``, yielding an empty Tribe.  It must expand to ``SP1/SP2/SP3``.
        """
        station_dict = {"EADB": {"network": "BP", "channel": "SP1"}}
        catalog = self._catalog_with_picks(["EADB"], station_dict)
        mock_read_events.return_value = catalog
        mock_read.return_value = Stream([Trace(data=np.zeros(100))])
        tribe_inst = MagicMock()
        tribe_inst.construct.return_value = tribe_inst
        MockTribe.return_value = tribe_inst

        make_templates(
            ["# header\n", "EADB  0.000  1  P\n"],
            ["/fake/file.ms"],
            station_dict,
            template_length=6.0,
            template_prepick=1.0,
            three_component=True,
        )
        channels = {p.waveform_id.channel_code for p in catalog.events[0].picks}
        assert channels == {"SP1", "SP2", "SP3"}
        assert "SPZ" not in channels

    @patch(f"{_MODULE}.Tribe")
    @patch(f"{_MODULE}.read")
    @patch(f"{_MODULE}.read_events")
    def test_borehole_channel_single_component(self, mock_read_events, mock_read, MockTribe):
        """three_component=False keeps the exact borehole channel (SP1, not SPZ)."""
        station_dict = {"EADB": {"network": "BP", "channel": "SP1"}}
        catalog = self._catalog_with_picks(["EADB"], station_dict)
        mock_read_events.return_value = catalog
        mock_read.return_value = Stream([Trace(data=np.zeros(100))])
        tribe_inst = MagicMock()
        tribe_inst.construct.return_value = tribe_inst
        MockTribe.return_value = tribe_inst

        make_templates(
            ["# header\n", "EADB  0.000  1  P\n"],
            ["/fake/file.ms"],
            station_dict,
            template_length=6.0,
            template_prepick=1.0,
            three_component=False,
        )
        picks = catalog.events[0].picks
        assert len(picks) == 1
        assert picks[0].waveform_id.channel_code == "SP1"


# ---------------------------------------------------------------------------
# TestDetectSignals
# ---------------------------------------------------------------------------
class TestDetectSignals:
    _MODULE = "swe.detection.matched_filter"

    @patch(f"{_MODULE}.make_templates")
    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_single_day_returns_party(self, mock_glob, mock_mt):
        party = _nonempty_party_mock(n=2)
        tribe = _mock_tribe_with_detect(party)
        mock_mt.return_value = tribe

        result = detect_signals(
            ["# h\n"],
            ["/f.ms"],
            {},
            16.0,
            2.0,
            "/data",
            UTCDateTime("2020-01-15"),
            UTCDateTime("2020-01-16"),
        )
        assert result is not None
        tribe.detect.assert_called_once()

    @patch(f"{_MODULE}.make_templates")
    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_multi_day_with_month_boundary(self, mock_glob, mock_mt):
        party = _nonempty_party_mock(n=1)
        tribe = _mock_tribe_with_detect(party)
        mock_mt.return_value = tribe

        detect_signals(
            ["# h\n"],
            ["/f.ms"],
            {},
            16.0,
            2.0,
            "/data",
            UTCDateTime("2020-01-31"),
            UTCDateTime("2020-02-02"),
        )
        assert tribe.detect.call_count == 2

    @patch(f"{_MODULE}.make_templates")
    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_no_day_files_still_runs(self, mock_glob, mock_mt):
        party = _empty_party_mock()
        tribe = _mock_tribe_with_detect(party)
        mock_mt.return_value = tribe

        result = detect_signals(
            ["# h\n"],
            ["/f.ms"],
            {},
            16.0,
            2.0,
            "/data",
            UTCDateTime("2020-01-15"),
            UTCDateTime("2020-01-16"),
        )
        tribe.detect.assert_called_once()
        assert result is None

    @patch(f"{_MODULE}.make_templates")
    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_detect_raises_matchfiltererror_continues(self, mock_glob, mock_mt, caplog):
        from eqcorrscan.core.match_filter.matched_filter import MatchFilterError

        tribe = MagicMock()
        tribe.detect.side_effect = MatchFilterError("boom")
        mock_mt.return_value = tribe

        with caplog.at_level(logging.WARNING):
            result = detect_signals(
                ["# h\n"],
                ["/f.ms"],
                {},
                16.0,
                2.0,
                "/data",
                UTCDateTime("2020-01-15"),
                UTCDateTime("2020-01-16"),
            )
        assert result is None
        assert "Detection failed" in caplog.text

    @patch(f"{_MODULE}.make_templates")
    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_no_detections_returns_none(self, mock_glob, mock_mt):
        party = _empty_party_mock()
        tribe = _mock_tribe_with_detect(party)
        mock_mt.return_value = tribe

        result = detect_signals(
            ["# h\n"],
            ["/f.ms"],
            {},
            16.0,
            2.0,
            "/data",
            UTCDateTime("2020-01-15"),
            UTCDateTime("2020-01-17"),
        )
        assert result is None

    @patch(f"{_MODULE}.make_templates")
    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_output_path_triggers_write(self, mock_glob, mock_mt, tmp_path):
        party = _nonempty_party_mock(n=1)
        tribe = _mock_tribe_with_detect(party)
        mock_mt.return_value = tribe

        detect_signals(
            ["# h\n"],
            ["/f.ms"],
            {},
            16.0,
            2.0,
            "/data",
            UTCDateTime("2020-01-15"),
            UTCDateTime("2020-01-16"),
            output_path=str(tmp_path / "out"),
        )
        party.write.assert_called_once()


# ---------------------------------------------------------------------------
# TestCullDetections
# ---------------------------------------------------------------------------
class TestCullDetections:
    _MODULE = "swe.detection.matched_filter"

    def _make_mock_stream(self, *, data=None, npts=6001):
        """Build a mock Stream that survives filter/trim/indexing.

        Returns a MagicMock whose __getitem__ yields a trace-like object
        with controllable ``data`` and ``stats``.
        """
        if data is None:
            data = np.ones(npts, dtype=np.float64) * 3.7  # non-zero after rint
        stats = MagicMock()
        stats.npts = len(data)
        trace_mock = MagicMock()
        trace_mock.data = data
        trace_mock.stats = stats

        st = MagicMock(spec=Stream)
        st.filter.return_value = st
        st.trim.return_value = st
        st.__getitem__ = lambda self, idx: trace_mock
        return st

    @patch(f"{_MODULE}.snr", return_value=[5.0])
    @patch(f"{_MODULE}.read")
    @patch(f"{_MODULE}.glob.glob")
    def test_happy_path_keeps_good_detection(self, mock_glob, mock_read, mock_snr):
        mock_glob.return_value = ["/data/YG.STA1.BHZ.2020-01-15.ms"]
        mock_read.return_value = self._make_mock_stream()

        det = _make_mock_detection(UTCDateTime("2020-01-15T10:00:00"))
        party = _make_mock_party(detections=[det])

        with (
            patch(f"{self._MODULE}.Party") as MockParty,
            patch(f"{self._MODULE}.Family") as MockFamily,
        ):
            MockFamily.return_value = MagicMock()
            MockParty.return_value = MagicMock()
            cull_detections(party, "/data", ("YG", "STA1", "BHZ"))
            kept = MockFamily.call_args[0][1]
            assert len(kept) == 1

    @patch(f"{_MODULE}.snr", return_value=[0.1])
    @patch(f"{_MODULE}.read")
    @patch(f"{_MODULE}.glob.glob")
    def test_snr_below_min_removed(self, mock_glob, mock_read, mock_snr):
        mock_glob.return_value = ["/data/f.ms"]
        mock_read.return_value = self._make_mock_stream()

        det = _make_mock_detection(UTCDateTime("2020-01-15T10:00:00"))
        party = _make_mock_party(detections=[det])

        with (
            patch(f"{self._MODULE}.Party") as MockParty,
            patch(f"{self._MODULE}.Family") as MockFamily,
        ):
            MockFamily.return_value = MagicMock()
            MockParty.return_value = MagicMock()
            cull_detections(party, "/data", ("YG", "STA1", "BHZ"), snr_min=1.0)
            kept = MockFamily.call_args[0][1]
            assert len(kept) == 0

    @patch(f"{_MODULE}.snr", return_value=[100.0])
    @patch(f"{_MODULE}.read")
    @patch(f"{_MODULE}.glob.glob")
    def test_snr_above_max_removed(self, mock_glob, mock_read, mock_snr):
        mock_glob.return_value = ["/data/f.ms"]
        mock_read.return_value = self._make_mock_stream()

        det = _make_mock_detection(UTCDateTime("2020-01-15T10:00:00"))
        party = _make_mock_party(detections=[det])

        with (
            patch(f"{self._MODULE}.Party") as MockParty,
            patch(f"{self._MODULE}.Family") as MockFamily,
        ):
            MockFamily.return_value = MagicMock()
            MockParty.return_value = MagicMock()
            cull_detections(party, "/data", ("YG", "STA1", "BHZ"), snr_max=25.0)
            kept = MockFamily.call_args[0][1]
            assert len(kept) == 0

    @patch(f"{_MODULE}.snr", return_value=[float("nan")])
    @patch(f"{_MODULE}.read")
    @patch(f"{_MODULE}.glob.glob")
    def test_nan_snr_removed(self, mock_glob, mock_read, mock_snr):
        mock_glob.return_value = ["/data/f.ms"]
        mock_read.return_value = self._make_mock_stream()

        det = _make_mock_detection(UTCDateTime("2020-01-15T10:00:00"))
        party = _make_mock_party(detections=[det])

        with (
            patch(f"{self._MODULE}.Party") as MockParty,
            patch(f"{self._MODULE}.Family") as MockFamily,
        ):
            MockFamily.return_value = MagicMock()
            MockParty.return_value = MagicMock()
            cull_detections(party, "/data", ("YG", "STA1", "BHZ"))
            kept = MockFamily.call_args[0][1]
            assert len(kept) == 0

    @patch(f"{_MODULE}.snr", return_value=[5.0])
    @patch(f"{_MODULE}.read")
    @patch(f"{_MODULE}.glob.glob")
    def test_high_zero_fraction_removed(self, mock_glob, mock_read, mock_snr):
        mock_glob.return_value = ["/data/f.ms"]
        # All-zero data -> zero_count > zero_fraction * npts
        mock_read.return_value = self._make_mock_stream(data=np.zeros(6001))

        det = _make_mock_detection(UTCDateTime("2020-01-15T10:00:00"))
        party = _make_mock_party(detections=[det])

        with (
            patch(f"{self._MODULE}.Party") as MockParty,
            patch(f"{self._MODULE}.Family") as MockFamily,
        ):
            MockFamily.return_value = MagicMock()
            MockParty.return_value = MagicMock()
            cull_detections(party, "/data", ("YG", "STA1", "BHZ"), zero_fraction=0.3)
            kept = MockFamily.call_args[0][1]
            assert len(kept) == 0

    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_no_day_files_skips_detection(self, mock_glob):
        det = _make_mock_detection(UTCDateTime("2020-01-15T10:00:00"))
        party = _make_mock_party(detections=[det])

        with (
            patch(f"{self._MODULE}.Party") as MockParty,
            patch(f"{self._MODULE}.Family") as MockFamily,
        ):
            MockFamily.return_value = MagicMock()
            MockParty.return_value = MagicMock()
            cull_detections(party, "/data", ("YG", "STA1", "BHZ"))
            kept = MockFamily.call_args[0][1]
            assert len(kept) == 0

    @patch(f"{_MODULE}.read", side_effect=OSError("corrupt file"))
    @patch(f"{_MODULE}.glob.glob", return_value=["/data/f.ms"])
    def test_exception_during_processing_skips_and_warns(self, mock_glob, mock_read, caplog):
        det = _make_mock_detection(UTCDateTime("2020-01-15T10:00:00"))
        party = _make_mock_party(detections=[det])

        with (
            patch(f"{self._MODULE}.Party") as MockParty,
            patch(f"{self._MODULE}.Family") as MockFamily,
        ):
            MockFamily.return_value = MagicMock()
            MockParty.return_value = MagicMock()
            with caplog.at_level(logging.WARNING):
                cull_detections(party, "/data", ("YG", "STA1", "BHZ"))
            kept = MockFamily.call_args[0][1]
            assert len(kept) == 0
            assert "Skipping detection" in caplog.text
            assert "OSError" in caplog.text


# ---------------------------------------------------------------------------
# TestGetDetections
# ---------------------------------------------------------------------------
class TestGetDetections:
    _MODULE = "swe.detection.matched_filter"

    def _make_catalog_party(self, picks_info):
        """Build a mock Party with catalog containing events and picks.

        Parameters
        ----------
        picks_info : list of (net, sta, cha, time)
            Each entry becomes a pick on a separate event.
        """
        events = []
        for net, sta, cha, time in picks_info:
            pick = MagicMock()
            pick.waveform_id.station_code = sta
            pick.waveform_id.network_code = net
            pick.waveform_id.channel_code = cha
            pick.time = time
            event = MagicMock()
            event.picks = [pick]
            events.append(event)

        family = MagicMock()
        family.catalog.events = events
        party = MagicMock()
        party.families = [family]
        return party

    @patch(f"{_MODULE}.read")
    @patch(f"{_MODULE}.glob.glob")
    def test_happy_path_returns_stream(self, mock_glob, mock_read):
        mock_glob.return_value = ["/data/YG.STA1.BHZ.2020-01-15.ms"]
        tr = Trace(data=np.ones(6001))
        tr.stats.sampling_rate = 100.0
        tr.stats.starttime = UTCDateTime("2020-01-15T00:00:00")
        mock_read.return_value = Stream([tr])

        party = self._make_catalog_party(
            [
                ("YG", "STA1", "BHZ", UTCDateTime("2020-01-15T10:00:00")),
            ]
        )

        result = get_detections(party, "/data", ("YG", "STA1", "BHZ"))
        assert len(result) == 1

    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_no_day_files_returns_empty_trace(self, mock_glob):
        party = self._make_catalog_party(
            [
                ("YG", "STA1", "BHZ", UTCDateTime("2020-01-15T10:00:00")),
            ]
        )

        result = get_detections(party, "/data", ("YG", "STA1", "BHZ"))
        assert len(result) == 1
        assert result[0].stats.npts == 0  # empty Trace

    @patch(f"{_MODULE}.read")
    @patch(f"{_MODULE}.glob.glob")
    def test_multiple_detections_same_day(self, mock_glob, mock_read):
        mock_glob.return_value = ["/data/f.ms"]
        tr = Trace(data=np.ones(6001))
        tr.stats.sampling_rate = 100.0
        tr.stats.starttime = UTCDateTime("2020-01-15T00:00:00")
        mock_read.return_value = Stream([tr])

        party = self._make_catalog_party(
            [
                ("YG", "STA1", "BHZ", UTCDateTime("2020-01-15T10:00:00")),
                ("YG", "STA1", "BHZ", UTCDateTime("2020-01-15T14:00:00")),
            ]
        )

        result = get_detections(party, "/data", ("YG", "STA1", "BHZ"))
        assert len(result) == 2

    def test_pick_matching_filters_by_station(self):
        """Only the matching network/station/channel pick is used."""
        pick_match = MagicMock()
        pick_match.waveform_id.station_code = "STA1"
        pick_match.waveform_id.network_code = "YG"
        pick_match.waveform_id.channel_code = "BHZ"
        pick_match.time = UTCDateTime("2020-01-15T10:00:00")

        pick_other = MagicMock()
        pick_other.waveform_id.station_code = "STA2"
        pick_other.waveform_id.network_code = "YG"
        pick_other.waveform_id.channel_code = "BHZ"
        pick_other.time = UTCDateTime("2020-01-15T12:00:00")

        event = MagicMock()
        event.picks = [pick_other, pick_match]
        family = MagicMock()
        family.catalog.events = [event]
        party = MagicMock()
        party.families = [family]

        with patch(f"{self._MODULE}.glob.glob", return_value=[]), patch(f"{self._MODULE}.read"):
            result = get_detections(party, "/data", ("YG", "STA1", "BHZ"))
        # Should find 1 detection from the matching pick
        assert len(result) == 1

    def test_no_events_returns_empty_stream(self):
        family = MagicMock()
        family.catalog.events = []
        party = MagicMock()
        party.families = [family]

        result = get_detections(party, "/data", ("YG", "STA1", "BHZ"))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TestDetectionsFromStacks
# ---------------------------------------------------------------------------
class TestDetectionsFromStacks:
    _MODULE = "swe.detection.matched_filter"

    def _make_stack(self, n_traces=1):
        """Build a small Stream to serve as a stack."""
        st = Stream()
        for i in range(n_traces):
            tr = Trace(data=np.random.default_rng(i).standard_normal(1200).astype(np.float64))
            tr.stats.network = "YG"
            tr.stats.station = f"STA{i + 1}"
            tr.stats.channel = "BHZ"
            tr.stats.sampling_rate = 100.0
            tr.stats.starttime = UTCDateTime("2020-01-15T00:00:00")
            st += tr
        return st

    @patch(f"{_MODULE}.Tribe")
    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_happy_path_with_pick_times(self, mock_glob, MockTribe):
        party = _nonempty_party_mock(n=1)
        tribe_inst = MagicMock()
        tribe_inst.construct.return_value = tribe_inst
        tribe_inst.detect.return_value = party
        MockTribe.return_value = tribe_inst

        stack = self._make_stack()
        picks = {"STA1": UTCDateTime("2020-01-15T00:00:05")}

        result = detections_from_stacks(
            stack,
            "/data",
            UTCDateTime("2020-01-15"),
            UTCDateTime("2020-01-16"),
            pick_times=picks,
        )
        assert result is not None

    @patch(f"{_MODULE}.Tribe")
    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_pick_times_none_uses_midpoint(self, mock_glob, MockTribe):
        party = _nonempty_party_mock(n=1)
        tribe_inst = MagicMock()
        tribe_inst.construct.return_value = tribe_inst
        tribe_inst.detect.return_value = party
        MockTribe.return_value = tribe_inst

        stack = self._make_stack()
        result = detections_from_stacks(
            stack,
            "/data",
            UTCDateTime("2020-01-15"),
            UTCDateTime("2020-01-16"),
            pick_times=None,
        )
        assert result is not None

    @patch(f"{_MODULE}.Tribe")
    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_no_detections_returns_none(self, mock_glob, MockTribe):
        party = _empty_party_mock()
        tribe_inst = MagicMock()
        tribe_inst.construct.return_value = tribe_inst
        tribe_inst.detect.return_value = party
        MockTribe.return_value = tribe_inst

        stack = self._make_stack()
        result = detections_from_stacks(
            stack,
            "/data",
            UTCDateTime("2020-01-15"),
            UTCDateTime("2020-01-16"),
        )
        assert result is None

    @patch(f"{_MODULE}.Tribe")
    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_detect_failure_continues_and_warns(self, mock_glob, MockTribe, caplog):
        from eqcorrscan.core.match_filter.matched_filter import MatchFilterError

        tribe_inst = MagicMock()
        tribe_inst.construct.return_value = tribe_inst
        tribe_inst.detect.side_effect = MatchFilterError("bad data")
        MockTribe.return_value = tribe_inst

        stack = self._make_stack()
        with caplog.at_level(logging.WARNING):
            result = detections_from_stacks(
                stack,
                "/data",
                UTCDateTime("2020-01-15"),
                UTCDateTime("2020-01-16"),
            )
        assert result is None
        assert "Stack detection failed" in caplog.text

    @patch(f"{_MODULE}.Tribe")
    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_process_len_defaults_to_short_auto(self, mock_glob, MockTribe):
        """With process_len=None the legacy auto value (max(60, dur+10)) is used."""
        party = _nonempty_party_mock(n=1)
        tribe_inst = MagicMock()
        tribe_inst.construct.return_value = tribe_inst
        tribe_inst.detect.return_value = party
        MockTribe.return_value = tribe_inst

        stack = self._make_stack()  # 1200 samples @ 100 Hz -> 12 s
        detections_from_stacks(
            stack,
            "/data",
            UTCDateTime("2020-01-15"),
            UTCDateTime("2020-01-16"),
        )
        assert tribe_inst.construct.call_args.kwargs["process_len"] == 60

    @patch(f"{_MODULE}.Tribe")
    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_process_len_override_forwarded(self, mock_glob, MockTribe):
        """An explicit process_len (e.g. day length) is passed to Tribe.construct."""
        party = _nonempty_party_mock(n=1)
        tribe_inst = MagicMock()
        tribe_inst.construct.return_value = tribe_inst
        tribe_inst.detect.return_value = party
        MockTribe.return_value = tribe_inst

        stack = self._make_stack()
        detections_from_stacks(
            stack,
            "/data",
            UTCDateTime("2020-01-15"),
            UTCDateTime("2020-01-16"),
            process_len=86400,
        )
        assert tribe_inst.construct.call_args.kwargs["process_len"] == 86400

    @patch(f"{_MODULE}.Tribe")
    @patch(f"{_MODULE}.glob.glob", return_value=[])
    def test_date_iteration_across_month_boundary(self, mock_glob, MockTribe):
        party = _nonempty_party_mock(n=1)
        tribe_inst = MagicMock()
        tribe_inst.construct.return_value = tribe_inst
        tribe_inst.detect.return_value = party
        MockTribe.return_value = tribe_inst

        stack = self._make_stack()
        detections_from_stacks(
            stack,
            "/data",
            UTCDateTime("2020-01-31"),
            UTCDateTime("2020-02-02"),
        )
        assert tribe_inst.detect.call_count == 2
