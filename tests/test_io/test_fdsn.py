"""Tests for swe.io.fdsn."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from obspy import Catalog, Inventory, Stream, Trace, UTCDateTime
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException
from obspy.core.event import Event, Magnitude, Origin
from obspy.core.inventory import Channel, Network, Response, Station

from swe.io import fdsn
from swe.io.fdsn import (
    DEFAULT_EXCLUDED_NETWORKS,
    catalog_to_dataframe,
    get_events,
    get_response_files,
    get_stations_in_region,
    get_waveforms,
    inventory_to_dataframe,
    load_response,
)

T0 = UTCDateTime("2020-01-01T00:00:00.0Z")
T1 = UTCDateTime("2020-01-01T23:59:59.9Z")


def _make_inventory(network_codes=("XX",), *, end_date=None, sensitivity=1.5e9, channel="BHZ"):
    """Synthetic Inventory: one station with one BHZ channel per network."""
    nets = []
    for i, code in enumerate(network_codes):
        # A PAZ stage (not just an overall sensitivity): ObsPy >= 1.5 needs a stage
        # to remove the sensitivity.
        resp = Response.from_paz(
            zeros=[],
            poles=[],
            stage_gain=sensitivity,
            input_units="M/S",
            output_units="COUNTS",
        )
        cha = Channel(
            code=channel,
            location_code="00",
            latitude=35.0 + i,
            longitude=-118.0,
            elevation=100.0,
            depth=0.0,
            sample_rate=40.0,
            start_date=UTCDateTime("2010-01-01"),
            end_date=end_date,
            response=resp,
        )
        sta = Station(
            code=f"S{i}",
            latitude=35.0 + i,
            longitude=-118.0,
            elevation=100.0,
            channels=[cha],
            start_date=UTCDateTime("2010-01-01"),
            end_date=end_date,
        )
        nets.append(Network(code=code, stations=[sta]))
    return Inventory(networks=nets, source="test")


def _trace(network="XX", station="S0", channel="BHZ", location="00"):
    return Trace(
        data=np.ones(100, dtype=np.float64),
        header={
            "network": network,
            "station": station,
            "location": location,
            "channel": channel,
            "sampling_rate": 40.0,
            "starttime": T0,
        },
    )


def _make_day_stream(network, station, channel, date_str, sr=100.0, n=8640000):
    """Create a synthetic day-long Stream."""
    start = UTCDateTime(f"{date_str}T00:00:00.0Z")
    data = np.random.default_rng(42).normal(0, 1, n).astype(np.float32)
    tr = Trace(
        data=data,
        header={
            "network": network,
            "station": station,
            "channel": channel,
            "sampling_rate": sr,
            "starttime": start,
        },
    )
    return Stream([tr])


class TestGetWaveforms:
    @patch("swe.io.fdsn.Client")
    def test_writes_daily_file(self, MockClient, tmp_path):
        """Should write a daily MSEED file."""
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        # Return a short synthetic stream (1 day)
        start = UTCDateTime("2020-01-01T00:00:00.0Z")
        end = UTCDateTime("2020-01-01T23:59:59.9Z")
        data = np.zeros(100, dtype=np.float32)
        tr = Trace(
            data=data,
            header={
                "network": "XX",
                "station": "TEST",
                "channel": "BHZ",
                "sampling_rate": 100.0,
                "starttime": start,
            },
        )
        mock_client.get_waveforms.return_value = Stream([tr])

        result = get_waveforms(
            "XX",
            "TEST",
            "BHZ",
            start,
            end,
            output_dir=tmp_path,
            client="IRIS",
        )
        assert len(result) == 1
        assert result[0].exists()
        assert "XX.TEST.BHZ" in result[0].name

    @patch("swe.io.fdsn.Client")
    def test_handles_no_data(self, MockClient, tmp_path):
        """Should handle missing data gracefully."""
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.get_waveforms.side_effect = FDSNException("No data")

        start = UTCDateTime("2020-01-01T00:00:00.0Z")
        end = UTCDateTime("2020-01-01T23:59:59.9Z")

        result = get_waveforms(
            "XX",
            "TEST",
            "BHZ",
            start,
            end,
            output_dir=tmp_path,
        )
        assert result == []

    @patch("swe.io.fdsn.Client")
    def test_typeerror_propagates(self, MockClient, tmp_path):
        """TypeError should not be caught - it propagates."""
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.get_waveforms.side_effect = TypeError("bad arg")

        start = UTCDateTime("2020-01-01T00:00:00.0Z")
        end = UTCDateTime("2020-01-01T23:59:59.9Z")

        with pytest.raises(TypeError, match="bad arg"):
            get_waveforms(
                "XX",
                "TEST",
                "BHZ",
                start,
                end,
                output_dir=tmp_path,
            )


class TestGetResponseFiles:
    @patch("swe.io.fdsn.Client")
    def test_writes_stationxml(self, MockClient, tmp_path):
        """One StationXML file per channel returned, named with real codes."""
        MockClient.return_value.get_stations.return_value = _make_inventory()
        result = get_response_files("XX", ["STA1", "STA2"], "BHZ", T0, T1, output_dir=tmp_path)
        # the mock returns the same station for both requests: same file, written twice
        assert [p.name for p in result] == ["XX.S0.00.BHZ.xml", "XX.S0.00.BHZ.xml"]
        assert MockClient.return_value.get_stations.call_count == 2

    @patch("swe.io.fdsn.Client")
    def test_multi_epoch_channel_written_once(self, MockClient, tmp_path):
        """Several epochs of one channel go into one file, listed once."""
        inv = _make_inventory(end_date=UTCDateTime("2015-01-01"))
        later = _make_inventory()
        later[0][0][0].start_date = UTCDateTime("2015-01-01")
        inv[0][0].channels.append(later[0][0][0])
        inv[0][0].end_date = None  # the station stays open; only the first channel epoch ended
        MockClient.return_value.get_stations.return_value = inv
        result = get_response_files("XX", ["S0"], "BHZ", T0, T1, output_dir=tmp_path)
        assert [p.name for p in result] == ["XX.S0.00.BHZ.xml"]
        assert len(load_response(_trace(), tmp_path)[0][0].channels) == 2

    @patch("swe.io.fdsn.Client")
    def test_wildcard_request_writes_real_codes(self, MockClient, tmp_path):
        """A '*' station / 'BH?' channel request yields per-station, per-channel real names."""
        inv = _make_inventory(("XX", "YY"))
        inv += _make_inventory(("XX",), channel="BHN")
        MockClient.return_value.get_stations.return_value = inv
        result = get_response_files("*", ["*"], "BH?", T0, T1, output_dir=tmp_path)
        assert sorted(p.name for p in result) == [
            "XX.S0.00.BHN.xml",
            "XX.S0.00.BHZ.xml",
            "YY.S1.00.BHZ.xml",
        ]
        assert (
            len(load_response(_trace(network="YY", station="S1"), tmp_path).select(network="YY"))
            == 1
        )

    @patch("swe.io.fdsn.Client")
    def test_round_trip_through_load_response(self, MockClient, tmp_path):
        """Files written by get_response_files must load via load_response (StationXML)."""
        MockClient.return_value.get_stations.return_value = _make_inventory(sensitivity=2.5e9)
        written = get_response_files("XX", ["S0"], "BH?", T0, T1, output_dir=tmp_path)
        assert written[0].read_text().lstrip().startswith("<?xml")
        tr = _trace()
        inv = load_response(tr, tmp_path)
        sens = inv[0][0][0].response.instrument_sensitivity.value
        assert sens == pytest.approx(2.5e9)
        tr.remove_sensitivity(inventory=inv)
        assert tr.data[0] == pytest.approx(1.0 / 2.5e9)

    @patch("swe.io.fdsn.Client")
    def test_handles_failure(self, MockClient, tmp_path):
        """Should handle failed response downloads."""
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.get_stations.side_effect = FDSNException("No response")

        start = UTCDateTime("2020-01-01T00:00:00.0Z")
        end = UTCDateTime("2020-01-01T23:59:59.9Z")

        result = get_response_files(
            "XX",
            ["FAIL"],
            "BHZ",
            start,
            end,
            output_dir=tmp_path,
        )
        assert result == []

    @patch("swe.io.fdsn.Client")
    def test_typeerror_propagates(self, MockClient, tmp_path):
        """TypeError should not be caught - it propagates."""
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.get_stations.side_effect = TypeError("bad arg")

        start = UTCDateTime("2020-01-01T00:00:00.0Z")
        end = UTCDateTime("2020-01-01T23:59:59.9Z")

        with pytest.raises(TypeError, match="bad arg"):
            get_response_files(
                "XX",
                ["FAIL"],
                "BHZ",
                start,
                end,
                output_dir=tmp_path,
            )


class TestLoadResponse:
    @patch("swe.io.fdsn.Client")
    def test_per_channel_calls_do_not_overwrite(self, MockClient, tmp_path):
        """Separate BHZ and BHN downloads into one directory both stay loadable."""
        MockClient.return_value.get_stations.side_effect = [
            _make_inventory(channel="BHZ"),
            _make_inventory(channel="BHN"),
        ]
        get_response_files("XX", ["S0"], "BHZ", T0, T1, output_dir=tmp_path)
        get_response_files("XX", ["S0"], "BHN", T0, T1, output_dir=tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "XX.S0.00.BHN.xml",
            "XX.S0.00.BHZ.xml",
        ]
        for cha in ("BHZ", "BHN"):
            inv = load_response(_trace(channel=cha), tmp_path)
            assert len(inv.select(channel=cha)) == 1

    def test_other_station_file_is_found(self, tmp_path):
        """Any {net}.{sta}.*.xml covering the trace is used when the exact name is absent."""
        _make_inventory().write(str(tmp_path / "XX.S0.all.xml"), format="STATIONXML")
        assert load_response(_trace(), tmp_path)[0].code == "XX"

    def test_legacy_resp_named_stationxml(self, tmp_path):
        """Legacy RESP.* files that actually hold StationXML (written by older versions) load."""
        _make_inventory().write(str(tmp_path / "RESP.XX.S0..BHZ"), format="STATIONXML")
        inv = load_response(_trace(), tmp_path)
        assert inv[0].code == "XX"

    def test_non_covering_file_does_not_hide_legacy(self, tmp_path):
        """A new file for BHZ must not shadow a legacy BHN file for a BHN trace."""
        _make_inventory(channel="BHZ").write(
            str(tmp_path / "XX.S0.00.BHZ.xml"), format="STATIONXML"
        )
        _make_inventory(channel="BHN").write(str(tmp_path / "RESP.XX.S0..BHN"), format="STATIONXML")
        inv = load_response(_trace(channel="BHN"), tmp_path)
        assert len(inv.select(channel="BHN")) == 1

    def test_prefers_exact_channel_file(self, tmp_path):
        """The exact-channel StationXML wins over a legacy per-channel file."""
        _make_inventory(sensitivity=1.0e9).write(
            str(tmp_path / "XX.S0.BHZ.xml"), format="STATIONXML"
        )
        _make_inventory(sensitivity=9.0e9).write(
            str(tmp_path / "RESP.XX.S0..BHZ"), format="STATIONXML"
        )
        inv = load_response(_trace(), tmp_path)
        assert inv[0][0][0].response.instrument_sensitivity.value == pytest.approx(1.0e9)

    def test_missing_lists_paths(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"No response file for XX\.S0\.00\.BHZ"):
            load_response(_trace(), tmp_path)

    def test_files_present_but_none_covers(self, tmp_path):
        """Wrong location code or an epoch that ends before the trace: not covering."""
        _make_inventory().write(str(tmp_path / "XX.S0.00.BHZ.xml"), format="STATIONXML")
        with pytest.raises(FileNotFoundError, match="covers"):
            load_response(_trace(location="10"), tmp_path)
        ended = tmp_path / "ended"
        ended.mkdir()
        _make_inventory(end_date=UTCDateTime("2015-01-01")).write(
            str(ended / "XX.S0.00.BHZ.xml"), format="STATIONXML"
        )
        with pytest.raises(FileNotFoundError, match="covers"):
            load_response(_trace(), ended)

    def test_unreadable_file_does_not_block_fallback(self, tmp_path):
        """A corrupt or unrelated XML next to a valid legacy file is skipped."""
        (tmp_path / "XX.S0.notes.xml").write_text("<notes>not station metadata</notes>")
        (tmp_path / "XX.S0.00.BHZ.xml").write_text(
            "<?xml version='1.0'?><FDSNStationXML"
        )  # truncated
        _make_inventory().write(str(tmp_path / "RESP.XX.S0..BHZ"), format="STATIONXML")
        assert load_response(_trace(), tmp_path)[0].code == "XX"

    @pytest.mark.parametrize("err", [TypeError("Unknown format"), ValueError("misread")])
    def test_legacy_retried_as_resp_on_any_parse_error(self, tmp_path, err):
        legacy = tmp_path / "RESP.XX.S0..BHZ"
        legacy.write_text("\nB050F03     Station:     S0\n")
        inv = _make_inventory()
        fdsn._read_inventory_cached.cache_clear()
        with patch("swe.io.fdsn.read_inventory", side_effect=[err, inv]) as spy:
            assert load_response(_trace(), tmp_path) is inv
        assert spy.call_args_list[-1].kwargs == {"format": "RESP"}

    def test_exact_hit_does_not_list_directory(self, tmp_path):
        _make_inventory().write(str(tmp_path / "XX.S0.00.BHZ.xml"), format="STATIONXML")
        with patch("pathlib.Path.glob", side_effect=AssertionError("listed")):
            load_response(_trace(), tmp_path)

    def test_parse_is_cached(self, tmp_path):
        _make_inventory().write(str(tmp_path / "XX.S0.00.BHZ.xml"), format="STATIONXML")
        fdsn._read_inventory_cached.cache_clear()
        with patch("swe.io.fdsn.read_inventory", wraps=fdsn.read_inventory) as spy:
            for _ in range(3):
                load_response(_trace(), tmp_path)
        assert spy.call_count == 1


class TestClientArgument:
    @patch("swe.io.fdsn.Client")
    def test_base_url_string_creates_client(self, MockClient, tmp_path):
        MockClient.return_value.get_waveforms.side_effect = FDSNException("No data")
        get_waveforms(
            "XX", "S0", "BHZ", T0, T1, output_dir=tmp_path, client="https://service.example.org"
        )
        MockClient.assert_called_once_with("https://service.example.org")

    @patch("swe.io.fdsn.Client")
    def test_client_instance_is_reused(self, MockClient, tmp_path):
        ready = MagicMock()
        ready.get_stations.side_effect = FDSNNoDataException("none")
        get_stations_in_region(
            starttime=T0,
            endtime=T1,
            minlatitude=0,
            maxlatitude=1,
            minlongitude=0,
            maxlongitude=1,
            client=ready,
        )
        MockClient.assert_not_called()
        ready.get_stations.assert_called_once()


class TestGetStationsInRegion:
    @patch("swe.io.fdsn.Client")
    def test_excludes_synthetic_network_by_default(self, MockClient):
        MockClient.return_value.get_stations.return_value = _make_inventory(("SY", "XX", "YY"))
        inv = get_stations_in_region(
            starttime=T0,
            endtime=T1,
            minlatitude=30,
            maxlatitude=40,
            minlongitude=-125,
            maxlongitude=-110,
        )
        assert [n.code for n in inv] == ["XX", "YY"]
        assert DEFAULT_EXCLUDED_NETWORKS == ("SY",)

    @patch("swe.io.fdsn.Client")
    def test_query_parameters_passed(self, MockClient):
        MockClient.return_value.get_stations.return_value = _make_inventory()
        get_stations_in_region(
            starttime=T0,
            endtime=T1,
            minlatitude=30,
            maxlatitude=40,
            minlongitude=-125,
            maxlongitude=-110,
            channel="?HZ",
            level="channel",
        )
        kw = MockClient.return_value.get_stations.call_args.kwargs
        assert kw["minlatitude"] == 30 and kw["maxlongitude"] == -110
        assert kw["channel"] == "?HZ" and kw["level"] == "channel"
        assert kw["starttime"] == T0 and kw["endtime"] == T1

    @patch("swe.io.fdsn.Client")
    def test_exclusion_is_configurable_and_case_insensitive(self, MockClient):
        MockClient.return_value.get_stations.return_value = _make_inventory(("SY", "XX"))
        kept = get_stations_in_region(
            starttime=T0,
            endtime=T1,
            minlatitude=0,
            maxlatitude=1,
            minlongitude=0,
            maxlongitude=1,
            exclude_networks=(),
        )
        assert [n.code for n in kept] == ["SY", "XX"]
        MockClient.return_value.get_stations.return_value = _make_inventory(("SY", "XX"))
        dropped = get_stations_in_region(
            starttime=T0,
            endtime=T1,
            minlatitude=0,
            maxlatitude=1,
            minlongitude=0,
            maxlongitude=1,
            exclude_networks=("xx",),
        )
        assert [n.code for n in dropped] == ["SY"]

    @patch("swe.io.fdsn.Client")
    def test_explicitly_requested_network_is_kept(self, MockClient):
        """network="SY" asks for the synthetic network; the default exclusion must not drop it."""
        MockClient.return_value.get_stations.return_value = _make_inventory(("SY",))
        inv = get_stations_in_region(
            starttime=T0,
            endtime=T1,
            minlatitude=0,
            maxlatitude=1,
            minlongitude=0,
            maxlongitude=1,
            network="SY",
        )
        assert [n.code for n in inv] == ["SY"]
        MockClient.return_value.get_stations.return_value = _make_inventory(("SY", "XX"))
        inv = get_stations_in_region(
            starttime=T0,
            endtime=T1,
            minlatitude=0,
            maxlatitude=1,
            minlongitude=0,
            maxlongitude=1,
            network="S*,XX",
        )
        assert [n.code for n in inv] == ["XX"]

    @patch("swe.io.fdsn.Client")
    def test_bare_string_exclusion(self, MockClient):
        """exclude_networks="SY" means the network SY, not the letters S and Y."""
        MockClient.return_value.get_stations.return_value = _make_inventory(("SY", "S", "XX"))
        inv = get_stations_in_region(
            starttime=T0,
            endtime=T1,
            minlatitude=0,
            maxlatitude=1,
            minlongitude=0,
            maxlongitude=1,
            exclude_networks="SY",
        )
        assert [n.code for n in inv] == ["S", "XX"]

    @patch("swe.io.fdsn.Client")
    def test_no_data_returns_empty_inventory(self, MockClient):
        MockClient.return_value.get_stations.side_effect = FDSNNoDataException("none")
        inv = get_stations_in_region(
            starttime=T0, endtime=T1, minlatitude=0, maxlatitude=1, minlongitude=0, maxlongitude=1
        )
        assert isinstance(inv, Inventory) and len(inv) == 0


class TestInventoryToDataframe:
    def test_station_level(self):
        df = inventory_to_dataframe(_make_inventory(("XX", "YY")))
        assert list(df.columns) == [
            "network",
            "station",
            "latitude",
            "longitude",
            "elevation",
            "start_date",
            "end_date",
        ]
        assert df["network"].tolist() == ["XX", "YY"]
        assert df["start_date"].iloc[0].startswith("2010-01-01")
        assert df["end_date"].isna().all()

    def test_channel_level_and_far_future_end(self):
        df = inventory_to_dataframe(
            _make_inventory(end_date=UTCDateTime("2599-12-31")), level="channel"
        )
        assert df[["location", "channel", "sample_rate"]].iloc[0].tolist() == ["00", "BHZ", 40.0]
        assert df["end_date"].iloc[0].startswith("2599-12-31")

    def test_empty_and_bad_level(self):
        assert inventory_to_dataframe(Inventory(networks=[], source="t")).empty
        with pytest.raises(ValueError, match="level"):
            inventory_to_dataframe(_make_inventory(), level="response")


class TestEvents:
    @patch("swe.io.fdsn.Client")
    def test_get_events_passthrough_and_default_client(self, MockClient):
        MockClient.return_value.get_events.return_value = Catalog()
        get_events(
            starttime=T0, endtime=T1, minmagnitude=6.0, latitude=10.0, longitude=20.0, maxradius=2.0
        )
        MockClient.assert_called_once_with("USGS")
        kw = MockClient.return_value.get_events.call_args.kwargs
        assert kw == dict(
            starttime=T0, endtime=T1, minmagnitude=6.0, latitude=10.0, longitude=20.0, maxradius=2.0
        )

    @patch("swe.io.fdsn.Client")
    def test_get_events_no_data(self, MockClient):
        MockClient.return_value.get_events.side_effect = FDSNNoDataException("none")
        cat = get_events(starttime=T0, endtime=T1)
        assert isinstance(cat, Catalog) and len(cat) == 0

    def test_catalog_to_dataframe(self):
        o = Origin(time=T0, latitude=10.5, longitude=20.8, depth=12000.0)
        m = Magnitude(mag=6.1, magnitude_type="mb")
        ev = Event(origins=[o], magnitudes=[m])
        ev.preferred_origin_id = o.resource_id
        bare = Event()
        df = catalog_to_dataframe(Catalog(events=[ev, bare]))
        row = df.iloc[0]
        assert row["depth_km"] == pytest.approx(12.0) and row["magnitude_type"] == "mb"
        assert row["time"].startswith("2020-01-01T00:00:00")
        assert df.iloc[1][["time", "latitude", "magnitude"]].isna().all()
