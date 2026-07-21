"""Tests for swe.association._dispatch - unified associate_picks dispatcher."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from obspy import UTCDateTime


@pytest.fixture()
def station_data():
    """Minimal station data for dispatch tests."""
    stations = ["STA1", "STA2", "STA3", "STA4"]
    location_dict = {
        "STA1": [-118.50, 34.00, 0.0],
        "STA2": [-118.48, 34.01, 0.0],
        "STA3": [-118.46, 34.02, 0.0],
        "STA4": [-118.44, 34.03, 0.0],
    }
    t0 = UTCDateTime("2024-01-01T00:00:00")
    station_picks = {
        "STA1": [t0 + 10.0, t0 + 30.0],
        "STA2": [t0 + 10.1, t0 + 30.1],
        "STA3": [t0 + 10.2, t0 + 30.2],
        "STA4": [t0 + 10.3, t0 + 30.3],
    }
    return station_picks, stations, location_dict


class TestDispatchRouting:
    """Verify method routing."""

    def test_dbscan_is_default(self, station_data):
        from swe.association import associate_picks

        picks, stations, locs = station_data
        result = associate_picks(picks, stations, locs)
        assert len(result) == 6
        assert isinstance(result.labels, np.ndarray)

    def test_dbscan_explicit_method(self, station_data):
        from swe.association import associate_picks

        picks, stations, locs = station_data
        result = associate_picks(picks, stations, locs, method="dbscan")
        assert len(result) == 6

    @patch("swe.association._dispatch._dbscan_associate")
    def test_dbscan_calls_clustering(self, mock_dbscan, station_data):
        from swe.association._dispatch import associate_picks

        mock_dbscan.return_value = ({}, {}, [], [], np.array([], dtype=int))
        picks, stations, locs = station_data
        associate_picks(picks, stations, locs, method="dbscan")
        mock_dbscan.assert_called_once()

    def test_pyocto_routes_to_backend(self, station_data):
        from swe.association._dispatch import associate_picks

        picks, stations, locs = station_data
        with patch("swe.association._pyocto_backend.associate_picks_pyocto") as mock_pyocto:
            mock_pyocto.return_value = ({}, {}, [], [], np.array([], dtype=int))
            associate_picks(picks, stations, locs, method="pyocto")
            mock_pyocto.assert_called_once()

    def test_gamma_routes_to_backend(self, station_data):
        from swe.association._dispatch import associate_picks

        picks, stations, locs = station_data
        with patch("swe.association._gamma_backend.associate_picks_gamma") as mock_gamma:
            mock_gamma.return_value = ({}, {}, [], [], np.array([], dtype=int))
            associate_picks(picks, stations, locs, method="gamma")
            mock_gamma.assert_called_once()

    def test_unsupported_method_raises(self, station_data):
        from swe.association import associate_picks

        picks, stations, locs = station_data
        with pytest.raises(ValueError, match="Unsupported"):
            associate_picks(picks, stations, locs, method="unknown")

    def test_method_case_insensitive(self, station_data):
        from swe.association import associate_picks

        picks, stations, locs = station_data
        result = associate_picks(picks, stations, locs, method="DBSCAN")
        assert isinstance(result.labels, np.ndarray)


class TestSWEPickInput:
    """Verify associate_picks accepts list[SWEPick] directly."""

    def test_swe_pick_list_accepted(self, station_data):
        from swe.association import associate_picks
        from swe.picks import SWEPick

        _, stations, locs = station_data
        t0 = UTCDateTime("2024-01-01T00:00:00")
        swe_picks = [
            SWEPick("STA1", "P", t0 + 10.0, 0.9),
            SWEPick("STA1", "S", t0 + 15.0, 0.7),
            SWEPick("STA2", "P", t0 + 10.1, 0.8),
            SWEPick("STA2", "S", t0 + 15.1, 0.6),
            SWEPick("STA3", "P", t0 + 10.2, 0.85),
            SWEPick("STA3", "S", t0 + 15.2, 0.65),
            SWEPick("STA4", "P", t0 + 10.3, 0.88),
            SWEPick("STA4", "S", t0 + 15.3, 0.68),
        ]
        result = associate_picks(swe_picks, stations, locs)
        assert isinstance(result.labels, np.ndarray)


class TestBackwardCompatibility:
    """Verify backward compatibility is preserved."""

    def test_tuple_unpacking(self, station_data):
        from swe.association import associate_picks

        picks, stations, locs = station_data
        sd, rsd, _pso, _pt, labels, _quality = associate_picks(picks, stations, locs)
        assert isinstance(sd, dict)
        assert isinstance(rsd, dict)
        assert isinstance(labels, np.ndarray)

    def test_old_import_path(self):
        """Importing from clustering module still works."""
        from swe.association.clustering import associate_picks

        assert callable(associate_picks)

    def test_dispatch_import_path(self):
        """Importing from __init__ gets the dispatch version."""
        from swe.association import associate_picks

        assert callable(associate_picks)

    def test_association_result_namedtuple(self, station_data):
        from swe.association import AssociationResult, associate_picks

        picks, stations, locs = station_data
        result = associate_picks(picks, stations, locs)
        assert isinstance(result, AssociationResult)
        assert hasattr(result, "station_distance")
        assert hasattr(result, "reverse_station_distance")
        assert hasattr(result, "pick_station_order")
        assert hasattr(result, "pick_times")
        assert hasattr(result, "labels")


class TestConversionUtilities:
    """Test shared conversion helpers."""

    def test_picks_to_dataframe(self):
        from swe.association._dispatch import picks_to_dataframe

        t0 = UTCDateTime("2024-01-01")
        picks = {"STA1": [t0, t0 + 1], "STA2": [t0 + 2]}
        df = picks_to_dataframe(picks)
        assert len(df) == 3
        assert set(df.columns) >= {"station", "time", "phase"}
        assert df["phase"].tolist() == ["P", "P", "P"]

    def test_picks_to_dataframe_with_phases(self):
        from swe.association._dispatch import picks_to_dataframe

        t0 = UTCDateTime("2024-01-01")
        picks = {"STA1": [t0, t0 + 1]}
        phases = {"STA1": ["P", "S"]}
        df = picks_to_dataframe(picks, pick_phases=phases)
        assert df["phase"].tolist() == ["P", "S"]

    def test_stations_to_dataframe(self):
        from swe.association._dispatch import stations_to_dataframe

        locs = {"STA1": [-118.5, 34.0, 100.0], "STA2": [-118.4, 34.1, 200.0]}
        df = stations_to_dataframe(locs)
        assert len(df) == 2
        assert set(df.columns) >= {"station", "longitude", "latitude", "elevation"}
