"""Tests for swe.association._gamma_backend - GaMMA adapter (mocked)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from obspy import UTCDateTime


@pytest.fixture()
def station_data():
    stations = ["STA1", "STA2", "STA3"]
    location_dict = {
        "STA1": [-118.50, 34.00, 0.0],
        "STA2": [-118.48, 34.01, 0.0],
        "STA3": [-118.46, 34.02, 0.0],
    }
    t0 = UTCDateTime("2024-01-01T00:00:00")
    station_picks = {
        "STA1": [t0 + 10.0],
        "STA2": [t0 + 10.1],
        "STA3": [t0 + 10.2],
    }
    return station_picks, stations, location_dict


@pytest.fixture()
def mock_gamma():
    """Create a mock gamma.utils module."""
    mock_utils = MagicMock()

    catalogs = pd.DataFrame(
        {
            "time": [UTCDateTime("2024-01-01T00:00:10").datetime],
            "longitude": [-118.48],
            "latitude": [34.01],
            "depth(km)": [5.0],
        }
    )
    assignments = pd.DataFrame(
        {
            "event_index": [0, 0, 0],
        }
    )
    mock_utils.association.return_value = (catalogs, assignments)
    return mock_utils


class TestGaMMABackend:
    """GaMMA backend tests with mocked gamma."""

    def test_associate_returns_result(self, station_data, mock_gamma):
        picks, stations, locs = station_data
        mock_module = MagicMock()
        mock_module.utils = mock_gamma
        with patch.dict(sys.modules, {"gamma": mock_module, "gamma.utils": mock_gamma}):
            from swe.association._gamma_backend import associate_picks_gamma

            result = associate_picks_gamma(picks, stations, locs)

        assert len(result) == 6
        assert isinstance(result.labels, np.ndarray)
        assert len(result.labels) == 3

    def test_labels_from_assignments(self, station_data, mock_gamma):
        picks, stations, locs = station_data
        mock_module = MagicMock()
        mock_module.utils = mock_gamma
        with patch.dict(sys.modules, {"gamma": mock_module, "gamma.utils": mock_gamma}):
            from swe.association._gamma_backend import associate_picks_gamma

            result = associate_picks_gamma(picks, stations, locs)

        # All 3 picks should be assigned to event 0
        assert (result.labels == 0).sum() == 3

    def test_empty_picks(self, mock_gamma):
        mock_module = MagicMock()
        mock_module.utils = mock_gamma
        with patch.dict(sys.modules, {"gamma": mock_module, "gamma.utils": mock_gamma}):
            from swe.association._gamma_backend import associate_picks_gamma

            result = associate_picks_gamma(
                {"STA1": []},
                ["STA1"],
                {"STA1": [0, 0, 0]},
            )
        assert len(result.labels) == 0

    def test_center_auto_inference(self, station_data, mock_gamma):
        _picks, _stations, locs = station_data
        mock_module = MagicMock()
        mock_module.utils = mock_gamma
        with patch.dict(sys.modules, {"gamma": mock_module, "gamma.utils": mock_gamma}):
            from swe.association._gamma_backend import _build_gamma_config

            config = _build_gamma_config(locs)

        center = config["center"]
        assert abs(center[0] - (-118.48)) < 0.01
        assert abs(center[1] - 34.01) < 0.01

    def test_custom_center(self, station_data, mock_gamma):
        _picks, _stations, locs = station_data
        mock_module = MagicMock()
        mock_module.utils = mock_gamma
        with patch.dict(sys.modules, {"gamma": mock_module, "gamma.utils": mock_gamma}):
            from swe.association._gamma_backend import _build_gamma_config

            config = _build_gamma_config(locs, center=(-118.5, 34.0))

        assert config["center"] == (-118.5, 34.0)

    def test_missing_gamma_raises(self):
        with patch.dict(sys.modules, {"gamma": None, "gamma.utils": None}):
            from swe.association._gamma_backend import _require_gamma

            with pytest.raises(ImportError, match="gamma-seismology"):
                _require_gamma()

    def test_result_tuple_unpacking(self, station_data, mock_gamma):
        picks, stations, locs = station_data
        mock_module = MagicMock()
        mock_module.utils = mock_gamma
        with patch.dict(sys.modules, {"gamma": mock_module, "gamma.utils": mock_gamma}):
            from swe.association._gamma_backend import associate_picks_gamma

            sd, rsd, _pso, _pt, labels, _quality = associate_picks_gamma(
                picks,
                stations,
                locs,
            )
        assert isinstance(sd, dict)
        assert isinstance(rsd, dict)
        assert isinstance(labels, np.ndarray)

    def test_config_contains_required_keys(self, station_data):
        from swe.association._gamma_backend import _build_gamma_config

        _, _, locs = station_data
        config = _build_gamma_config(locs)
        assert "center" in config
        assert "method" in config
        assert config["method"] == "BGMM"
        assert "min_picks_per_eq" in config
        assert "use_amplitude" in config
