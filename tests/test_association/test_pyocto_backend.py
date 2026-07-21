"""Tests for swe.association._pyocto_backend - PyOcto adapter (mocked)."""

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
def mock_pyocto():
    """Create a mock pyocto module."""
    mock_module = MagicMock()

    # Mock VelocityModel1D
    mock_vel_model = MagicMock()
    mock_module.associator.VelocityModel1D.return_value = mock_vel_model

    # Mock OctoAssociator
    mock_associator = MagicMock()
    events_df = pd.DataFrame({"time": [10.0], "x": [0.0], "y": [0.0], "z": [5.0]})
    assignments = np.array([0, 0, 0])
    mock_associator.associate.return_value = (events_df, assignments)
    mock_module.OctoAssociator.return_value = mock_associator

    return mock_module


class TestPyOctoBackend:
    """PyOcto backend tests with mocked pyocto."""

    def test_associate_returns_result(self, station_data, mock_pyocto):
        picks, stations, locs = station_data
        with patch.dict(sys.modules, {"pyocto": mock_pyocto}):
            from swe.association._pyocto_backend import associate_picks_pyocto

            result = associate_picks_pyocto(picks, stations, locs)

        assert len(result) == 6
        assert isinstance(result.labels, np.ndarray)
        assert len(result.labels) == 3

    def test_empty_picks(self, mock_pyocto):
        with patch.dict(sys.modules, {"pyocto": mock_pyocto}):
            from swe.association._pyocto_backend import associate_picks_pyocto

            result = associate_picks_pyocto(
                {"STA1": []},
                ["STA1"],
                {"STA1": [0, 0, 0]},
            )
        assert len(result.labels) == 0

    def test_velocity_model_passed(self, station_data, mock_pyocto):
        picks, stations, locs = station_data
        with patch.dict(sys.modules, {"pyocto": mock_pyocto}):
            from swe.association._pyocto_backend import associate_picks_pyocto

            associate_picks_pyocto(
                picks,
                stations,
                locs,
                velocity_model={"vp": 5.5, "vs": 3.2},
            )
        mock_pyocto.associator.VelocityModel1D.assert_called_once()

    def test_custom_limits(self, station_data, mock_pyocto):
        picks, stations, locs = station_data
        with patch.dict(sys.modules, {"pyocto": mock_pyocto}):
            from swe.association._pyocto_backend import associate_picks_pyocto

            associate_picks_pyocto(
                picks,
                stations,
                locs,
                xlim=(-10, 10),
                ylim=(-10, 10),
                zlim=(0, 20),
            )
        call_kwargs = mock_pyocto.OctoAssociator.call_args
        assert call_kwargs[1]["xlim"] == (-10, 10)

    def test_missing_pyocto_raises(self):
        # Ensure pyocto is not available
        with patch.dict(sys.modules, {"pyocto": None}):
            from swe.association._pyocto_backend import _require_pyocto

            with pytest.raises(ImportError, match="pyocto"):
                _require_pyocto()

    def test_result_tuple_unpacking(self, station_data, mock_pyocto):
        picks, stations, locs = station_data
        with patch.dict(sys.modules, {"pyocto": mock_pyocto}):
            from swe.association._pyocto_backend import associate_picks_pyocto

            sd, rsd, _pso, _pt, labels, _quality = associate_picks_pyocto(
                picks,
                stations,
                locs,
            )
        assert isinstance(sd, dict)
        assert isinstance(rsd, dict)
        assert isinstance(labels, np.ndarray)
