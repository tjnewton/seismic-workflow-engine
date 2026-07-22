"""Tests for swe.velocity.raytrace."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestComputeTravelTimes:
    """Tests for compute_travel_times() with mocked ttcrpy."""

    def _make_mock_grid(self, n_receivers):
        """Create a mock Grid3d that returns predictable travel times."""
        mock_grid = MagicMock()
        call_count = [0]

        def mock_raytrace(src, rcv, slowness=None):
            tt = np.full(len(rcv), 1.0 + call_count[0] * 0.1)
            call_count[0] += 1
            rays = None
            return tt, rays

        mock_grid.raytrace.side_effect = mock_raytrace
        return mock_grid

    def test_output_shape(self, tmp_path):
        """Result should be (N_sources, M_receivers)."""
        model_file = tmp_path / "model.txt"
        model_file.write_text("0.0 5.0\n10.0 6.0\n20.0 7.0\n")

        mock_grid = self._make_mock_grid(n_receivers=4)
        mock_grid3d_cls = MagicMock(return_value=mock_grid)
        mock_ttcrpy = MagicMock()
        mock_ttcrpy.rgrid.Grid3d = mock_grid3d_cls

        with patch.dict("sys.modules", {"ttcrpy": mock_ttcrpy, "ttcrpy.rgrid": mock_ttcrpy.rgrid}):
            import importlib

            import swe.velocity.raytrace as mod

            importlib.reload(mod)

            sources = np.array([[0.0, 0.0, 5.0], [1.0, 1.0, 10.0]])
            receivers = np.array(
                [
                    [5.0, 0.0, 0.0],
                    [5.0, 5.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [10.0, 5.0, 0.0],
                ]
            )
            result = mod.compute_travel_times(str(model_file), sources, receivers)

        assert result.shape == (2, 4)

    def test_import_error_without_ttcrpy(self, tmp_path):
        """Should raise ImportError when ttcrpy is not installed."""
        import importlib

        import swe.velocity.raytrace as mod

        with patch.dict("sys.modules", {"ttcrpy": None, "ttcrpy.rgrid": None}):
            importlib.reload(mod)
            sources = np.array([[0.0, 0.0, 0.0]])
            receivers = np.array([[1.0, 0.0, 0.0]])
            with pytest.raises(ImportError, match="ttcrpy"):
                mod.compute_travel_times("model.txt", sources, receivers)

    def test_unsupported_model_type(self, tmp_path):
        """Should raise ValueError for unsupported model types."""
        model_file = tmp_path / "model.txt"
        model_file.write_text("0.0 5.0\n10.0 6.0\n")

        mock_ttcrpy = MagicMock()
        with patch.dict("sys.modules", {"ttcrpy": mock_ttcrpy, "ttcrpy.rgrid": mock_ttcrpy.rgrid}):
            import importlib

            import swe.velocity.raytrace as mod

            importlib.reload(mod)

            sources = np.array([[0.0, 0.0, 0.0]])
            receivers = np.array([[1.0, 0.0, 0.0]])
            with pytest.raises(ValueError, match="Unsupported model_type"):
                mod.compute_travel_times(str(model_file), sources, receivers, model_type="3d")

    def test_1d_model_loading(self, tmp_path):
        """Should correctly load a two-column velocity model."""
        model_file = tmp_path / "velocity.txt"
        model_file.write_text("0.0 3.0\n5.0 5.0\n15.0 6.5\n")

        mock_grid = self._make_mock_grid(n_receivers=2)
        mock_grid3d_cls = MagicMock(return_value=mock_grid)
        mock_ttcrpy = MagicMock()
        mock_ttcrpy.rgrid.Grid3d = mock_grid3d_cls

        with patch.dict("sys.modules", {"ttcrpy": mock_ttcrpy, "ttcrpy.rgrid": mock_ttcrpy.rgrid}):
            import importlib

            import swe.velocity.raytrace as mod

            importlib.reload(mod)

            sources = np.array([[0.0, 0.0, 2.0]])
            receivers = np.array([[5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
            result = mod.compute_travel_times(str(model_file), sources, receivers)

        assert result.shape == (1, 2)
        # Grid3d constructor should have been called
        mock_grid3d_cls.assert_called_once()

    def test_single_source_single_receiver(self, tmp_path):
        """Should handle degenerate case of one source and one receiver."""
        model_file = tmp_path / "model.txt"
        model_file.write_text("0.0 5.0\n10.0 6.0\n")

        mock_grid = self._make_mock_grid(n_receivers=1)
        mock_grid3d_cls = MagicMock(return_value=mock_grid)
        mock_ttcrpy = MagicMock()
        mock_ttcrpy.rgrid.Grid3d = mock_grid3d_cls

        with patch.dict("sys.modules", {"ttcrpy": mock_ttcrpy, "ttcrpy.rgrid": mock_ttcrpy.rgrid}):
            import importlib

            import swe.velocity.raytrace as mod

            importlib.reload(mod)

            sources = np.array([[0.0, 0.0, 0.0]])
            receivers = np.array([[5.0, 5.0, 5.0]])
            result = mod.compute_travel_times(str(model_file), sources, receivers)

        assert result.shape == (1, 1)


# ---------------------------------------------------------------------------
# TestComputeTravelTimesWithVelocityModel
# ---------------------------------------------------------------------------


class TestComputeTravelTimesWithVelocityModel:
    """Tests for compute_travel_times() when passed a VelocityModel."""

    def _make_mock_grid(self, n_receivers):
        mock_grid = MagicMock()
        call_count = [0]

        def mock_raytrace(src, rcv, slowness=None):
            tt = np.full(len(rcv), 0.5 + call_count[0] * 0.1)
            call_count[0] += 1
            return tt, None

        mock_grid.raytrace.side_effect = mock_raytrace
        return mock_grid

    def test_accepts_velocity_model(self):
        """Should accept a VelocityModel instead of a file path."""
        from swe.velocity.model import VelocityModel

        x = np.linspace(0, 100, 6)
        y = np.linspace(0, 100, 6)
        z = np.linspace(0, 50, 4)
        slowness = np.full((5, 5, 3), 1.0 / 5000.0)
        vm = VelocityModel(x=x, y=y, z=z, slowness=slowness, model_type="3d")

        mock_grid = self._make_mock_grid(n_receivers=3)
        mock_grid3d_cls = MagicMock(return_value=mock_grid)
        mock_ttcrpy = MagicMock()
        mock_ttcrpy.rgrid.Grid3d = mock_grid3d_cls

        with patch.dict("sys.modules", {"ttcrpy": mock_ttcrpy, "ttcrpy.rgrid": mock_ttcrpy.rgrid}):
            import importlib

            import swe.velocity.raytrace as mod

            importlib.reload(mod)

            sources = np.array([[10.0, 10.0, 10.0], [50.0, 50.0, 25.0]])
            receivers = np.array(
                [
                    [20.0, 20.0, 0.0],
                    [80.0, 80.0, 0.0],
                    [50.0, 50.0, 0.0],
                ]
            )
            result = mod.compute_travel_times(vm, sources, receivers)

        assert result.shape == (2, 3)
        # cell_slowness=True should have been used.
        mock_grid3d_cls.assert_called_once()
        call_kwargs = mock_grid3d_cls.call_args
        assert (
            call_kwargs[1].get("cell_slowness") is True
            or call_kwargs.kwargs.get("cell_slowness") is True
        )

    def test_ignores_model_type_arg(self):
        """When VelocityModel is passed, model_type parameter is ignored."""
        from swe.velocity.model import VelocityModel

        x = np.linspace(0, 10, 3)
        y = np.linspace(0, 10, 3)
        z = np.linspace(0, 10, 3)
        slowness = np.full((2, 2, 2), 0.0002)
        vm = VelocityModel(x=x, y=y, z=z, slowness=slowness, model_type="1d")

        mock_grid = self._make_mock_grid(n_receivers=1)
        mock_grid3d_cls = MagicMock(return_value=mock_grid)
        mock_ttcrpy = MagicMock()
        mock_ttcrpy.rgrid.Grid3d = mock_grid3d_cls

        with patch.dict("sys.modules", {"ttcrpy": mock_ttcrpy, "ttcrpy.rgrid": mock_ttcrpy.rgrid}):
            import importlib

            import swe.velocity.raytrace as mod

            importlib.reload(mod)

            sources = np.array([[5.0, 5.0, 5.0]])
            receivers = np.array([[0.0, 0.0, 0.0]])
            # model_type="3d" should NOT raise, because VelocityModel is used.
            result = mod.compute_travel_times(vm, sources, receivers, model_type="3d")

        assert result.shape == (1, 1)


# ---------------------------------------------------------------------------
# TestComputeTravelTimeGrid
# ---------------------------------------------------------------------------


class TestComputeTravelTimeGrid:
    """Tests for compute_travel_time_grid() with mocked ttcrpy."""

    def test_output_shape(self):
        """Travel times should have shape (n_stations, nx, ny, nz)."""
        from swe.velocity.model import VelocityModel

        x = np.linspace(0, 100, 6)
        y = np.linspace(0, 100, 6)
        z = np.linspace(0, 50, 4)
        slowness = np.full((5, 5, 3), 1.0 / 5000.0)
        vm = VelocityModel(x=x, y=y, z=z, slowness=slowness, model_type="3d")

        stations = np.array([[10.0, 10.0, 0.0], [90.0, 90.0, 0.0]])

        mock_grid = MagicMock()

        def mock_raytrace(src, rcv, slowness=None):
            n = len(rcv)
            return np.ones(n) * 0.01, None

        mock_grid.raytrace.side_effect = mock_raytrace
        mock_grid3d_cls = MagicMock(return_value=mock_grid)
        mock_ttcrpy = MagicMock()
        mock_ttcrpy.rgrid.Grid3d = mock_grid3d_cls

        with patch.dict("sys.modules", {"ttcrpy": mock_ttcrpy, "ttcrpy.rgrid": mock_ttcrpy.rgrid}):
            import importlib

            import swe.velocity.raytrace as mod

            importlib.reload(mod)

            ttg = mod.compute_travel_time_grid(vm, stations)

        assert ttg.travel_times.shape == (2, 6, 6, 4)
        np.testing.assert_array_equal(ttg.grid_x, x)
        np.testing.assert_array_equal(ttg.grid_y, y)
        np.testing.assert_array_equal(ttg.grid_z, z)
        np.testing.assert_array_equal(ttg.station_coords, stations)

    def test_single_station(self):
        """Should work with a single station."""
        from swe.velocity.model import VelocityModel

        x = np.linspace(0, 10, 3)
        y = np.linspace(0, 10, 3)
        z = np.linspace(0, 10, 3)
        slowness = np.full((2, 2, 2), 0.0002)
        vm = VelocityModel(x=x, y=y, z=z, slowness=slowness, model_type="1d")

        stations = np.array([[5.0, 5.0, 5.0]])

        mock_grid = MagicMock()
        mock_grid.raytrace.return_value = (np.ones(27) * 0.005, None)
        mock_grid3d_cls = MagicMock(return_value=mock_grid)
        mock_ttcrpy = MagicMock()
        mock_ttcrpy.rgrid.Grid3d = mock_grid3d_cls

        with patch.dict("sys.modules", {"ttcrpy": mock_ttcrpy, "ttcrpy.rgrid": mock_ttcrpy.rgrid}):
            import importlib

            import swe.velocity.raytrace as mod

            importlib.reload(mod)

            ttg = mod.compute_travel_time_grid(vm, stations)

        assert ttg.travel_times.shape == (1, 3, 3, 3)

    def test_import_error(self):
        """Should raise ImportError when ttcrpy is unavailable."""
        from swe.velocity.model import VelocityModel

        x = np.linspace(0, 10, 3)
        y = np.linspace(0, 10, 3)
        z = np.linspace(0, 10, 3)
        slowness = np.full((2, 2, 2), 0.0002)
        vm = VelocityModel(x=x, y=y, z=z, slowness=slowness, model_type="1d")

        stations = np.array([[5.0, 5.0, 5.0]])

        with patch.dict("sys.modules", {"ttcrpy": None, "ttcrpy.rgrid": None}):
            import importlib

            import swe.velocity.raytrace as mod

            importlib.reload(mod)

            with pytest.raises(ImportError, match="ttcrpy"):
                mod.compute_travel_time_grid(vm, stations)
