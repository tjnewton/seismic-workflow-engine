"""Tests for swe.velocity.model."""

from __future__ import annotations

import numpy as np
import pytest

from swe.velocity.model import (
    TravelTimeGrid,
    VelocityModel,
    build_velocity_model_3d,
    load_travel_time_grid,
    read_velocity_model_1d,
    read_velocity_model_3d,
    save_travel_time_grid,
    save_velocity_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_model():
    """Build a minimal VelocityModel for reuse across tests."""
    x = np.linspace(0.0, 100.0, 6)
    y = np.linspace(0.0, 100.0, 6)
    z = np.linspace(0.0, 50.0, 4)
    slowness = np.full((5, 5, 3), 1.0 / 5000.0)
    return VelocityModel(x=x, y=y, z=z, slowness=slowness, model_type="3d")


def _make_simple_ttg():
    """Build a minimal TravelTimeGrid for reuse across tests."""
    gx = np.linspace(0.0, 100.0, 6)
    gy = np.linspace(0.0, 100.0, 6)
    gz = np.linspace(0.0, 50.0, 4)
    stations = np.array([[10.0, 20.0, 0.0], [80.0, 70.0, 0.0]])
    tt = np.random.default_rng(42).random((2, 6, 6, 4))
    return TravelTimeGrid(
        grid_x=gx,
        grid_y=gy,
        grid_z=gz,
        station_coords=stations,
        travel_times=tt,
    )


# ---------------------------------------------------------------------------
# TestReadVelocityModel1d
# ---------------------------------------------------------------------------


class TestReadVelocityModel1d:
    def test_shape_and_bounds(self, tmp_path):
        """Grid axes and slowness shape should match grid_bounds/spacing."""
        model_file = tmp_path / "model.txt"
        model_file.write_text("0.0 3000.0\n25.0 5000.0\n50.0 6000.0\n")

        model = read_velocity_model_1d(
            model_file,
            grid_bounds=((0.0, 100.0), (0.0, 100.0), (0.0, 50.0)),
            grid_spacing=(20.0, 20.0, 10.0),
        )
        assert model.model_type == "1d"
        assert model.slowness.shape == (
            len(model.x) - 1,
            len(model.y) - 1,
            len(model.z) - 1,
        )

    def test_slowness_values(self, tmp_path):
        """Slowness should be 1/velocity at cell centres."""
        model_file = tmp_path / "constant.txt"
        model_file.write_text("0.0 5000.0\n100.0 5000.0\n")

        model = read_velocity_model_1d(
            model_file,
            grid_bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 100.0)),
            grid_spacing=(10.0, 10.0, 10.0),
        )
        np.testing.assert_allclose(model.slowness, 1.0 / 5000.0)

    def test_laterally_constant(self, tmp_path):
        """1-D model should have identical slowness columns in x-y."""
        model_file = tmp_path / "model.txt"
        model_file.write_text("0.0 3000.0\n50.0 6000.0\n")

        model = read_velocity_model_1d(
            model_file,
            grid_bounds=((0.0, 40.0), (0.0, 40.0), (0.0, 50.0)),
            grid_spacing=(10.0, 10.0, 10.0),
        )
        # All x-y slices should be identical.
        for ix in range(model.slowness.shape[0]):
            for iy in range(model.slowness.shape[1]):
                np.testing.assert_array_equal(model.slowness[ix, iy, :], model.slowness[0, 0, :])

    def test_max_cells_per_axis_exceeded(self, tmp_path):
        """Should raise ValueError when grid is too large."""
        model_file = tmp_path / "model.txt"
        model_file.write_text("0.0 5000.0\n100.0 5000.0\n")

        with pytest.raises(ValueError, match="max_cells_per_axis"):
            read_velocity_model_1d(
                model_file,
                grid_bounds=((0.0, 1000.0), (0.0, 10.0), (0.0, 10.0)),
                grid_spacing=(1.0, 10.0, 10.0),
                max_cells_per_axis=50,
            )


# ---------------------------------------------------------------------------
# TestReadVelocityModel3d
# ---------------------------------------------------------------------------


class TestReadVelocityModel3d:
    def test_round_trip(self, tmp_path):
        """Save then load should recover the same arrays."""
        original = _make_simple_model()
        path = tmp_path / "model.npz"
        save_velocity_model(original, path)

        loaded = read_velocity_model_3d(path)
        assert loaded.model_type == "3d"
        np.testing.assert_array_equal(loaded.x, original.x)
        np.testing.assert_array_equal(loaded.y, original.y)
        np.testing.assert_array_equal(loaded.z, original.z)
        np.testing.assert_array_equal(loaded.slowness, original.slowness)


# ---------------------------------------------------------------------------
# TestBuildVelocityModel3d
# ---------------------------------------------------------------------------


class TestBuildVelocityModel3d:
    def test_assigns_3d_model_type(self):
        x = np.linspace(0, 10, 3)
        y = np.linspace(0, 10, 3)
        z = np.linspace(0, 10, 3)
        slowness = np.ones((2, 2, 2))
        model = build_velocity_model_3d(x, y, z, slowness)
        assert model.model_type == "3d"

    def test_shape_validation(self):
        """Should reject slowness with wrong shape."""
        x = np.linspace(0, 10, 4)
        y = np.linspace(0, 10, 4)
        z = np.linspace(0, 10, 4)
        bad_slowness = np.ones((3, 3, 2))  # should be (3, 3, 3)
        with pytest.raises(ValueError, match="slowness shape"):
            build_velocity_model_3d(x, y, z, bad_slowness)

    def test_preserves_data(self):
        """Arrays should be stored correctly."""
        x = np.array([0.0, 5.0, 10.0])
        y = np.array([0.0, 5.0, 10.0])
        z = np.array([0.0, 5.0])
        slowness = np.full((2, 2, 1), 0.0002)
        model = build_velocity_model_3d(x, y, z, slowness)
        np.testing.assert_allclose(model.slowness, 0.0002)


# ---------------------------------------------------------------------------
# TestSaveLoadVelocityModel
# ---------------------------------------------------------------------------


class TestSaveLoadVelocityModel:
    def test_round_trip_integrity(self, tmp_path):
        """Saved and reloaded arrays should be identical."""
        original = _make_simple_model()
        path = save_velocity_model(original, tmp_path / "vm.npz")
        data = np.load(path)
        np.testing.assert_array_equal(data["x"], original.x)
        np.testing.assert_array_equal(data["slowness"], original.slowness)

    def test_returns_path(self, tmp_path):
        model = _make_simple_model()
        result = save_velocity_model(model, tmp_path / "vm.npz")
        assert result == tmp_path / "vm.npz"


# ---------------------------------------------------------------------------
# TestSaveLoadTravelTimeGrid
# ---------------------------------------------------------------------------


class TestSaveLoadTravelTimeGrid:
    def test_round_trip_integrity(self, tmp_path):
        """Saved and reloaded TravelTimeGrid should be identical."""
        original = _make_simple_ttg()
        path = tmp_path / "ttg.npz"
        save_travel_time_grid(original, path)
        loaded = load_travel_time_grid(path)

        np.testing.assert_array_equal(loaded.grid_x, original.grid_x)
        np.testing.assert_array_equal(loaded.grid_y, original.grid_y)
        np.testing.assert_array_equal(loaded.grid_z, original.grid_z)
        np.testing.assert_array_equal(loaded.station_coords, original.station_coords)
        np.testing.assert_array_equal(loaded.travel_times, original.travel_times)

    def test_returns_path(self, tmp_path):
        ttg = _make_simple_ttg()
        result = save_travel_time_grid(ttg, tmp_path / "ttg.npz")
        assert result == tmp_path / "ttg.npz"
