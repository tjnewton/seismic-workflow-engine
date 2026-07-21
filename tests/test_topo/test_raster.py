"""Tests for swe.topo.raster."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestElevationsFromRaster:
    """Tests for elevations_from_raster() with mocked rasterio."""

    def _make_mock_src(self, *, crs="EPSG:4326", sample_values=None):
        """Create a mock rasterio dataset."""
        if sample_values is None:
            sample_values = [100.0, 200.0, 300.0]
        mock_src = MagicMock()
        mock_src.crs = crs
        mock_src.sample.return_value = iter([[v] for v in sample_values])
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)
        return mock_src

    @patch("swe.topo.raster.rasterio", create=True)
    def test_basic_sampling(self, mock_rasterio_module):
        """Should return elevation values from raster sampling."""
        import importlib
        import swe.topo.raster as mod

        mock_src = self._make_mock_src(sample_values=[100.0, 200.0, 300.0])
        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value = mock_src

        with patch.dict("sys.modules", {"rasterio": mock_rasterio}):
            importlib.reload(mod)
            coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            result = mod.elevations_from_raster("dem.tif", coords)

        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [100.0, 200.0, 300.0])

    @patch("swe.topo.raster.rasterio", create=True)
    def test_single_coordinate(self, mock_rasterio_module):
        """Should handle a single (1, 2) coordinate array."""
        import importlib
        import swe.topo.raster as mod

        mock_src = self._make_mock_src(sample_values=[42.0])
        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value = mock_src

        with patch.dict("sys.modules", {"rasterio": mock_rasterio}):
            importlib.reload(mod)
            coords = np.array([10.0, 20.0])
            result = mod.elevations_from_raster("dem.tif", coords)

        assert result.shape == (1,)
        assert result[0] == 42.0

    def test_import_error_without_rasterio(self):
        """Should raise ImportError when rasterio is not available."""
        import importlib
        import sys
        import swe.topo.raster as mod

        with patch.dict("sys.modules", {"rasterio": None}):
            importlib.reload(mod)
            coords = np.array([[1.0, 2.0]])
            with pytest.raises(ImportError, match="rasterio"):
                mod.elevations_from_raster("dem.tif", coords)

    @patch("swe.topo.raster.rasterio", create=True)
    def test_crs_reprojection(self, mock_rasterio_module):
        """Should reproject coordinates when target_crs differs from raster CRS."""
        import importlib
        import swe.topo.raster as mod

        mock_src = self._make_mock_src(
            crs="EPSG:32610", sample_values=[500.0]
        )
        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value = mock_src

        mock_transformer = MagicMock()
        mock_transformer.transform.return_value = (
            np.array([500000.0]),
            np.array([4649776.0]),
        )
        mock_pyproj = MagicMock()
        mock_pyproj.Transformer.from_crs.return_value = mock_transformer

        with patch.dict("sys.modules", {
            "rasterio": mock_rasterio,
            "pyproj": mock_pyproj,
        }):
            importlib.reload(mod)
            coords = np.array([[-122.0, 42.0]])
            result = mod.elevations_from_raster(
                "dem.tif", coords, target_crs="EPSG:4326"
            )

        assert result.shape == (1,)
        assert result[0] == 500.0
        mock_pyproj.Transformer.from_crs.assert_called_once_with(
            "EPSG:4326", "EPSG:32610", always_xy=True
        )

    @patch("swe.topo.raster.rasterio", create=True)
    def test_same_crs_no_reprojection(self, mock_rasterio_module):
        """When target_crs matches raster CRS, no reprojection should happen."""
        import importlib
        import swe.topo.raster as mod

        mock_src = self._make_mock_src(
            crs="EPSG:4326", sample_values=[150.0]
        )
        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value = mock_src

        with patch.dict("sys.modules", {"rasterio": mock_rasterio}):
            importlib.reload(mod)
            coords = np.array([[-122.0, 42.0]])
            result = mod.elevations_from_raster(
                "dem.tif", coords, target_crs="EPSG:4326"
            )

        assert result[0] == 150.0


# ---------------------------------------------------------------------------
# TestGridFromRaster
# ---------------------------------------------------------------------------

class TestGridFromRaster:
    """Tests for grid_from_raster() with mocked rasterio."""

    def _make_mock_src(self, *, nrows=5, ncols=5, crs="EPSG:32610", nodata=None):
        """Create a mock rasterio dataset with a synthetic grid."""
        mock_src = MagicMock()
        mock_src.crs = crs
        mock_src.nodata = nodata

        # Create a simple affine-like transform.
        mock_transform = MagicMock()
        mock_transform.a = 10.0   # pixel width
        mock_transform.c = 100.0  # x origin (upper-left)
        mock_transform.e = -10.0  # pixel height (negative = north-up)
        mock_transform.f = 500.0  # y origin (upper-left)
        mock_src.transform = mock_transform

        data = np.arange(nrows * ncols, dtype=np.float32).reshape(nrows, ncols)
        mock_src.read.return_value = data
        mock_src.window_transform.return_value = mock_transform
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)
        return mock_src

    def test_basic_grid_extraction(self):
        """Should return an ElevationGrid with correct shapes."""
        import importlib

        mock_src = self._make_mock_src(nrows=4, ncols=6)
        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value = mock_src

        mock_windows = MagicMock()
        mock_rasterio.windows = mock_windows

        with patch.dict("sys.modules", {
            "rasterio": mock_rasterio,
            "rasterio.windows": mock_windows,
        }):
            import swe.topo.raster as mod
            importlib.reload(mod)

            result = mod.grid_from_raster("dem.tif")

        assert result.elevation.shape == (4, 6)
        assert len(result.x) == 6
        assert len(result.y) == 4
        assert result.crs == "EPSG:32610"

    def test_with_bounds(self):
        """Should apply from_bounds windowing when bounds given."""
        import importlib

        mock_src = self._make_mock_src(nrows=3, ncols=3)
        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value = mock_src

        mock_from_bounds = MagicMock(return_value=MagicMock())
        mock_windows = MagicMock()
        mock_windows.from_bounds = mock_from_bounds
        mock_rasterio.windows = mock_windows

        with patch.dict("sys.modules", {
            "rasterio": mock_rasterio,
            "rasterio.windows": mock_windows,
        }):
            import swe.topo.raster as mod
            importlib.reload(mod)

            result = mod.grid_from_raster(
                "dem.tif",
                bounds=(100.0, 400.0, 130.0, 430.0),
            )

        # from_bounds should have been called.
        mock_from_bounds.assert_called_once()
        assert result.elevation is not None

    def test_fill_nodata(self):
        """Should call fillnodata when nodata is present and fill_nodata=True."""
        import importlib

        mock_src = self._make_mock_src(nrows=3, ncols=3, nodata=-9999.0)
        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value = mock_src

        mock_windows = MagicMock()
        mock_rasterio.windows = mock_windows

        mock_fill = MagicMock()
        mock_fillnodata = MagicMock(return_value=np.zeros((3, 3)))
        mock_fill.fillnodata = mock_fillnodata

        with patch.dict("sys.modules", {
            "rasterio": mock_rasterio,
            "rasterio.windows": mock_windows,
            "rasterio.fill": mock_fill,
        }):
            import swe.topo.raster as mod
            importlib.reload(mod)

            mod.grid_from_raster("dem.tif", fill_nodata=True)

        mock_fillnodata.assert_called_once()

    def test_import_error_without_rasterio(self):
        """Should raise ImportError when rasterio is not available."""
        import importlib
        import swe.topo.raster as mod

        with patch.dict("sys.modules", {"rasterio": None, "rasterio.windows": None}):
            importlib.reload(mod)
            with pytest.raises(ImportError, match="rasterio"):
                mod.grid_from_raster("dem.tif")


# ---------------------------------------------------------------------------
# TestTerrainCorrection
# ---------------------------------------------------------------------------

class TestTerrainCorrection:
    """Tests for terrain_correction() with mocked rasterio."""

    def test_updates_z_column(self):
        """Should replace z-values with sampled elevations."""
        import importlib

        mock_src = MagicMock()
        mock_src.crs = "EPSG:32610"
        mock_src.sample.return_value = iter([[100.0], [200.0], [300.0]])
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value = mock_src

        mock_windows = MagicMock()
        mock_rasterio.windows = mock_windows

        with patch.dict("sys.modules", {
            "rasterio": mock_rasterio,
            "rasterio.windows": mock_windows,
        }):
            import swe.topo.raster as mod
            importlib.reload(mod)

            coords = np.array([
                [10.0, 20.0, 0.0],
                [30.0, 40.0, 0.0],
                [50.0, 60.0, 0.0],
            ])
            result = mod.terrain_correction(coords, "dem.tif")

        # z-column should be updated, x-y should be unchanged.
        np.testing.assert_array_equal(result[:, :2], coords[:, :2])
        np.testing.assert_array_equal(result[:, 2], [100.0, 200.0, 300.0])

    def test_does_not_modify_input(self):
        """Should return a copy, not modify the input array."""
        import importlib

        mock_src = MagicMock()
        mock_src.crs = "EPSG:32610"
        mock_src.sample.return_value = iter([[999.0]])
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value = mock_src

        mock_windows = MagicMock()
        mock_rasterio.windows = mock_windows

        with patch.dict("sys.modules", {
            "rasterio": mock_rasterio,
            "rasterio.windows": mock_windows,
        }):
            import swe.topo.raster as mod
            importlib.reload(mod)

            coords = np.array([[10.0, 20.0, 0.0]])
            original_z = coords[0, 2]
            result = mod.terrain_correction(coords, "dem.tif")

        assert coords[0, 2] == original_z
        assert result[0, 2] == 999.0
