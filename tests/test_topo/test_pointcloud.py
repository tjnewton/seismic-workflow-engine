"""Tests for swe.topo.pointcloud."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestLoadPointCloud:
    """Tests for load_point_cloud() with mocked laspy."""

    def _make_mock_las(
        self,
        n=10,
        *,
        has_color=True,
        has_classification=True,
    ):
        """Create a mock LAS object with controllable fields."""
        rng = np.random.default_rng(42)
        mock_las = MagicMock()
        mock_las.x = rng.uniform(0, 100, n)
        mock_las.y = rng.uniform(0, 100, n)
        mock_las.z = rng.uniform(0, 50, n)

        if has_color:
            mock_las.red = rng.integers(0, 65535, n).astype(np.uint16)
            mock_las.green = rng.integers(0, 65535, n).astype(np.uint16)
            mock_las.blue = rng.integers(0, 65535, n).astype(np.uint16)
        else:
            # Remove colour attributes.
            del mock_las.red
            del mock_las.green
            del mock_las.blue

        if has_classification:
            mock_las.classification = rng.choice([1, 2, 6], n)
        else:
            del mock_las.classification

        return mock_las

    def test_basic_loading(self):
        """Should load xyz, colors, classification from LAS."""
        mock_las = self._make_mock_las(n=20)
        mock_laspy = MagicMock()
        mock_laspy.read.return_value = mock_las

        with patch.dict("sys.modules", {"laspy": mock_laspy}):
            import importlib

            import swe.topo.pointcloud as mod

            importlib.reload(mod)

            result = mod.load_point_cloud("test.las")

        assert result.points.shape == (20, 3)
        assert result.colors is not None
        assert result.colors.shape == (20, 3)
        assert result.classification is not None
        assert result.classification.shape == (20,)

    def test_bounds_filtering(self):
        """Should filter points to the given bounding box."""
        mock_las = MagicMock()
        mock_las.x = np.array([10.0, 50.0, 90.0, 50.0])
        mock_las.y = np.array([10.0, 50.0, 90.0, 50.0])
        mock_las.z = np.array([5.0, 25.0, 45.0, 25.0])
        del mock_las.red
        del mock_las.green
        del mock_las.blue
        del mock_las.classification

        mock_laspy = MagicMock()
        mock_laspy.read.return_value = mock_las

        with patch.dict("sys.modules", {"laspy": mock_laspy}):
            import importlib

            import swe.topo.pointcloud as mod

            importlib.reload(mod)

            result = mod.load_point_cloud(
                "test.las",
                bounds=((20.0, 80.0), (20.0, 80.0), (10.0, 40.0)),
            )

        # Only the two middle points should survive.
        assert result.points.shape[0] == 2

    def test_classification_filter(self):
        """Should keep only points with specified classification codes."""
        mock_las = MagicMock()
        mock_las.x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mock_las.y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mock_las.z = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mock_las.classification = np.array([2, 6, 2, 1, 6])
        del mock_las.red
        del mock_las.green
        del mock_las.blue

        mock_laspy = MagicMock()
        mock_laspy.read.return_value = mock_las

        with patch.dict("sys.modules", {"laspy": mock_laspy}):
            import importlib

            import swe.topo.pointcloud as mod

            importlib.reload(mod)

            result = mod.load_point_cloud(
                "test.las",
                classification_filter=[2],
            )

        assert result.points.shape[0] == 2
        np.testing.assert_array_equal(result.classification, [2, 2])

    def test_subsample(self):
        """Should keep every n-th point."""
        mock_las = MagicMock()
        mock_las.x = np.arange(10.0)
        mock_las.y = np.arange(10.0)
        mock_las.z = np.arange(10.0)
        del mock_las.red
        del mock_las.green
        del mock_las.blue
        del mock_las.classification

        mock_laspy = MagicMock()
        mock_laspy.read.return_value = mock_las

        with patch.dict("sys.modules", {"laspy": mock_laspy}):
            import importlib

            import swe.topo.pointcloud as mod

            importlib.reload(mod)

            result = mod.load_point_cloud("test.las", subsample=3)

        # Every 3rd point: indices 0, 3, 6, 9 -> 4 points.
        assert result.points.shape[0] == 4

    def test_import_error_without_laspy(self):
        """Should raise ImportError when laspy is not installed."""
        import importlib

        import swe.topo.pointcloud as mod

        with patch.dict("sys.modules", {"laspy": None}):
            importlib.reload(mod)
            with pytest.raises(ImportError, match="laspy"):
                mod.load_point_cloud("test.las")

    def test_missing_color_fields(self):
        """Should return None for colors when LAS lacks RGB."""
        mock_las = self._make_mock_las(n=5, has_color=False)
        mock_laspy = MagicMock()
        mock_laspy.read.return_value = mock_las

        with patch.dict("sys.modules", {"laspy": mock_laspy}):
            import importlib

            import swe.topo.pointcloud as mod

            importlib.reload(mod)

            result = mod.load_point_cloud("test.las")

        assert result.colors is None

    def test_missing_classification_field(self):
        """Should return None for classification when LAS lacks it."""
        mock_las = self._make_mock_las(n=5, has_classification=False)
        mock_laspy = MagicMock()
        mock_laspy.read.return_value = mock_las

        with patch.dict("sys.modules", {"laspy": mock_laspy}):
            import importlib

            import swe.topo.pointcloud as mod

            importlib.reload(mod)

            result = mod.load_point_cloud("test.las")

        assert result.classification is None
