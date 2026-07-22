"""Tests for swe.roughness.spatial."""

from __future__ import annotations

import numpy as np
import pytest

from swe.roughness.spatial import (
    SpatialCorrelationResult,
    compute_spatial_frequency_map,
    correlate_frequency_roughness,
)


class TestComputeSpatialFrequencyMap:
    """Tests for compute_spatial_frequency_map."""

    def test_basic_grid(self):
        """Grid should have correct shape and contain valid frequencies."""
        rng = np.random.default_rng(42)
        locs = rng.uniform(0, 100, (50, 2))
        freqs = rng.uniform(30, 40, 50)

        x_grid, y_grid, freq_grid = compute_spatial_frequency_map(locs, freqs, grid_spacing=10.0)

        assert freq_grid.shape == (len(y_grid), len(x_grid))
        # Some cells should have data
        assert np.any(~np.isnan(freq_grid))

    def test_smoothing(self):
        """Smoothing should produce a smoother grid."""
        rng = np.random.default_rng(42)
        locs = rng.uniform(0, 100, (200, 2))
        freqs = rng.uniform(30, 40, 200)

        _, _, grid_raw = compute_spatial_frequency_map(locs, freqs, grid_spacing=10.0)
        _, _, grid_smooth = compute_spatial_frequency_map(
            locs, freqs, grid_spacing=10.0, smoothing_radius=20.0
        )

        # Smoothed should have more non-NaN values (interpolation fills gaps)
        assert np.sum(~np.isnan(grid_smooth)) >= np.sum(~np.isnan(grid_raw))

    def test_invalid_locations_shape(self):
        """Non-2D locations should raise ValueError."""
        with pytest.raises(ValueError, match="shape \\(N, 2\\)"):
            compute_spatial_frequency_map(np.array([1, 2, 3]), np.array([30.0]))

    def test_length_mismatch_raises(self):
        """Mismatched locations/frequencies should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_spatial_frequency_map(
                np.array([[0, 0], [1, 1]]),
                np.array([30.0]),
            )


class TestCorrelateFrequencyRoughness:
    """Tests for correlate_frequency_roughness."""

    def test_correlated_data(self):
        """High correlation for perfectly correlated synthetic data."""
        from swe.topo.raster import ElevationGrid

        # Build a grid where roughness varies spatially:
        # left half is smooth, right half is rough.
        dx = 0.5
        nx, ny = 60, 50
        x = np.arange(nx) * dx
        y = np.arange(ny) * dx
        rng = np.random.default_rng(42)

        xx, _yy = np.meshgrid(x, y)
        # Roughness amplitude scales with x position
        amplitude = xx / xx.max()  # 0 on left, 1 on right
        elevation = amplitude * rng.standard_normal((ny, nx))
        grid = ElevationGrid(x=x, y=y, elevation=elevation, crs=None)

        # Place events across the grid - avoid edges
        n_events = 40
        neighborhood_radius = 2.0
        margin = neighborhood_radius + dx
        ex = rng.uniform(x[0] + margin, x[-1] - margin, n_events)
        ey = rng.uniform(y[0] + margin, y[-1] - margin, n_events)
        locs = np.column_stack([ex, ey])

        # Compute the same roughness the function will compute
        from swe.roughness.spatial import correlate_frequency_roughness

        # Pre-compute local roughness to build correlated frequencies
        local_r = np.zeros(n_events)
        for i, (evx, evy) in enumerate(locs):
            ix = np.searchsorted(x, evx)
            iy = np.searchsorted(y, evy)
            r_cells = int(np.ceil(neighborhood_radius / dx))
            sub = elevation[
                max(0, iy - r_cells) : min(ny, iy + r_cells + 1),
                max(0, ix - r_cells) : min(nx, ix + r_cells + 1),
            ]
            local_r[i] = np.std(sub)

        # Make frequency strictly proportional to roughness
        freqs = 20.0 + 100.0 * local_r

        result = correlate_frequency_roughness(
            locs,
            freqs,
            grid,
            roughness_metric="rms_height",
            neighborhood_radius=neighborhood_radius,
        )

        assert isinstance(result, SpatialCorrelationResult)
        assert result.correlation_coefficient > 0.8
        assert result.p_value < 0.05

    def test_invalid_metric_raises(self, synthetic_elevation_grid):
        """Invalid roughness metric should raise ValueError."""
        locs = np.array([[1.0, 1.0], [2.0, 2.0]])
        freqs = np.array([30.0, 35.0])
        with pytest.raises(ValueError, match="roughness_metric must be"):
            correlate_frequency_roughness(
                locs,
                freqs,
                synthetic_elevation_grid,
                roughness_metric="invalid",
            )

    def test_slope_variance_metric(self, synthetic_elevation_grid):
        """slope_variance metric should produce a finite result."""
        grid = synthetic_elevation_grid
        rng = np.random.default_rng(42)
        n_events = 15
        ex = rng.uniform(grid.x[3], grid.x[-4], n_events)
        ey = rng.uniform(grid.y[3], grid.y[-4], n_events)
        locs = np.column_stack([ex, ey])
        freqs = rng.uniform(30, 40, n_events)

        result = correlate_frequency_roughness(
            locs,
            freqs,
            grid,
            roughness_metric="slope_variance",
            neighborhood_radius=2.0,
        )

        assert np.isfinite(result.correlation_coefficient)

    def test_too_few_events_returns_nan(self, synthetic_elevation_grid):
        """Fewer than 3 events yields NaN correlation."""
        locs = np.array([[0.0, 0.0]])
        freqs = np.array([34.5])
        result = correlate_frequency_roughness(
            locs,
            freqs,
            synthetic_elevation_grid,
            neighborhood_radius=0.01,  # tiny radius -> likely <4 cells
        )
        assert np.isnan(result.correlation_coefficient)
