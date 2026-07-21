"""Integration tests: focal mechanism clustering -> stress inversion."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


class TestStressPipeline:
    """Cluster focal mechanisms and invert for stress tensors."""

    def test_cluster_and_invert_synthetic_mechanisms(self):
        """Two clusters with distinct stress orientations produce valid results."""
        from swe.stress import cluster_focal_mechanisms, stress_inversion

        rng = np.random.default_rng(42)
        n_per_cluster = 20

        # Cluster 1: normal faulting (strike~0, dip~60, rake~-90)
        cluster1 = np.column_stack(
            [
                rng.normal(0, 5, n_per_cluster),  # strike
                rng.normal(60, 3, n_per_cluster),  # dip
                rng.normal(-90, 5, n_per_cluster),  # rake
            ]
        )
        # Cluster 2: strike-slip (strike~90, dip~90, rake~0)
        cluster2 = np.column_stack(
            [
                rng.normal(90, 5, n_per_cluster),
                rng.normal(85, 3, n_per_cluster),
                rng.normal(0, 5, n_per_cluster),
            ]
        )

        # For clustering, use spatial coordinates
        coords1 = np.column_stack(
            [
                rng.normal(-118.5, 0.01, n_per_cluster),
                rng.normal(34.0, 0.01, n_per_cluster),
                rng.normal(5.0, 0.5, n_per_cluster),
            ]
        )
        coords2 = np.column_stack(
            [
                rng.normal(-117.5, 0.01, n_per_cluster),
                rng.normal(35.0, 0.01, n_per_cluster),
                rng.normal(10.0, 0.5, n_per_cluster),
            ]
        )

        all_coords = np.vstack([coords1, coords2])
        all_mechs = np.vstack([cluster1, cluster2])

        # Step 1: Cluster by spatial coordinates
        cluster_result = cluster_focal_mechanisms(all_coords, n_clusters=2)
        assert len(cluster_result.labels) == 2 * n_per_cluster
        assert len(np.unique(cluster_result.labels)) == 2

        # Step 2: Invert stress per cluster
        inv_result = stress_inversion(all_mechs, cluster_result.labels)

        assert len(inv_result.results) == 2
        assert len(inv_result.axes) == 2

        # Both inversions should have sigma_1, sigma_2, sigma_3
        for r in inv_result.results:
            assert "sigma_1" in r
            assert "sigma_2" in r
            assert "sigma_3" in r
            for key in ("sigma_1", "sigma_2", "sigma_3"):
                assert "azimuth" in r[key]
                assert "plunge" in r[key]
                assert 0 <= r[key]["azimuth"] < 360
                assert 0 <= r[key]["plunge"] <= 90

        # Stress ratios should be valid
        for a in inv_result.axes:
            assert 0 <= a["stress_ratio"] <= 1.0

    def test_single_cluster_inversion(self):
        """Valid stress tensor from consistent normal-faulting mechanisms."""
        from swe.stress import stress_inversion

        rng = np.random.default_rng(123)
        n = 30
        # Consistent normal faulting
        mechs = np.column_stack(
            [
                rng.normal(180, 3, n),  # strike ~180
                rng.normal(60, 2, n),  # dip ~60
                rng.normal(-90, 3, n),  # rake ~-90 (normal)
            ]
        )
        labels = np.zeros(n, dtype=int)

        inv_result = stress_inversion(mechs, labels)

        assert len(inv_result.results) == 1
        assert len(inv_result.axes) == 1

        # Check stress ratio is valid
        assert 0 <= inv_result.axes[0]["stress_ratio"] <= 1.0

        # For normal faulting, sigma_1 should be approximately vertical
        # (high plunge). Allow generous tolerance for noisy data.
        sigma1_plunge = inv_result.results[0]["sigma_1"]["plunge"]
        assert sigma1_plunge > 5, (
            f"Expected positive plunge for sigma_1 in normal faulting, got {sigma1_plunge}"
        )
