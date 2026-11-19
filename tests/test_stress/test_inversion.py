"""Tests for swe.stress.inversion."""

import numpy as np
import pytest

from swe.stress.inversion import (
    _invert,
    _strike_dip_rake_to_fault_slip,
    _vector_to_azimuth_plunge,
    cluster_focal_mechanisms,
    split_focal_data,
    stress_inversion,
    stress_inversion_with_bootstrap,
)


class TestClusterFocalMechanisms:
    """Tests for cluster_focal_mechanisms with synthetic data."""

    def test_two_clusters(self):
        """Two well-separated groups should be assigned to two clusters."""
        rng = np.random.default_rng(42)
        group_a = rng.normal(loc=0.0, scale=0.5, size=(30, 3))
        group_b = rng.normal(loc=10.0, scale=0.5, size=(30, 3))
        data = np.vstack([group_a, group_b])

        labels, centers = cluster_focal_mechanisms(data, n_clusters=2)

        assert len(labels) == 60
        assert centers.shape == (2, 3)
        assert set(labels) == {0, 1}

        # Verify the two centers are far apart
        dist = np.linalg.norm(centers[0] - centers[1])
        assert dist > 5.0

    def test_single_cluster(self):
        """Requesting one cluster should label everything the same."""
        rng = np.random.default_rng(7)
        data = rng.normal(loc=5.0, scale=1.0, size=(20, 2))

        labels, centers = cluster_focal_mechanisms(data, n_clusters=1)

        assert len(labels) == 20
        assert np.all(labels == 0)
        assert centers.shape == (1, 2)

    def test_deterministic_with_random_state(self):
        """Same random_state should produce identical results."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(40, 3))

        labels1, centers1 = cluster_focal_mechanisms(data, n_clusters=3, random_state=42)
        labels2, centers2 = cluster_focal_mechanisms(data, n_clusters=3, random_state=42)

        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_array_almost_equal(centers1, centers2)

    def test_cluster_sizes(self):
        """Each cluster should have at least one member."""
        rng = np.random.default_rng(99)
        data = rng.normal(size=(50, 4))

        labels, _centers = cluster_focal_mechanisms(data, n_clusters=5)

        for i in range(5):
            count = np.sum(labels == i)
            assert count >= 1, f"Cluster {i} has no members"

    def test_n_init_and_max_iter(self):
        """Custom n_init and max_iter should not raise."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(30, 2))

        labels, _centers = cluster_focal_mechanisms(data, n_clusters=2, n_init=5, max_iter=100)

        assert len(labels) == 30


class TestStrikeDipRakeConversion:
    """Tests for _strike_dip_rake_to_fault_slip."""

    def test_orthogonality(self):
        """Normal and slip vectors should be orthogonal for each plane."""
        n1, d1, n2, d2 = _strike_dip_rake_to_fault_slip(30.0, 60.0, -90.0)
        assert abs(np.dot(n1, d1)) < 1e-10
        assert abs(np.dot(n2, d2)) < 1e-10

    def test_unit_vectors(self):
        """All returned vectors should have unit length."""
        n1, d1, n2, d2 = _strike_dip_rake_to_fault_slip(120.0, 45.0, 30.0)
        for v in [n1, d1, n2, d2]:
            assert abs(np.linalg.norm(v) - 1.0) < 1e-10

    def test_vertical_dip(self):
        """Vertical fault (dip=90) should still produce valid vectors."""
        n1, d1, _n2, _d2 = _strike_dip_rake_to_fault_slip(0.0, 90.0, 0.0)
        assert abs(np.dot(n1, d1)) < 1e-10

    def test_pure_normal_fault(self):
        """Pure normal fault (rake=-90): slip has a z-component (vertical motion)."""
        _n1, d1, _, _ = _strike_dip_rake_to_fault_slip(0.0, 45.0, -90.0)
        # z=Down convention: slip vector should have significant vertical component
        assert abs(d1[2]) > 0.5


class TestVectorToAzimuthPlunge:
    """Tests for _vector_to_azimuth_plunge."""

    def test_north_horizontal(self):
        """Pointing north horizontally: azimuth=0, plunge=0."""
        az, pl = _vector_to_azimuth_plunge(np.array([1.0, 0.0, 0.0]))
        assert abs(az) < 1e-6 or abs(az - 360.0) < 1e-6
        assert abs(pl) < 1e-6

    def test_east_horizontal(self):
        """Pointing east horizontally: azimuth=90, plunge=0."""
        az, pl = _vector_to_azimuth_plunge(np.array([0.0, 1.0, 0.0]))
        assert abs(az - 90.0) < 1e-6
        assert abs(pl) < 1e-6

    def test_vertical_down(self):
        """Pointing straight down: plunge=90."""
        _az, pl = _vector_to_azimuth_plunge(np.array([0.0, 0.0, 1.0]))
        assert abs(pl - 90.0) < 1e-6

    def test_plunge_always_positive(self):
        """Plunge should always be in [0, 90]."""
        _az, pl = _vector_to_azimuth_plunge(np.array([0.0, 0.0, -1.0]))
        assert 0.0 <= pl <= 90.0


class TestInvert:
    """Tests for _invert (Vavrycuk stress inversion)."""

    def test_synthetic_normal_faulting(self):
        """Synthetic normal-faulting mechanisms should recover one near-vertical axis."""
        # Normal faulting regime: the vertical direction should correspond to
        # one of the principal stresses (sigma_3 in deviatoric convention,
        # since the overburden is the most compressive absolute stress but
        # deviatoric inversion removes the mean).
        rng = np.random.default_rng(42)
        n_mechs = 20
        strikes = rng.uniform(0, 360, n_mechs)
        dips = rng.uniform(40, 70, n_mechs)  # moderate dips
        rakes = rng.uniform(-110, -70, n_mechs)  # normal faulting

        focal_mechs = np.column_stack([strikes, dips, rakes])
        results, axes = _invert(focal_mechs)

        # Check output structure
        assert "sigma_1" in results
        assert "sigma_2" in results
        assert "sigma_3" in results
        assert "azimuth" in results["sigma_1"]
        assert "plunge" in results["sigma_1"]

        # At least one principal axis should be near-vertical (plunge > 60)
        plunges = [
            results["sigma_1"]["plunge"],
            results["sigma_2"]["plunge"],
            results["sigma_3"]["plunge"],
        ]
        max_plunge = max(plunges)
        assert max_plunge > 60.0, (
            f"Expected one near-vertical axis for normal faulting, max plunge was {max_plunge}"
        )

        # Stress ratio should be between 0 and 1
        assert 0.0 <= axes["stress_ratio"] <= 1.0

    def test_synthetic_strike_slip(self):
        """Strike-slip mechanisms should recover sigma_1 near horizontal."""
        rng = np.random.default_rng(123)
        n_mechs = 20
        # Strike-slip: vertical dip, rake ~0 or ~180
        strikes = rng.uniform(30, 50, n_mechs)  # consistent strike
        dips = rng.uniform(80, 90, n_mechs)
        rakes = rng.uniform(-20, 20, n_mechs)

        focal_mechs = np.column_stack([strikes, dips, rakes])
        results, _axes = _invert(focal_mechs)

        # For strike-slip, sigma_1 and sigma_3 should be near-horizontal
        sigma1_plunge = results["sigma_1"]["plunge"]
        sigma3_plunge = results["sigma_3"]["plunge"]
        assert sigma1_plunge < 45.0 or sigma3_plunge < 45.0

    def test_single_mechanism(self):
        """Should handle a single focal mechanism without error."""
        focal_mechs = np.array([[45.0, 60.0, -90.0]])
        results, axes = _invert(focal_mechs)
        assert "sigma_1" in results
        assert axes["n_iterations"] >= 1

    def test_output_structure(self):
        """Verify the output dictionary structure."""
        rng = np.random.default_rng(7)
        focal_mechs = rng.uniform(low=[0, 10, -180], high=[360, 80, 180], size=(10, 3))
        results, axes = _invert(focal_mechs)

        for key in ["sigma_1", "sigma_2", "sigma_3"]:
            assert key in results
            assert "azimuth" in results[key]
            assert "plunge" in results[key]
            assert 0.0 <= results[key]["azimuth"] < 360.0
            assert 0.0 <= results[key]["plunge"] <= 90.0

        assert "eigenvectors" in axes
        assert "eigenvalues" in axes
        assert "stress_ratio" in axes
        assert "n_iterations" in axes

    def test_stress_inversion_runs(self):
        """stress_inversion should now run without NotImplementedError."""
        rng = np.random.default_rng(42)
        data = np.column_stack(
            [
                rng.uniform(0, 360, 15),  # strike
                rng.uniform(20, 70, 15),  # dip
                rng.uniform(-180, 180, 15),  # rake
            ]
        )
        labels = np.array([0] * 8 + [1] * 7)

        inv_result = stress_inversion(data, labels)
        assert len(inv_result.results) == 2
        assert len(inv_result.axes) == 2
        for r in inv_result.results:
            assert "sigma_1" in r

    def test_empty_raises(self):
        """Empty array should raise ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            _invert(np.empty((0, 3)))


# ---------------------------------------------------------------------------
# Additional edge-case tests for _invert
# ---------------------------------------------------------------------------


class TestInvertEdgeCases:
    """Additional edge cases for the Vavrycuk inversion."""

    def test_coplanar_mechanisms(self):
        """All-identical strike/dip/rake should still converge without error."""
        focal_mechs = np.tile([120.0, 45.0, -90.0], (10, 1))
        results, axes = _invert(focal_mechs)
        assert "sigma_1" in results
        assert axes["n_iterations"] >= 1

    def test_two_mechanisms(self):
        """Minimum viable count (2 mechanisms) should produce a result."""
        focal_mechs = np.array(
            [
                [30.0, 60.0, -90.0],
                [210.0, 45.0, 90.0],
            ]
        )
        results, _axes = _invert(focal_mechs)
        assert "sigma_1" in results

    def test_high_noise_mechanisms(self):
        """Random strike/dip/rake should not crash."""
        rng = np.random.default_rng(42)
        focal_mechs = np.column_stack(
            [
                rng.uniform(0, 360, 30),
                rng.uniform(5, 85, 30),
                rng.uniform(-180, 180, 30),
            ]
        )
        results, _axes = _invert(focal_mechs)
        assert "sigma_1" in results

    def test_convergence_reported(self):
        """n_iterations should be between 1 and 20."""
        rng = np.random.default_rng(42)
        focal_mechs = np.column_stack(
            [
                rng.uniform(0, 360, 15),
                rng.uniform(30, 70, 15),
                rng.uniform(-110, -70, 15),
            ]
        )
        _, axes = _invert(focal_mechs)
        assert 1 <= axes["n_iterations"] <= 20

    def test_stress_tensor_is_deviatoric(self):
        """The reconstructed stress tensor should have trace approx 0."""
        rng = np.random.default_rng(42)
        focal_mechs = np.column_stack(
            [
                rng.uniform(0, 360, 20),
                rng.uniform(30, 70, 20),
                rng.uniform(-110, -70, 20),
            ]
        )
        _, axes = _invert(focal_mechs)
        eigenvalues = np.array(axes["eigenvalues"])
        # Sum of eigenvalues of a deviatoric tensor should be ~0
        np.testing.assert_almost_equal(np.sum(eigenvalues), 0.0, decimal=10)


# ---------------------------------------------------------------------------
# stress_inversion public API tests
# ---------------------------------------------------------------------------


class TestStressInversionPublicAPI:
    """Tests for the public stress_inversion function."""

    def test_three_clusters(self):
        """Three clusters should produce 3 result dicts."""
        rng = np.random.default_rng(42)
        data = np.column_stack(
            [
                rng.uniform(0, 360, 30),
                rng.uniform(20, 70, 30),
                rng.uniform(-180, 180, 30),
            ]
        )
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
        inv_result = stress_inversion(data, labels)
        assert len(inv_result.results) == 3
        assert len(inv_result.axes) == 3

    def test_single_cluster(self):
        """All labels=0 should produce exactly 1 result."""
        rng = np.random.default_rng(42)
        data = np.column_stack(
            [
                rng.uniform(0, 360, 10),
                rng.uniform(20, 70, 10),
                rng.uniform(-180, 180, 10),
            ]
        )
        labels = np.zeros(10, dtype=int)
        inv_result = stress_inversion(data, labels)
        assert len(inv_result.results) == 1
        assert len(inv_result.axes) == 1

    def test_mismatched_labels(self):
        """Non-contiguous labels should still work (one result per unique label)."""
        rng = np.random.default_rng(42)
        data = np.column_stack(
            [
                rng.uniform(0, 360, 12),
                rng.uniform(20, 70, 12),
                rng.uniform(-180, 180, 12),
            ]
        )
        labels = np.array([0, 0, 0, 5, 5, 5, 10, 10, 10, 99, 99, 99])
        inv_result = stress_inversion(data, labels)
        assert len(inv_result.results) == 4


# ---------------------------------------------------------------------------
# split_focal_data tests
# ---------------------------------------------------------------------------


class TestSplitFocalData:
    """Tests for split_focal_data (file I/O)."""

    def test_splits_by_label(self, tmp_path):
        """Two clusters should produce two output files."""
        events = np.array(
            [
                [1.0, 2.0, 3.0, 30.0, 60.0, -90.0],
                [1.1, 2.1, 3.1, 35.0, 55.0, -85.0],
                [4.0, 5.0, 6.0, 120.0, 45.0, 90.0],
            ]
        )
        labels = np.array([0, 0, 1])
        split_focal_data(events, labels, output_dir=tmp_path)
        assert (tmp_path / "events_0.txt").exists()
        assert (tmp_path / "events_1.txt").exists()

    def test_adds_default_magnitude(self, tmp_path):
        """6-column data should get a default magnitude column (5.0) appended."""
        events = np.array([[1.0, 2.0, 3.0, 30.0, 60.0, -90.0]])
        labels = np.array([0])
        split_focal_data(events, labels, output_dir=tmp_path)
        content = (tmp_path / "events_0.txt").read_text().strip()
        # 7th column (magnitude) should be 5.0
        fields = content.split()
        assert fields[6] == "5.0"

    def test_seven_column_data(self, tmp_path):
        """7-column data should not get an extra magnitude column."""
        events = np.array([[1.0, 2.0, 3.0, 30.0, 60.0, -90.0, 3.5]])
        labels = np.array([0])
        split_focal_data(events, labels, output_dir=tmp_path)
        content = (tmp_path / "events_0.txt").read_text().strip()
        fields = content.split()
        # 7th column should be the original magnitude, not 5.0
        assert fields[6] == "3.5"


class TestStressInversionWithBootstrap:
    """Tests for bootstrap stress inversion."""

    def test_bootstrap_produces_axes(self):
        """bootstrap_axes should be populated."""
        rng = np.random.default_rng(42)
        data = np.column_stack(
            [
                rng.uniform(0, 360, 15),
                rng.uniform(30, 70, 15),
                rng.uniform(-110, -70, 15),
            ]
        )
        labels = np.zeros(15, dtype=int)

        result = stress_inversion_with_bootstrap(
            data,
            labels,
            n_bootstrap=10,
            random_state=42,
        )

        assert result.bootstrap_axes is not None
        assert len(result.bootstrap_axes) == 1
        assert "sigma_1" in result.bootstrap_axes[0]
        assert "azimuth_std" in result.bootstrap_axes[0]["sigma_1"]
        assert result.bootstrap_axes[0]["sigma_1"]["n_samples"] > 0

    def test_backward_compat_without_bootstrap(self):
        """stress_inversion should return bootstrap_axes=None."""
        rng = np.random.default_rng(42)
        data = np.column_stack(
            [
                rng.uniform(0, 360, 10),
                rng.uniform(20, 70, 10),
                rng.uniform(-180, 180, 10),
            ]
        )
        labels = np.zeros(10, dtype=int)

        result = stress_inversion(data, labels)
        assert result.bootstrap_axes is None
