"""Tests for swe.moment_tensor.tensor - core types and algebra."""

from __future__ import annotations

import numpy as np
import pytest

from swe.moment_tensor.tensor import (
    MomentTensor,
    MomentTensorDecomposition,
    decompose_moment_tensor,
    from_matrix,
    moment_magnitude,
    scalar_moment,
    strike_dip_rake_to_tensor,
    tensor_to_strike_dip_rake,
    to_matrix,
)


# ---------------------------------------------------------------------------
# Scalar moment and moment magnitude
# ---------------------------------------------------------------------------


class TestScalarMoment:
    """Tests for scalar_moment computation."""

    def test_identity_like_tensor(self):
        m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        m0 = scalar_moment(m)
        assert m0 == pytest.approx(np.sqrt(3.0 / 2.0))

    def test_zero_tensor(self):
        m = np.zeros((3, 3), dtype=np.float64)
        assert scalar_moment(m) == 0.0

    def test_accepts_moment_tensor_namedtuple(self):
        mt = MomentTensor(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        m0 = scalar_moment(mt)
        assert m0 == pytest.approx(np.sqrt(3.0 / 2.0))


class TestMomentMagnitude:
    """Tests for Mw computation."""

    def test_known_value(self):
        # M0 = 1e16 N·m -> Mw ≈ 4.27
        mw = moment_magnitude(1e16)
        expected = (2.0 / 3.0) * np.log10(1e16) - 10.7
        assert mw == pytest.approx(expected)

    def test_zero_moment(self):
        assert moment_magnitude(0.0) == -999.0

    def test_negative_moment(self):
        assert moment_magnitude(-1.0) == -999.0


# ---------------------------------------------------------------------------
# Matrix conversion
# ---------------------------------------------------------------------------


class TestMatrixConversion:
    """Tests for to_matrix and from_matrix."""

    def test_to_matrix(self):
        mt = MomentTensor(1.0, 2.0, 3.0, 0.5, 0.3, 0.1)
        m = to_matrix(mt)
        assert m.shape == (3, 3)
        assert m[0, 0] == 1.0
        assert m[1, 1] == 2.0
        assert m[2, 2] == 3.0
        assert m[0, 1] == 0.5
        assert m[1, 0] == 0.5  # symmetric

    def test_from_matrix(self):
        m = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=np.float64)
        mt = from_matrix(m)
        assert mt.mxx == 1.0
        assert mt.myy == 4.0
        assert mt.mzz == 6.0
        assert mt.mxy == 2.0
        assert mt.scalar_moment > 0

    def test_round_trip(self):
        mt = MomentTensor(1.0, 2.0, 3.0, 0.5, 0.3, 0.1, 1.0, 0.0)
        m = to_matrix(mt)
        mt2 = from_matrix(m)
        assert mt2.mxx == pytest.approx(mt.mxx)
        assert mt2.myy == pytest.approx(mt.myy)
        assert mt2.mzz == pytest.approx(mt.mzz)


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


class TestDecomposition:
    """Tests for ISO/DC/CLVD decomposition."""

    def test_pure_dc(self, pure_dc_mt):
        """A pure DC tensor should decompose to ~100% DC."""
        decomp = decompose_moment_tensor(pure_dc_mt)
        assert isinstance(decomp, MomentTensorDecomposition)
        assert decomp.dc_pct == pytest.approx(100.0, abs=5.0)
        assert decomp.iso_pct == pytest.approx(0.0, abs=2.0)
        assert decomp.clvd_pct == pytest.approx(0.0, abs=5.0)

    def test_pure_isotropic(self, isotropic_mt):
        """A pure isotropic tensor should decompose to ~100% ISO."""
        decomp = decompose_moment_tensor(isotropic_mt)
        assert decomp.iso_pct == pytest.approx(100.0, abs=1.0)
        assert decomp.dc_pct == pytest.approx(0.0, abs=1.0)
        assert decomp.clvd_pct == pytest.approx(0.0, abs=1.0)

    def test_clvd_tensor(self, clvd_mt):
        """A CLVD-rich tensor should have significant CLVD percentage."""
        decomp = decompose_moment_tensor(clvd_mt)
        assert decomp.clvd_pct > 20.0
        assert decomp.iso_pct == pytest.approx(0.0, abs=1.0)

    def test_percentages_sum_to_100(self, pure_dc_mt):
        decomp = decompose_moment_tensor(pure_dc_mt)
        total = decomp.iso_pct + decomp.dc_pct + decomp.clvd_pct
        assert total == pytest.approx(100.0, abs=0.1)


# ---------------------------------------------------------------------------
# Strike/dip/rake ↔ tensor conversion
# ---------------------------------------------------------------------------


class TestSDRConversion:
    """Tests for strike_dip_rake_to_tensor and tensor_to_strike_dip_rake."""

    @pytest.mark.parametrize("strike,dip,rake", [
        (0.0, 45.0, 90.0),    # thrust
        (90.0, 90.0, 0.0),    # strike-slip
        (180.0, 45.0, -90.0), # normal fault
        (270.0, 30.0, 45.0),  # oblique
    ])
    def test_round_trip(self, strike, dip, rake):
        """strike/dip/rake -> MT -> strike/dip/rake should recover equivalent mechanism."""
        mt = strike_dip_rake_to_tensor(strike, dip, rake, m0=1.0)

        # Check the tensor is a valid DC
        m = to_matrix(mt)
        trace = np.trace(m)
        assert trace == pytest.approx(0.0, abs=1e-10)  # deviatoric

        # Extract back
        s2, d2, r2 = tensor_to_strike_dip_rake(mt)

        # Rebuild and compare radiation patterns (since there are two
        # equivalent nodal planes, we compare the tensors)
        mt2 = strike_dip_rake_to_tensor(s2, d2, r2, m0=1.0)
        m2 = to_matrix(mt2)

        # Tensors should match (possibly with sign convention differences)
        np.testing.assert_allclose(np.abs(m), np.abs(m2), atol=0.15)

    def test_pure_dc_trace_zero(self):
        """DC tensor should have zero trace."""
        mt = strike_dip_rake_to_tensor(45.0, 60.0, 90.0)
        m = to_matrix(mt)
        assert np.trace(m) == pytest.approx(0.0, abs=1e-10)

    def test_scalar_moment_scaling(self):
        """M0 parameter scales the tensor linearly."""
        mt1 = strike_dip_rake_to_tensor(0.0, 45.0, 90.0, m0=1.0)
        mt2 = strike_dip_rake_to_tensor(0.0, 45.0, 90.0, m0=2.0)
        m1 = to_matrix(mt1)
        m2 = to_matrix(mt2)
        np.testing.assert_allclose(m2, 2.0 * m1)
