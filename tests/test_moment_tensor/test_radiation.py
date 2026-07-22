"""Tests for swe.moment_tensor.radiation - general MT radiation patterns."""

from __future__ import annotations

import numpy as np

from swe.moment_tensor.radiation import (
    _direction_cosines,
    radiation_pattern_p,
    radiation_pattern_sh,
    radiation_pattern_sv,
)
from swe.moment_tensor.tensor import strike_dip_rake_to_tensor


class TestDirectionCosines:
    """Tests for _direction_cosines convention.

    The convention used is consistent with Aki & Richards eq. 4.88
    and ``strike_dip_rake_to_tensor``: gamma_z = cos(takeoff).
    """

    def test_takeoff_zero(self):
        """takeoff=0 (straight up) should give gamma = (0, 0, +1)."""
        gx, gy, gz = _direction_cosines(
            np.array([0.0]),
            np.array([0.0]),
        )
        np.testing.assert_allclose(gx, 0.0, atol=1e-15)
        np.testing.assert_allclose(gy, 0.0, atol=1e-15)
        np.testing.assert_allclose(gz, 1.0, atol=1e-15)

    def test_takeoff_pi(self):
        """takeoff=pi (straight down) should give gamma = (0, 0, -1)."""
        gx, gy, gz = _direction_cosines(
            np.array([0.0]),
            np.array([np.pi]),
        )
        np.testing.assert_allclose(gx, 0.0, atol=1e-15)
        np.testing.assert_allclose(gy, 0.0, atol=1e-15)
        np.testing.assert_allclose(gz, -1.0, atol=1e-15)

    def test_horizontal_ray_north(self):
        """takeoff=pi/2, azimuth=0 should give (1, 0, 0) - due North."""
        gx, gy, gz = _direction_cosines(
            np.array([0.0]),
            np.array([np.pi / 2]),
        )
        np.testing.assert_allclose(gx, 1.0, atol=1e-15)
        np.testing.assert_allclose(gy, 0.0, atol=1e-15)
        np.testing.assert_allclose(gz, 0.0, atol=1e-15)

    def test_consistent_with_focal_solve(self):
        """radiation_pattern_p should agree with focal.solve._radiation_pattern
        for any DC mechanism, confirming the direction cosine convention is correct."""
        from swe.focal.solve import _radiation_pattern

        strike, dip, rake = 30.0, 45.0, -90.0
        mt = strike_dip_rake_to_tensor(strike, dip, rake, m0=1.0)

        azimuths = np.arange(0, 360, 30, dtype=np.float64)
        takeoffs = np.full_like(azimuths, 60.0)

        rad_new = radiation_pattern_p(mt, azimuths, takeoffs)
        rad_old = _radiation_pattern(
            np.radians(strike),
            np.radians(dip),
            np.radians(rake),
            np.radians(azimuths),
            np.radians(takeoffs),
        )
        np.testing.assert_array_equal(np.sign(rad_new), np.sign(rad_old))


class TestPRadiation:
    """Tests for P-wave radiation pattern."""

    def test_pure_dc_symmetry(self, pure_dc_mt):
        """P radiation should change sign across nodal planes."""
        azimuths = np.arange(0, 360, 10, dtype=np.float64)
        takeoffs = np.full_like(azimuths, 45.0)
        rad = radiation_pattern_p(pure_dc_mt, azimuths, takeoffs)

        # Should have both positive and negative values
        assert np.any(rad > 0)
        assert np.any(rad < 0)

    def test_isotropic_all_positive(self, isotropic_mt):
        """Isotropic (explosion) should give positive radiation everywhere."""
        azimuths = np.arange(0, 360, 30, dtype=np.float64)
        takeoffs = np.array([30, 60, 90, 120, 150] * 12, dtype=np.float64)[:12]
        rad = radiation_pattern_p(isotropic_mt, azimuths, takeoffs)
        assert np.all(rad > 0)

    def test_consistent_with_focal_solve(self):
        """DC radiation should match the existing focal.solve._radiation_pattern.

        Since both compute Aki & Richards P-wave pattern, the predicted
        polarities (signs) should agree for any DC mechanism.
        """
        from swe.focal.solve import _radiation_pattern

        strike, dip, rake = 45.0, 60.0, 90.0
        mt = strike_dip_rake_to_tensor(strike, dip, rake, m0=1.0)

        azimuths = np.arange(0, 360, 15, dtype=np.float64)
        takeoffs = np.full_like(azimuths, 50.0)

        # New general radiation
        rad_new = radiation_pattern_p(mt, azimuths, takeoffs)

        # Old DC-only radiation
        rad_old = _radiation_pattern(
            np.radians(strike),
            np.radians(dip),
            np.radians(rake),
            np.radians(azimuths),
            np.radians(takeoffs),
        )

        # Signs (polarities) must match
        np.testing.assert_array_equal(np.sign(rad_new), np.sign(rad_old))

    def test_scalar_output(self):
        """Single azimuth/takeoff pair should return a scalar-like value."""
        mt = strike_dip_rake_to_tensor(0.0, 45.0, 90.0)
        rad = radiation_pattern_p(mt, np.array([90.0]), np.array([45.0]))
        assert rad.shape == (1,)

    def test_zero_tensor_zero_radiation(self):
        """Zero MT should produce zero radiation."""
        mt = strike_dip_rake_to_tensor(0.0, 0.0, 0.0, m0=0.0)
        azimuths = np.array([0, 90, 180, 270], dtype=np.float64)
        takeoffs = np.full(4, 45.0)
        rad = radiation_pattern_p(mt, azimuths, takeoffs)
        np.testing.assert_allclose(rad, 0.0, atol=1e-15)


class TestSVRadiation:
    """Tests for SV-wave radiation pattern."""

    def test_returns_array(self, pure_dc_mt):
        az = np.array([0, 90, 180, 270], dtype=np.float64)
        to = np.full(4, 45.0)
        rad = radiation_pattern_sv(pure_dc_mt, az, to)
        assert rad.shape == (4,)


class TestSHRadiation:
    """Tests for SH-wave radiation pattern."""

    def test_returns_array(self, pure_dc_mt):
        az = np.array([0, 90, 180, 270], dtype=np.float64)
        to = np.full(4, 45.0)
        rad = radiation_pattern_sh(pure_dc_mt, az, to)
        assert rad.shape == (4,)

    def test_isotropic_sh_zero(self, isotropic_mt):
        """Isotropic source should have zero SH radiation."""
        az = np.arange(0, 360, 30, dtype=np.float64)
        to = np.full_like(az, 45.0)
        rad = radiation_pattern_sh(isotropic_mt, az, to)
        np.testing.assert_allclose(rad, 0.0, atol=1e-12)
