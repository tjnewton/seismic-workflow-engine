"""Tests for swe.scattering.born."""

from __future__ import annotations

import numpy as np
import pytest

from swe.scattering import (
    born_point_scattering,
    born_radiation_pattern,
    scattering_wavenumber,
    sphere_form_factor,
    voxel_form_factor,
)

THETA = np.linspace(0.05, np.pi - 0.05, 37)
HOST = dict(vp=3000.0, vs=1700.0, density=2000.0)


def _exact_small_fluid_sphere(theta, *, vp, density, vp_in, density_in):
    """Rayleigh limit for a small fluid sphere in fluid: (kappa_e - kappa)/kappa + 3(rho_e - rho)/(2 rho_e + rho) cos."""
    k_host, k_in = density * vp**2, density_in * vp_in**2  # bulk moduli
    monopole = k_host / k_in - 1.0  # (kappa_e - kappa) / kappa with kappa = 1 / K
    dipole = 3.0 * (density_in - density) / (2.0 * density_in + density)
    return monopole + dipole * np.cos(theta)


class TestFluidLimit:
    def test_matches_exact_small_sphere_at_small_contrast(self):
        """Fluid-limit Born equals the exact Rayleigh result to first order in contrast."""
        vp, rho, dvp, drho = 1500.0, 1000.0, 15.0, 10.0  # 1% contrasts
        born = born_radiation_pattern(THETA, vp=vp, vs=0.0, density=rho, dvp=dvp, drho=drho)
        exact = _exact_small_fluid_sphere(
            THETA, vp=vp, density=rho, vp_in=vp + dvp, density_in=rho + drho
        )
        rel = np.linalg.norm(born - exact) / np.linalg.norm(exact)
        assert rel < 0.03

    def test_strong_density_contrast_departs_and_is_flagged(self):
        """At a density ratio of 3 the Born dipole (2) far exceeds the exact one (6/7)."""
        vp, rho = 1500.0, 1000.0
        kw = dict(vp=vp, vs=0.0, density=rho, drho=2.0 * rho)
        # dipole coefficient = (pattern(0) - pattern(pi)) / 2, for Born and exact alike
        born_dip = (born_radiation_pattern(0.0, **kw) - born_radiation_pattern(np.pi, **kw)) / 2
        ex = [
            _exact_small_fluid_sphere(t, vp=vp, density=rho, vp_in=vp, density_in=3.0 * rho)
            for t in (0.0, np.pi)
        ]
        assert born_dip == pytest.approx(2.0) and (ex[0] - ex[1]) / 2 == pytest.approx(6.0 / 7.0)
        born_back = born_radiation_pattern(np.pi, **kw)
        assert abs(born_back - ex[1]) > 0.5 * abs(ex[1])
        res = born_point_scattering(
            frequency=1.0,
            volume=1e3,
            distance=1e5,
            theta=0.0,
            vp=vp,
            vs=0.0,
            density=rho,
            drho=2.0 * rho,
        )
        assert not res.weak_contrast


class TestElasticPatterns:
    def test_density_only_pp_vanishes_forward(self):
        """Elastic P->P scattering by a density perturbation is zero in the forward direction."""
        assert born_radiation_pattern(0.0, **HOST, drho=100.0) == pytest.approx(0.0, abs=1e-12)

    def test_ps_vanishes_forward_and_backward(self):
        for th in (0.0, np.pi):
            assert born_radiation_pattern(
                th, **HOST, dvp=30.0, dvs=17.0, drho=20.0, mode="PS"
            ) == pytest.approx(0.0, abs=1e-12)

    def test_velocity_form_matches_lame_form(self):
        """Pattern in (drho, dvp, dvs) equals the textbook velocity form of the P->P pattern."""
        drho, dvp, dvs = 40.0, 60.0, 34.0
        g2 = (HOST["vs"] / HOST["vp"]) ** 2
        expect = (
            (drho / HOST["density"]) * (np.cos(THETA) - 1.0 + 2.0 * g2 * np.sin(THETA) ** 2)
            - 2.0 * dvp / HOST["vp"]
            + 4.0 * g2 * np.sin(THETA) ** 2 * dvs / HOST["vs"]
        )
        got = born_radiation_pattern(THETA, **HOST, dvp=dvp, dvs=dvs, drho=drho)
        np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-15)

    def test_amplitude_scaling(self):
        kw = dict(volume=1e6, theta=np.pi, **HOST, dvp=-300.0)
        a = born_point_scattering(frequency=1.0, distance=1e4, **kw).amplitude
        assert born_point_scattering(frequency=2.0, distance=1e4, **kw).amplitude == pytest.approx(
            4 * a
        )
        assert born_point_scattering(frequency=1.0, distance=2e4, **kw).amplitude == pytest.approx(
            a / 2
        )
        k = 2 * np.pi / HOST["vp"]
        assert a == pytest.approx(
            k**2 * 1e6 / (4 * np.pi * 1e4) * born_radiation_pattern(np.pi, **HOST, dvp=-300.0)
        )

    def test_flags(self):
        res = born_point_scattering(
            frequency=1.0, volume=1e6, distance=5e4, theta=1.0, **HOST, dvp=30.0
        )
        assert res.rayleigh and res.weak_contrast and res.far_field
        near = born_point_scattering(
            frequency=1.0, volume=1e6, distance=100.0, theta=1.0, **HOST, dvp=30.0
        )
        assert not near.far_field
        big = born_point_scattering(
            frequency=20.0, volume=1e9, distance=5e4, theta=1.0, **HOST, dvp=30.0
        )
        assert not big.rayleigh

    def test_bad_inputs(self):
        with pytest.raises(ValueError, match="mode"):
            born_radiation_pattern(0.0, **HOST, mode="SS")
        with pytest.raises(ValueError, match="vs > 0"):
            born_radiation_pattern(0.0, vp=1500.0, vs=0.0, density=1000.0, mode="PS")
        with pytest.raises(ValueError, match="positive"):
            born_point_scattering(frequency=0.0, volume=1.0, distance=1.0, theta=0.0, **HOST)


class TestFormFactors:
    def test_sphere_limits(self):
        assert sphere_form_factor(0.0, 10.0) == pytest.approx(1.0)
        assert sphere_form_factor(1e-6, 10.0) == pytest.approx(1.0)
        x = 4.493409457909064  # first zero of tan(x) = x: sin x - x cos x = 0
        assert sphere_form_factor(x / 10.0, 10.0) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("qa", [0.0, 0.5, 1.5, 3.0])
    def test_voxel_sphere_matches_analytic(self, qa):
        a = 10.0
        f, vol = voxel_form_factor(
            lambda x, y, z: x**2 + y**2 + z**2 <= a**2,
            bounds=[(-1.1 * a, 1.1 * a)] * 3,
            q_vector=(qa / a, 0.0, 0.0),
            n=64,
        )
        assert vol == pytest.approx(4 / 3 * np.pi * a**3, rel=0.02)
        assert abs(f) == pytest.approx(abs(sphere_form_factor(qa / a, a)), abs=0.02)

    def test_polyhedron_close_to_volume_equivalent_sphere_in_rayleigh_regime(self):
        """A faceted polyhedral inclusion (octahedron) scatters like its volume-equivalent sphere at ka < 0.3."""
        s = 10.0
        f, vol = voxel_form_factor(
            lambda x, y, z: np.abs(x) + np.abs(y) + np.abs(z) <= s,
            bounds=[(-1.1 * s, 1.1 * s)] * 3,
            q_vector=(0.0, 0.0, 0.0),
            n=64,
        )
        a_eq = (3 * vol / (4 * np.pi)) ** (1 / 3)
        for ka in (0.1, 0.2, 0.3):
            q = 2 * ka / a_eq  # backscatter: q = 2k
            fp, _ = voxel_form_factor(
                lambda x, y, z: np.abs(x) + np.abs(y) + np.abs(z) <= s,
                bounds=[(-1.1 * s, 1.1 * s)] * 3,
                q_vector=(0.0, 0.0, q),
                n=64,
            )
            assert abs(abs(fp) - sphere_form_factor(q, a_eq)) < 0.10
        assert abs(f) == pytest.approx(1.0)

    def test_empty_body_raises(self):
        with pytest.raises(ValueError, match="inside"):
            voxel_form_factor(
                lambda x, y, z: x > 1e9, bounds=[(-1, 1)] * 3, q_vector=(0, 0, 0), n=8
            )

    def test_scattering_wavenumber(self):
        assert scattering_wavenumber(0.0, k_in=1.0, k_out=1.0) == pytest.approx(0.0)
        assert scattering_wavenumber(np.pi, k_in=1.0, k_out=1.0) == pytest.approx(2.0)
        assert scattering_wavenumber(np.pi / 2, k_in=3.0, k_out=4.0) == pytest.approx(5.0)


class TestAmplitudeAndPhaseConventions:
    def test_shear_perturbation_in_fluid_host_raises(self):
        with pytest.raises(ValueError, match="fluid host"):
            born_point_scattering(
                frequency=1.0,
                volume=1e3,
                distance=1e5,
                theta=1.0,
                vp=1500.0,
                vs=0.0,
                density=1000.0,
                dvs=2000.0,
            )

    def test_form_factor_sign_and_phase_are_kept(self):
        kw = dict(frequency=1.0, volume=1e6, distance=1e4, theta=np.pi, **HOST, dvp=-300.0)
        a = born_point_scattering(**kw).amplitude
        assert born_point_scattering(**kw, form_factor=-0.5).amplitude == pytest.approx(-0.5 * a)
        c = born_point_scattering(**kw, form_factor=0.5j).amplitude
        assert isinstance(c, complex) and c == pytest.approx(0.5j * a)

    def test_phase_convention_matches_scattering_vector(self):
        """F(q) = (1/V) int exp(-i q.x): a sphere shifted by x0 gets phase exp(-i q.x0)."""
        a, x0, q = 5.0, 3.0, 0.4
        f, _ = voxel_form_factor(
            lambda x, y, z: (x - x0) ** 2 + y**2 + z**2 <= a**2,
            bounds=[(x0 - 1.1 * a, x0 + 1.1 * a), (-1.1 * a, 1.1 * a), (-1.1 * a, 1.1 * a)],
            q_vector=(q, 0.0, 0.0),
            n=64,
        )
        assert f == pytest.approx(sphere_form_factor(q, a) * np.exp(-1j * q * x0), abs=0.02)

    def test_many_q_at_once(self):
        a = 10.0
        qs = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
        f, _ = voxel_form_factor(
            lambda x, y, z: x**2 + y**2 + z**2 <= a**2,
            bounds=[(-1.1 * a, 1.1 * a)] * 3,
            q_vector=qs,
            n=48,
        )
        assert f.shape == (3,)
        expect = [abs(sphere_form_factor(np.linalg.norm(v), a)) for v in qs]
        np.testing.assert_allclose(np.abs(f), expect, atol=0.02)
        with pytest.raises(ValueError, match="q_vector"):
            voxel_form_factor(
                lambda x, y, z: x**2 + y**2 + z**2 <= 0.25,
                bounds=[(-1, 1)] * 3,
                q_vector=(1.0, 2.0),
                n=8,
            )


class TestComplexTypesAndChunking:
    def test_complex_form_factor_types(self):
        kw = dict(frequency=1.0, volume=1e6, distance=1e4, theta=np.pi, **HOST, dvp=-300.0)
        a = born_point_scattering(**kw).amplitude
        c64 = born_point_scattering(**kw, form_factor=np.complex64(0.3 - 0.4j)).amplitude
        assert isinstance(c64, complex) and c64 == pytest.approx((0.3 - 0.4j) * a, rel=1e-6)
        zero_imag = born_point_scattering(**kw, form_factor=complex(0.5, 0.0)).amplitude
        assert isinstance(zero_imag, complex)

    def test_chunked_q_matches_per_q(self):
        a = 8.0
        qs = np.column_stack([np.linspace(0.0, 0.5, 130), np.zeros(130), np.zeros(130)])
        f, _ = voxel_form_factor(
            lambda x, y, z: x**2 + y**2 + z**2 <= a**2,
            bounds=[(-1.1 * a, 1.1 * a)] * 3,
            q_vector=qs,
            n=24,
        )
        single = [
            voxel_form_factor(
                lambda x, y, z: x**2 + y**2 + z**2 <= a**2,
                bounds=[(-1.1 * a, 1.1 * a)] * 3,
                q_vector=q,
                n=24,
            )[0]
            for q in qs[::43]
        ]
        np.testing.assert_allclose(f[::43], single, rtol=1e-12, atol=1e-12)


class TestVoxelValidation:
    def test_body_clipped_by_bounds_raises(self):
        with pytest.raises(ValueError, match="clipped"):
            voxel_form_factor(
                lambda x, y, z: x**2 + y**2 + z**2 <= 4.0,
                bounds=[(-1, 1)] * 3,
                q_vector=(0.0, 0.0, 0.0),
                n=16,
            )

    def test_empty_or_bad_q_raises(self):
        inside = lambda x, y, z: x**2 + y**2 + z**2 <= 0.25  # noqa: E731
        for bad in (np.zeros((0, 3)), np.zeros((2, 3, 1))):
            with pytest.raises(ValueError, match="q_vector"):
                voxel_form_factor(inside, bounds=[(-1, 1)] * 3, q_vector=bad, n=8)


class TestVoxelArgumentChecks:
    def test_small_n_and_reversed_bounds_raise(self):
        inside = lambda x, y, z: x**2 + y**2 + z**2 <= 0.25  # noqa: E731
        with pytest.raises(ValueError, match="at least 3"):
            voxel_form_factor(inside, bounds=[(-1, 1)] * 3, q_vector=(0, 0, 0), n=2)
        with pytest.raises(ValueError, match="min < max"):
            voxel_form_factor(inside, bounds=[(1, -1), (-1, 1), (-1, 1)], q_vector=(0, 0, 0), n=8)

    def test_bad_q_is_rejected_before_voxelizing(self):
        def inside(x, y, z):
            raise AssertionError("inside() must not run for a bad q_vector")

        with pytest.raises(ValueError, match="q_vector"):
            voxel_form_factor(inside, bounds=[(-1, 1)] * 3, q_vector=np.zeros((3, 4)), n=8)


class TestHostAndFraunhofer:
    def test_impossible_host_raises(self):
        with pytest.raises(ValueError, match="impossible host"):
            born_point_scattering(
                frequency=1.0,
                volume=1e3,
                distance=1e5,
                theta=1.0,
                vp=2000.0,
                vs=3000.0,
                density=2000.0,
            )

    def test_far_field_needs_fraunhofer_distance(self):
        vp = 3000.0
        k = 2 * np.pi * 5.0 / vp
        a = 5.0 / k  # ka = 5
        vol = 4 / 3 * np.pi * a**3
        res = born_point_scattering(
            frequency=5.0, volume=vol, distance=20.0 / k, theta=1.0, **HOST, dvp=30.0
        )
        assert res.kr > 10 and not res.far_field  # kr = 20 but r < 2 D^2 / wavelength
        far = born_point_scattering(
            frequency=5.0, volume=vol, distance=100.0 * a, theta=1.0, **HOST, dvp=30.0
        )
        assert far.far_field

    def test_float_n_raises(self):
        with pytest.raises(ValueError, match="integer"):
            voxel_form_factor(
                lambda x, y, z: x**2 + y**2 + z**2 <= 0.25,
                bounds=[(-1, 1)] * 3,
                q_vector=(0, 0, 0),
                n=48.0,
            )


class TestNonFiniteAndShapes:
    @pytest.mark.parametrize(
        "bad", [dict(frequency=np.nan), dict(volume=np.inf), dict(distance=np.nan)]
    )
    def test_non_finite_amplitude_inputs_raise(self, bad):
        kw = dict(frequency=1.0, volume=1e3, distance=1e5, theta=1.0, **HOST) | bad
        with pytest.raises(ValueError, match="finite"):
            born_point_scattering(**kw)

    def test_non_finite_host_raises(self):
        with pytest.raises(ValueError, match="finite"):
            born_radiation_pattern(1.0, vp=np.nan, vs=1000.0, density=2000.0)

    def test_array_theta_rejected_clearly(self):
        with pytest.raises(ValueError, match="theta must be a scalar"):
            born_point_scattering(
                frequency=1.0, volume=1e3, distance=1e5, theta=np.array([0.1, 0.2]), **HOST
            )

    def test_malformed_bounds_raise(self):
        with pytest.raises(ValueError, match="bounds"):
            voxel_form_factor(
                lambda x, y, z: x**2 <= 0.25,
                bounds=[(-1, 1), (-1, 1), 2.0],
                q_vector=(0, 0, 0),
                n=8,
            )
