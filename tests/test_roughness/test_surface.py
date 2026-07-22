"""Tests for swe.roughness.surface."""

from __future__ import annotations

import numpy as np
import pytest

from swe.roughness.surface import (
    RoughnessPSDResult,
    compute_roughness_psd_1d,
    compute_roughness_psd_2d,
    dominant_roughness_wavelength,
)


class TestComputeRoughnessPSD1D:
    """Tests for compute_roughness_psd_1d."""

    def test_dominant_wavelength_recovery(self, synthetic_surface_profile):
        """Dominant wavelength from known profile should be within 10%."""
        x, z, lambda_true = synthetic_surface_profile
        result = compute_roughness_psd_1d(x, z)

        assert isinstance(result, RoughnessPSDResult)
        assert abs(result.dominant_wavelength - lambda_true) / lambda_true < 0.10

    def test_spectral_slope_self_affine(self):
        """Spectral slope of a self-affine profile should be near -1-2H."""
        rng = np.random.default_rng(123)
        n = 4096
        dx = 0.01
        x = np.arange(n) * dx

        # Generate fractional Brownian noise with H=0.8
        # PSD ~ k^(-1-2H) = k^(-2.6)
        freqs = np.fft.rfftfreq(n, d=dx)
        psd_target = np.zeros_like(freqs)
        psd_target[1:] = freqs[1:] ** (-2.6)
        amp = np.sqrt(psd_target)
        phase = rng.uniform(-np.pi, np.pi, len(freqs))
        spectrum = amp * np.exp(1j * phase)
        spectrum[0] = 0
        z = np.fft.irfft(spectrum, n=n)

        result = compute_roughness_psd_1d(x, z, detrend=True)
        # Slope should be roughly -2.6 (within 0.5)
        assert abs(result.spectral_slope - (-2.6)) < 0.5

    def test_no_detrend(self, synthetic_surface_profile):
        """Works without detrending."""
        x, z, _ = synthetic_surface_profile
        result = compute_roughness_psd_1d(x, z, detrend=False)
        assert len(result.psd) > 0

    def test_profile_direction_stored(self, synthetic_surface_profile):
        """Profile direction label is preserved."""
        x, z, _ = synthetic_surface_profile
        result = compute_roughness_psd_1d(x, z, profile_direction="along-strike")
        assert result.profile_direction == "along-strike"

    def test_too_few_points_raises(self):
        """Fewer than 4 points should raise ValueError."""
        with pytest.raises(ValueError, match="at least 4"):
            compute_roughness_psd_1d(np.array([0, 1, 2]), np.array([0, 1, 0]))

    def test_length_mismatch_raises(self):
        """Mismatched x/z lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_roughness_psd_1d(np.arange(10), np.arange(5))

    def test_psd_2d_is_none(self, synthetic_surface_profile):
        """1-D computation should leave psd_2d as None."""
        x, z, _ = synthetic_surface_profile
        result = compute_roughness_psd_1d(x, z)
        assert result.psd_2d is None


class TestComputeRoughnessPSD2D:
    """Tests for compute_roughness_psd_2d."""

    def test_produces_valid_result(self, synthetic_elevation_grid):
        """2-D PSD should produce a valid result with psd_2d."""
        result = compute_roughness_psd_2d(synthetic_elevation_grid)

        assert isinstance(result, RoughnessPSDResult)
        assert result.psd_2d is not None
        assert len(result.psd) > 0
        assert not np.isnan(result.dominant_wavelength)

    def test_dominant_wavelength_reasonable(self, synthetic_elevation_grid):
        """Dominant wavelength should be within range of known values."""
        result = compute_roughness_psd_2d(synthetic_elevation_grid)
        # Known dominant wavelengths are 2.0 and 3.0 m
        assert 0.5 < result.dominant_wavelength < 10.0

    def test_spectral_slope_finite(self, synthetic_elevation_grid):
        """Spectral slope should be a finite number."""
        result = compute_roughness_psd_2d(synthetic_elevation_grid)
        assert np.isfinite(result.spectral_slope)


class TestDominantRoughnessWavelength:
    """Tests for dominant_roughness_wavelength."""

    def test_peak_method(self, synthetic_surface_profile):
        """Peak method should recover known dominant wavelength."""
        x, z, lambda_true = synthetic_surface_profile
        psd_result = compute_roughness_psd_1d(x, z)
        wl = dominant_roughness_wavelength(psd_result, method="peak")
        assert abs(wl - lambda_true) / lambda_true < 0.10

    def test_break_method(self, synthetic_surface_profile):
        """Break method should produce a finite wavelength."""
        x, z, _ = synthetic_surface_profile
        psd_result = compute_roughness_psd_1d(x, z)
        wl = dominant_roughness_wavelength(psd_result, method="break")
        assert np.isfinite(wl)
        assert wl > 0

    def test_wavelength_range_filter(self, synthetic_surface_profile):
        """Wavelength range should filter correctly."""
        x, z, _ = synthetic_surface_profile
        psd_result = compute_roughness_psd_1d(x, z)
        wl = dominant_roughness_wavelength(
            psd_result,
            min_wavelength=0.01,
            max_wavelength=0.3,
            method="peak",
        )
        assert 0.01 <= wl <= 0.3

    def test_invalid_method_raises(self, synthetic_surface_profile):
        """Invalid method should raise ValueError."""
        x, z, _ = synthetic_surface_profile
        psd_result = compute_roughness_psd_1d(x, z)
        with pytest.raises(ValueError, match="method must be"):
            dominant_roughness_wavelength(psd_result, method="invalid")

    def test_empty_range_returns_nan(self, synthetic_surface_profile):
        """Empty wavelength range should return NaN."""
        x, z, _ = synthetic_surface_profile
        psd_result = compute_roughness_psd_1d(x, z)
        wl = dominant_roughness_wavelength(
            psd_result,
            min_wavelength=1000.0,
            max_wavelength=2000.0,
        )
        assert np.isnan(wl)
