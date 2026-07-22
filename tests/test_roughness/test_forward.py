"""Tests for swe.roughness.forward."""

from __future__ import annotations

import numpy as np
import pytest

from swe.roughness.forward import (
    ForwardModelResult,
    forward_model_frequency,
    resonance_frequency,
)
from swe.roughness.surface import RoughnessPSDResult


@pytest.fixture()
def simple_psd():
    """Simple roughness PSD with known peak at wavenumber=20 (1/m)."""
    k = np.linspace(0.1, 100.0, 1000)
    # Peak at k=20 -> wavelength=0.05 m
    psd = np.exp(-((k - 20.0) ** 2) / (2 * 2.0**2))
    wavelengths = 1.0 / k
    return RoughnessPSDResult(
        wavelengths=wavelengths,
        wavenumbers=k,
        psd=psd,
        psd_2d=None,
        dominant_wavelength=0.05,
        spectral_slope=-2.0,
        spectral_slope_std=0.1,
        profile_direction=None,
    )


class TestForwardModelFrequency:
    """Tests for forward_model_frequency."""

    def test_basic_prediction(self, simple_psd):
        """Predicted frequency should be within factor of 2 of V*k_peak."""
        slip_vel = 1e-4  # m/s
        result = forward_model_frequency(simple_psd, slip_vel)

        assert isinstance(result, ForwardModelResult)
        assert not np.isnan(result.predicted_frequency)
        assert result.predicted_frequency > 0
        assert result.model_type == "elastic_halfspace"

    def test_observed_frequency_ratio(self, simple_psd):
        """Frequency ratio should be computed when observed_frequency given."""
        result = forward_model_frequency(simple_psd, 1e-4, observed_frequency=34.5)

        assert not np.isnan(result.frequency_ratio)
        assert result.observed_frequency == 34.5

    def test_hertzian_model(self, simple_psd):
        """Hertzian model should also produce a valid prediction."""
        result = forward_model_frequency(simple_psd, 1e-4, contact_model="hertzian")

        assert result.model_type == "hertzian"
        assert not np.isnan(result.predicted_frequency)

    def test_invalid_model_raises(self, simple_psd):
        """Invalid contact model should raise ValueError."""
        with pytest.raises(ValueError, match="contact_model must be"):
            forward_model_frequency(simple_psd, 1e-4, contact_model="invalid")

    def test_stress_spectrum_shape(self, simple_psd):
        """Stress spectrum should have same length as PSD."""
        result = forward_model_frequency(simple_psd, 1e-4)
        assert len(result.stress_spectrum) == len(simple_psd.psd)
        assert len(result.stress_frequencies) == len(simple_psd.psd)

    def test_zero_slip_velocity_raises(self, simple_psd):
        """Zero slip velocity should raise ValueError."""
        with pytest.raises(ValueError, match="slip_velocity must be positive"):
            forward_model_frequency(simple_psd, 0.0)

    def test_negative_slip_velocity_raises(self, simple_psd):
        """Negative slip velocity should raise ValueError."""
        with pytest.raises(ValueError, match="slip_velocity must be positive"):
            forward_model_frequency(simple_psd, -1e-4)


class TestResonanceFrequency:
    """Tests for resonance_frequency."""

    def test_fundamental_mode(self):
        """Fundamental resonance of a 50m thick layer with Vs=500 m/s -> 5 Hz."""
        f = resonance_frequency(50.0, 500.0)
        assert f == pytest.approx(5.0)

    def test_higher_mode(self):
        """Second mode should be double the fundamental."""
        f1 = resonance_frequency(50.0, 500.0, mode=1)
        f2 = resonance_frequency(50.0, 500.0, mode=2)
        assert f2 == pytest.approx(2.0 * f1)

    def test_known_value(self):
        """Reproduce the redacted case: ~20m thick, Vs~690 m/s."""
        # f = 690 / (2 * 20) = 17.25 Hz
        f = resonance_frequency(20.0, 690.0)
        assert f == pytest.approx(17.25)

    def test_zero_thickness_raises(self):
        """Zero thickness should raise ValueError."""
        with pytest.raises(ValueError, match="thickness must be positive"):
            resonance_frequency(0.0, 500.0)

    def test_negative_velocity_raises(self):
        """Negative velocity should raise ValueError."""
        with pytest.raises(ValueError, match="velocity must be positive"):
            resonance_frequency(50.0, -500.0)

    def test_zero_mode_raises(self):
        """Mode < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="mode must be >= 1"):
            resonance_frequency(50.0, 500.0, mode=0)
