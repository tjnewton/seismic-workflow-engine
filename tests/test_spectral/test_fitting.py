"""Tests for swe.spectral.fitting."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Trace, UTCDateTime

from swe.spectral.fitting import (
    SpectralFitResult,
    StressDropResult,
    brune_spectrum,
    compute_displacement_spectrum,
    compute_stress_drop,
    fit_brune_model,
    spectral_fit_batch,
)


class TestBruneSpectrum:
    """Tests for the brune_spectrum function."""

    def test_returns_omega0_at_low_frequency(self):
        freqs = np.array([0.01, 0.1])
        result = brune_spectrum(freqs, omega_0=1e-6, corner_frequency=5.0)
        np.testing.assert_allclose(result, 1e-6, rtol=0.01)

    def test_falls_off_at_high_frequency(self):
        freqs = np.array([50.0, 100.0])
        result = brune_spectrum(freqs, omega_0=1e-6, corner_frequency=5.0)
        assert np.all(result < 1e-6 * 0.01)

    def test_corner_frequency_halfpoint(self):
        """At f=fc, amplitude should be omega_0 / 2 for n_gamma=2."""
        freqs = np.array([5.0])
        result = brune_spectrum(freqs, omega_0=1e-6, corner_frequency=5.0, n_gamma=2.0)
        expected = 1e-6 / 2.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_custom_n_gamma(self):
        freqs = np.array([10.0])
        result_2 = brune_spectrum(freqs, omega_0=1.0, corner_frequency=5.0, n_gamma=2.0)
        result_3 = brune_spectrum(freqs, omega_0=1.0, corner_frequency=5.0, n_gamma=3.0)
        # Steeper falloff with higher n_gamma
        assert result_3[0] < result_2[0]

    def test_returns_ndarray(self):
        freqs = np.linspace(0.1, 50.0, 100)
        result = brune_spectrum(freqs, omega_0=1e-6, corner_frequency=5.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == freqs.shape


class TestComputeDisplacementSpectrum:
    """Tests for compute_displacement_spectrum."""

    def test_returns_tuple(self, brune_trace):
        freqs, amps = compute_displacement_spectrum(brune_trace)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(amps, np.ndarray)
        assert len(freqs) == len(amps)

    def test_respects_freq_min(self, brune_trace):
        freqs, _ = compute_displacement_spectrum(brune_trace, freq_min=2.0)
        assert freqs[0] >= 2.0

    def test_respects_freq_max(self, brune_trace):
        freqs, _ = compute_displacement_spectrum(brune_trace, freq_max=20.0)
        assert freqs[-1] <= 20.0

    def test_no_response_uses_raw(self, brune_trace):
        """Trace without response metadata should still return spectra."""
        freqs, amps = compute_displacement_spectrum(brune_trace)
        assert len(freqs) > 0
        assert np.all(amps >= 0)


class TestFitBruneModel:
    """Tests for fit_brune_model."""

    def test_returns_spectral_fit_result(self, brune_trace):
        result = fit_brune_model(brune_trace, freq_min=0.5, freq_max=45.0)
        assert isinstance(result, SpectralFitResult)

    def test_recovers_corner_frequency(self, brune_trace):
        """Corner frequency should be recovered within ±50%."""
        result = fit_brune_model(brune_trace, freq_min=0.5, freq_max=45.0)
        assert not np.isnan(result.corner_frequency)
        # fc=5.0 Hz in the fixture
        assert 2.5 < result.corner_frequency < 10.0

    def test_recovers_omega0(self, brune_trace):
        """Long-period level should be within an order of magnitude."""
        result = fit_brune_model(brune_trace, freq_min=0.5, freq_max=45.0)
        assert not np.isnan(result.long_period_level)

    def test_station_name(self, brune_trace):
        result = fit_brune_model(brune_trace)
        assert result.station == "XX.BRN1.BHZ"

    def test_short_trace_returns_nan(self):
        """Very short trace should return NaN gracefully."""
        header = {
            "network": "XX",
            "station": "STA",
            "channel": "BHZ",
            "sampling_rate": 10.0,
            "npts": 4,
            "starttime": UTCDateTime(2024, 1, 1),
        }
        short = Trace(data=np.random.default_rng(42).standard_normal(4), header=header)
        result = fit_brune_model(short, freq_min=2.0)
        assert np.isnan(result.corner_frequency)

    def test_fitted_spectrum_shape_matches_frequencies(self, brune_trace):
        result = fit_brune_model(brune_trace)
        assert result.frequencies.shape == result.fitted_spectrum.shape
        assert result.frequencies.shape == result.observed_spectrum.shape

    def test_cornerless_spectrum_stays_within_band(self):
        """A spectrum with no corner in the band must not diverge past freq_max.

        Regression: previously the unbounded least-squares fit could return a
        corner frequency far above the data (~1e5 Hz) for a weak, cornerless
        stack.  fc is now constrained to the fitted [freq_min, freq_max] band.
        """
        sr = 100.0
        npts = 4096
        rng = np.random.default_rng(7)
        header = {
            "network": "XX",
            "station": "FLAT",
            "channel": "BHZ",
            "sampling_rate": sr,
            "npts": npts,
            "starttime": UTCDateTime(2024, 1, 1),
        }
        # White noise -> no Brune corner within 1.5-7 Hz
        flat = Trace(data=rng.standard_normal(npts).astype(np.float64), header=header)
        result = fit_brune_model(flat, freq_min=1.5, freq_max=7.0)
        assert not np.isnan(result.corner_frequency)
        assert 1.5 - 1e-6 <= result.corner_frequency <= 7.0 + 1e-6


class TestComputeStressDrop:
    """Tests for compute_stress_drop."""

    def test_returns_stress_drop_result(self, brune_trace):
        fit = fit_brune_model(brune_trace)
        result = compute_stress_drop(fit, magnitude=2.0)
        assert isinstance(result, StressDropResult)

    def test_stress_drop_positive(self, brune_trace):
        fit = fit_brune_model(brune_trace)
        result = compute_stress_drop(fit, magnitude=2.0)
        assert result.stress_drop_mpa > 0

    def test_seismic_moment_from_magnitude(self, brune_trace):
        fit = fit_brune_model(brune_trace)
        result = compute_stress_drop(fit, magnitude=3.0)
        expected_m0 = 10.0 ** (1.5 * 3.0 + 9.05)
        np.testing.assert_allclose(result.seismic_moment, expected_m0, rtol=1e-10)

    def test_explicit_seismic_moment(self, brune_trace):
        fit = fit_brune_model(brune_trace)
        result = compute_stress_drop(fit, seismic_moment=1e14)
        assert result.seismic_moment == 1e14

    def test_madariaga_vs_brune_k(self, brune_trace):
        fit = fit_brune_model(brune_trace)
        r_mad = compute_stress_drop(fit, magnitude=2.0, source_model="madariaga")
        r_brn = compute_stress_drop(fit, magnitude=2.0, source_model="brune")
        # Very similar k-constants, so results should be close
        np.testing.assert_allclose(r_mad.source_radius_m, r_brn.source_radius_m, rtol=0.01)

    def test_raises_without_moment_or_magnitude(self, brune_trace):
        fit = fit_brune_model(brune_trace)
        with pytest.raises(ValueError, match="seismic_moment or magnitude"):
            compute_stress_drop(fit)

    def test_raises_unknown_source_model(self, brune_trace):
        fit = fit_brune_model(brune_trace)
        with pytest.raises(ValueError, match="Unknown source model"):
            compute_stress_drop(fit, magnitude=2.0, source_model="unknown")


class TestSpectralFitBatch:
    """Tests for spectral_fit_batch."""

    def test_batch_returns_list(self, brune_trace):
        results = spectral_fit_batch(
            [brune_trace, brune_trace],
            magnitudes=[2.0, 3.0],
        )
        assert len(results) == 2
        assert all(isinstance(r, StressDropResult) for r in results)

    def test_raises_without_magnitudes_or_moments(self, brune_trace):
        with pytest.raises(ValueError, match="magnitudes or seismic_moments"):
            spectral_fit_batch([brune_trace])

    def test_raises_mismatched_lengths(self, brune_trace):
        with pytest.raises(ValueError, match="magnitudes length"):
            spectral_fit_batch(
                [brune_trace, brune_trace],
                magnitudes=[2.0],
            )
