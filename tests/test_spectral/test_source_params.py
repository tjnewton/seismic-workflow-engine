"""Tests for swe.spectral.source_params (LFE-robust source parameters)."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Trace, UTCDateTime

from swe.spectral.source_params import (
    FalloffResult,
    MomentResult,
    SourceDurationResult,
    high_frequency_falloff,
    moment_from_spectral_level,
    source_time_function_width,
)


def _trace(data, sr=250.0, station="TEST"):
    return Trace(
        data=np.asarray(data, dtype=np.float64),
        header={
            "sampling_rate": sr,
            "starttime": UTCDateTime("2015-02-17T00:00:00"),
            "station": station,
        },
    )


class TestMomentFromSpectralLevel:
    def _boxcar_displacement(self, *, height, width_s, sr=250.0, total_s=20.0):
        """A short interior displacement boxcar of known area = height*width."""
        n = int(total_s * sr)
        data = np.zeros(n)
        w = int(width_s * sr)
        start = n // 2  # well inside, away from the taper
        data[start : start + w] = height
        return _trace(data, sr=sr)

    def test_omega0_equals_pulse_area(self):
        """Low-frequency plateau Omega_0 should equal the pulse area (m*s)."""
        height, width_s = 2.0, 0.1  # area = 0.2 m*s; corner ~ 1/0.1 = 10 Hz
        tr = self._boxcar_displacement(height=height, width_s=width_s)
        res = moment_from_spectral_level(
            tr,
            distance_m=25_000.0,
            plateau_freq_min=0.5,
            plateau_freq_max=2.0,
            taper_fraction=0.01,
        )
        assert res.omega_0 == pytest.approx(height * width_s, rel=0.05)

    def test_moment_formula_value(self):
        """M0 matches 4*pi*rho*c^3*R*Omega_0/(U*F) for known inputs."""
        height, width_s = 1.0, 0.1  # area 0.1 m*s
        tr = self._boxcar_displacement(height=height, width_s=width_s)
        rho, c, R, F = 2700.0, 3500.0, 25_000.0, 1.0
        res = moment_from_spectral_level(
            tr,
            distance_m=R,
            density_kg_m3=rho,
            velocity_m_s=c,
            phase="S",
            free_surface_factor=F,
            plateau_freq_min=0.5,
            plateau_freq_max=2.0,
            taper_fraction=0.01,
        )
        u = np.sqrt(2.0 / 5.0)  # S-wave RMS radiation
        expected = 4.0 * np.pi * rho * c**3 * R * res.omega_0 / (u * F)
        assert res.seismic_moment == pytest.approx(expected, rel=1e-6)
        # And Mw consistent with Hanks & Kanamori
        assert res.moment_magnitude == pytest.approx(
            (2.0 / 3.0) * (np.log10(res.seismic_moment) - 9.05), rel=1e-6
        )

    def test_moment_scales_linearly_with_distance(self):
        tr = self._boxcar_displacement(height=1.0, width_s=0.1)
        kw = dict(plateau_freq_min=0.5, plateau_freq_max=2.0, taper_fraction=0.01)
        m1 = moment_from_spectral_level(tr, distance_m=10_000.0, **kw).seismic_moment
        m2 = moment_from_spectral_level(tr, distance_m=20_000.0, **kw).seismic_moment
        assert m2 == pytest.approx(2.0 * m1, rel=1e-6)

    def test_free_surface_factor_halves_moment(self):
        tr = self._boxcar_displacement(height=1.0, width_s=0.1)
        kw = dict(
            distance_m=25_000.0, plateau_freq_min=0.5, plateau_freq_max=2.0, taper_fraction=0.01
        )
        m_borehole = moment_from_spectral_level(tr, free_surface_factor=1.0, **kw)
        m_surface = moment_from_spectral_level(tr, free_surface_factor=2.0, **kw)
        assert m_surface.seismic_moment == pytest.approx(0.5 * m_borehole.seismic_moment, rel=1e-6)

    def test_invalid_inputs(self):
        tr = self._boxcar_displacement(height=1.0, width_s=0.1)
        with pytest.raises(ValueError):
            moment_from_spectral_level(tr, distance_m=-1.0)
        with pytest.raises(ValueError):
            moment_from_spectral_level(tr, distance_m=1000.0, phase="Z")
        with pytest.raises(ValueError):
            moment_from_spectral_level(
                tr, distance_m=1000.0, plateau_freq_min=5.0, plateau_freq_max=2.0
            )

    def test_returns_named_tuple(self):
        tr = self._boxcar_displacement(height=1.0, width_s=0.1)
        res = moment_from_spectral_level(tr, distance_m=25_000.0)
        assert isinstance(res, MomentResult)
        assert res.station == "TEST"


class TestSourceTimeFunctionWidth:
    def _wavelet(self, sigma_s, carrier_hz=8.0, sr=250.0, total_s=8.0):
        """A Gaussian-modulated carrier: a realistic band-limited LFE wavelet
        whose analytic envelope is a Gaussian of width sigma_s."""
        n = int(total_s * sr)
        t = np.arange(n) / sr
        t0 = total_s / 2.0
        envelope = np.exp(-0.5 * ((t - t0) / sigma_s) ** 2)
        data = envelope * np.sin(2 * np.pi * carrier_hz * (t - t0))
        return _trace(data, sr=sr)

    def test_gaussian_fwhm_recovered(self):
        sigma = 0.2
        tr = self._wavelet(sigma)
        res = source_time_function_width(tr, method="gaussian_fwhm")
        expected_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
        assert res.duration_s == pytest.approx(expected_fwhm, rel=0.1)
        assert res.fit_quality > 0.9

    def test_envelope_halfwidth_method(self):
        sigma = 0.2
        tr = self._wavelet(sigma)
        res = source_time_function_width(tr, method="envelope_halfwidth")
        # Half-amplitude width of a Gaussian envelope equals its FWHM
        assert res.duration_s == pytest.approx(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma, rel=0.25)
        assert np.isnan(res.fit_quality)

    def test_signal_window(self):
        tr = self._wavelet(0.2)
        res = source_time_function_width(tr, method="gaussian_fwhm", signal_window=(2.0, 6.0))
        assert res.duration_s > 0

    def test_invalid_method(self):
        tr = self._wavelet(0.2)
        with pytest.raises(ValueError, match="unknown method"):
            source_time_function_width(tr, method="bogus")

    def test_empty_window_raises(self):
        tr = self._wavelet(0.2)
        with pytest.raises(ValueError, match="empty"):
            source_time_function_width(tr, signal_window=(5.0, 5.0))

    def test_returns_named_tuple(self):
        tr = self._wavelet(0.2)
        assert isinstance(source_time_function_width(tr), SourceDurationResult)


class TestHighFrequencyFalloff:
    def _powerlaw_trace(self, gamma, sr=250.0, n=4096, seed=0):
        """Synthesize a trace whose displacement spectrum ~ f**-gamma."""
        rng = np.random.default_rng(seed)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        amp = np.zeros_like(freqs)
        nz = freqs > 0
        amp[nz] = freqs[nz] ** (-gamma)
        phase = rng.uniform(0, 2 * np.pi, size=freqs.shape)
        spectrum = amp * np.exp(1j * phase)
        data = np.fft.irfft(spectrum, n=n)
        return _trace(data, sr=sr)

    def test_recovers_gamma_2(self):
        tr = self._powerlaw_trace(gamma=2.0)
        res = high_frequency_falloff(tr, freq_min=5.0, freq_max=50.0, taper_fraction=0.01)
        assert res.gamma == pytest.approx(2.0, abs=0.3)

    def test_recovers_gamma_3(self):
        tr = self._powerlaw_trace(gamma=3.0)
        res = high_frequency_falloff(tr, freq_min=5.0, freq_max=50.0, taper_fraction=0.01)
        assert res.gamma == pytest.approx(3.0, abs=0.3)
        # LFE-like steep falloff is distinguishable from omega-squared
        assert res.gamma > 2.5

    def test_invalid_band(self):
        tr = self._powerlaw_trace(gamma=2.0)
        with pytest.raises(ValueError, match="freq_min must be"):
            high_frequency_falloff(tr, freq_min=50.0, freq_max=5.0)

    def test_too_few_samples(self):
        tr = self._powerlaw_trace(gamma=2.0)
        with pytest.raises(ValueError, match=">=3"):
            high_frequency_falloff(tr, freq_min=49.99, freq_max=50.0)

    def test_returns_named_tuple(self):
        tr = self._powerlaw_trace(gamma=2.0)
        res = high_frequency_falloff(tr, freq_min=5.0, freq_max=50.0)
        assert isinstance(res, FalloffResult)
        assert res.gamma_std >= 0.0
