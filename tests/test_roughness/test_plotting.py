"""Tests for swe.roughness._plotting."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from swe.roughness._plotting import (
    plot_forward_model_comparison,
    plot_frequency_roughness_correlation,
    plot_frequency_time_series,
    plot_roughness_psd,
    plot_scaling_law_fit,
    plot_spatial_frequency_map,
)
from swe.roughness.forward import ForwardModelResult
from swe.roughness.frequency import FrequencyTimeSeriesResult, ScalingLawFitResult
from swe.roughness.spatial import SpatialCorrelationResult
from swe.roughness.surface import RoughnessPSDResult


@pytest.fixture()
def freq_ts_result():
    """Minimal FrequencyTimeSeriesResult for plotting."""
    n = 10
    return FrequencyTimeSeriesResult(
        times=np.linspace(730120, 730130, n),
        median_frequency=np.full(n, 34.5),
        std_frequency=np.full(n, 1.0),
        count_per_bin=np.ones(n, dtype=np.int64) * 5,
        bin_width_days=1.0,
        all_frequencies=np.random.default_rng(42).normal(34.5, 1.0, 50),
        all_times=np.random.default_rng(42).uniform(730120, 730130, 50),
    )


@pytest.fixture()
def psd_result():
    """Minimal RoughnessPSDResult for plotting."""
    k = np.linspace(0.1, 50.0, 100)
    psd = k ** (-2.0)
    return RoughnessPSDResult(
        wavelengths=1.0 / k,
        wavenumbers=k,
        psd=psd,
        psd_2d=None,
        dominant_wavelength=0.5,
        spectral_slope=-2.0,
        spectral_slope_std=0.1,
        profile_direction="along-strike",
    )


@pytest.fixture()
def scaling_result():
    """Minimal ScalingLawFitResult for plotting."""
    n = 10
    vel = np.linspace(1e-7, 1e-6, n)
    obs = vel / 1e-8
    pred = obs * 0.98
    return ScalingLawFitResult(
        lambda_0=1e-8,
        lambda_0_std=1e-9,
        r_squared=0.95,
        predicted_frequency=pred,
        observed_frequency=obs,
        slip_velocity=vel,
        residuals=obs - pred,
    )


@pytest.fixture()
def spatial_result():
    """Minimal SpatialCorrelationResult for plotting."""
    n = 20
    rng = np.random.default_rng(42)
    roughness = rng.uniform(0.1, 1.0, n)
    freq = 20.0 + 15.0 * roughness + rng.normal(0, 0.5, n)
    return SpatialCorrelationResult(
        event_x=rng.uniform(0, 100, n),
        event_y=rng.uniform(0, 100, n),
        event_frequency=freq,
        local_roughness=roughness,
        correlation_coefficient=0.85,
        p_value=0.001,
        predicted_frequency=20.0 + 15.0 * roughness,
        roughness_metric="rms_height",
    )


@pytest.fixture()
def forward_result():
    """Minimal ForwardModelResult for plotting."""
    k = np.linspace(0.1, 100, 200)
    stress = np.exp(-((k - 30) ** 2) / 50.0) * k
    return ForwardModelResult(
        predicted_frequency=3.0e-3,
        observed_frequency=34.5,
        frequency_ratio=3.0e-3 / 34.5,
        roughness_psd=k ** (-2),
        slip_velocity=1e-4,
        contact_stiffness=1e9,
        stress_spectrum=stress,
        stress_frequencies=1e-4 * k,
        model_type="elastic_halfspace",
    )


class TestPlotFrequencyTimeSeries:
    """Tests for plot_frequency_time_series."""

    def test_returns_figure(self, freq_ts_result):
        fig = plot_frequency_time_series(freq_ts_result)
        assert isinstance(fig, Figure)

    def test_with_velocity_overlay(self, freq_ts_result):
        vel_times = np.linspace(730120, 730130, 20)
        vel_values = np.linspace(1e-7, 1e-6, 20)
        fig = plot_frequency_time_series(
            freq_ts_result,
            slip_velocity_times=vel_times,
            slip_velocity_values=vel_values,
        )
        assert isinstance(fig, Figure)


class TestPlotRoughnessPSD:
    """Tests for plot_roughness_psd."""

    def test_returns_figure(self, psd_result):
        fig = plot_roughness_psd(psd_result)
        assert isinstance(fig, Figure)

    def test_no_slope_fit(self, psd_result):
        fig = plot_roughness_psd(psd_result, show_slope_fit=False)
        assert isinstance(fig, Figure)

    def test_no_dominant_highlight(self, psd_result):
        fig = plot_roughness_psd(psd_result, highlight_dominant=False)
        assert isinstance(fig, Figure)


class TestPlotScalingLawFit:
    """Tests for plot_scaling_law_fit."""

    def test_returns_figure(self, scaling_result):
        fig = plot_scaling_law_fit(scaling_result)
        assert isinstance(fig, Figure)


class TestPlotSpatialFrequencyMap:
    """Tests for plot_spatial_frequency_map."""

    def test_returns_figure(self):
        x = np.arange(10, dtype=np.float64)
        y = np.arange(8, dtype=np.float64)
        grid = np.random.default_rng(42).uniform(30, 40, (8, 10))
        fig = plot_spatial_frequency_map(x, y, grid)
        assert isinstance(fig, Figure)

    def test_with_stations(self):
        x = np.arange(10, dtype=np.float64)
        y = np.arange(8, dtype=np.float64)
        grid = np.random.default_rng(42).uniform(30, 40, (8, 10))
        stations = np.array([[2.0, 3.0], [7.0, 5.0]])
        fig = plot_spatial_frequency_map(x, y, grid, station_coords=stations)
        assert isinstance(fig, Figure)


class TestPlotFrequencyRoughnessCorrelation:
    """Tests for plot_frequency_roughness_correlation."""

    def test_returns_figure(self, spatial_result):
        fig = plot_frequency_roughness_correlation(spatial_result)
        assert isinstance(fig, Figure)


class TestPlotForwardModelComparison:
    """Tests for plot_forward_model_comparison."""

    def test_returns_figure(self, forward_result):
        fig = plot_forward_model_comparison(forward_result)
        assert isinstance(fig, Figure)
