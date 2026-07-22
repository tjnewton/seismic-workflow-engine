"""Tests for swe.roughness.frequency."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Trace, UTCDateTime

from swe.roughness.frequency import (
    FrequencyTimeSeriesResult,
    ScalingLawFitResult,
    compute_frequency_time_series,
    fit_scaling_law,
    infer_roughness_wavelength,
)


class TestComputeFrequencyTimeSeries:
    """Tests for compute_frequency_time_series."""

    def test_basic_frequency_recovery(self, frequency_traces):
        """Recovered median frequency should be close to the true value."""
        traces, pick_times, freq_true = frequency_traces
        result = compute_frequency_time_series(traces, pick_times, halfwin=0.06)

        assert isinstance(result, FrequencyTimeSeriesResult)
        assert len(result.all_frequencies) > 0

        # Median of all measurements should be within 20% of true
        median_freq = np.nanmedian(result.median_frequency)
        assert abs(median_freq - freq_true) / freq_true < 0.2

    def test_bin_counts(self, frequency_traces):
        """Each bin should have a reasonable count."""
        traces, pick_times, _ = frequency_traces
        result = compute_frequency_time_series(traces, pick_times, bin_width_days=2.0)

        assert len(result.times) > 0
        assert np.all(result.count_per_bin >= 0)
        assert result.count_per_bin.sum() == len(result.all_frequencies)

    def test_frequency_bounds_filtering(self, frequency_traces):
        """Frequency bounds should filter out-of-range measurements."""
        traces, pick_times, _ = frequency_traces
        result = compute_frequency_time_series(
            traces,
            pick_times,
            min_frequency=30.0,
            max_frequency=40.0,
        )

        assert np.all(result.all_frequencies >= 30.0)
        assert np.all(result.all_frequencies <= 40.0)

    def test_nan_handling(self):
        """Traces with NaN instantaneous frequency are skipped."""
        # Very short trace where peak is at boundary -> NaN
        sr = 100.0
        npts = 10
        data = np.zeros(npts)
        data[0] = 10.0  # peak at boundary
        header = {
            "network": "XX",
            "station": "T01",
            "channel": "BHZ",
            "sampling_rate": sr,
            "npts": npts,
            "starttime": UTCDateTime(2024, 1, 1),
        }
        trace = Trace(data=data, header=header)
        result = compute_frequency_time_series([trace], [730120.0], halfwin=0.06)
        assert len(result.all_frequencies) == 0

    def test_length_mismatch_raises(self, frequency_traces):
        """Mismatched traces/picks lengths raises ValueError."""
        traces, pick_times, _ = frequency_traces
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_frequency_time_series(traces[:5], pick_times)

    def test_bin_width_days_stored(self, frequency_traces):
        """bin_width_days is preserved in the result."""
        traces, pick_times, _ = frequency_traces
        result = compute_frequency_time_series(traces, pick_times, bin_width_days=3.0)
        assert result.bin_width_days == 3.0


class TestFitScalingLaw:
    """Tests for fit_scaling_law."""

    def test_synthetic_scaling(self):
        """With f = V / lambda_true, recovered lambda_0 should match."""
        lambda_true = 1e-5  # 10 microns
        times = np.linspace(730120.0, 730130.0, 20)
        velocities = np.linspace(1e-6, 5e-6, 20)
        frequencies = velocities / lambda_true

        # Build a FrequencyTimeSeriesResult
        ts = FrequencyTimeSeriesResult(
            times=times,
            median_frequency=frequencies,
            std_frequency=np.zeros_like(frequencies),
            count_per_bin=np.ones(len(times), dtype=np.int64) * 10,
            bin_width_days=0.5,
            all_frequencies=frequencies,
            all_times=times,
        )

        result = fit_scaling_law(ts, times, velocities)

        assert isinstance(result, ScalingLawFitResult)
        assert abs(result.lambda_0 - lambda_true) / lambda_true < 0.2
        assert result.r_squared > 0.95

    def test_velocity_unit_conversion(self):
        """Velocity unit conversion should produce consistent results."""
        lambda_true = 1e-3
        times = np.linspace(0, 10, 10)
        vel_ms = np.linspace(1e-4, 1e-3, 10)
        freqs = vel_ms / lambda_true

        ts = FrequencyTimeSeriesResult(
            times=times,
            median_frequency=freqs,
            std_frequency=np.zeros(10),
            count_per_bin=np.ones(10, dtype=np.int64),
            bin_width_days=1.0,
            all_frequencies=freqs,
            all_times=times,
        )

        # m/s
        r1 = fit_scaling_law(ts, times, vel_ms, velocity_unit="m/s")

        # Convert to m/day
        vel_mday = vel_ms * 86400.0
        r2 = fit_scaling_law(ts, times, vel_mday, velocity_unit="m/day")

        assert abs(r1.lambda_0 - r2.lambda_0) / r1.lambda_0 < 0.01

    def test_unknown_velocity_unit(self):
        """Unknown velocity unit raises ValueError."""
        ts = FrequencyTimeSeriesResult(
            times=np.array([0.0, 1.0]),
            median_frequency=np.array([10.0, 20.0]),
            std_frequency=np.zeros(2),
            count_per_bin=np.ones(2, dtype=np.int64),
            bin_width_days=1.0,
            all_frequencies=np.array([10.0, 20.0]),
            all_times=np.array([0.0, 1.0]),
        )
        with pytest.raises(ValueError, match="Unknown velocity_unit"):
            fit_scaling_law(
                ts, np.array([0.0, 1.0]), np.array([1.0, 2.0]), velocity_unit="furlongs/fortnight"
            )

    def test_too_few_bins(self):
        """Fewer than 2 valid bins returns NaN result."""
        ts = FrequencyTimeSeriesResult(
            times=np.array([0.0]),
            median_frequency=np.array([10.0]),
            std_frequency=np.zeros(1),
            count_per_bin=np.ones(1, dtype=np.int64),
            bin_width_days=1.0,
            all_frequencies=np.array([10.0]),
            all_times=np.array([0.0]),
        )
        result = fit_scaling_law(ts, np.array([0.0]), np.array([1.0]))
        assert np.isnan(result.lambda_0)


class TestInferRoughnessWavelength:
    """Tests for infer_roughness_wavelength."""

    def test_scalar(self):
        """Scalar inputs should return float."""
        result = infer_roughness_wavelength(34.5, 1.06e-6)
        expected = 1.06e-6 / 34.5
        assert isinstance(result, float)
        assert abs(result - expected) / expected < 1e-10

    def test_array(self):
        """Array inputs should return array."""
        f = np.array([30.0, 35.0, 40.0])
        v = np.array([1e-6, 1e-6, 1e-6])
        result = infer_roughness_wavelength(f, v)
        expected = v / f
        np.testing.assert_allclose(result, expected)

    def test_zero_frequency_raises(self):
        """Zero frequency should raise ValueError."""
        with pytest.raises(ValueError, match="frequency must be positive"):
            infer_roughness_wavelength(0.0, 1e-6)

    def test_negative_velocity_raises(self):
        """Negative velocity should raise ValueError."""
        with pytest.raises(ValueError, match="slip_velocity must be non-negative"):
            infer_roughness_wavelength(34.5, -1e-6)
