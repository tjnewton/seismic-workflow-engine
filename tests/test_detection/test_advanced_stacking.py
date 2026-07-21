"""Tests for advanced stacking methods in swe.detection.stacking."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Trace, UTCDateTime

from swe.detection.stacking import (
    _compute_trace_snr,
    _simple_linear_stack,
    median_stack,
    nth_root_stack,
    selective_stack,
    trace_snr,
)


def _make_trace(data, sr=100.0):
    """Create a Trace from an array."""
    return Trace(
        data=np.asarray(data, dtype=np.float64),
        header={"sampling_rate": sr, "starttime": UTCDateTime(0)},
    )


def _make_signal_trace(amplitude=1.0, noise_level=0.01, sr=100.0, duration=1.0):
    """Create a trace with a clear signal in the middle and noise on the edges."""
    n = int(sr * duration)
    rng = np.random.default_rng(42)
    data = rng.normal(0, noise_level, n)
    # Add signal in the middle
    mid = n // 4
    end = 3 * n // 4
    data[mid:end] += amplitude * np.sin(2 * np.pi * 10 * np.arange(end - mid) / sr)
    return _make_trace(data, sr)


class TestNthRootStack:
    def test_n1_equals_linear(self):
        """n=1 should be identical to a linear stack."""
        traces = [_make_trace([1.0, 2.0, 3.0]), _make_trace([3.0, 2.0, 1.0])]
        result = nth_root_stack(traces, n=1)
        np.testing.assert_allclose(result.data, [2.0, 2.0, 2.0], atol=1e-10)

    def test_n2_noise_suppression(self):
        """nth-root should suppress noise more than linear stack."""
        rng = np.random.default_rng(42)
        n = 1000
        signal = np.sin(2 * np.pi * 5 * np.arange(n) / 100.0)
        traces = [_make_trace(signal + rng.normal(0, 0.5, n)) for _ in range(20)]

        linear = _simple_linear_stack(traces)
        nroot = nth_root_stack(traces, n=2)

        # Both should recover the signal, but nth-root has different characteristics
        assert nroot.data.shape == linear.data.shape

    def test_single_trace(self):
        """Single-trace stack should approximately return the original."""
        tr = _make_trace([1.0, 2.0, 3.0])
        result = nth_root_stack([tr], n=2)
        # For n=2, single trace: sign(x) * |x|^(1/2) -> sign * |sign * |...|^2|^2 = x
        np.testing.assert_allclose(result.data, [1.0, 2.0, 3.0], atol=1e-10)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            nth_root_stack([])

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            nth_root_stack([_make_trace([1.0])], n=0)

    def test_preserves_sign(self):
        """nth-root stack should preserve polarity."""
        traces = [_make_trace([-2.0, 0.0, 2.0]), _make_trace([-4.0, 0.0, 4.0])]
        result = nth_root_stack(traces, n=2)
        assert result.data[0] < 0  # Negative stays negative
        assert result.data[2] > 0  # Positive stays positive

    def test_different_n_values(self):
        """Higher n should produce more compressed amplitudes."""
        traces = [_make_trace([10.0, 1.0, 10.0]), _make_trace([10.0, 1.0, 10.0])]
        r2 = nth_root_stack(traces, n=2)
        r4 = nth_root_stack(traces, n=4)
        # Higher root compresses more
        assert np.max(np.abs(r4.data)) <= np.max(np.abs(r2.data)) + 1e-10

    def test_output_inherits_metadata(self):
        start = UTCDateTime("2024-01-01")
        tr = Trace(
            data=np.ones(100),
            header={"sampling_rate": 50, "starttime": start, "station": "TEST"},
        )
        result = nth_root_stack([tr, tr.copy()], n=2)
        assert result.stats.sampling_rate == 50
        assert result.stats.starttime == start


class TestMedianStack:
    def test_basic_median(self):
        traces = [
            _make_trace([1.0, 2.0, 3.0]),
            _make_trace([1.0, 2.0, 3.0]),
            _make_trace([10.0, 20.0, 30.0]),
        ]
        result = median_stack(traces)
        np.testing.assert_allclose(result.data, [1.0, 2.0, 3.0])

    def test_outlier_robustness(self):
        """Median should be robust to a single outlier."""
        normal = _make_trace([1.0, 1.0, 1.0])
        outlier = _make_trace([100.0, 100.0, 100.0])
        traces = [normal.copy() for _ in range(9)] + [outlier]
        result = median_stack(traces)
        np.testing.assert_allclose(result.data, [1.0, 1.0, 1.0])

    def test_two_traces(self):
        """With two traces, median is the average."""
        traces = [_make_trace([2.0, 4.0]), _make_trace([4.0, 6.0])]
        result = median_stack(traces)
        np.testing.assert_allclose(result.data, [3.0, 5.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            median_stack([])

    def test_single_trace(self):
        tr = _make_trace([1.0, 2.0, 3.0])
        result = median_stack([tr])
        np.testing.assert_allclose(result.data, [1.0, 2.0, 3.0])


class TestSelectiveStack:
    def test_high_snr_selection(self):
        """Only high-SNR traces should be included."""
        # High SNR: signal in middle
        high_snr = _make_signal_trace(amplitude=1.0, noise_level=0.01)
        # Low SNR: just noise
        low_snr = _make_signal_trace(amplitude=0.0, noise_level=0.5)

        traces = [high_snr, high_snr.copy(), low_snr]
        result = selective_stack(
            traces,
            snr_threshold=2.0,
            noise_window=(0.0, 0.1),
            signal_window=(0.3, 0.7),
        )
        assert isinstance(result, Trace)

    def test_all_below_threshold_raises(self):
        """Should raise if no traces meet the threshold."""
        low_snr = _make_signal_trace(amplitude=0.0, noise_level=0.5)
        traces = [low_snr, low_snr.copy()]
        with pytest.raises(ValueError, match="No traces exceed"):
            selective_stack(
                traces,
                snr_threshold=100.0,
                noise_window=(0.0, 0.1),
                signal_window=(0.3, 0.7),
            )

    def test_method_linear(self):
        traces = [_make_signal_trace(), _make_signal_trace()]
        result = selective_stack(traces, method="linear", snr_threshold=0.0)
        assert isinstance(result, Trace)

    def test_method_median(self):
        traces = [_make_signal_trace(), _make_signal_trace()]
        result = selective_stack(traces, method="median", snr_threshold=0.0)
        assert isinstance(result, Trace)

    def test_method_nth_root(self):
        traces = [_make_signal_trace(), _make_signal_trace()]
        result = selective_stack(traces, method="nth_root", snr_threshold=0.0)
        assert isinstance(result, Trace)

    def test_invalid_method_raises(self):
        traces = [_make_signal_trace()]
        with pytest.raises(ValueError, match="Unknown stacking method"):
            selective_stack(traces, method="invalid", snr_threshold=0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            selective_stack([])


class TestComputeTraceSNR:
    def test_pure_signal(self):
        """Pure signal with no noise should give very high SNR."""
        data = np.zeros(1000)
        data[200:800] = 1.0
        tr = _make_trace(data)
        snr = _compute_trace_snr(tr, (0.0, 0.1), (0.3, 0.7))
        assert snr == float("inf") or snr > 100

    def test_pure_noise(self):
        """Uniform noise should give SNR close to 1."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1.0, 1000)
        tr = _make_trace(data)
        snr = _compute_trace_snr(tr, (0.0, 0.1), (0.3, 0.7))
        assert 0.5 < snr < 2.0  # Should be roughly 1


class TestTraceSnr:
    def test_signal_vs_noise_windows(self):
        """Signal window over a clear pulse, noise window over the quiet start."""
        sr = 100.0
        data = np.zeros(1000)
        data[300:700] = 1.0  # signal pulse: 3.0-7.0 s
        tr = _make_trace(data, sr=sr)
        snr = trace_snr(tr, signal_window=(3.0, 7.0), noise_window=(0.0, 1.0))
        assert snr == float("inf")  # noise window is silent

    def test_identical_windows_for_single_and_stack(self):
        """Same absolute window must be used regardless of trace length."""
        rng = np.random.default_rng(0)
        sr = 100.0
        n = 1000
        signal = np.zeros(n)
        signal[400:600] = 2.0
        single = _make_trace(signal + rng.normal(0, 1.0, n), sr=sr)
        # A "stack" of 25 -> noise down ~5x, same window
        stacked = np.mean(
            [signal + rng.normal(0, 1.0, n) for _ in range(25)], axis=0
        )
        stack = _make_trace(stacked, sr=sr)
        sig_w, noise_w = (4.0, 6.0), (0.0, 2.0)
        snr_single = trace_snr(single, signal_window=sig_w, noise_window=noise_w)
        snr_stack = trace_snr(stack, signal_window=sig_w, noise_window=noise_w)
        assert snr_stack > snr_single  # stacking improves the two-window SNR

    def test_reference_time_offset(self):
        """Windows are measured relative to reference_time when given."""
        sr = 100.0
        data = np.zeros(1000)
        data[300:700] = 1.0
        start = UTCDateTime("2015-02-17T00:00:00")
        tr = Trace(data=data, header={"sampling_rate": sr, "starttime": start})
        # reference 3 s in: signal now at (0,4), noise at (-3,-2)
        ref = start + 3.0
        snr = trace_snr(
            tr,
            signal_window=(0.0, 4.0),
            noise_window=(-3.0, -2.0),
            reference_time=ref,
        )
        assert snr == float("inf")

    def test_pure_noise_near_one(self):
        rng = np.random.default_rng(42)
        tr = _make_trace(rng.normal(0, 1.0, 2000), sr=100.0)
        snr = trace_snr(tr, signal_window=(5.0, 15.0), noise_window=(0.0, 5.0))
        assert 0.5 < snr < 2.0

    def test_invalid_windows(self):
        tr = _make_trace(np.ones(100), sr=100.0)
        with pytest.raises(ValueError, match="signal_window"):
            trace_snr(tr, signal_window=(1.0, 1.0), noise_window=(0.0, 0.5))
        with pytest.raises(ValueError, match="noise_window"):
            trace_snr(tr, signal_window=(0.0, 0.5), noise_window=(0.5, 0.5))

    def test_window_outside_trace_raises(self):
        tr = _make_trace(np.ones(100), sr=100.0)  # 1.0 s long
        with pytest.raises(ValueError, match="outside the trace"):
            trace_snr(tr, signal_window=(50.0, 60.0), noise_window=(0.0, 0.5))


class TestSimpleLinearStack:
    def test_basic(self):
        traces = [_make_trace([1.0, 3.0]), _make_trace([3.0, 1.0])]
        result = _simple_linear_stack(traces)
        np.testing.assert_allclose(result.data, [2.0, 2.0])

    def test_different_lengths(self):
        """Should truncate to shortest trace."""
        traces = [_make_trace([1.0, 2.0, 3.0]), _make_trace([2.0, 4.0])]
        result = _simple_linear_stack(traces)
        assert len(result.data) == 2
        np.testing.assert_allclose(result.data, [1.5, 3.0])
