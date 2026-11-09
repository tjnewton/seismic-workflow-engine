"""Tests for swe.detection.injection."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Trace, UTCDateTime
from scipy.stats import norm

from swe.detection import clopper_pearson, inject, inject_trace, injection_recovery


class TestClopperPearson:
    def test_edges(self):
        lo, hi = clopper_pearson(0, 20)
        assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 20))
        lo, hi = clopper_pearson(20, 20)
        assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 20))

    def test_known_value(self):
        lo, hi = clopper_pearson(5, 10)  # textbook: (0.187, 0.813)
        assert lo == pytest.approx(0.1871, abs=1e-4) and hi == pytest.approx(0.8129, abs=1e-4)

    @pytest.mark.parametrize("p", [0.01, 0.5, 0.99])
    def test_coverage_holds_near_zero_and_one(self, p):
        """Clopper-Pearson is conservative: coverage >= nominal, including near p = 0 and 1."""
        rng = np.random.default_rng(1)
        n, reps = 50, 3000
        k = rng.binomial(n, p, size=reps)
        cover = np.mean([lo <= p <= hi for lo, hi in (clopper_pearson(int(x), n) for x in k)])
        assert cover >= 0.95

    def test_bad_inputs(self):
        with pytest.raises(ValueError):
            clopper_pearson(3, 2)
        with pytest.raises(ValueError):
            clopper_pearson(0, 0)
        with pytest.raises(ValueError, match="confidence"):
            clopper_pearson(1, 2, confidence=1.0)


class TestInject:
    def test_signal_lands_at_offset_with_amplitude(self):
        data = np.zeros((6, 8))
        out = inject(data, np.ones((2, 3)), offset=(1, 4), scale=2.5)
        assert out[1:3, 4:7].tolist() == [[2.5] * 3] * 2
        assert out.sum() == pytest.approx(2.5 * 6) and data.sum() == 0.0

    def test_one_dimensional_scalar_offset(self):
        out = inject(np.arange(5.0), np.array([10.0, 20.0]), offset=3)
        assert out.tolist() == [0.0, 1.0, 2.0, 13.0, 24.0]

    def test_out_of_bounds_and_dims(self):
        with pytest.raises(ValueError, match="fit"):
            inject(np.zeros(5), np.ones(3), offset=4)
        with pytest.raises(ValueError, match="dimensions"):
            inject(np.zeros((3, 3)), np.ones(2), offset=0)

    def test_trace(self):
        t0 = UTCDateTime("2020-01-01")
        tr = Trace(np.zeros(100), header={"sampling_rate": 10.0, "starttime": t0})
        out = inject_trace(tr, np.array([1.0, -1.0]), time=t0 + 2.0, scale=3.0)
        assert out.data[20] == 3.0 and out.data[21] == -3.0 and tr.data.sum() == 0.0


class TestInjectionRecovery:
    def test_recovers_analytic_detection_probability(self):
        """Single-sample threshold detector in unit Gaussian noise: P_d = 1 - Phi(thr - amp)."""
        thr = 2.0

        def make(rng, params):
            return inject(rng.normal(size=64), np.array([1.0]), offset=32, scale=params["amp"])

        def detect(data, params):
            return data[32] > thr

        df = injection_recovery(
            make_trial=make,
            detect=detect,
            grid=[{"amp": a} for a in (0.0, 1.0, 2.0, 3.0)],
            n_trials=400,
            random_state=7,
        )
        expect = 1 - norm.cdf(thr - df.amp)
        assert ((df.ci_low <= expect) & (expect <= df.ci_high)).all()
        assert list(df.columns) == [
            "amp",
            "n_trials",
            "n_detected",
            "p_detect",
            "ci_low",
            "ci_high",
        ]

    def test_seed_reproducible(self):
        def make(rng, params):
            return rng.normal()

        def detect(x, params):
            return x > 0

        a = injection_recovery(
            make_trial=make, detect=detect, grid=[{}], n_trials=50, random_state=3
        )
        b = injection_recovery(
            make_trial=make, detect=detect, grid=[{}], n_trials=50, random_state=3
        )
        assert a.equals(b)

    def test_bad_inputs(self):
        with pytest.raises(ValueError):
            injection_recovery(
                make_trial=lambda r, p: 0, detect=lambda d, p: True, grid=[], n_trials=5
            )


class TestMasksEncodingAndCollisions:
    def test_masked_gaps_stay_masked(self):
        data = np.ma.masked_array(np.zeros(6), mask=[0, 0, 1, 1, 0, 0])
        out = inject(data, np.ones(4), offset=1, scale=2.0)
        assert np.ma.isMaskedArray(out) and out.mask.tolist() == data.mask.tolist()
        assert out.filled(-9).tolist() == [0.0, 2.0, -9, -9, 2.0, 0.0]

    def test_integer_mseed_trace_can_be_written_after_injection(self, tmp_path):
        from obspy import read

        tr = Trace(
            np.zeros(100, dtype=np.int32),
            header={"sampling_rate": 10.0, "starttime": UTCDateTime(0)},
        )
        tr.write(str(tmp_path / "in.mseed"), format="MSEED", encoding="STEIM2")
        tr = read(str(tmp_path / "in.mseed"))[0]
        out = inject_trace(tr, np.array([1.5]), time=tr.stats.starttime + 1.0)
        out.write(str(tmp_path / "out.mseed"), format="MSEED")
        assert read(str(tmp_path / "out.mseed"))[0].data[10] == pytest.approx(1.5)

    def test_rows_independent_of_other_grid_entries(self):
        """Each entry has its own random stream: adding entries leaves a row unchanged."""

        def make(rng, params):
            return rng.normal() + params["amp"]

        def detect(x, params):
            return x > 1.5

        both = injection_recovery(
            make_trial=make,
            detect=detect,
            grid=[{"amp": 1.0}, {"amp": 2.0}],
            n_trials=200,
            random_state=5,
        )
        alone = injection_recovery(
            make_trial=make, detect=detect, grid=[{"amp": 1.0}], n_trials=200, random_state=5
        )
        assert both.n_detected.iloc[0] == alone.n_detected.iloc[0]

    def test_grid_keys_colliding_with_results_raise(self):
        with pytest.raises(ValueError, match="collide"):
            injection_recovery(
                make_trial=lambda r, p: 0,
                detect=lambda d, p: True,
                grid=[{"n_trials": 5}],
                n_trials=5,
            )


class TestStreamsOffsetsAndRounding:
    def test_rows_independent_of_order(self):
        def make(rng, params):
            return rng.normal() + params["amp"]

        def detect(x, params):
            return x > 1.5

        grid = [{"amp": 1.0}, {"amp": 2.0}]
        ab = injection_recovery(
            make_trial=make, detect=detect, grid=grid, n_trials=200, random_state=5
        )
        ba = injection_recovery(
            make_trial=make, detect=detect, grid=grid[::-1], n_trials=200, random_state=5
        )
        assert ab.set_index("amp").n_detected.to_dict() == ba.set_index("amp").n_detected.to_dict()

    def test_offset_forms(self):
        assert inject(np.zeros(5), np.ones(1), offset=3)[3] == 1.0
        assert inject(np.zeros(5), np.ones(1), offset=np.array(2))[2] == 1.0
        assert inject(np.zeros((3, 3)), np.ones((1, 1)), offset=(np.int64(1), 2))[1, 2] == 1.0
        for bad in (2.5, 3.0, True):
            with pytest.raises(ValueError, match="integer"):
                inject(np.zeros(5), np.ones(1), offset=bad)

    def test_half_sample_ties_round_up(self):
        t0 = UTCDateTime(0)
        tr = Trace(np.zeros(20), header={"sampling_rate": 10.0, "starttime": t0})
        assert int(np.argmax(inject_trace(tr, np.ones(1), time=t0 + 0.25).data)) == 3
        assert int(np.argmax(inject_trace(tr, np.ones(1), time=t0 + 0.35).data)) == 4


class TestStreamKeys:
    def test_equal_values_of_different_types_share_a_stream(self):
        from swe.detection.injection import _params_key

        assert (
            _params_key({"amp": 1})
            == _params_key({"amp": 1.0})
            == _params_key({"amp": np.float64(1.0)})
        )
        assert _params_key({"a": 1, "b": 2}) == _params_key({"b": 2, "a": 1})

    def test_long_arrays_are_not_truncated(self):
        from swe.detection.injection import _params_key

        b = np.zeros(2000)
        b[1000] = 1.0
        assert _params_key({"x": np.zeros(2000)}) != _params_key({"x": b})

    def test_bad_keys_and_values_raise(self):
        from swe.detection.injection import _params_key

        with pytest.raises(ValueError, match="strings"):
            _params_key({1: 2})
        with pytest.raises(ValueError, match="JSON"):
            _params_key({"f": object()})


class TestHalfSampleTiesAndComplex:
    def test_all_half_sample_ties_round_up_at_100_hz(self):
        t0 = UTCDateTime("2020-01-01T00:00:00")
        tr = Trace(np.zeros(2100), header={"sampling_rate": 100.0, "starttime": t0})
        late = [
            k
            for k in range(1999)
            if int(np.argmax(inject_trace(tr, np.ones(1), time=t0 + (k + 0.5) / 100.0).data))
            != k + 1
        ]
        assert late == []

    def test_complex_raster_keeps_phase(self):
        out = inject(np.zeros((3, 3), dtype=complex), np.full((1, 1), 1j), offset=(1, 1))
        assert np.iscomplexobj(out) and out[1, 1] == 1j


class TestGridAndSeedValidation:
    def test_generator_grid_is_read_once_and_fully(self):
        df = injection_recovery(
            make_trial=lambda r, p: r.normal(),
            detect=lambda d, p: True,
            grid=({"amp": a} for a in (1.0, 2.0)),
            n_trials=3,
        )
        assert df.amp.tolist() == [1.0, 2.0]

    def test_large_ints_keep_distinct_streams(self):
        from swe.detection.injection import _params_key

        assert _params_key({"id": 2**53}) != _params_key({"id": 2**53 + 1})
        assert _params_key({"amp": np.array(3.0)}) == _params_key({"amp": 3})

    def test_negative_random_state_raises(self):
        with pytest.raises(ValueError, match="random_state"):
            injection_recovery(
                make_trial=lambda r, p: 0,
                detect=lambda d, p: True,
                grid=[{}],
                n_trials=1,
                random_state=-1,
            )


class TestInjectDtypesAndTies:
    def test_complex_scale_on_real_data(self):
        out = inject(np.zeros(4), np.ones(1), offset=1, scale=2j)
        assert np.iscomplexobj(out) and out[1] == 2j

    def test_masked_fill_value_preserved(self):
        data = np.ma.masked_array(np.zeros(4), mask=[0, 1, 0, 0], fill_value=-999.0)
        assert inject(data, np.ones(1), offset=0).fill_value == -999.0

    def test_low_rate_ties_are_exact(self):
        t0 = UTCDateTime(0)
        tr = Trace(np.zeros(4), header={"sampling_rate": 0.1, "starttime": t0})
        just_before = UTCDateTime(ns=t0.ns + 5_000_000_000 - 5)
        assert int(np.argmax(inject_trace(tr, np.ones(1), time=just_before).data)) == 0
        assert int(np.argmax(inject_trace(tr, np.ones(1), time=t0 + 5.0).data)) == 1


class TestContracts:
    def test_bad_entry_fails_before_any_trial(self):
        calls = []

        def make(rng, params):
            calls.append(params)
            return 0

        with pytest.raises(ValueError, match="JSON"):
            injection_recovery(
                make_trial=make,
                detect=lambda d, p: True,
                grid=[{"a": 1}, {"a": object()}],
                n_trials=5,
            )
        assert calls == []

    def test_integer_counts_required(self):
        with pytest.raises(ValueError, match="integer"):
            injection_recovery(
                make_trial=lambda r, p: 0, detect=lambda d, p: True, grid=[{}], n_trials=2.5
            )
        with pytest.raises(ValueError, match="integers"):
            clopper_pearson(3.0, 10)
        with pytest.raises(ValueError, match="random_state"):
            injection_recovery(
                make_trial=lambda r, p: 0,
                detect=lambda d, p: True,
                grid=[{}],
                n_trials=1,
                random_state=True,
            )

    def test_mask_free_masked_input_comes_back_plain_and_writable(self, tmp_path):
        out = inject(np.ma.masked_array(np.zeros(4)), np.ones(1), offset=1)
        assert not np.ma.isMaskedArray(out)
        tr = Trace(
            np.ma.masked_array(np.zeros(100)),
            header={"sampling_rate": 10.0, "starttime": UTCDateTime(0)},
        )
        inject_trace(tr, np.ones(1), time=UTCDateTime(1)).write(
            str(tmp_path / "x.mseed"), format="MSEED"
        )

    def test_parameter_columns_come_first_and_names_must_match(self):
        df = injection_recovery(
            make_trial=lambda r, p: 0,
            detect=lambda d, p: True,
            grid=[{"amp": 1.0, "width": 2}, {"amp": 2.0, "width": 3}],
            n_trials=2,
        )
        assert list(df.columns)[:2] == ["amp", "width"] and df.width.tolist() == [2, 3]
        with pytest.raises(ValueError, match="same parameter names"):
            injection_recovery(
                make_trial=lambda r, p: 0,
                detect=lambda d, p: True,
                grid=[{"a": 1}, {"b": 1}],
                n_trials=2,
            )

    def test_namedtuple_and_dataclass_keys(self):
        from collections import namedtuple
        from dataclasses import dataclass

        from swe.detection.injection import _params_key

        Loc, Size = namedtuple("Loc", "a b"), namedtuple("Size", "a b")
        assert _params_key({"p": Loc(1, 2)}) != _params_key({"p": Size(1, 2)})

        @dataclass
        class D:
            x: float

        assert _params_key({"p": D(1)}) == _params_key({"p": D(1.0)})


class TestMediumReviewContracts:
    def test_masked_signal_samples_are_not_injected(self):
        out = inject(np.zeros(2), np.ma.masked_array([1.0, 9e20], mask=[0, 1]), offset=0)
        assert out.tolist() == [1.0, 0.0]

    def test_canonical_keys_do_not_collide(self):
        from collections import namedtuple
        from dataclasses import dataclass

        from swe.detection.injection import _params_key

        assert _params_key({"p": {1: "a"}}) != _params_key({"p": {"1": "a"}})
        nt = namedtuple("NT", "a")
        assert _params_key({"p": nt(1)}) != _params_key(
            {"p": {"namedtuple": "NT", "fields": {"a": 1}}}
        )

        @dataclass
        class Inner:
            x: int

        @dataclass
        class Outer:
            inner: Inner

        assert _params_key({"p": Outer(Inner(1))}) != _params_key({"p": Outer({"x": 1})})

    def test_confidence_and_duplicates_checked_before_trials(self):
        calls = []

        def make(rng, params):
            calls.append(1)
            return 0

        with pytest.raises(ValueError, match="confidence"):
            injection_recovery(
                make_trial=make, detect=lambda d, p: True, grid=[{}], n_trials=5, confidence=95
            )
        with pytest.raises(ValueError, match="duplicate"):
            injection_recovery(
                make_trial=make, detect=lambda d, p: True, grid=[{"a": 1}, {"a": 1.0}], n_trials=5
            )
        assert calls == []
