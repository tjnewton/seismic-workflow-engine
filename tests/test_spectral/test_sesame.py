"""Tests for the SESAME H/V criteria, lognormal statistics, anti-trigger and smoothing."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from scipy.signal import iirpeak, lfilter

from swe.spectral import (
    AntiTrigger,
    HVSRResult,
    compute_hvsr,
    compute_ssr,
    sesame_criteria,
    sesame_thresholds,
)
from swe.spectral.ratios import _konno_ohmachi_smoothing, _samples

SR = 50.0


def _stream(*, gain=6.0, f_r=2.0, duration=900.0, seed=1, spike_at=None):
    """Vertical white noise; horizontals white noise plus a resonance at f_r (Q = 4)."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    b, a = iirpeak(f_r, 4.0, fs=SR)
    st = Stream()
    for ch in ("HHZ", "HHN", "HHE"):
        x = rng.standard_normal(n)
        if ch != "HHZ" and gain:
            x = x + gain * lfilter(b, a, rng.standard_normal(n))
        if spike_at is not None:
            i = int(spike_at * SR)
            x[i : i + 25] += 200.0
        header = {
            "network": "XX",
            "station": "S1",
            "channel": ch,
            "sampling_rate": SR,
            "starttime": UTCDateTime(0),
        }
        st += Trace(x, header=header)
    return st


def _hvsr(st, **kw):
    return compute_hvsr(
        st, window_length=kw.pop("window_length", 30.0), freq_min=0.2, freq_max=20.0, **kw
    )


class TestSesameCriteria:
    def test_resonance_is_reliable_and_clear(self):
        c = sesame_criteria(_hvsr(_stream()))
        assert abs(c.f0 / 2.0 - 1) < 0.05
        assert c.a0 > 2
        assert c.reliable
        assert c.clear
        assert c.clear_peak[:4] == (True, True, True, True)

    def test_flat_curve_is_not_clear(self):
        c = sesame_criteria(_hvsr(_stream(gain=0.0)))
        assert c.a0 < 2
        assert not c.clear_peak[2]
        assert not c.clear

    def test_window_too_short_for_f0(self):
        # f0 = 2 Hz needs lw > 10 / f0 = 5 s (reliability i)
        c = sesame_criteria(_hvsr(_stream(), window_length=4.0))
        assert not c.reliability[0]
        assert not c.reliable

    def test_too_few_cycles(self):
        # 50 s of 20 s windows at 50% overlap: 4 windows, nc = 20 * 4 * ~2 Hz ~ 160 < 200
        c = sesame_criteria(
            _hvsr(_stream(duration=50.0), window_length=20.0), search_range=(1.0, 4.0)
        )
        assert c.n_cycles == pytest.approx(20.0 * 4 * c.f0)
        assert c.n_cycles < 200
        assert not c.reliability[1]
        assert sesame_criteria(_hvsr(_stream()), search_range=(1.0, 4.0)).reliability[1]

    def test_dc_never_chosen_as_f0(self):
        st = _stream(duration=120.0)
        for tr in st:
            tr.data = tr.data + (500.0 if tr.stats.channel != "HHZ" else 0.0)
        r = compute_hvsr(st, window_length=30.0, freq_min=0.0, freq_max=20.0)
        assert r.frequencies[0] > 0  # the DC bin is never kept
        assert sesame_criteria(r).f0 > 0

    def test_nan_sigma_is_skipped(self):
        r = _hvsr(_stream(duration=300.0))
        log_std = r.hvsr_log_std.copy()
        i = int(np.argmin(np.abs(r.frequencies - 2.0)))
        log_std[i + 3] = np.nan  # e.g. a bin where only one window has H/V > 0
        c = sesame_criteria(r._replace(hvsr_log_std=log_std))
        assert np.isfinite(c.sigma_a_f0)
        assert abs(c.f0 / 2.0 - 1) < 0.05
        assert c.clear_peak[3]  # the NaN bin must not capture the A * sigma_A peak search

    def test_search_range_moves_f0(self):
        r = _hvsr(_stream())
        c = sesame_criteria(r, search_range=(5.0, 20.0))
        assert 5.0 <= c.f0 <= 20.0

    def test_search_range_empty_raises(self):
        with pytest.raises(ValueError, match="selects no frequency"):
            sesame_criteria(_hvsr(_stream(duration=120.0)), search_range=(30.0, 40.0))

    def test_result_without_lognormal_curves_raises(self):
        f = np.linspace(0.5, 10, 20)
        r = HVSRResult(f, np.ones(20), np.zeros(20), np.ones((3, 20)), 1.0, 1.0, 3, "XX.S1")
        with pytest.raises(ValueError, match="lognormal"):
            sesame_criteria(r)


class TestSesameThresholds:
    @pytest.mark.parametrize(
        ("f0", "eps_frac", "theta"),
        [
            (0.1, 0.25, 3.0),
            (0.2, 0.20, 2.5),
            (0.49, 0.20, 2.5),
            (0.5, 0.15, 2.0),
            (1.0, 0.10, 1.78),
            (2.0, 0.05, 1.58),
            (20.0, 0.05, 1.58),
        ],
    )
    def test_table(self, f0, eps_frac, theta):
        eps, th = sesame_thresholds(f0)
        assert eps == pytest.approx(eps_frac * f0)
        assert th == theta

    @pytest.mark.parametrize("f0", [0.0, -1.0, np.nan, np.inf])
    def test_bad_f0_raises(self, f0):
        with pytest.raises(ValueError):
            sesame_thresholds(f0)


class TestLognormalStatistics:
    def test_median_and_log_std(self):
        r = _hvsr(_stream(duration=300.0))
        logs = np.log(r.hvsr_per_window)
        np.testing.assert_allclose(r.hvsr_median, np.exp(logs.mean(axis=0)), rtol=1e-12)
        np.testing.assert_allclose(r.hvsr_log_std, logs.std(axis=0, ddof=1), rtol=1e-12)
        assert r.window_length == 30.0
        assert r.n_rejected == 0


class TestAntiTrigger:
    def test_rejects_windows_with_a_transient(self):
        st = _stream(gain=0.0, spike_at=400.0)
        total = _hvsr(st).n_windows
        r = _hvsr(st, anti_trigger=AntiTrigger(sta=1.0, lta=30.0, max_ratio=2.0, min_ratio=0.2))
        assert r.n_windows + r.n_rejected == total
        # the spike's windows plus the LTA warm-up and recovery, not the rest of the record
        assert 3 <= r.n_rejected <= 10

    def test_quiet_record_keeps_windows_after_warm_up(self):
        st = _stream(gain=0.0)
        total = _hvsr(st).n_windows
        r = _hvsr(st, anti_trigger=AntiTrigger())
        assert 0 < r.n_rejected <= 3  # only windows starting within the first lta seconds
        assert r.n_windows + r.n_rejected == total

    def test_everything_rejected_raises(self):
        with pytest.raises(ValueError, match="every window"):
            _hvsr(_stream(duration=120.0), anti_trigger=AntiTrigger(max_ratio=0.1, min_ratio=0.0))

    def test_record_shorter_than_lta_raises(self):
        with pytest.raises(ValueError, match="shorter than the anti-trigger lta"):
            _hvsr(_stream(duration=20.0), window_length=10.0, anti_trigger=AntiTrigger(lta=30.0))

    def test_band_settle_zones_count_in_the_length_check(self):
        # 40 s is longer than lta = 30 s, but not than lta + 2 x 4 / 0.5 Hz = 46 s
        at = AntiTrigger(lta=30.0, band=(0.5, 10.0))
        with pytest.raises(ValueError, match="plus the band-pass settle zones"):
            _hvsr(_stream(duration=40.0), window_length=10.0, anti_trigger=at)

    def test_offset_does_not_hide_a_transient(self):
        st = _stream(gain=0.0, spike_at=400.0)
        for tr in st:
            tr.data = tr.data + 1e5
        assert _hvsr(st, anti_trigger=AntiTrigger()).n_rejected >= 3

    def test_mapping_is_accepted(self):
        st = _stream(gain=0.0, spike_at=400.0)
        a = _hvsr(st, anti_trigger={"sta": 1.0, "lta": 30.0})
        b = _hvsr(st, anti_trigger=AntiTrigger(sta=1.0, lta=30.0))
        assert a.n_rejected == b.n_rejected > 0

    def test_sta_not_shorter_than_lta_raises(self):
        with pytest.raises(ValueError, match="sta < lta"):
            _hvsr(_stream(duration=120.0), anti_trigger=AntiTrigger(sta=30.0, lta=30.0))


def _ko_reference(spectrum, freqs, bandwidth):
    """The previous loop implementation, kept as a regression reference."""
    out = np.zeros_like(spectrum, dtype=float)
    for i, fc in enumerate(freqs):
        if fc <= 0:
            out[i] = spectrum[i]
            continue
        ratio = freqs / fc
        valid = ratio > 0
        w = np.zeros_like(freqs)
        arg = bandwidth * np.log10(ratio[valid])
        with np.errstate(divide="ignore", invalid="ignore"):
            w[valid] = np.where(np.abs(arg) < 1e-10, 1.0, (np.sin(arg) / arg) ** 4)
        out[i] = np.sum(w * spectrum) / np.sum(w)
    return out


class TestSmoothing:
    def test_konno_ohmachi_matches_reference(self):
        rng = np.random.default_rng(3)
        f = np.fft.rfftfreq(2000, 1 / 100)[:600]  # includes DC
        s = np.abs(rng.standard_normal(f.size)) + 1
        np.testing.assert_allclose(
            _konno_ohmachi_smoothing(s, f, bandwidth=40.0), _ko_reference(s, f, 40.0), rtol=1e-10
        )

    def test_konno_ohmachi_rows(self):
        rng = np.random.default_rng(4)
        f = np.linspace(0.1, 20, 300)
        s = np.abs(rng.standard_normal((3, f.size))) + 1
        out = _konno_ohmachi_smoothing(s, f, bandwidth=40.0)
        for k in range(3):
            np.testing.assert_allclose(out[k], _ko_reference(s[k], f, 40.0), rtol=1e-10)

    def test_none_leaves_spectra_unsmoothed(self):
        st = _stream(duration=120.0)
        a = _hvsr(st, smoothing_type="none")
        b = _hvsr(st)
        assert a.hvsr_per_window.shape == b.hvsr_per_window.shape
        assert np.std(np.diff(a.hvsr_mean)) > np.std(np.diff(b.hvsr_mean))

    def test_unknown_smoothing_raises_hvsr(self):
        with pytest.raises(ValueError, match="smoothing_type"):
            _hvsr(_stream(duration=120.0), smoothing_type="parzen")

    def test_unknown_smoothing_raises_ssr(self):
        st = _stream(duration=60.0)
        with pytest.raises(ValueError, match="smoothing_type"):
            compute_ssr([st[1]], [st[0]], smoothing_type="parzen")

    def test_unknown_merge_raises(self):
        with pytest.raises(ValueError, match="merge_method"):
            _hvsr(_stream(duration=120.0), merge_method="max")


def test_konno_ohmachi_blocks_match_single_block(monkeypatch):
    import swe.spectral.ratios as ratios

    rng = np.random.default_rng(5)
    f = np.linspace(0.1, 20, 257)
    s = np.abs(rng.standard_normal((2, f.size))) + 1
    whole = _konno_ohmachi_smoothing(s, f, bandwidth=40.0)
    monkeypatch.setattr(ratios, "_KO_BLOCK_ELEMENTS", 3 * f.size)  # blocks of 3 centre frequencies
    np.testing.assert_allclose(_konno_ohmachi_smoothing(s, f, bandwidth=40.0), whole, rtol=1e-13)


@pytest.mark.parametrize("overlap", [1.0, 1.5, -0.1])
def test_bad_overlap_raises(overlap):
    with pytest.raises(ValueError, match="overlap"):
        _hvsr(_stream(duration=120.0), overlap=overlap)


def test_overlap_without_a_step_raises():
    with pytest.raises(ValueError, match="no step"):
        compute_hvsr(
            _stream(duration=60.0), window_length=0.1, overlap=0.99, freq_min=0.2, freq_max=20.0
        )


def test_window_chunks_match_one_chunk(monkeypatch):
    import swe.spectral.ratios as ratios

    st = _stream(duration=300.0)
    whole = _hvsr(st)
    monkeypatch.setattr(ratios, "_WINDOW_CHUNK", 3)
    np.testing.assert_allclose(_hvsr(st).hvsr_per_window, whole.hvsr_per_window, rtol=1e-12)


class TestDeadData:
    def test_zero_filled_window_is_not_a_peak(self):
        st = _stream()
        clean = sesame_criteria(_hvsr(st), search_range=(1.0, 4.0))
        st.select(channel="HHE")[0].data[: int(40 * SR)] = 0.0  # merge(fill_value=0) style gap
        assert _hvsr(st).n_rejected >= 1  # by default the zero run is a gap
        r = _hvsr(st, zero_gap=None)  # zeros as data: the dead windows are still rejected
        c = sesame_criteria(r, search_range=(1.0, 4.0))
        assert r.n_rejected >= 1
        assert not (r.hvsr_per_window == 0).all(axis=1).any()
        assert c.n_cycles == pytest.approx(30.0 * r.n_windows * c.f0)
        assert c.sigma_f == pytest.approx(clean.sigma_f, rel=0.25)
        assert c.clear_peak[4] == clean.clear_peak[4]

    def test_nan_sigma_in_band_does_not_fail_reliability(self):
        r = _hvsr(_stream())
        log_std = r.hvsr_log_std.copy()
        log_std[int(np.argmin(np.abs(r.frequencies - 2.5)))] = np.nan
        assert sesame_criteria(r._replace(hvsr_log_std=log_std)).reliability[2]

    def test_dead_component_gives_no_warning_and_a_clear_error(self):
        import warnings

        st = _stream(duration=120.0)
        st.select(channel="HHE")[0].data[:] = 0.0
        with warnings.catch_warnings(), pytest.raises(ValueError, match="dead, railed or constant"):
            warnings.simplefilter("error")
            _hvsr(st, smoothing_type="none", zero_gap=None)
        with pytest.raises(ValueError, match="contains a gap"):
            _hvsr(st)  # by default the all-zero component is one long gap


class TestAntiTriggerConfig:
    def test_min_not_below_max_raises(self):
        with pytest.raises(ValueError, match="min_ratio < max_ratio"):
            _hvsr(_stream(duration=120.0), anti_trigger=AntiTrigger(min_ratio=2.5, max_ratio=2.0))

    @pytest.mark.parametrize("bad", [True, [1.0, 30.0, 2.0, 0.2], 3.0])
    def test_wrong_type_raises(self, bad):
        with pytest.raises(TypeError, match="AntiTrigger or a mapping"):
            _hvsr(_stream(duration=120.0), anti_trigger=bad)

    def test_warm_up_named_in_error(self):
        with pytest.raises(ValueError, match="first lta"):
            _hvsr(_stream(gain=0.0, duration=60.0), window_length=40.0, anti_trigger=AntiTrigger())


class TestThirdReview:
    def test_single_window_is_refused(self):
        r = compute_hvsr(_stream(duration=45.0), window_length=40.0, freq_min=0.2, freq_max=20.0)
        assert r.n_windows == 1
        assert not np.isfinite(r.hvsr_log_std).any()  # undefined, not zero
        with pytest.raises(ValueError, match="at least 2 windows"):
            sesame_criteria(r)

    def test_window_length_is_the_length_used(self):
        r = compute_hvsr(_stream(duration=120.0), window_length=10.01, freq_min=0.2, freq_max=20.0)
        assert r.window_length == pytest.approx(int(10.01 * SR) / SR)

    def test_mismatched_sampling_rates_raise(self):
        st = _stream(duration=120.0)
        st.select(channel="HHN")[0].stats.sampling_rate = 100.0
        with pytest.raises(ValueError, match="sampling rate"):
            _hvsr(st)

    def test_duplicate_horizontal_raises(self):
        st = _stream(duration=120.0)
        st.select(channel="HHE")[0].stats.channel = "HHN"
        with pytest.raises(ValueError, match="more than one trace"):
            _hvsr(st)


class TestFourthReview:
    def test_warm_up_rejected_even_with_zero_min_ratio(self):
        st = _stream(gain=0.0, duration=600.0, spike_at=2.0)
        r = _hvsr(st, window_length=20.0, anti_trigger=AntiTrigger(min_ratio=0.0))
        assert r.n_rejected >= 2  # windows starting within the first 30 s

    def test_lowercase_duplicate_horizontal_raises(self):
        st = _stream(duration=120.0)
        st.select(channel="HHE")[0].stats.channel = "HHn"
        with pytest.raises(ValueError, match="more than one trace"):
            _hvsr(st)

    def test_selection_does_not_depend_on_trace_order(self):
        st = _stream(duration=120.0)
        a = _hvsr(st)
        st.traces = [st.traces[2], st.traces[0], st.traces[1]]  # E, Z, N
        np.testing.assert_allclose(_hvsr(st).hvsr_per_window, a.hvsr_per_window)

    def test_unpaired_horizontals_raise(self):
        st = _stream(duration=120.0)
        st.select(channel="HHE")[0].stats.channel = "HH1"
        with pytest.raises(ValueError, match="N/E or 1/2"):
            _hvsr(st)

    def test_two_verticals_raise(self):
        st = _stream(duration=120.0)
        extra = st.select(channel="HHZ")[0].copy()
        extra.stats.channel = "BHZ"
        st += extra
        with pytest.raises(ValueError, match=r"\['Z'\]"):
            _hvsr(st)

    def test_chunked_lognormal_stats_match_direct(self, monkeypatch):
        import swe.spectral.ratios as ratios

        r = _hvsr(_stream(duration=300.0))
        monkeypatch.setattr(ratios, "_WINDOW_CHUNK", 4)
        med, lstd = ratios._lognormal_stats(r.hvsr_per_window)
        logs = np.log(r.hvsr_per_window)
        np.testing.assert_allclose(med, np.exp(logs.mean(axis=0)), rtol=1e-12)
        np.testing.assert_allclose(lstd, logs.std(axis=0, ddof=1), rtol=1e-10)


def _gappy(gap=(200.0, 250.0), duration=450.0):
    """Merged stream with a gap on every component (masked samples)."""
    st = _stream(duration=duration)
    out = Stream()
    for tr in st:
        a = tr.slice(tr.stats.starttime, tr.stats.starttime + gap[0])
        b = tr.slice(tr.stats.starttime + gap[1], tr.stats.endtime)
        out += a.copy()
        out += b.copy()
    return out.merge()


class TestFifthReview:
    def test_merged_gap_windows_are_rejected(self):
        st = _gappy()
        assert np.ma.isMaskedArray(st[0].data)
        r = _hvsr(st, window_length=20.0)
        assert r.n_rejected >= 4
        assert not (r.hvsr_per_window == 0).all(axis=1).any()  # no all-zero curves
        assert np.isfinite(r.hvsr_mean).all()

    def test_nan_in_one_horizontal_only(self):
        st = _stream(duration=300.0)
        st.select(channel="HHN")[0].data[5000:5100] = np.nan
        r = _hvsr(st)
        assert np.isfinite(r.hvsr_mean).all()
        assert r.n_rejected >= 1
        assert np.isfinite(r.peak_amplitude)

    def test_anti_trigger_keeps_clean_windows_after_a_gap(self):
        st = _gappy()
        plain = _hvsr(st, window_length=20.0)
        r = _hvsr(st, window_length=20.0, anti_trigger=AntiTrigger(max_ratio=3.0))
        # windows well after the gap (and its LTA recovery) survive
        assert r.n_windows >= plain.n_windows // 2

    def test_empty_band_raises(self):
        with pytest.raises(ValueError, match="no FFT frequency"):
            compute_hvsr(_stream(duration=120.0), window_length=30.0, freq_min=30.0, freq_max=40.0)

    @pytest.mark.parametrize("frac", [-0.1, 0.7, 1.5])
    def test_bad_taper_raises(self, frac):
        with pytest.raises(ValueError, match="taper_fraction"):
            _hvsr(_stream(duration=120.0), taper_fraction=frac)

    def test_single_window_std_is_undefined(self):
        r = compute_hvsr(_stream(duration=45.0), window_length=40.0, freq_min=0.2, freq_max=20.0)
        assert np.isnan(r.hvsr_std).all()


class TestSixthReview:
    def test_gap_rejections_are_the_gap_plus_its_warm_up(self):
        st = _stream(gain=0.0, duration=600.0)
        for tr in st:
            tr.data[int(300 * SR) : int(330 * SR)] = np.nan
        r = _hvsr(st, window_length=20.0, anti_trigger=AntiTrigger())
        # 3 record warm-up + 4 overlapping the gap + 3 in the post-gap warm-up (330-360 s)
        assert r.n_rejected == 10
        assert r.n_windows == 59 - 10

    def test_sampling_rates_equal_to_rounding_are_accepted(self):
        st = _stream(duration=120.0)
        for tr in st.select(channel="HH[NE]"):
            tr.stats.sampling_rate = SR * (1 + 1e-9)
        assert _hvsr(st).n_windows > 0

    def test_bad_settings_fail_before_data_work(self):
        with pytest.raises(ValueError, match="sta < lta"):
            _hvsr(_stream(duration=120.0), anti_trigger=AntiTrigger(sta=40.0, lta=30.0))

    def test_single_event_ssr_spread_is_undefined(self):
        st = _stream(duration=60.0)
        r = compute_ssr([st[1]], [st[0]])
        assert np.isnan(r.ssr_std).all()


class TestSeventhReview:
    @staticmethod
    def _microseism_stream(seed=7):
        rng = np.random.default_rng(seed)
        sr, n = 100.0, 60000
        from scipy.signal import butter, sosfiltfilt

        sos = butter(4, (0.15, 0.3), btype="bandpass", fs=sr, output="sos")  # sharp skirts
        st = Stream()
        for ch in ("HHZ", "HHN", "HHE"):
            x = rng.standard_normal(n)
            micro = sosfiltfilt(sos, rng.standard_normal(n))
            x = x + 10 * micro / micro.std()
            st += Trace(
                x,
                header={
                    "network": "XX",
                    "station": "S1",
                    "channel": ch,
                    "sampling_rate": sr,
                    "starttime": UTCDateTime(0),
                },
            )
        return st

    def test_band_lets_microseism_records_through(self):
        st = self._microseism_stream()
        kw = dict(window_length=20.0, freq_min=0.5, freq_max=20.0)
        with pytest.raises(ValueError, match="every window was rejected"):
            compute_hvsr(st, anti_trigger=AntiTrigger(), **kw)
        r = compute_hvsr(st, anti_trigger=AntiTrigger(band=(1.0, 20.0)), **kw)
        assert r.n_windows > r.n_rejected

    def test_bad_band_raises(self):
        with pytest.raises(ValueError, match="band"):
            _hvsr(_stream(duration=120.0), anti_trigger=AntiTrigger(band=(1.0, 40.0)))

    def test_zero_filled_gap_is_a_gap(self):
        st = _stream(duration=300.0)
        for tr in st:
            tr.data[int(100 * SR) : int(130 * SR)] = 0.0  # merge(fill_value=0)
        r = _hvsr(st, window_length=20.0)
        assert r.n_rejected >= 4
        r0 = _hvsr(st, window_length=20.0, zero_gap=None)
        assert r0.n_rejected < r.n_rejected  # only windows wholly inside the zeros (dead) go

    def test_unused_duplicate_component_is_ignored(self):
        st = _stream(duration=120.0)
        for k in range(2):
            extra = st.select(channel="HHN")[0].copy()
            extra.stats.channel = "EH1"
            extra.stats.starttime += 1000 * k
            st += extra
        assert _hvsr(st).n_windows > 0

    def test_empty_channel_code_is_skipped(self, caplog):
        st = _stream(duration=120.0)
        st += Trace(np.zeros(100))
        assert _hvsr(st).n_windows > 0
        assert "Cannot classify" in caplog.text

    def test_non_overlapping_components_raise(self):
        st = _stream(duration=120.0)
        st.select(channel="HHZ")[0].stats.starttime += 1000.0
        with pytest.raises(ValueError, match="do not overlap"):
            _hvsr(st)

    def test_plot_omits_undefined_band(self):
        import matplotlib

        matplotlib.use("Agg")
        from swe.spectral import plot_hvsr

        r = compute_hvsr(_stream(duration=45.0), window_length=40.0, freq_min=0.2, freq_max=20.0)
        fig = plot_hvsr(r)
        labels = [
            t.get_text()
            for ax in fig.axes
            for leg in [ax.get_legend()]
            if leg
            for t in leg.get_texts()
        ]
        assert not any("\u03c3" in s for s in labels)


class TestEighthReview:
    def test_dc_offset_does_not_make_a_peak(self):
        st = _stream(gain=0.0, duration=600.0)
        clean = _hvsr(st)
        for tr, off in zip(st, (5000.0, -20000.0, 30000.0), strict=True):
            tr.data = tr.data + off
        shifted = _hvsr(st)
        np.testing.assert_allclose(shifted.hvsr_median, clean.hvsr_median, rtol=1e-6)

    def test_isolated_zero_counts_at_one_sps_are_data(self):
        rng = np.random.default_rng(11)
        st = Stream()
        for ch in ("LHZ", "LHN", "LHE"):
            x = np.round(rng.standard_normal(86400) * 3.0)  # integer counts: many exact zeros
            st += Trace(
                x,
                header={
                    "network": "XX",
                    "station": "S1",
                    "channel": ch,
                    "sampling_rate": 1.0,
                    "starttime": UTCDateTime(0),
                },
            )
        r = compute_hvsr(st, window_length=1000.0, freq_min=0.01, freq_max=0.4)
        assert r.n_rejected == 0

    def test_gap_error_mentions_zero_gap(self):
        st = _stream(duration=120.0)
        st.select(channel="HHE")[0].data[:] = 0.0
        with pytest.raises(ValueError, match="zero_gap=None"):
            _hvsr(st)


class TestNinthReview:
    def test_zero_runs_match_a_loop(self):
        from swe.spectral.ratios import _zero_runs

        rng = np.random.default_rng(13)
        x = np.round(rng.standard_normal(5000) * 0.7)
        x[1000:1050] = 0.0
        for run in (1, 3, 10, 40):
            ref = np.zeros(len(x), dtype=bool)
            i = 0
            while i < len(x):
                if x[i] == 0:
                    j = i
                    while j < len(x) and x[j] == 0:
                        j += 1
                    if j - i >= run:
                        ref[i:j] = True
                    i = j
                else:
                    i += 1
            np.testing.assert_array_equal(_zero_runs(x, run), ref)

    def test_band_edges_cost_only_their_windows(self):
        st = TestSeventhReview._microseism_stream(seed=9)
        at = AntiTrigger(band=(1.0, 20.0), max_ratio=3.0, min_ratio=0.1)
        r = compute_hvsr(st, window_length=20.0, freq_min=0.5, freq_max=20.0, anti_trigger=at)
        # by design: windows overlapping the first settle + lta (34 s) and the last settle (4 s)
        assert r.n_rejected <= 6


class TestTenthReview:
    def test_anti_trigger_uses_amplitude_ratio(self):
        # a stationary record whose amplitude STA/LTA stays inside tight limits keeps its windows
        # (an energy ratio would have squared the swings)
        r = _hvsr(
            _stream(gain=0.0, duration=600.0),
            anti_trigger=AntiTrigger(max_ratio=1.6, min_ratio=0.6),
        )
        assert r.n_windows > r.n_rejected

    def test_ssr_rate_mismatch_raises(self):
        st = _stream(duration=60.0)
        ref = st[0].copy()
        ref.stats.sampling_rate = 2 * SR
        with pytest.raises(ValueError, match="one sampling rate"):
            compute_ssr([st[1]], [ref])

    def test_ssr_events_of_different_length(self):
        st = _stream(duration=60.0)
        short = st[2].copy()
        short.data = short.data[:-1]
        r = compute_ssr([st[1].copy(), short], [st[0].copy(), st[0].copy()])
        assert r.ssr_per_event.shape == (2, len(r.frequencies))

    def test_ssr_masked_gap_raises(self):
        st = _stream(duration=60.0)
        tgt = st[1].copy()
        tgt.data = np.ma.masked_array(tgt.data, mask=np.zeros(tgt.data.size, dtype=bool))
        tgt.data.mask[100:200] = True
        with pytest.raises(ValueError, match="gap-free"):
            compute_ssr([tgt], [st[0]])


class TestEleventhReview:
    def test_ssr_mixed_rates_across_events_raise(self):
        st = _stream(duration=60.0)
        slow = [tr.copy() for tr in st]
        for tr in slow:
            tr.resample(SR / 2)
        with pytest.raises(ValueError, match="one sampling rate"):
            compute_ssr([st[1], slow[1]], [st[0], slow[0]])

    def test_freq_min_zero_drops_dc(self):
        r = compute_hvsr(_stream(duration=120.0), window_length=30.0, freq_min=0.0, freq_max=20.0)
        assert r.frequencies.min() > 0
        assert r.peak_frequency > 0


class TestTwelfthReview:
    def test_ssr_nan_sample_raises(self):
        st = _stream(duration=60.0)
        tgt = st[1].copy()
        tgt.data[500] = np.nan
        with pytest.raises(ValueError, match="masked or NaN"):
            compute_ssr([tgt], [st[0]])

    def test_railed_component_windows_are_rejected(self):
        st = _stream(duration=600.0)
        st.select(channel="HHZ")[0].data[int(200 * SR) : int(300 * SR)] = 8388607.0
        r = _hvsr(st, window_length=20.0)
        assert r.n_rejected >= 5
        assert not (r.hvsr_per_window == 0).all(axis=1).any()

    def test_ssr_does_not_depend_on_event_order(self):
        st = _stream(duration=60.0)
        long_, short = (st[1].copy(), st[0].copy()), (st[2].copy(), st[0].copy())
        short[0].data, short[1].data = short[0].data[:1500], short[1].data[:1500]
        a = compute_ssr([long_[0], short[0]], [long_[1], short[1]])
        b = compute_ssr([short[0], long_[0]], [short[1], long_[1]])
        np.testing.assert_allclose(a.frequencies, b.frequencies)
        np.testing.assert_allclose(a.ssr_mean, b.ssr_mean, rtol=1e-12)


class TestThirteenthReview:
    def test_stuck_float64_component_is_rejected(self):
        st = _stream(duration=600.0)
        for tr in st:  # integer counts, then remove_sensitivity-style float64 (same gain on all)
            tr.data = np.round(tr.data * 100).astype(np.int32)
        z = st.select(channel="HHZ")[0]
        z.data[int(300 * SR) :] = z.data[int(300 * SR)]  # the vertical sticks at its last value
        for tr in st:
            tr.data = tr.data / 3.7e8
        r = _hvsr(st, window_length=20.0)
        assert r.n_rejected >= 28
        assert np.isfinite(r.peak_amplitude) and r.peak_amplitude < 100


class TestFourteenthReview:
    def test_ssr_dead_reference_raises(self):
        st = _stream(duration=60.0)
        ref = st[0].copy()
        ref.data[:] = 0.0
        with pytest.raises(ValueError, match="never varies"):
            compute_ssr([st[1]], [ref])

    def test_ssr_zero_filled_gap_raises(self):
        st = _stream(duration=60.0)
        tgt = st[1].copy()
        tgt.data[1000:1150] = 0.0  # 3 s at 50 Hz
        with pytest.raises(ValueError, match="zero-filled gap"):
            compute_ssr([tgt], [st[0]])

    def test_short_dropout_is_a_gap(self):
        st = _stream(duration=300.0)
        for tr in st:
            tr.data[int(100 * SR) : int(100.5 * SR)] = 0.0  # 0.5 s packet loss
        assert _hvsr(st, window_length=20.0).n_rejected >= 1

    def test_gap_straddling_the_common_range_is_found(self):
        st = _stream(duration=300.0)
        z = st.select(channel="HHZ")[0]
        z.data[int(9 * SR) : int(10.6 * SR)] = 0.0  # 1.6 s gap at 9-10.6 s on Z
        for tr in st.select(channel="HH[NE]"):  # horizontals start at 10.5 s
            tr.data = tr.data[int(10.5 * SR) :]
            tr.stats.starttime += 10.5
        # inside the common range only 0.1 s (5 samples) of the gap remains, under the
        # 12-sample threshold; measured on the whole trace, it is still a gap
        r = _hvsr(st, window_length=20.0)
        assert r.n_rejected >= 1

    def test_components_from_different_sites_raise(self):
        st = _stream(duration=120.0)
        st.select(channel="HHZ")[0].stats.station = "S2"
        with pytest.raises(ValueError, match="different sites"):
            _hvsr(st)


class TestFifteenthReview:
    def test_ssr_checks_the_analysed_samples(self):
        st = _stream(duration=60.0)
        tgt = st[1].copy()
        tgt.data = tgt.data[:2000]
        ref = st[0].copy()
        ref.data[:2000] = 5.0  # flat over the analysed prefix, varying after it
        with pytest.raises(ValueError, match="never varies"):
            compute_ssr([tgt], [ref])

    def test_ssr_ignores_padding_beyond_the_analysed_samples(self):
        st = _stream(duration=60.0)
        tgt = st[1].copy()
        tgt.data = np.concatenate([tgt.data, np.zeros(1000)])  # trim(pad=True, fill_value=0)
        r = compute_ssr([tgt], [st[0]])
        assert r.n_events == 1

    def test_ssr_frequency_floor_is_the_shortest_event(self):
        st = _stream(duration=60.0)
        short_t, short_r = st[2].copy(), st[0].copy()
        short_t.data, short_r.data = short_t.data[:500], short_r.data[:500]  # 10 s at 50 Hz
        r = compute_ssr([st[1], short_t], [st[0], short_r], freq_min=0.01)
        assert r.frequencies.min() >= SR / 500 - 1e-12


def test_ssr_keeps_the_first_resolvable_bin():
    rng = np.random.default_rng(17)
    for sr, n in ((20.0, 121), (100.0, 3003)):  # rfftfreq's first bin falls one ulp below sr/n
        tr = [
            Trace(rng.standard_normal(n), header={"sampling_rate": sr, "channel": c})
            for c in ("HHZ", "HHN")
        ]
        r = compute_ssr([tr[1]], [tr[0]], freq_min=0.0, freq_max=sr / 2)
        assert r.frequencies[0] == pytest.approx(sr / n)


def test_sesame_criteria_accepts_a_cached_pipeline_output(tmp_path):
    from swe.pipeline.serialization import deserialize_output, serialize_output

    r = _hvsr(_stream(duration=300.0))
    serialize_output(tmp_path, "compute_hvsr", r)
    cached = deserialize_output(tmp_path, "compute_hvsr")
    assert isinstance(cached, HVSRResult)  # pickled, so a cache hit keeps the type
    a, b = sesame_criteria(r), sesame_criteria(cached)
    assert (a.f0, a.reliability, a.clear_peak) == (b.f0, b.reliability, b.clear_peak)
    assert sesame_criteria(r._asdict()).f0 == a.f0


def test_window_length_is_rounded_to_samples_not_truncated():
    w = 40.3  # 40.3 * 50 Hz is 2014.9999999999998 in floating point
    assert int(w * SR) == 2014
    assert _hvsr(_stream(duration=300.0), window_length=w).window_length == 2015 / SR
    # an np.float64 length (numpy 1.x round() of it is a float) still gives integer samples
    assert _hvsr(_stream(duration=300.0), window_length=np.float64(w)).window_length == 2015 / SR
    assert type(_samples(np.float64(2014.9999999999998))) is int


def test_site_label_carries_the_location_code():
    st = _stream(duration=300.0)
    assert _hvsr(st).station == "XX.S1"
    for tr in st:
        tr.stats.location = "10"
    assert _hvsr(st).station == "XX.S1.10"
    r = compute_ssr([st[1]], [st[0]])
    assert (r.target_station, r.reference_station) == ("XX.S1.10", "XX.S1.10")
    for tr in st:
        tr.stats.network = ""
    assert _hvsr(st).station == ".S1.10"  # an empty network keeps its place
