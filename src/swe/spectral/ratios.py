"""Spectral ratio methods: HVSR and SSR.

Computes Horizontal-to-Vertical Spectral Ratios (HVSR) for site
characterisation and Standard Spectral Ratios (SSR) for relative
site response estimation.

References
----------
Nakamura, Y. (1989). A method for dynamic characteristics estimation
    of subsurface using microtremor on the ground surface. QR of RTRI,
    30(1), 25-33.
SESAME (2004). Guidelines for the implementation of the H/V spectral
    ratio technique on ambient vibrations. European Commission Project
    No. EVG1-CT-2000-00026.
Borcherdt, R.D. (1970). Effects of local geology on ground motion near
    San Francisco Bay. Bull. Seismol. Soc. Am., 60(1), 29-61.
Konno, K. & Ohmachi, T. (1998). Ground-motion characteristics estimated
    from spectral ratio between horizontal and vertical components of
    microtremor. Bull. Seismol. Soc. Am., 88(1), 228-241.

The SESAME (2004) reliability and clear-peak criteria (its section 2.1 and
Table 2) are evaluated by :func:`sesame_criteria` on the lognormal H/V curve:
the median ``exp(mean(ln H/V))`` and the factor ``sigma_A = exp(std(ln H/V))``
by which it is multiplied or divided. Windows with transients can be rejected
by the SESAME "anti-trigger" (:class:`AntiTrigger`).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from obspy import Stream
    from obspy.core.trace import Trace

logger = logging.getLogger(__name__)

__all__ = [
    "AntiTrigger",
    "HVSRResult",
    "SSRResult",
    "SesameCriteria",
    "compute_hvsr",
    "compute_ssr",
    "sesame_criteria",
    "sesame_thresholds",
]

SMOOTHING_TYPES = ("konno_ohmachi", "none")
_KO_BLOCK_ELEMENTS = 4_000_000  # weights per block (32 MB), bounds memory for long windows
_WINDOW_CHUNK = 256  # windows transformed at a time
_MIN_ZERO_RUN = (
    10  # samples: shorter runs of exact zeros are data, not gaps (low-rate integer counts)
)


class HVSRResult(NamedTuple):
    """Result from HVSR computation."""

    frequencies: NDArray[np.floating]
    """Frequency array (Hz)."""
    hvsr_mean: NDArray[np.floating]
    """Mean HVSR curve across windows."""
    hvsr_std: NDArray[np.floating]
    """Standard deviation of HVSR across windows (NaN with a single window)."""
    hvsr_per_window: NDArray[np.floating]
    """Individual HVSR curves, shape ``(n_windows, n_freqs)``."""
    peak_frequency: float
    """Frequency of the maximum of the arithmetic mean curve (:func:`sesame_criteria`
    uses the peak of the lognormal median instead)."""
    peak_amplitude: float
    """Maximum of the arithmetic mean curve."""
    n_windows: int
    """Number of time windows used."""
    station: str
    """``NET.STA``, or ``NET.STA.LOC`` when the location code is set."""
    hvsr_median: NDArray[np.floating] | None = None
    """Lognormal median H/V curve, ``exp(mean(ln H/V))`` across windows."""
    hvsr_log_std: NDArray[np.floating] | None = None
    """Standard deviation of ``ln H/V`` across windows (``sigma_A = exp`` of it)."""
    window_length: float | None = None
    """Window length (s) used."""
    n_rejected: int = 0
    """Number of windows rejected: those containing a gap (NaN, masked or zero-filled
    samples), those in which a component never varies (dead, railed or stuck), and, if
    enabled, those failing the anti-trigger."""


class AntiTrigger(NamedTuple):
    """SESAME (2004) anti-trigger window selection: keep windows free of transients.

    A window is kept only if the STA/LTA ratio of mean absolute amplitude stays
    within ``[min_ratio, max_ratio]`` at every sample, on every component. STA/LTA runs on
    each gap-free segment, and a segment's first ``lta`` seconds (at the record start
    and after every gap) are its warm-up: no transient check is possible there, so
    windows overlapping a warm-up are rejected. SESAME suggests ``sta`` of about
    0.5-2 s, ``lta`` of several tens of seconds and ``max_ratio`` of about 1.5-2.
    """

    sta: float = 1.0
    """Short-term average length (s)."""
    lta: float = 30.0
    """Long-term average length (s)."""
    max_ratio: float = 2.0
    """Largest STA/LTA allowed inside a window."""
    min_ratio: float = 0.2
    """Smallest STA/LTA allowed inside a window."""
    band: tuple[float, float] | None = None
    """Optional ``(f_low, f_high)`` band-pass (Hz, zero-phase Butterworth order 4) applied
    before STA/LTA. Raw broadband records dominated by microseisms or drift swing
    through the ratio limits unfiltered; a band above the microseism peak (for
    example ``(1.0, 20.0)``) checks the band the H/V analysis uses."""


class SesameCriteria(NamedTuple):
    """SESAME (2004) criteria for a reliable H/V curve and a clear H/V peak."""

    f0: float
    """Peak frequency of the lognormal median curve (Hz)."""
    a0: float
    """Median H/V amplitude at ``f0``."""
    sigma_f: float
    """Standard deviation of the per-window peak frequencies (Hz)."""
    sigma_a_f0: float
    """``sigma_A`` (multiplicative factor) at ``f0``."""
    n_cycles: float
    """Number of significant cycles ``lw * nw * f0``."""
    reliability: tuple[bool, bool, bool]
    """Reliability criteria i-iii."""
    clear_peak: tuple[bool, bool, bool, bool, bool, bool]
    """Clear-peak criteria i-vi."""

    @property
    def reliable(self) -> bool:
        """All three reliability criteria hold."""
        return all(self.reliability)

    @property
    def clear(self) -> bool:
        """At least five of the six clear-peak criteria hold."""
        return sum(self.clear_peak) >= 5


class SSRResult(NamedTuple):
    """Result from Standard Spectral Ratio computation."""

    frequencies: NDArray[np.floating]
    """Frequency array (Hz)."""
    ssr_mean: NDArray[np.floating]
    """Mean SSR across events."""
    ssr_std: NDArray[np.floating]
    """Standard deviation of SSR across events."""
    ssr_per_event: NDArray[np.floating]
    """Individual SSR curves, shape ``(n_events, n_freqs)``."""
    target_station: str
    """Target ``NET.STA`` (``NET.STA.LOC`` when the location code is set)."""
    reference_station: str
    """Reference ``NET.STA`` (``NET.STA.LOC`` when the location code is set)."""
    n_events: int
    """Number of events used."""


def _konno_ohmachi_smoothing(
    spectra: NDArray[np.floating],
    frequencies: NDArray[np.floating],
    *,
    bandwidth: float = 40.0,
) -> NDArray[np.floating]:
    """Apply Konno-Ohmachi (1998) logarithmic smoothing.

    Each output value is the window-weighted mean of the spectrum, with the
    window ``(sin(b log10(f/fc)) / (b log10(f/fc)))**4`` centred on the output
    frequency ``fc`` and normalised to unit sum. Vectorized: the windows are built
    in blocks of centre frequencies and applied to every row at once. Values at
    ``f <= 0`` are returned unchanged and take no part in the other windows.
    (ObsPy's matrix path for 2-D input applies the row-normalised windows
    transposed, so it is not used here.)

    Parameters
    ----------
    spectra : numpy.ndarray
        Amplitude spectrum, or a 2-D array of spectra (one per row).
    frequencies : numpy.ndarray
        Corresponding frequency array (Hz).
    bandwidth : float
        Smoothing bandwidth parameter (b). Higher values give less
        smoothing.

    Returns
    -------
    numpy.ndarray
        Smoothed spectra, same shape as ``spectra``.
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    out: NDArray[np.floating] = np.array(spectra, dtype=np.float64)
    pos = np.flatnonzero(freqs > 0)
    if pos.size == 0:
        return out
    f = freqs[pos]
    src = out[..., pos]  # fancy indexing already copies
    block = max(1, _KO_BLOCK_ELEMENTS // f.size)
    for c in range(0, f.size, block):
        with np.errstate(divide="ignore", invalid="ignore"):
            arg = bandwidth * np.log10(f[None, :] / f[c : c + block, None])
            w = np.where(np.abs(arg) < 1e-10, 1.0, (np.sin(arg) / arg) ** 4)
        w /= w.sum(axis=1, keepdims=True)
        out[..., pos[c : c + block]] = src @ w.T
    return out


def _check_smoothing(smoothing_type: str) -> None:
    if smoothing_type not in SMOOTHING_TYPES:
        raise ValueError(f"smoothing_type must be one of {SMOOTHING_TYPES}, got {smoothing_type!r}")


def _smooth(
    spectra: NDArray[np.floating],
    freqs: NDArray[np.floating],
    smoothing_type: str,
    bandwidth: float,
) -> NDArray[np.floating]:
    if smoothing_type == "konno_ohmachi":
        return _konno_ohmachi_smoothing(spectra, freqs, bandwidth=bandwidth)
    out: NDArray[np.floating] = np.asarray(spectra, dtype=np.float64)
    return out


def _lognormal_stats(
    hv: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Per-bin ``exp(mean ln)`` and ``std ln`` (ddof 1) over the rows with H/V > 0.

    NaN where fewer than one (median) or two (log std) rows are positive. Two passes
    over chunks of rows, so the temporaries stay at chunk size.
    """
    n_valid = np.zeros(hv.shape[1])
    s1 = np.zeros(hv.shape[1])
    for c in range(0, hv.shape[0], _WINDOW_CHUNK):
        blk = hv[c : c + _WINDOW_CHUNK]
        pos = blk > 0
        n_valid += pos.sum(axis=0)
        s1 += np.where(pos, np.log(np.where(pos, blk, 1.0)), 0.0).sum(axis=0)
    mean_log = np.divide(s1, n_valid, out=np.full(n_valid.shape, np.nan), where=n_valid > 0)
    ss = np.zeros(hv.shape[1])
    for c in range(0, hv.shape[0], _WINDOW_CHUNK):
        blk = hv[c : c + _WINDOW_CHUNK]
        pos = blk > 0
        ss += np.where(pos, (np.log(np.where(pos, blk, 1.0)) - mean_log) ** 2, 0.0).sum(axis=0)
    log_std = np.sqrt(
        np.divide(ss, n_valid - 1, out=np.full(n_valid.shape, np.nan), where=n_valid > 1)
    )
    return np.exp(mean_log), log_std


def _cosine_taper(n: int, fraction: float) -> NDArray[np.floating]:
    """Ones with raised-cosine ramps of ``max(1, int(fraction * n))`` samples at each end."""
    if not 0.0 <= fraction <= 0.5:
        raise ValueError(f"taper_fraction must be in [0, 0.5], got {fraction}")
    taper = np.ones(n)
    n_taper = max(1, int(fraction * n))
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))
    taper[:n_taper] = ramp
    taper[-n_taper:] = ramp[::-1]
    return taper


def _demeaned(x: NDArray[np.floating]) -> NDArray[np.floating]:
    out: NDArray[np.floating] = x - np.mean(x)
    return out


def _samples(x: float) -> int:
    """Nearest whole number of samples as a Python int (numpy 1.x ``round`` of np.float64 is a float)."""
    return int(np.rint(x))


def _site_label(tr: Trace) -> str:
    """``NET.STA``, or ``NET.STA.LOC`` when the location code is set (co-located sensors)."""
    s = tr.stats
    return f"{s.network}.{s.station}" + (f".{s.location}" if s.location else "")


def _zero_runs(x: NDArray[np.floating], run: int) -> NDArray[np.bool_]:
    """True on samples inside runs of at least ``run`` consecutive exact zeros."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (x == 0).astype(np.int8), [0]])))
    a, b = edges[::2], edges[1::2]
    keep = (b - a) >= run
    marks = np.zeros(len(x) + 1, dtype=np.int64)
    np.add.at(marks, a[keep], 1)
    np.add.at(marks, b[keep], -1)
    out: NDArray[np.bool_] = np.cumsum(marks[:-1]) > 0
    return out


def _windows_all(
    mask: NDArray[np.bool_], starts: list[int], window_samples: int
) -> NDArray[np.bool_]:
    """True for each window ``[st, st + window_samples)`` in which every sample of ``mask`` is True."""
    bad = np.concatenate([[0], np.cumsum(~mask)])
    st = np.asarray(starts)
    return np.asarray(bad[st + window_samples] - bad[st] == 0)


def _anti_trigger_mask(
    data: list[NDArray[np.floating]],
    starts: list[int],
    window_samples: int,
    sr: float,
    at: AntiTrigger,
) -> NDArray[np.bool_]:
    """True for windows whose STA/LTA stays inside the limits on every component.

    STA/LTA runs on each gap-free segment separately, demeaned, so a gap neither
    carries NaN forward nor drags the LTA down. Each segment's first ``lta``
    seconds are its warm-up: no transient check is possible there, so it counts as
    failing, whatever ``min_ratio`` is. With a band-pass, four periods of the low
    corner at each segment end are also unverifiable (filter edge transients).
    """
    from obspy.signal.trigger import classic_sta_lta
    from scipy.signal import butter, sosfiltfilt

    sos = None
    settle = 0
    if at.band is not None:
        lo, hi = at.band
        if not 0 < lo < hi < sr / 2:
            raise ValueError(
                f"anti-trigger band must satisfy 0 < low < high < Nyquist, got {at.band}"
            )
        sos = butter(4, (lo, hi), btype="bandpass", fs=sr, output="sos")
        settle = _samples(4.0 / lo * sr)  # four periods of the low corner
    nsta, nlta = _samples(at.sta * sr), _samples(at.lta * sr)
    if not (0 < nsta < nlta):
        raise ValueError(f"need 0 < sta < lta in samples, got sta={nsta}, lta={nlta}")
    if min(len(x) for x in data) <= nlta + 2 * settle:  # the per-segment test below
        raise ValueError(
            f"record shorter than the anti-trigger lta ({at.lta} s)"
            + (f" plus the band-pass settle zones (2 x {settle / sr:g} s)" if settle else "")
        )
    ok = np.ones(len(starts), dtype=bool)
    for x in data:
        good = np.zeros(len(x), dtype=bool)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], np.isfinite(x).astype(np.int8), [0]])))
        for a, b in zip(edges[::2], edges[1::2], strict=True):
            if b - a <= nlta + 2 * settle:
                continue  # too short to establish the LTA
            seg = x[a:b] - np.mean(x[a:b])  # an offset would swamp the energy
            if sos is not None:
                # the filter's edge transients are unverifiable: drop `settle` samples at each end
                # before STA/LTA, so they neither spike the ratio nor inflate the LTA
                seg = sosfiltfilt(sos, seg)[settle : len(seg) - settle]
            # ObsPy's STA/LTA averages squared samples; on sqrt(|x|) it becomes a ratio of mean
            # absolute amplitudes, the quantity SESAME's thresholds refer to
            ratio = classic_sta_lta(np.sqrt(np.abs(seg)), nsta, nlta)
            g = (ratio >= at.min_ratio) & (ratio <= at.max_ratio)
            g[:nlta] = False
            good[a + settle : b - settle] = g
        ok &= _windows_all(good, starts, window_samples)
    return ok


def _identify_components(stream: Stream) -> dict[str, list[Trace]]:
    """Group traces by orientation, from the last character of the channel code.

    Vertical (``Z`` or ``3``) goes under ``"Z"``; horizontals under ``"N"``, ``"E"``,
    ``"1"`` or ``"2"`` (case-insensitive). Other channels are skipped with a warning.
    """
    groups: dict[str, list[Trace]] = {}
    for tr in stream:
        key = tr.stats.channel[-1:].upper()  # an empty code gives "" and is skipped
        if key == "3":
            key = "Z"
        if key in ("Z", "N", "E", "1", "2"):
            groups.setdefault(key, []).append(tr)
        else:
            logger.warning("Cannot classify channel %s; skipping", tr.stats.channel)
    return groups


def compute_hvsr(
    stream: Stream,
    *,
    window_length: float = 40.0,
    overlap: float = 0.5,
    smoothing_type: str = "konno_ohmachi",
    smoothing_bandwidth: float = 40.0,
    freq_min: float = 0.1,
    freq_max: float = 50.0,
    merge_method: str = "geometric_mean",
    taper_fraction: float = 0.05,
    anti_trigger: AntiTrigger | Mapping[str, Any] | None = None,
    zero_gap: float | None = 0.25,
) -> HVSRResult:
    """Compute Horizontal-to-Vertical Spectral Ratio (HVSR).

    Divides the stream into overlapping time windows and computes
    the mean H/V spectral ratio with Konno-Ohmachi smoothing.

    Parameters
    ----------
    stream : obspy.Stream
        Three-component waveform data.
    window_length : float
        Length of each time window in seconds.
    overlap : float
        Fractional overlap between windows (0 to <1).
    smoothing_type : {"konno_ohmachi", "none"}
        Smoothing method.
    smoothing_bandwidth : float
        Bandwidth parameter for Konno-Ohmachi smoothing.
    freq_min : float
        Minimum frequency (Hz) for the output.
    freq_max : float
        Maximum frequency (Hz) for the output.
    merge_method : str
        Method to combine horizontal components: ``"geometric_mean"``
        or ``"quadratic_mean"``.
    taper_fraction : float
        Taper fraction applied to each window.
    anti_trigger : AntiTrigger or mapping, optional
        If given, windows with transients are rejected (SESAME anti-trigger).
        A mapping (for example from pipeline YAML) is converted with
        ``AntiTrigger(**anti_trigger)``. Each component is demeaned first.
    zero_gap : float or None
        Runs of exact zeros lasting at least this many seconds (and at least
        10 samples) count as gaps (zero-filled merges, including short packet-loss
        dropouts); runs are measured on the whole trace before trimming to the
        common time range. Windows containing gaps
        are rejected. ``None`` treats zeros as data. NaN and masked samples are
        always gaps. Zero-filled gaps are found only in unprocessed data:
        demeaning, filtering or response removal turns them into other values,
        so merge preprocessed data without ``fill_value`` (masked gaps) instead.
        Each window is demeaned before the taper, so a DC offset does not leak
        into the low frequencies.

    Returns
    -------
    HVSRResult
        HVSR results: the arithmetic mean and standard deviation, the
        lognormal median and log standard deviation, per-window curves, the
        peak of the mean curve, and the window bookkeeping.

    Raises
    ------
    ValueError
        If the stream lacks required components, windows are too short,
        ``overlap`` is outside ``[0, 1)``, the smoothing or merge method is
        unknown, the record is shorter than the anti-trigger LTA, or every
        window is rejected.
    """
    _check_smoothing(smoothing_type)
    if merge_method not in ("geometric_mean", "quadratic_mean"):
        raise ValueError(f"Unknown merge_method: {merge_method!r}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if isinstance(anti_trigger, Mapping):  # pipeline YAML passes plain mappings
        anti_trigger = AntiTrigger(**anti_trigger)  # values are checked below
    if anti_trigger is not None:
        if not isinstance(anti_trigger, AntiTrigger):
            raise TypeError(
                f"anti_trigger must be an AntiTrigger or a mapping, got {anti_trigger!r}"
            )
        if not 0.0 <= anti_trigger.min_ratio < anti_trigger.max_ratio:
            raise ValueError(f"need 0 <= min_ratio < max_ratio, got {anti_trigger}")
        if not 0.0 < anti_trigger.sta < anti_trigger.lta:
            raise ValueError(f"need 0 < sta < lta, got {anti_trigger}")
    if not 0.0 <= taper_fraction <= 0.5:
        raise ValueError(f"taper_fraction must be in [0, 0.5], got {taper_fraction}")
    if zero_gap is not None and not zero_gap > 0:
        raise ValueError(f"zero_gap must be positive or None, got {zero_gap}")
    groups = _identify_components(stream)
    n_horizontal = sum(len(v) for k, v in groups.items() if k != "Z")
    if n_horizontal < 2:
        raise ValueError(f"Need at least 2 horizontal components, got {n_horizontal}")
    if "Z" not in groups:
        raise ValueError("No vertical component found in stream")

    # One trace per component: an N/E or 1/2 horizontal pair and one vertical
    pair = next((p for p in (("N", "E"), ("1", "2")) if set(p) <= groups.keys()), None)
    used = ("Z", *pair) if pair is not None else tuple(groups)
    repeated = sorted(k for k in used if len(groups[k]) > 1)  # only the components used
    if repeated:
        raise ValueError(
            f"more than one trace for component(s) {repeated} (a gappy stream, or several band or"
            " location codes): select one, or merge the gaps (gappy windows are then rejected)"
        )
    if pair is None:
        horiz = sorted(k for k in groups if k != "Z")
        raise ValueError(f"need an N/E or 1/2 horizontal pair, got components {horiz}")
    vert_trace = groups["Z"][0]
    horiz_traces = [groups[pair[0]][0], groups[pair[1]][0]]
    station = _site_label(vert_trace)
    sites = {
        (t.stats.network, t.stats.station, t.stats.location) for t in (vert_trace, *horiz_traces)
    }
    if len(sites) > 1:
        raise ValueError(f"the three components come from different sites: {sorted(sites)}")

    sr = vert_trace.stats.sampling_rate
    if any(not math.isclose(tr.stats.sampling_rate, sr, rel_tol=1e-6) for tr in horiz_traces):
        raise ValueError("all three components must share one sampling rate")
    # round, not truncate: 40.3 s at 100 Hz is 4029.999... samples
    window_samples = _samples(window_length * sr)
    step_samples = _samples(window_samples * (1.0 - overlap))

    if window_samples < 4:
        raise ValueError(
            f"Window length {window_length}s is too short (need at least 4 samples at {sr} Hz)"
        )
    if step_samples < 1:
        raise ValueError(
            f"overlap {overlap} leaves no step between {window_samples}-sample windows"
        )

    # Find common time range
    t_start = max(tr.stats.starttime for tr in [vert_trace, *horiz_traces])
    t_end = min(tr.stats.endtime for tr in [vert_trace, *horiz_traces])
    if t_start >= t_end:
        raise ValueError("the three components do not overlap in time")

    # Trim all to common range
    def as_float(tr: Trace) -> NDArray[np.floating]:
        # gaps -> NaN on the whole trace (masked samples; zero runs, measured before trimming so
        # a gap straddling the common range keeps its full length), then trim to the common range
        full = np.ma.filled(np.ma.array(tr.data, dtype=np.float64, copy=True), np.nan)
        if zero_gap is not None:
            full[_zero_runs(full, max(_MIN_ZERO_RUN, _samples(zero_gap * sr)))] = np.nan
        i0 = round((t_start - tr.stats.starttime) * sr)
        return full[i0 : i0 + round((t_end - t_start) * sr) + 1]

    vert_data = as_float(vert_trace)
    horiz_data = [as_float(tr) for tr in horiz_traces]

    total_samples = min(len(vert_data), *(len(h) for h in horiz_data))

    # Generate window start indices
    starts: list[int] = []
    idx = 0
    while idx + window_samples <= total_samples:
        starts.append(idx)
        idx += step_samples

    if len(starts) == 0:
        raise ValueError(f"Trace too short for window_length={window_length}s")

    # Windows containing a gap on any component are rejected: NaN or masked samples, or
    # (merge(fill_value=0), as swe.io.fdsn does) a run of exact zeros of at least zero_gap seconds
    finite: NDArray[np.bool_] = np.ones(total_samples, dtype=bool)
    for x in (vert_data, *horiz_data):
        finite &= np.isfinite(x[:total_samples])
    gap_free = _windows_all(finite, starts, window_samples)
    n_rejected = int((~gap_free).sum())
    starts = [st for st, k in zip(starts, gap_free, strict=True) if k]
    if not starts:
        raise ValueError(
            "every window contains a gap (NaN or masked samples, or a run of exact zeros; "
            "set zero_gap=None if zeros are data)"
        )
    if anti_trigger is not None:
        keep = _anti_trigger_mask(
            [vert_data, *horiz_data], starts, window_samples, sr, anti_trigger
        )
        n_rejected += int((~keep).sum())
        starts = [st for st, k in zip(starts, keep, strict=True) if k]
        if not starts:
            settle = (
                f", and the band-pass settle zones of {4.0 / anti_trigger.band[0]:g} s at each end"
                if anti_trigger.band is not None
                else ""
            )
            raise ValueError(
                "every window was rejected by the anti-trigger (windows overlapping the first "
                f"lta = {anti_trigger.lta} s of the record or after a gap{settle} are always"
                " rejected)"
            )

    # Frequency array for the FFT
    freqs = np.fft.rfftfreq(window_samples, d=1.0 / sr)
    freq_mask = (freqs > 0) & (freqs >= freq_min) & (freqs <= freq_max)  # no DC bin
    out_freqs = freqs[freq_mask]
    if out_freqs.size == 0:
        raise ValueError(f"no FFT frequency lies in [{freq_min}, {freq_max}] Hz")

    taper = _cosine_taper(window_samples, taper_fraction)
    n_windows = len(starts)
    hvsr_array = np.empty((n_windows, out_freqs.size))
    alive = np.ones(n_windows, dtype=bool)
    # Windows in chunks, end to end (FFT, one smoothing call for V, H1, H2, H/V): memory stays
    # at a chunk plus the per-window result.
    for c in range(0, n_windows, _WINDOW_CHUNK):
        chunk = starts[c : c + _WINDOW_CHUNK]
        raw = [
            np.stack([x[st : st + window_samples] for st in chunk])
            for x in (vert_data, *horiz_data)
        ]
        # a component that never varies in a window (dead, railed, stuck) has no H/V; tested on the
        # raw samples, since demeaning a float64 constant leaves a rounding residual
        alive[c : c + len(chunk)] = np.all([np.ptp(w, axis=1) > 0 for w in raw], axis=0)
        amp = np.concatenate(
            [
                np.abs(np.fft.rfft((w - w.mean(axis=1, keepdims=True)) * taper, axis=1))[
                    :, freq_mask
                ]
                for w in raw
            ]
        )
        v_spec, h1, h2 = np.split(_smooth(amp, out_freqs, smoothing_type, smoothing_bandwidth), 3)
        if merge_method == "geometric_mean":
            h_combined = np.sqrt(h1 * h2)
        else:
            h_combined = np.sqrt((h1**2 + h2**2) / 2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            hvsr_array[c : c + len(chunk)] = np.where(v_spec > 0, h_combined / v_spec, 0.0)
        # a component with an identically zero spectrum (dead, railed or constant) has no H/V

    n_rejected += int((~alive).sum())
    if not alive.any():
        raise ValueError("every window has a dead, railed or constant component")
    hvsr_array = hvsr_array[alive]
    n_windows = hvsr_array.shape[0]
    hvsr_mean = np.mean(hvsr_array, axis=0)
    hvsr_std = (
        np.std(hvsr_array, axis=0, ddof=1) if n_windows > 1 else np.full_like(hvsr_mean, np.nan)
    )
    # Lognormal statistics over the windows with H/V > 0 in each bin; NaN where fewer than one
    # (median) or two (log std) such windows exist. Computed by hand: no warnings filter needed.
    hvsr_median, hvsr_log_std = _lognormal_stats(hvsr_array)

    peak_idx = int(np.argmax(hvsr_mean))
    peak_frequency = float(out_freqs[peak_idx])
    peak_amplitude = float(hvsr_mean[peak_idx])

    return HVSRResult(
        frequencies=out_freqs,
        hvsr_mean=hvsr_mean,
        hvsr_std=hvsr_std,
        hvsr_per_window=hvsr_array,
        peak_frequency=peak_frequency,
        peak_amplitude=peak_amplitude,
        n_windows=n_windows,
        station=station,
        hvsr_median=hvsr_median,
        hvsr_log_std=hvsr_log_std,
        window_length=window_samples / sr,  # the length used, after rounding to samples
        n_rejected=n_rejected,
    )


def compute_ssr(
    target_traces: list[Trace],
    reference_traces: list[Trace],
    *,
    smoothing_type: str = "konno_ohmachi",
    smoothing_bandwidth: float = 40.0,
    freq_min: float = 0.1,
    freq_max: float = 50.0,
    water_level: float = 1e-10,
    taper_fraction: float = 0.05,
    zero_gap: float | None = 0.25,
) -> SSRResult:
    """Compute Standard Spectral Ratios (SSR).

    Computes the spectral ratio between target and reference station
    recordings for each event, then stacks across events.

    Parameters
    ----------
    target_traces : list of Trace
        Target station waveforms, one per event.
    reference_traces : list of Trace
        Reference station waveforms, one per event (paired with target).
    smoothing_type : {"konno_ohmachi", "none"}
        Smoothing method.
    smoothing_bandwidth : float
        Bandwidth parameter for smoothing.
    freq_min : float
        Minimum frequency (Hz).
    freq_max : float
        Maximum frequency (Hz).
    water_level : float
        Small value added to the reference spectrum denominator to
        prevent division by zero.
    taper_fraction : float
        Taper fraction applied before FFT.
    zero_gap : float or None
        As in :func:`compute_hvsr`: a run of exact zeros of at least this many
        seconds (and 10 samples) marks a zero-filled gap, and the call raises.
        ``None`` treats zeros as data: pass it for synthetics that are exactly zero
        before the first arrival. A trace that never varies also raises.

    Returns
    -------
    SSRResult
        SSR results including mean, standard deviation, and per-event
        ratios.

    Raises
    ------
    ValueError
        If input lists are empty or have mismatched lengths; the smoothing
        method is unknown; traces do not share one sampling rate; an analysed
        trace has masked or NaN samples, never varies, or holds a zero-filled
        gap; or no FFT frequency lies in the band.
    """
    _check_smoothing(smoothing_type)
    if len(target_traces) == 0:
        raise ValueError("target_traces must not be empty")
    if len(target_traces) != len(reference_traces):
        raise ValueError(
            f"target_traces ({len(target_traces)}) and reference_traces "
            f"({len(reference_traces)}) must have the same length"
        )

    target_station = _site_label(target_traces[0])
    reference_station = _site_label(reference_traces[0])

    sr = target_traces[0].stats.sampling_rate
    # each pair is analysed over its first npts = min(len) samples, so the checks look at those
    lengths = [
        min(len(t.data), len(r.data)) for t, r in zip(target_traces, reference_traces, strict=False)
    ]
    for (tgt, ref), npts in zip(
        zip(target_traces, reference_traces, strict=False), lengths, strict=True
    ):
        for tr in (tgt, ref):
            if not math.isclose(tr.stats.sampling_rate, sr, rel_tol=1e-6):
                raise ValueError(
                    f"every trace must share one sampling rate ({sr} Hz, got {tr.stats.sampling_rate} Hz)"
                )
            d = tr.data[:npts]
            if np.ma.is_masked(d) or not np.isfinite(d).all():
                raise ValueError(
                    "SSR needs gap-free event windows; a trace has masked or NaN samples"
                )
            if np.ptp(d) == 0:
                raise ValueError(f"trace {tr.id} never varies (dead or stuck channel)")
            if (
                zero_gap is not None
                and _zero_runs(
                    np.asarray(d, dtype=np.float64), max(_MIN_ZERO_RUN, _samples(zero_gap * sr))
                ).any()
            ):
                raise ValueError(
                    f"trace {tr.id} has a zero-filled gap (set zero_gap=None if zeros are data)"
                )
    # One FFT length for every event (the longest pair, shorter ones zero-padded), so the grid
    # does not depend on event order; each event's own length is used for its taper. Bins below
    # the shortest event's resolution (1 / its duration) are dropped: padding cannot resolve them.
    nfft = max(lengths)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    f_floor = sr / min(lengths)
    # bin k is at k * sr / nfft; compare on indices (k * min_len >= nfft), not floats
    resolved = np.arange(freqs.size) * min(lengths) >= nfft
    freq_mask = resolved & (freqs >= freq_min) & (freqs <= freq_max)  # no DC bin
    out_freqs = freqs[freq_mask]
    if out_freqs.size == 0:
        raise ValueError(
            f"no FFT frequency lies in [{max(freq_min, f_floor)}, {freq_max}] Hz "
            "(the lower bound includes the shortest event's resolution)"
        )

    ssr_events: list[NDArray] = []
    for (tgt, ref), npts in zip(
        zip(target_traces, reference_traces, strict=False), lengths, strict=True
    ):
        taper = _cosine_taper(npts, taper_fraction)
        tgt_data = _demeaned(tgt.data[:npts].astype(np.float64)) * taper  # no DC leakage
        ref_data = _demeaned(ref.data[:npts].astype(np.float64)) * taper
        tgt_s = np.abs(np.fft.rfft(tgt_data, n=nfft))[freq_mask]
        ref_s = np.abs(np.fft.rfft(ref_data, n=nfft))[freq_mask]
        tgt_s, ref_s = _smooth(
            np.stack([tgt_s, ref_s]), out_freqs, smoothing_type, smoothing_bandwidth
        )
        ssr_events.append(tgt_s / (ref_s + water_level))

    ssr_array = np.array(ssr_events)
    ssr_mean = np.mean(ssr_array, axis=0)
    ssr_std = (  # undefined (NaN) for a single event, as HVSR's single-window std
        np.std(ssr_array, axis=0, ddof=1) if len(ssr_events) > 1 else np.full_like(ssr_mean, np.nan)
    )

    return SSRResult(
        frequencies=out_freqs,
        ssr_mean=ssr_mean,
        ssr_std=ssr_std,
        ssr_per_event=ssr_array,
        target_station=target_station,
        reference_station=reference_station,
        n_events=len(ssr_events),
    )


# SESAME (2004) Table 2: (upper frequency bound in Hz, epsilon as a fraction of f0, theta)
_SESAME_TABLE = (
    (0.2, 0.25, 3.0),
    (0.5, 0.20, 2.5),
    (1.0, 0.15, 2.0),
    (2.0, 0.10, 1.78),
    (np.inf, 0.05, 1.58),
)


def sesame_thresholds(f0: float) -> tuple[float, float]:
    """SESAME (2004) stability thresholds ``(epsilon(f0) in Hz, theta(f0))``.

    Parameters
    ----------
    f0 : float
        H/V peak frequency (Hz). Each frequency range is closed below and open
        above: ``[0.2, 0.5)``, ``[0.5, 1.0)``, ``[1.0, 2.0)``, ``[2.0, inf)``.

    Returns
    -------
    tuple of float
        ``epsilon`` (Hz) for the condition ``sigma_f < epsilon`` and ``theta``
        for ``sigma_A(f0) < theta``.

    Raises
    ------
    ValueError
        If ``f0`` is not positive and finite.
    """
    if not (np.isfinite(f0) and f0 > 0):
        raise ValueError(f"f0 must be positive and finite, got {f0}")
    for upper, eps, theta in _SESAME_TABLE:
        if f0 < upper:
            return eps * f0, theta
    raise AssertionError("unreachable")  # pragma: no cover


def sesame_criteria(
    result: HVSRResult | Mapping[str, Any] | Sequence[Any],
    *,
    search_range: tuple[float, float] | None = None,
) -> SesameCriteria:
    """Evaluate the SESAME (2004) reliability and clear-peak criteria.

    Uses the lognormal median curve ``A(f)`` and ``sigma_A(f) = exp(std ln H/V)``
    of :func:`compute_hvsr`. ``f0`` is the peak of ``A`` within ``search_range``
    (positive frequencies where ``A`` and ``sigma_A`` are finite);
    ``sigma_f`` is the standard deviation of the per-window peak frequencies in
    the same range, over windows whose curve is not identically zero there (they
    alone count as ``nw``; at least two are required). SESAME also recommends at
    least 10 windows (its Table 1); that is a recording guideline, not one of
    the criteria, and is not checked.

    Reliability (all three must hold): (i) ``f0 > 10 / lw``; (ii)
    ``nc = lw * nw * f0 > 200``; (iii) ``sigma_A(f) < 2`` for
    ``0.5 f0 < f < 2 f0`` if ``f0 > 0.5`` Hz, else ``< 3``.

    Clear peak (at least five of six): (i) some ``f`` in ``[f0/4, f0]`` has
    ``A(f) < A0/2``; (ii) likewise in ``[f0, 4 f0]``; (iii) ``A0 > 2``; (iv) the
    peaks of ``A * sigma_A`` and ``A / sigma_A`` lie within 5% of ``f0``; (v)
    ``sigma_f < epsilon(f0)``; (vi) ``sigma_A(f0) < theta(f0)``
    (:func:`sesame_thresholds`). Criteria (i) and (ii) see only the part of their
    interval inside the computed frequency band.

    Parameters
    ----------
    result : HVSRResult, mapping or sequence
        Output of :func:`compute_hvsr`, or its fields as a mapping or in field
        order (the form a cached pipeline step returns it in).
    search_range : tuple of float, optional
        ``(f_min, f_max)`` (Hz) in which to look for the peak; default, the
        whole band. It also bounds the per-window and ``A * sigma_A`` peak
        searches, so a range narrower than the peak makes criteria (iv) and (v)
        trivially true.

    Returns
    -------
    SesameCriteria
        The quantities and the criteria, with ``reliable`` and ``clear``.

    Raises
    ------
    ValueError
        If ``result`` lacks the lognormal curves, or ``search_range`` selects
        no frequency.
    """
    if isinstance(result, Mapping):  # a pipeline step's cached output comes back as JSON
        result = HVSRResult(**result)
    elif not isinstance(result, HVSRResult):
        result = HVSRResult(*result)
    if result.hvsr_median is None or result.hvsr_log_std is None or result.window_length is None:
        raise ValueError("result has no lognormal curves; compute it with compute_hvsr")
    f = np.asarray(result.frequencies, dtype=float)
    amp = np.asarray(result.hvsr_median, dtype=float)
    sig_a = np.exp(np.asarray(result.hvsr_log_std, dtype=float))
    if not np.isfinite(amp).any():
        raise ValueError(
            "no frequency has a finite median H/V: is a component dead or zero-filled?"
        )
    positive = np.asarray(result.hvsr_per_window) > 0
    n_live = int(positive.any(axis=1).sum())
    if n_live < 2:
        raise ValueError(f"the SESAME spreads need at least 2 windows with H/V > 0, got {n_live}")
    lo, hi = search_range if search_range is not None else (-np.inf, np.inf)
    sel = np.flatnonzero((f >= lo) & (f <= hi) & (f > 0) & np.isfinite(amp) & np.isfinite(sig_a))
    if sel.size == 0:
        raise ValueError(f"search_range {search_range} selects no frequency")
    i0 = sel[np.argmax(amp[sel])]
    f0, a0 = float(f[i0]), float(amp[i0])
    curves = np.asarray(result.hvsr_per_window, dtype=float)[:, sel]
    live = positive[:, sel].any(axis=1)  # a zero curve (dead or zero-filled window) has no peak
    peaks = f[sel][np.argmax(curves[live], axis=1)]
    sigma_f = float(np.std(peaks, ddof=1))  # >= 2 live windows: sel needs a finite sigma_A
    lw, nw = float(result.window_length), int(live.sum())
    n_cycles = lw * nw * f0

    band = (f > 0.5 * f0) & (f < 2.0 * f0)
    sig_band = sig_a[band][np.isfinite(sig_a[band])]  # NaN bins are skipped, as in the f0 search
    limit = 2.0 if f0 > 0.5 else 3.0
    # f0's own bin is in the band and finite (the f0 search requires it), so sig_band is never empty
    reliability = (bool(f0 > 10.0 / lw), bool(n_cycles > 200.0), bool(np.all(sig_band < limit)))

    eps, theta = sesame_thresholds(f0)
    f_up = f[sel][np.argmax((amp * sig_a)[sel])]
    f_lo = f[sel][np.argmax((amp / sig_a)[sel])]
    clear_peak = (
        bool(np.any(amp[(f >= f0 / 4) & (f <= f0)] < a0 / 2)),
        bool(np.any(amp[(f >= f0) & (f <= 4 * f0)] < a0 / 2)),
        bool(a0 > 2.0),
        bool(abs(f_up / f0 - 1) <= 0.05 and abs(f_lo / f0 - 1) <= 0.05),
        bool(sigma_f < eps),
        bool(sig_a[i0] < theta),
    )
    return SesameCriteria(
        f0=f0,
        a0=a0,
        sigma_f=sigma_f,
        sigma_a_f0=float(sig_a[i0]),
        n_cycles=float(n_cycles),
        reliability=reliability,
        clear_peak=clear_peak,
    )
