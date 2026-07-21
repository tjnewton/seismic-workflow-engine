"""Tests for swe.spectral._plotting."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure
from obspy import UTCDateTime

from swe.spectral._plotting import (
    plot_brune_fit,
    plot_coda_decay,
    plot_coda_q_vs_frequency,
    plot_hvsr,
    plot_ssr,
    plot_stress_drop_summary,
)
from swe.spectral.coda import estimate_coda_q, coda_q_frequency_bands
from swe.spectral.fitting import fit_brune_model, compute_stress_drop
from swe.spectral.ratios import compute_hvsr, compute_ssr


@pytest.fixture()
def hvsr_result(three_component_stream):
    return compute_hvsr(three_component_stream, window_length=20.0)


@pytest.fixture()
def ssr_result(event_trace_pair):
    target, reference = event_trace_pair
    return compute_ssr([target], [reference])


@pytest.fixture()
def coda_result(coda_trace):
    origin = coda_trace.stats.starttime + 5.0
    return estimate_coda_q(coda_trace, origin, freq_min=1.0, freq_max=10.0)


@pytest.fixture()
def coda_freq_result(coda_trace):
    origin = coda_trace.stats.starttime + 5.0
    return coda_q_frequency_bands(coda_trace, origin)


@pytest.fixture()
def brune_fit_result(brune_trace):
    return fit_brune_model(brune_trace, freq_min=0.5, freq_max=45.0)


@pytest.fixture()
def stress_drop_results(brune_trace):
    fit = fit_brune_model(brune_trace, freq_min=0.5, freq_max=45.0)
    return [
        compute_stress_drop(fit, magnitude=m) for m in [1.0, 2.0, 3.0]
    ]


class TestPlotHVSR:
    def test_returns_figure(self, hvsr_result):
        fig = plot_hvsr(hvsr_result)
        assert isinstance(fig, Figure)

    def test_no_individual_windows(self, hvsr_result):
        fig = plot_hvsr(hvsr_result, show_individual=False)
        assert isinstance(fig, Figure)


class TestPlotSSR:
    def test_returns_figure(self, ssr_result):
        fig = plot_ssr(ssr_result)
        assert isinstance(fig, Figure)

    def test_no_individual_events(self, ssr_result):
        fig = plot_ssr(ssr_result, show_individual=False)
        assert isinstance(fig, Figure)


class TestPlotCodaDecay:
    def test_returns_figure(self, coda_trace, coda_result):
        origin = coda_trace.stats.starttime + 5.0
        fig = plot_coda_decay(coda_trace, coda_result, origin)
        assert isinstance(fig, Figure)


class TestPlotCodaQVsFrequency:
    def test_returns_figure(self, coda_freq_result):
        fig = plot_coda_q_vs_frequency(coda_freq_result)
        assert isinstance(fig, Figure)


class TestPlotBruneFit:
    def test_returns_figure(self, brune_fit_result):
        fig = plot_brune_fit(brune_fit_result)
        assert isinstance(fig, Figure)


class TestPlotStressDropSummary:
    def test_returns_figure(self, stress_drop_results):
        fig = plot_stress_drop_summary(stress_drop_results)
        assert isinstance(fig, Figure)
