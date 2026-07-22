"""Spectral analysis for seismic waveforms."""

from __future__ import annotations

from swe.spectral._plotting import (
    plot_brune_fit,
    plot_coda_decay,
    plot_coda_q_vs_frequency,
    plot_hvsr,
    plot_ssr,
    plot_stress_drop_summary,
)
from swe.spectral.coda import (
    CodaQFrequencyResult,
    CodaQResult,
    coda_q_frequency_bands,
    estimate_coda_q,
)
from swe.spectral.fitting import (
    SpectralFitResult,
    StressDropResult,
    brune_spectrum,
    compute_displacement_spectrum,
    compute_stress_drop,
    fit_brune_model,
    spectral_fit_batch,
)
from swe.spectral.ratios import (
    HVSRResult,
    SSRResult,
    compute_hvsr,
    compute_ssr,
)
from swe.spectral.source_params import (
    FalloffResult,
    MomentResult,
    SourceDurationResult,
    high_frequency_falloff,
    moment_from_spectral_level,
    source_time_function_width,
)

__all__ = [
    # coda
    "CodaQFrequencyResult",
    "CodaQResult",
    # source parameters (LFE-robust)
    "FalloffResult",
    # ratios
    "HVSRResult",
    "MomentResult",
    "SSRResult",
    "SourceDurationResult",
    # fitting
    "SpectralFitResult",
    "StressDropResult",
    "brune_spectrum",
    "coda_q_frequency_bands",
    "compute_displacement_spectrum",
    "compute_hvsr",
    "compute_ssr",
    "compute_stress_drop",
    "estimate_coda_q",
    "fit_brune_model",
    "high_frequency_falloff",
    "moment_from_spectral_level",
    # plotting
    "plot_brune_fit",
    "plot_coda_decay",
    "plot_coda_q_vs_frequency",
    "plot_hvsr",
    "plot_ssr",
    "plot_stress_drop_summary",
    "source_time_function_width",
    "spectral_fit_batch",
]
