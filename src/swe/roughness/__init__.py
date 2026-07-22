"""Roughness-frequency scaling analysis.

Tools for characterising surface roughness, tracking temporal
frequency evolution, testing the tribological scaling law
f = V_slip / lambda_dominant, and mapping spatial correlations
between roughness and seismic source frequency.
"""

from __future__ import annotations

from swe.roughness._plotting import (
    plot_forward_model_comparison,
    plot_frequency_roughness_correlation,
    plot_frequency_time_series,
    plot_roughness_psd,
    plot_scaling_law_fit,
    plot_spatial_frequency_map,
)
from swe.roughness.forward import (
    ForwardModelResult,
    forward_model_frequency,
    resonance_frequency,
)
from swe.roughness.frequency import (
    FrequencyTimeSeriesResult,
    ScalingLawFitResult,
    compute_frequency_time_series,
    fit_scaling_law,
    infer_roughness_wavelength,
)
from swe.roughness.spatial import (
    SpatialCorrelationResult,
    compute_spatial_frequency_map,
    correlate_frequency_roughness,
)
from swe.roughness.surface import (
    RoughnessPSDResult,
    compute_roughness_psd_1d,
    compute_roughness_psd_2d,
    dominant_roughness_wavelength,
)

__all__ = [
    # forward
    "ForwardModelResult",
    # frequency
    "FrequencyTimeSeriesResult",
    # surface
    "RoughnessPSDResult",
    "ScalingLawFitResult",
    # spatial
    "SpatialCorrelationResult",
    "compute_frequency_time_series",
    "compute_roughness_psd_1d",
    "compute_roughness_psd_2d",
    "compute_spatial_frequency_map",
    "correlate_frequency_roughness",
    "dominant_roughness_wavelength",
    "fit_scaling_law",
    "forward_model_frequency",
    "infer_roughness_wavelength",
    # plotting
    "plot_forward_model_comparison",
    "plot_frequency_roughness_correlation",
    "plot_frequency_time_series",
    "plot_roughness_psd",
    "plot_scaling_law_fit",
    "plot_spatial_frequency_map",
    "resonance_frequency",
]
