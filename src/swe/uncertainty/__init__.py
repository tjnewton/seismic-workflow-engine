"""Uncertainty propagation for seismic workflow results."""

from __future__ import annotations

from swe.uncertainty.conversions import (
    location_uncertainty_from_dict,
    location_uncertainty_to_dict,
    location_uncertainty_to_quantity_errors,
    magnitude_uncertainty_from_dict,
    magnitude_uncertainty_to_dict,
    quantity_error_to_uncertain_value,
    uncertain_value_from_dict,
    uncertain_value_to_dict,
)
from swe.uncertainty.magnitude import propagate_location_to_magnitude
from swe.uncertainty.plotting import (
    plot_location_uncertainty,
    plot_magnitude_pdf,
    plot_pick_uncertainty,
    plot_uncertainty_summary,
)
from swe.uncertainty.report import summarize_pipeline_uncertainty
from swe.uncertainty.propagation import (
    hypocenters_to_location_uncertainty,
    monte_carlo_location_uncertainty,
)
from swe.uncertainty.types import (
    LocationUncertainty,
    MagnitudeUncertainty,
    UncertainValue,
)

__all__ = [
    # types
    "UncertainValue",
    "LocationUncertainty",
    "MagnitudeUncertainty",
    # conversions
    "location_uncertainty_from_dict",
    "location_uncertainty_to_dict",
    "location_uncertainty_to_quantity_errors",
    "magnitude_uncertainty_from_dict",
    "magnitude_uncertainty_to_dict",
    "quantity_error_to_uncertain_value",
    "uncertain_value_from_dict",
    "uncertain_value_to_dict",
    # propagation
    "hypocenters_to_location_uncertainty",
    "monte_carlo_location_uncertainty",
    "propagate_location_to_magnitude",
    "summarize_pipeline_uncertainty",
    # plotting
    "plot_location_uncertainty",
    "plot_magnitude_pdf",
    "plot_pick_uncertainty",
    "plot_uncertainty_summary",
]
