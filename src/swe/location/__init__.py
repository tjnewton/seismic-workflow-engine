"""Earthquake hypocenter location methods."""

from __future__ import annotations

from swe.location.amplitude import (
    AmplitudeLocationResult,
    LocationResult,
    SyntheticEventResult,
    amplocate,
    locate_by_travel_time,
    locate_by_travel_time_with_uncertainty,
    synthetic_event,
)
from swe.location.nonlinloc import (
    HypocenterResult,
    NLLLocationResult,
    catalog_to_obs,
    extract_nll_locations,
    hypocenter_to_obspy_origin,
    picks_to_nonlinloc,
    process_nll_hypocenters,
    run_nonlinloc,
)
from swe.location.travel_time import locate_by_travel_time_grid

__all__ = [
    "AmplitudeLocationResult",
    "HypocenterResult",
    "LocationResult",
    "NLLLocationResult",
    "SyntheticEventResult",
    "amplocate",
    "catalog_to_obs",
    "extract_nll_locations",
    "hypocenter_to_obspy_origin",
    "locate_by_travel_time",
    "locate_by_travel_time_grid",
    "locate_by_travel_time_with_uncertainty",
    "picks_to_nonlinloc",
    "process_nll_hypocenters",
    "run_nonlinloc",
    "synthetic_event",
]
