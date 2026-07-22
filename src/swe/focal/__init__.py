"""Focal mechanism determination from P-wave first-motion polarities.

Provides first-motion polarity measurement, takeoff angle computation,
and grid-search focal mechanism solving.  Results feed directly into
:func:`swe.stress.stress_inversion`.

See also :mod:`swe.moment_tensor` for full moment tensor inversion
(ISO + DC + CLVD decomposition) using the same polarity observations.
"""

from __future__ import annotations

from swe.focal.polarity import (
    PolarityObservation,
    PolarityResult,
    measure_polarities,
    measure_polarity,
)
from swe.focal.solve import (
    FocalMechanism,
    FocalMechanismResult,
    focal_mechanism_gridsearch,
    focal_mechanism_uncertainty,
    focal_mechanisms_batch,
    polarity_result_to_event_dict,
)
from swe.focal.takeoff import (
    RayParameters,
    compute_takeoff_angles,
    compute_takeoff_angles_1d,
)

__all__ = [
    # solve
    "FocalMechanism",
    "FocalMechanismResult",
    # polarity
    "PolarityObservation",
    "PolarityResult",
    # takeoff
    "RayParameters",
    "compute_takeoff_angles",
    "compute_takeoff_angles_1d",
    "focal_mechanism_gridsearch",
    "focal_mechanism_uncertainty",
    "focal_mechanisms_batch",
    "measure_polarities",
    "measure_polarity",
    "polarity_result_to_event_dict",
]
