"""Magnitude estimation for seismic events."""

from __future__ import annotations

from swe.magnitude.catalog import (
    attach_duration_magnitudes,
    attach_local_magnitudes,
    attach_magnitudes_to_catalog,
)
from swe.magnitude.duration import (
    DurationMagnitudeBatchResult,
    DurationMagnitudeResult,
    coda_duration,
    duration_magnitude,
    duration_magnitudes_batch,
)
from swe.magnitude.local import (
    LocalMagnitudeResult,
    local_magnitude,
    wood_anderson_amplitude,
)
from swe.magnitude.relative import (
    RelativeMomentResult,
    relative_moment_dataframe,
    relative_moments_from_party,
    relative_moments_from_streams,
)

__all__ = [
    # duration
    "DurationMagnitudeBatchResult",
    "DurationMagnitudeResult",
    # local
    "LocalMagnitudeResult",
    # relative
    "RelativeMomentResult",
    "attach_duration_magnitudes",
    # catalog integration
    "attach_local_magnitudes",
    "attach_magnitudes_to_catalog",
    "coda_duration",
    "duration_magnitude",
    "duration_magnitudes_batch",
    "local_magnitude",
    "relative_moment_dataframe",
    "relative_moments_from_party",
    "relative_moments_from_streams",
    "wood_anderson_amplitude",
]
