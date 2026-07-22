"""Moment tensor inversion and decomposition.

Extends the focal mechanism pipeline in :mod:`swe.focal` to full moment
tensor representation.  Provides polarity-based and amplitude-based
grid-search inversion, ISO/DC/CLVD decomposition, and moment magnitude
computation.
"""

from __future__ import annotations

from swe.moment_tensor.amplitude_inversion import (
    AmplitudeMTResult,
    moment_tensor_from_amplitude_ratios,
    moment_tensor_from_amplitudes,
    moment_tensor_from_s_p_ratios,
)
from swe.moment_tensor.amplitudes import (
    ThreeComponentAmplitudes,
    three_component_amplitudes,
)
from swe.moment_tensor.inversion import (
    MomentTensorResult,
    moment_tensor_batch,
    moment_tensor_from_polarities,
    moment_tensor_uncertainty,
)
from swe.moment_tensor.tensor import (
    MomentTensor,
    MomentTensorDecomposition,
    decompose_moment_tensor,
    from_matrix,
    moment_magnitude,
    scalar_moment,
    strike_dip_rake_to_tensor,
    tensor_to_strike_dip_rake,
    to_matrix,
)

__all__ = [
    "AmplitudeMTResult",
    "MomentTensor",
    "MomentTensorDecomposition",
    "MomentTensorResult",
    "ThreeComponentAmplitudes",
    "decompose_moment_tensor",
    "from_matrix",
    "moment_magnitude",
    "moment_tensor_batch",
    "moment_tensor_from_amplitude_ratios",
    "moment_tensor_from_amplitudes",
    "moment_tensor_from_polarities",
    "moment_tensor_from_s_p_ratios",
    "moment_tensor_uncertainty",
    "scalar_moment",
    "strike_dip_rake_to_tensor",
    "tensor_to_strike_dip_rake",
    "three_component_amplitudes",
    "to_matrix",
]
