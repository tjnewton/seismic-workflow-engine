"""Shared fixtures for moment tensor tests."""

from __future__ import annotations

import numpy as np
import pytest

from swe.moment_tensor.tensor import MomentTensor, strike_dip_rake_to_tensor


@pytest.fixture()
def pure_dc_mt() -> MomentTensor:
    """A pure double-couple moment tensor (strike=30, dip=60, rake=90)."""
    return strike_dip_rake_to_tensor(30.0, 60.0, 90.0, m0=1e16)


@pytest.fixture()
def isotropic_mt() -> MomentTensor:
    """A pure isotropic (explosive) moment tensor."""
    return MomentTensor(
        mxx=1.0, myy=1.0, mzz=1.0, mxy=0.0, mxz=0.0, myz=0.0,
        scalar_moment=np.sqrt(3.0 / 2.0), moment_magnitude=0.0,
    )


@pytest.fixture()
def clvd_mt() -> MomentTensor:
    """A deviatoric tensor with significant CLVD component."""
    # Eigenvalues: 2, -1, -1 -> ε = 0.5
    return MomentTensor(
        mxx=2.0, myy=-1.0, mzz=-1.0, mxy=0.0, mxz=0.0, myz=0.0,
        scalar_moment=0.0, moment_magnitude=0.0,
    )


@pytest.fixture()
def synthetic_polarities():
    """Synthetic polarity observations from a known focal mechanism.

    Strike=0, dip=45, rake=90 (pure thrust).
    Stations distributed around the focal sphere.
    """
    rng = np.random.default_rng(42)
    n_stations = 20
    azimuths = rng.uniform(0, 360, n_stations)
    takeoff_angles = rng.uniform(30, 150, n_stations)

    # Compute true polarities from the known mechanism
    mt = strike_dip_rake_to_tensor(0.0, 45.0, 90.0, m0=1.0)
    from swe.moment_tensor.radiation import radiation_pattern_p

    rad = radiation_pattern_p(mt, azimuths, takeoff_angles)
    polarities = np.sign(rad).astype(np.int32)

    # Ensure we have enough non-zero polarities
    polarities[polarities == 0] = 1

    return {
        "azimuths": azimuths,
        "takeoff_angles": takeoff_angles,
        "polarities": polarities,
        "true_mt": mt,
        "true_sdr": (0.0, 45.0, 90.0),
    }
