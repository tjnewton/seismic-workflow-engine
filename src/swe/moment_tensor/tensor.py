"""Core moment tensor types and algebra.

Provides :class:`MomentTensor` (6 independent components of the symmetric
3x3 tensor), decomposition into isotropic + DC + CLVD components, and
conversion to/from strike-dip-rake focal mechanism parameters.

References
----------
Jost, M.L. & Herrmann, R.B. (1989). A student's guide to and review of
    moment tensors. Seismol. Res. Lett., 60(2), 37-57.
Knopoff, L. & Randall, M.J. (1970). The compensated linear-vector dipole:
    A possible mechanism for deep earthquakes. J. Geophys. Res., 75,
    4957-4963.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "MomentTensor",
    "MomentTensorDecomposition",
    "decompose_moment_tensor",
    "from_matrix",
    "moment_magnitude",
    "scalar_moment",
    "strike_dip_rake_to_tensor",
    "tensor_to_strike_dip_rake",
    "to_matrix",
]


class MomentTensor(NamedTuple):
    """A symmetric 3x3 moment tensor (6 independent components).

    Uses the convention: x=North, y=East, z=Down (NED).

    Parameters
    ----------
    mxx : float
        North-North component.
    myy : float
        East-East component.
    mzz : float
        Down-Down component.
    mxy : float
        North-East component.
    mxz : float
        North-Down component.
    myz : float
        East-Down component.
    scalar_moment : float
        Scalar seismic moment M₀ in N·m.
    moment_magnitude : float
        Moment magnitude Mw.
    """

    mxx: float
    myy: float
    mzz: float
    mxy: float
    mxz: float
    myz: float
    scalar_moment: float = 0.0
    moment_magnitude: float = 0.0


class MomentTensorDecomposition(NamedTuple):
    """Decomposition of a moment tensor into ISO + DC + CLVD.

    Parameters
    ----------
    iso_pct : float
        Isotropic component percentage ``[0, 100]``.
    dc_pct : float
        Double-couple component percentage ``[0, 100]``.
    clvd_pct : float
        Compensated linear vector dipole percentage ``[0, 100]``.
    full_tensor : MomentTensor
        Original (input) tensor.
    dc_tensor : MomentTensor
        Best double-couple extracted from the deviatoric part.
    iso_scalar : float
        Trace / 3 of the original tensor.
    """

    iso_pct: float
    dc_pct: float
    clvd_pct: float
    full_tensor: MomentTensor
    dc_tensor: MomentTensor
    iso_scalar: float


def to_matrix(mt: MomentTensor) -> NDArray[np.floating]:
    """Convert a :class:`MomentTensor` to a 3x3 NumPy array.

    Parameters
    ----------
    mt : MomentTensor
        Input tensor.

    Returns
    -------
    NDArray
        Symmetric ``(3, 3)`` matrix.
    """
    return np.array(
        [
            [mt.mxx, mt.mxy, mt.mxz],
            [mt.mxy, mt.myy, mt.myz],
            [mt.mxz, mt.myz, mt.mzz],
        ],
        dtype=np.float64,
    )


def from_matrix(
    m: NDArray[np.floating],
    *,
    m0: float | None = None,
) -> MomentTensor:
    """Create a :class:`MomentTensor` from a 3x3 matrix.

    Parameters
    ----------
    m : NDArray
        Symmetric ``(3, 3)`` moment tensor matrix.
    m0 : float, optional
        Override scalar moment.  If ``None``, computed from *m*.

    Returns
    -------
    MomentTensor
    """
    if m0 is None:
        m0 = scalar_moment(m)
    mw = moment_magnitude(m0)
    return MomentTensor(
        mxx=float(m[0, 0]),
        myy=float(m[1, 1]),
        mzz=float(m[2, 2]),
        mxy=float(m[0, 1]),
        mxz=float(m[0, 2]),
        myz=float(m[1, 2]),
        scalar_moment=m0,
        moment_magnitude=mw,
    )


def scalar_moment(m: NDArray[np.floating] | MomentTensor) -> float:
    r"""Compute scalar seismic moment M₀.

    .. math::
        M_0 = \frac{1}{\sqrt{2}} \sqrt{\sum_{ij} m_{ij}^2}

    Parameters
    ----------
    m : NDArray or MomentTensor
        Symmetric ``(3, 3)`` matrix or :class:`MomentTensor`.

    Returns
    -------
    float
        Scalar moment M₀.
    """
    if isinstance(m, MomentTensor):
        m = to_matrix(m)
    m = np.asarray(m, dtype=np.float64)
    return float(np.sqrt(np.sum(m**2) / 2.0))


def moment_magnitude(m0: float) -> float:
    r"""Compute moment magnitude from scalar moment.

    .. math::
        M_w = \frac{2}{3} \log_{10}(M_0) - 10.7

    Parameters
    ----------
    m0 : float
        Scalar seismic moment in N·m.

    Returns
    -------
    float
        Moment magnitude Mw.
    """
    if m0 <= 0:
        return -999.0
    return (2.0 / 3.0) * np.log10(m0) - 10.7


def decompose_moment_tensor(mt: MomentTensor) -> MomentTensorDecomposition:
    """Decompose a moment tensor into isotropic + DC + CLVD components.

    Uses the Knopoff & Randall (1970) decomposition via eigenvalue analysis
    of the deviatoric part.

    Parameters
    ----------
    mt : MomentTensor
        Input moment tensor.

    Returns
    -------
    MomentTensorDecomposition
        Percentages and component tensors.
    """
    m = to_matrix(mt)

    # Isotropic part
    iso = np.trace(m) / 3.0
    m_dev = m - iso * np.eye(3)

    # Eigendecomposition of deviatoric part
    eigenvalues, eigenvectors = np.linalg.eigh(m_dev)

    # Sort by absolute value descending
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Decompose deviatoric eigenvalues into DC + CLVD
    # Following Jost & Herrmann (1989):
    # |λ1| >= |λ2| >= |λ3|
    # ε = -λ_min / λ_max where λ_min is the smallest absolute eigenvalue
    abs_eigs = np.abs(eigenvalues)
    max_abs = abs_eigs[0]

    if max_abs < 1e-30:
        # Zero deviatoric tensor
        dc_pct = 0.0
        clvd_pct = 0.0
    else:
        epsilon = abs_eigs[2] / max_abs
        dc_pct = float((1.0 - 2.0 * epsilon) * 100.0)
        clvd_pct = float(2.0 * epsilon * 100.0)

    # Isotropic percentage relative to total
    iso_abs = abs(iso)
    dev_m0 = scalar_moment(m_dev)

    total_contrib = iso_abs * np.sqrt(6.0) + dev_m0 * np.sqrt(2.0)
    if total_contrib > 0:
        iso_pct = float(iso_abs * np.sqrt(6.0) / total_contrib * 100.0)
    else:
        iso_pct = 0.0

    # Rescale DC/CLVD to share the non-isotropic portion
    non_iso_pct = 100.0 - iso_pct
    dc_pct = dc_pct * non_iso_pct / 100.0
    clvd_pct = clvd_pct * non_iso_pct / 100.0

    # Reconstruct best DC tensor from dominant eigenvalue pair
    # DC part: remove the CLVD by zeroing the smallest eigenvalue
    dc_eigs = eigenvalues.copy()
    dc_eigs[2] = 0.0
    # Adjust to make trace zero: if λ1 + λ2 != 0, redistribute
    dc_eigs[1] = -dc_eigs[0]
    dc_mat = eigenvectors @ np.diag(dc_eigs) @ eigenvectors.T
    dc_mt = from_matrix(dc_mat)

    return MomentTensorDecomposition(
        iso_pct=iso_pct,
        dc_pct=dc_pct,
        clvd_pct=clvd_pct,
        full_tensor=mt,
        dc_tensor=dc_mt,
        iso_scalar=float(iso),
    )


def strike_dip_rake_to_tensor(
    strike: float,
    dip: float,
    rake: float,
    *,
    m0: float = 1.0,
) -> MomentTensor:
    """Convert a focal mechanism (strike/dip/rake) to a moment tensor.

    Parameters
    ----------
    strike : float
        Strike in degrees ``[0, 360)``.
    dip : float
        Dip in degrees ``[0, 90]``.
    rake : float
        Rake in degrees ``[-180, 180)``.
    m0 : float
        Scalar moment to scale the tensor.

    Returns
    -------
    MomentTensor
        Pure double-couple moment tensor.
    """
    s = np.radians(strike)
    d = np.radians(dip)
    r = np.radians(rake)

    sin_d = np.sin(d)
    cos_d = np.cos(d)
    sin_2d = np.sin(2.0 * d)
    cos_2d = np.cos(2.0 * d)
    sin_r = np.sin(r)
    cos_r = np.cos(r)
    sin_s = np.sin(s)
    cos_s = np.cos(s)
    sin_2s = np.sin(2.0 * s)
    cos_2s = np.cos(2.0 * s)

    # Aki & Richards (2002) Table 4.4, NED convention
    mxx = -m0 * (sin_d * cos_r * sin_2s + sin_2d * sin_r * sin_s**2)
    myy = m0 * (sin_d * cos_r * sin_2s - sin_2d * sin_r * cos_s**2)
    mzz = m0 * sin_2d * sin_r
    mxy = m0 * (sin_d * cos_r * cos_2s + 0.5 * sin_2d * sin_r * sin_2s)
    mxz = -m0 * (cos_d * cos_r * cos_s + cos_2d * sin_r * sin_s)
    myz = -m0 * (cos_d * cos_r * sin_s - cos_2d * sin_r * cos_s)

    mw = moment_magnitude(m0)

    return MomentTensor(
        mxx=float(mxx),
        myy=float(myy),
        mzz=float(mzz),
        mxy=float(mxy),
        mxz=float(mxz),
        myz=float(myz),
        scalar_moment=m0,
        moment_magnitude=mw,
    )


def tensor_to_strike_dip_rake(
    mt: MomentTensor,
) -> tuple[float, float, float]:
    """Extract best double-couple strike/dip/rake from a moment tensor.

    Performs eigendecomposition of the deviatoric part and converts the
    T and P axes to a focal mechanism.

    Parameters
    ----------
    mt : MomentTensor
        Input moment tensor.

    Returns
    -------
    strike : float
        Strike in degrees ``[0, 360)``.
    dip : float
        Dip in degrees ``[0, 90]``.
    rake : float
        Rake in degrees ``[-180, 180)``.
    """
    m = to_matrix(mt)

    # Remove isotropic part
    iso = np.trace(m) / 3.0
    m_dev = m - iso * np.eye(3)

    eigenvalues, eigenvectors = np.linalg.eigh(m_dev)

    # Sort: T (most positive), B (intermediate), P (most negative)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    t_axis = eigenvectors[:, 0]  # tension
    p_axis = eigenvectors[:, 2]  # pressure

    # Fault normal and slip from T and P axes
    # n = (T + P) / |T + P|, d = (T - P) / |T - P|
    n_vec = t_axis + p_axis
    d_vec = t_axis - p_axis

    n_norm = np.linalg.norm(n_vec)
    d_norm = np.linalg.norm(d_vec)

    if n_norm < 1e-12 or d_norm < 1e-12:
        return 0.0, 0.0, 0.0

    n_vec = n_vec / n_norm
    d_vec = d_vec / d_norm

    # Ensure fault normal points upward (nz < 0 in z-Down convention)
    if n_vec[2] > 0:
        n_vec = -n_vec
        d_vec = -d_vec

    # Extract strike, dip, rake from fault normal and slip vector
    # NED convention: n = (-sin(d)*sin(s), sin(d)*cos(s), -cos(d))
    # cos(dip) = -nz
    dip = np.arccos(np.clip(-n_vec[2], -1.0, 1.0))
    sin_d = np.sin(dip)

    if sin_d < 1e-10:
        # Near-horizontal fault: strike undefined
        strike = 0.0
        rake = np.arctan2(-d_vec[0], d_vec[1])
    else:
        strike = np.arctan2(-n_vec[0] / sin_d, n_vec[1] / sin_d)
        # Rake from slip vector
        cos_s = np.cos(strike)
        sin_s = np.sin(strike)
        rake = np.arctan2(
            -d_vec[2] / sin_d,
            d_vec[0] * cos_s + d_vec[1] * sin_s,
        )

    strike = float(np.degrees(strike)) % 360.0
    dip = float(np.degrees(dip))
    rake = float(np.degrees(rake))

    if rake >= 180.0:
        rake -= 360.0
    if rake < -180.0:
        rake += 360.0

    return strike, dip, rake
