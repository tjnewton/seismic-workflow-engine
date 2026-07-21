"""Grid-search moment tensor inversion from P-wave polarities.

Searches over the moment tensor space (full, deviatoric, or
double-couple constrained) to find the tensor that best explains
observed P-wave first-motion polarities.

References
----------
Tape, W. & Tape, C. (2015). A uniform parametrization of moment
    tensors. Geophys. J. Int., 202, 2074-2081.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from swe.moment_tensor.radiation import radiation_pattern_p
from swe.moment_tensor.tensor import (
    MomentTensor,
    decompose_moment_tensor,
    from_matrix,
    strike_dip_rake_to_tensor,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MomentTensorResult",
    "moment_tensor_batch",
    "moment_tensor_from_polarities",
    "moment_tensor_uncertainty",
]


class MomentTensorResult(NamedTuple):
    """Result from moment tensor grid-search inversion.

    Parameters
    ----------
    best : MomentTensor or None
        Best-fitting moment tensor, or ``None`` if no solution found.
    accepted : list of MomentTensor
        All solutions within the misfit threshold.
    misfits : NDArray
        Misfit values for the full grid.
    n_polarities : int
        Number of polarity observations used.
    iso_pct_range : tuple of float
        ``(min, max)`` of ISO percentage across accepted solutions.
    dc_pct_range : tuple of float
        ``(min, max)`` of DC percentage across accepted solutions.
    clvd_pct_range : tuple of float
        ``(min, max)`` of CLVD percentage across accepted solutions.
    """

    best: MomentTensor | None
    accepted: list[MomentTensor]
    misfits: NDArray[np.floating]
    n_polarities: int
    iso_pct_range: tuple[float, float] = (0.0, 0.0)
    dc_pct_range: tuple[float, float] = (0.0, 0.0)
    clvd_pct_range: tuple[float, float] = (0.0, 0.0)


def _generate_dc_grid(
    *,
    strike_step: float = 10.0,
    dip_step: float = 10.0,
    rake_step: float = 10.0,
) -> list[MomentTensor]:
    """Generate moment tensors for a double-couple grid search."""
    tensors: list[MomentTensor] = []
    for strike in np.arange(0.0, 360.0, strike_step):
        for dip in np.arange(0.0, 90.0 + dip_step / 2.0, dip_step):
            for rake in np.arange(-180.0, 180.0, rake_step):
                tensors.append(
                    strike_dip_rake_to_tensor(strike, dip, rake, m0=1.0)
                )
    return tensors


def _generate_deviatoric_grid(
    *,
    n_gamma: int = 12,
    n_delta: int = 7,
    n_strike: int = 36,
    n_dip: int = 10,
    n_rake: int = 36,
) -> list[MomentTensor]:
    """Generate deviatoric moment tensors using a simplified parameterization.

    Uses lune coordinates (gamma, delta) for the source type and
    (strike, dip, rake) for orientation.  gamma controls the CLVD
    fraction and delta=0 for deviatoric.
    """
    tensors: list[MomentTensor] = []

    # gamma in [-pi/6, pi/6] parameterizes the CLVD fraction
    gammas = np.linspace(-np.pi / 6.0, np.pi / 6.0, n_gamma)

    for gamma in gammas:
        # Eigenvalue ratios for deviatoric source (trace = 0)
        # Following Tape & Tape parameterization on the lune
        sin_g = np.sin(gamma)
        cos_g = np.cos(gamma)
        # Eigenvalues proportional to:
        lam1 = cos_g + sin_g / np.sqrt(3.0)
        lam2 = -2.0 * sin_g / np.sqrt(3.0)
        lam3 = -cos_g + sin_g / np.sqrt(3.0)
        # Normalize to unit moment
        norm = np.sqrt((lam1**2 + lam2**2 + lam3**2) / 2.0)
        if norm < 1e-30:
            continue
        lam1 /= norm
        lam2 /= norm
        lam3 /= norm

        # Sample orientations
        for strike in np.linspace(0.0, 360.0, n_strike, endpoint=False):
            for dip in np.linspace(0.0, 90.0, n_dip + 1):
                for rake in np.linspace(-180.0, 180.0, n_rake, endpoint=False):
                    # Build rotation matrix from strike/dip/rake
                    # to get the eigenvector frame
                    s = np.radians(strike)
                    d = np.radians(dip)
                    r = np.radians(rake)

                    # Eigenvector frame from fault geometry
                    sin_d = np.sin(d)
                    cos_d = np.cos(d)
                    sin_s = np.sin(s)
                    cos_s = np.cos(s)
                    sin_r = np.sin(r)
                    cos_r = np.cos(r)

                    # Fault normal (n) and slip direction (d)
                    n = np.array([
                        -sin_d * sin_s,
                        sin_d * cos_s,
                        -cos_d,
                    ])
                    slip = np.array([
                        cos_r * cos_s + sin_r * cos_d * sin_s,
                        cos_r * sin_s - sin_r * cos_d * cos_s,
                        -sin_r * sin_d,
                    ])
                    # Third axis (null/B axis)
                    b = np.cross(n, slip)
                    b_norm = np.linalg.norm(b)
                    if b_norm < 1e-12:
                        continue
                    b = b / b_norm

                    # T and P axes
                    t = (slip + n) / np.sqrt(2.0)
                    p = (slip - n) / np.sqrt(2.0)

                    # Rotation matrix: columns are T, B, P eigenvectors
                    R = np.column_stack([t, b, p])

                    # Build moment tensor
                    diag = np.diag([lam1, lam2, lam3])
                    m = R @ diag @ R.T

                    tensors.append(from_matrix(m, m0=1.0))

    return tensors


def _generate_full_grid(
    *,
    n_iso: int = 5,
    n_gamma: int = 12,
    n_strike: int = 36,
    n_dip: int = 10,
    n_rake: int = 36,
) -> list[MomentTensor]:
    """Generate full moment tensors including isotropic component."""
    tensors: list[MomentTensor] = []

    iso_values = np.linspace(-1.0, 1.0, n_iso)
    gammas = np.linspace(-np.pi / 6.0, np.pi / 6.0, n_gamma)

    for iso_frac in iso_values:
        for gamma in gammas:
            sin_g = np.sin(gamma)
            cos_g = np.cos(gamma)
            lam1 = cos_g + sin_g / np.sqrt(3.0)
            lam2 = -2.0 * sin_g / np.sqrt(3.0)
            lam3 = -cos_g + sin_g / np.sqrt(3.0)

            # Scale deviatoric and add isotropic
            dev_scale = np.sqrt(1.0 - iso_frac**2) if abs(iso_frac) < 1.0 else 0.0
            lam1 = dev_scale * lam1 + iso_frac / np.sqrt(3.0)
            lam2 = dev_scale * lam2 + iso_frac / np.sqrt(3.0)
            lam3 = dev_scale * lam3 + iso_frac / np.sqrt(3.0)

            norm = np.sqrt((lam1**2 + lam2**2 + lam3**2) / 2.0)
            if norm < 1e-30:
                continue
            lam1 /= norm
            lam2 /= norm
            lam3 /= norm

            for strike in np.linspace(0.0, 360.0, n_strike, endpoint=False):
                for dip in np.linspace(0.0, 90.0, n_dip + 1):
                    for rake in np.linspace(-180.0, 180.0, n_rake, endpoint=False):
                        s = np.radians(strike)
                        d = np.radians(dip)
                        r = np.radians(rake)

                        sin_d = np.sin(d)
                        cos_d = np.cos(d)
                        sin_s = np.sin(s)
                        cos_s = np.cos(s)
                        sin_r = np.sin(r)
                        cos_r = np.cos(r)

                        n_vec = np.array([
                            -sin_d * sin_s,
                            sin_d * cos_s,
                            -cos_d,
                        ])
                        slip = np.array([
                            cos_r * cos_s + sin_r * cos_d * sin_s,
                            cos_r * sin_s - sin_r * cos_d * cos_s,
                            -sin_r * sin_d,
                        ])
                        b = np.cross(n_vec, slip)
                        b_norm = np.linalg.norm(b)
                        if b_norm < 1e-12:
                            continue
                        b = b / b_norm

                        t = (slip + n_vec) / np.sqrt(2.0)
                        p = (slip - n_vec) / np.sqrt(2.0)

                        R = np.column_stack([t, b, p])
                        diag = np.diag([lam1, lam2, lam3])
                        m = R @ diag @ R.T

                        tensors.append(from_matrix(m, m0=1.0))

    return tensors


def moment_tensor_from_polarities(
    azimuths: NDArray[np.floating],
    takeoff_angles: NDArray[np.floating],
    polarities: NDArray[np.signedinteger] | NDArray[np.floating],
    *,
    constraint: str = "deviatoric",
    max_misfit: float = 0.2,
    min_polarities: int = 8,
    weights: NDArray[np.floating] | None = None,
    strike_step: float = 10.0,
    dip_step: float = 10.0,
    rake_step: float = 10.0,
) -> MomentTensorResult:
    """Find moment tensors consistent with observed P-wave polarities.

    Performs a grid search over the moment tensor space, evaluating the
    P-wave radiation pattern for each candidate tensor and comparing
    predicted polarities with observations.

    Parameters
    ----------
    azimuths : NDArray
        Source-to-station azimuths in degrees.
    takeoff_angles : NDArray
        Takeoff angles in degrees (0=up, 180=down).
    polarities : NDArray
        Observed polarities: +1 (compressional), -1 (dilatational).
        Stations with polarity 0 are excluded.
    constraint : str
        Search space constraint:

        - ``"dc"``: double-couple only (3 parameters)
        - ``"deviatoric"``: deviatoric tensors (5 parameters, default)
        - ``"full"``: full 6-parameter tensor
    max_misfit : float
        Maximum misfit fraction ``[0, 1]`` for accepting solutions.
    min_polarities : int
        Minimum non-zero polarity observations required.
    weights : NDArray, optional
        Per-observation weights (e.g. from polarity confidence).
    strike_step : float
        Grid step for strike in degrees (DC constraint).
    dip_step : float
        Grid step for dip in degrees (DC constraint).
    rake_step : float
        Grid step for rake in degrees (DC constraint).

    Returns
    -------
    MomentTensorResult
        Best solution, all accepted solutions, misfits, and count.
    """
    azimuths = np.asarray(azimuths, dtype=np.float64)
    takeoff_angles = np.asarray(takeoff_angles, dtype=np.float64)
    polarities = np.asarray(polarities, dtype=np.float64)

    # Filter out zero polarities
    valid = polarities != 0
    az = azimuths[valid]
    to = takeoff_angles[valid]
    pol = polarities[valid]

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)[valid]
    else:
        w = np.ones_like(pol)

    n_pol = len(pol)
    if n_pol < min_polarities:
        logger.warning(
            "Only %d valid polarities (need %d), cannot invert",
            n_pol,
            min_polarities,
        )
        return MomentTensorResult(
            best=None,
            accepted=[],
            misfits=np.array([], dtype=np.float64),
            n_polarities=n_pol,
        )

    total_weight = np.sum(w)

    # Generate candidate tensors
    constraint = constraint.lower()
    if constraint == "dc":
        candidates = _generate_dc_grid(
            strike_step=strike_step,
            dip_step=dip_step,
            rake_step=rake_step,
        )
    elif constraint == "deviatoric":
        candidates = _generate_deviatoric_grid()
    elif constraint == "full":
        candidates = _generate_full_grid()
    else:
        raise ValueError(
            f"Unknown constraint: {constraint!r}. "
            f"Supported: 'dc', 'deviatoric', 'full'"
        )

    logger.info(
        "Searching %d candidate moment tensors (constraint=%r, %d polarities)",
        len(candidates),
        constraint,
        n_pol,
    )

    all_misfits: list[float] = []
    accepted: list[tuple[MomentTensor, float]] = []

    for mt in candidates:
        # Compute predicted P-wave radiation
        rad = radiation_pattern_p(mt, az, to)
        predicted = np.sign(rad)

        # Weighted misfit
        mismatch = (predicted != pol).astype(np.float64)
        misfit = float(np.sum(w * mismatch) / total_weight)

        all_misfits.append(misfit)

        if misfit <= max_misfit:
            accepted.append((mt, misfit))

    # Sort by misfit
    accepted.sort(key=lambda x: x[1])

    accepted_mts = [mt for mt, _ in accepted]
    best = accepted_mts[0] if accepted_mts else None

    if best is not None:
        logger.info(
            "Best MT misfit: %.3f (%d solutions within threshold)",
            accepted[0][1],
            len(accepted_mts),
        )
    else:
        logger.warning("No moment tensor solution found within misfit threshold")

    # Compute component ranges from accepted solutions
    iso_pct_range = (0.0, 0.0)
    dc_pct_range = (0.0, 0.0)
    clvd_pct_range = (0.0, 0.0)
    if accepted_mts:
        iso_pcts: list[float] = []
        dc_pcts: list[float] = []
        clvd_pcts: list[float] = []
        for mt in accepted_mts:
            dec = decompose_moment_tensor(mt)
            iso_pcts.append(dec.iso_pct)
            dc_pcts.append(dec.dc_pct)
            clvd_pcts.append(dec.clvd_pct)
        iso_pct_range = (min(iso_pcts), max(iso_pcts))
        dc_pct_range = (min(dc_pcts), max(dc_pcts))
        clvd_pct_range = (min(clvd_pcts), max(clvd_pcts))

    return MomentTensorResult(
        best=best,
        accepted=accepted_mts,
        misfits=np.array(all_misfits, dtype=np.float64),
        n_polarities=n_pol,
        iso_pct_range=iso_pct_range,
        dc_pct_range=dc_pct_range,
        clvd_pct_range=clvd_pct_range,
    )


def moment_tensor_batch(
    events: list[dict],
    *,
    constraint: str = "deviatoric",
    max_misfit: float = 0.2,
    min_polarities: int = 8,
    strike_step: float = 10.0,
    dip_step: float = 10.0,
    rake_step: float = 10.0,
) -> list[MomentTensorResult]:
    """Invert moment tensors for multiple events.

    Parameters
    ----------
    events : list of dict
        Each dict must contain:

        - ``'azimuths'``: array of azimuths (degrees)
        - ``'takeoff_angles'``: array of takeoff angles (degrees)
        - ``'polarities'``: array of polarity observations
        - ``'weights'`` (optional): array of weights
    constraint : str
        ``"dc"``, ``"deviatoric"``, or ``"full"``.
    max_misfit : float
        Maximum allowed misfit fraction.
    min_polarities : int
        Minimum non-zero polarity observations.
    strike_step, dip_step, rake_step : float
        Grid step sizes for DC constraint.

    Returns
    -------
    list of MomentTensorResult
        One result per event.
    """
    results: list[MomentTensorResult] = []

    for i, event in enumerate(events):
        logger.info(
            "Inverting moment tensor for event %d of %d", i + 1, len(events)
        )
        result = moment_tensor_from_polarities(
            azimuths=event["azimuths"],
            takeoff_angles=event["takeoff_angles"],
            polarities=event["polarities"],
            constraint=constraint,
            max_misfit=max_misfit,
            min_polarities=min_polarities,
            weights=event.get("weights"),
            strike_step=strike_step,
            dip_step=dip_step,
            rake_step=rake_step,
        )
        results.append(result)

    return results


def moment_tensor_uncertainty(result: MomentTensorResult) -> dict:
    """Summarise component ranges from a moment tensor inversion.

    Parameters
    ----------
    result : MomentTensorResult
        Result from :func:`moment_tensor_from_polarities`.

    Returns
    -------
    dict
        Keys: ``iso_pct_range``, ``dc_pct_range``, ``clvd_pct_range``,
        ``n_accepted``.
    """
    return {
        "iso_pct_range": result.iso_pct_range,
        "dc_pct_range": result.dc_pct_range,
        "clvd_pct_range": result.clvd_pct_range,
        "n_accepted": len(result.accepted),
    }
