"""Amplitude-based moment tensor inversion.

Complements the polarity-based method in :mod:`swe.moment_tensor.inversion`
by using P-wave amplitude ratios and absolute amplitudes to constrain the
moment tensor.  This is particularly important for characterising
low-frequency earthquakes (LFEs) where first-motion polarities are
ambiguous or unreadable.

The approach fits observed amplitude ratios (P/SV, P/SH, SV/SH) to
theoretical radiation patterns for candidate moment tensors, using the
same grid-search framework as the polarity inversion.

References
----------
Julian, B.R. & Foulger, G.R. (1996). Earthquake mechanisms from
    linear-programming inversion of seismic-wave amplitude ratios.
    Bull. Seismol. Soc. Am., 86, 972-980.
Hardebeck, J.L. & Shearer, P.M. (2003). Using S/P amplitude ratios to
    constrain the focal mechanisms of small earthquakes. Bull. Seismol.
    Soc. Am., 93, 2434-2444.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from swe.moment_tensor.radiation import (
    radiation_pattern_p,
    radiation_pattern_sh,
    radiation_pattern_sv,
)
from swe.moment_tensor.tensor import (
    MomentTensor,
    decompose_moment_tensor,
    from_matrix,
    strike_dip_rake_to_tensor,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AmplitudeMTResult",
    "moment_tensor_from_amplitudes",
    "moment_tensor_from_amplitude_ratios",
    "moment_tensor_from_s_p_ratios",
]


class AmplitudeMTResult(NamedTuple):
    """Result from amplitude-based moment tensor inversion.

    Parameters
    ----------
    best : MomentTensor or None
        Best-fitting moment tensor.
    accepted : list of MomentTensor
        All solutions within the misfit threshold.
    misfits : NDArray
        L2 misfit values for the full grid.
    n_observations : int
        Number of amplitude observations used.
    variance_reduction : float
        Variance reduction of the best-fitting solution (0-1).
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
    n_observations: int
    variance_reduction: float = 0.0
    iso_pct_range: tuple[float, float] = (0.0, 0.0)
    dc_pct_range: tuple[float, float] = (0.0, 0.0)
    clvd_pct_range: tuple[float, float] = (0.0, 0.0)


def moment_tensor_from_amplitudes(
    azimuths: NDArray[np.floating],
    takeoff_angles: NDArray[np.floating],
    amplitudes_p: NDArray[np.floating],
    *,
    amplitudes_sv: NDArray[np.floating] | None = None,
    amplitudes_sh: NDArray[np.floating] | None = None,
    constraint: str = "deviatoric",
    max_misfit: float = 0.5,
    min_observations: int = 6,
    weights: NDArray[np.floating] | None = None,
    strike_step: float = 10.0,
    dip_step: float = 10.0,
    rake_step: float = 10.0,
) -> AmplitudeMTResult:
    """Find moment tensors that best fit observed P-wave amplitudes.

    Uses an L2 misfit between normalised observed and predicted
    amplitude patterns.

    Parameters
    ----------
    azimuths : NDArray
        Source-to-station azimuths in degrees.
    takeoff_angles : NDArray
        Takeoff angles in degrees (0=up, 180=down).
    amplitudes_p : NDArray
        Observed P-wave amplitudes (absolute or relative).
    amplitudes_sv : NDArray, optional
        Observed SV-wave amplitudes.
    amplitudes_sh : NDArray, optional
        Observed SH-wave amplitudes.
    constraint : str
        ``"dc"``, ``"deviatoric"``, or ``"full"``.
    max_misfit : float
        Maximum normalised L2 misfit for accepting solutions.
    min_observations : int
        Minimum number of amplitude observations required.
    weights : NDArray, optional
        Per-station weights.
    strike_step, dip_step, rake_step : float
        Grid steps for DC constraint (degrees).

    Returns
    -------
    AmplitudeMTResult
    """
    azimuths = np.asarray(azimuths, dtype=np.float64)
    takeoff_angles = np.asarray(takeoff_angles, dtype=np.float64)
    amplitudes_p = np.asarray(amplitudes_p, dtype=np.float64)

    # Filter out zero/NaN amplitudes
    valid = np.isfinite(amplitudes_p) & (amplitudes_p != 0)
    az = azimuths[valid]
    to = takeoff_angles[valid]
    amp_p = amplitudes_p[valid]

    n_obs = len(amp_p)
    if n_obs < min_observations:
        logger.warning(
            "Only %d valid amplitudes (need %d), cannot invert",
            n_obs,
            min_observations,
        )
        return AmplitudeMTResult(
            best=None,
            accepted=[],
            misfits=np.array([], dtype=np.float64),
            n_observations=n_obs,
        )

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)[valid]
    else:
        w = np.ones_like(amp_p)

    # Normalise observed amplitudes
    amp_p_norm = amp_p / np.max(np.abs(amp_p))

    # Build amplitude vectors for SV/SH if provided
    amp_sv_norm = None
    if amplitudes_sv is not None:
        amp_sv = np.asarray(amplitudes_sv, dtype=np.float64)[valid]
        amp_sv_norm = amp_sv / np.max(np.abs(amp_sv)) if np.any(amp_sv != 0) else None

    amp_sh_norm = None
    if amplitudes_sh is not None:
        amp_sh = np.asarray(amplitudes_sh, dtype=np.float64)[valid]
        amp_sh_norm = amp_sh / np.max(np.abs(amp_sh)) if np.any(amp_sh != 0) else None

    # Generate candidates (reuse polarity inversion grids)
    from swe.moment_tensor.inversion import (
        _generate_dc_grid,
        _generate_deviatoric_grid,
        _generate_full_grid,
    )

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
        "Searching %d candidate MTs (amplitude-based, constraint=%r, %d observations)",
        len(candidates),
        constraint,
        n_obs,
    )

    total_weight = np.sum(w)
    all_misfits: list[float] = []
    accepted: list[tuple[MomentTensor, float]] = []

    # Observed variance for variance reduction
    obs_var = float(np.sum(w * amp_p_norm**2) / total_weight)

    for mt in candidates:
        # Predicted P amplitudes
        pred_p = radiation_pattern_p(mt, az, to)
        pred_max = np.max(np.abs(pred_p))
        if pred_max < 1e-30:
            all_misfits.append(float("inf"))
            continue
        pred_p_norm = pred_p / pred_max

        # L2 misfit (P)
        residual = amp_p_norm - pred_p_norm
        misfit = float(np.sum(w * residual**2) / total_weight)

        # Add SV misfit if available
        if amp_sv_norm is not None:
            pred_sv = radiation_pattern_sv(mt, az, to)
            pred_sv_max = np.max(np.abs(pred_sv))
            if pred_sv_max > 1e-30:
                pred_sv_norm = pred_sv / pred_sv_max
                misfit += float(np.sum(w * (amp_sv_norm - pred_sv_norm) ** 2) / total_weight)

        # Add SH misfit if available
        if amp_sh_norm is not None:
            pred_sh = radiation_pattern_sh(mt, az, to)
            pred_sh_max = np.max(np.abs(pred_sh))
            if pred_sh_max > 1e-30:
                pred_sh_norm = pred_sh / pred_sh_max
                misfit += float(np.sum(w * (amp_sh_norm - pred_sh_norm) ** 2) / total_weight)

        all_misfits.append(misfit)

        if misfit <= max_misfit:
            accepted.append((mt, misfit))

    accepted.sort(key=lambda x: x[1])
    accepted_mts = [mt for mt, _ in accepted]
    best = accepted_mts[0] if accepted_mts else None

    # Variance reduction for best solution
    vr = 0.0
    if best is not None and obs_var > 0:
        best_misfit = accepted[0][1]
        vr = max(0.0, 1.0 - best_misfit / obs_var)

    if best is not None:
        logger.info(
            "Best amplitude MT misfit: %.4f (VR=%.1f%%, %d solutions)",
            accepted[0][1],
            vr * 100,
            len(accepted_mts),
        )
    else:
        logger.warning("No amplitude-based MT solution within misfit threshold")

    # Component ranges
    iso_pct_range = (0.0, 0.0)
    dc_pct_range = (0.0, 0.0)
    clvd_pct_range = (0.0, 0.0)
    if accepted_mts:
        iso_pcts = []
        dc_pcts = []
        clvd_pcts = []
        for mt in accepted_mts:
            dec = decompose_moment_tensor(mt)
            iso_pcts.append(dec.iso_pct)
            dc_pcts.append(dec.dc_pct)
            clvd_pcts.append(dec.clvd_pct)
        iso_pct_range = (min(iso_pcts), max(iso_pcts))
        dc_pct_range = (min(dc_pcts), max(dc_pcts))
        clvd_pct_range = (min(clvd_pcts), max(clvd_pcts))

    return AmplitudeMTResult(
        best=best,
        accepted=accepted_mts,
        misfits=np.array(all_misfits, dtype=np.float64),
        n_observations=n_obs,
        variance_reduction=vr,
        iso_pct_range=iso_pct_range,
        dc_pct_range=dc_pct_range,
        clvd_pct_range=clvd_pct_range,
    )


def moment_tensor_from_amplitude_ratios(
    azimuths: NDArray[np.floating],
    takeoff_angles: NDArray[np.floating],
    sv_p_ratios: NDArray[np.floating],
    *,
    sh_p_ratios: NDArray[np.floating] | None = None,
    constraint: str = "deviatoric",
    max_misfit: float = 0.5,
    min_observations: int = 6,
    weights: NDArray[np.floating] | None = None,
    strike_step: float = 10.0,
    dip_step: float = 10.0,
    rake_step: float = 10.0,
) -> AmplitudeMTResult:
    """Find moment tensors from SV/P (and optionally SH/P) amplitude ratios.

    Amplitude ratios are less sensitive to path effects and site
    amplification than absolute amplitudes, making them useful for
    networks with heterogeneous station characteristics.

    Parameters
    ----------
    azimuths : NDArray
        Source-to-station azimuths in degrees.
    takeoff_angles : NDArray
        Takeoff angles in degrees (0=up, 180=down).
    sv_p_ratios : NDArray
        Observed SV/P amplitude ratios (log10 scale).
    sh_p_ratios : NDArray, optional
        Observed SH/P amplitude ratios (log10 scale).
    constraint : str
        ``"dc"``, ``"deviatoric"``, or ``"full"``.
    max_misfit : float
        Maximum normalised L2 misfit.
    min_observations : int
        Minimum observations required.
    weights : NDArray, optional
        Per-station weights.
    strike_step, dip_step, rake_step : float
        Grid steps for DC constraint (degrees).

    Returns
    -------
    AmplitudeMTResult
    """
    azimuths = np.asarray(azimuths, dtype=np.float64)
    takeoff_angles = np.asarray(takeoff_angles, dtype=np.float64)
    sv_p_ratios = np.asarray(sv_p_ratios, dtype=np.float64)

    valid = np.isfinite(sv_p_ratios)
    az = azimuths[valid]
    to = takeoff_angles[valid]
    svp = sv_p_ratios[valid]

    n_obs = len(svp)
    if n_obs < min_observations:
        logger.warning(
            "Only %d valid ratio observations (need %d), cannot invert",
            n_obs,
            min_observations,
        )
        return AmplitudeMTResult(
            best=None,
            accepted=[],
            misfits=np.array([], dtype=np.float64),
            n_observations=n_obs,
        )

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)[valid]
    else:
        w = np.ones_like(svp)

    shp = None
    if sh_p_ratios is not None:
        shp = np.asarray(sh_p_ratios, dtype=np.float64)[valid]

    from swe.moment_tensor.inversion import (
        _generate_dc_grid,
        _generate_deviatoric_grid,
        _generate_full_grid,
    )

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
        "Searching %d candidates (amplitude ratios, constraint=%r, %d observations)",
        len(candidates),
        constraint,
        n_obs,
    )

    total_weight = np.sum(w)
    all_misfits: list[float] = []
    accepted: list[tuple[MomentTensor, float]] = []

    obs_var = float(np.sum(w * svp**2) / total_weight)

    for mt in candidates:
        pred_p = radiation_pattern_p(mt, az, to)
        pred_sv = radiation_pattern_sv(mt, az, to)

        # Avoid division by zero
        safe_p = np.where(np.abs(pred_p) > 1e-30, pred_p, 1e-30)
        pred_svp = np.log10(np.abs(pred_sv / safe_p) + 1e-30)

        residual = svp - pred_svp
        misfit = float(np.sum(w * residual**2) / total_weight)

        if shp is not None:
            pred_sh = radiation_pattern_sh(mt, az, to)
            pred_shp = np.log10(np.abs(pred_sh / safe_p) + 1e-30)
            misfit += float(np.sum(w * (shp - pred_shp) ** 2) / total_weight)

        all_misfits.append(misfit)
        if misfit <= max_misfit:
            accepted.append((mt, misfit))

    accepted.sort(key=lambda x: x[1])
    accepted_mts = [mt for mt, _ in accepted]
    best = accepted_mts[0] if accepted_mts else None

    vr = 0.0
    if best is not None and obs_var > 0:
        vr = max(0.0, 1.0 - accepted[0][1] / obs_var)

    # Component ranges
    iso_pct_range = (0.0, 0.0)
    dc_pct_range = (0.0, 0.0)
    clvd_pct_range = (0.0, 0.0)
    if accepted_mts:
        iso_pcts = []
        dc_pcts = []
        clvd_pcts = []
        for mt in accepted_mts:
            dec = decompose_moment_tensor(mt)
            iso_pcts.append(dec.iso_pct)
            dc_pcts.append(dec.dc_pct)
            clvd_pcts.append(dec.clvd_pct)
        iso_pct_range = (min(iso_pcts), max(iso_pcts))
        dc_pct_range = (min(dc_pcts), max(dc_pcts))
        clvd_pct_range = (min(clvd_pcts), max(clvd_pcts))

    return AmplitudeMTResult(
        best=best,
        accepted=accepted_mts,
        misfits=np.array(all_misfits, dtype=np.float64),
        n_observations=n_obs,
        variance_reduction=vr,
        iso_pct_range=iso_pct_range,
        dc_pct_range=dc_pct_range,
        clvd_pct_range=clvd_pct_range,
    )


def moment_tensor_from_s_p_ratios(
    azimuths: NDArray[np.floating],
    takeoff_angles: NDArray[np.floating],
    s_p_ratios: NDArray[np.floating],
    *,
    constraint: str = "dc",
    max_misfit: float = 0.5,
    min_observations: int = 6,
    weights: NDArray[np.floating] | None = None,
    strike_step: float = 10.0,
    dip_step: float = 10.0,
    rake_step: float = 10.0,
) -> AmplitudeMTResult:
    """Find moment tensors from **total** S/P amplitude ratios.

    Unlike :func:`moment_tensor_from_amplitude_ratios` (which needs SV/P and
    SH/P separately, and therefore rotated three-component data), this uses the
    rotation-invariant total shear amplitude: the predicted ratio is
    ``sqrt(SV**2 + SH**2) / |P|``.  It is the defensible inversion for borehole
    arrays whose component orientations are not well constrained, paired with
    :func:`swe.moment_tensor.three_component_amplitudes`.  S/P ratios are also
    insensitive to common path/site amplification (Hardebeck & Shearer 2003).

    Parameters
    ----------
    azimuths : NDArray
        Source-to-station azimuths in degrees.
    takeoff_angles : NDArray
        Takeoff angles in degrees (0=up, 180=down).
    s_p_ratios : NDArray
        Observed total S/P amplitude ratios (linear, not log).  Non-finite or
        non-positive entries are dropped.
    constraint : str
        ``"dc"``, ``"deviatoric"``, or ``"full"``.
    max_misfit : float
        Maximum normalised L2 misfit (in log10-ratio space).
    min_observations : int
        Minimum observations required.
    weights : NDArray, optional
        Per-station weights.
    strike_step, dip_step, rake_step : float
        Grid steps for the DC constraint (degrees).

    Returns
    -------
    AmplitudeMTResult
    """
    azimuths = np.asarray(azimuths, dtype=np.float64)
    takeoff_angles = np.asarray(takeoff_angles, dtype=np.float64)
    s_p_ratios = np.asarray(s_p_ratios, dtype=np.float64)

    valid = np.isfinite(s_p_ratios) & (s_p_ratios > 0)
    az = azimuths[valid]
    to = takeoff_angles[valid]
    obs = np.log10(s_p_ratios[valid])

    n_obs = len(obs)
    if n_obs < min_observations:
        logger.warning(
            "Only %d valid S/P observations (need %d), cannot invert",
            n_obs,
            min_observations,
        )
        return AmplitudeMTResult(
            best=None,
            accepted=[],
            misfits=np.array([], dtype=np.float64),
            n_observations=n_obs,
        )

    w = (
        np.asarray(weights, dtype=np.float64)[valid]
        if weights is not None
        else np.ones_like(obs)
    )

    from swe.moment_tensor.inversion import (
        _generate_dc_grid,
        _generate_deviatoric_grid,
        _generate_full_grid,
    )

    constraint = constraint.lower()
    if constraint == "dc":
        candidates = _generate_dc_grid(
            strike_step=strike_step, dip_step=dip_step, rake_step=rake_step
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
        "Searching %d candidates (total S/P ratios, constraint=%r, %d observations)",
        len(candidates),
        constraint,
        n_obs,
    )

    total_weight = np.sum(w)
    # Variance about the mean of the (log) observations: the conventional VR
    # denominator (mean removed), so VR reflects fit beyond a constant offset.
    obs_mean = float(np.sum(w * obs) / total_weight)
    obs_var = float(np.sum(w * (obs - obs_mean) ** 2) / total_weight)

    all_misfits: list[float] = []
    accepted: list[tuple[MomentTensor, float]] = []

    for mt in candidates:
        pred_p = radiation_pattern_p(mt, az, to)
        pred_sv = radiation_pattern_sv(mt, az, to)
        pred_sh = radiation_pattern_sh(mt, az, to)
        safe_p = np.where(np.abs(pred_p) > 1e-30, np.abs(pred_p), 1e-30)
        pred_s = np.sqrt(pred_sv**2 + pred_sh**2)
        pred_ratio = np.log10(pred_s / safe_p + 1e-30)

        residual = obs - pred_ratio
        misfit = float(np.sum(w * residual**2) / total_weight)
        all_misfits.append(misfit)
        if misfit <= max_misfit:
            accepted.append((mt, misfit))

    accepted.sort(key=lambda x: x[1])
    accepted_mts = [mt for mt, _ in accepted]
    best = accepted_mts[0] if accepted_mts else None

    vr = 0.0
    if best is not None and obs_var > 0:
        vr = max(0.0, 1.0 - accepted[0][1] / obs_var)

    iso_pct_range = (0.0, 0.0)
    dc_pct_range = (0.0, 0.0)
    clvd_pct_range = (0.0, 0.0)
    if accepted_mts:
        iso_pcts, dc_pcts, clvd_pcts = [], [], []
        for mt in accepted_mts:
            dec = decompose_moment_tensor(mt)
            iso_pcts.append(dec.iso_pct)
            dc_pcts.append(dec.dc_pct)
            clvd_pcts.append(dec.clvd_pct)
        iso_pct_range = (min(iso_pcts), max(iso_pcts))
        dc_pct_range = (min(dc_pcts), max(dc_pcts))
        clvd_pct_range = (min(clvd_pcts), max(clvd_pcts))

    return AmplitudeMTResult(
        best=best,
        accepted=accepted_mts,
        misfits=np.array(all_misfits, dtype=np.float64),
        n_observations=n_obs,
        variance_reduction=vr,
        iso_pct_range=iso_pct_range,
        dc_pct_range=dc_pct_range,
        clvd_pct_range=clvd_pct_range,
    )
