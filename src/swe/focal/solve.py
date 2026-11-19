"""Grid-search focal mechanism solver from P-wave first-motion polarities.

Evaluates the P-wave radiation pattern over a grid of strike, dip,
and rake values to find focal mechanisms consistent with observed
first-motion polarities.

References
----------
Aki, K. & Richards, P.G. (2002). Quantitative Seismology (2nd ed.).
    University Science Books. Eq. 4.88.
Hardebeck, J.L. & Shearer, P.M. (2002). A new method for determining
    first-motion focal mechanisms. Bull. Seismol. Soc. Am., 92(6),
    2264-2276.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "FocalMechanism",
    "FocalMechanismResult",
    "focal_mechanism_gridsearch",
    "focal_mechanism_uncertainty",
    "focal_mechanisms_batch",
    "polarity_result_to_event_dict",
]


class FocalMechanism(NamedTuple):
    """A single focal mechanism solution."""

    strike: float
    """Strike angle in degrees [0, 360)."""
    dip: float
    """Dip angle in degrees [0, 90]."""
    rake: float
    """Rake angle in degrees [-180, 180)."""
    strike2: float
    """Auxiliary plane strike in degrees."""
    dip2: float
    """Auxiliary plane dip in degrees."""
    rake2: float
    """Auxiliary plane rake in degrees."""
    misfit: float
    """Fraction of misfit polarities."""
    n_polarities: int
    """Number of polarity observations used."""


class FocalMechanismResult(NamedTuple):
    """Result from focal mechanism grid search."""

    mechanisms: list[FocalMechanism]
    """All solutions within the misfit threshold, sorted by misfit."""
    best: FocalMechanism | None
    """Best-fitting mechanism, or ``None`` if no solution found."""
    focal_data: NDArray[np.floating]
    """Best mechanism as ``(1, 3)`` array of ``[strike, dip, rake]``,
    directly compatible with :func:`swe.stress.stress_inversion` input.
    Empty array if no solution found."""
    all_misfits: NDArray[np.floating]
    """Misfit values for the full grid search."""
    strike_std: float = 0.0
    """Circular standard deviation of strike across accepted mechanisms."""
    dip_std: float = 0.0
    """Standard deviation of dip across accepted mechanisms."""
    rake_std: float = 0.0
    """Standard deviation of rake across accepted mechanisms."""


def _radiation_pattern(
    strike_rad: float,
    dip_rad: float,
    rake_rad: float,
    azimuth_rad: NDArray[np.floating],
    takeoff_rad: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute P-wave radiation pattern amplitude.

    Evaluates the far-field P-wave radiation pattern (Aki & Richards,
    eq. 4.88) for a double-couple source.

    Parameters
    ----------
    strike_rad, dip_rad, rake_rad : float
        Focal mechanism parameters in radians.
    azimuth_rad : NDArray
        Source-to-station azimuths in radians.
    takeoff_rad : NDArray
        Takeoff angles in radians (0=up, pi=down).

    Returns
    -------
    NDArray
        Radiation pattern amplitude at each station.
        Positive = compressional, negative = dilatational.
    """
    # Angles relative to strike
    phi = azimuth_rad - strike_rad

    sin_d = np.sin(dip_rad)
    cos_d = np.cos(dip_rad)
    sin_r = np.sin(rake_rad)
    cos_r = np.cos(rake_rad)
    sin_t = np.sin(takeoff_rad)
    cos_t = np.cos(takeoff_rad)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_2d = np.sin(2.0 * dip_rad)
    cos_2d = np.cos(2.0 * dip_rad)
    sin_2phi = np.sin(2.0 * phi)
    np.cos(2.0 * phi)

    # P-wave radiation pattern (Aki & Richards eq. 4.88)
    sin_2t = np.sin(2.0 * takeoff_rad)

    rad = (
        cos_r * sin_d * sin_t**2 * sin_2phi
        - cos_r * cos_d * sin_2t * cos_phi
        + sin_r * sin_2d * (cos_t**2 - sin_t**2 * sin_phi**2)
        + sin_r * cos_2d * sin_2t * sin_phi
    )

    return np.asarray(rad)


def _auxiliary_plane(strike: float, dip: float, rake: float) -> tuple[float, float, float]:
    """Compute the auxiliary nodal plane from the primary plane.

    Parameters
    ----------
    strike, dip, rake : float
        Primary plane parameters in degrees.

    Returns
    -------
    strike2, dip2, rake2 : float
        Auxiliary plane parameters in degrees.
    """
    s = np.radians(strike)
    d = np.radians(dip)
    r = np.radians(rake)

    sin_d = np.sin(d)
    cos_d = np.cos(d)
    sin_r = np.sin(r)
    cos_r = np.cos(r)

    # Dip of auxiliary plane
    dip2 = np.arccos(abs(sin_r) * sin_d)
    sin_d2 = np.sin(dip2)

    if abs(sin_d2) < 1e-10:
        # Degenerate case (vertical or horizontal)
        return float(np.degrees(s + np.pi)), 0.0, 0.0

    # Strike of auxiliary plane
    cos_s2_diff = (
        -1.0 / (np.tan(d) * np.tan(dip2)) if abs(np.tan(d) * np.tan(dip2)) > 1e-10 else 0.0
    )
    cos_s2_diff = np.clip(cos_s2_diff, -1.0, 1.0)
    s2_diff = np.arccos(cos_s2_diff)

    if rake > 0:
        strike2 = s + s2_diff
    else:
        strike2 = s - s2_diff

    # Rake of auxiliary plane
    sin_r2 = cos_d / sin_d2
    sin_r2 = np.clip(sin_r2, -1.0, 1.0)
    if cos_r < 0:
        rake2 = np.pi - np.arcsin(sin_r2)
    else:
        rake2 = np.arcsin(sin_r2)

    strike2 = float(np.degrees(strike2)) % 360.0
    dip2 = float(np.degrees(dip2))
    rake2 = float(np.degrees(rake2))

    # Normalize rake to [-180, 180)
    if rake2 >= 180.0:
        rake2 -= 360.0
    if rake2 < -180.0:
        rake2 += 360.0

    return strike2, dip2, rake2


def focal_mechanism_gridsearch(
    azimuths: NDArray[np.floating],
    takeoff_angles: NDArray[np.floating],
    polarities: NDArray[np.signedinteger] | NDArray[np.floating],
    *,
    strike_step: float = 5.0,
    dip_step: float = 5.0,
    rake_step: float = 5.0,
    max_misfit: float = 0.2,
    min_polarities: int = 8,
    weights: NDArray[np.floating] | None = None,
) -> FocalMechanismResult:
    """Find focal mechanisms consistent with observed P-wave polarities.

    Performs a grid search over strike, dip, and rake, evaluating the
    P-wave radiation pattern at each station and comparing the predicted
    polarity with the observed polarity.

    Parameters
    ----------
    azimuths : NDArray
        Source-to-station azimuths in degrees (clockwise from North).
    takeoff_angles : NDArray
        Takeoff angles in degrees (0=up, 90=horizontal, 180=down).
    polarities : NDArray
        Observed polarities: +1 (up/compressional), -1 (down/dilatational).
        Stations with polarity 0 are excluded.
    strike_step : float
        Grid step for strike in degrees.
    dip_step : float
        Grid step for dip in degrees.
    rake_step : float
        Grid step for rake in degrees.
    max_misfit : float
        Maximum allowed misfit fraction [0, 1] for accepting solutions.
    min_polarities : int
        Minimum number of non-zero polarity observations required.
    weights : NDArray, optional
        Weights for each polarity observation.  If provided, misfit is
        computed as a weighted fraction.

    Returns
    -------
    FocalMechanismResult
        All mechanisms within ``max_misfit``, the best solution, and
        the ``focal_data`` array for stress inversion.
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
            "Only %d valid polarities (need %d), cannot solve",
            n_pol,
            min_polarities,
        )
        return FocalMechanismResult(
            mechanisms=[],
            best=None,
            focal_data=np.empty((0, 3), dtype=np.float64),
            all_misfits=np.array([], dtype=np.float64),
        )

    # Convert to radians for computation
    az_rad = np.radians(az)
    to_rad = np.radians(to)

    # Build the grid
    strikes = np.arange(0.0, 360.0, strike_step)
    dips = np.arange(0.0, 90.0 + dip_step / 2, dip_step)
    rakes = np.arange(-180.0, 180.0, rake_step)

    total_weight = np.sum(w)
    all_misfits: list[float] = []
    accepted: list[tuple[float, float, float, float]] = []

    for strike in strikes:
        s_rad = np.radians(strike)
        for dip in dips:
            d_rad = np.radians(dip)
            for rake in rakes:
                r_rad = np.radians(rake)

                # Compute predicted polarities
                rad = _radiation_pattern(s_rad, d_rad, r_rad, az_rad, to_rad)
                predicted = np.sign(rad)

                # Misfit: weighted fraction of incorrect polarities
                mismatch = (predicted != pol).astype(np.float64)
                misfit = float(np.sum(w * mismatch) / total_weight)

                all_misfits.append(misfit)

                if misfit <= max_misfit:
                    accepted.append((strike, dip, rake, misfit))

    # Sort by misfit
    accepted.sort(key=lambda x: x[3])

    # Build FocalMechanism objects
    mechanisms: list[FocalMechanism] = []
    for strike, dip, rake, misfit in accepted:
        s2, d2, r2 = _auxiliary_plane(strike, dip, rake)
        mechanisms.append(
            FocalMechanism(
                strike=strike,
                dip=dip,
                rake=rake,
                strike2=s2,
                dip2=d2,
                rake2=r2,
                misfit=misfit,
                n_polarities=n_pol,
            )
        )

    best = mechanisms[0] if mechanisms else None

    if best is not None:
        focal_data = np.array([[best.strike, best.dip, best.rake]], dtype=np.float64)
    else:
        focal_data = np.empty((0, 3), dtype=np.float64)

    # Compute angular uncertainty from accepted mechanisms
    strike_std = 0.0
    dip_std = 0.0
    rake_std = 0.0
    if len(mechanisms) > 1:
        strikes_rad = np.radians([m.strike for m in mechanisms])
        strike_std = float(
            np.degrees(np.sqrt(-2.0 * np.log(max(abs(np.mean(np.exp(1j * strikes_rad))), 1e-30))))
        )
        dip_std = float(np.std([m.dip for m in mechanisms], ddof=1))
        rake_std = float(np.std([m.rake for m in mechanisms], ddof=1))

    return FocalMechanismResult(
        mechanisms=mechanisms,
        best=best,
        focal_data=focal_data,
        all_misfits=np.array(all_misfits, dtype=np.float64),
        strike_std=strike_std,
        dip_std=dip_std,
        rake_std=rake_std,
    )


def focal_mechanisms_batch(
    events: list[dict],
    *,
    strike_step: float = 5.0,
    dip_step: float = 5.0,
    rake_step: float = 5.0,
    max_misfit: float = 0.2,
    min_polarities: int = 8,
) -> tuple[list[FocalMechanismResult], NDArray[np.floating]]:
    """Solve focal mechanisms for multiple events.

    Parameters
    ----------
    events : list of dict
        Each dict must contain:
        - ``'azimuths'``: array of azimuths (degrees)
        - ``'takeoff_angles'``: array of takeoff angles (degrees)
        - ``'polarities'``: array of polarity observations
        - ``'weights'`` (optional): array of weights
    strike_step, dip_step, rake_step : float
        Grid step sizes in degrees.
    max_misfit : float
        Maximum allowed misfit fraction.
    min_polarities : int
        Minimum non-zero polarity observations.

    Returns
    -------
    results : list of FocalMechanismResult
        One result per event.
    focal_data : NDArray
        Stacked ``(N, 3)`` array of ``[strike, dip, rake]`` for all
        events that produced a solution.  Directly compatible with
        :func:`swe.stress.stress_inversion`.
    """
    results: list[FocalMechanismResult] = []
    focal_rows: list[NDArray[np.floating]] = []

    for i, event in enumerate(events):
        logger.info("Solving focal mechanism for event %d of %d", i + 1, len(events))

        result = focal_mechanism_gridsearch(
            azimuths=event["azimuths"],
            takeoff_angles=event["takeoff_angles"],
            polarities=event["polarities"],
            strike_step=strike_step,
            dip_step=dip_step,
            rake_step=rake_step,
            max_misfit=max_misfit,
            min_polarities=min_polarities,
            weights=event.get("weights"),
        )
        results.append(result)

        if result.focal_data.size > 0:
            focal_rows.append(result.focal_data)

    if focal_rows:
        focal_data = np.vstack(focal_rows)
    else:
        focal_data = np.empty((0, 3), dtype=np.float64)

    return results, focal_data


def focal_mechanism_uncertainty(result: FocalMechanismResult) -> dict:
    """Summarize angular uncertainty from a focal mechanism grid search.

    Parameters
    ----------
    result : FocalMechanismResult
        Result from :func:`focal_mechanism_gridsearch`.

    Returns
    -------
    dict
        Keys: ``strike_std``, ``dip_std``, ``rake_std``, ``n_accepted``.
    """
    return {
        "strike_std": result.strike_std,
        "dip_std": result.dip_std,
        "rake_std": result.rake_std,
        "n_accepted": len(result.mechanisms),
    }


def polarity_result_to_event_dict(
    polarity_result,
    azimuths: NDArray[np.floating],
    takeoffs: NDArray[np.floating],
    *,
    use_confidence_as_weights: bool = True,
) -> dict:
    """Bridge a :class:`~swe.focal.polarity.PolarityResult` to a grid-search event dict.

    Parameters
    ----------
    polarity_result : PolarityResult
        Output from :func:`~swe.focal.polarity.measure_polarities`.
    azimuths : NDArray
        Source-to-station azimuths (degrees), aligned with
        *polarity_result.observations*.
    takeoffs : NDArray
        Takeoff angles (degrees), aligned with *polarity_result.observations*.
    use_confidence_as_weights : bool
        If *True*, use polarity confidence values as weights.

    Returns
    -------
    dict
        Event dict compatible with :func:`focal_mechanism_gridsearch` and
        :func:`focal_mechanisms_batch`.
    """
    azimuths = np.asarray(azimuths, dtype=np.float64)
    takeoffs = np.asarray(takeoffs, dtype=np.float64)

    event: dict = {
        "azimuths": azimuths,
        "takeoff_angles": takeoffs,
        "polarities": polarity_result.polarities,
    }
    if use_confidence_as_weights:
        event["weights"] = polarity_result.confidences

    return event
