"""Monte Carlo uncertainty propagation from picks to locations.

Perturbs pick times by Gaussian noise based on their
:attr:`~swe.picks.SWEPick.time_uncertainty_s`, relocates multiple
times, and computes scatter-cloud statistics.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from swe.uncertainty.types import LocationUncertainty

logger = logging.getLogger(__name__)

__all__ = [
    "hypocenters_to_location_uncertainty",
    "monte_carlo_location_uncertainty",
]


def monte_carlo_location_uncertainty(
    pick_times: NDArray[np.floating],
    pick_uncertainties: NDArray[np.floating],
    locate_fn: Callable[..., Any],
    locate_kwargs: dict[str, Any],
    *,
    n_samples: int = 100,
    random_state: int = 0,
) -> LocationUncertainty:
    """Estimate location uncertainty via Monte Carlo pick-time perturbation.

    For each realisation, Gaussian noise with sigma equal to each pick's
    ``time_uncertainty_s`` is added to the observed times, and the
    location function is re-run.  The resulting scatter cloud is
    summarised as a :class:`~swe.uncertainty.types.LocationUncertainty`.

    Parameters
    ----------
    pick_times : ndarray, shape (n_picks,)
        Observed pick arrival times in seconds (or any unit consistent
        with the locate function).
    pick_uncertainties : ndarray, shape (n_picks,)
        1-sigma pick-time uncertainties in the same units.
    locate_fn : callable
        Location function.  Must accept ``observed_times`` as first
        positional argument (or keyword) and return an object with
        ``x``, ``y``, ``z`` attributes.
    locate_kwargs : dict
        Additional keyword arguments forwarded to *locate_fn*.  The
        ``observed_times`` key is overwritten in each realisation.
    n_samples : int
        Number of Monte Carlo realisations.
    random_state : int
        Seed for the random number generator.

    Returns
    -------
    LocationUncertainty
        Summary of the scatter cloud.
    """
    rng = np.random.default_rng(random_state)
    pick_times = np.asarray(pick_times, dtype=np.float64)
    pick_uncertainties = np.asarray(pick_uncertainties, dtype=np.float64)

    if pick_times.shape != pick_uncertainties.shape:
        raise ValueError(
            f"pick_times shape {pick_times.shape} does not match "
            f"pick_uncertainties shape {pick_uncertainties.shape}"
        )

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    for _ in range(n_samples):
        noise = rng.normal(0.0, pick_uncertainties)
        perturbed = pick_times + noise
        kwargs = {**locate_kwargs, "observed_times": perturbed}
        try:
            result = locate_fn(**kwargs)
            xs.append(float(result.x))
            ys.append(float(result.y))
            zs.append(float(result.z))
        except Exception:
            logger.debug("Monte Carlo relocation failed, skipping sample")
            continue

    if len(xs) < 2:
        logger.warning(
            "Monte Carlo location: only %d successful relocations (need >= 2 for uncertainty)",
            len(xs),
        )
        return LocationUncertainty(method="monte_carlo", n_samples=len(xs))

    coords = np.column_stack([xs, ys, zs])
    cov = np.cov(coords, rowvar=False)

    horiz_uncertainty = float(np.sqrt(cov[0, 0] + cov[1, 1]))
    depth_uncertainty = float(np.sqrt(cov[2, 2]))

    return LocationUncertainty(
        horiz_uncertainty_m=horiz_uncertainty,
        depth_uncertainty_m=depth_uncertainty,
        covariance_matrix=cov,
        n_samples=len(xs),
        method="monte_carlo",
    )


def hypocenters_to_location_uncertainty(
    hypocenters: list[Any],
) -> LocationUncertainty:
    """Convert NonLinLoc HypocenterResult list to a LocationUncertainty.

    Takes the mean and scatter of NLL's per-event uncertainty fields.

    Parameters
    ----------
    hypocenters : list of HypocenterResult
        Parsed hypocenters from :func:`~swe.location.nonlinloc.process_nll_hypocenters`.

    Returns
    -------
    LocationUncertainty
        Aggregated uncertainty.
    """
    if not hypocenters:
        return LocationUncertainty(method="nll_scatter")

    horiz_vals = [h.horiz_uncertainty_m for h in hypocenters]
    depth_vals = [h.depth_uncertainty_m for h in hypocenters]
    rms_vals = [h.rms for h in hypocenters]

    return LocationUncertainty(
        horiz_uncertainty_m=float(np.mean(horiz_vals)),
        depth_uncertainty_m=float(np.mean(depth_vals)),
        rms=float(np.mean(rms_vals)),
        n_samples=len(hypocenters),
        method="nll_scatter",
    )
