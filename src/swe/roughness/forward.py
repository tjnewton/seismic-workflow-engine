"""Forward models for frequency prediction from surface roughness.

Provides a spectral-domain asperity contact model and a structural
resonance alternative.  Both are lightweight, hypothesis-agnostic
tools for testing frequency predictions against observations.

References
----------
Persson, B. N. J. (2006). Contact mechanics for randomly rough
    surfaces. Surface Science Reports, 61, 201-227.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from swe.roughness.surface import RoughnessPSDResult

logger = logging.getLogger(__name__)

__all__ = [
    "ForwardModelResult",
    "forward_model_frequency",
    "resonance_frequency",
]


class ForwardModelResult(NamedTuple):
    """Result of forward frequency prediction.

    Attributes
    ----------
    predicted_frequency : float
        Predicted dominant frequency (Hz).
    observed_frequency : float or None
        Observed frequency (Hz) if provided.
    frequency_ratio : float
        Ratio of predicted to observed frequency, or ``NaN``.
    roughness_psd : ndarray
        Roughness PSD used in the model.
    slip_velocity : float
        Slip velocity used (m/s).
    contact_stiffness : float
        Effective contact stiffness parameter (Pa/m).
    stress_spectrum : ndarray
        Predicted stress perturbation spectrum.
    stress_frequencies : ndarray
        Temporal frequencies corresponding to the stress spectrum (Hz).
    model_type : str
        Label for the contact model used.
    """

    predicted_frequency: float
    observed_frequency: float | None
    frequency_ratio: float
    roughness_psd: NDArray[np.floating]
    slip_velocity: float
    contact_stiffness: float
    stress_spectrum: NDArray[np.floating]
    stress_frequencies: NDArray[np.floating]
    model_type: str


def forward_model_frequency(
    roughness_psd: RoughnessPSDResult,
    slip_velocity: float,
    *,
    normal_stress: float = 1e6,
    shear_modulus: float = 1e9,
    poisson_ratio: float = 0.25,
    contact_model: str = "elastic_halfspace",
    observed_frequency: float | None = None,
) -> ForwardModelResult:
    """Predict dominant radiated frequency from roughness and slip velocity.

    Computes the stress perturbation spectrum as the product of the
    roughness PSD and an elastic Green's function, then converts
    spatial wavenumber to temporal frequency via f = V * k / (2 pi).

    Parameters
    ----------
    roughness_psd : RoughnessPSDResult
        Surface roughness PSD.
    slip_velocity : float
        Slip velocity (m/s).
    normal_stress : float
        Normal stress on the interface (Pa).
    shear_modulus : float
        Shear modulus of the material (Pa).
    poisson_ratio : float
        Poisson's ratio.
    contact_model : str
        Contact mechanics model: ``"elastic_halfspace"`` or ``"hertzian"``.
    observed_frequency : float, optional
        If provided, computes the predicted/observed ratio.

    Returns
    -------
    ForwardModelResult
    """
    valid_models = ("elastic_halfspace", "hertzian")
    if contact_model not in valid_models:
        raise ValueError(f"contact_model must be one of {valid_models}, got {contact_model!r}")
    if slip_velocity <= 0:
        raise ValueError(f"slip_velocity must be positive, got {slip_velocity}")

    k = roughness_psd.wavenumbers
    psd = roughness_psd.psd

    # Elastic Green's function: G(k) = k * E* / 2
    # where E* = E / (1 - nu^2), E = 2 * mu * (1 + nu)
    youngs = 2.0 * shear_modulus * (1.0 + poisson_ratio)
    e_star = youngs / (1.0 - poisson_ratio**2)

    if contact_model == "elastic_halfspace":
        # Stress perturbation: sigma(k) ~ PSD(k) * k * E*
        green = k * e_star
    else:
        # Hertzian: sigma(k) ~ PSD(k) * k^(3/2) * E*
        green = k**1.5 * e_star

    stress_spectrum = psd * green
    contact_stiffness = float(e_star)

    # Convert spatial wavenumber to temporal frequency: f = V * k
    temporal_freq = slip_velocity * k

    # Peak stress perturbation frequency
    if len(stress_spectrum) > 0 and np.any(stress_spectrum > 0):
        peak_idx = np.argmax(stress_spectrum)
        predicted_freq = float(temporal_freq[peak_idx])
    else:
        predicted_freq = np.nan

    # Frequency ratio
    if observed_frequency is not None and observed_frequency > 0:
        ratio = predicted_freq / observed_frequency if not np.isnan(predicted_freq) else np.nan
    else:
        ratio = np.nan

    logger.info(
        "Forward model (%s): predicted f=%.3f Hz, V=%.3e m/s",
        contact_model,
        predicted_freq,
        slip_velocity,
    )

    return ForwardModelResult(
        predicted_frequency=predicted_freq,
        observed_frequency=observed_frequency,
        frequency_ratio=ratio,
        roughness_psd=psd,
        slip_velocity=slip_velocity,
        contact_stiffness=contact_stiffness,
        stress_spectrum=stress_spectrum,
        stress_frequencies=temporal_freq,
        model_type=contact_model,
    )


def resonance_frequency(
    thickness: float,
    velocity: float,
    *,
    mode: int = 1,
) -> float:
    """Predict structural resonance frequency for a layer.

    Computes f = n * V_s / (2 * H) for a layer of thickness H
    with shear-wave velocity V_s, where n is the mode number.

    Parameters
    ----------
    thickness : float
        Layer thickness (m).  Must be positive.
    velocity : float
        Shear-wave velocity (m/s).  Must be positive.
    mode : int
        Mode number (1 = fundamental).  Must be positive.

    Returns
    -------
    float
        Resonance frequency (Hz).

    Raises
    ------
    ValueError
        If any input is non-positive.
    """
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    if velocity <= 0:
        raise ValueError(f"velocity must be positive, got {velocity}")
    if mode < 1:
        raise ValueError(f"mode must be >= 1, got {mode}")

    freq = mode * velocity / (2.0 * thickness)
    logger.info(
        "Resonance frequency: f=%.2f Hz (H=%.1f m, Vs=%.0f m/s, mode=%d)",
        freq,
        thickness,
        velocity,
        mode,
    )
    return freq
