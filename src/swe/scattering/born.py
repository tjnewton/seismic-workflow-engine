"""Born single-scattering by a compact elastic inclusion in a homogeneous full space.

An incident plane P wave of unit displacement amplitude meets a small inclusion of volume
``V`` with perturbations ``(dvp, dvs, drho)`` of the host ``(vp, vs, density)``. To first order
in the perturbations, the inclusion radiates like a point force plus a moment tensor (the
"stress glut"). In the far field, at distance ``r`` and scattering angle ``theta`` (measured from
the incident propagation direction), the scattered amplitudes are

``P->P:  u_P = k_a**2 V F(q) / (4 pi r) * [ (drho/rho) cos(theta) - (dlam + 2 dmu cos(theta)**2) / (rho vp**2) ]``

``P->S:  u_S = k_b**2 V F(q) / (4 pi r) * [ (drho/rho) sin(theta) - 2 dmu sin(theta) cos(theta) / (rho vp vs) ]``

Here ``k_a = 2 pi f / vp``, ``k_b = 2 pi f / vs``, and ``dlam, dmu`` are the first-order Lamé
perturbations implied by ``(dvp, dvs, drho)``. ``F(q)`` is the inclusion's form factor at
scattering wavenumber ``q`` (``F = 1`` for a point scatterer). These are the standard Born
radiation patterns (Wu & Aki 1985, Geophysics 50, 582-595). In the fluid limit (``vs = 0``) the
P->P pattern reduces to Rayleigh scattering by a small fluid sphere at first order in contrast.

Validity flags: Rayleigh regime ``k a < 0.5`` (``a`` the volume-equivalent radius), weak
contrast (every relative perturbation below 0.3), far field ``k r > 10`` *and* beyond the
Fraunhofer distance ``2 D**2 / wavelength`` (``D = 2a``), where a plane-wave form factor
holds. Outside them the Born amplitude is an estimate, not a bound, and should be checked
numerically.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

from swe.utils.helpers import is_integer

logger = logging.getLogger(__name__)

RAYLEIGH_KA = 0.5
WEAK_CONTRAST = 0.3
FAR_FIELD_KR = 10.0
_CHUNK_BYTES = 64 * 1024 * 1024
_MODES = ("PP", "PS")


class BornResult(NamedTuple):
    """Born far-field scattered amplitude and its validity flags."""

    amplitude: float | complex
    """Scattered / incident displacement amplitude: complex if the form factor is complex, else signed real."""
    ka: float
    """Wavenumber of the scattered wave times the volume-equivalent radius."""
    kr: float
    """Wavenumber of the scattered wave times the distance."""
    rayleigh: bool
    """True when ``ka`` is below 0.5."""
    weak_contrast: bool
    """True when every relative perturbation is below 0.3 in magnitude."""
    far_field: bool
    """True when ``kr`` exceeds 10 and the distance exceeds the body's Fraunhofer distance."""


def _lame_perturbations(vp, vs, density, dvp, dvs, drho):
    """First-order (dlam, dmu) from velocity and density perturbations."""
    dmu = drho * vs**2 + 2.0 * density * vs * dvs
    dlam = drho * (vp**2 - 2.0 * vs**2) + 2.0 * density * (vp * dvp - 2.0 * vs * dvs)
    return dlam, dmu


def born_radiation_pattern(
    theta: float | np.ndarray,
    *,
    vp: float,
    vs: float,
    density: float,
    dvp: float = 0.0,
    dvs: float = 0.0,
    drho: float = 0.0,
    mode: str = "PP",
) -> float | np.ndarray:
    """Dimensionless Born radiation pattern (the bracketed factor in the module docstring).

    Parameters
    ----------
    theta : float or ndarray
        Scattering angle(s) in radians, from the incident propagation direction.
    vp, vs, density : float
        Host P and S velocity and density (any consistent units). ``vs`` may be 0
        (fluid host) for ``mode="PP"``.
    dvp, dvs, drho : float
        Absolute perturbations inside the inclusion.
    mode : {"PP", "PS"}
        Incident P to scattered P, or incident P to scattered S.

    Returns
    -------
    float or ndarray
        Pattern value(s).

    Raises
    ------
    ValueError
        For an unknown mode, non-positive ``vp`` or ``density``, ``mode="PS"`` with
        ``vs <= 0``, or a shear perturbation in a fluid host (``vs = 0``, ``dvs != 0``).
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if not (vp > 0 and density > 0 and vs >= 0) or not np.isfinite([vp, vs, density]).all():
        raise ValueError("vp and density must be positive and finite, vs non-negative and finite")
    if vs > 0 and vp**2 <= 4.0 / 3.0 * vs**2:
        raise ValueError(
            f"impossible host: vp={vp}, vs={vs} gives a non-positive bulk modulus (need vp > 1.155 vs)"
        )
    if vs == 0 and dvs != 0:
        raise ValueError("a shear perturbation (dvs) is undefined in a fluid host (vs = 0)")
    theta = np.asarray(theta, dtype=float)
    dlam, dmu = _lame_perturbations(vp, vs, density, dvp, dvs, drho)
    c = np.cos(theta)
    if mode == "PP":
        out = (drho / density) * c - (dlam + 2.0 * dmu * c**2) / (density * vp**2)
    else:
        if vs <= 0:
            raise ValueError("mode='PS' needs vs > 0")
        s = np.sin(theta)
        out = (drho / density) * s - 2.0 * dmu * s * c / (density * vp * vs)
    return float(out) if out.ndim == 0 else out


def scattering_wavenumber(
    theta: float | np.ndarray, *, k_in: float, k_out: float
) -> float | np.ndarray:
    """Magnitude of the scattering vector ``q = k_out r_hat - k_in e_hat`` for angle *theta*."""
    q = np.sqrt(np.maximum(k_in**2 + k_out**2 - 2.0 * k_in * k_out * np.cos(theta), 0.0))
    return float(q) if np.ndim(q) == 0 else q


def sphere_form_factor(q: float | np.ndarray, radius: float) -> float | np.ndarray:
    """Born form factor of a uniform sphere: ``3 (sin x - x cos x) / x**3`` with ``x = q radius``."""
    x = np.asarray(q, dtype=float) * radius
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1.0, x)
    f = np.where(small, 1.0 - x**2 / 10.0, 3.0 * (np.sin(xs) - xs * np.cos(xs)) / xs**3)
    return float(f) if f.ndim == 0 else f


def voxel_form_factor(
    inside: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    *,
    bounds: Sequence[tuple[float, float]],
    q_vector: Sequence[float] | np.ndarray,
    n: int = 48,
) -> tuple[complex | np.ndarray, float]:
    """Form factor and volume of an arbitrary body by voxel quadrature.

    Uses the Born convention of :func:`scattering_wavenumber`: ``q = k_out r_hat - k_in e_hat``
    and ``F(q) = (1/V) integral exp(-i q . x) dV``.

    Parameters
    ----------
    inside : callable
        Vectorized ``inside(x, y, z) -> bool array``; True for points in the body. It is
        called once, on broadcastable (sparse) coordinate grids.
    bounds : sequence of (min, max)
        Box enclosing the body, one pair per axis (x, y, z).
    q_vector : sequence of float or ndarray
        One scattering vector (3 components) or an ``(m, 3)`` array of them (reciprocal
        length units). The body is voxelized once for all of them.
    n : int
        Voxels per axis.

    Returns
    -------
    (complex or ndarray, float)
        ``F`` (a complex scalar, or an array of length m) and the volume ``V``. Center the
        body near the origin: the phase of ``F`` depends on the reference point, its
        magnitude does not.

    Raises
    ------
    ValueError
        If no voxel center falls inside the body, the body touches the edge of *bounds*
        (it would be clipped), *q_vector* has the wrong shape or is empty (checked before
        voxelizing), ``n < 3``, or a bound pair is not increasing.
    """
    q = np.asarray(q_vector, dtype=float)
    single = q.ndim == 1
    q = np.atleast_2d(q)
    if q.ndim != 2 or q.shape[1] != 3 or q.shape[0] == 0:
        raise ValueError(
            f"q_vector must be (3,) or a non-empty (m, 3) array, got shape {np.shape(q_vector)}"
        )
    if not is_integer(n) or n < 3:
        raise ValueError(f"n must be an integer of at least 3 voxels per axis, got {n!r}")
    if len(bounds) != 3 or any(np.shape(b) != (2,) or not b[0] < b[1] for b in bounds):
        raise ValueError(f"bounds must be three (min, max) pairs with min < max, got {bounds}")
    axes = [np.linspace(lo, hi, n, endpoint=False) + (hi - lo) / (2 * n) for lo, hi in bounds]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij", sparse=True)
    mask = np.broadcast_to(np.asarray(inside(gx, gy, gz), dtype=bool), (n, n, n))
    if not mask.any():
        raise ValueError("no voxel center lies inside the body; check bounds or increase n")
    faces = (mask[0], mask[-1], mask[:, 0], mask[:, -1], mask[:, :, 0], mask[:, :, -1])
    if any(f.any() for f in faces):
        raise ValueError("the body reaches the edge of bounds and would be clipped; widen bounds")
    ix, iy, iz = np.nonzero(mask)
    pts = np.column_stack([axes[0][ix], axes[1][iy], axes[2][iz]])
    dv = float(np.prod([(hi - lo) / n for lo, hi in bounds]))
    volume = float(len(pts) * dv)
    # bound peak memory to about _CHUNK_BYTES: per chunk the phases and one cos or sin temporary
    # coexist, each (voxels x chunk) float64
    chunk = max(1, _CHUNK_BYTES // (2 * 8 * len(pts)))
    parts = []
    for i in range(0, len(q), chunk):
        ph = pts @ q[i : i + chunk].T
        parts.append(np.cos(ph).mean(axis=0) - 1j * np.sin(ph).mean(axis=0))
    f = np.concatenate(parts)
    return (complex(f[0]) if single else f), volume


def _fraunhofer_distance(k: float, radius: float) -> float:
    """Fraunhofer distance 2 D**2 / wavelength for a body of diameter D = 2 radius."""
    return 4.0 * k * radius**2 / np.pi


def born_point_scattering(
    *,
    frequency: float,
    volume: float,
    distance: float,
    theta: float,
    vp: float,
    vs: float,
    density: float,
    dvp: float = 0.0,
    dvs: float = 0.0,
    drho: float = 0.0,
    mode: str = "PP",
    form_factor: float | complex = 1.0,
) -> BornResult:
    """Far-field Born amplitude of the wave scattered by a compact inclusion.

    Parameters
    ----------
    frequency : float
        Frequency (Hz).
    volume : float
        Inclusion volume (length units cubed, consistent with *distance* and the velocities).
    distance : float
        Scatterer-to-receiver distance.
    theta : float
        Scattering angle (radians) from the incident propagation direction.
    vp, vs, density, dvp, dvs, drho : float
        Host properties and perturbations (see :func:`born_radiation_pattern`).
    mode : {"PP", "PS"}
        Scattered wave type.
    form_factor : float or complex
        ``F(q)`` of the body (see :func:`sphere_form_factor`, :func:`voxel_form_factor`);
        1 for a point scatterer. It multiplies the amplitude as given, so a negative real
        ``F`` flips the sign and a complex ``F`` makes the amplitude complex. Evaluate it at
        ``q = scattering_wavenumber(theta, k_in=2 pi f / vp, k_out=2 pi f / v_out)``, with
        ``v_out = vp`` for ``mode="PP"`` and ``v_out = vs`` for ``mode="PS"``.

    Returns
    -------
    BornResult
        Amplitude ratio and validity flags.

    Raises
    ------
    ValueError
        For non-positive or non-finite frequency, volume or distance, a non-scalar *theta*
        (and see :func:`born_radiation_pattern`).
    """
    if not all(np.isfinite(x) and x > 0 for x in (frequency, volume, distance)):
        raise ValueError("frequency, volume and distance must be positive and finite")
    if np.ndim(theta) != 0:
        raise ValueError(
            "theta must be a scalar; loop or use born_radiation_pattern for arrays of angles"
        )
    pattern = born_radiation_pattern(
        theta, vp=vp, vs=vs, density=density, dvp=dvp, dvs=dvs, drho=drho, mode=mode
    )
    k = 2.0 * np.pi * frequency / (vp if mode == "PP" else vs)
    amp = k**2 * volume * form_factor / (4.0 * np.pi * distance) * float(pattern)
    radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    contrasts = [abs(dvp) / vp, abs(drho) / density] + ([abs(dvs) / vs] if vs > 0 else [])
    result = BornResult(
        amplitude=complex(amp) if np.iscomplexobj(form_factor) else float(np.real(amp)),
        ka=float(k * radius),
        kr=float(k * distance),
        rayleigh=bool(k * radius < RAYLEIGH_KA),
        weak_contrast=bool(max(contrasts) < WEAK_CONTRAST),
        far_field=bool(k * distance > FAR_FIELD_KR and distance > _fraunhofer_distance(k, radius)),
    )
    if not (result.rayleigh and result.weak_contrast and result.far_field):
        logger.info("Born estimate outside its validity regime: %s", result)
    return result
