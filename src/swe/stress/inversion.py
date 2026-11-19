"""Stress tensor inversion from focal mechanism data.

Provides functions for K-means clustering of focal mechanisms and
stress inversion using the Vavrycuk adaptation of Michael's method.

References
----------
Michael, A.J. (1984). Determination of stress from slip data: faults
    and folds. J. Geophys. Res., 89(B13), 11517-11526.
Vavrycuk, V. (2014). Iterative joint inversion for stress and fault
    orientations from focal mechanisms. Geophys. J. Int., 199(1), 69-77.
    doi:10.1093/gji/ggu224
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.linalg import lstsq
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class StressInversionResult(NamedTuple):
    """Result from stress inversion on clustered focal mechanisms."""

    results: list[dict]
    """Inversion result dictionaries, one per cluster."""
    axes: list[dict]
    """Stress axes orientations, one per cluster."""
    bootstrap_axes: list[dict] | None = None
    """Per-cluster bootstrap statistics, or ``None`` if not computed."""


class ClusterResult(NamedTuple):
    """Result from K-means clustering of focal mechanisms."""

    labels: np.ndarray
    """Cluster label for each sample."""
    centers: np.ndarray
    """Cluster center coordinates."""


# ---------------------------------------------------------------------------
# Vavrycuk (2014) stress inversion
# ---------------------------------------------------------------------------


def _deg2rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * np.pi / 180.0


def _strike_dip_rake_to_fault_slip(
    strike: float,
    dip: float,
    rake: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert strike/dip/rake to fault normal and slip vectors for both nodal planes.

    Uses the Aki & Richards convention.

    Parameters
    ----------
    strike, dip, rake : float
        In degrees.

    Returns
    -------
    n1 : numpy.ndarray
        Fault normal for nodal plane 1, shape ``(3,)``.
    d1 : numpy.ndarray
        Slip direction for nodal plane 1, shape ``(3,)``.
    n2 : numpy.ndarray
        Fault normal for nodal plane 2, shape ``(3,)``.
    d2 : numpy.ndarray
        Slip direction for nodal plane 2, shape ``(3,)``.
    """
    s = _deg2rad(strike)
    d = _deg2rad(dip)
    r = _deg2rad(rake)

    # Nodal plane 1: fault normal
    n1 = np.array(
        [
            -np.sin(d) * np.sin(s),
            np.sin(d) * np.cos(s),
            -np.cos(d),
        ]
    )

    # Nodal plane 1: slip direction
    d1 = np.array(
        [
            np.cos(r) * np.cos(s) + np.sin(r) * np.cos(d) * np.sin(s),
            np.cos(r) * np.sin(s) - np.sin(r) * np.cos(d) * np.cos(s),
            -np.sin(r) * np.sin(d),
        ]
    )

    # Nodal plane 2 (auxiliary): normal is the slip direction of plane 1,
    # and slip direction is the normal of plane 1
    n2 = d1.copy()
    d2 = n1.copy()

    return n1, d1, n2, d2


def _build_design_matrix(
    normals: np.ndarray,
) -> np.ndarray:
    """Build the design matrix for Michael's (1984) linear inversion.

    For each fault with normal *n*, the relationship between the stress
    tensor and the resolved shear traction yields a row in the design
    matrix relating 5 independent stress-tensor components to the 3
    components of the slip vector.

    Parameters
    ----------
    normals : numpy.ndarray
        Fault normals, shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        Design matrix, shape ``(3*N, 5)``.

    Notes
    -----
    We parameterize the deviatoric stress tensor with 5 independent
    components: ``[s11, s12, s13, s22, s23]`` (with
    ``s33 = -(s11 + s22)``).  For each mechanism *k*, three equations
    relate the stress to the expected slip direction on the fault.
    """
    n_mechs = normals.shape[0]
    A = np.zeros((3 * n_mechs, 5))
    for k in range(n_mechs):
        n = normals[k]
        _nx, _ny, _nz = n[0], n[1], n[2]
        row = 3 * k

        # Traction T_i = sum_j sigma_ij * n_j
        # Resolved shear: tau_i = T_i - (T . n) * n_i
        # With sigma parameterized as 5 components:
        #   sigma = [[s11, s12, s13],
        #            [s12, s22, s23],
        #            [s13, s23, -(s11+s22)]]

        # T_x = s11*nx + s12*ny + s13*nz
        # T_y = s12*nx + s22*ny + s23*nz
        # T_z = s13*nx + s23*ny - (s11+s22)*nz

        # Normal traction: Tn = T.n = T_x*nx + T_y*ny + T_z*nz
        # Shear: tau_x = T_x - Tn*nx, etc.

        # For each of the 5 params, compute the contribution to tau:
        for p in range(5):
            # Build a sigma matrix with only param p = 1
            sig = np.zeros((3, 3))
            if p == 0:  # s11
                sig[0, 0] = 1.0
                sig[2, 2] = -1.0
            elif p == 1:  # s12
                sig[0, 1] = 1.0
                sig[1, 0] = 1.0
            elif p == 2:  # s13
                sig[0, 2] = 1.0
                sig[2, 0] = 1.0
            elif p == 3:  # s22
                sig[1, 1] = 1.0
                sig[2, 2] = -1.0
            elif p == 4:  # s23
                sig[1, 2] = 1.0
                sig[2, 1] = 1.0

            T = sig @ n
            Tn = T @ n
            tau = T - Tn * n

            A[row, p] = tau[0]
            A[row + 1, p] = tau[1]
            A[row + 2, p] = tau[2]

    return A


def _stress_from_params(params: np.ndarray) -> np.ndarray:
    """Reconstruct 3x3 stress tensor from 5 deviatoric parameters.

    Parameters
    ----------
    params : numpy.ndarray
        ``[s11, s12, s13, s22, s23]``.

    Returns
    -------
    numpy.ndarray
        Symmetric stress tensor, shape ``(3, 3)``.
    """
    s11, s12, s13, s22, s23 = params
    s33 = -(s11 + s22)
    return np.array(
        [
            [s11, s12, s13],
            [s12, s22, s23],
            [s13, s23, s33],
        ]
    )


def _principal_stresses(
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Decompose stress tensor into principal stresses and stress ratio.

    Parameters
    ----------
    sigma : numpy.ndarray
        Symmetric stress tensor, shape ``(3, 3)``.

    Returns
    -------
    eigenvalues : numpy.ndarray
        Principal stresses ``[sigma_1, sigma_2, sigma_3]`` sorted so
        that ``sigma_1 >= sigma_2 >= sigma_3`` (most compressive first).
    eigenvectors : numpy.ndarray
        Columns are principal stress directions, shape ``(3, 3)``.
    stress_ratio : float
        ``R = (sigma_1 - sigma_2) / (sigma_1 - sigma_3)``.
    """
    vals, vecs = np.linalg.eigh(sigma)
    # eigh returns ascending; we want descending (sigma_1 >= sigma_2 >= sigma_3)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    denom = vals[0] - vals[2]
    stress_ratio = (vals[0] - vals[1]) / denom if abs(denom) > 1e-12 else 0.0

    return vals, vecs, stress_ratio


def _vector_to_azimuth_plunge(v: np.ndarray) -> tuple[float, float]:
    """Convert a 3-D direction vector to azimuth and plunge (degrees).

    Convention: x=North, y=East, z=Down.

    Parameters
    ----------
    v : numpy.ndarray
        Direction vector, shape ``(3,)``.

    Returns
    -------
    azimuth : float
        Azimuth in degrees [0, 360).
    plunge : float
        Plunge in degrees [0, 90].
    """
    # Ensure downward-pointing convention
    if v[2] < 0:
        v = -v

    length = np.linalg.norm(v)
    if length < 1e-12:
        return 0.0, 0.0

    v = v / length
    plunge = np.degrees(np.arcsin(v[2]))
    azimuth = np.degrees(np.arctan2(v[1], v[0]))
    if azimuth < 0:
        azimuth += 360.0

    return float(azimuth), float(plunge)


def _select_fault_planes(
    focal_mechanisms: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the fault plane (of two nodal planes) most consistent with stress.

    For each mechanism, the nodal plane whose predicted slip direction
    is closest to the observed slip direction (given the current stress
    tensor) is selected.

    Parameters
    ----------
    focal_mechanisms : numpy.ndarray
        ``(N, 3+)`` with columns ``[strike, dip, rake, ...]``.
    sigma : numpy.ndarray
        Current stress tensor estimate, shape ``(3, 3)``.

    Returns
    -------
    normals : numpy.ndarray
        Selected fault normals, shape ``(N, 3)``.
    slips : numpy.ndarray
        Corresponding slip directions, shape ``(N, 3)``.
    """
    n = focal_mechanisms.shape[0]
    normals = np.empty((n, 3))
    slips = np.empty((n, 3))

    for i in range(n):
        strike, dip, rake = focal_mechanisms[i, :3]
        n1, d1, n2, d2 = _strike_dip_rake_to_fault_slip(strike, dip, rake)

        best_misfit = np.inf
        best_n = n1
        best_d = d1

        for nk, dk in [(n1, d1), (n2, d2)]:
            # Predicted traction on the fault
            T = sigma @ nk
            Tn = T @ nk
            tau = T - Tn * nk
            tau_norm = np.linalg.norm(tau)
            if tau_norm > 1e-12:
                tau_hat = tau / tau_norm
            else:
                tau_hat = tau

            # Misfit = angle between predicted shear and observed slip
            cos_angle = np.clip(tau_hat @ dk, -1.0, 1.0)
            misfit = np.arccos(abs(cos_angle))

            if misfit < best_misfit:
                best_misfit = misfit
                best_n = nk
                best_d = dk

        normals[i] = best_n
        slips[i] = best_d

    return normals, slips


def _invert(focal_mechanisms: np.ndarray) -> tuple[dict, dict]:
    """Perform Vavrycuk (2014) iterative stress inversion on focal mechanisms.

    Parameters
    ----------
    focal_mechanisms : numpy.ndarray
        Array of shape ``(N, 3+)`` with at minimum columns
        ``[strike, dip, rake]`` per mechanism (degrees).

    Returns
    -------
    inversion_results : dict
        Dictionary with keys ``sigma_1``, ``sigma_2``, ``sigma_3``,
        each containing ``azimuth`` and ``plunge`` (degrees).
    stress_axes : dict
        Additional information: ``eigenvectors``, ``eigenvalues``,
        ``stress_ratio`` (R), ``n_iterations``.
    """
    focal_mechanisms = np.atleast_2d(focal_mechanisms)
    n_mechs = focal_mechanisms.shape[0]

    if n_mechs < 1:
        raise ValueError("At least one focal mechanism is required.")

    # Initial fault-plane selection: use nodal plane 1 for all mechanisms
    normals = np.empty((n_mechs, 3))
    slips = np.empty((n_mechs, 3))
    for i in range(n_mechs):
        strike, dip, rake = focal_mechanisms[i, :3]
        n1, d1, _, _ = _strike_dip_rake_to_fault_slip(strike, dip, rake)
        normals[i] = n1
        slips[i] = d1

    max_iter = 20
    sigma = None

    for iteration in range(max_iter):
        # Build design matrix from current normals
        A = _build_design_matrix(normals)

        # Build observation vector from slip directions
        b = slips.ravel()

        # Solve least-squares: A @ params ≈ b
        params, _, _, _ = lstsq(A, b)

        sigma = _stress_from_params(params)

        # Select fault planes based on updated stress
        new_normals, new_slips = _select_fault_planes(focal_mechanisms, sigma)

        # Check convergence: fault-plane selection unchanged
        if np.allclose(new_normals, normals, atol=1e-8):
            logger.debug(
                "Vavrycuk inversion converged after %d iterations",
                iteration + 1,
            )
            normals = new_normals
            slips = new_slips
            break

        normals = new_normals
        slips = new_slips
    else:
        logger.warning(
            "Vavrycuk inversion did not converge after %d iterations",
            max_iter,
        )

    # Decompose final stress tensor
    assert sigma is not None
    eigenvalues, eigenvectors, stress_ratio = _principal_stresses(sigma)

    # Convert principal axes to azimuth/plunge
    axes_info = {}
    labels = ["sigma_1", "sigma_2", "sigma_3"]
    inversion_results = {}
    for j, label in enumerate(labels):
        az, pl = _vector_to_azimuth_plunge(eigenvectors[:, j])
        inversion_results[label] = {"azimuth": az, "plunge": pl}
        axes_info[label] = eigenvectors[:, j].tolist()

    stress_axes = {
        "eigenvectors": eigenvectors.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "stress_ratio": stress_ratio,
        "n_iterations": iteration + 1,
        "principal_axes": axes_info,
    }

    return inversion_results, stress_axes


def stress_inversion(
    data: np.ndarray | list,
    labels: np.ndarray,
) -> StressInversionResult:
    """Perform a 1-D stress inversion on clustered focal mechanism data.

    Runs a separate inversion for each cluster identified by *labels*.
    Uses the Vavrycuk adaptation of Michael's method.  No damping is
    implemented yet.

    Parameters
    ----------
    data : numpy.ndarray or list of MomentTensor
        Focal mechanism data.  Accepts either:

        - An ``(N, M)`` array with at least columns
          ``[strike, dip, rake]`` (degrees).
        - A list of :class:`~swe.moment_tensor.tensor.MomentTensor`
          objects, which are automatically converted to strike/dip/rake.
    labels : numpy.ndarray
        Cluster label for each event (from :func:`cluster_focal_mechanisms`
        or similar).

    Returns
    -------
    results : list of dict
        Inversion result dictionaries, one per cluster.  Each dict has keys
        ``sigma_1``, ``sigma_2``, ``sigma_3`` with ``azimuth`` and
        ``plunge`` sub-keys.
    axes : list of dict
        Stress axes orientations, one per cluster.
    """
    # Accept list[MomentTensor] - convert to (N, 3) SDR array
    if isinstance(data, list):
        from swe.moment_tensor.tensor import MomentTensor, tensor_to_strike_dip_rake

        rows = []
        for item in data:
            if isinstance(item, MomentTensor):
                s, d, r = tensor_to_strike_dip_rake(item)
                rows.append([s, d, r])
            else:
                raise TypeError(f"Expected MomentTensor, got {type(item).__name__}")
        data = np.array(rows, dtype=np.float64)
    unique_labels = np.unique(labels)
    results: list[dict] = []
    axes: list[dict] = []

    for i, label in enumerate(unique_labels):
        logger.info(
            "Inverting cluster %d of %d (label=%d)",
            i + 1,
            len(unique_labels),
            label,
        )
        focal_mechs = data[np.where(labels == label)[0], :]
        inversion_results, stress_axes = _invert(focal_mechs)
        results.append(inversion_results)
        axes.append(stress_axes)

        sigma1_az = round(inversion_results["sigma_1"]["azimuth"], 1)
        sigma1_pl = round(inversion_results["sigma_1"]["plunge"], 1)
        logger.info("  Sigma 1  azimuth: %s  plunge: %s", sigma1_az, sigma1_pl)
        sigma2_az = round(inversion_results["sigma_2"]["azimuth"], 1)
        sigma2_pl = round(inversion_results["sigma_2"]["plunge"], 1)
        logger.info("  Sigma 2  azimuth: %s  plunge: %s", sigma2_az, sigma2_pl)
        sigma3_az = round(inversion_results["sigma_3"]["azimuth"], 1)
        sigma3_pl = round(inversion_results["sigma_3"]["plunge"], 1)
        logger.info("  Sigma 3  azimuth: %s  plunge: %s", sigma3_az, sigma3_pl)

    return StressInversionResult(results, axes)


def stress_inversion_with_bootstrap(
    data: np.ndarray | list,
    labels: np.ndarray,
    *,
    n_bootstrap: int = 100,
    random_state: int = 0,
) -> StressInversionResult:
    """Stress inversion with bootstrap uncertainty estimation.

    Runs :func:`stress_inversion` on the original data, then resamples
    focal mechanisms with replacement *n_bootstrap* times per cluster to
    estimate the spread of principal stress orientations.

    Parameters
    ----------
    data : numpy.ndarray or list of MomentTensor
        Focal mechanism data (same format as :func:`stress_inversion`).
    labels : numpy.ndarray
        Cluster label for each event.
    n_bootstrap : int
        Number of bootstrap iterations per cluster.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    StressInversionResult
        Same as :func:`stress_inversion`, with ``bootstrap_axes``
        populated.
    """
    # Accept list[MomentTensor] - convert to (N, 3) SDR array
    if isinstance(data, list):
        from swe.moment_tensor.tensor import MomentTensor, tensor_to_strike_dip_rake

        rows = []
        for item in data:
            if isinstance(item, MomentTensor):
                s, d, r = tensor_to_strike_dip_rake(item)
                rows.append([s, d, r])
            else:
                raise TypeError(f"Expected MomentTensor, got {type(item).__name__}")
        data = np.array(rows, dtype=np.float64)

    # Run original inversion
    base_result = stress_inversion(data, labels)

    rng = np.random.default_rng(random_state)
    unique_labels = np.unique(labels)
    bootstrap_axes: list[dict] = []

    for label in unique_labels:
        mask = labels == label
        cluster_data = data[mask]
        n_mechs = len(cluster_data)

        boot_results: list[dict] = []
        for _ in range(n_bootstrap):
            indices = rng.integers(0, n_mechs, size=n_mechs)
            resampled = cluster_data[indices]
            boot_labels = np.zeros(n_mechs, dtype=int)
            try:
                boot_inv = stress_inversion(resampled, boot_labels)
                boot_results.append(boot_inv.results[0])
            except (ValueError, np.linalg.LinAlgError):
                continue

        # Compute std of azimuths and plunges across bootstrap samples
        cluster_boot: dict = {}
        for axis_name in ["sigma_1", "sigma_2", "sigma_3"]:
            azimuths = [r[axis_name]["azimuth"] for r in boot_results if axis_name in r]
            plunges = [r[axis_name]["plunge"] for r in boot_results if axis_name in r]
            if azimuths:
                az_rad = np.radians(azimuths)
                az_std = float(
                    np.degrees(
                        np.sqrt(-2.0 * np.log(max(abs(np.mean(np.exp(1j * az_rad))), 1e-30)))
                    )
                )
                pl_std = float(np.std(plunges, ddof=1)) if len(plunges) > 1 else 0.0
            else:
                az_std = 0.0
                pl_std = 0.0
            cluster_boot[axis_name] = {
                "azimuth_std": az_std,
                "plunge_std": pl_std,
                "n_samples": len(azimuths),
            }
        bootstrap_axes.append(cluster_boot)

    return StressInversionResult(
        results=base_result.results,
        axes=base_result.axes,
        bootstrap_axes=bootstrap_axes,
    )


def cluster_focal_mechanisms(
    data: np.ndarray,
    n_clusters: int,
    *,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 0,
) -> ClusterResult:
    """Cluster focal mechanism data using K-means.

    Parameters
    ----------
    data : numpy.ndarray
        Coordinate data to cluster, shape ``(N, D)``.  Typically spatial
        coordinates of earthquakes (longitude, latitude, depth).
    n_clusters : int
        Number of clusters.
    n_init : int
        Number of K-means initializations.
    max_iter : int
        Maximum iterations per K-means run.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    labels : numpy.ndarray
        Cluster label for each sample.
    centers : numpy.ndarray
        Cluster center coordinates, shape ``(n_clusters, D)``.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    kmeans.fit(data)
    return ClusterResult(kmeans.labels_, kmeans.cluster_centers_)


def split_focal_data(
    events: np.ndarray,
    labels: np.ndarray,
    *,
    output_dir: str | Path = ".",
) -> None:
    """Split focal mechanism data by cluster and write to text files.

    Each cluster's events are written as a whitespace-delimited text file
    named ``events_{label}.txt`` in *output_dir*.  If magnitude (7th
    column) is missing, a default value of 5.0 is used.

    Parameters
    ----------
    events : numpy.ndarray
        Event data with columns: ``lon, lat, depth, strike, dip, rake``
        and optionally ``magnitude``.
    labels : numpy.ndarray
        Cluster label for each event.
    output_dir : str or Path
        Directory to write output files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label in np.unique(labels):
        cluster_data = events[np.where(labels == label)[0], :]

        # Add default magnitude if not present
        if cluster_data.shape[1] < 7:
            mags = np.full((len(cluster_data), 1), 5.0)
            cluster_data = np.hstack((cluster_data, mags))

        # Add X, Y, ID columns
        n = len(cluster_data)
        x_y_id = np.column_stack(
            [
                np.full(n, "X"),
                np.full(n, "Y"),
                np.full(n, str(label)),
            ]
        )
        cluster_data_out = np.hstack((cluster_data.astype(str), x_y_id))

        output_file = output_dir / f"events_{label}.txt"
        np.savetxt(str(output_file), cluster_data_out, delimiter=" ", fmt="%s")
        logger.info("Wrote cluster %s to %s", label, output_file)
