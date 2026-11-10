"""Terrain morphometry on gridded elevation: slope, aspect, local planes, curvature, templates.

All functions take a 2-D elevation array of shape ``(ny, nx)`` and signed grid spacings ``dx``
(along columns, eastward positive) and ``dy`` (along rows; negative for north-up rasters whose
row 0 is the northern edge), in the same length unit as the elevation. Use a projected CRS.
NaN cells (nodata) make NaN only the outputs whose stencil, filter or window contains them;
convert nodata sentinels (e.g. from ``grid_from_raster(..., fill_nodata=False)``) to NaN first.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import correlate1d, gaussian_filter

from swe.topo.raster import ElevationGrid

__all__ = [
    "PlaneFit",
    "curvature",
    "faceted_template",
    "grid_spacing",
    "local_plane_fit",
    "slope_aspect",
    "template_correlation",
]


class PlaneFit(NamedTuple):
    """Moving-window least-squares plane fit."""

    slope: NDArray[np.floating]
    """Slope of the fitted plane (degrees)."""
    aspect: NDArray[np.floating]
    """Downslope azimuth of the fitted plane (degrees clockwise from north; NaN where flat)."""
    residual_std: NDArray[np.floating]
    """Residual standard deviation about the plane, ``sqrt(RSS / (n - 3))`` (elevation units)."""


def _as_float(a: np.ndarray) -> np.ndarray:
    """Float array with masked (e.g. rasterio ``read(masked=True)``) and infinite cells as NaN."""
    out: np.ndarray = np.ma.filled(np.ma.asarray(a, dtype=float), np.nan)
    clean: np.ndarray = np.where(np.isinf(out), np.nan, out)  # a new array: never edit the input
    return clean


def _check(elevation: np.ndarray, dx: float, dy: float) -> np.ndarray:
    z = _as_float(elevation)
    if z.ndim != 2 or min(z.shape) < 3:
        raise ValueError(f"elevation must be a 2-D array of at least 3 x 3, got shape {z.shape}")
    if np.ndim(dx) or np.ndim(dy):
        raise ValueError("dx and dy must be scalars (a uniform grid); see grid_spacing")
    if dx == 0 or dy == 0 or not np.isfinite([dx, dy]).all():
        raise ValueError(f"dx and dy must be finite and non-zero, got {dx}, {dy}")
    return z


def grid_spacing(grid: ElevationGrid) -> tuple[float, float]:
    """Signed ``(dx, dy)`` of an :class:`~swe.topo.raster.ElevationGrid`, for these functions.

    *dy* is negative for a north-up grid (``y`` decreasing with the row index), as
    :func:`grid_from_raster` returns.

    The CRS is checked with pyproj or rasterio when either is installed: a geographic CRS is
    rejected, and a non-metre unit raises a warning (spacing and elevation must share a unit).

    Raises
    ------
    ValueError
        For non-uniform coordinates, an elevation array not shaped ``(len(y), len(x))``, or a
        geographic (degree) CRS.
    """
    x, y = np.asarray(grid.x), np.asarray(grid.y)
    # coordinate rounding: a float32 axis at UTM magnitudes steps unevenly by its own resolution
    tol = [
        4 * np.finfo(a.dtype if a.dtype.kind == "f" else float).eps * float(np.abs(a).max())
        for a in (x, y)
    ]
    x, y = x.astype(float), y.astype(float)
    if x.size < 2 or y.size < 2:
        raise ValueError("grid needs at least two x and two y coordinates")
    if np.shape(grid.elevation) != (y.size, x.size):
        raise ValueError(
            f"elevation shape {np.shape(grid.elevation)} is not (len(y), len(x))"
            f" = {(y.size, x.size)}"
        )
    dx, dy = float(x[-1] - x[0]) / (x.size - 1), float(y[-1] - y[0]) / (y.size - 1)
    if not (
        np.allclose(np.diff(x), dx, rtol=1e-6, atol=tol[0])
        and np.allclose(np.diff(y), dy, rtol=1e-6, atol=tol[1])
    ):
        raise ValueError("grid coordinates are not uniformly spaced")
    if dx == 0 or dy == 0:
        raise ValueError("grid coordinates repeat (zero spacing)")
    if grid.crs is not None:
        info = _crs_info(grid.crs)
        if info is None:
            warnings.warn(
                f"grid CRS {grid.crs!r} not checked (no pyproj or rasterio, or unparseable)",
                stacklevel=2,
            )
        else:
            geographic, units = info
            if geographic:
                raise ValueError(f"grid CRS {grid.crs} is geographic; reproject to a projected CRS")
            unit = next((u for u in units if u.lower() not in ("metre", "meter", "m")), None)
            if unit is not None:
                warnings.warn(
                    f"grid CRS unit is {unit}; dx, dy and elevation must share one unit",
                    stacklevel=2,
                )
    return dx, dy


def _crs_info(crs: str) -> tuple[bool, list[str]] | None:
    """(is_geographic, axis unit names) from pyproj, else rasterio (horizontal unit only).

    Each installed backend is tried in turn; None if none is installed or none can parse the CRS.
    """
    try:
        from pyproj import CRS as ProjCRS

        pc = ProjCRS.from_user_input(crs)
        return bool(pc.is_geographic), [a.unit_name for a in pc.axis_info]  # compound: + vertical
    except Exception:  # not installed, CRSError, missing PROJ data: try rasterio
        pass
    try:
        from rasterio.crs import CRS as RioCRS

        c = RioCRS.from_user_input(crs)
        return bool(c.is_geographic), [c.linear_units] if c.linear_units else []
    except Exception:  # an unparseable or local CRS is left unchecked
        return None


def _centred(z: np.ndarray) -> np.ndarray:
    """*z* minus the median of its finite cells, so window sums cancel less against an offset.

    The median, unlike the mean, keeps the direct window sums of :func:`local_plane_fit` away
    from a stray nodata sentinel; FFT-based correlation is not protected (convert to NaN).
    """
    ok = np.isfinite(z)
    return z - np.median(z[ok]) if ok.any() else z


def _mask_edges(a: np.ndarray, hy: int, hx: int) -> None:
    """Set the outer *hy* rows and *hx* columns of *a* to NaN in place (safe for 0)."""
    a[:hy, :] = a[a.shape[0] - hy :, :] = np.nan
    a[:, :hx] = a[:, a.shape[1] - hx :] = np.nan


def _window_sum(a: np.ndarray, wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
    """Direct sum over each window of ``a * wx(column offset) * wy(row offset)``.

    Separable and local, so rounding and NaN stay within the window (no FFT, no running sum).
    """
    out: np.ndarray = correlate1d(
        correlate1d(a, wx, axis=1, mode="constant"), wy, axis=0, mode="constant"
    )
    return out


def _aspect(
    scale: np.ndarray, dzdx: np.ndarray, dzdy: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """Downslope azimuth, degrees clockwise from north (dzdy is the northward gradient).

    NaN where the gradient is within rounding of zero for the local elevation size *scale*.
    """
    tol = 64 * np.finfo(float).eps * scale / min(abs(dx), abs(dy))
    az = np.degrees(np.arctan2(-dzdx, -dzdy)) % 360.0
    return np.where(np.hypot(dzdx, dzdy) <= tol, np.nan, az)


def slope_aspect(elevation: np.ndarray, *, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope (degrees) and downslope aspect (degrees clockwise from north) by central differences.

    Edge cells use one-sided differences (``numpy.gradient``).

    Parameters
    ----------
    elevation : ndarray
        2-D elevation, shape ``(ny, nx)``.
    dx, dy : float
        Signed spacing along columns (east) and rows (north if positive).

    Returns
    -------
    (ndarray, ndarray)
        Slope and aspect, same shape as *elevation*; aspect is NaN where the surface is flat.

    Raises
    ------
    ValueError
        For a non-2-D or smaller than 3 x 3 array, or zero or non-finite spacing.
    """
    z = _check(elevation, dx, dy)
    dzdrow, dzdcol = np.gradient(z)
    dzdx, dzdy = dzdcol / dx, dzdrow / dy  # dzdy is the gradient toward +y (north if dy > 0)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = _aspect(np.abs(z), dzdx, dzdy, dx, dy)
    bad = ~np.isfinite(z)
    slope[bad] = aspect[bad] = np.nan  # the stencil skips the centre cell
    return slope, aspect


def local_plane_fit(elevation: np.ndarray, *, dx: float, dy: float, size: int) -> PlaneFit:
    """Least-squares plane ``z = a x + b y + c`` in a ``size`` x ``size`` window around each cell.

    The planes are computed from separable moving-window moments (exact for interior cells);
    cells within ``size // 2`` of the edge are NaN. The moments are one-pass about the field
    median, so *residual_std* below about ``2e-8`` times the largest ``|z - median|`` in the
    window is rounding (this includes the window's own relief).

    Parameters
    ----------
    elevation : ndarray
        2-D elevation.
    dx, dy : float
        Signed spacings (see module docstring).
    size : int
        Odd window width in cells, at least 3.

    Returns
    -------
    PlaneFit
        Slope, aspect and residual standard deviation of the local plane.

    Raises
    ------
    ValueError
        For an even, too-small or larger-than-grid *size* (and see :func:`slope_aspect`).
    """
    z = _check(elevation, dx, dy)
    if not isinstance(size, (int, np.integer)) or size < 3 or size % 2 == 0:
        raise ValueError(f"size must be an odd integer >= 3, got {size!r}")
    if size > min(z.shape):
        raise ValueError(f"size {size} is larger than the grid {z.shape}")
    z = _centred(z)
    h = int(size) // 2
    off = np.arange(-h, h + 1, dtype=float)
    ones = np.ones_like(off)
    n = off.size**2

    # x and y offsets from the window centre are separable and sum to zero, so the normal
    # equations decouple: a = S_xz / S_xx, b = S_yz / S_yy, c = mean(z)
    sz, szz = _window_sum(z, ones, ones), _window_sum(z**2, ones, ones)
    sxz, syz = _window_sum(z, off * dx, ones), _window_sum(z, ones, off * dy)
    s_off2 = off.size * (off**2).sum()  # window sum of squared offsets, in cells
    a, b = sxz / (s_off2 * dx**2), syz / (s_off2 * dy**2)
    c = sz / n
    # residual sum of squares of z about the plane: S_zz - c S_z - a S_xz - b S_yz
    rss = np.maximum(szz - c * sz - a * sxz - b * syz, 0.0)
    residual_std = np.sqrt(rss / (n - 3))
    slope = np.degrees(np.arctan(np.hypot(a, b)))
    aspect = _aspect(np.sqrt(np.abs(szz) / n), a, b, dx, dy)  # window RMS of z
    for arr in (slope, aspect, residual_std):
        _mask_edges(arr, h, h)
    return PlaneFit(slope=slope, aspect=aspect, residual_std=residual_std)


def curvature(
    elevation: np.ndarray, *, dx: float, dy: float, sigma: float = 0.0, mask_edges: bool = True
) -> np.ndarray:
    """Laplacian of the elevation (``d2z/dx2 + d2z/dy2``), by second central differences.

    Second derivatives amplify noise, so the elevation can first be smoothed with an isotropic
    Gaussian of standard deviation *sigma*, in the length unit of *dx* and *dy* (0 = none).
    A NaN cell makes NaN every output whose smoothing footprint (about 4 *sigma*) or
    difference stencil contains it. With smoothing, cells within that radius of the grid edge
    are NaN (the edge padding would fake curvature there) unless *mask_edges* is False, which
    is exact only where the surface is flat at the edges, such as a zero-padded
    :func:`faceted_template`. Without smoothing, edge cells take the second difference of their
    inward neighbour.

    Raises
    ------
    ValueError
        For negative *sigma* (and see :func:`slope_aspect`).
    """
    z = _check(elevation, dx, dy)
    if not 0 <= sigma < np.inf:
        raise ValueError(f"sigma must be finite and non-negative, got {sigma}")
    sig = (sigma / abs(dy), sigma / abs(dx))  # in cells (rows, columns)
    if sigma > 0:
        # NaN spreads over the filter support (about 4 sigma). Renormalising over the valid cells
        # instead would bias the mean at hole edges and fake curvature there.
        z = gaussian_filter(z, sigma=sig, mode="nearest")
    d2col = np.pad(np.diff(z, 2, axis=1), ((0, 0), (1, 1)), mode="edge") / dx**2
    d2row = np.pad(np.diff(z, 2, axis=0), ((1, 1), (0, 0)), mode="edge") / dy**2
    out: np.ndarray = d2col + d2row
    if sigma > 0 and mask_edges:  # "nearest" padding bends the smoothed surface: mask the radius
        _mask_edges(out, *(int(4.0 * s_ + 0.5) + 1 for s_ in sig))
    return out


def faceted_template(
    *,
    n_sides: int,
    base: float,
    height: float,
    dx: float,
    dy: float | None = None,
    rotation: float = 0.0,
    pad: int = 0,
) -> np.ndarray:
    """Elevation of a regular ``n_sides``-faceted apex landform (planar facets meeting at a summit).

    The base is a regular polygon. At ``rotation=0`` one edge faces row 0 (north on a north-up
    raster), so a square is aligned with the grid. Elevation falls linearly from ``height`` at
    the centre to 0 at the base, and is 0 outside. Correlation is not rotation-invariant: to
    search for landforms of unknown orientation, take the maximum over a set of rotations.

    Parameters
    ----------
    n_sides : int
        Number of facets (at least 3).
    base : float
        Twice the apothem (the width across flats; the side length of a square).
    height : float
        Summit height above the base.
    dx, dy : float
        Grid spacing along columns and rows (same unit as *base*); signs are ignored and *dy*
        defaults to *dx*. Pass the raster's spacings so the template matches non-square cells.
    rotation : float
        Rotation in degrees, clockwise as displayed with row 0 at the top.
    pad : int
        Zero cells added on every side. To correlate :func:`curvature` smoothed by *sigma*
        with ``mask_edges=False``, pad by at least ``4 * sigma / min(dx, |dy|) + 1`` cells so
        the smoothed lobe is not cut at the border.

    Returns
    -------
    ndarray
        Template with odd dimensions, summit at the centre cell. For odd *n_sides* the polygon
        is not symmetric about the summit, so the side facing an edge carries extra zero rows.

    Raises
    ------
    ValueError
        For fewer than 3 sides, or non-finite or non-positive dimensions.
    """
    if not isinstance(n_sides, (int, np.integer)) or n_sides < 3:
        raise ValueError(f"n_sides must be an integer >= 3, got {n_sides!r}")
    dy = dx if dy is None else dy
    sizes = (base, height, abs(dx), abs(dy))
    if not all(0 < v < np.inf for v in sizes) or not np.isfinite(rotation):
        raise ValueError("base, height, dx and dy must be positive and finite, rotation finite")
    if not isinstance(pad, (int, np.integer)) or pad < 0:
        raise ValueError(f"pad must be a non-negative integer, got {pad!r}")
    dx, dy, r_in = abs(dx), abs(dy), base / 2  # r_in: apothem
    if r_in < max(dx, dy):
        raise ValueError(f"base / 2 = {r_in} must span at least one cell ({max(dx, dy)})")
    # facet normals at -pi/2 + 2 pi k / n (one toward row 0), plus the rotation (clockwise on
    # screen, since y increases with the row index); vertices lie midway between normals
    normals = np.radians(rotation) - np.pi / 2 + 2 * np.pi * np.arange(n_sides) / n_sides
    vertices = normals + np.pi / n_sides
    circ = r_in / np.cos(np.pi / n_sides)
    hx = int(np.ceil(circ * np.abs(np.cos(vertices)).max() / dx - 1e-9)) + pad
    hy = int(np.ceil(circ * np.abs(np.sin(vertices)).max() / dy - 1e-9)) + pad
    xx, yy = np.meshgrid(np.arange(-hx, hx + 1) * dx, np.arange(-hy, hy + 1) * dy)
    # distance to the polygon edge along each facet normal: max over facets of the projection
    proj = xx * np.cos(normals[0]) + yy * np.sin(normals[0])
    for a in normals[1:]:
        np.maximum(proj, xx * np.cos(a) + yy * np.sin(a), out=proj)
    out: np.ndarray = np.clip(height * (1.0 - proj / r_in), 0.0, None)
    return out


def template_correlation(field: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Local normalized cross-correlation of *template* with *field* (``same`` shape).

    At each cell, the Pearson correlation between the template and the window of *field*
    centred there. Cells whose window leaves the field, contains NaN, or is constant are NaN.

    The window mean and variance are direct local sums about the field median, so a window
    whose variance is below about ``64 * eps * (Kx + Ky)`` of its mean square about that median
    reads as constant (NaN); for a window far from the median this limits the resolvable
    std / offset to about ``sqrt(64 * eps * (Kx + Ky))``. The cross term uses an FFT, whose
    rounding scales with the whole field: windows whose variance is below that rounding are
    NaN, and a nodata sentinel left in the field corrupts every window. Correlate a derived or
    detrended field such as :func:`curvature`, not raw elevation with a strong regional trend.

    Parameters
    ----------
    field : ndarray
        2-D field to search.
    template : ndarray
        2-D finite template with odd dimensions, not constant.

    Returns
    -------
    ndarray
        Correlation in [-1, 1], NaN where undefined.

    Raises
    ------
    ValueError
        For non-2-D inputs, even template dimensions, a non-finite or constant template, or a
        template larger than the field.
    """
    f, t = _as_float(field), _as_float(template)
    if f.ndim != 2 or t.ndim != 2:
        raise ValueError("field and template must be 2-D")
    if t.shape[0] % 2 == 0 or t.shape[1] % 2 == 0:
        raise ValueError(f"template dimensions must be odd, got {t.shape}")
    if t.shape[0] > f.shape[0] or t.shape[1] > f.shape[1]:
        raise ValueError(f"template {t.shape} is larger than field {f.shape}")
    if not np.isfinite(t).all():
        raise ValueError("template must be finite")
    if np.ptp(t) == 0:
        raise ValueError("template is constant")
    t0 = t - t.mean()
    tn = np.sqrt((t0**2).sum())
    bad = ~np.isfinite(f)
    # zero the NaN cells (their windows are masked below) and remove the median offset, which
    # the correlation ignores, so the window variance does not cancel against it
    f = np.where(bad, 0.0, _centred(f))
    n = t.size

    wy, wx = np.ones(t.shape[0]), np.ones(t.shape[1])

    def box(a):
        return _window_sum(a, wx, wy)

    from scipy.signal import fftconvolve  # deferred: scipy.signal adds about 0.6 s to import

    num = fftconvolve(f, t0[::-1, ::-1], mode="same")
    f2 = f**2
    s1, s2 = box(f), box(f2)
    var = np.maximum(s2 - s1**2 / n, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ncc: np.ndarray = np.clip(num / (np.sqrt(var) * tn), -1.0, 1.0)
    eps = np.finfo(float).eps
    # constant window, to the rounding of the local sums; or variance below the FFT's
    # rounding of the cross term, which scales with the whole field
    taps = t.shape[0] + t.shape[1]  # the separable sums run over Kx + Ky terms per window
    undefined = (var <= 64 * eps * taps * s2) | (var <= (1e3 * eps) ** 2 * float(f2.sum()))
    if bad.any():
        undefined |= box(bad.astype(float)) > 0.5
    ncc[undefined] = np.nan
    _mask_edges(ncc, t.shape[0] // 2, t.shape[1] // 2)
    return ncc
