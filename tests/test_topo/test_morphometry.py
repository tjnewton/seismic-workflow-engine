"""Tests for swe.topo.morphometry."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from swe.topo import (
    ElevationGrid,
    curvature,
    faceted_template,
    grid_spacing,
    local_plane_fit,
    slope_aspect,
    template_correlation,
)

DX = 5.0


def _grid(n=41):
    off = (np.arange(n) - n // 2) * DX
    return np.meshgrid(off, off)  # x along columns, y along rows (row index increasing)


class TestSlopeAspect:
    def test_east_rising_plane_faces_west(self):
        x, _ = _grid()
        slope, aspect = slope_aspect(0.1 * x, dx=DX, dy=DX)
        assert slope[20, 20] == pytest.approx(np.degrees(np.arctan(0.1)))
        assert aspect[20, 20] == pytest.approx(270.0)

    def test_north_up_raster_sign_convention(self):
        """Row 0 is the north edge (dy < 0); elevation rising northward faces south."""
        n = 21
        north = np.arange(n)[::-1, None] * DX * np.ones((1, n))  # largest at row 0 (north)
        _, aspect = slope_aspect(0.2 * north, dx=DX, dy=-DX)
        assert aspect[10, 10] == pytest.approx(180.0)

    def test_nan_cell_gives_nan(self):
        x, _ = _grid(7)
        z = 0.1 * x
        z[3, 3] = np.nan
        slope, aspect = slope_aspect(z, dx=DX, dy=DX)
        assert np.isnan(slope[3, 3]) and np.isnan(aspect[3, 3])

    def test_masked_cells_are_nodata(self):
        x, _ = _grid(9)
        z = np.ma.masked_array(0.1 * x, mask=np.zeros(x.shape, bool))
        z[4, 4] = np.ma.masked  # the stored value underneath stays a sentinel
        z.data[4, 4] = -9999.0
        slope, _ = slope_aspect(z, dx=DX, dy=DX)
        assert np.isnan(slope[4, 4])
        assert np.nanmax(slope) == pytest.approx(np.degrees(np.arctan(0.1)))

    def test_flat_tolerance_is_local(self):
        """An extreme value elsewhere must not make every aspect NaN."""
        x, _ = _grid(21)
        z = 0.1 * x
        z[0, 0] = -3.4e38
        _, aspect = slope_aspect(z, dx=DX, dy=DX)
        assert aspect[10, 10] == pytest.approx(270.0)

    def test_infinite_cell_is_nodata(self):
        x, _ = _grid(15)
        z = 0.1 * x
        z[7, 7] = np.inf
        slope, _ = slope_aspect(z, dx=DX, dy=DX)
        assert np.nanmax(slope) == pytest.approx(np.degrees(np.arctan(0.1)))
        assert np.isfinite(curvature(z, dx=DX, dy=DX)).sum() == z.size - 5  # 5-point stencil
        assert np.isinf(z[7, 7])  # the caller's array is not modified

    def test_array_spacing_is_rejected_clearly(self):
        with pytest.raises(ValueError, match="scalars"):
            slope_aspect(np.zeros((5, 5)), dx=np.full(4, DX), dy=DX)

    def test_flat_aspect_is_nan_and_bad_inputs(self):
        _, aspect = slope_aspect(np.zeros((5, 5)), dx=DX, dy=DX)
        assert np.isnan(aspect).all()
        with pytest.raises(ValueError, match="2-D"):
            slope_aspect(np.zeros(5), dx=DX, dy=DX)
        with pytest.raises(ValueError, match="non-zero"):
            slope_aspect(np.zeros((5, 5)), dx=0.0, dy=DX)


class TestLocalPlaneFit:
    def test_recovers_plane_exactly(self):
        x, y = _grid()
        fit = local_plane_fit(0.05 * x - 0.02 * y + 3.0, dx=DX, dy=DX, size=5)
        assert np.nanmax(fit.residual_std) == pytest.approx(0.0, abs=1e-6)
        assert fit.slope[20, 20] == pytest.approx(np.degrees(np.arctan(np.hypot(0.05, 0.02))))
        expect_aspect = np.degrees(np.arctan2(-0.05, 0.02)) % 360
        assert fit.aspect[20, 20] == pytest.approx(expect_aspect)
        assert np.isnan(fit.slope[0, 0]) and np.isnan(fit.slope[-1, -1])

    def test_residual_std_is_unbiased(self):
        """Mean residual variance equals the noise variance (RSS / (n - 3), not RSS / n)."""
        x, _ = _grid(101)
        rng = np.random.default_rng(0)
        fit = local_plane_fit(0.05 * x + rng.normal(0, 1.0, x.shape), dx=DX, dy=DX, size=5)
        assert np.nanmean(fit.residual_std**2) == pytest.approx(1.0, rel=0.05)

    def test_nan_stays_local(self):
        x, y = _grid(61)
        z = 0.05 * x - 0.02 * y + 3.0
        z[30, 30] = np.nan
        fit = local_plane_fit(z, dx=DX, dy=DX, size=5)
        interior = fit.slope[2:-2, 2:-2]
        assert np.isnan(interior).sum() == 25  # only the windows that contain the hole
        assert np.isnan(fit.slope[28:33, 28:33]).all() and np.isfinite(fit.slope[30, 40])

    def test_flat_surface_aspect_is_nan(self):
        fit = local_plane_fit(np.full((30, 30), 1234.567), dx=DX, dy=-DX, size=5)
        assert np.isnan(fit.aspect).all()
        assert np.nanmax(fit.slope) == pytest.approx(0.0, abs=1e-9)

    def test_large_offset_keeps_precision(self):
        """A 5000 m plateau with 10 um roughness: the offset must not swamp the residual."""
        x, y = _grid(61)
        rng = np.random.default_rng(3)
        z = 5000.0 + 0.05 * x - 0.02 * y + rng.normal(0, 1e-5, x.shape)
        fit = local_plane_fit(z, dx=DX, dy=DX, size=5)
        assert np.sqrt(np.nanmean(fit.residual_std**2)) == pytest.approx(1e-5, rel=0.1)

    def test_one_sentinel_does_not_break_the_grid(self):
        x, _ = _grid(21)
        z = 0.1 * x
        z[0, 0] = -3.4e38
        fit = local_plane_fit(z, dx=DX, dy=DX, size=5)
        assert fit.slope[10, 10] == pytest.approx(np.degrees(np.arctan(0.1)))
        assert fit.aspect[10, 10] == pytest.approx(270.0)

    def test_numpy_integer_size(self):
        x, _ = _grid()
        fit = local_plane_fit(0.1 * x, dx=DX, dy=DX, size=np.int64(5))
        assert fit.aspect[20, 20] == pytest.approx(270.0)

    def test_bad_size(self):
        for size in (4, 1, 3.0):
            with pytest.raises(ValueError, match="odd integer"):
                local_plane_fit(np.zeros((9, 9)), dx=DX, dy=DX, size=size)
        with pytest.raises(ValueError, match="larger than the grid"):
            local_plane_fit(np.zeros((9, 9)), dx=DX, dy=DX, size=11)


class TestCurvatureAndTemplate:
    def test_paraboloid_laplacian(self):
        x, y = _grid()
        lap = curvature(x**2 + y**2, dx=DX, dy=DX)
        assert lap == pytest.approx(np.full(x.shape, 4.0))  # edges included
        for sigma in (-1.0, np.inf):
            with pytest.raises(ValueError, match="sigma"):
                curvature(x, dx=DX, dy=DX, sigma=sigma)

    def test_smoothing_spreads_nodata_within_its_footprint_only(self):
        """No renormalized (biased) values at a hole edge: NaN within ~4 sigma, exact beyond."""
        x, y = _grid()
        z = 0.1 * x + 0.2 * y  # a plane has zero curvature, so any finite value must be 0
        z[20, 20] = np.nan
        lap = curvature(z, dx=DX, dy=DX, sigma=2 * DX)
        assert np.abs(lap[np.isfinite(lap)]).max() == pytest.approx(0.0, abs=1e-12)
        assert np.isnan(lap[20, 12:29]).all() and np.isfinite(lap[20, [10, 30]]).all()
        assert np.isnan(lap[:9]).all() and np.isnan(lap[:, -9:]).all()  # 4 sigma + 1 at the edges

    def test_cell_scale_structure_is_resolved(self):
        """A checkerboard has Laplacian -8 z on a unit grid (a 2-cell-wide stencil gives 0)."""
        checker = (-1.0) ** np.add.outer(np.arange(9), np.arange(9))
        lap = curvature(checker, dx=1.0, dy=1.0)
        assert lap[1:-1, 1:-1] == pytest.approx(-8.0 * checker[1:-1, 1:-1])

    def test_smoothing_is_isotropic_in_length(self):
        """Smoothing x**4 + y**4 by an isotropic Gaussian of sigma s adds 12 s**2 per axis.

        The discrete second difference of x**4 adds 2 h**2 per axis at the origin.
        """
        dx, dy, s = 10.0, 30.0, 60.0
        yy, xx = np.mgrid[-40:41, -40:41].astype(float)
        lap = curvature((xx * dx) ** 4 + (yy * dy) ** 4, dx=dx, dy=-dy, sigma=s)
        assert lap[40, 40] == pytest.approx(2 * (dx**2 + dy**2) + 24 * s**2, rel=0.01)

    def test_faceted_template_shape(self):
        t = faceted_template(n_sides=4, base=100.0, height=30.0, dx=DX, pad=2)
        c = t.shape[0] // 2
        assert t.shape[0] % 2 == 1 and t[c, c] == pytest.approx(30.0)
        assert t[0, 0] == 0.0 and t[c, 0] == 0.0
        assert faceted_template(n_sides=3, base=100.0, height=10.0, dx=DX).max() == pytest.approx(
            10.0
        )
        for bad in (
            {"n_sides": 2},
            {"n_sides": 4, "pad": 1.5},
            {"n_sides": 4, "base": np.inf},
            {"n_sides": 4, "base": 4.0},  # less than two cells across
        ):
            with pytest.raises(ValueError):
                faceted_template(**{"base": 100.0, "height": 10.0, "dx": DX, **bad})

    def test_rotation_and_non_square_cells(self):
        """A 100 m square turned 45 degrees reaches 70.7 m toward row 0; dy sets the row count."""
        c = 20
        flat = faceted_template(n_sides=4, base=100.0, height=10.0, dx=DX, pad=10)
        turned = faceted_template(n_sides=4, base=100.0, height=10.0, dx=DX, rotation=45.0)
        assert flat[c - 14, c] == 0 and turned.shape == (31, 31) and turned[15 - 14, 15] > 0
        aniso = faceted_template(n_sides=4, base=100.0, height=10.0, dx=DX, dy=-2 * DX)
        assert aniso.shape == (11, 21)  # 50 m half-width: 5 rows of 10 m, 10 columns of 5 m
        assert aniso[5, 10] == pytest.approx(10.0) and aniso[0, 10] == 0 and aniso[5, 0] == 0

    def test_triangle_edge_faces_row_zero(self):
        """Apothem 50 m toward row 0 (the facing edge); the opposite vertex is 100 m away."""
        t = faceted_template(n_sides=np.int64(3), base=100.0, height=10.0, dx=-DX)
        cy, cx = t.shape[0] // 2, t.shape[1] // 2
        assert t[cy - 9, cx] > 0 and t[cy - 10, cx] == 0
        assert t[cy + 19, cx] > 0 and t.shape[0] == 2 * 20 + 1  # vertex inside the template
        assert t[cy, :] == pytest.approx(t[cy, ::-1])  # symmetric east-west

    def test_correlation_peak_at_planted_location(self):
        t = faceted_template(n_sides=4, base=100.0, height=30.0, dx=DX, pad=2)
        f = np.zeros((81, 81))
        h = t.shape[0] // 2
        f[30 - h : 30 + h + 1, 50 - h : 50 + h + 1] += t
        ncc = template_correlation(f, t)
        assert np.nanargmax(ncc) == np.ravel_multi_index((30, 50), f.shape)
        assert np.nanmax(ncc) == pytest.approx(1.0)
        assert np.isnan(ncc[0, 0])

    def test_correlation_nan_stays_local(self):
        rng = np.random.default_rng(1)
        f = rng.normal(size=(60, 60))
        f[30, 30] = np.nan
        ncc = template_correlation(f, rng.normal(size=(5, 5)))
        assert np.isnan(ncc[2:-2, 2:-2]).sum() == 25
        assert np.isfinite(ncc[30, 40])

    def test_correlation_ignores_large_offset(self):
        t = faceted_template(n_sides=4, base=50.0, height=1.0, dx=DX, pad=1)
        f = np.random.default_rng(2).normal(0, 0.01, (40, 40))
        h = t.shape[0] // 2
        f[20 - h : 21 + h, 20 - h : 21 + h] += t
        assert template_correlation(f + 4000.0, t) == pytest.approx(
            template_correlation(f, t), abs=1e-6, nan_ok=True
        )

    def test_quiet_region_is_not_masked_by_a_loud_one(self):
        rng = np.random.default_rng(5)
        f = rng.normal(size=(60, 60))
        f[:, :30] *= 1e-7
        ncc = template_correlation(f, rng.normal(size=(5, 5)))
        assert np.isfinite(ncc[2:-2, 2:28]).all()

    def test_window_far_from_the_median_is_resolved(self):
        """std / offset = 1e-6 on a 21 x 21 template: within the documented sqrt(64 eps (Kx+Ky))."""
        rng = np.random.default_rng(7)
        f = rng.normal(0, 1.0, (80, 80))
        f[20:60, 20:60] = 1000.0 + rng.normal(0, 1e-3, (40, 40))
        t = rng.normal(size=(21, 21))
        ncc = template_correlation(f, t)
        w = f[30:51, 30:51]
        expect = np.corrcoef(w.ravel(), t.ravel())[0, 1]
        assert ncc[40, 40] == pytest.approx(expect, abs=0.05)

    def test_correlation_is_unit_free(self):
        rng = np.random.default_rng(4)
        f, t = rng.normal(size=(40, 40)), rng.normal(size=(7, 7))
        assert template_correlation(1e-7 * f, t) == pytest.approx(
            template_correlation(f, t), abs=1e-9, nan_ok=True
        )

    def test_correlation_input_checks(self):
        with pytest.raises(ValueError, match="odd"):
            template_correlation(np.zeros((9, 9)), np.ones((2, 3)))
        with pytest.raises(ValueError, match="constant"):
            template_correlation(np.zeros((9, 9)), np.ones((3, 3)))
        with pytest.raises(ValueError, match="constant"):  # caught before mean subtraction
            template_correlation(np.zeros((9, 9)), np.full((5, 5), 0.1))
        with pytest.raises(ValueError, match="finite"):
            template_correlation(np.zeros((9, 9)), np.full((3, 3), np.nan))
        with pytest.raises(ValueError, match="larger"):
            template_correlation(np.zeros((3, 3)), np.eye(5))


class TestSensitivityAndSpecificity:
    """Acceptance: a faceted landform is found; a dome and a ridge of the same size are not."""

    def test_curvature_correlation_separates_facets_from_dome_and_ridge(self):
        rng = np.random.default_rng(0)
        t = faceted_template(n_sides=4, base=200.0, height=60.0, dx=DX, pad=4)
        n, c, h = 201, 100, t.shape[0] // 2
        yy, xx = np.mgrid[-c : c + 1, -c : c + 1] * DX
        faceted = np.zeros((n, n))
        faceted[c - h : c + h + 1, c - h : c + h + 1] += t
        dome = 60.0 * np.clip(1 - (xx**2 + yy**2) / 100.0**2, 0, None)
        ridge = 60.0 * np.clip(1 - np.abs(xx) / 100.0, 0, None)
        tc = curvature(t, dx=DX, dy=-DX, sigma=DX, mask_edges=False)
        threshold = 0.6
        scores = {}
        for name, surf in (("faceted", faceted), ("dome", dome), ("ridge", ridge)):
            noisy = surf + rng.normal(0, 2.0, surf.shape)  # 2 m noise
            scores[name] = np.nanmax(
                template_correlation(curvature(noisy, dx=DX, dy=-DX, sigma=DX), tc)
            )
        assert scores["faceted"] >= threshold
        assert scores["dome"] < threshold and scores["ridge"] < threshold

    def test_rotation_search_finds_a_turned_landform(self):
        """Turned 30 degrees, it is found at the right rotation, well above a dome or ridge."""
        rng = np.random.default_rng(0)
        n, c = 201, 100
        yy, xx = np.mgrid[-c : c + 1, -c : c + 1] * DX
        kw = {"n_sides": 4, "base": 200.0, "height": 60.0, "dx": DX, "pad": 4}
        planted = faceted_template(rotation=30.0, **kw)
        hy, hx = planted.shape[0] // 2, planted.shape[1] // 2
        faceted = np.zeros((n, n))
        faceted[c - hy : c + hy + 1, c - hx : c + hx + 1] += planted
        dome = 60.0 * np.clip(1 - (xx**2 + yy**2) / 100.0**2, 0, None)
        ridge = 60.0 * np.clip(1 - np.abs(xx) / 100.0, 0, None)
        rotations = np.arange(0.0, 90.0, 15.0)  # a square repeats every 90 degrees
        templates = [
            curvature(faceted_template(rotation=r, **kw), dx=DX, dy=-DX, sigma=DX, mask_edges=False)
            for r in rotations
        ]
        best = {}
        for name, surf in (("faceted", faceted), ("dome", dome), ("ridge", ridge)):
            cv = curvature(surf + rng.normal(0, 2.0, surf.shape), dx=DX, dy=-DX, sigma=DX)
            best[name] = [np.nanmax(template_correlation(cv, t)) for t in templates]
        assert rotations[np.argmax(best["faceted"])] == 30.0
        assert max(best["faceted"]) > 1.5 * max(best["dome"] + best["ridge"])


class TestGridSpacing:
    def test_north_up_grid_gives_negative_dy(self):
        g = ElevationGrid(
            x=np.arange(5) * 10.0,
            y=100.0 - np.arange(4) * 10.0,
            elevation=np.zeros((4, 5)),
            crs=None,
        )
        assert grid_spacing(g) == (10.0, -10.0)

    def test_rejects_non_uniform_and_transposed(self):
        z = np.zeros((3, 3))
        uneven = ElevationGrid(x=np.array([0.0, 1.0, 3.0]), y=np.arange(3.0), elevation=z, crs=None)
        with pytest.raises(ValueError, match="uniformly"):
            grid_spacing(uneven)
        tiny = uneven._replace(x=np.array([0.0, 1e-9, 2e-9, 3.5e-9]), elevation=np.zeros((3, 4)))
        with pytest.raises(ValueError, match="uniformly"):  # relative check, not atol=1e-8
            grid_spacing(tiny)
        transposed = ElevationGrid(
            x=np.arange(4.0), y=np.arange(3.0), elevation=np.zeros((4, 3)), crs=None
        )
        with pytest.raises(ValueError, match="shape"):
            grid_spacing(transposed)
        # float32 at UTM magnitude steps by 0.5 m, unevenly; the endpoint spacing is still right
        north = (5e6 + np.arange(1001) * 0.25).astype(np.float32)
        f32 = ElevationGrid(x=np.arange(3.0), y=north, elevation=np.zeros((1001, 3)), crs=None)
        assert grid_spacing(f32)[1] == pytest.approx(0.25, rel=0.01)
        repeated = ElevationGrid(x=np.full(3, 5.0), y=np.arange(3.0), elevation=z, crs=None)
        with pytest.raises(ValueError, match="zero spacing"):
            grid_spacing(repeated)

    @pytest.mark.parametrize("backend", ["pyproj", "rasterio"])
    def test_crs_checks(self, monkeypatch, backend):
        units = {
            "EPSG:4326": (True, ["degree"]),
            "EPSG:32606": (False, ["metre"]),
            "EPSG:2263": (False, ["US survey foot"]),
            "EPSG:26910+6360": (False, ["metre", "US survey foot"]),  # compound: feet vertical
        }

        class _CRS:  # stand-in for pyproj.CRS / rasterio.crs.CRS (CI installs neither)
            def __init__(self, crs):
                self.is_geographic, us = units[crs]
                self.linear_units = us[0]
                self.axis_info = [types.SimpleNamespace(unit_name=u) for u in us]

            @classmethod
            def from_user_input(cls, crs):
                if crs not in units:
                    raise RuntimeError(f"cannot parse {crs}")  # like pyproj's CRSError
                return cls(crs)

        if backend == "pyproj":
            monkeypatch.setitem(sys.modules, "pyproj", types.SimpleNamespace(CRS=_CRS))
        else:
            monkeypatch.setitem(sys.modules, "pyproj", None)
            monkeypatch.setitem(sys.modules, "rasterio", types.ModuleType("rasterio"))
            monkeypatch.setitem(sys.modules, "rasterio.crs", types.SimpleNamespace(CRS=_CRS))
        grid = ElevationGrid(
            x=np.arange(3.0), y=np.arange(3.0), elevation=np.zeros((3, 3)), crs="EPSG:32606"
        )
        assert grid_spacing(grid) == (1.0, 1.0)
        with pytest.raises(ValueError, match="geographic"):
            grid_spacing(grid._replace(crs="EPSG:4326"))
        with pytest.warns(UserWarning, match="unit"):
            grid_spacing(grid._replace(crs="EPSG:2263"))
        if backend == "pyproj":  # rasterio reports the horizontal unit only
            with pytest.warns(UserWarning, match="unit"):
                grid_spacing(grid._replace(crs="EPSG:26910+6360"))
        with pytest.warns(UserWarning, match="not checked"):
            assert grid_spacing(grid._replace(crs="local")) == (1.0, 1.0)

    def test_crs_unchecked_without_backends(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyproj", None)
        monkeypatch.setitem(sys.modules, "rasterio.crs", None)
        grid = ElevationGrid(
            x=np.arange(3.0), y=np.arange(3.0), elevation=np.zeros((3, 3)), crs="EPSG:32606"
        )
        with pytest.warns(UserWarning, match="not checked"):
            assert grid_spacing(grid) == (1.0, 1.0)
