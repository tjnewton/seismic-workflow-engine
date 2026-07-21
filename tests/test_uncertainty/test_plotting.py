"""Tests for swe.uncertainty.plotting - uncertainty visualisation."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from swe.uncertainty.plotting import (
    plot_location_uncertainty,
    plot_magnitude_pdf,
    plot_pick_uncertainty,
    plot_uncertainty_summary,
)
from swe.uncertainty.types import (
    LocationUncertainty,
    MagnitudeUncertainty,
    UncertainValue,
)


class TestPlotPickUncertainty:
    def test_basic(self):
        times = np.array([1.0, 2.0, 3.0])
        uncertainties = np.array([0.1, 0.2, 0.15])
        fig = plot_pick_uncertainty(times, uncertainties)
        assert isinstance(fig, Figure)

    def test_with_labels(self):
        times = np.array([1.0, 2.0])
        uncertainties = np.array([0.1, 0.2])
        fig = plot_pick_uncertainty(
            times, uncertainties, station_labels=["STA1", "STA2"]
        )
        assert isinstance(fig, Figure)

    def test_with_figsize(self):
        times = np.array([1.0])
        uncertainties = np.array([0.1])
        fig = plot_pick_uncertainty(times, uncertainties, figsize=(6, 4))
        assert isinstance(fig, Figure)


class TestPlotLocationUncertainty:
    def test_basic(self):
        locations = [(-117.0, 35.0, 5.0), (-117.1, 35.1, 8.0)]
        uncertainties = [
            LocationUncertainty(horiz_uncertainty_m=500, depth_uncertainty_m=1000),
            LocationUncertainty(horiz_uncertainty_m=300, depth_uncertainty_m=500),
        ]
        fig = plot_location_uncertainty(locations, uncertainties)
        assert isinstance(fig, Figure)

    def test_with_covariance(self):
        cov = np.diag([100.0, 200.0, 300.0])
        locations = [(-117.0, 35.0, 5.0)]
        uncertainties = [
            LocationUncertainty(
                horiz_uncertainty_m=500,
                depth_uncertainty_m=1000,
                covariance_matrix=cov,
            ),
        ]
        fig = plot_location_uncertainty(locations, uncertainties)
        assert isinstance(fig, Figure)

    def test_zero_uncertainty(self):
        locations = [(-117.0, 35.0, 5.0)]
        uncertainties = [LocationUncertainty()]
        fig = plot_location_uncertainty(locations, uncertainties)
        assert isinstance(fig, Figure)


class TestPlotMagnitudePDF:
    def test_basic(self):
        magnitudes = np.array([2.0, 2.5, 3.0, 2.8])
        uncertainties = np.array([0.1, 0.15, 0.2, 0.12])
        fig = plot_magnitude_pdf(magnitudes, uncertainties)
        assert isinstance(fig, Figure)

    def test_single_event(self):
        fig = plot_magnitude_pdf(np.array([3.0]), np.array([0.1]))
        assert isinstance(fig, Figure)

    def test_zero_uncertainties_clamped(self):
        """Very small uncertainties should be clamped, not crash."""
        fig = plot_magnitude_pdf(np.array([3.0, 3.5]), np.array([0.0, 0.0]))
        assert isinstance(fig, Figure)


class TestPlotUncertaintySummary:
    def test_uncertain_values(self):
        results = {
            "Pick Time": UncertainValue(value=0.05, uncertainty=0.02),
            "Location": UncertainValue(value=500, uncertainty=100),
        }
        fig = plot_uncertainty_summary(results)
        assert isinstance(fig, Figure)

    def test_location_uncertainty(self):
        results = {
            "Location": LocationUncertainty(
                horiz_uncertainty_m=500, depth_uncertainty_m=1000
            ),
        }
        fig = plot_uncertainty_summary(results)
        assert isinstance(fig, Figure)

    def test_magnitude_uncertainty(self):
        results = {
            "Magnitude": MagnitudeUncertainty(
                mag_uncertainty=0.15, station_std=0.1
            ),
        }
        fig = plot_uncertainty_summary(results)
        assert isinstance(fig, Figure)

    def test_dict_values(self):
        results = {
            "Custom": {"value": 1.5, "uncertainty": 0.3},
        }
        fig = plot_uncertainty_summary(results)
        assert isinstance(fig, Figure)

    def test_float_values(self):
        results = {"RMS": 0.05, "Residual": 0.1}
        fig = plot_uncertainty_summary(results)
        assert isinstance(fig, Figure)

    def test_mixed_types(self):
        results = {
            "Pick": UncertainValue(value=0.05, uncertainty=0.02),
            "Location": LocationUncertainty(horiz_uncertainty_m=500),
            "Magnitude": MagnitudeUncertainty(mag_uncertainty=0.15),
            "RMS": 0.05,
        }
        fig = plot_uncertainty_summary(results)
        assert isinstance(fig, Figure)
