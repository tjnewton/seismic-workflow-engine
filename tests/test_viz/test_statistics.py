"""Tests for swe.viz.statistics - edge cases and label/binning validation."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from swe.viz.statistics import (
    plot_date_distribution,
    plot_distribution,
    plot_moment_distribution,
)

# ---------------------------------------------------------------------------
# plot_distribution edge cases
# ---------------------------------------------------------------------------


class TestPlotDistributionEdgeCases:
    """Edge-case tests for plot_distribution."""

    def test_empty_array(self):
        """Empty input should not crash and should return a Figure."""
        fig = plot_distribution([])
        assert isinstance(fig, Figure)

    def test_single_value(self):
        """A single data point should produce a valid histogram."""
        fig = plot_distribution([42.0])
        assert isinstance(fig, Figure)

    def test_nan_values(self):
        """NaN values in the input should not crash."""
        fig = plot_distribution([1.0, np.nan, 3.0])
        assert isinstance(fig, Figure)

    def test_all_identical_values(self):
        """All-identical values (zero-width range) should still return a Figure."""
        fig = plot_distribution([5.0] * 100)
        assert isinstance(fig, Figure)

    def test_xlabel_applied(self):
        """Custom xlabel should appear on the x-axis."""
        fig = plot_distribution([1.0, 2.0, 3.0], xlabel="Custom Label")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Custom Label"

    def test_title_applied(self):
        """Provided title should appear on the axes."""
        fig = plot_distribution([1.0, 2.0], title="My Title")
        ax = fig.axes[0]
        assert ax.get_title() == "My Title"

    def test_no_title_when_empty_string(self):
        """An empty-string title should leave the title unset."""
        fig = plot_distribution([1.0, 2.0], title="")
        ax = fig.axes[0]
        assert ax.get_title() == ""

    def test_custom_bins(self):
        """Different bin counts should produce different patch counts."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=500).tolist()
        fig10 = plot_distribution(data, bins=10)
        fig100 = plot_distribution(data, bins=100)
        patches_10 = len(fig10.axes[0].patches)
        patches_100 = len(fig100.axes[0].patches)
        assert patches_10 < patches_100

    def test_custom_figsize(self):
        """Custom figsize should be reflected in the Figure."""
        fig = plot_distribution([1.0, 2.0], figsize=(12, 6))
        w, h = fig.get_size_inches()
        assert abs(w - 12.0) < 0.1
        assert abs(h - 6.0) < 0.1

    def test_negative_values(self):
        """Negative values should be handled correctly."""
        fig = plot_distribution([-5, -3, -1, 1, 3, 5])
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_moment_distribution edge cases
# ---------------------------------------------------------------------------


class TestPlotMomentDistributionEdgeCases:
    """Edge-case tests for plot_moment_distribution."""

    def test_quantile_lines_present(self):
        """Twin axis should have vertical lines for quantile annotations."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=3.0, scale=0.5, size=200).tolist()
        fig = plot_moment_distribution(data)
        # There should be at least 2 axes (main + twin)
        assert len(fig.axes) >= 2
        # The twin axis should have lines from the quantile overlay
        twin_ax = fig.axes[1]
        assert len(twin_ax.lines) >= 5  # 5 quantile lines

    def test_single_magnitude(self):
        """Single data point should still produce a valid figure."""
        fig = plot_moment_distribution([2.5])
        assert isinstance(fig, Figure)

    def test_negative_magnitudes(self):
        """All-negative magnitudes are valid and should not crash."""
        fig = plot_moment_distribution([-2.0, -1.5, -1.0, -0.5])
        assert isinstance(fig, Figure)

    def test_no_title_when_empty_string(self):
        """An empty-string title should leave the title unset."""
        fig = plot_moment_distribution([1.0, 2.0, 3.0], title="")
        ax = fig.axes[0]
        assert ax.get_title() == ""


# ---------------------------------------------------------------------------
# plot_date_distribution edge cases
# ---------------------------------------------------------------------------


class TestPlotDateDistributionEdgeCases:
    """Edge-case tests for plot_date_distribution."""

    def test_empty_dates(self):
        """Empty date list should not crash."""
        fig = plot_date_distribution([])
        assert isinstance(fig, Figure)

    def test_single_date(self):
        """Single UTCDateTime should produce a valid Figure."""
        from obspy import UTCDateTime

        fig = plot_date_distribution([UTCDateTime("2024-01-15")])
        assert isinstance(fig, Figure)

    def test_default_title_format(self):
        """No title provided should produce 'Detections (n=N)' default."""
        from obspy import UTCDateTime

        dates = [UTCDateTime("2024-01-01") + i * 3600 for i in range(10)]
        fig = plot_date_distribution(dates)
        ax = fig.axes[0]
        assert ax.get_title() == "Detections (n=10)"
