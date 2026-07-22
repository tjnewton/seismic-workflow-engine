"""Tests for swe.magnitude.relative."""

from __future__ import annotations

import numpy as np
import pandas as pd
from obspy import Stream

from swe.magnitude.relative import (
    RelativeMomentResult,
    relative_moment_dataframe,
    relative_moments_from_party,
    relative_moments_from_streams,
)

from .conftest import _make_trace


class TestRelativeMomentsFromParty:
    """Tests for relative_moments_from_party."""

    def test_returns_list_of_results(self, mock_party):
        results = relative_moments_from_party(mock_party)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], RelativeMomentResult)

    def test_result_shapes(self, mock_party):
        results = relative_moments_from_party(mock_party)
        r = results[0]
        assert len(r.relative_moments) == 10
        assert len(r.relative_magnitudes) == 10
        assert len(r.detect_times) == 10

    def test_moments_positive(self, mock_party):
        results = relative_moments_from_party(mock_party)
        assert np.all(results[0].relative_moments > 0)

    def test_magnitude_formula(self, mock_party):
        """Magnitudes should be (2/3) * log10(moment)."""
        results = relative_moments_from_party(mock_party)
        r = results[0]
        expected = (2.0 / 3.0) * np.log10(r.relative_moments)
        np.testing.assert_allclose(r.relative_magnitudes, expected)

    def test_min_cc_filter(self, mock_party):
        results_all = relative_moments_from_party(mock_party, min_cc=0.0)
        results_filtered = relative_moments_from_party(mock_party, min_cc=1.0)
        assert len(results_filtered[0].relative_moments) <= len(results_all[0].relative_moments)

    def test_high_min_cc_empty(self, mock_party):
        results = relative_moments_from_party(mock_party, min_cc=100.0)
        assert len(results[0].relative_moments) == 0
        assert len(results[0].relative_magnitudes) == 0
        assert len(results[0].detect_times) == 0

    def test_multi_family(self, mock_party_multi):
        results = relative_moments_from_party(mock_party_multi)
        assert len(results) == 2
        assert results[0].template_name != results[1].template_name

    def test_template_name_preserved(self, mock_party):
        results = relative_moments_from_party(mock_party)
        assert results[0].template_name == "tmpl_000"


class TestRelativeMomentsFromStreams:
    """Tests for relative_moments_from_streams."""

    def test_amplitude_ratio(self, mock_family):
        # Build detection streams with 2x amplitude
        det_streams = []
        for _ in range(len(mock_family.detections)):
            st = Stream()
            for tmpl_tr in mock_family.template.st:
                tr = _make_trace(
                    network=tmpl_tr.stats.network,
                    station=tmpl_tr.stats.station,
                    channel=tmpl_tr.stats.channel,
                    amplitude=2.0,
                )
                st += tr
            det_streams.append(st)

        result = relative_moments_from_streams(mock_family, det_streams)
        assert isinstance(result, RelativeMomentResult)
        assert len(result.relative_moments) == 10
        # All ratios should be ~2.0 (same rng seed means same shape)
        np.testing.assert_allclose(
            result.relative_moments,
            2.0,
            atol=0.1,
        )

    def test_min_cc_filter(self, mock_family):
        det_streams = [Stream() for _ in mock_family.detections]
        result = relative_moments_from_streams(
            mock_family,
            det_streams,
            min_cc=100.0,
        )
        assert len(result.relative_moments) == 0

    def test_empty_detection_streams(self, mock_family):
        """Empty streams should fall back to cc/n_chans."""
        det_streams = [Stream() for _ in mock_family.detections]
        result = relative_moments_from_streams(mock_family, det_streams)
        assert len(result.relative_moments) == 10
        assert np.all(result.relative_moments > 0)


class TestRelativeMomentDataframe:
    """Tests for relative_moment_dataframe."""

    def test_returns_dataframe(self, mock_party):
        df = relative_moment_dataframe(mock_party)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, mock_party):
        df = relative_moment_dataframe(mock_party)
        expected_cols = {
            "template_name",
            "detect_time",
            "relative_moment",
            "relative_magnitude",
        }
        assert set(df.columns) == expected_cols

    def test_row_count(self, mock_party):
        df = relative_moment_dataframe(mock_party)
        assert len(df) == 10

    def test_multi_family_rows(self, mock_party_multi):
        df = relative_moment_dataframe(mock_party_multi)
        assert len(df) == 10  # 5 per family * 2 families

    def test_min_cc_filter(self, mock_party):
        df_all = relative_moment_dataframe(mock_party, min_cc=0.0)
        df_filtered = relative_moment_dataframe(mock_party, min_cc=100.0)
        assert len(df_filtered) == 0
        assert len(df_all) > 0
