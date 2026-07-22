"""Tests for swe.picking.deep_learning."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from swe.picking.deep_learning import _SUPPORTED_MODELS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stream(
    stations: list[str] | None = None,
    *,
    npts: int = 6000,
    sampling_rate: float = 100.0,
    starttime: UTCDateTime | None = None,
) -> Stream:
    """Build a synthetic 3-component ObsPy Stream for testing."""
    if stations is None:
        stations = ["STA1", "STA2"]
    if starttime is None:
        starttime = UTCDateTime("2024-01-01T00:00:00.0Z")

    traces = []
    rng = np.random.default_rng(42)
    for sta in stations:
        for cha in ("HHZ", "HHN", "HHE"):
            tr = Trace(data=rng.standard_normal(npts).astype(np.float32))
            tr.stats.station = sta
            tr.stats.channel = cha
            tr.stats.network = "XX"
            tr.stats.location = ""
            tr.stats.sampling_rate = sampling_rate
            tr.stats.starttime = starttime
            traces.append(tr)
    return Stream(traces=traces)


def _make_mock_picks(
    stations: list[str],
    starttime: UTCDateTime,
) -> list:
    """Create fake SeisBench Pick objects as SimpleNamespace instances."""
    picks = []
    for i, sta in enumerate(stations):
        picks.append(
            SimpleNamespace(
                trace_id=f"XX.{sta}..HHZ",
                phase="P",
                peak_time=starttime + 10.0 + i * 5.0,
                peak_value=0.85 - i * 0.1,
            )
        )
        picks.append(
            SimpleNamespace(
                trace_id=f"XX.{sta}..HHZ",
                phase="S",
                peak_time=starttime + 15.0 + i * 5.0,
                peak_value=0.75 - i * 0.1,
            )
        )
    return picks


def _make_mock_seisbench(picks):
    """Build mock seisbench and seisbench.models modules."""
    mock_model = MagicMock()
    mock_model.annotate.return_value = Stream()
    mock_model.classify.return_value = picks

    mock_model_cls = MagicMock()
    mock_model_cls.from_pretrained.return_value = mock_model

    mock_sbm = MagicMock()
    for name in _SUPPORTED_MODELS:
        setattr(mock_sbm, name, mock_model_cls)

    mock_seisbench_pkg = MagicMock()
    mock_seisbench_pkg.models = mock_sbm

    return mock_sbm, mock_model_cls, mock_model, mock_seisbench_pkg


def _patch_seisbench(mock_sbm, mock_seisbench_pkg):
    """Context manager that injects mock seisbench into sys.modules."""
    return patch.dict(
        "sys.modules",
        {
            "seisbench": mock_seisbench_pkg,
            "seisbench.models": mock_sbm,
        },
    )


# ---------------------------------------------------------------------------
# Tests for predict_phase_arrivals
# ---------------------------------------------------------------------------


class TestPredictPhaseArrivals:
    """Tests for predict_phase_arrivals with mocked SeisBench."""

    def test_basic_picks(self):
        """Returns correctly-structured pick dicts."""
        from swe.picking.deep_learning import predict_phase_arrivals

        stations = ["STA1", "STA2"]
        t0 = UTCDateTime("2024-01-01T00:00:00.0Z")
        stream = _make_stream(stations, starttime=t0)
        mock_picks = _make_mock_picks(stations, t0)
        mock_sbm, _, _, mock_pkg = _make_mock_seisbench(mock_picks)

        with _patch_seisbench(mock_sbm, mock_pkg):
            results = predict_phase_arrivals(stream, model_name="PhaseNet")

        assert len(results) == 4  # 2 stations x (P + S)
        for pick in results:
            assert hasattr(pick, "station")
            assert hasattr(pick, "phase")
            assert hasattr(pick, "time")
            assert hasattr(pick, "confidence")
            assert pick.phase in ("P", "S")
            assert isinstance(pick.confidence, float)

    def test_station_extracted_from_trace_id(self):
        """Station code is extracted from the SEED trace ID."""
        from swe.picking.deep_learning import predict_phase_arrivals

        t0 = UTCDateTime("2024-01-01T00:00:00.0Z")
        stream = _make_stream(["MYSTA"], starttime=t0)
        picks = [
            SimpleNamespace(
                trace_id="XX.MYSTA..HHZ",
                phase="P",
                peak_time=t0 + 10.0,
                peak_value=0.9,
            )
        ]
        mock_sbm, _, _, mock_pkg = _make_mock_seisbench(picks)

        with _patch_seisbench(mock_sbm, mock_pkg):
            results = predict_phase_arrivals(stream, model_name="PhaseNet")

        assert results[0].station == "MYSTA"

    def test_trace_id_without_dots(self):
        """If trace_id has no dots, the whole string is used as station."""
        from swe.picking.deep_learning import predict_phase_arrivals

        t0 = UTCDateTime("2024-01-01T00:00:00.0Z")
        stream = _make_stream(["STA1"], starttime=t0)
        picks = [
            SimpleNamespace(
                trace_id="RAWID",
                phase="P",
                peak_time=t0 + 5.0,
                peak_value=0.7,
            )
        ]
        mock_sbm, _, _, mock_pkg = _make_mock_seisbench(picks)

        with _patch_seisbench(mock_sbm, mock_pkg):
            results = predict_phase_arrivals(stream, model_name="PhaseNet")

        assert results[0].station == "RAWID"

    def test_no_picks_returns_empty(self):
        """No model output yields an empty list."""
        from swe.picking.deep_learning import predict_phase_arrivals

        stream = _make_stream(["STA1"])
        mock_sbm, _, _, mock_pkg = _make_mock_seisbench([])

        with _patch_seisbench(mock_sbm, mock_pkg):
            results = predict_phase_arrivals(stream, model_name="PhaseNet")

        assert results == []

    def test_unsupported_model_raises(self):
        """An unsupported model_name should raise ValueError."""
        from swe.picking.deep_learning import predict_phase_arrivals

        stream = _make_stream(["STA1"])
        mock_sbm, _, _, mock_pkg = _make_mock_seisbench([])

        with (
            _patch_seisbench(mock_sbm, mock_pkg),
            pytest.raises(ValueError, match="Unsupported model"),
        ):
            predict_phase_arrivals(stream, model_name="NotAModel")

    def test_import_error_without_seisbench(self):
        """Missing seisbench raises a clear ImportError."""
        from swe.picking.deep_learning import predict_phase_arrivals

        stream = _make_stream(["STA1"])

        # Ensure seisbench is not importable
        with (
            patch.dict("sys.modules", {"seisbench": None, "seisbench.models": None}),
            pytest.raises(ImportError, match="seisbench is required"),
        ):
            predict_phase_arrivals(stream, model_name="PhaseNet")

    def test_eqtransformer_model(self):
        """EQTransformer model name is accepted."""
        from swe.picking.deep_learning import predict_phase_arrivals

        t0 = UTCDateTime("2024-01-01T00:00:00.0Z")
        stream = _make_stream(["STA1"], starttime=t0)
        picks = [
            SimpleNamespace(
                trace_id="XX.STA1..HHZ",
                phase="P",
                peak_time=t0 + 10.0,
                peak_value=0.8,
            )
        ]
        mock_sbm, _, _, mock_pkg = _make_mock_seisbench(picks)

        with _patch_seisbench(mock_sbm, mock_pkg):
            results = predict_phase_arrivals(stream, model_name="EQTransformer")

        assert len(results) == 1
        assert results[0].phase == "P"

    def test_custom_thresholds_passed(self):
        """Custom thresholds are forwarded to model.classify()."""
        from swe.picking.deep_learning import predict_phase_arrivals

        stream = _make_stream(["STA1"])
        mock_sbm, _, mock_model, mock_pkg = _make_mock_seisbench([])

        with _patch_seisbench(mock_sbm, mock_pkg):
            predict_phase_arrivals(
                stream,
                model_name="PhaseNet",
                p_threshold=0.5,
                s_threshold=0.6,
                batch_size=128,
            )

        mock_model.classify.assert_called_once_with(
            stream,
            batch_size=128,
            P_threshold=0.5,
            S_threshold=0.6,
        )

    def test_pretrained_parameter_forwarded(self):
        """The pretrained weight set name is passed to from_pretrained."""
        from swe.picking.deep_learning import predict_phase_arrivals

        stream = _make_stream(["STA1"])
        mock_sbm, mock_cls, _, mock_pkg = _make_mock_seisbench([])

        with _patch_seisbench(mock_sbm, mock_pkg):
            predict_phase_arrivals(
                stream,
                model_name="PhaseNet",
                pretrained="stead",
            )

        mock_cls.from_pretrained.assert_called_once_with("stead")


# ---------------------------------------------------------------------------
# Tests for pick_stream
# ---------------------------------------------------------------------------


class TestPickStream:
    """Tests for pick_stream convenience wrapper."""

    def test_returns_swe_pick_list(self):
        """pick_stream returns list[SWEPick]."""
        from swe.picking.deep_learning import pick_stream
        from swe.picks import SWEPick

        stations = ["STA1", "STA2"]
        t0 = UTCDateTime("2024-01-01T00:00:00.0Z")
        stream = _make_stream(stations, starttime=t0)
        mock_picks = _make_mock_picks(stations, t0)
        mock_sbm, _, _, mock_pkg = _make_mock_seisbench(mock_picks)

        with _patch_seisbench(mock_sbm, mock_pkg):
            result = pick_stream(stream, model_name="PhaseNet")

        assert isinstance(result, list)
        assert len(result) == 4  # 2 stations x (P + S)
        assert all(isinstance(p, SWEPick) for p in result)
        # Phase and confidence preserved
        phases = {p.phase for p in result}
        assert phases == {"P", "S"}
        assert all(p.confidence > 0 for p in result)

    def test_preserves_phase_info(self):
        """Phase type is preserved in each SWEPick."""
        from swe.picking.deep_learning import pick_stream

        t0 = UTCDateTime("2024-01-01T00:00:00.0Z")
        stream = _make_stream(["STA1"], starttime=t0)
        picks = [
            SimpleNamespace(
                trace_id="XX.STA1..HHZ",
                phase="S",
                peak_time=t0 + 20.0,
                peak_value=0.7,
            ),
            SimpleNamespace(
                trace_id="XX.STA1..HHZ",
                phase="P",
                peak_time=t0 + 10.0,
                peak_value=0.9,
            ),
        ]
        mock_sbm, _, _, mock_pkg = _make_mock_seisbench(picks)

        with _patch_seisbench(mock_sbm, mock_pkg):
            result = pick_stream(stream, model_name="PhaseNet")

        p_picks = [p for p in result if p.phase == "P"]
        s_picks = [p for p in result if p.phase == "S"]
        assert len(p_picks) == 1
        assert len(s_picks) == 1

    def test_empty_stream_returns_empty_list(self):
        """No picks yields an empty list."""
        from swe.picking.deep_learning import pick_stream

        stream = _make_stream(["STA1"])
        mock_sbm, _, _, mock_pkg = _make_mock_seisbench([])

        with _patch_seisbench(mock_sbm, mock_pkg):
            result = pick_stream(stream, model_name="PhaseNet")

        assert result == []

    def test_convertible_to_station_dict(self):
        """Output can be converted to dict format via to_station_picks.

        This ensures backward compatibility with associate_picks.
        """
        from swe.picking.deep_learning import pick_stream
        from swe.picks import to_station_picks

        stations = ["STA1", "STA2"]
        t0 = UTCDateTime("2024-01-01T00:00:00.0Z")
        stream = _make_stream(stations, starttime=t0)
        mock_picks = _make_mock_picks(stations, t0)
        mock_sbm, _, _, mock_pkg = _make_mock_seisbench(mock_picks)

        with _patch_seisbench(mock_sbm, mock_pkg):
            swe_picks = pick_stream(stream, model_name="PhaseNet")

        station_picks, pick_phases = to_station_picks(swe_picks)

        assert isinstance(station_picks, dict)
        assert "STA1" in station_picks
        assert "STA2" in station_picks
        for _station, times in station_picks.items():
            assert isinstance(times, list)
            for t in times:
                assert isinstance(t, UTCDateTime)
        # Phases are also preserved
        assert "STA1" in pick_phases
        for phase in pick_phases["STA1"]:
            assert phase in ("P", "S")
