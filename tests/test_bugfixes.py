"""Regression tests for verified bug fixes.

Each test targets a specific bug from the comprehensive audit.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime


class _PicklableForHmacTest:
    """Module-level class so pickle can serialize it."""

    def __init__(self, v=42):
        self.v = v


# ---------------------------------------------------------------------------
# Bug #4: IndexError in marker file NSLC parsing
# ---------------------------------------------------------------------------
class TestBug4NslcGuard:
    """NSLC lines with fewer than 4 dot-separated parts must be skipped."""

    def test_incomplete_nslc_skipped(self, tmp_path):
        """Lines with only 2 dot-parts (e.g. 'NET.STA') should not crash."""
        from swe.detection.matched_filter import markers_to_template

        # Only 2-part NSLC: "YG.STA1" - missing location and channel
        content = (
            "phase: 2020-01-01 00:00:10.0000  0 "
            "YG.STA1                 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 2020-01-01   "
            "00:00:10.0000 S        None False\n"
        )
        marker_path = tmp_path / "short_nslc.mrkr"
        marker_path.write_text(content)

        result = markers_to_template(marker_path, prepick_offset=2.0)
        assert result.templates == []
        assert result.station_dict == {}

    def test_three_part_nslc_skipped(self, tmp_path):
        """Lines with 3-part NSLC should also be skipped."""
        from swe.detection.matched_filter import markers_to_template

        content = (
            "phase: 2020-01-01 00:00:10.0000  0 "
            "YG.STA1.LOC              xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 2020-01-01   "
            "00:00:10.0000 S        None False\n"
        )
        marker_path = tmp_path / "three_nslc.mrkr"
        marker_path.write_text(content)

        result = markers_to_template(marker_path, prepick_offset=2.0)
        assert result.templates == []

    def test_four_part_nslc_accepted(self, tmp_path):
        """Full 4-part NSLC should be parsed correctly."""
        from swe.detection.matched_filter import markers_to_template

        content = (
            "phase: 2020-01-01 00:00:10.0000  0 "
            "YG.STA1..BHZ    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 2020-01-01   "
            "00:00:10.0000 S        None False\n"
        )
        marker_path = tmp_path / "full_nslc.mrkr"
        marker_path.write_text(content)

        result = markers_to_template(marker_path, prepick_offset=2.0)
        assert "STA1" in result.station_dict


# ---------------------------------------------------------------------------
# Bug #1: Unsafe pickle deserialization - HMAC signing
# ---------------------------------------------------------------------------
class TestBug1HmacSigning:
    """Pickle files must be HMAC-signed; unsigned files are rejected.

    Pickle is used here as an intentional fallback for complex scientific
    objects (ObsPy Stream/Catalog) that cannot be JSON-serialized.  HMAC
    signing mitigates the risk of loading tampered files.
    """

    def test_signed_pickle_roundtrips(self, tmp_path):
        from swe.pipeline.serialization import deserialize_output, serialize_output

        fmt = serialize_output(tmp_path, "s1", _PicklableForHmacTest(42))
        assert fmt == "pickle"
        result = deserialize_output(tmp_path, "s1")
        assert result.v == 42

    def test_unsigned_pickle_rejected(self, tmp_path):
        """A pickle without an HMAC sig file should be rejected.

        This test intentionally creates raw pickle files to verify the
        HMAC guard rejects them.  This is a security test, not production code.
        """
        import pickle

        from swe.pipeline.serialization import deserialize_output

        pkl_path = tmp_path / "s1.output.pkl"
        meta_path = tmp_path / "s1.output.pkl.meta"
        with open(pkl_path, "wb") as fh:
            pickle.dump(_PicklableForHmacTest(), fh)

        import swe

        meta_path.write_text(
            json.dumps(
                {
                    "swe_version": swe.__version__,
                    "python_version": "3.x",
                }
            )
        )

        with pytest.raises(FileNotFoundError, match="Unsigned or tampered"):
            deserialize_output(tmp_path, "s1")

    def test_tampered_pickle_rejected(self, tmp_path):
        """A pickle with a modified sig should be rejected."""
        from swe.pipeline.serialization import deserialize_output, serialize_output

        serialize_output(tmp_path, "s1", _PicklableForHmacTest())

        # Tamper with the signature
        sig_path = tmp_path / "s1.output.pkl.sig"
        sig_path.write_text("0000dead0000")

        with pytest.raises(FileNotFoundError, match="Unsigned or tampered"):
            deserialize_output(tmp_path, "s1")

    def test_json_still_works_without_sig(self, tmp_path):
        """JSON serialization should still work (no HMAC needed)."""
        from swe.pipeline.serialization import deserialize_output, serialize_output

        fmt = serialize_output(tmp_path, "s1", {"key": "val"})
        assert fmt == "json"
        result = deserialize_output(tmp_path, "s1")
        assert result == {"key": "val"}


# ---------------------------------------------------------------------------
# Bug #12: Broad exception catch in serialization
# ---------------------------------------------------------------------------
class TestBug12NarrowExceptionCatch:
    """Serialization should only catch serialization-related exceptions."""

    def test_specific_pickle_error_caught(self, tmp_path):
        from swe.pipeline.serialization import serialize_output

        obj = lambda: None  # lambdas are not picklable  # noqa: E731
        fmt = serialize_output(tmp_path, "s1", obj)
        assert fmt == "none"


# ---------------------------------------------------------------------------
# Bug #2 + #3: Race conditions in parallel pipeline
# ---------------------------------------------------------------------------
class TestBug2And3RaceConditions:
    """Parallel pipeline execution must be thread-safe."""

    def test_parallel_execution_does_not_lose_outputs(self, tmp_path):
        """All parallel steps should appear in outputs dict."""
        from swe.pipeline.core import Pipeline, PipelineStep

        # 3 steps with no dependencies (all in same tier)
        pipeline = Pipeline(
            steps=[
                PipelineStep(name="a", function="os.getcwd", params={}),
                PipelineStep(name="b", function="os.getcwd", params={}),
                PipelineStep(name="c", function="os.getcwd", params={}),
            ]
        )

        call_count = {"n": 0}
        lock = threading.Lock()

        def _slow_fn(**kwargs):
            with lock:
                call_count["n"] += 1
                n = call_count["n"]
            time.sleep(0.01)  # simulate work
            return {"result": n}

        with patch("swe.pipeline.core._import_callable", return_value=_slow_fn):
            outputs = pipeline.execute(run_dir=tmp_path / "runs", parallel=True)

        assert "a" in outputs
        assert "b" in outputs
        assert "c" in outputs


# ---------------------------------------------------------------------------
# Bug #3: Provenance file locking
# ---------------------------------------------------------------------------
class TestBug3ProvenanceFileLocking:
    """Concurrent provenance writes should not lose records."""

    def test_concurrent_saves_preserve_all_records(self, tmp_path):
        from swe.pipeline.provenance import StepProvenance, load_provenance, save_provenance

        def _make_records(prefix: str, n: int) -> list[StepProvenance]:
            now = time.time()
            return [
                StepProvenance(
                    step_name=f"{prefix}_{i}",
                    function_path="mod.fn",
                    resolved_params={},
                    input_hash="abc",
                    started_at=now,
                    completed_at=now + 0.1,
                    duration_seconds=0.1,
                    return_type="dict",
                    cache_hit=False,
                )
                for i in range(n)
            ]

        threads = []
        for j in range(5):
            t = threading.Thread(
                target=save_provenance,
                args=(tmp_path, _make_records(f"batch{j}", 3)),
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        loaded = load_provenance(tmp_path)
        assert len(loaded) == 15  # 5 batches * 3 records


# ---------------------------------------------------------------------------
# Bug #5: Parameter resolution error messages
# ---------------------------------------------------------------------------
class TestBug5ParameterResolutionErrors:
    """Bad ${step.key} references should produce clear error messages."""

    def test_missing_step_raises_keyerror(self):
        from swe.pipeline.core import _resolve_params

        with pytest.raises(KeyError, match="not found in outputs"):
            _resolve_params({"x": "${nonexistent.key}"}, {})

    def test_missing_dict_key_raises_keyerror(self):
        from swe.pipeline.core import _resolve_params

        outputs = {"s1": {"a": 1}}
        with pytest.raises(KeyError, match=r"key 'missing'.*not found"):
            _resolve_params({"x": "${s1.missing}"}, outputs)

    def test_missing_attr_raises_attributeerror(self):
        from swe.pipeline.core import _resolve_params

        outputs = {"s1": SimpleNamespace(a=1)}
        with pytest.raises(AttributeError, match="has no attribute 'missing'"):
            _resolve_params({"x": "${s1.missing}"}, outputs)

    def test_valid_reference_still_works(self):
        from swe.pipeline.core import _resolve_params

        outputs = {"s1": {"path": "/data"}}
        result = _resolve_params({"x": "${s1.path}"}, outputs)
        assert result["x"] == "/data"


# ---------------------------------------------------------------------------
# Bug #6: NaN propagation in relative moment calculation
# ---------------------------------------------------------------------------
class TestBug6NanPropagation:
    """NaN det_amp should not corrupt the median ratio."""

    def test_nan_det_amp_excluded(self):
        from swe.magnitude.relative import relative_moments_from_streams

        # Template with known amplitude
        tmpl_tr = Trace(
            data=np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float64),
            header={"network": "XX", "station": "ST00", "channel": "BHZ", "sampling_rate": 100.0},
        )
        template = SimpleNamespace(name="test", st=Stream([tmpl_tr]))

        detection = SimpleNamespace(
            detect_val=0.8,
            no_chans=1,
            detect_time=UTCDateTime(2024, 1, 1),
            template_name="test",
        )
        family = SimpleNamespace(template=template, detections=[detection])

        # Detection stream with NaN data
        nan_tr = Trace(
            data=np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64),
            header={"network": "XX", "station": "ST00", "channel": "BHZ", "sampling_rate": 100.0},
        )
        det_streams = [Stream([nan_tr])]

        result = relative_moments_from_streams(family, det_streams)
        # The NaN ratio should be excluded; fallback to cc/n_chans
        assert len(result.relative_moments) == 1
        assert not np.isnan(result.relative_moments[0])


# ---------------------------------------------------------------------------
# Bug #7: IndexError in signal max-pool lookup
# ---------------------------------------------------------------------------
class TestBug7SignalMaxPoolLookup:
    """pick_time_index missing from max_pool_indices should not crash."""

    def test_get_trace_properties_with_edge_pick(self):
        from swe.utils.signal import get_trace_properties

        sr = 100.0
        npts = 2000
        data = np.random.default_rng(42).standard_normal(npts).astype(np.float64)
        tr = Trace(
            data=data,
            header={
                "sampling_rate": sr,
                "starttime": UTCDateTime(2024, 1, 1),
            },
        )
        pick_time = tr.stats.starttime + 10.0

        # Should not raise IndexError
        dy, _d2y, _curvature, _fits, mads = get_trace_properties(tr, pick_time, duration=5.0)
        assert len(dy) > 0
        assert len(mads) == 6


# ---------------------------------------------------------------------------
# Bug #8: Velocity lookup asymmetry at layer boundaries
# ---------------------------------------------------------------------------
class TestBug8VelocityLookupSymmetry:
    """Upward and downward rays at exact layer boundaries should use consistent velocity."""

    def test_symmetric_boundary_velocity(self):
        from swe.focal.takeoff import _compute_horizontal_distance

        # A 2-layer model: 4 km/s above 5 km, 6 km/s below
        depths = np.array([0.0, 5.0])
        velocities = np.array([4.0, 6.0])

        # Downward ray starting exactly at boundary (z=5.0)
        h_down = _compute_horizontal_distance(5.0, 10.0, 135.0, depths, velocities)
        # Should use the below-boundary velocity (6.0), not total internal reflection
        assert h_down < 1e6

    def test_1d_downward_ray_at_boundary(self):
        """Downward ray at exact layer boundary should get the correct layer."""
        from swe.focal.takeoff import compute_takeoff_angles_1d

        source = {"x": 0.0, "y": 0.0, "z": 5.0}  # at boundary
        stations = {"STA1": [10.0, 0.0, 10.0]}  # below
        profile = [[0.0, 4.0], [5.0, 6.0]]

        result = compute_takeoff_angles_1d(source, stations, velocity_profile=profile)
        assert len(result.stations) == 1
        assert 90.0 < result.takeoff_angles[0] < 180.0  # downgoing


# ---------------------------------------------------------------------------
# Bug #9: Window bounds off-by-one
# ---------------------------------------------------------------------------
class TestBug9WindowBounds:
    """times[index + window_length] must not go out of bounds."""

    def test_window_end_inclusive(self):
        from swe.picking.windows import find_picking_windows

        # Create a minimal MiniSEED file
        npts = 500
        sr = 250.0
        starttime = UTCDateTime(2024, 1, 1, 0, 0, 0)
        data = np.random.default_rng(42).standard_normal(npts).astype(np.float64)
        tr = Trace(
            data=data,
            header={
                "network": "XX",
                "station": "STA1",
                "location": "",
                "channel": "BHZ",
                "sampling_rate": sr,
                "starttime": starttime,
            },
        )

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".ms", delete=False) as tmp:
            tr.write(tmp.name, format="MSEED")
            tmp_path = tmp.name

        try:
            result = find_picking_windows(
                [tmp_path],
                window_length=240,
                window_offset=120,
                time_periods=[(starttime, starttime + npts / sr)],
                sampling_rate=sr,
            )
            # Should not crash with IndexError
            for ws, we, _sta, _cha in result.times:
                assert we >= ws
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Bug #10: Validate NonLinLoc binary exists
# ---------------------------------------------------------------------------
class TestBug10ValidateBinary:
    """run_nonlinloc should raise FileNotFoundError for missing binary."""

    def test_missing_binary_raises(self, tmp_path):
        from swe.location.nonlinloc import run_nonlinloc

        ctrl = tmp_path / "test.nll_control"
        ctrl.write_text("CONTROL\n")

        with pytest.raises(FileNotFoundError, match="NonLinLoc binary not found"):
            run_nonlinloc(ctrl, nll_binary="nonexistent_binary_xyz_12345")


# ---------------------------------------------------------------------------
# Bug #11: Silent phase misalignment
# ---------------------------------------------------------------------------
class TestBug11PhaseMisalignment:
    """Mismatched pick_phases length should raise ValueError."""

    def test_mismatched_lengths_raises(self):
        from swe.association._dispatch import picks_to_dataframe

        station_picks = {"STA1": [UTCDateTime(2024, 1, 1), UTCDateTime(2024, 1, 2)]}
        pick_phases = {"STA1": ["P"]}  # only 1 phase for 2 picks

        with pytest.raises(ValueError, match="matching lengths"):
            picks_to_dataframe(station_picks, pick_phases=pick_phases)

    def test_matching_lengths_works(self):
        from swe.association._dispatch import picks_to_dataframe

        station_picks = {"STA1": [UTCDateTime(2024, 1, 1), UTCDateTime(2024, 1, 2)]}
        pick_phases = {"STA1": ["P", "S"]}

        df = picks_to_dataframe(station_picks, pick_phases=pick_phases)
        assert len(df) == 2
        assert list(df["phase"]) == ["P", "S"]


# ---------------------------------------------------------------------------
# Bug #13: Coordinate validation in geo_to_local
# ---------------------------------------------------------------------------
class TestBug13CoordinateValidation:
    """2-element (lon, lat) tuples should not crash geo_to_local."""

    def test_two_element_coords_default_elevation(self):
        from swe.utils.helpers import geo_to_local

        location_dict = {
            "STA1": [-122.0, 37.0],  # no elevation
            "STA2": [-122.1, 37.1, 0.5],  # with elevation
        }
        local_coords, _origin = geo_to_local(location_dict)
        assert "STA1" in local_coords
        assert "STA2" in local_coords

    def test_all_two_element_coords(self):
        from swe.utils.helpers import geo_to_local

        location_dict = {
            "STA1": [-122.0, 37.0],
            "STA2": [-122.1, 37.1],
        }
        local_coords, origin = geo_to_local(location_dict)
        assert len(local_coords) == 2
        assert origin[2] == 0.0  # default elevation


# ---------------------------------------------------------------------------
# Bug #14: Named constant for meters per degree
# ---------------------------------------------------------------------------
class TestBug14NamedConstant:
    """Verify _METERS_PER_DEG_LAT is used and has correct value."""

    def test_constant_exists(self):
        from swe.location.nonlinloc import _METERS_PER_DEG_LAT

        assert _METERS_PER_DEG_LAT == 111_320.0

    def test_hypocenter_conversion_uses_constant(self):
        from swe.location.nonlinloc import HypocenterResult, hypocenter_to_obspy_origin

        hyp = HypocenterResult(
            x_km=1.0,
            y_km=2.0,
            z_m=3000.0,
            horiz_uncertainty_m=1000.0,
        )
        origin = hypocenter_to_obspy_origin(hyp, reference_lat=0.0, reference_lon=0.0)
        expected_deg = 1000.0 / 111_320.0
        assert abs(origin.latitude_errors.uncertainty - expected_deg) < 1e-10
