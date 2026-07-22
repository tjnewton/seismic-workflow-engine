"""Tests for swe.pipeline.serialization."""

from __future__ import annotations

import json

import numpy as np
import pytest

from swe.pipeline.serialization import (
    SWEJSONEncoder,
    _VersionMismatchError,
    deserialize_output,
    serialize_output,
)


class _PicklableCustom:
    """Module-level class so pickle can serialize it."""

    def __init__(self, value=42):
        self.value = value


class TestSWEJSONEncoder:
    """Tests for the custom JSON encoder."""

    def test_numpy_int(self):
        result = json.loads(json.dumps({"x": np.int64(42)}, cls=SWEJSONEncoder))
        assert result["x"] == 42

    def test_numpy_float(self):
        result = json.loads(json.dumps({"x": np.float64(3.14)}, cls=SWEJSONEncoder))
        assert abs(result["x"] - 3.14) < 1e-10

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = json.loads(json.dumps({"x": arr}, cls=SWEJSONEncoder))
        assert result["x"] == [1, 2, 3]

    def test_utcdatetime(self):
        from obspy import UTCDateTime

        t = UTCDateTime("2024-01-15T12:00:00")
        result = json.loads(json.dumps({"t": t}, cls=SWEJSONEncoder))
        assert "2024-01-15" in result["t"]

    def test_namedtuple(self):
        from collections import namedtuple

        Point = namedtuple("Point", ["x", "y"])
        p = Point(1, 2)
        # namedtuples are tuples, so JSON encodes them as arrays
        result = json.loads(json.dumps({"p": p}, cls=SWEJSONEncoder))
        assert result["p"] == [1, 2]

    def test_plain_dict_passthrough(self):
        """Plain dicts should serialize without custom encoding."""
        result = json.loads(json.dumps({"d": {"a": 1, "b": 2}}, cls=SWEJSONEncoder))
        assert result["d"] == {"a": 1, "b": 2}


class TestSerializeOutput:
    """Tests for serialize_output / deserialize_output."""

    def test_dict_uses_json(self, tmp_path):
        """Plain dicts should be stored as JSON."""
        fmt = serialize_output(tmp_path, "s1", {"key": "val"})
        assert fmt == "json"
        assert (tmp_path / "s1.output.json").exists()

    def test_list_uses_json(self, tmp_path):
        """Lists should be stored as JSON."""
        fmt = serialize_output(tmp_path, "s1", [1, 2, 3])
        assert fmt == "json"

    def test_roundtrip_json(self, tmp_path):
        """JSON-serialized output should round-trip correctly."""
        original = {"a": 1, "b": [2, 3], "c": "hello"}
        serialize_output(tmp_path, "s1", original)
        result = deserialize_output(tmp_path, "s1")
        assert result == original

    def test_complex_object_uses_pickle(self, tmp_path):
        """Objects that can't be JSON-encoded should use pickle.

        Pickle is used only for complex scientific objects (ObsPy types)
        that cannot be JSON-serialized, with version guards on deserialization.
        """
        fmt = serialize_output(tmp_path, "s1", _PicklableCustom())
        assert fmt == "pickle"
        assert (tmp_path / "s1.output.pkl").exists()
        assert (tmp_path / "s1.output.pkl.meta").exists()

    def test_roundtrip_pickle(self, tmp_path):
        """Pickle-serialized output should round-trip correctly.

        Pickle is used only for complex scientific objects (ObsPy types)
        that cannot be JSON-serialized, with version guards on deserialization.
        """
        serialize_output(tmp_path, "s1", _PicklableCustom(99))
        result = deserialize_output(tmp_path, "s1")
        assert result.value == 99

    def test_pickle_version_mismatch(self, tmp_path):
        """Pickle with different SWE version should raise _VersionMismatchError.

        Pickle is used only for complex scientific objects (ObsPy types)
        that cannot be JSON-serialized, with version guards on deserialization.
        """
        serialize_output(tmp_path, "s1", object())

        # Tamper with the metadata
        meta_path = tmp_path / "s1.output.pkl.meta"
        meta = json.loads(meta_path.read_text())
        meta["swe_version"] = "99.99.99"
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(_VersionMismatchError):
            deserialize_output(tmp_path, "s1")

    def test_deserialize_missing_step(self, tmp_path):
        """Should raise FileNotFoundError for missing step output."""
        with pytest.raises(FileNotFoundError):
            deserialize_output(tmp_path, "nonexistent")

    def test_numpy_array_roundtrip(self, tmp_path):
        """Numpy arrays should serialize as JSON lists."""
        arr = np.array([1.0, 2.0, 3.0])
        serialize_output(tmp_path, "s1", {"data": arr})
        result = deserialize_output(tmp_path, "s1")
        assert result["data"] == [1.0, 2.0, 3.0]
