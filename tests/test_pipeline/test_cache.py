"""Tests for swe.pipeline.cache."""

from __future__ import annotations

import collections
import copy
import dataclasses
import datetime
import enum
import json
import re
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import numpy as np
import pytest
from obspy import UTCDateTime
from obspy.core.event import Event, Origin

from swe.pipeline import cache
from swe.pipeline.cache import compute_step_hash, find_cached_result
from swe.pipeline.run_context import RunContext
from swe.pipeline.serialization import serialize_output


def _dummy_func(x=1):
    return x * 2


def _other_func(x=1):
    return x * 3


def _h(value):
    return compute_step_hash("mod.fn", {"v": value}, _dummy_func)


class _Color(enum.IntEnum):
    RED = 1
    GREEN = 2


@dataclasses.dataclass(frozen=True)
class _Pt:
    x: int


class _PtWithNote(_Pt):  # not a dataclass itself: ``note`` is state outside the fields
    def __init__(self, x, note):
        super().__init__(x)
        object.__setattr__(self, "note", note)


@dataclasses.dataclass
class _Unset:
    x: float
    y: float = dataclasses.field(init=False)


@dataclasses.dataclass
class _DictDC(dict):  # a dataclass whose contents live in the dict, outside its fields
    name: str = "n"

    def __init__(self, items):
        dict.__init__(self, items)
        self.name = "n"


class _FakeFields(tuple):  # not a NamedTuple: _fields covers one of two elements
    _fields = ("a",)
    a = property(lambda s: s[0])


def _zone_from_file(d):
    import zoneinfo

    try:
        tz = zoneinfo.ZoneInfo("UTC" if d == 1.0 else "Etc/GMT+1")
    except zoneinfo.ZoneInfoNotFoundError:  # no system tz database or tzdata package
        pytest.skip("no IANA time zone database on this host")
    return datetime.datetime(2020, 1, 1, tzinfo=tz)


def _void_with_object(d):
    a = np.zeros(1, dtype=[("a", int), ("b", object)])
    a[0]["b"] = [d]
    return a[0]


class _TZ(datetime.tzinfo):  # no __repr__: its repr is an address
    def __init__(self, h):
        self.h = h

    def utcoffset(self, dt):
        return datetime.timedelta(hours=self.h)


@dataclasses.dataclass(eq=False)
class _DequeDC(collections.deque):  # contents live in the deque, outside the fields
    name: str = "n"

    def __init__(self, items):
        collections.deque.__init__(self, items)
        self.name = "n"


def _nan_payload(n):
    import struct

    return struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000 | n))[0]


@dataclasses.dataclass(eq=False)
class _TupleDC(tuple):  # contents live in the tuple, outside the fields
    name: str = "n"

    def __new__(cls, items):
        return tuple.__new__(cls, items)

    def __init__(self, items):
        self.name = "n"


class _IterSlots(_Pt):  # __slots__ given as an iterator: exhausted once the class is built
    __slots__ = iter(("note",))

    def __init__(self, x, note):
        super().__init__(x)
        object.__setattr__(self, "note", note)


class _DictProp(_Pt):  # a property named __dict__ hides the real instance dict
    __dict__ = property(lambda s: {"x": 1})

    def __init__(self, x, note):
        super().__init__(x)
        object.__setattr__(self, "note", note)


@dataclasses.dataclass
class _HidingGetattr:  # reports a clean __dict__ while storing more
    x: int

    def __init__(self, x, note):
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "note", note)

    def __getattribute__(self, name):
        if name == "__dict__":
            return {"x": object.__getattribute__(self, "x")}
        return object.__getattribute__(self, name)


_SIDE: dict = {}


class _Undecorated(_Pt):  # not decorated: its tag lives in a side table, outside the fields
    def __init__(self, x, tag):
        super().__init__(x)
        _SIDE[id(self)] = tag


def _distinct_numpy_int_types():
    """np.longlong, np.int64 and np.int32 values, minus aliases (on Windows int64 is longlong)."""
    seen, out = set(), []
    for t in (np.longlong, np.int64, np.int32):
        if t not in seen:
            seen.add(t)
            out.append(t(1))
    return out


def _utc_with_attribute(d):
    import warnings

    t = UTCDateTime(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # older obspy warns on setting a custom attribute
        try:
            t.note = d
        except AttributeError:  # newer obspy (__slots__) forbids it: nothing to hide
            pytest.skip("this obspy forbids instance attributes on UTCDateTime")
    return t


def _readonly(a):
    a.flags.writeable = False
    return a


class _SlotNote(_Pt):  # extra state in __slots__, outside the dataclass fields
    __slots__ = ("note",)

    def __init__(self, x, note):
        super().__init__(x)
        object.__setattr__(self, "note", note)


class _NotePath(PurePosixPath):  # a path subclass with a __dict__, so it can carry state
    pass


def _note_path(note):
    p = _NotePath("a")
    p.note = note
    return p


class _MyTime(UTCDateTime):
    pass


class _Pair(NamedTuple):
    a: int
    b: int


_BIG = np.arange(2000.0)
_BIG_CHANGED = _BIG.copy()
_BIG_CHANGED[1000] = -1.0


class TestComputeStepHash:
    """Tests for compute_step_hash."""

    def test_stream_param_makes_the_step_uncacheable(self):
        """A Stream has no exact key: a fresh hash every time, never a shared one."""
        from obspy import Stream, Trace

        p = {"stream": Stream([Trace(np.zeros(10))])}
        assert compute_step_hash("mod.fn", p, _dummy_func) != compute_step_hash(
            "mod.fn", p, _dummy_func
        )

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([{1: "a"}, {"1": "a"}], id="int-vs-str-dict-keys"),
            pytest.param([(1, 2), [1, 2]], id="tuple-vs-list"),
            pytest.param([10**5000, 10**5000 + 1], id="ints-beyond-the-json-digit-limit"),
            pytest.param([0.5, np.float32(0.5), np.float64(0.5)], id="numpy-scalar-dtype"),
            pytest.param([1, np.int64(1), True], id="int-numpy-bool"),
            pytest.param(
                [
                    datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(
                        2020, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(hours=1))
                    ),
                    datetime.datetime(2020, 1, 1),
                ],
                id="timezone-aware",
            ),
            pytest.param(
                [
                    np.array(["2020-01-01"], dtype="datetime64[D]"),
                    np.array(["2020-01-02"], dtype="datetime64[D]"),
                ],
                id="datetime64-arrays-keyed",
            ),
            pytest.param([np.array([1, 2], dtype=np.int32), np.array([1, 2]), [1, 2]], id="dtype"),
            pytest.param([np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(4)], id="shape"),
            pytest.param(
                [np.arange(6.0).reshape(2, 3), np.asfortranarray(np.arange(6.0).reshape(2, 3))],
                id="memory-order",
            ),
            pytest.param([{"a": 1, "b": 2}, {"b": 2, "a": 1}], id="dict-order"),
            pytest.param([float("nan"), _nan_payload(1), 0.0, -0.0], id="float-bit-patterns"),
            pytest.param(_distinct_numpy_int_types(), id="numpy-scalar-types"),
            pytest.param([(_BIG, {1}), (_BIG_CHANGED, {1})], id="large-array-elided-by-repr"),
            pytest.param([(np.array([0.1]), {1}), (np.array([0.1 + 1e-12]), {1})], id="rounded"),
            pytest.param(
                [UTCDateTime(ns=1), UTCDateTime(ns=2), "1970-01-01T00:00:00.000000Z"],
                id="utcdatetime-ns",
            ),
            pytest.param(
                [
                    *({"x": 1, "__type__": "_Pt"}, Path("a"), {"__path__": "a"}, "a"),
                    *(Decimal("1"), {1}, frozenset({1})),
                    *(datetime.date(2020, 1, 1), datetime.datetime(2020, 1, 1)),
                ],
                id="tagged-value-vs-dict",
            ),
        ],
    )
    def test_different_values_never_share_a_key(self, values):
        """Equal values (fresh copies) share a key; different types or contents never do."""
        keys = [_h(v) for v in values]
        assert keys == [_h(copy.deepcopy(v)) for v in values]
        assert len(set(keys)) == len(keys)

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(
                lambda d: Event(
                    resource_id="smi:local/e", origins=[Origin(resource_id="smi:local/o", depth=d)]
                ),
                id="event-repr-omits-origins",
            ),
            pytest.param(lambda d: _Pt(int(d)), id="dataclass"),
            pytest.param(lambda d: _Pair(1, d), id="namedtuple"),
            pytest.param(lambda d: np.ma.array([1.0, d]), id="masked-array"),
            pytest.param(_utc_with_attribute, id="utcdatetime-with-instance-attribute"),
            pytest.param(lambda d: _PtWithNote(1, d), id="state-outside-dataclass-fields"),
            pytest.param(_Unset, id="unset-dataclass-field"),
            pytest.param(lambda d: _DictDC({d: 1}), id="dataclass-subclassing-dict"),
            pytest.param(_void_with_object, id="numpy-scalar-with-object-field"),
            pytest.param(lambda d: _Color.RED if d == 1.0 else _Color.GREEN, id="enum-member"),
            pytest.param(lambda d: re.I | (re.M if d == 1.0 else re.S), id="flag"),
            pytest.param(lambda d: _FakeFields((1, d)), id="tuple-subclass-with-fake-fields"),
            pytest.param(_zone_from_file, id="zoneinfo"),
            pytest.param(
                lambda d: datetime.datetime(2020, 1, 1, tzinfo=_TZ(d)), id="custom-tzinfo"
            ),
            pytest.param(lambda d: np.ma.array(np.matrix([[1.0, d]])), id="masked-matrix"),
            pytest.param(lambda d: _DequeDC([d]), id="dataclass-subclassing-deque"),
            pytest.param(lambda d: _TupleDC((d,)), id="dataclass-subclassing-tuple"),
            pytest.param(lambda d: _IterSlots(1, d), id="slots-given-as-an-iterator"),
            pytest.param(lambda d: _DictProp(1, d), id="dict-hidden-by-a-property"),
            pytest.param(lambda d: _HidingGetattr(1, d), id="custom-getattribute"),
            pytest.param(lambda d: _Undecorated(1, d), id="undecorated-dataclass-subclass"),
            pytest.param(lambda d: _SlotNote(1, d), id="state-in-slots-outside-fields"),
            pytest.param(_note_path, id="path-subclass-with-state"),
            pytest.param(lambda d: _MyTime(ns=int(d)), id="utcdatetime-subclass"),
        ],
    )
    def test_value_without_exact_key_is_uncacheable(self, make):
        """No exact key: a fresh key each time (never looked up), not a shared one or a crash."""
        a, b = make(1.0), make(2.0)
        assert _h(a) != _h(b)
        assert _h(a).startswith(cache.UNCACHEABLE)
        with pytest.raises(cache._NoExactKey):  # rejected by design, not by a crash
            cache._key(a)

    def test_deterministic(self):
        """Same inputs should produce the same hash."""
        h1 = compute_step_hash("mod.fn", {"a": 1}, _dummy_func)
        h2 = compute_step_hash("mod.fn", {"a": 1}, _dummy_func)
        assert h1 == h2

    def test_different_params_differ(self):
        """Different params should produce different hashes."""
        h1 = compute_step_hash("mod.fn", {"a": 1}, _dummy_func)
        h2 = compute_step_hash("mod.fn", {"a": 2}, _dummy_func)
        assert h1 != h2

    def test_different_function_path_differ(self):
        """Different function paths should produce different hashes."""
        h1 = compute_step_hash("mod.fn1", {"a": 1}, _dummy_func)
        h2 = compute_step_hash("mod.fn2", {"a": 1}, _dummy_func)
        assert h1 != h2

    def test_different_source_differ(self):
        """Different function source should produce different hashes."""
        h1 = compute_step_hash("mod.fn", {"a": 1}, _dummy_func)
        h2 = compute_step_hash("mod.fn", {"a": 1}, _other_func)
        assert h1 != h2

    def test_returns_hex_string(self):
        """Hash should be a valid hex string."""
        h = compute_step_hash("mod.fn", {}, _dummy_func)
        assert len(h) == 64  # SHA-256
        int(h, 16)  # should not raise


class TestFindCachedResult:
    """Tests for find_cached_result."""

    def test_finds_matching_cache(self, tmp_path):
        """Should return cached output when hash matches."""
        ctx = RunContext.create(tmp_path)
        serialize_output(ctx.steps_dir, "s1", {"result": 42})
        ctx.mark_step_completed("s1", "hash_abc")

        result = find_cached_result("s1", "hash_abc", tmp_path)
        assert result == {"result": 42}

    def test_unloadable_cached_pickle_is_a_miss(self, tmp_path):
        """A cached pickle whose class no longer imports is a miss, not an exception."""
        from swe.pipeline.serialization import _sign_file

        ctx = RunContext.create(tmp_path)
        serialize_output(ctx.steps_dir, "s1", (1, 2))
        ctx.mark_step_completed("s1", "h")
        pkl = ctx.steps_dir / "s1.output.pkl"
        pkl.write_bytes(b"cno_such_module_xyz\nCls\n)\x81.")
        _sign_file(pkl)
        assert find_cached_result("s1", "h", tmp_path) is None

    def test_returns_none_on_miss(self, tmp_path):
        """Should return None when no cache matches."""
        ctx = RunContext.create(tmp_path)
        serialize_output(ctx.steps_dir, "s1", {"result": 42})
        ctx.mark_step_completed("s1", "hash_abc")

        result = find_cached_result("s1", "different_hash", tmp_path)
        assert result is None

    def test_returns_none_when_empty(self, tmp_path):
        """Should return None when no runs exist."""
        result = find_cached_result("s1", "any_hash", tmp_path)
        assert result is None

    def test_returns_none_for_nonexistent_dir(self, tmp_path):
        """Should return None when base_dir doesn't exist."""
        result = find_cached_result("s1", "any_hash", tmp_path / "nope")
        assert result is None

    def test_prefers_most_recent_run(self, tmp_path):
        """Should scan most recent runs first."""
        # Create two runs with deterministic ordering by using
        # explicit directory names rather than auto-generated IDs.

        run1_dir = tmp_path / "20240101T000000_aaaaaaaa"
        run1_steps = run1_dir / "steps"
        run1_steps.mkdir(parents=True)
        serialize_output(run1_steps, "s1", {"run": 1})
        manifest1 = {
            "run_id": "20240101T000000_aaaaaaaa",
            "steps": {"s1": {"status": "completed", "hash": "same_hash"}},
        }
        (run1_dir / "manifest.json").write_text(json.dumps(manifest1))

        run2_dir = tmp_path / "20240102T000000_bbbbbbbb"
        run2_steps = run2_dir / "steps"
        run2_steps.mkdir(parents=True)
        serialize_output(run2_steps, "s1", {"run": 2})
        manifest2 = {
            "run_id": "20240102T000000_bbbbbbbb",
            "steps": {"s1": {"status": "completed", "hash": "same_hash"}},
        }
        (run2_dir / "manifest.json").write_text(json.dumps(manifest2))

        result = find_cached_result("s1", "same_hash", tmp_path)
        assert result == {"run": 2}


def test_hashing_a_masked_array_leaves_it_unchanged():
    """Reading the key must not set a masked array's fill value (the property does)."""
    m = np.ma.array([1.0, 2.0])
    assert m._fill_value is None
    compute_step_hash("mod.fn", {"m": m}, _dummy_func)
    assert m._fill_value is None


def test_read_only_array_is_keyed_apart_from_a_writable_one():
    """In-place writes fail on a read-only input, so the two must not share a cached output."""
    a, b = np.arange(3.0), _readonly(np.arange(3.0))
    assert compute_step_hash("mod.fn", {"a": a}, _dummy_func) != compute_step_hash(
        "mod.fn", {"a": b}, _dummy_func
    )
