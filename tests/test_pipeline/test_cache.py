"""Tests for swe.pipeline.cache."""

from __future__ import annotations

import json

from swe.pipeline.cache import compute_step_hash, find_cached_result
from swe.pipeline.run_context import RunContext
from swe.pipeline.serialization import serialize_output


def _dummy_func(x=1):
    return x * 2


def _other_func(x=1):
    return x * 3


class TestComputeStepHash:
    """Tests for compute_step_hash."""

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
