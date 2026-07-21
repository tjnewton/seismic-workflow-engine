"""Tests for swe.pipeline.run_context."""

from __future__ import annotations

import json

import pytest

from swe.pipeline.run_context import RunContext, _find_latest_run, _generate_run_id


class TestRunContext:
    """Tests for RunContext create/load lifecycle."""

    def test_create_produces_directory(self, tmp_path):
        """create() should set up run_dir, steps_dir, and manifest."""
        ctx = RunContext.create(tmp_path)
        assert ctx.run_dir.exists()
        assert ctx.steps_dir.exists()
        assert (ctx.run_dir / "manifest.json").exists()

    def test_manifest_contains_run_id(self, tmp_path):
        """Manifest should record the run ID."""
        ctx = RunContext.create(tmp_path)
        manifest = json.loads((ctx.run_dir / "manifest.json").read_text())
        assert manifest["run_id"] == ctx.run_id

    def test_load_by_run_id(self, tmp_path):
        """load() should restore a previously created context."""
        ctx1 = RunContext.create(tmp_path)
        ctx2 = RunContext.load(tmp_path, ctx1.run_id)
        assert ctx2.run_id == ctx1.run_id
        assert ctx2.run_dir == ctx1.run_dir

    def test_load_latest(self, tmp_path):
        """load('latest') should find the most recent run."""
        import json

        # Create two runs with deterministic ordering
        run1_dir = tmp_path / "20240101T000000_aaaaaaaa"
        (run1_dir / "steps").mkdir(parents=True)
        (run1_dir / "manifest.json").write_text(
            json.dumps({"run_id": "20240101T000000_aaaaaaaa"})
        )

        run2_dir = tmp_path / "20240102T000000_bbbbbbbb"
        (run2_dir / "steps").mkdir(parents=True)
        (run2_dir / "manifest.json").write_text(
            json.dumps({"run_id": "20240102T000000_bbbbbbbb"})
        )

        loaded = RunContext.load(tmp_path, "latest")
        assert loaded.run_id == "20240102T000000_bbbbbbbb"

    def test_load_nonexistent_raises(self, tmp_path):
        """load() with a bad run ID should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            RunContext.load(tmp_path, "nonexistent_run_id")

    def test_load_latest_empty_raises(self, tmp_path):
        """load('latest') with no runs should raise FileNotFoundError."""
        tmp_path.mkdir(exist_ok=True)
        with pytest.raises(FileNotFoundError):
            RunContext.load(tmp_path, "latest")

    def test_mark_step_completed(self, tmp_path):
        """Completed steps should be recorded in the manifest."""
        ctx = RunContext.create(tmp_path)
        ctx.mark_step_completed("step_a", "abc123")
        assert ctx.is_step_completed("step_a")
        assert not ctx.is_step_completed("step_b")

    def test_mark_step_failed(self, tmp_path):
        """Failed steps should be recorded but not counted as completed."""
        ctx = RunContext.create(tmp_path)
        ctx.mark_step_failed("step_a")
        assert not ctx.is_step_completed("step_a")

        manifest = json.loads((ctx.run_dir / "manifest.json").read_text())
        assert manifest["steps"]["step_a"]["status"] == "failed"
        assert manifest["status"] == "failed"

    def test_get_step_output_json(self, tmp_path):
        """get_step_output should deserialize JSON-stored outputs."""
        from swe.pipeline.serialization import serialize_output

        ctx = RunContext.create(tmp_path)
        serialize_output(ctx.steps_dir, "step_a", {"key": "value"})
        result = ctx.get_step_output("step_a")
        assert result == {"key": "value"}

    def test_get_step_output_missing(self, tmp_path):
        """get_step_output should raise for non-existent step."""
        ctx = RunContext.create(tmp_path)
        with pytest.raises(FileNotFoundError):
            ctx.get_step_output("no_such_step")


class TestGenerateRunId:
    """Tests for _generate_run_id."""

    def test_format(self):
        """Run IDs should match YYYYMMDDTHHMMSS_<hex> format."""
        import re

        run_id = _generate_run_id()
        assert re.match(r"\d{8}T\d{6}_[0-9a-f]{8}", run_id)

    def test_uniqueness(self):
        """Two calls should produce different IDs."""
        assert _generate_run_id() != _generate_run_id()
