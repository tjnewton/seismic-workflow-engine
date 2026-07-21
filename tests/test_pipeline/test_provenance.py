"""Tests for swe.pipeline.provenance."""

from __future__ import annotations

import json
import time

import pytest

from swe.pipeline.provenance import (
    StepProvenance,
    format_provenance,
    load_provenance,
    save_provenance,
)


def _make_record(step_name: str = "s1", cache_hit: bool = False) -> StepProvenance:
    """Create a minimal provenance record for testing."""
    now = time.time()
    return StepProvenance(
        step_name=step_name,
        function_path="mod.fn",
        resolved_params={"x": 1},
        input_hash="abc123",
        started_at=now,
        completed_at=now + 1.5,
        duration_seconds=1.5,
        return_type="dict",
        cache_hit=cache_hit,
    )


class TestSaveLoadProvenance:
    """Tests for save/load provenance lifecycle."""

    def test_save_creates_file(self, tmp_path):
        """save_provenance should create provenance.json."""
        records = [_make_record()]
        path = save_provenance(tmp_path, records)
        assert path.exists()
        assert path.name == "provenance.json"

    def test_roundtrip(self, tmp_path):
        """Records should survive save/load cycle."""
        save_provenance(tmp_path, [_make_record("s1"), _make_record("s2")])
        loaded = load_provenance(tmp_path)
        assert len(loaded) == 2
        assert loaded[0]["step_name"] == "s1"
        assert loaded[1]["step_name"] == "s2"

    def test_append_to_existing(self, tmp_path):
        """Saving twice should append, not overwrite."""
        save_provenance(tmp_path, [_make_record("s1")])
        save_provenance(tmp_path, [_make_record("s2")])
        loaded = load_provenance(tmp_path)
        assert len(loaded) == 2

    def test_load_empty(self, tmp_path):
        """load_provenance with no file should return empty list."""
        assert load_provenance(tmp_path) == []

    def test_records_contain_versions(self, tmp_path):
        """Each record should include python_version and swe_version."""
        save_provenance(tmp_path, [_make_record()])
        loaded = load_provenance(tmp_path)
        assert "python_version" in loaded[0]
        assert "swe_version" in loaded[0]


class TestUncertaintySummary:
    """Tests for uncertainty_summary field on StepProvenance."""

    def test_uncertainty_summary_defaults_none(self):
        """Without explicit uncertainty_summary, field should be None."""
        record = _make_record()
        assert record.uncertainty_summary is None

    def test_uncertainty_summary_roundtrip(self, tmp_path):
        """uncertainty_summary should survive save/load."""
        record = _make_record()
        record = StepProvenance(
            step_name=record.step_name,
            function_path=record.function_path,
            resolved_params=record.resolved_params,
            input_hash=record.input_hash,
            started_at=record.started_at,
            completed_at=record.completed_at,
            duration_seconds=record.duration_seconds,
            return_type=record.return_type,
            cache_hit=record.cache_hit,
            uncertainty_summary={"ml_stderr": 0.05, "depth_uncertainty_m": 120.0},
        )
        save_provenance(tmp_path, [record])
        loaded = load_provenance(tmp_path)
        assert loaded[0]["uncertainty_summary"]["ml_stderr"] == 0.05
        assert loaded[0]["uncertainty_summary"]["depth_uncertainty_m"] == 120.0


class TestFormatProvenance:
    """Tests for format_provenance display."""

    def test_empty_message(self):
        """Empty records should produce a helpful message."""
        assert "No provenance" in format_provenance([])

    def test_includes_step_name(self):
        """Output should include the step name."""
        records = [
            {
                "step_name": "download",
                "function_path": "swe.io.fdsn.get_waveforms",
                "duration_seconds": 2.5,
                "cache_hit": False,
                "return_type": "dict",
            }
        ]
        output = format_provenance(records)
        assert "download" in output
        assert "swe.io.fdsn.get_waveforms" in output

    def test_cache_hit_label(self):
        """Cache hits should show 'HIT'."""
        records = [
            {
                "step_name": "s1",
                "function_path": "mod.fn",
                "duration_seconds": 0.001,
                "cache_hit": True,
                "return_type": "dict",
            }
        ]
        assert "HIT" in format_provenance(records)
