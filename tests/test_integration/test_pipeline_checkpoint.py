"""Integration tests: pipeline checkpointing, caching, and resume."""

from __future__ import annotations

import textwrap

import pytest

from swe.pipeline.core import Pipeline

pytestmark = pytest.mark.integration


def _step_add(*, a: int, b: int) -> dict:
    """Simple step: return sum."""
    return {"result": a + b}


def _step_multiply(*, value: int, factor: int) -> dict:
    """Simple step: return product."""
    return {"result": value * factor}


def _step_fail(**kwargs) -> dict:
    """Step that always fails."""
    raise RuntimeError("intentional failure")


class TestPipelineCheckpoint:
    """Pipeline checkpointing and resume integration tests."""

    def test_checkpoint_creates_run_dir(self, tmp_path):
        """Executing a pipeline should create a run directory."""
        yaml_content = textwrap.dedent("""\
            steps:
              - name: add
                function: tests.test_integration.test_pipeline_checkpoint._step_add
                params:
                  a: 3
                  b: 4
        """)
        config = tmp_path / "pipeline.yml"
        config.write_text(yaml_content)

        run_dir = tmp_path / "runs"
        pipeline = Pipeline.from_yaml(config)
        pipeline.execute(run_dir=run_dir)

        assert run_dir.exists()
        runs = list(run_dir.iterdir())
        assert len(runs) == 1

    def test_resume_after_failure(self, tmp_path):
        """After a failure, resume should skip completed steps."""
        yaml_content = textwrap.dedent("""\
            steps:
              - name: add
                function: tests.test_integration.test_pipeline_checkpoint._step_add
                params:
                  a: 3
                  b: 4
              - name: fail
                function: tests.test_integration.test_pipeline_checkpoint._step_fail
        """)
        config = tmp_path / "pipeline.yml"
        config.write_text(yaml_content)

        run_dir = tmp_path / "runs"
        pipeline = Pipeline.from_yaml(config)

        # First run: step 1 succeeds, step 2 fails
        with pytest.raises(RuntimeError, match="intentional failure"):
            pipeline.execute(run_dir=run_dir)

        # Now fix the pipeline (replace fail with multiply)
        yaml_fixed = textwrap.dedent("""\
            steps:
              - name: add
                function: tests.test_integration.test_pipeline_checkpoint._step_add
                params:
                  a: 3
                  b: 4
              - name: multiply
                function: tests.test_integration.test_pipeline_checkpoint._step_multiply
                params:
                  value: "${add.result}"
                  factor: 10
        """)
        config.write_text(yaml_fixed)
        pipeline2 = Pipeline.from_yaml(config)

        # Resume: add should be loaded from checkpoint
        outputs = pipeline2.execute(run_dir=run_dir, resume="latest")
        assert outputs["add"] == {"result": 7}
        assert outputs["multiply"] == {"result": 70}

    def test_cache_hit_on_rerun(self, tmp_path):
        """Re-running the same pipeline should produce cache hits."""
        yaml_content = textwrap.dedent("""\
            steps:
              - name: add
                function: tests.test_integration.test_pipeline_checkpoint._step_add
                params:
                  a: 10
                  b: 20
        """)
        config = tmp_path / "pipeline.yml"
        config.write_text(yaml_content)

        run_dir = tmp_path / "runs"
        pipeline = Pipeline.from_yaml(config)

        # First run
        out1 = pipeline.execute(run_dir=run_dir)
        assert out1["add"] == {"result": 30}

        # Second run: should hit cache
        out2 = pipeline.execute(run_dir=run_dir)
        assert out2["add"] == {"result": 30}

    def test_provenance_saved(self, tmp_path):
        """Provenance file should be created after execution."""
        yaml_content = textwrap.dedent("""\
            steps:
              - name: add
                function: tests.test_integration.test_pipeline_checkpoint._step_add
                params:
                  a: 1
                  b: 2
        """)
        config = tmp_path / "pipeline.yml"
        config.write_text(yaml_content)

        run_dir = tmp_path / "runs"
        pipeline = Pipeline.from_yaml(config)
        pipeline.execute(run_dir=run_dir)

        # Find the run directory
        runs = list(run_dir.iterdir())
        assert len(runs) == 1
        prov_path = runs[0] / "provenance.json"
        assert prov_path.exists()

        import json

        records = json.loads(prov_path.read_text())
        assert len(records) == 1
        assert records[0]["step_name"] == "add"
        assert records[0]["function_path"].endswith("_step_add")

    def test_three_step_pipeline_with_references(self, tmp_path):
        """Multi-step pipeline with ${ref} should work end-to-end."""
        yaml_content = textwrap.dedent("""\
            steps:
              - name: step1
                function: tests.test_integration.test_pipeline_checkpoint._step_add
                params:
                  a: 5
                  b: 3
              - name: step2
                function: tests.test_integration.test_pipeline_checkpoint._step_multiply
                params:
                  value: "${step1.result}"
                  factor: 2
              - name: step3
                function: tests.test_integration.test_pipeline_checkpoint._step_multiply
                params:
                  value: "${step2.result}"
                  factor: 10
        """)
        config = tmp_path / "pipeline.yml"
        config.write_text(yaml_content)

        run_dir = tmp_path / "runs"
        pipeline = Pipeline.from_yaml(config)
        outputs = pipeline.execute(run_dir=run_dir)

        assert outputs["step1"] == {"result": 8}
        assert outputs["step2"] == {"result": 16}
        assert outputs["step3"] == {"result": 160}

    def test_parallel_independent_steps(self, tmp_path):
        """Independent steps should complete with --parallel."""
        yaml_content = textwrap.dedent("""\
            steps:
              - name: add1
                function: tests.test_integration.test_pipeline_checkpoint._step_add
                params:
                  a: 1
                  b: 2
              - name: add2
                function: tests.test_integration.test_pipeline_checkpoint._step_add
                params:
                  a: 10
                  b: 20
        """)
        config = tmp_path / "pipeline.yml"
        config.write_text(yaml_content)

        run_dir = tmp_path / "runs"
        pipeline = Pipeline.from_yaml(config)
        outputs = pipeline.execute(run_dir=run_dir, parallel=True)

        assert outputs["add1"] == {"result": 3}
        assert outputs["add2"] == {"result": 30}

    def test_no_cache_forces_reexecution(self, tmp_path):
        """--no-cache should force re-execution of all steps."""
        call_count = {"n": 0}

        def _counting_add(*, a: int, b: int) -> dict:
            call_count["n"] += 1
            return {"result": a + b}

        import unittest.mock as mock

        yaml_content = textwrap.dedent("""\
            steps:
              - name: add
                function: tests.test_integration.test_pipeline_checkpoint._step_add
                params:
                  a: 1
                  b: 2
        """)
        config = tmp_path / "pipeline.yml"
        config.write_text(yaml_content)

        run_dir = tmp_path / "runs"
        pipeline = Pipeline.from_yaml(config)

        with mock.patch(
            "swe.pipeline.core._import_callable",
            return_value=_counting_add,
        ):
            pipeline.execute(run_dir=run_dir)
            pipeline.execute(run_dir=run_dir, no_cache=True)

        assert call_count["n"] == 2
