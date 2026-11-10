"""Tests for swe.pipeline.core."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from swe.pipeline.core import Pipeline, PipelineStep, _extract_uncertainty, _resolve_params


class TestPipelineFromYaml:
    """Tests for Pipeline.from_yaml()."""

    def test_loads_steps(self, tmp_path):
        """Should load steps from a valid YAML file."""
        config = tmp_path / "pipeline.yml"
        config.write_text(
            "steps:\n"
            "  - name: step1\n"
            "    function: os.path.exists\n"
            "    params:\n"
            "      path: /tmp\n"
            "  - name: step2\n"
            "    function: os.path.isdir\n"
            "    params:\n"
            "      s: /tmp\n"
        )
        pipeline = Pipeline.from_yaml(config)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].name == "step1"
        assert pipeline.steps[1].function == "os.path.isdir"

    def test_missing_steps_key(self, tmp_path):
        """Should raise ValueError if 'steps' key is missing."""
        config = tmp_path / "bad.yml"
        config.write_text("foo: bar\n")
        with pytest.raises(ValueError, match="steps"):
            Pipeline.from_yaml(config)

    def test_empty_params(self, tmp_path):
        """Steps without params should default to empty dict."""
        config = tmp_path / "noparam.yml"
        config.write_text("steps:\n  - name: step1\n    function: os.getcwd\n")
        pipeline = Pipeline.from_yaml(config)
        assert pipeline.steps[0].params == {}

    def test_cache_flag_from_yaml(self, tmp_path):
        """Should parse ``cache: false`` from YAML."""
        config = tmp_path / "nocache.yml"
        config.write_text("steps:\n  - name: step1\n    function: os.getcwd\n    cache: false\n")
        pipeline = Pipeline.from_yaml(config)
        assert pipeline.steps[0].cache is False

    def test_cache_defaults_true(self, tmp_path):
        """Steps without explicit cache key should default to True."""
        config = tmp_path / "default.yml"
        config.write_text("steps:\n  - name: step1\n    function: os.getcwd\n")
        pipeline = Pipeline.from_yaml(config)
        assert pipeline.steps[0].cache is True


class TestPipelineExecute:
    """Tests for Pipeline.execute()."""

    def test_executes_steps(self, tmp_path):
        """Should call each step function with resolved params."""
        mock_fn = MagicMock(return_value={"result": 42})
        pipeline = Pipeline(
            steps=[
                PipelineStep(name="s1", function="os.getcwd", params={}),
            ]
        )

        with patch("swe.pipeline.core._import_callable", return_value=mock_fn):
            outputs = pipeline.execute(run_dir=tmp_path / "runs")

        mock_fn.assert_called_once_with()
        assert "s1" in outputs

    def test_skip_steps(self, tmp_path):
        """Skipped steps should not be executed."""
        mock_fn = MagicMock()
        pipeline = Pipeline(
            steps=[
                PipelineStep(name="s1", function="os.getcwd", params={}),
                PipelineStep(name="s2", function="os.getcwd", params={}),
            ]
        )

        with patch("swe.pipeline.core._import_callable", return_value=mock_fn):
            outputs = pipeline.execute(skip={"s1"}, run_dir=tmp_path / "runs")

        assert mock_fn.call_count == 1
        assert "s1" not in outputs
        assert "s2" in outputs

    def test_param_resolution(self, tmp_path):
        """${ref} params should resolve from prior step outputs."""
        fn1 = MagicMock(return_value={"outpath": "/data/file.ms"})
        fn2 = MagicMock(return_value="done")

        pipeline = Pipeline(
            steps=[
                PipelineStep(name="s1", function="mod.fn1", params={}),
                PipelineStep(
                    name="s2",
                    function="mod.fn2",
                    params={"input_file": "${s1.outpath}"},
                ),
            ]
        )

        call_count = [0]

        def _import_side_effect(path):
            call_count[0] += 1
            return fn1 if call_count[0] == 1 else fn2

        with patch(
            "swe.pipeline.core._import_callable",
            side_effect=_import_side_effect,
        ):
            pipeline.execute(run_dir=tmp_path / "runs")

        fn2.assert_called_once_with(input_file="/data/file.ms")

    def test_resume_skips_completed(self, tmp_path):
        """Resumed run should skip already-completed steps."""
        run_dir = tmp_path / "runs"
        mock_fn = MagicMock(return_value={"val": 1})

        pipeline = Pipeline(
            steps=[
                PipelineStep(name="s1", function="os.getcwd", params={}),
                PipelineStep(name="s2", function="os.getcwd", params={}),
            ]
        )

        # First run: execute both steps
        with patch("swe.pipeline.core._import_callable", return_value=mock_fn):
            pipeline.execute(run_dir=run_dir)

        assert mock_fn.call_count == 2

        # Reset and resume: both should be skipped (cached checkpoint)
        mock_fn.reset_mock()
        with patch("swe.pipeline.core._import_callable", return_value=mock_fn):
            outputs = pipeline.execute(run_dir=run_dir, resume="latest")

        assert mock_fn.call_count == 0
        assert "s1" in outputs
        assert "s2" in outputs

    def test_resume_refuses_a_run_written_before_typed_checkpoints(self, tmp_path):
        """A run without the output-format field holds lossy JSON checkpoints: no resume."""
        import json

        run_dir = tmp_path / "runs"
        pipeline = Pipeline(steps=[PipelineStep(name="s1", function="os.getcwd", params={})])
        with patch("swe.pipeline.core._import_callable", return_value=MagicMock(return_value=1)):
            pipeline.execute(run_dir=run_dir)
        (manifest,) = run_dir.glob("*/manifest.json")
        m = json.loads(manifest.read_text())
        del m["output_format"]
        manifest.write_text(json.dumps(m))
        with pytest.raises(RuntimeError, match="before checkpoints kept their types"):
            pipeline.execute(run_dir=run_dir, resume="latest")

    def test_resume_stops_clearly_on_a_checkpoint_from_another_version(self, tmp_path):
        """A pickle written by another SWE version stops the resume with the step named."""
        import json

        run_dir = tmp_path / "runs"
        mock_fn = MagicMock(return_value=(1, 2))
        pipeline = Pipeline(steps=[PipelineStep(name="s1", function="os.getcwd", params={})])
        with patch("swe.pipeline.core._import_callable", return_value=mock_fn):
            pipeline.execute(run_dir=run_dir)
        (meta,) = run_dir.glob("*/steps/s1.output.pkl.meta")
        meta.write_text(json.dumps({"swe_version": "0.0.0-other"}))
        with pytest.raises(RuntimeError, match="checkpoint of step 's1' cannot be loaded"):
            pipeline.execute(run_dir=run_dir, resume="latest")

    def test_resume_restores_a_typed_output(self, tmp_path):
        """A tuple output is pickled, so a resume gets the tuple back, not a list."""
        run_dir = tmp_path / "runs"
        mock_fn = MagicMock(return_value=(1, 2))
        pipeline = Pipeline(steps=[PipelineStep(name="s1", function="os.getcwd", params={})])
        with patch("swe.pipeline.core._import_callable", return_value=mock_fn):
            pipeline.execute(run_dir=run_dir)
            outputs = pipeline.execute(run_dir=run_dir, resume="latest")
        assert outputs["s1"] == (1, 2)

    def test_failed_step_raises(self, tmp_path):
        """A step that raises should be recorded as failed and re-raised."""
        run_dir = tmp_path / "runs"

        def _boom(**kwargs):
            raise RuntimeError("step failed")

        pipeline = Pipeline(
            steps=[
                PipelineStep(name="explode", function="os.getcwd", params={}),
            ]
        )

        with (
            patch("swe.pipeline.core._import_callable", return_value=_boom),
            pytest.raises(RuntimeError, match="step failed"),
        ):
            pipeline.execute(run_dir=run_dir)

    def test_no_cache_flag(self, tmp_path):
        """With no_cache=True, steps should always execute."""
        run_dir = tmp_path / "runs"
        call_count = [0]

        def _counter(**kwargs):
            call_count[0] += 1
            return {"n": call_count[0]}

        pipeline = Pipeline(
            steps=[
                PipelineStep(name="s1", function="os.getcwd", params={}),
            ]
        )

        with patch("swe.pipeline.core._import_callable", return_value=_counter):
            pipeline.execute(run_dir=run_dir)
            pipeline.execute(run_dir=run_dir, no_cache=True)

        assert call_count[0] == 2


class TestResolveParams:
    """Tests for _resolve_params."""

    def test_simple_reference(self):
        """Should replace ${step.key} with the value."""
        outputs = {"step1": {"path": "/tmp/data"}}
        result = _resolve_params({"f": "${step1.path}"}, outputs, set(outputs))
        assert result["f"] == "/tmp/data"

    def test_whole_output_reference(self):
        """``${step}`` passes the earlier step's whole output, of any type."""
        out = object()
        assert _resolve_params({"f": "${step1}", "g": ["${step1}"]}, {"step1": out}, {"step1"}) == {
            "f": out,
            "g": [out],
        }

    def test_bare_name_that_is_not_a_step_stays_literal(self):
        """``${HOME}``-style text passed through before ``${step}`` existed, and still does."""
        assert _resolve_params({"p": "${HOME}/x", "q": "${HOME}"}, {"s": 1}, {"s"}) == {
            "p": "${HOME}/x",
            "q": "${HOME}",
        }

    def test_bare_ref_to_a_pipeline_step_without_output_raises(self):
        """A known step must have output, as for ``${step.key}``."""
        with pytest.raises(KeyError, match="not found in outputs"):
            _resolve_params({"p": "${load}"}, {}, {"load"})

    def test_no_reference(self):
        """Strings without ${} should pass through."""
        result = _resolve_params({"f": "hello"}, {}, set())
        assert result["f"] == "hello"

    def test_numeric_passthrough(self):
        """Non-string values should pass through."""
        result = _resolve_params({"n": 42, "x": 3.14}, {}, set())
        assert result["n"] == 42
        assert result["x"] == 3.14

    def test_nested_dict(self):
        """Should resolve references in nested dicts."""
        outputs = {"s1": {"val": 100}}
        result = _resolve_params({"nested": {"x": "${s1.val}"}}, outputs, set(outputs))
        assert result["nested"]["x"] == 100

    def test_list_resolution(self):
        """Should resolve references inside lists."""
        outputs = {"s1": {"a": "foo", "b": "bar"}}
        result = _resolve_params({"items": ["${s1.a}", "${s1.b}"]}, outputs, set(outputs))
        assert result["items"] == ["foo", "bar"]


class TestExtractUncertainty:
    """Tests for _extract_uncertainty helper."""

    def test_none_input(self):
        """None input should return None."""
        assert _extract_uncertainty(None) is None

    def test_no_uncertainty_fields(self):
        """Dict without uncertainty fields returns None."""
        result = _extract_uncertainty({"key": "value"})
        assert result is None

    def test_extracts_from_dict(self):
        """Should extract recognised uncertainty fields from a dict."""
        obj = {"ml_stderr": 0.05, "ml_std": 0.12, "rms": 0.8}
        summary = _extract_uncertainty(obj)
        assert summary is not None
        assert summary["ml_stderr"] == 0.05
        assert summary["ml_std"] == 0.12
        assert summary["rms"] == 0.8

    def test_extracts_from_namedtuple(self):
        """Should extract recognised uncertainty fields from a NamedTuple."""
        from swe.stress.inversion import StressInversionResult

        result = StressInversionResult(
            results=[{"sigma_1": {}}],
            axes=[{"stress_ratio": 0.5}],
            bootstrap_axes=[{"sigma_1": {"azimuth_std": 5.0}}],
        )
        summary = _extract_uncertainty(result)
        assert summary is not None
        assert "bootstrap_axes" in summary

    def test_skips_zero_scalars(self):
        """Zero-valued scalars (defaults) should be skipped."""
        result = _extract_uncertainty({"ml_stderr": 0, "depth_uncertainty_m": 0.0})
        assert result is None

    def test_skips_empty_containers(self):
        """Empty dicts/lists should be skipped."""
        result = _extract_uncertainty({"cluster_quality": {}, "bootstrap_axes": []})
        assert result is None

    def test_tuple_converted_to_list(self):
        """Tuple values should be converted to lists for JSON safety."""
        result = _extract_uncertainty({"iso_pct_range": (10.0, 25.0)})
        assert result is not None
        assert result["iso_pct_range"] == [10.0, 25.0]

    def test_ignores_arbitrary_objects(self):
        """Arbitrary objects (e.g. MagicMock) should return None."""
        from unittest.mock import MagicMock

        assert _extract_uncertainty(MagicMock()) is None
