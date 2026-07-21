"""Tests for swe.cli.pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from swe.cli.pipeline import Pipeline, PipelineStep, _resolve_params


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
        config.write_text(
            "steps:\n"
            "  - name: step1\n"
            "    function: os.getcwd\n"
        )
        pipeline = Pipeline.from_yaml(config)
        assert pipeline.steps[0].params == {}


class TestPipelineExecute:
    """Tests for Pipeline.execute()."""

    def test_executes_steps(self, tmp_path):
        """Should call each step function with resolved params."""
        mock_fn = MagicMock(return_value={"result": 42})
        pipeline = Pipeline(steps=[
            PipelineStep(name="s1", function="os.getcwd", params={}),
        ])

        with patch("swe.pipeline.core._import_callable", return_value=mock_fn):
            outputs = pipeline.execute(run_dir=tmp_path / "runs")

        mock_fn.assert_called_once_with()
        assert "s1" in outputs

    def test_skip_steps(self, tmp_path):
        """Skipped steps should not be executed."""
        mock_fn = MagicMock()
        pipeline = Pipeline(steps=[
            PipelineStep(name="s1", function="os.getcwd", params={}),
            PipelineStep(name="s2", function="os.getcwd", params={}),
        ])

        with patch("swe.pipeline.core._import_callable", return_value=mock_fn):
            outputs = pipeline.execute(skip={"s1"}, run_dir=tmp_path / "runs")

        assert mock_fn.call_count == 1
        assert "s1" not in outputs
        assert "s2" in outputs

    def test_param_resolution(self, tmp_path):
        """${ref} params should resolve from prior step outputs."""
        fn1 = MagicMock(return_value={"outpath": "/data/file.ms"})
        fn2 = MagicMock(return_value="done")

        pipeline = Pipeline(steps=[
            PipelineStep(name="s1", function="mod.fn1", params={}),
            PipelineStep(
                name="s2",
                function="mod.fn2",
                params={"input_file": "${s1.outpath}"},
            ),
        ])

        call_count = [0]

        def _import_side_effect(path):
            call_count[0] += 1
            return fn1 if call_count[0] == 1 else fn2

        with patch("swe.pipeline.core._import_callable", side_effect=_import_side_effect):
            pipeline.execute(run_dir=tmp_path / "runs")

        fn2.assert_called_once_with(input_file="/data/file.ms")


class TestResolveParams:
    """Tests for _resolve_params."""

    def test_simple_reference(self):
        """Should replace ${step.key} with the value."""
        outputs = {"step1": {"path": "/tmp/data"}}
        result = _resolve_params({"f": "${step1.path}"}, outputs)
        assert result["f"] == "/tmp/data"

    def test_no_reference(self):
        """Strings without ${} should pass through."""
        result = _resolve_params({"f": "hello"}, {})
        assert result["f"] == "hello"

    def test_numeric_passthrough(self):
        """Non-string values should pass through."""
        result = _resolve_params({"n": 42, "x": 3.14}, {})
        assert result["n"] == 42
        assert result["x"] == 3.14

    def test_nested_dict(self):
        """Should resolve references in nested dicts."""
        outputs = {"s1": {"val": 100}}
        result = _resolve_params({"nested": {"x": "${s1.val}"}}, outputs)
        assert result["nested"]["x"] == 100

    def test_list_resolution(self):
        """Should resolve references inside lists."""
        outputs = {"s1": {"a": "foo", "b": "bar"}}
        result = _resolve_params({"items": ["${s1.a}", "${s1.b}"]}, outputs)
        assert result["items"] == ["foo", "bar"]
