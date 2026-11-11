"""Tests for swe.pipeline.validation - dry-run mode."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from swe.pipeline.core import Pipeline, PipelineStep
from swe.pipeline.templates import get_template, list_templates
from swe.pipeline.validation import validate_pipeline

EXAMPLES = sorted((Path(__file__).parents[2] / "examples").glob("*.yml"))


class TestValidatePipeline:
    def test_valid_simple_pipeline(self):
        """A valid pipeline should produce no errors."""
        pipeline = Pipeline(
            steps=[
                PipelineStep(
                    name="step1",
                    function="swe.moment_tensor.tensor.scalar_moment",
                    params={"m": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
                ),
            ]
        )
        errors, _warnings = validate_pipeline(pipeline)
        assert len(errors) == 0

    @pytest.mark.parametrize(
        ("params", "reason"),
        [({}, "missing a required argument: 'm'"), ({"m": 1, "mm": 2}, "unexpected keyword")],
    )
    def test_params_must_fit_the_signature(self, params, reason):
        """Params the ``function(**params)`` call would reject are errors before any step runs."""
        pipeline = Pipeline(
            steps=[
                PipelineStep(
                    name="s", function="swe.moment_tensor.tensor.scalar_moment", params=params
                )
            ]
        )
        errors, _warnings = validate_pipeline(pipeline)
        assert len(errors) == 1
        assert "params do not fit" in errors[0]
        assert reason in errors[0]

    def test_callable_without_signature_is_not_checked(self):
        """``inspect.signature`` fails for some builtins; such steps are not reported."""
        pipeline = Pipeline(steps=[PipelineStep(name="s", function="builtins.max", params={})])
        errors, _warnings = validate_pipeline(pipeline)
        assert errors == []

    def test_empty_pipeline(self):
        """An empty pipeline should be reported as an error."""
        pipeline = Pipeline(steps=[])
        errors, _warnings = validate_pipeline(pipeline)
        assert len(errors) == 1
        assert "no steps" in errors[0].lower()

    def test_duplicate_step_names(self):
        pipeline = Pipeline(
            steps=[
                PipelineStep(name="step1", function="swe.moment_tensor.tensor.scalar_moment"),
                PipelineStep(name="step1", function="swe.moment_tensor.tensor.moment_magnitude"),
            ]
        )
        errors, _ = validate_pipeline(pipeline)
        assert any("Duplicate" in e for e in errors)

    def test_empty_step_name(self):
        pipeline = Pipeline(
            steps=[
                PipelineStep(name="", function="swe.moment_tensor.tensor.scalar_moment"),
            ]
        )
        errors, _ = validate_pipeline(pipeline)
        assert any("empty name" in e.lower() for e in errors)

    def test_empty_function_path(self):
        pipeline = Pipeline(
            steps=[
                PipelineStep(name="step1", function=""),
            ]
        )
        errors, _ = validate_pipeline(pipeline)
        assert any("empty function" in e.lower() for e in errors)

    def test_invalid_module(self):
        pipeline = Pipeline(
            steps=[
                PipelineStep(
                    name="step1",
                    function="swe.nonexistent_module.func",
                ),
            ]
        )
        errors, _ = validate_pipeline(pipeline)
        assert any("cannot import" in e.lower() for e in errors)

    def test_invalid_function(self):
        pipeline = Pipeline(
            steps=[
                PipelineStep(
                    name="step1",
                    function="swe.moment_tensor.tensor.nonexistent_function",
                ),
            ]
        )
        errors, _ = validate_pipeline(pipeline)
        assert any("no attribute" in e.lower() for e in errors)

    def test_invalid_function_path_no_module(self):
        pipeline = Pipeline(
            steps=[
                PipelineStep(name="step1", function="no_dots_here"),
            ]
        )
        errors, _ = validate_pipeline(pipeline)
        assert any("invalid function path" in e.lower() for e in errors)

    def test_forward_reference(self):
        """References to steps that come later should be errors."""
        pipeline = Pipeline(
            steps=[
                PipelineStep(
                    name="step1",
                    function="swe.moment_tensor.tensor.scalar_moment",
                    params={"x": "${step2.output}"},
                ),
                PipelineStep(
                    name="step2",
                    function="swe.moment_tensor.tensor.scalar_moment",
                ),
            ]
        )
        errors, _ = validate_pipeline(pipeline)
        assert any("forward reference" in e.lower() for e in errors)

    def test_unknown_reference(self):
        pipeline = Pipeline(
            steps=[
                PipelineStep(
                    name="step1",
                    function="swe.moment_tensor.tensor.scalar_moment",
                    params={"x": "${nonexistent.value}"},
                ),
            ]
        )
        errors, _ = validate_pipeline(pipeline)
        assert any("unknown step" in e.lower() for e in errors)

    def test_valid_reference(self):
        """A valid forward reference to an earlier step should pass."""
        pipeline = Pipeline(
            steps=[
                PipelineStep(
                    name="step1",
                    function="swe.moment_tensor.tensor.scalar_moment",
                ),
                PipelineStep(
                    name="step2",
                    function="swe.moment_tensor.tensor.scalar_moment",
                    params={"m": "${step1.output}"},
                ),
            ]
        )
        errors, _ = validate_pipeline(pipeline)
        # Only potential errors would be from non-callable check
        ref_errors = [e for e in errors if "reference" in e.lower()]
        assert len(ref_errors) == 0

    def test_nested_reference_in_dict(self):
        """References nested in dicts should be detected."""
        pipeline = Pipeline(
            steps=[
                PipelineStep(
                    name="step1",
                    function="swe.moment_tensor.tensor.scalar_moment",
                    params={"nested": {"key": "${nonexistent.value}"}},
                ),
            ]
        )
        errors, _ = validate_pipeline(pipeline)
        assert any("unknown step" in e.lower() for e in errors)

    def test_reference_in_list(self):
        """References in lists should be detected."""
        pipeline = Pipeline(
            steps=[
                PipelineStep(
                    name="step1",
                    function="swe.moment_tensor.tensor.scalar_moment",
                    params={"items": ["${nonexistent.value}"]},
                ),
            ]
        )
        errors, _ = validate_pipeline(pipeline)
        assert any("unknown step" in e.lower() for e in errors)


class TestShippedPipelines:
    """Every built-in template and example passes a dry run, so ``swe run`` can call each step."""

    @pytest.mark.parametrize("name", list_templates())
    def test_template_validates(self, tmp_path, name):
        path = tmp_path / f"{name}.yml"
        path.write_text(get_template(name))
        errors, _warnings = validate_pipeline(Pipeline.from_yaml(path))
        assert errors == []

    def test_readme_pipelines_validate(self, tmp_path):
        """Every YAML pipeline shown in the README passes a dry run."""
        readme = Path(__file__).resolve().parents[2] / "README.md"
        blocks = [
            b for b in re.findall(r"```yaml\n(.*?)```", readme.read_text(), re.S) if "steps:" in b
        ]
        assert blocks
        for i, block in enumerate(blocks):
            path = tmp_path / f"readme_{i}.yml"
            path.write_text(block)
            errors, _warnings = validate_pipeline(Pipeline.from_yaml(path))
            assert errors == [], (i, errors)

    @pytest.mark.parametrize("params", [None, ["x"]], ids=["null", "list"])
    def test_params_that_are_not_a_mapping_are_an_error(self, params):
        """``params:`` left empty (null) or given as a list cannot be passed as keywords."""
        step = PipelineStep(
            name="s", function="swe.moment_tensor.tensor.scalar_moment", params=params
        )
        errors, _warnings = validate_pipeline(Pipeline(steps=[step]))
        assert [e for e in errors if "params" in e] == [
            f"Step 's': params must be a mapping, got {type(params).__name__}."
        ]

    def test_examples_are_found(self):
        """An empty glob would turn the example test into a silent skip."""
        assert EXAMPLES

    def test_yaml_params_null_is_empty_and_a_list_is_rejected(self, tmp_path):
        """``params:`` left empty loads as no params; a list fails at load, before any step runs."""
        ok = tmp_path / "ok.yml"
        ok.write_text("steps:\n  - name: s\n    function: os.getcwd\n    params:\n")
        assert Pipeline.from_yaml(ok).steps[0].params == {}
        bad = tmp_path / "bad.yml"
        bad.write_text("steps:\n  - name: s\n    function: os.getcwd\n    params: [1]\n")
        with pytest.raises(ValueError, match="params must be a mapping"):
            Pipeline.from_yaml(bad)

    @pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
    def test_example_validates(self, path):
        errors, _warnings = validate_pipeline(Pipeline.from_yaml(path))
        assert errors == []
