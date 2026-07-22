"""Tests for swe.pipeline.validation - dry-run mode."""

from __future__ import annotations

from swe.pipeline.core import Pipeline, PipelineStep
from swe.pipeline.validation import validate_pipeline


class TestValidatePipeline:
    def test_valid_simple_pipeline(self):
        """A valid pipeline should produce no errors."""
        pipeline = Pipeline(
            steps=[
                PipelineStep(
                    name="step1",
                    function="swe.moment_tensor.tensor.scalar_moment",
                    params={},
                ),
            ]
        )
        errors, _warnings = validate_pipeline(pipeline)
        assert len(errors) == 0

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
