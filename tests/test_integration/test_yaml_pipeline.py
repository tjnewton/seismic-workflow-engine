"""Integration tests: YAML pipeline orchestration with real SWE functions."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from swe.cli.pipeline import Pipeline

pytestmark = pytest.mark.integration


def _synthetic_event_from_lists(
    true_location, station_coords, velocity=5000.0, source_amplitude=1.0
):
    """Thin wrapper that converts YAML lists to numpy arrays."""
    from swe.location.amplitude import synthetic_event

    return synthetic_event(
        tuple(true_location),
        np.array(station_coords, dtype=float),
        velocity=velocity,
        source_amplitude=source_amplitude,
    )


def _locate_from_lists(observed_times, station_coords, velocity_model, grid_bounds, grid_spacing):
    """Thin wrapper that converts YAML lists to numpy/tuple types."""
    from swe.location.amplitude import locate_by_travel_time

    return locate_by_travel_time(
        np.asarray(observed_times, dtype=float),
        np.array(station_coords, dtype=float),
        velocity_model,
        grid_bounds=tuple(tuple(b) for b in grid_bounds),
        grid_spacing=tuple(grid_spacing),
    )


class TestYAMLPipeline:
    """Pipeline execution with parameter references and real SWE functions."""

    def test_two_step_pipeline_with_param_reference(self, tmp_path):
        """${step1.key} references resolve correctly between real functions."""
        yaml_content = textwrap.dedent("""\
            steps:
              - name: make_event
                function: tests.test_integration.test_yaml_pipeline._synthetic_event_from_lists
                params:
                  true_location: [500.0, 500.0, 200.0]
                  station_coords: [[0, 0, 0], [1000, 0, 0], [0, 1000, 0], [1000, 1000, 0]]
                  velocity: 5000.0
                  source_amplitude: 1.0
              - name: locate
                function: tests.test_integration.test_yaml_pipeline._locate_from_lists
                params:
                  observed_times: "${make_event.travel_times}"
                  station_coords: [[0, 0, 0], [1000, 0, 0], [0, 1000, 0], [1000, 1000, 0]]
                  velocity_model: 5000.0
                  grid_bounds: [[0, 1000], [0, 1000], [0, 400]]
                  grid_spacing: [100.0, 100.0, 100.0]
        """)
        config = tmp_path / "pipeline.yml"
        config.write_text(yaml_content)

        pipeline = Pipeline.from_yaml(config)
        outputs = pipeline.execute()

        assert "make_event" in outputs
        assert "locate" in outputs
        loc = outputs["locate"]
        assert hasattr(loc, "x")
        assert hasattr(loc, "misfit")
        assert np.isfinite(loc.x)
        assert np.isfinite(loc.misfit)

    def test_skip_step_in_pipeline(self, tmp_path):
        """Skipped steps are excluded from outputs dict."""
        yaml_content = textwrap.dedent("""\
            steps:
              - name: step_a
                function: tests.test_integration.test_yaml_pipeline._synthetic_event_from_lists
                params:
                  true_location: [100.0, 100.0, 50.0]
                  station_coords: [[0, 0, 0], [200, 0, 0]]
                  velocity: 5000.0
              - name: step_b
                function: tests.test_integration.test_yaml_pipeline._synthetic_event_from_lists
                params:
                  true_location: [100.0, 100.0, 50.0]
                  station_coords: [[0, 0, 0], [200, 0, 0]]
                  velocity: 5000.0
        """)
        config = tmp_path / "pipeline.yml"
        config.write_text(yaml_content)

        pipeline = Pipeline.from_yaml(config)
        outputs = pipeline.execute(skip={"step_b"})

        assert "step_a" in outputs
        assert "step_b" not in outputs

    def test_association_pipeline_from_yaml(self, tmp_path):
        """Multi-step pipeline with parameter reference chaining."""
        yaml_content = textwrap.dedent("""\
            steps:
              - name: event1
                function: tests.test_integration.test_yaml_pipeline._synthetic_event_from_lists
                params:
                  true_location: [500.0, 500.0, 200.0]
                  station_coords: [[0, 0, 0], [1000, 0, 0], [0, 1000, 0]]
                  velocity: 5000.0
              - name: event2
                function: tests.test_integration.test_yaml_pipeline._synthetic_event_from_lists
                params:
                  true_location: [800.0, 200.0, 100.0]
                  station_coords: [[0, 0, 0], [1000, 0, 0], [0, 1000, 0]]
                  velocity: 5000.0
        """)
        config = tmp_path / "pipeline.yml"
        config.write_text(yaml_content)

        pipeline = Pipeline.from_yaml(config)
        outputs = pipeline.execute()

        assert len(outputs) == 2
        assert outputs["event1"].travel_times is not None
        assert outputs["event2"].travel_times is not None
