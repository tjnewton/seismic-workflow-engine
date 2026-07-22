"""Integration test: uncertainty propagation through a YAML pipeline."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from swe.pipeline.core import Pipeline, PipelineStep
from swe.uncertainty.report import summarize_pipeline_uncertainty
from swe.uncertainty.types import LocationUncertainty, MagnitudeUncertainty


def _make_synthetic_stream(n_stations: int = 4) -> Stream:
    """Create a synthetic Stream for testing."""
    rng = np.random.default_rng(42)
    traces = []
    for i in range(n_stations):
        data = rng.standard_normal(1000).astype(np.float64)
        header = {
            "network": "XX",
            "station": f"ST{i:02d}",
            "channel": "BHZ",
            "sampling_rate": 100.0,
            "npts": 1000,
            "starttime": UTCDateTime(2024, 1, 1),
        }
        traces.append(Trace(data=data, header=header))
    return Stream(traces)


class TestUncertaintyPipelineIntegration:
    """End-to-end test: picks -> location with uncertainty -> magnitude with uncertainty."""

    def test_full_propagation_pipeline(self, tmp_path):
        """Run a 3-step pipeline with uncertainty propagation via ${ref} syntax."""

        # Step 1: Mock pick function returning picks with time_uncertainty_s
        def mock_pick(**kwargs):
            from swe.picks import SWEPick

            return [
                SWEPick(
                    "ST00", "P", UTCDateTime(2024, 1, 1, 0, 0, 10), 0.9, time_uncertainty_s=0.05
                ),
                SWEPick(
                    "ST01", "P", UTCDateTime(2024, 1, 1, 0, 0, 10.5), 0.85, time_uncertainty_s=0.075
                ),
                SWEPick(
                    "ST02", "P", UTCDateTime(2024, 1, 1, 0, 0, 11), 0.8, time_uncertainty_s=0.1
                ),
                SWEPick(
                    "ST03", "P", UTCDateTime(2024, 1, 1, 0, 0, 12), 0.7, time_uncertainty_s=0.15
                ),
            ]

        # Step 2: Mock location function returning LocationResult with uncertainty
        def mock_locate(**kwargs):
            from swe.location.amplitude import LocationResult

            lu = LocationUncertainty(
                horiz_uncertainty_m=500.0,
                depth_uncertainty_m=200.0,
                rms=0.12,
                n_samples=100,
                method="monte_carlo",
            )
            return LocationResult(x=100.0, y=200.0, z=5000.0, misfit=0.05, uncertainty=lu)

        # Step 3: Mock magnitude function that uses location uncertainty
        def mock_magnitude(**kwargs):
            location_uncertainty = kwargs.get("location_uncertainty")
            mu = None
            if location_uncertainty is not None:
                from swe.uncertainty.magnitude import propagate_location_to_magnitude

                mu = propagate_location_to_magnitude(
                    ml_std=0.15,
                    ml_stderr=0.07,
                    n_valid_stations=4,
                    distances_km=np.array([10.0, 20.0, 30.0, 40.0]),
                    location_uncertainty=location_uncertainty,
                )
            return {
                "ml": 2.5,
                "ml_stderr": 0.07,
                "magnitude_uncertainty": mu,
            }

        pipeline = Pipeline(
            steps=[
                PipelineStep(
                    name="pick_phases",
                    function="mock.pick",
                    params={},
                    propagate_uncertainty=True,
                ),
                PipelineStep(
                    name="locate_events",
                    function="mock.locate",
                    params={},
                    propagate_uncertainty=True,
                ),
                PipelineStep(
                    name="estimate_magnitudes",
                    function="mock.magnitude",
                    params={
                        "location_uncertainty": "${locate_events.uncertainty}",
                    },
                    propagate_uncertainty=True,
                ),
            ]
        )

        call_count = [0]

        def _import_side_effect(path):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_pick
            elif call_count[0] == 2:
                return mock_locate
            else:
                return mock_magnitude

        with patch(
            "swe.pipeline.core._import_callable",
            side_effect=_import_side_effect,
        ):
            outputs = pipeline.execute(run_dir=tmp_path / "runs")

        # Verify outputs
        assert "pick_phases" in outputs
        assert "locate_events" in outputs
        assert "estimate_magnitudes" in outputs

        # Verify location uncertainty was propagated
        loc_result = outputs["locate_events"]
        assert loc_result.uncertainty is not None
        assert loc_result.uncertainty.horiz_uncertainty_m == 500.0

        # Verify magnitude uncertainty was computed
        mag_result = outputs["estimate_magnitudes"]
        assert mag_result["magnitude_uncertainty"] is not None
        assert isinstance(mag_result["magnitude_uncertainty"], MagnitudeUncertainty)
        assert mag_result["magnitude_uncertainty"].distance_contribution > 0

    def test_propagate_uncertainty_yaml_flag(self, tmp_path):
        """Test that propagate_uncertainty is parsed from YAML."""
        config = tmp_path / "pipeline.yml"
        config.write_text(
            "steps:\n"
            "  - name: step1\n"
            "    function: os.getcwd\n"
            "    propagate_uncertainty: true\n"
            "  - name: step2\n"
            "    function: os.getcwd\n"
        )
        pipeline = Pipeline.from_yaml(config)
        assert pipeline.steps[0].propagate_uncertainty is True
        assert pipeline.steps[1].propagate_uncertainty is False


class TestSummarizePipelineUncertainty:
    """Tests for the uncertainty report generator."""

    def test_with_uncertainty_data(self):
        records = [
            {
                "step_name": "locate_events",
                "uncertainty_summary": {
                    "horiz_uncertainty_m": 500.0,
                    "depth_uncertainty_m": 200.0,
                    "rms": 0.12,
                },
            },
            {
                "step_name": "estimate_magnitudes",
                "uncertainty_summary": {
                    "ml_stderr": 0.07,
                    "mag_uncertainty": 0.15,
                    "distance_contribution": 0.13,
                },
            },
        ]
        report = summarize_pipeline_uncertainty(records)
        assert "Uncertainty Propagation Summary" in report
        assert "locate_events" in report
        assert "500.0" in report
        assert "estimate_magnitudes" in report
        assert "0.1500" in report

    def test_empty_records(self):
        report = summarize_pipeline_uncertainty([])
        assert "No uncertainty data found" in report

    def test_no_uncertainty_in_any_step(self):
        records = [
            {"step_name": "step1", "uncertainty_summary": None},
            {"step_name": "step2"},
        ]
        report = summarize_pipeline_uncertainty(records)
        assert "No uncertainty data found" in report
