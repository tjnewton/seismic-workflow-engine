"""Tests for swe.pipeline.templates - built-in pipeline templates."""

from __future__ import annotations

import pytest
import yaml

from swe.pipeline.templates import TEMPLATES, get_template, list_templates


class TestListTemplates:
    def test_returns_sorted_list(self):
        result = list_templates()
        assert isinstance(result, list)
        assert result == sorted(result)

    def test_contains_known_templates(self):
        result = list_templates()
        assert "full" in result
        assert "detection" in result
        assert "stress" in result
        assert "spectral" in result
        assert "magnitude" in result
        assert "uncertainty" in result

    def test_matches_templates_dict(self):
        result = list_templates()
        assert set(result) == set(TEMPLATES)


class TestGetTemplate:
    @pytest.mark.parametrize("name", list_templates())
    def test_returns_string(self, name):
        content = get_template(name)
        assert isinstance(content, str)
        assert len(content) > 0

    @pytest.mark.parametrize("name", list_templates())
    def test_valid_yaml(self, name):
        """Every template should be parseable as valid YAML."""
        content = get_template(name)
        data = yaml.safe_load(content)
        assert isinstance(data, dict)
        assert "steps" in data
        assert isinstance(data["steps"], list)
        assert len(data["steps"]) > 0

    @pytest.mark.parametrize("name", list_templates())
    def test_steps_have_required_fields(self, name):
        """Each step should have 'name' and 'function' fields."""
        content = get_template(name)
        data = yaml.safe_load(content)
        for step in data["steps"]:
            assert "name" in step, f"Step missing 'name' in template {name}"
            assert "function" in step, f"Step missing 'function' in template {name}"

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError, match="Unknown template"):
            get_template("nonexistent_template")

    def test_full_template_has_all_stages(self):
        """Full template should cover the main pipeline stages."""
        content = get_template("full")
        data = yaml.safe_load(content)
        step_names = [s["name"] for s in data["steps"]]
        assert "download_waveforms" in step_names
        assert "pick_phases" in step_names
        assert "associate_events" in step_names

    def test_spectral_hvsr_step_fits_compute_hvsr(self):
        """The HVSR step passes a stream by reference, in params compute_hvsr accepts."""
        import inspect

        from swe.spectral import compute_hvsr

        steps = {s["name"]: s for s in yaml.safe_load(get_template("spectral"))["steps"]}
        params = steps["compute_hvsr"]["params"]
        assert params["stream"] == "${load_hvsr_station}"
        inspect.signature(compute_hvsr).bind(**params)

    def test_detection_template_has_detect_step(self):
        content = get_template("detection")
        data = yaml.safe_load(content)
        funcs = [s["function"] for s in data["steps"]]
        assert any("detect" in f for f in funcs)
