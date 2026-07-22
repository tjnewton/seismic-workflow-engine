"""Tests for swe.config - user configuration management."""

from __future__ import annotations

from pathlib import Path

import yaml

from swe.config import SWEConfig, get_config_path, load_config, save_config


class TestSWEConfig:
    def test_defaults(self):
        cfg = SWEConfig()
        assert cfg.fdsn_client == "IRIS"
        assert cfg.output_dir == "."
        assert cfg.picker_backend == "seisbench"
        assert cfg.picker_model == "PhaseNet"
        assert cfg.associator_backend == "dbscan"
        assert cfg.run_dir == ".swe_runs"
        assert cfg.parallel is False
        assert cfg.log_level == "WARNING"

    def test_from_dict(self):
        cfg = SWEConfig.from_dict({"fdsn_client": "SCEDC", "parallel": True})
        assert cfg.fdsn_client == "SCEDC"
        assert cfg.parallel is True
        # Other fields should be defaults
        assert cfg.output_dir == "."

    def test_from_dict_ignores_unknown(self):
        cfg = SWEConfig.from_dict({"unknown_key": "value", "fdsn_client": "GFZ"})
        assert cfg.fdsn_client == "GFZ"
        assert not hasattr(cfg, "unknown_key")


class TestGetConfigPath:
    def test_returns_path(self):
        path = get_config_path()
        assert isinstance(path, Path)
        assert path.name == "config.yml"
        assert ".swe" in str(path)


class TestLoadConfig:
    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = load_config(config_path=tmp_path / "nonexistent.yml")
        assert cfg.fdsn_client == "IRIS"

    def test_loads_from_file(self, tmp_path):
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump({"fdsn_client": "SCEDC", "parallel": True}))

        cfg = load_config(config_path=config_path)
        assert cfg.fdsn_client == "SCEDC"
        assert cfg.parallel is True

    def test_invalid_yaml_returns_defaults(self, tmp_path):
        config_path = tmp_path / "config.yml"
        config_path.write_text("{{invalid yaml content")

        cfg = load_config(config_path=config_path)
        assert cfg.fdsn_client == "IRIS"  # defaults

    def test_non_dict_yaml_returns_defaults(self, tmp_path):
        config_path = tmp_path / "config.yml"
        config_path.write_text("- just\n- a\n- list\n")

        cfg = load_config(config_path=config_path)
        assert cfg.fdsn_client == "IRIS"


class TestSaveConfig:
    def test_save_and_reload(self, tmp_path):
        config_path = tmp_path / "subdir" / "config.yml"
        cfg = SWEConfig(fdsn_client="GFZ", parallel=True)

        saved_path = save_config(cfg, config_path=config_path)
        assert saved_path.exists()
        assert saved_path.parent.exists()

        loaded = load_config(config_path=saved_path)
        assert loaded.fdsn_client == "GFZ"
        assert loaded.parallel is True

    def test_creates_directory(self, tmp_path):
        config_path = tmp_path / "deep" / "nested" / "config.yml"
        save_config(SWEConfig(), config_path=config_path)
        assert config_path.exists()
