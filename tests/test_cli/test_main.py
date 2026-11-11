"""Tests for swe.cli.main."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from swe.cli.main import cli


class TestCLI:
    """Tests for the top-level CLI group."""

    def test_version(self):
        """--version should print the package version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "swe" in result.output
        assert "0.3.0" in result.output

    def test_help(self):
        """--help should list available subcommands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "download" in result.output
        assert "noise" in result.output
        assert "pick" in result.output
        assert "associate" in result.output
        assert "detect" in result.output
        assert "locate" in result.output
        assert "plot" in result.output
        assert "pipeline" in result.output

    def test_run_help(self):
        """swe run --help should show the run subcommand help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "CONFIG_FILE" in result.output
        assert "--dry-run" in result.output

    def test_download_help(self):
        """swe download --help should show download options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "--help"])
        assert result.exit_code == 0
        assert "--network" in result.output

    def test_verbose_flag(self):
        """The -v flag should be accepted."""
        runner = CliRunner()
        result = runner.invoke(cli, ["-v", "--help"])
        assert result.exit_code == 0

    def test_run_missing_config(self):
        """swe run with non-existent config should fail."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "nonexistent.yml"])
        assert result.exit_code != 0


class TestPipelineGroup:
    """Tests for the pipeline subcommand group."""

    def test_pipeline_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pipeline", "--help"])
        assert result.exit_code == 0
        assert "template" in result.output
        assert "list" in result.output

    def test_pipeline_list(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pipeline", "list"])
        assert result.exit_code == 0
        assert "full" in result.output
        assert "detection" in result.output
        assert "stress" in result.output
        assert "spectral" in result.output

    def test_pipeline_template_full(self, tmp_path):
        runner = CliRunner()
        output = str(tmp_path / "test_pipeline.yml")
        result = runner.invoke(cli, ["pipeline", "template", "full", "-o", output])
        assert result.exit_code == 0
        assert Path(output).exists()
        content = Path(output).read_text()
        assert "steps:" in content

    def test_pipeline_template_detection(self, tmp_path):
        runner = CliRunner()
        output = str(tmp_path / "detection.yml")
        result = runner.invoke(cli, ["pipeline", "template", "detection", "-o", output])
        assert result.exit_code == 0
        assert Path(output).exists()

    def test_pipeline_template_default_output(self, tmp_path):
        runner = CliRunner()
        # Change to tmp_path so the file is created there
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["pipeline", "template", "stress"])
            assert result.exit_code == 0
            assert Path("stress_pipeline.yml").exists()

    def test_pipeline_template_existing_file(self, tmp_path):
        runner = CliRunner()
        output = str(tmp_path / "existing.yml")
        Path(output).write_text("existing content")
        result = runner.invoke(cli, ["pipeline", "template", "full", "-o", output])
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_pipeline_template_invalid_workflow(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pipeline", "template", "nonexistent"])
        assert result.exit_code != 0


class TestDryRun:
    """Tests for the --dry-run flag."""

    def test_dry_run_valid_pipeline(self, tmp_path):
        config = tmp_path / "valid.yml"
        config.write_text(
            "steps:\n  - name: step1\n    function: swe.moment_tensor.tensor.scalar_moment\n"
            "    params:\n      m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n"
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--dry-run", str(config)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower() or "passed" in result.output.lower()

    def test_malformed_pipeline_is_an_error_line_not_a_traceback(self, tmp_path):
        config = tmp_path / "bad.yml"
        config.write_text("steps:\n  - name: s\n    function: os.getcwd\n    params: [1]\n")
        result = CliRunner().invoke(cli, ["run", str(config)])
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)  # click's error exit, nothing raised
        assert "Error: Step 's': params must be a mapping, got list" in result.output

    def test_dry_run_invalid_function(self, tmp_path):
        config = tmp_path / "invalid.yml"
        config.write_text("steps:\n  - name: step1\n    function: swe.nonexistent.module.func\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--dry-run", str(config)])
        assert result.exit_code != 0
        assert "error" in result.output.lower()
