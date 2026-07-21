"""Tests for swe.cli.commands - individual CLI subcommands."""

from __future__ import annotations

import csv
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner
from obspy import Stream, Trace, UTCDateTime

from swe.cli.main import cli


@pytest.fixture()
def mseed_dir(tmp_path):
    """Create a temporary directory with a synthetic MiniSEED file."""
    st = Stream()
    tr = Trace(
        data=np.random.default_rng(0).normal(0, 1, 6000).astype(np.float64),
        header={
            "network": "XX",
            "station": "STA1",
            "channel": "BHZ",
            "sampling_rate": 100.0,
            "starttime": UTCDateTime("2024-01-01T00:00:00"),
            "npts": 6000,
        },
    )
    st.append(tr)
    outfile = tmp_path / "XX.STA1.BHZ.2024-01-01.mseed"
    st.write(str(outfile), format="MSEED")
    return tmp_path


@pytest.fixture()
def picks_csv(tmp_path):
    """Create a picks CSV file."""
    path = tmp_path / "picks.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["station", "phase", "time", "confidence"])
        for sta in ["STA1", "STA2", "STA3", "STA4", "STA5"]:
            writer.writerow([sta, "P", "2024-01-01T00:00:10.000000Z", "0.9"])
            writer.writerow([sta, "P", "2024-01-01T00:00:30.000000Z", "0.85"])
    return path


@pytest.fixture()
def stations_csv(tmp_path):
    """Create a stations CSV file."""
    path = tmp_path / "stations.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["station", "lon", "lat", "elev"])
        for i, sta in enumerate(["STA1", "STA2", "STA3", "STA4", "STA5"]):
            writer.writerow([sta, -118.5 + i * 0.02, 34.0 + i * 0.01, 0.0])
    return path


class TestNoiseCommand:
    """Tests for the noise CLI command."""

    def test_noise_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["noise", "--help"])
        assert result.exit_code == 0
        assert "--input-dir" in result.output
        assert "--sta" in result.output
        assert "--lta" in result.output
        assert "--threshold" in result.output

    def test_noise_runs_on_mseed(self, mseed_dir, tmp_path):
        runner = CliRunner()
        output = tmp_path / "out.npz"
        result = runner.invoke(
            cli,
            [
                "noise",
                "--input-dir",
                str(mseed_dir),
                "--start",
                "2024-01-01T00:00:00",
                "--end",
                "2024-01-01T00:01:00",
                "--sta",
                "15",
                "--lta",
                "250",
                "--threshold",
                "3.0",
                "--window-length",
                "240",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "noise window" in result.output.lower()

    def test_noise_no_files_error(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "noise",
                "--input-dir",
                str(empty),
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-02",
            ],
        )
        assert result.exit_code != 0
        assert "No MiniSEED" in result.output


class TestPickCommand:
    """Tests for the pick CLI command."""

    def test_pick_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pick", "--help"])
        assert result.exit_code == 0
        assert "--method" in result.output
        assert "--model" in result.output
        assert "--p-threshold" in result.output

    @patch("swe.picking.predict_phase_arrivals")
    def test_pick_seisbench_writes_csv(self, mock_predict, mseed_dir, tmp_path):
        from swe.picking import PhaseArrival

        mock_predict.return_value = [
            PhaseArrival("STA1", "P", UTCDateTime("2024-01-01T00:00:10"), 0.95),
            PhaseArrival("STA1", "S", UTCDateTime("2024-01-01T00:00:12"), 0.80),
        ]
        output = tmp_path / "picks.csv"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "pick",
                "--input-dir",
                str(mseed_dir),
                "--method",
                "seisbench",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "2 pick(s)" in result.output
        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert lines[0] == "station,phase,time,confidence"
        assert len(lines) == 3

    @patch("swe.picking.find_picking_windows")
    def test_pick_stalta_method(self, mock_fpw, mseed_dir):
        from swe.picking import PickingWindowResult

        mock_fpw.return_value = PickingWindowResult(
            windows=np.array([]),
            windows_untransformed=[],
            times=[],
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "pick",
                "--input-dir",
                str(mseed_dir),
                "--method",
                "stalta",
                "--start",
                "2024-01-01T00:00:00",
                "--end",
                "2024-01-01T00:01:00",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "STA/LTA" in result.output

    def test_pick_no_files_error(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "pick",
                "--input-dir",
                str(empty),
            ],
        )
        assert result.exit_code != 0
        assert "No MiniSEED" in result.output


class TestAssociateCommand:
    """Tests for the associate CLI command."""

    def test_associate_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["associate", "--help"])
        assert result.exit_code == 0
        assert "--picks" in result.output
        assert "--stations" in result.output
        assert "--eps" in result.output
        assert "--format" in result.output

    def test_associate_quakeml(self, picks_csv, stations_csv, tmp_path):
        output = tmp_path / "catalog.xml"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "associate",
                "--picks",
                str(picks_csv),
                "--stations",
                str(stations_csv),
                "--eps",
                "2.0",
                "--min-samples",
                "3",
                "--min-picks",
                "3",
                "--temporal-threshold",
                "15.0",
                "--spatial-threshold",
                "10.0",
                "--output",
                str(output),
                "--format",
                "quakeml",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Associated" in result.output
        assert output.exists()

    def test_associate_markers_format(self, picks_csv, stations_csv, tmp_path):
        output = tmp_path / "events.mrkr"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "associate",
                "--picks",
                str(picks_csv),
                "--stations",
                str(stations_csv),
                "--eps",
                "2.0",
                "--min-samples",
                "3",
                "--min-picks",
                "3",
                "--temporal-threshold",
                "15.0",
                "--spatial-threshold",
                "10.0",
                "--output",
                str(output),
                "--format",
                "markers",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Associated" in result.output


class TestDetectCommand:
    """Tests for the detect CLI command."""

    def test_detect_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["detect", "--help"])
        assert result.exit_code == 0
        assert "--marker-file" in result.output
        assert "--template-dir" in result.output
        assert "--data-dir" in result.output
        assert "--thresh-type" in result.output

    def test_detect_missing_marker_file(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "detect",
                "--marker-file",
                str(tmp_path / "nonexistent.mrkr"),
                "--template-dir",
                str(tmp_path),
                "--data-dir",
                str(tmp_path),
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-02",
            ],
        )
        # Should fail due to nonexistent marker file
        assert result.exit_code != 0


class TestPlotCommand:
    """Tests for the plot CLI command."""

    def test_plot_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["plot", "--help"])
        assert result.exit_code == 0
        assert "--plot-type" in result.output
        assert "--output" in result.output
        assert "--figsize" in result.output
        assert "--dpi" in result.output
        assert "--freq-max" in result.output

    def test_plot_waveform_to_file(self, mseed_dir, tmp_path):
        output = tmp_path / "waveform.png"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "plot",
                "--input",
                str(mseed_dir),
                "--plot-type",
                "waveform",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Saved" in result.output
        assert output.exists()

    def test_plot_spectrogram_to_file(self, mseed_dir, tmp_path):
        output = tmp_path / "spectrogram.png"
        runner = CliRunner()
        mseed_files = list(mseed_dir.glob("*.mseed"))
        result = runner.invoke(
            cli,
            [
                "plot",
                "--input",
                str(mseed_files[0]),
                "--plot-type",
                "spectrogram",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_plot_spectrum_to_file(self, mseed_dir, tmp_path):
        output = tmp_path / "spectrum.png"
        runner = CliRunner()
        mseed_files = list(mseed_dir.glob("*.mseed"))
        result = runner.invoke(
            cli,
            [
                "plot",
                "--input",
                str(mseed_files[0]),
                "--plot-type",
                "spectrum",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_plot_with_figsize(self, mseed_dir, tmp_path):
        output = tmp_path / "small.png"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "plot",
                "--input",
                str(mseed_dir),
                "--output",
                str(output),
                "--figsize",
                "8x4",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_plot_bad_figsize(self, mseed_dir, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "plot",
                "--input",
                str(mseed_dir),
                "--output",
                str(tmp_path / "out.png"),
                "--figsize",
                "invalid",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid figsize" in result.output

    def test_plot_channel_filter(self, mseed_dir, tmp_path):
        output = tmp_path / "filtered.png"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "plot",
                "--input",
                str(mseed_dir),
                "--output",
                str(output),
                "--channel",
                "BHZ",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_plot_no_matching_channel(self, mseed_dir, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "plot",
                "--input",
                str(mseed_dir),
                "--output",
                str(tmp_path / "out.png"),
                "--channel",
                "XYZ",
            ],
        )
        assert result.exit_code != 0
        assert "No traces" in result.output
