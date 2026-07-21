"""Tests for swe.location.nonlinloc."""

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from obspy import Catalog, UTCDateTime
from obspy.core.event import (
    Event,
    Origin,
    Pick,
    QuantityError,
    WaveformStreamID,
)

from swe.location.nonlinloc import (
    _format_sec_fraction,
    _measure_amplitude,
    catalog_to_obs,
    extract_nll_locations,
    hypocenter_to_obspy_origin,
    picks_to_nonlinloc,
    process_nll_hypocenters,
    run_nonlinloc,
)


# Minimal NonLinLoc .hyp file content for testing
MINIMAL_HYP = """\
NLLOC "./loc/RR" "LOCATED" "Location completed."
HYPOCENTER  x 694.337  y 5155.760  z 0.050  OT 56.45
QUALITY  Pmax  -0.01  MFmin  2.0  MFmax  4.5  RMS  0.15  Nphs  10
STATISTICS  ExpectX 694.337  Y 5155.760  Z 0.050  CovXX 0.001  CovXY 0.0  CovXZ 0.0  CovYY 0.001  CovYZ 0.0  CovZZ 0.0025
QML_OriginUncertainty  horUncert  0.020  minHorUncert  0.010  maxHorUncert  0.030  azMaxHorUncert  45.0
SCATTER  include  4
 694.330 5155.755 0.048 0.1
 694.335 5155.758 0.051 0.2
 694.340 5155.762 0.049 0.3
 694.345 5155.765 0.052 0.15
END_NLLOC
"""

# HYP with a rejected event (no "Location completed")
REJECTED_HYP = """\
NLLOC "./loc/RR" "REJECTED" "Location aborted."
HYPOCENTER  x 694.000  y 5155.000  z 0.100  OT 10.00
QUALITY  Pmax  -0.01  MFmin  2.0  MFmax  4.5  RMS  0.50  Nphs  3
QML_OriginUncertainty  horUncert  0.100  minHorUncert  0.050  maxHorUncert  0.200  azMaxHorUncert  90.0
END_NLLOC
"""


class TestExtractNllLocations:
    def test_parses_hypocenters(self, tmp_path):
        hyp_file = tmp_path / "test.hyp"
        hyp_file.write_text(MINIMAL_HYP)

        result = extract_nll_locations(hyp_file)
        assert len(result.hypocenters) == 1
        assert result.hypocenters[0, 0] == pytest.approx(694.337)
        assert result.hypocenters[0, 1] == pytest.approx(5155.760)
        # z is multiplied by 1000
        assert result.hypocenters[0, 2] == pytest.approx(50.0)

    def test_parses_scatter_points(self, tmp_path):
        hyp_file = tmp_path / "test.hyp"
        hyp_file.write_text(MINIMAL_HYP)

        result = extract_nll_locations(hyp_file)
        assert len(result.pdfs) == 4
        assert list(result.pdfs.columns) == ["x", "y", "z", "weight"]

    def test_rejects_incomplete_events(self, tmp_path):
        hyp_file = tmp_path / "rejected.hyp"
        hyp_file.write_text(REJECTED_HYP)

        result = extract_nll_locations(hyp_file)
        assert len(result.hypocenters) == 0
        assert len(result.pdfs) == 0

    def test_empty_file(self, tmp_path):
        hyp_file = tmp_path / "empty.hyp"
        hyp_file.write_text("")

        result = extract_nll_locations(hyp_file)
        assert len(result.pdfs) == 0
        assert result.hypocenters.shape == (0, 4)


class TestProcessNllHypocenters:
    def test_extracts_uncertainty(self, tmp_path):
        hyp_file = tmp_path / "test.hyp"
        hyp_file.write_text(MINIMAL_HYP)

        hypocenters = process_nll_hypocenters(hyp_file)
        assert len(hypocenters) == 1
        # maxHorUncert: 0.030 * 1000 = 30.0
        assert hypocenters[0][3] == pytest.approx(30.0)

    def test_multiple_events(self, tmp_path):
        hyp_file = tmp_path / "multi.hyp"
        hyp_file.write_text(MINIMAL_HYP + MINIMAL_HYP)

        hypocenters = process_nll_hypocenters(hyp_file)
        assert len(hypocenters) == 2

    def test_depth_uncertainty_parsed(self, tmp_path):
        """depth_uncertainty_m should be sqrt(CovZZ) * 1000."""
        hyp_file = tmp_path / "test.hyp"
        hyp_file.write_text(MINIMAL_HYP)

        hypocenters = process_nll_hypocenters(hyp_file)
        assert len(hypocenters) == 1
        # CovZZ = 0.0025 -> sqrt = 0.05 -> * 1000 = 50.0
        assert hypocenters[0].depth_uncertainty_m == pytest.approx(50.0)

    def test_rms_parsed(self, tmp_path):
        """rms should be parsed from the QUALITY line."""
        hyp_file = tmp_path / "test.hyp"
        hyp_file.write_text(MINIMAL_HYP)

        hypocenters = process_nll_hypocenters(hyp_file)
        assert hypocenters[0].rms == pytest.approx(0.15)

    def test_backward_compat_positional_access(self, tmp_path):
        """Existing positional access [0:4] must still work."""
        hyp_file = tmp_path / "test.hyp"
        hyp_file.write_text(MINIMAL_HYP)

        hypocenters = process_nll_hypocenters(hyp_file)
        h = hypocenters[0]
        x, y, z, horiz = h[0], h[1], h[2], h[3]
        assert x == pytest.approx(694.337)
        assert horiz == pytest.approx(30.0)


class TestHypocenterToObspyOrigin:
    """Tests for hypocenter_to_obspy_origin."""

    def test_basic_conversion(self, tmp_path):
        """Should produce an Origin with uncertainty fields."""
        hyp_file = tmp_path / "test.hyp"
        hyp_file.write_text(MINIMAL_HYP)

        hypocenters = process_nll_hypocenters(hyp_file)
        origin = hypocenter_to_obspy_origin(
            hypocenters[0], reference_lat=45.0, reference_lon=7.0
        )

        assert origin.depth == pytest.approx(hypocenters[0].z_m)
        assert origin.depth_errors is not None
        assert origin.depth_errors.uncertainty == pytest.approx(50.0)
        assert origin.origin_uncertainty is not None
        assert origin.quality is not None
        assert origin.quality.standard_error == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Helpers for catalog_to_obs tests
# ---------------------------------------------------------------------------


def _make_pick(station, phase, time_str, uncertainty=None):
    """Create an ObsPy Pick with optional uncertainty."""
    wid = WaveformStreamID(
        network_code="XX", station_code=station,
        channel_code="BHZ", location_code="",
    )
    pick = Pick(
        waveform_id=wid,
        time=UTCDateTime(time_str),
        phase_hint=phase,
    )
    if uncertainty is not None:
        pick.time_errors = QuantityError(uncertainty=uncertainty)
    return pick


def _make_catalog(events_data):
    """Build a Catalog from a list of lists of (station, phase, time) tuples."""
    cat = Catalog()
    for picks_data in events_data:
        ev = Event()
        for item in picks_data:
            station, phase, time_str = item[:3]
            unc = item[3] if len(item) > 3 else None
            ev.picks.append(_make_pick(station, phase, time_str, unc))
        cat.events.append(ev)
    return cat


class TestCatalogToObs:
    def test_single_event_p_and_s(self, tmp_path):
        """Event with P and S picks produces both lines."""
        cat = _make_catalog([
            [
                ("STA1", "P", "2024-01-15T10:30:00.1234"),
                ("STA1", "S", "2024-01-15T10:30:01.5678"),
                ("STA2", "P", "2024-01-15T10:30:00.5000"),
            ],
        ])
        obs_file = tmp_path / "test.obs"
        result = catalog_to_obs(cat, output_path=obs_file)

        assert result.exists()
        content = result.read_text()
        assert "EQEVENT" in content
        assert "STA1" in content
        assert "STA2" in content
        # Should contain both P and S lines
        lines = [ln for ln in content.splitlines() if not ln.startswith("#")]
        assert len(lines) == 3

    def test_multiple_events(self, tmp_path):
        """Multiple events produce separate EQEVENT blocks."""
        cat = _make_catalog([
            [("STA1", "P", "2024-01-15T10:30:00")],
            [("STA2", "P", "2024-01-15T11:00:00")],
        ])
        obs_file = tmp_path / "multi.obs"
        catalog_to_obs(cat, output_path=obs_file)

        content = obs_file.read_text()
        assert content.count("EQEVENT") == 2

    def test_custom_uncertainty(self, tmp_path):
        """Pick-level uncertainty is written to the GAU field."""
        cat = _make_catalog([
            [("STA1", "P", "2024-01-15T10:30:00", 0.05)],
        ])
        obs_file = tmp_path / "unc.obs"
        catalog_to_obs(cat, output_path=obs_file)

        content = obs_file.read_text()
        assert "5.00e-02" in content

    def test_default_uncertainty(self, tmp_path):
        """Picks without time_errors use default_uncertainty."""
        cat = _make_catalog([
            [("STA1", "P", "2024-01-15T10:30:00")],
        ])
        obs_file = tmp_path / "default_unc.obs"
        catalog_to_obs(cat, output_path=obs_file, default_uncertainty=0.2)

        content = obs_file.read_text()
        assert "2.00e-01" in content

    def test_skips_duplicate_station_phase(self, tmp_path):
        """Duplicate station+phase combos within one event are skipped."""
        cat = _make_catalog([
            [
                ("STA1", "P", "2024-01-15T10:30:00"),
                ("STA1", "P", "2024-01-15T10:30:00.1"),
            ],
        ])
        obs_file = tmp_path / "dedup.obs"
        catalog_to_obs(cat, output_path=obs_file)

        content = obs_file.read_text()
        phase_lines = [ln for ln in content.splitlines() if not ln.startswith("#")]
        assert len(phase_lines) == 1

    def test_empty_catalog(self, tmp_path):
        """Empty catalog produces empty file."""
        cat = Catalog()
        obs_file = tmp_path / "empty.obs"
        catalog_to_obs(cat, output_path=obs_file)

        content = obs_file.read_text()
        assert content == ""

    def test_skips_unknown_phases(self, tmp_path):
        """Phases other than P/S are skipped."""
        cat = _make_catalog([
            [
                ("STA1", "P", "2024-01-15T10:30:00"),
                ("STA1", "Lg", "2024-01-15T10:30:05"),
            ],
        ])
        obs_file = tmp_path / "phases.obs"
        catalog_to_obs(cat, output_path=obs_file)

        content = obs_file.read_text()
        phase_lines = [ln for ln in content.splitlines() if not ln.startswith("#")]
        assert len(phase_lines) == 1

    def test_roundtrip_obs_to_hyp(self, tmp_path):
        """Write .obs, verify it's parseable NLL format."""
        cat = _make_catalog([
            [
                ("STA1", "P", "2024-06-01T12:00:05.1234", 0.02),
                ("STA2", "S", "2024-06-01T12:00:07.5678", 0.05),
            ],
        ])
        obs_file = tmp_path / "roundtrip.obs"
        catalog_to_obs(cat, output_path=obs_file)

        content = obs_file.read_text()
        lines = [ln for ln in content.splitlines() if not ln.startswith("#")]
        assert len(lines) == 2

        # Verify format: station field, date, GAU keyword
        for line in lines:
            assert "GAU" in line
            assert "20240601" in line


class TestFormatSecFraction:
    """Tests for _format_sec_fraction edge cases."""

    def test_zero_microseconds(self):
        """microsecond=0 should produce '00.0000'."""
        assert _format_sec_fraction(0, 0.0) == "00.0000"

    def test_normal_microseconds(self):
        """microsecond=123400 (0.1234) should produce '00.1234'."""
        assert _format_sec_fraction(0, 0.1234) == "00.1234"

    def test_leading_zero_preserved(self):
        """microsecond=50000 (0.05) should produce '00.0500', not '00.5'."""
        assert _format_sec_fraction(0, 0.05) == "00.0500"

    def test_overflow_carries_to_next_second(self):
        """microsecond=999999 rounds to 1.0, should carry over to next second."""
        result = _format_sec_fraction(30, 0.999999)
        assert result == "31.0000"

    def test_nonzero_second(self):
        """Second field is formatted with leading zero."""
        assert _format_sec_fraction(5, 0.5) == "05.5000"


class TestRunNonlinloc:
    def test_missing_control_file(self, tmp_path):
        """Should raise FileNotFoundError for missing control file."""
        with pytest.raises(FileNotFoundError, match="Control file not found"):
            run_nonlinloc(tmp_path / "nonexistent.nll_control")

    @patch("swe.location.nonlinloc.shutil.which", return_value="/usr/bin/NLLoc")
    @patch("swe.location.nonlinloc.subprocess.run")
    def test_invokes_subprocess(self, mock_run, mock_which, tmp_path):
        """Should call subprocess.run with correct arguments."""
        ctrl = tmp_path / "test.nll_control"
        ctrl.write_text("# NLL control file")

        mock_run.return_value = subprocess.CompletedProcess(
            args=["NLLoc", str(ctrl)], returncode=0,
            stdout="OK", stderr="",
        )

        result = run_nonlinloc(ctrl)
        mock_run.assert_called_once_with(
            ["NLLoc", str(ctrl)],
            capture_output=True, text=True, check=True, timeout=300.0,
        )
        assert result.returncode == 0

    @patch("swe.location.nonlinloc.shutil.which", return_value="/opt/nll/bin/NLLoc")
    @patch("swe.location.nonlinloc.subprocess.run")
    def test_custom_binary(self, mock_run, mock_which, tmp_path):
        """Custom nll_binary is passed through."""
        ctrl = tmp_path / "test.nll_control"
        ctrl.write_text("# NLL control file")

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )

        run_nonlinloc(ctrl, nll_binary="/opt/nll/bin/NLLoc")
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0][0] == "/opt/nll/bin/NLLoc"

    @patch("swe.location.nonlinloc.shutil.which", return_value="/usr/bin/NLLoc")
    @patch("swe.location.nonlinloc.subprocess.run")
    def test_propagates_called_process_error(self, mock_run, mock_which, tmp_path):
        """CalledProcessError from NLL is propagated."""
        ctrl = tmp_path / "test.nll_control"
        ctrl.write_text("# NLL control file")

        mock_run.side_effect = subprocess.CalledProcessError(
            1, "NLLoc", stderr="error"
        )

        with pytest.raises(subprocess.CalledProcessError):
            run_nonlinloc(ctrl)


# ---------------------------------------------------------------------------
# picks_to_nonlinloc tests
# ---------------------------------------------------------------------------

# Snuffler uncertainty marker with two datetime pairs
_UNCERTAINTY_MARKER = (
    "phase: 2024-01-15 10:30:00.0000 2024-01-15 10:30:00.2000  0 "
    "XX.LONGSTATION..BHZ  "
    "abc123def456ghi789jklmnopqrst "
    "2024-01-15  10:30:00.1000 P         None False\n"
)

# Simple phase marker (single datetime)
_SIMPLE_PHASE_MARKER = (
    "phase: 2024-01-15 10:30:00.1000  0 XX.STA1..BHZ  "
    "evt001 2024-01-15  10:30:00.1000 P         None False\n"
)

# Non-phase line
_EVENT_LINE = (
    "event: 2024-01-15 10:30:00.0000  0 evt001          "
    "0.0          0.0 None         None None  Event None\n"
)


class TestPicksToNonlinloc:
    """Tests for picks_to_nonlinloc field-based parsing."""

    @patch("swe.location.nonlinloc._measure_amplitude")
    def test_simple_phase_marker(self, mock_amp, tmp_path):
        """Should parse a simple phase marker line."""
        mock_amp.return_value = (1.0, 0.01)
        marker_file = tmp_path / "test.mrkr"
        marker_file.write_text(_SIMPLE_PHASE_MARKER)

        result = picks_to_nonlinloc(
            marker_file, tmp_path, output_path=tmp_path / "out.obs",
        )
        content = result.read_text()
        assert "EQEVENT" in content
        assert "STA1" in content

    @patch("swe.location.nonlinloc._measure_amplitude")
    def test_skips_non_phase_lines(self, mock_amp, tmp_path):
        """Non-phase lines should be skipped."""
        mock_amp.return_value = (1.0, 0.01)
        marker_file = tmp_path / "test.mrkr"
        marker_file.write_text(_EVENT_LINE + _SIMPLE_PHASE_MARKER)

        result = picks_to_nonlinloc(
            marker_file, tmp_path, output_path=tmp_path / "out.obs",
        )
        content = result.read_text()
        lines = [ln for ln in content.splitlines() if not ln.startswith("#")]
        assert len(lines) == 1

    @patch("swe.location.nonlinloc._measure_amplitude")
    def test_nan_amplitude_replaced_with_zero(self, mock_amp, tmp_path):
        """NaN amplitude from _measure_amplitude should become 0.0 in output."""
        mock_amp.return_value = (np.nan, np.nan)
        marker_file = tmp_path / "test.mrkr"
        marker_file.write_text(_SIMPLE_PHASE_MARKER)

        result = picks_to_nonlinloc(
            marker_file, tmp_path, output_path=tmp_path / "out.obs",
        )
        content = result.read_text()
        assert "0.00e+00" in content


# ---------------------------------------------------------------------------
# _measure_amplitude tests
# ---------------------------------------------------------------------------


class TestMeasureAmplitude:
    """Tests for _measure_amplitude."""

    def test_returns_nan_on_missing_files(self, tmp_path):
        """Should return NaN when no day files exist."""
        amp, period = _measure_amplitude(
            tmp_path, "STA1", "BHZ",
            UTCDateTime("2024-01-15T10:30:00"),
            UTCDateTime("2024-01-15T10:30:01"),
            UTCDateTime("2024-01-15T10:30:00.5"),
        )
        assert np.isnan(amp)
        assert np.isnan(period)

    @patch("swe.location.nonlinloc.read")
    def test_returns_nan_on_read_error(self, mock_read, tmp_path):
        """Should return NaN when read raises an error."""
        # Create a dummy file to pass the glob check
        day_file = tmp_path / "XX.STA1..BHZ.2024-01-15.ms"
        day_file.write_text("dummy")
        mock_read.side_effect = OSError("corrupt file")

        amp, period = _measure_amplitude(
            tmp_path, "STA1", "BHZ",
            UTCDateTime("2024-01-15T10:30:00"),
            UTCDateTime("2024-01-15T10:30:01"),
            UTCDateTime("2024-01-15T10:30:00.5"),
        )
        assert np.isnan(amp)
        assert np.isnan(period)
