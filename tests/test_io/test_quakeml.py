"""Tests for swe.io.quakeml."""

import pytest
from obspy import UTCDateTime
from obspy.core.event import (
    Catalog,
    CreationInfo,
    Event,
    Magnitude,
    Origin,
    Pick,
    WaveformStreamID,
)
from pathlib import Path

from swe.io.quakeml import read_quakeml, write_quakeml


def _make_catalog(n_events=2):
    """Create a synthetic ObsPy Catalog for testing."""
    cat = Catalog()
    for i in range(n_events):
        event = Event()
        origin = Origin(
            time=UTCDateTime(2020, 1, 1 + i),
            latitude=46.0 + i * 0.1,
            longitude=-122.0 + i * 0.1,
            depth=5000.0 + i * 1000,
        )
        event.origins.append(origin)
        event.preferred_origin_id = origin.resource_id
        mag = Magnitude(mag=2.5 + i * 0.5, magnitude_type="ML")
        event.magnitudes.append(mag)
        cat.events.append(event)
    return cat


class TestQuakeML:
    """Tests for QuakeML read/write wrappers."""

    def test_roundtrip(self, tmp_path):
        """Write then read should recover the same catalog structure."""
        cat = _make_catalog(3)
        output = tmp_path / "catalog.xml"

        result_path = write_quakeml(cat, output)
        assert result_path.exists()

        read_cat = read_quakeml(output)
        assert len(read_cat) == 3

    def test_agency_id_default(self, tmp_path):
        """Events without agency_id should get 'SWE' by default."""
        cat = _make_catalog(1)
        output = tmp_path / "agency.xml"
        write_quakeml(cat, output)

        read_cat = read_quakeml(output)
        event = read_cat[0]
        assert event.creation_info.agency_id == "SWE"

    def test_custom_agency_id(self, tmp_path):
        """Custom agency_id should be written."""
        cat = _make_catalog(1)
        output = tmp_path / "custom.xml"
        write_quakeml(cat, output, agency_id="TEST")

        read_cat = read_quakeml(output)
        assert read_cat[0].creation_info.agency_id == "TEST"

    def test_empty_catalog(self, tmp_path):
        """Empty catalog should round-trip without error."""
        cat = Catalog()
        output = tmp_path / "empty.xml"
        write_quakeml(cat, output)

        read_cat = read_quakeml(output)
        assert len(read_cat) == 0

    def test_preserves_origin_data(self, tmp_path):
        """Origin lat/lon/depth should survive round-trip."""
        cat = _make_catalog(1)
        output = tmp_path / "origin.xml"
        write_quakeml(cat, output)

        read_cat = read_quakeml(output)
        orig = read_cat[0].origins[0]
        assert abs(orig.latitude - 46.0) < 0.01
        assert abs(orig.longitude - (-122.0)) < 0.01


# ---------------------------------------------------------------------------
# Round-trip preservation tests
# ---------------------------------------------------------------------------


class TestQuakeMLRoundTrip:
    """Extended round-trip tests for data preservation."""

    def test_preserves_magnitude_data(self, tmp_path):
        """Magnitude value and type should survive a write/read cycle."""
        cat = _make_catalog(1)
        output = tmp_path / "mag.xml"
        write_quakeml(cat, output)

        read_cat = read_quakeml(output)
        mag = read_cat[0].magnitudes[0]
        assert abs(mag.mag - 2.5) < 0.01
        assert mag.magnitude_type == "ML"

    def test_preserves_pick_data(self, tmp_path):
        """Picks attached to events should survive round-trip."""
        cat = _make_catalog(1)
        pick = Pick(
            time=UTCDateTime("2020-01-01T00:01:00"),
            waveform_id=WaveformStreamID(
                network_code="XX", station_code="STA1", channel_code="BHZ",
            ),
            phase_hint="P",
        )
        cat[0].picks.append(pick)
        output = tmp_path / "picks.xml"
        write_quakeml(cat, output)

        read_cat = read_quakeml(output)
        assert len(read_cat[0].picks) == 1
        assert read_cat[0].picks[0].phase_hint == "P"

    def test_preserves_event_count_large(self, tmp_path):
        """50 events should round-trip correctly."""
        cat = Catalog()
        for i in range(50):
            event = Event()
            origin = Origin(
                time=UTCDateTime(2020, 1, 1) + i * 86400,
                latitude=46.0, longitude=-122.0, depth=5000.0,
            )
            event.origins.append(origin)
            event.preferred_origin_id = origin.resource_id
            cat.events.append(event)
        output = tmp_path / "large.xml"
        write_quakeml(cat, output)

        read_cat = read_quakeml(output)
        assert len(read_cat) == 50

    def test_string_path_accepted(self, tmp_path):
        """str path (instead of Path) should work for both write and read."""
        cat = _make_catalog(1)
        output = str(tmp_path / "strpath.xml")
        result_path = write_quakeml(cat, output)
        assert result_path.exists()

        read_cat = read_quakeml(output)
        assert len(read_cat) == 1

    def test_return_type_is_path(self, tmp_path):
        """write_quakeml should return a Path object."""
        cat = _make_catalog(1)
        output = tmp_path / "rettype.xml"
        result = write_quakeml(cat, output)
        assert isinstance(result, Path)

    def test_overwrite_existing_file(self, tmp_path):
        """Second write to the same path should overwrite cleanly."""
        output = tmp_path / "overwrite.xml"
        write_quakeml(_make_catalog(2), output)
        write_quakeml(_make_catalog(5), output)

        read_cat = read_quakeml(output)
        assert len(read_cat) == 5


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


class TestQuakeMLErrors:
    """Error-path tests for QuakeML I/O."""

    def test_read_nonexistent_file_raises(self, tmp_path):
        """Reading a nonexistent file should raise an exception."""
        fake_path = tmp_path / "does_not_exist.xml"
        with pytest.raises(Exception):
            read_quakeml(fake_path)

    def test_agency_id_preserved_when_set(self, tmp_path):
        """Events with an existing agency_id should keep it, not overwrite."""
        cat = Catalog()
        event = Event()
        event.creation_info = CreationInfo(agency_id="USGS")
        origin = Origin(
            time=UTCDateTime(2020, 1, 1), latitude=46.0,
            longitude=-122.0, depth=5000.0,
        )
        event.origins.append(origin)
        cat.events.append(event)

        output = tmp_path / "agency.xml"
        write_quakeml(cat, output, agency_id="SWE")

        read_cat = read_quakeml(output)
        assert read_cat[0].creation_info.agency_id == "USGS"
