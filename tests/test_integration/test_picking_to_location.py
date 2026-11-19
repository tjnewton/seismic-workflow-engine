"""Integration tests: picking -> association -> catalog -> location."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


class TestPickToAssociation:
    """Verify that pick formats feed cleanly into association."""

    def test_pick_format_compatible_with_association(
        self, synthetic_station_network, synthetic_pick_times
    ):
        """pick_stream output format directly feeds associate_picks."""
        from swe.association import associate_picks

        net = synthetic_station_network
        result = associate_picks(
            synthetic_pick_times,
            net["stations"],
            net["location_dict"],
            dbscan_eps=1.0,
            dbscan_min_samples=3,
            min_picks_per_cluster=3,
        )
        # Should return an AssociationResult (6-tuple with cluster_quality)
        assert len(result) == 6
        assert isinstance(result.station_distance, dict)
        assert isinstance(result.labels, np.ndarray)

    def test_association_clusters_synthetic_events(
        self, synthetic_station_network, synthetic_pick_times
    ):
        """Two well-separated events should form two DBSCAN clusters."""
        from swe.association import associate_picks

        net = synthetic_station_network
        result = associate_picks(
            synthetic_pick_times,
            net["stations"],
            net["location_dict"],
            dbscan_eps=1.2,
            dbscan_min_samples=3,
            min_picks_per_cluster=3,
            temporal_threshold=15.0,
            spatial_threshold=10.0,
        )
        non_noise = set(result.labels) - {-1}
        assert len(non_noise) >= 2, f"Expected at least 2 clusters, got {len(non_noise)}"


class TestAssociationToCatalog:
    """Verify association results convert to ObsPy Catalog."""

    def test_association_result_to_catalog_preserves_pick_count(
        self, synthetic_station_network, synthetic_pick_times
    ):
        """Total picks in Catalog matches non-noise association labels."""
        from swe.association import associate_picks, association_to_catalog

        net = synthetic_station_network
        result = associate_picks(
            synthetic_pick_times,
            net["stations"],
            net["location_dict"],
            dbscan_eps=1.0,
            dbscan_min_samples=3,
            min_picks_per_cluster=3,
            temporal_threshold=5.0,
            spatial_threshold=10.0,
        )
        catalog = association_to_catalog(result)

        non_noise_count = int(np.sum(result.labels != -1))
        catalog_pick_count = sum(len(ev.picks) for ev in catalog.events)
        assert catalog_pick_count == non_noise_count


class TestCatalogToLocation:
    """Verify catalog picks provide valid inputs for location."""

    def test_catalog_picks_provide_valid_travel_times(self, synthetic_station_network):
        """LocationResult from synthetic travel times has finite coords."""
        from swe.location import locate_by_travel_time

        net = synthetic_station_network
        station_coords = net["station_coords"]

        # Synthetic source at (5000, 2500, 1000) meters
        true_loc = (5000.0, 2500.0, 1000.0)
        velocity = 5000.0  # m/s
        dists = np.sqrt(np.sum((station_coords - true_loc) ** 2, axis=1))
        observed_times = dists / velocity

        result = locate_by_travel_time(
            observed_times,
            station_coords,
            velocity,
            grid_bounds=((0, 10000), (0, 5000), (0, 2000)),
            grid_spacing=(500.0, 500.0, 500.0),
        )
        assert np.isfinite(result.x)
        assert np.isfinite(result.y)
        assert np.isfinite(result.z)
        assert result.misfit >= 0.0

    def test_located_events_have_reasonable_coordinates(self, synthetic_station_network):
        """Synthetic event travel times recover source position within grid."""
        from swe.location import locate_by_travel_time

        net = synthetic_station_network
        station_coords = net["station_coords"]

        true_loc = (5000.0, 2500.0, 1000.0)
        velocity = 5000.0
        dists = np.sqrt(np.sum((station_coords - true_loc) ** 2, axis=1))
        observed_times = dists / velocity

        result = locate_by_travel_time(
            observed_times,
            station_coords,
            velocity,
            grid_bounds=((0, 10000), (0, 5000), (0, 2000)),
            grid_spacing=(250.0, 250.0, 250.0),
        )
        # Should recover source within ~500m given 250m grid spacing
        assert abs(result.x - true_loc[0]) <= 500.0
        assert abs(result.y - true_loc[1]) <= 500.0
        assert abs(result.z - true_loc[2]) <= 500.0


class TestFullPipeline:
    """End-to-end: mocked DL picks -> real association -> real location."""

    def test_full_pipeline_produces_located_events(
        self, synthetic_station_network, synthetic_pick_times
    ):
        """Full pipeline from picks to located events produces results."""
        from swe.association import associate_picks, association_to_catalog
        from swe.location import locate_by_travel_time

        net = synthetic_station_network

        # Step 1: Association
        result = associate_picks(
            synthetic_pick_times,
            net["stations"],
            net["location_dict"],
            dbscan_eps=1.0,
            dbscan_min_samples=3,
            min_picks_per_cluster=3,
            temporal_threshold=5.0,
            spatial_threshold=10.0,
        )

        # Step 2: Convert to catalog
        catalog = association_to_catalog(result)
        assert len(catalog.events) >= 1

        # Step 3: For each event, extract pick times and locate
        station_coords = net["station_coords"]
        stations = net["stations"]
        velocity = 5000.0

        for event in catalog.events:
            if len(event.picks) < 3:
                continue
            # Build observed times array (relative seconds from first pick)
            pick_times_utc = sorted(p.time for p in event.picks)
            origin_time = pick_times_utc[0]

            sta_indices = []
            obs_times = []
            for pick in event.picks:
                sta = pick.waveform_id.station_code
                if sta in stations:
                    idx = stations.index(sta)
                    sta_indices.append(idx)
                    obs_times.append(float(pick.time - origin_time))

            if len(sta_indices) < 3:
                continue

            coords = station_coords[sta_indices]
            obs_arr = np.array(obs_times)

            loc_result = locate_by_travel_time(
                obs_arr,
                coords,
                velocity,
                grid_bounds=((0, 10000), (0, 5000), (0, 2000)),
                grid_spacing=(500.0, 500.0, 500.0),
            )
            assert np.isfinite(loc_result.x)
            assert np.isfinite(loc_result.misfit)
