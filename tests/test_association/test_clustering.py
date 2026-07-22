"""Tests for swe.association.clustering."""

from unittest.mock import patch

import numpy as np
import pytest
from matplotlib.dates import date2num
from obspy import UTCDateTime

from swe.association.clustering import (
    AssociationResult,
    associate_picks,
    build_pick_data,
    build_pick_dict,
    cluster_picks,
    get_cluster_stats,
    process_clusters,
    save_associations,
)


class TestClusterPicks:
    """Tests for cluster_picks with synthetic 2-D data."""

    def test_two_distinct_clusters(self):
        """Two well-separated blobs should yield two clusters."""
        rng = np.random.default_rng(42)
        # Cluster A centred at (0, 0, 0, 0)
        cluster_a = rng.normal(loc=0.0, scale=0.1, size=(30, 4))
        # Cluster B centred at (10, 10, 10, 10)
        cluster_b = rng.normal(loc=10.0, scale=0.1, size=(30, 4))
        data = np.vstack([cluster_a, cluster_b])

        labels = cluster_picks(
            data,
            dbscan_eps=0.8,
            dbscan_min_samples=3,
        )

        assert len(labels) == 60
        unique = set(labels)
        # Remove noise label if present
        unique.discard(-1)
        assert len(unique) == 2, f"Expected 2 clusters, got {len(unique)}"

    def test_all_noise_with_tight_eps(self):
        """Widely scattered points with very tight eps should all be noise."""
        rng = np.random.default_rng(99)
        data = rng.uniform(0, 100, size=(20, 4))

        labels = cluster_picks(
            data,
            dbscan_eps=0.001,
            dbscan_min_samples=5,
        )

        assert len(labels) == 20
        assert all(label == -1 for label in labels), "Expected all noise labels"

    def test_single_cluster(self):
        """Tightly grouped points should form a single cluster.

        After standardisation the data has unit variance, so eps must be
        large enough to capture the neighbourhood.
        """
        rng = np.random.default_rng(7)
        data = rng.normal(loc=5.0, scale=1.0, size=(50, 4))

        labels = cluster_picks(
            data,
            dbscan_eps=3.0,
            dbscan_min_samples=3,
        )

        unique = set(labels)
        unique.discard(-1)
        assert len(unique) == 1

    def test_pca_option(self):
        """Clustering with PCA enabled should still produce valid labels."""
        rng = np.random.default_rng(42)
        cluster_a = rng.normal(loc=0.0, scale=0.1, size=(25, 4))
        cluster_b = rng.normal(loc=10.0, scale=0.1, size=(25, 4))
        data = np.vstack([cluster_a, cluster_b])

        labels = cluster_picks(
            data,
            use_pca=True,
            pca_components=2,
            dbscan_eps=0.8,
            dbscan_min_samples=3,
        )

        assert len(labels) == 50

    def test_scale_time(self):
        """Scaling the time dimension should not crash and should return labels."""
        rng = np.random.default_rng(123)
        data = rng.normal(loc=0.0, scale=1.0, size=(30, 4))

        labels = cluster_picks(
            data,
            scale_time=True,
            scale_factor=10.0,
            dbscan_eps=1.0,
            dbscan_min_samples=3,
        )

        assert len(labels) == 30

    def test_unsupported_algorithm_raises(self):
        """An unsupported algorithm name should raise ValueError."""
        data = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        with pytest.raises(ValueError, match="Unsupported"):
            cluster_picks(data, clustering_algo="OPTICS")


class TestGetClusterStats:
    """Tests for get_cluster_stats with known labels."""

    def test_single_cluster(self):
        """Single cluster should return one median."""
        labels = np.array([0, 0, 0, 0])
        pick_times = [1.0, 2.0, 3.0, 4.0]

        unique_labels, medians = get_cluster_stats(labels, pick_times)

        assert unique_labels == {0}
        assert medians[0] == pytest.approx(2.5)

    def test_noise_and_signal(self):
        """Noise and signal clusters should have separate medians."""
        labels = np.array([-1, -1, 0, 0, 0, 1, 1])
        pick_times = [0.5, 0.6, 10.0, 11.0, 12.0, 20.0, 22.0]

        unique_labels, medians = get_cluster_stats(labels, pick_times)

        assert unique_labels == {-1, 0, 1}
        assert medians[-1] == pytest.approx(0.55)
        assert medians[0] == pytest.approx(11.0)
        assert medians[1] == pytest.approx(21.0)

    def test_empty_labels(self):
        """Empty labels should return empty results."""
        labels = np.array([], dtype=int)
        pick_times = []

        unique_labels, medians = get_cluster_stats(labels, pick_times)

        assert unique_labels == set()
        assert medians == {}

    def test_all_noise(self):
        """All noise should return only the -1 label."""
        labels = np.array([-1, -1, -1])
        pick_times = [1.0, 2.0, 3.0]

        unique_labels, medians = get_cluster_stats(labels, pick_times)

        assert unique_labels == {-1}
        assert medians[-1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Tests for build_pick_dict
# ---------------------------------------------------------------------------


class TestBuildPickDict:
    """Tests for build_pick_dict."""

    def test_basic_picks_grouped_by_station(self):
        """3 stations with 2 picks each should produce correct grouping."""
        reverse = {0: "STA1", 1: "STA2", 2: "STA3"}
        # Build pick times as matplotlib date numbers
        base = UTCDateTime("2024-01-01T00:00:00")
        times = [date2num((base + i * 10).datetime) for i in range(6)]
        station_order = [0, 1, 2, 0, 1, 2]

        result = build_pick_dict(
            np.array(times),
            station_order,
            reverse,
            duplicate_threshold=10.0,
            sampling_rate=250.0,
        )

        for sta in ("STA1", "STA2", "STA3"):
            assert len(result[sta]) == 2
            for val in result[sta]:
                assert isinstance(val, UTCDateTime)

    def test_empty_pick_times_returns_empty_lists(self):
        """Empty pick_times triggers early return with empty lists."""
        reverse = {0: "STA1", 1: "STA2"}
        result = build_pick_dict([], [], reverse)

        assert result == {"STA1": [], "STA2": []}

    def test_duplicate_picks_removed(self):
        """Picks within threshold/sampling_rate seconds are deduplicated."""
        reverse = {0: "STA1"}
        base = UTCDateTime("2024-01-01T00:00:00")
        t1 = date2num(base.datetime)
        t2 = date2num((base + 0.01).datetime)  # 10 ms apart

        result = build_pick_dict(
            [t1, t2],
            [0, 0],
            reverse,
            duplicate_threshold=10.0,
            sampling_rate=250.0,  # threshold = 10/250 = 0.04 s
        )

        assert len(result["STA1"]) == 1

    def test_station_list_override(self):
        """station_list overrides reverse_station_distance keys."""
        reverse = {0: "STA1", 1: "STA2"}
        result = build_pick_dict(
            [],
            [],
            reverse,
            station_list=["X", "Y", "Z"],
        )

        assert set(result.keys()) == {"X", "Y", "Z"}

    def test_all_picks_are_duplicates(self):
        """When all picks are duplicates, only the first survives."""
        reverse = {0: "STA1"}
        base = UTCDateTime("2024-01-01T00:00:00")
        # 5 picks each 1 ms apart - all within 10/250=0.04 s
        times = [date2num((base + i * 0.001).datetime) for i in range(5)]
        station_order = [0] * 5

        result = build_pick_dict(
            times,
            station_order,
            reverse,
            duplicate_threshold=10.0,
            sampling_rate=250.0,
        )

        assert len(result["STA1"]) == 1


# ---------------------------------------------------------------------------
# Tests for build_pick_data
# ---------------------------------------------------------------------------


class TestBuildPickData:
    """Tests for build_pick_data."""

    def _make_station_picks(self, base, offsets):
        """Helper: create station_picks dict from time offsets."""
        return {sta: [base + dt for dt in dts] for sta, dts in offsets.items()}

    def test_basic_transformation(self):
        """Pick array should contain [lon, lat, elev, mpl_date]."""
        base = UTCDateTime("2024-06-15T12:00:00")
        station_picks = {"STA1": [base]}
        location_dict = {"STA1": [-120.0, 35.0, 500.0]}
        station_distance = {"STA1": 0}

        pa, pso, pt = build_pick_data(
            station_picks,
            location_dict,
            station_distance,
        )

        assert len(pa) == 1
        assert pa[0][0] == -120.0
        assert pa[0][1] == 35.0
        assert pa[0][2] == 500.0
        assert pa[0][3] == pytest.approx(base.matplotlib_date)
        assert pso == [0]
        assert pt[0] == pytest.approx(base.matplotlib_date)

    def test_empty_station_picks(self):
        """Empty station_picks -> empty outputs."""
        pa, pso, pt = build_pick_data(
            {"STA1": []},
            {"STA1": [0.0, 0.0, 0.0]},
            {"STA1": 0},
        )

        assert pa == []
        assert pso == []
        assert pt == []

    def test_multiple_picks_per_station(self):
        """Multiple picks per station expand correctly."""
        base = UTCDateTime("2024-06-15T12:00:00")
        station_picks = {"A": [base, base + 10], "B": [base + 5]}
        location_dict = {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}
        station_distance = {"A": 0, "B": 1}

        pa, pso, _pt = build_pick_data(
            station_picks,
            location_dict,
            station_distance,
        )

        assert len(pa) == 3
        assert pso.count(0) == 2
        assert pso.count(1) == 1

    def test_location_dict_not_mutated(self):
        """build_pick_data should not mutate the caller's location_dict."""
        base = UTCDateTime("2024-06-15T12:00:00")
        station_picks = {"STA1": [base]}
        location_dict = {"STA1": [1.0, 2.0, 3.0]}
        station_distance = {"STA1": 0}

        build_pick_data(station_picks, location_dict, station_distance)

        assert location_dict["STA1"] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Tests for process_clusters
# ---------------------------------------------------------------------------


class TestProcessClusters:
    """Tests for process_clusters."""

    @staticmethod
    def _make_pick_times(base, offsets_sec):
        """Return mpl-date pick_times from a base UTCDateTime + offsets."""
        return [date2num((base + dt).datetime) for dt in offsets_sec]

    def test_small_cluster_dissolved(self):
        """Clusters below min_picks_per_cluster get dissolved to noise."""
        labels = np.array([0, 0, -1])
        pick_times = [1.0, 1.001, 0.5]
        station_order = [0, 1, 2]

        result = process_clusters(
            labels,
            pick_times,
            station_order,
            min_picks_per_cluster=4,
        )

        assert np.all(result == -1)

    def test_large_cluster_retained(self):
        """A cluster meeting all criteria is retained."""
        base = UTCDateTime("2024-01-01T12:00:00")
        times = self._make_pick_times(base, [0, 0.1, 0.2, 0.3, 0.4])
        labels = np.array([0, 0, 0, 0, 0])
        station_order = [0, 1, 2, 3, 4]

        result = process_clusters(
            labels,
            times,
            station_order,
            min_picks_per_cluster=4,
            temporal_threshold=1.0,
            spatial_threshold=10.0,
        )

        assert np.sum(result == 0) >= 4

    def test_duplicate_station_picks_removed(self):
        """Two picks on the same station within a cluster - farther one expelled."""
        base = UTCDateTime("2024-01-01T12:00:00")
        # 5 picks: stations [0,1,2,3,0] - station 0 duplicated
        times = self._make_pick_times(base, [0, 0.1, 0.2, 0.3, 0.5])
        labels = np.array([0, 0, 0, 0, 0])
        station_order = [0, 1, 2, 3, 0]

        result = process_clusters(
            labels,
            times,
            station_order,
            min_picks_per_cluster=4,
            temporal_threshold=2.0,
            spatial_threshold=10.0,
        )

        # One of the station-0 picks should have been expelled
        cluster_members = np.where(result == 0)[0]
        cluster_stations = [station_order[i] for i in cluster_members]
        assert cluster_stations.count(0) == 1

    def test_temporal_misfit_removed(self):
        """A pick far from the cluster median in time is expelled."""
        base = UTCDateTime("2024-01-01T12:00:00")
        # 4 close picks + 1 outlier at +100s
        times = self._make_pick_times(base, [0, 0.1, 0.2, 0.3, 100])
        labels = np.array([0, 0, 0, 0, 0])
        station_order = [0, 1, 2, 3, 4]

        result = process_clusters(
            labels,
            times,
            station_order,
            min_picks_per_cluster=4,
            temporal_threshold=1.0,
            spatial_threshold=10.0,
        )

        # Outlier (index 4) should be expelled
        assert result[4] == -1

    def test_spatial_misfit_removed(self):
        """A pick on a distant station index is expelled."""
        base = UTCDateTime("2024-01-01T12:00:00")
        times = self._make_pick_times(base, [0, 0.1, 0.2, 0.3, 0.4])
        labels = np.array([0, 0, 0, 0, 0])
        # Stations 0-3 are close; station 100 is a spatial outlier
        station_order = [0, 1, 2, 3, 100]

        result = process_clusters(
            labels,
            times,
            station_order,
            min_picks_per_cluster=4,
            temporal_threshold=2.0,
            spatial_threshold=5.0,
        )

        assert result[4] == -1

    def test_merge_clusters_enabled(self):
        """Two close, disjoint-station clusters should merge."""
        base = UTCDateTime("2024-01-01T12:00:00")
        # Cluster 0: stations 0-3, cluster 1: stations 4-7
        times_0 = self._make_pick_times(base, [0, 0.1, 0.2, 0.3])
        times_1 = self._make_pick_times(base, [0.05, 0.15, 0.25, 0.35])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        station_order = [0, 1, 2, 3, 4, 5, 6, 7]

        result = process_clusters(
            labels,
            times_0 + times_1,
            station_order,
            min_picks_per_cluster=4,
            temporal_threshold=2.0,
            spatial_threshold=10.0,
            merge_clusters=True,
        )

        non_noise = result[result != -1]
        if len(non_noise) > 0:
            assert len(set(non_noise)) == 1, "Clusters should have merged"

    def test_noise_reassignment(self):
        """A noise pick near a cluster should be reassigned when process_noise=True."""
        base = UTCDateTime("2024-01-01T12:00:00")
        # 4 cluster picks + 1 noise pick close in time and on a nearby station
        times = self._make_pick_times(base, [0, 0.1, 0.2, 0.3, 0.15])
        labels = np.array([0, 0, 0, 0, -1])
        station_order = [0, 1, 2, 3, 5]  # station 5 is nearby

        result = process_clusters(
            labels,
            times,
            station_order,
            min_picks_per_cluster=4,
            temporal_threshold=2.0,
            spatial_threshold=10.0,
            process_noise=True,
        )

        assert result[4] == 0

    def test_returns_ndarray(self):
        """Return type should be a numpy array."""
        result = process_clusters(
            np.array([-1, -1]),
            [1.0, 2.0],
            [0, 1],
        )

        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Tests for associate_picks
# ---------------------------------------------------------------------------


class TestAssociatePicks:
    """Tests for associate_picks."""

    def test_empty_picks_returns_early(self):
        """Empty station_picks -> AssociationResult with empty arrays."""
        result = associate_picks(
            {"A": [], "B": []},
            ["A", "B"],
            {"A": [0.0, 0.0, 0.0], "B": [1.0, 1.0, 1.0]},
        )

        assert result.station_distance == {"A": 0, "B": 1}
        assert result.reverse_station_distance == {0: "A", 1: "B"}
        assert result.pick_station_order == []
        assert result.pick_times == []
        assert len(result.labels) == 0

    def test_station_distance_maps_correct(self):
        """Forward/reverse maps should match station_list order."""
        station_list = ["X", "Y", "Z"]
        result = associate_picks(
            {"X": [], "Y": [], "Z": []},
            station_list,
            {"X": [0, 0, 0], "Y": [1, 1, 1], "Z": [2, 2, 2]},
        )

        for idx, sta in enumerate(station_list):
            assert result.station_distance[sta] == idx
            assert result.reverse_station_distance[idx] == sta

    def test_returns_association_result_structure(self):
        """Pipeline should return an AssociationResult with matching lengths."""
        base = UTCDateTime("2024-01-01T12:00:00")
        station_picks = {
            "A": [base + i for i in range(5)],
            "B": [base + i for i in range(5)],
        }
        station_list = ["A", "B"]
        location_dict = {"A": [0.0, 0.0, 0.0], "B": [1.0, 1.0, 1.0]}

        result = associate_picks(
            station_picks,
            station_list,
            location_dict,
            dbscan_eps=2.0,
            dbscan_min_samples=2,
            min_picks_per_cluster=2,
            temporal_threshold=10.0,
            spatial_threshold=10.0,
        )

        assert isinstance(result, AssociationResult)
        assert len(result.labels) == len(result.pick_station_order) == len(result.pick_times)

    @patch("swe.association.clustering.process_clusters")
    @patch("swe.association.clustering.cluster_picks")
    def test_calls_process_clusters_twice(self, mock_cluster, mock_process):
        """process_clusters should be called twice: merge=True then merge=False."""
        base = UTCDateTime("2024-01-01T12:00:00")
        station_picks = {"A": [base], "B": [base + 1]}
        station_list = ["A", "B"]
        location_dict = {"A": [0.0, 0.0, 0.0], "B": [1.0, 1.0, 1.0]}

        mock_cluster.return_value = np.array([0, 0])
        mock_process.return_value = np.array([0, 0])

        associate_picks(station_picks, station_list, location_dict)

        assert mock_process.call_count == 2
        first_call_kwargs = mock_process.call_args_list[0][1]
        second_call_kwargs = mock_process.call_args_list[1][1]
        assert first_call_kwargs["merge_clusters"] is True
        assert second_call_kwargs["merge_clusters"] is False


# ---------------------------------------------------------------------------
# Tests for save_associations
# ---------------------------------------------------------------------------


class TestSaveAssociations:
    """Tests for save_associations."""

    @patch("swe.association.clustering.write_events_to_markers")
    def test_calls_write_events_with_correct_structure(self, mock_write):
        """Events dict should have correct number of keys, each with pick dicts."""
        base = UTCDateTime("2024-01-01T12:00:00")
        times = [date2num((base + i * 0.1).datetime) for i in range(6)]
        labels = np.array([0, 0, 0, 1, 1, 1])
        station_order = [0, 1, 2, 0, 1, 2]
        reverse = {0: "STA1", 1: "STA2", 2: "STA3"}

        save_associations("out.mrkr", labels, times, station_order, reverse)

        mock_write.assert_called_once()
        events = mock_write.call_args[0][0]
        assert len(events) == 2
        for picks in events.values():
            assert len(picks) == 3
            for p in picks:
                assert "station" in p
                assert "time" in p

    @patch("swe.association.clustering.write_events_to_markers")
    def test_noise_labels_excluded(self, mock_write):
        """Cluster -1 should be omitted from output."""
        base = UTCDateTime("2024-01-01T12:00:00")
        times = [date2num((base + i).datetime) for i in range(5)]
        labels = np.array([-1, -1, 0, 0, 0])
        station_order = [0, 1, 2, 3, 4]
        reverse = {i: f"S{i}" for i in range(5)}

        save_associations("out.mrkr", labels, times, station_order, reverse)

        events = mock_write.call_args[0][0]
        assert len(events) == 1  # only cluster 0

    @patch("swe.association.clustering.write_events_to_markers")
    def test_station_channel_naming_convention(self, mock_write):
        """All stations use the default channel (or the one passed via kwarg)."""
        base = UTCDateTime("2024-01-01T12:00:00")
        times = [date2num(base.datetime), date2num((base + 0.1).datetime)]
        labels = np.array([0, 0])
        station_order = [0, 1]
        reverse = {0: "UNET", 1: "STA1"}

        save_associations("out.mrkr", labels, times, station_order, reverse)

        events = mock_write.call_args[0][0]
        picks = next(iter(events.values()))
        station_names = {p["station"] for p in picks}
        assert "UNET.DP1" in station_names
        assert "STA1.DP1" in station_names

    @patch("swe.association.clustering.write_events_to_markers")
    def test_event_hash_is_deterministic(self, mock_write):
        """Same inputs should produce identical hash keys."""
        base = UTCDateTime("2024-01-01T12:00:00")
        times = [date2num((base + i * 0.1).datetime) for i in range(3)]
        labels = np.array([0, 0, 0])
        station_order = [0, 1, 2]
        reverse = {0: "A", 1: "B", 2: "C"}

        save_associations("a.mrkr", labels, times, station_order, reverse)
        keys1 = set(mock_write.call_args[0][0].keys())

        save_associations("b.mrkr", labels, times, station_order, reverse)
        keys2 = set(mock_write.call_args[0][0].keys())

        assert keys1 == keys2


# ---------------------------------------------------------------------------
# Tests for AssociationResult
# ---------------------------------------------------------------------------


class TestComputeClusterQuality:
    """Tests for compute_cluster_quality."""

    def test_basic_quality_metrics(self):
        """Two clusters should produce correct quality dicts."""
        from swe.association.clustering import compute_cluster_quality

        labels = np.array([0, 0, 0, 1, 1, -1])
        pick_times = np.array([0.0, 0.001, 0.002, 0.010, 0.011, 0.050])
        pick_station_order = np.array([0, 1, 2, 3, 4, 5])
        reverse = {i: f"S{i}" for i in range(6)}

        quality = compute_cluster_quality(
            labels,
            pick_times,
            pick_station_order,
            reverse,
        )

        assert set(quality.keys()) == {0, 1}
        assert quality[0]["n_picks"] == 3
        assert quality[0]["n_stations"] == 3
        assert quality[1]["n_picks"] == 2
        assert quality[1]["n_stations"] == 2
        assert quality[0]["time_spread_s"] > 0
        assert quality[0]["station_coverage"] == pytest.approx(3 / 6)

    def test_noise_excluded(self):
        """Noise cluster -1 should not appear in quality."""
        from swe.association.clustering import compute_cluster_quality

        labels = np.array([-1, -1, -1])
        pick_times = np.array([1.0, 2.0, 3.0])
        pick_station_order = np.array([0, 1, 2])
        reverse = {0: "A", 1: "B", 2: "C"}

        quality = compute_cluster_quality(
            labels,
            pick_times,
            pick_station_order,
            reverse,
        )

        assert quality == {}

    def test_single_pick_cluster_zero_spread(self):
        """A cluster with one pick should have zero time spread."""
        from swe.association.clustering import compute_cluster_quality

        labels = np.array([0])
        pick_times = np.array([5.0])
        pick_station_order = np.array([0])
        reverse = {0: "STA1"}

        quality = compute_cluster_quality(
            labels,
            pick_times,
            pick_station_order,
            reverse,
        )

        assert quality[0]["time_spread_s"] == 0.0
        assert quality[0]["n_picks"] == 1
        assert quality[0]["station_coverage"] == pytest.approx(1.0)

    def test_associate_picks_populates_cluster_quality(self):
        """associate_picks should populate cluster_quality on the result."""
        base = UTCDateTime("2024-01-01T12:00:00")
        station_picks = {
            "A": [base + i for i in range(5)],
            "B": [base + i for i in range(5)],
        }
        station_list = ["A", "B"]
        location_dict = {"A": [0.0, 0.0, 0.0], "B": [1.0, 1.0, 1.0]}

        result = associate_picks(
            station_picks,
            station_list,
            location_dict,
            dbscan_eps=2.0,
            dbscan_min_samples=2,
            min_picks_per_cluster=2,
            temporal_threshold=10.0,
            spatial_threshold=10.0,
        )

        assert isinstance(result, AssociationResult)
        # cluster_quality should be a dict (possibly empty if all noise)
        assert result.cluster_quality is not None
        assert isinstance(result.cluster_quality, dict)


class TestAssociationResult:
    """Tests for the AssociationResult NamedTuple."""

    def _make_result(self):
        return AssociationResult(
            station_distance={"A": 0, "B": 1},
            reverse_station_distance={0: "A", 1: "B"},
            pick_station_order=[0, 1, 0, 1],
            pick_times=[1.0, 2.0, 3.0, 4.0],
            labels=np.array([0, 0, 1, 1]),
        )

    def test_named_field_access(self):
        """Fields should be accessible by name."""
        result = self._make_result()
        assert result.station_distance == {"A": 0, "B": 1}
        assert result.reverse_station_distance == {0: "A", 1: "B"}
        assert result.pick_station_order == [0, 1, 0, 1]
        assert result.pick_times == [1.0, 2.0, 3.0, 4.0]
        np.testing.assert_array_equal(result.labels, [0, 0, 1, 1])

    def test_tuple_unpacking_still_works(self):
        """Existing tuple-unpacking code must remain compatible."""
        result = self._make_result()
        sd, rsd, pso, pt, labels, _quality = result
        assert sd == {"A": 0, "B": 1}
        assert rsd == {0: "A", 1: "B"}
        assert pso == [0, 1, 0, 1]
        assert pt == [1.0, 2.0, 3.0, 4.0]
        np.testing.assert_array_equal(labels, [0, 0, 1, 1])

    def test_is_tuple_subclass(self):
        """AssociationResult should be a tuple subclass."""
        result = self._make_result()
        assert isinstance(result, tuple)
        assert len(result) == 6

    def test_associate_picks_returns_named_tuple(self):
        """associate_picks should return an AssociationResult."""
        result = associate_picks(
            {"A": [], "B": []},
            ["A", "B"],
            {"A": [0.0, 0.0, 0.0], "B": [1.0, 1.0, 1.0]},
        )
        assert isinstance(result, AssociationResult)
        assert result.station_distance == {"A": 0, "B": 1}
        assert result.labels.size == 0
