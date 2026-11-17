import numpy as np

from bevdrive.tracking import DetectionMeasurement, QueryAssociationTrackManager


def _detection(x: float) -> DetectionMeasurement:
    return DetectionMeasurement("pedestrian", 0.9, (x, 0.0, 0.0), (0.6, 0.6, 1.7), 0.0)


def test_query_association_manager_preserves_track_for_matching_embedding() -> None:
    tracker = QueryAssociationTrackManager(max_distance_m=3.0, min_score=0.2)
    first = tracker.update_with_embeddings([_detection(0.0)], [np.array([1.0, 0.0])], timestamp_us=1_000_000, scene_id="scene")
    second = tracker.update_with_embeddings([_detection(0.2)], [np.array([1.0, 0.0])], timestamp_us=1_100_000, scene_id="scene")
    assert first[0].track_id == second[0].track_id


def test_query_association_manager_resets_scene_state() -> None:
    tracker = QueryAssociationTrackManager(min_score=0.2)
    tracker.update_with_embeddings([_detection(0.0)], [np.array([1.0, 0.0])], timestamp_us=1_000_000, scene_id="a")
    next_scene = tracker.update_with_embeddings([_detection(0.0)], [np.array([1.0, 0.0])], timestamp_us=1_000_000, scene_id="b")
    assert next_scene[0].track_id == 1
