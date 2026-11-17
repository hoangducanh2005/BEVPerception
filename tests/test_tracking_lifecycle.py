import numpy as np

from bevdrive.tracking.track_manager import DetectionMeasurement, TrackManager


def _measurement(x: float, *, score: float = 0.9) -> DetectionMeasurement:
    return DetectionMeasurement(
        class_name="pedestrian",
        score=score,
        center_world_xyz=(x, 0.0, 0.0),
        size_lwh=(0.6, 0.6, 1.7),
        yaw_rad=0.0,
    )


def test_scene_reset_clears_tracks_and_restarts_ids() -> None:
    tracker = TrackManager(max_distance_m=3.0, max_missed_frames=3, min_score=0.2)
    first = tracker.update([_measurement(0.0)], timestamp_us=1_000_000, scene_id="scene-a")
    assert first[0].track_id == 1
    assert tracker.active_track_count == 1

    second_scene = tracker.update([_measurement(0.0)], timestamp_us=1_000_000, scene_id="scene-b")
    assert second_scene[0].track_id == 1
    assert tracker.scene_id == "scene-b"
    assert tracker.active_track_count == 1


def test_timestamp_delta_is_used_by_constant_velocity_prediction() -> None:
    tracker = TrackManager(max_distance_m=3.0, max_missed_frames=3, min_score=0.2)
    tracker.update([_measurement(0.0)], timestamp_us=1_000_000, scene_id="scene-a")
    tracker.update([_measurement(1.0)], timestamp_us=1_100_000, scene_id="scene-a")
    assignment = tracker.update([], timestamp_us=1_600_000, scene_id="scene-a")
    assert assignment == []
    assert np.isfinite(tracker._tracks[1].filter.state).all()
    assert tracker._tracks[1].age == 3
