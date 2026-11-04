import numpy as np
import pytest

from bevdrive.contracts.detection import Detection3D


def test_detection_serialization_preserves_units_and_frame() -> None:
    detection = Detection3D(
        scene_id="scene-1",
        sample_token="sample-1",
        timestamp_us=1_000_000,
        class_name="pedestrian",
        score=0.9,
        center_xyz_m=(1.0, 2.0, 3.0),
        size_lwh_m=(0.6, 0.6, 1.7),
        yaw_rad=0.2,
        velocity_xy_mps=(0.1, -0.2),
        coordinate_frame="ego_at_timestamp",
        source="test",
    )
    restored = Detection3D.from_mapping(detection.to_dict())
    assert restored == detection


@pytest.mark.parametrize("field", ["center_xyz_m", "size_lwh_m"])
def test_detection_rejects_wrong_vector_shape(field: str) -> None:
    values = {
        "scene_id": "scene-1",
        "sample_token": "sample-1",
        "timestamp_us": 1,
        "class_name": "car",
        "score": 0.5,
        "center_xyz_m": (0.0, 0.0, 0.0),
        "size_lwh_m": (1.0, 1.0, 1.0),
        "yaw_rad": 0.0,
    }
    values[field] = (1.0, 2.0)
    with pytest.raises(ValueError):
        Detection3D(**values)


def test_detection_rejects_nan() -> None:
    with pytest.raises(ValueError):
        Detection3D(
            scene_id="scene-1",
            sample_token="sample-1",
            timestamp_us=1,
            class_name="car",
            score=0.5,
            center_xyz_m=(np.nan, 0.0, 0.0),
            size_lwh_m=(1.0, 1.0, 1.0),
            yaw_rad=0.0,
        )
