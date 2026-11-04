import numpy as np

from bevdrive.geometry.projection import project_points
from bevdrive.geometry.transforms import invert_transform, transform_points


def test_transform_inverse_round_trip() -> None:
    transform = np.array(
        [[0.0, -1.0, 0.0, 2.0], [1.0, 0.0, 0.0, -1.0], [0.0, 0.0, 1.0, 0.5], [0, 0, 0, 1]],
        dtype=float,
    )
    points = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 1.0]])
    round_trip = transform_points(transform_points(points, transform), invert_transform(transform))
    np.testing.assert_allclose(round_trip, points, atol=1e-8)


def test_projection_depth_and_image_bounds() -> None:
    intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    points = np.array([[0.0, 0.0, 10.0], [6.0, 0.0, 10.0], [0.0, 0.0, -1.0]])
    result = project_points(points, intrinsic, image_size=(100, 80))
    np.testing.assert_allclose(result.uv[0], [50.0, 40.0])
    assert result.valid.tolist() == [True, False, False]


def test_invalid_transform_is_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        invert_transform(np.eye(3))
