import numpy as np


def test_lidar2sensor_encoding_matches_mmdet3d_loader_formula():
    """mmdet3d LoadPointsFromMultiSweeps applies p @ M[:3,:3] - M[:3,3]; we encode M to equal R p + t."""

    rng = np.random.default_rng(0)
    yaw = 0.3
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    translation = np.array([4.0, -1.5, 0.2])
    m = np.eye(4)
    m[:3, :3] = rotation.T
    m[:3, 3] = -translation
    points = rng.normal(size=(50, 3))
    loader = points @ m[:3, :3] - m[:3, 3]
    expected = points @ rotation.T + translation
    assert np.allclose(loader, expected)
