import numpy as np
import torch

from bevdrive.temporal.bev_alignment import BEVGrid, aggregate, resample, sample_at, warp_to_current

GRID = BEVGrid(-10.0, 10.0, -10.0, 10.0, 20, 20, "xy")  # 1 m cells, tensor[ix, iy]
GRID_YX = BEVGrid(-10.0, 10.0, -10.0, 10.0, 20, 20, "yx")


def _pose(x=0.0, y=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    t = np.eye(4)
    t[:2, :2] = [[c, -s], [s, c]]
    t[:2, 3] = [x, y]
    return t


def _blob(grid, xy):
    feat = torch.zeros(1, 1, *grid.shape)
    i, j = grid.index_of(np.array([xy]))
    feat[0, 0, i[0], j[0]] = 1.0
    return feat


def test_identity_resample_is_exact():
    feat = torch.randn(1, 3, *GRID.shape)
    out, valid = resample(feat, GRID, GRID, np.eye(4))
    assert valid.all()
    assert torch.allclose(out, feat, atol=1e-5)


def test_layout_conversion_is_transpose():
    feat = torch.randn(1, 2, *GRID.shape)
    out, _ = resample(feat, GRID, GRID_YX)
    assert torch.allclose(out, feat.transpose(-1, -2), atol=1e-5)


def test_cell_centers_and_sampling_agree():
    feat = _blob(GRID, (3.5, -6.5))
    assert abs(float(sample_at(feat, GRID, torch.tensor([[3.5, -6.5]]))[0, 0, 0]) - 1.0) < 1e-5
    assert abs(float(sample_at(feat, GRID, torch.tensor([[-6.5, 3.5]]))[0, 0, 0])) < 1e-5


def test_static_point_lands_at_expected_cell_after_ego_motion():
    # Ego moves +3 m in x and rotates 90 deg; a static world point must be re-expressed exactly.
    prev_pose, cur_pose = _pose(), _pose(3.0, 0.0, np.pi / 2)
    world = np.array([5.5, 2.5, 0.0, 1.0])
    prev_xy = (np.linalg.inv(prev_pose) @ world)[:2]
    cur_xy = (np.linalg.inv(cur_pose) @ world)[:2]
    warped, valid = warp_to_current(_blob(GRID, prev_xy), prev_pose, cur_pose, GRID)
    i, j = GRID.index_of(cur_xy[None])
    assert float(warped[0, 0, i[0], j[0]]) > 0.99
    assert float(warped.sum()) > 0.99
    assert valid[i[0], j[0]]


def test_gated_aggregation_suppresses_changed_cells():
    cur = torch.zeros(1, 4, *GRID.shape)
    cur[:, 0] = 1.0
    static_prev = cur.clone()
    moved_prev = cur.clone()
    moved_prev[:, :, 5, 5] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # content changed in one cell
    valid = torch.ones(GRID.shape, dtype=torch.bool)
    _, diag = aggregate(cur, [(moved_prev, valid, 0.5)], mode="gated")
    gate = diag["gate_newest"][0, 0]
    assert gate[5, 5] < 0.01 and gate[0, 0] > 0.97
    out, _ = aggregate(cur, [(static_prev, valid, 0.5)], mode="gated")
    assert torch.allclose(out, cur, atol=1e-5)
