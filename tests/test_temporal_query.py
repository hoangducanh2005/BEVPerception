import numpy as np
import torch

from bevdrive.tracking.temporal_query import TemporalBEVMemory, TemporalQueryBank


def test_temporal_bev_identity_pose_preserves_feature() -> None:
    feature = torch.randn(1, 8, 20, 20)
    pose = np.eye(4, dtype=np.float64)
    memory = TemporalBEVMemory()
    memory.update(feature, pose)
    aligned, stats = memory.update(feature, pose)
    assert aligned is not None
    assert stats.has_previous is True
    assert stats.valid_fraction == 1.0
    assert stats.mean_abs_delta is not None and stats.mean_abs_delta < 1e-5
    assert stats.cosine_similarity is not None and stats.cosine_similarity > 0.999


def test_temporal_query_bank_persists_track_embedding() -> None:
    feature = torch.ones(1, 4, 20, 20)
    pose = np.eye(4, dtype=np.float64)

    class Assignment:
        track_id = 7
        center_world_xyz = (0.0, 0.0, 0.0)

    bank = TemporalQueryBank(ema=0.7)
    first = bank.update([Assignment()], feature, pose)
    second = bank.update([Assignment()], feature, pose)
    assert first["num_queries"] == 1
    assert second["num_queries"] == 1
    assert second["num_queries_updated"] == 1
    assert second["mean_query_cosine"] is not None and second["mean_query_cosine"] > 0.999
