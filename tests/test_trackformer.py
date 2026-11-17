"""Smoke/unit tests for the query-based tracker (random BEV, untrained or UniAD-initialised weights).

These check shapes, coordinate maths, lifecycle and gradient flow. They are
*not* evidence of tracking quality.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from bevdrive.trackformer import BEVFusionTrackFormer, ClipMatcher, Instances, TrackFormerConfig, boxes_to_code, code_to_boxes, load_uniad_track_weights
from bevdrive.trackformer.layers import deformable_attn_single_level
from bevdrive.trackformer.query_modules import RuntimeTrackerBase


def _pose(x=0.0, y=0.0, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    t = np.eye(4)
    t[:2, :2] = [[c, -s], [s, c]]
    t[:2, 3] = [x, y]
    return t


def _proposals(n=5):
    boxes = np.zeros((n, 9))
    boxes[:, 0] = np.linspace(-20, 20, n)
    boxes[:, 1] = 5.0
    boxes[:, 2] = -1.5
    boxes[:, 3:6] = [0.7, 0.6, 1.7]
    return {"boxes": boxes, "scores": np.linspace(0.9, 0.5, n), "labels": np.full(n, 8)}


def test_box_code_roundtrip():
    boxes = torch.tensor([[1.0, -2.0, -1.0, 4.5, 1.9, 1.6, 0.7, 1.0, -0.5]])
    assert torch.allclose(code_to_boxes(boxes_to_code(boxes)), boxes, atol=1e-5)


def test_deformable_attention_matches_mmcv_pytorch_reference():
    mmcv_ops = pytest.importorskip("mmcv.ops.multi_scale_deform_attn")
    torch.manual_seed(0)
    bs, h, w, heads, dh, nq, pts = 1, 7, 9, 2, 4, 5, 3
    value = torch.randn(bs, h * w, heads, dh)
    loc = torch.rand(bs, nq, heads, pts, 2)
    attn = torch.rand(bs, nq, heads, pts).softmax(-1)
    ours = deformable_attn_single_level(value, (h, w), loc, attn)
    ref = mmcv_ops.multi_scale_deformable_attn_pytorch(value, torch.tensor([[h, w]]), loc[:, :, :, None], attn[:, :, :, None])
    assert torch.allclose(ours, ref, atol=1e-5)


def test_velo_update_moves_reference_points_with_ego_and_velocity():
    model = BEVFusionTrackFormer(TrackFormerConfig(query_mode="proposal"))
    xyz = torch.tensor([[10.0, 5.0, -1.0]])
    ref = torch.logit(model._norm_xyz(xyz))
    out = model.velo_update(ref, torch.tensor([[2.0, 0.0]]), _pose(), _pose(3.0, 0.0), 0.5)
    got = model._denorm_xyz(out.sigmoid())
    assert torch.allclose(got, torch.tensor([[10.0 + 1.0 - 3.0, 5.0, -1.0]]), atol=1e-3)


def test_runtime_tracker_lifecycle():
    base = RuntimeTrackerBase(score_thresh=0.4, filter_score_thresh=0.35, miss_tolerance=2)
    inst = Instances(obj_idxes=torch.tensor([-1, -1, 3]), scores=torch.tensor([0.9, 0.2, 0.1]), disappear_time=torch.tensor([0, 0, 1]))
    base.max_obj_id = 4
    base.update(inst)
    assert inst.obj_idxes.tolist() == [4, -1, -1]  # new id born; low-score track 3 killed at miss_tolerance


def test_inference_keeps_identity_and_trains_end_to_end():
    torch.manual_seed(0)
    model = BEVFusionTrackFormer(TrackFormerConfig(query_mode="proposal", score_thresh=0.0, filter_score_thresh=0.0))
    model.eval()
    bev = torch.randn(1, 512, 180, 180)
    r1 = model.forward_inference(bev, None, _proposals(), _pose(), 0.0, "s")
    assert len(r1["ids"]) > 0 and r1["boxes"].shape[1] == 9
    r2 = model.forward_inference(bev, bev, _proposals(), _pose(0.5), 0.5, "s")
    assert set(r1["ids"]).issubset(set(r2["ids"]))  # track queries persist
    assert r2["from_track_query"].any()
    # training: gradients reach adapter, decoder and QIM through a 2-frame clip
    model.train()
    gt = {"code": boxes_to_code(torch.tensor(_proposals()["boxes"], dtype=torch.float32)), "labels": torch.full((5,), 8), "obj_ids": torch.arange(5), "vel_valid": torch.ones(5, dtype=torch.bool)}
    frames = [{"bev": bev, "history": None, "proposals": _proposals(), "l2g": _pose(), "timestamp": 0.0, "gt": gt}, {"bev": bev, "history": bev, "proposals": _proposals(), "l2g": _pose(0.5), "timestamp": 0.5, "gt": gt}]
    losses = model.forward_train_clip(frames, ClipMatcher(active_dist_m=1e3))  # untrained: force survival so QIM is exercised
    losses["loss"].backward()
    assert model.bev_adapter.proj[1].weight.grad is not None
    assert model.decoder.layers[0].attentions[1].sampling_offsets.weight.grad is not None
    assert model.query_interact.linear_feat1.weight.grad is not None


@pytest.mark.skipif(not Path("/home/anhhd/GIT/UniAD/ckpts/uniad_base_track_map.pth").exists(), reason="UniAD checkpoint not available")
def test_uniad_weights_load_into_compatible_modules():
    model = BEVFusionTrackFormer(TrackFormerConfig(query_mode="hybrid"))
    report = load_uniad_track_weights(model)
    assert not report["shape_mismatch"]
    assert any(k.startswith("decoder.layers.5.attentions.1") for k in report["loaded"])
    assert "query_embedding.weight" in report["loaded"]
    assert all(k.startswith(("bev_adapter.proj", "proposal_")) for k in report["not_initialised"])
