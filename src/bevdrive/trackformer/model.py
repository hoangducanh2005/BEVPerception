"""BEVFusion-TrackFormer: UniAD-style query-based 3D MOT on fused camera-LiDAR BEV.

Per frame (online, batch size 1):

  BEVFusion fused BEV (frozen) --+--> BEVAdapter --> value map (UniAD grid)
  Phase-2 aligned history -------+
  detection queries  (new objects): BEVFusion proposals  [+ UniAD learned queries]
  track queries      (existing objects): carried from t-1, reference points
                      propagated with ego motion + predicted velocity
  [det queries ; track queries] --6-layer deformable decoder (self-attn among
  all queries => track/detection competition, cross-attn into BEV)--> class,
  box, velocity, embedding per query
  RuntimeTrackerBase: confident detection queries are born as new ids, tracks
  with low score sleep and die after ``miss_tolerance`` frames
  MemoryBank (embedding history attention) + QIM (query update) -> next frame

Identity is carried by the query itself: there is no Hungarian association at
inference. Hungarian matching is used only in training (ClipMatcher) to assign
new ground-truth objects to detection queries, exactly as in MOTR/UniAD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from bevdrive.temporal.bev_alignment import UNIAD_GRID, sample_at

from .bev_adapter import BEVAdapter
from .instances import Instances
from .layers import TrackDecoder, inverse_sigmoid, make_cls_branch, make_reg_branch
from .query_modules import MemoryBank, QueryInteractionModule, RuntimeTrackerBase


@dataclass
class TrackFormerConfig:
    embed_dims: int = 256
    num_layers: int = 6
    num_classes: int = 10
    pc_range: tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
    bev_in_channels: int = 512
    use_temporal_bev: bool = True
    query_mode: str = "proposal"  # "proposal" | "learned" | "hybrid"
    num_learned_queries: int = 900
    include_sdc_query: bool = False  # UniAD appends one ego (SDC) query that never becomes a track
    ref_z_from_query: bool = True  # UniAD: after propagation, track ref z = reference_points(query_pos)
    reset_ids_per_scene: bool = True  # UniAD keeps counting ids across scenes (False reproduces that)
    max_proposals: int = 200
    proposal_min_score: float = 0.02
    memory_len: int = 4
    score_thresh: float = 0.4
    filter_score_thresh: float = 0.35
    miss_tolerance: int = 5
    qim_update_query_pos: bool = True
    qim_fp_ratio: float = 0.3
    qim_random_drop: float = 0.1
    code_weights: tuple[float, ...] = field(default=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2))


def sine_embed(x: torch.Tensor, num_feats: int = 64, temperature: float = 10000.0) -> torch.Tensor:
    """[N, D] in [0,1] -> [N, D*2*num_feats/2*2] sinusoidal embedding."""

    scale = 2 * math.pi
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    pos = x[..., None] * scale / dim_t
    pos = torch.stack([pos[..., 0::2].sin(), pos[..., 1::2].cos()], dim=-1).flatten(-2)
    return pos.flatten(-2)


def boxes_to_code(boxes: torch.Tensor) -> torch.Tensor:
    """mmdet3d 9-dof (x,y,z_b,dx,dy,dz,yaw,vx,vy) -> UniAD code (cx,cy,log dx,log dy,cz,log dz,sin,cos,vx,vy)."""

    dims = boxes[:, 3:6].clamp_min(1e-3).log()
    return torch.cat([boxes[:, 0:2], dims[:, 0:2], boxes[:, 2:3], dims[:, 2:3], boxes[:, 6:7].sin(), boxes[:, 6:7].cos(), boxes[:, 7:9]], dim=1)


def code_to_boxes(code: torch.Tensor) -> torch.Tensor:
    yaw = torch.atan2(code[:, 6:7], code[:, 7:8])
    dims = code[:, [2, 3, 5]].exp()
    return torch.cat([code[:, 0:2], code[:, 4:5], dims, yaw, code[:, 8:10]], dim=1)


class BEVFusionTrackFormer(nn.Module):
    def __init__(self, cfg: TrackFormerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or TrackFormerConfig()
        c = cfg.embed_dims
        self.bev_adapter = BEVAdapter(cfg.bev_in_channels, cfg.use_temporal_bev, c)
        if cfg.query_mode in ("learned", "hybrid"):
            self.query_embedding = nn.Embedding(cfg.num_learned_queries + int(cfg.include_sdc_query), 2 * c)
        self.reference_points = nn.Linear(c, 3)
        if cfg.query_mode in ("proposal", "hybrid"):
            self.proposal_pos = nn.Sequential(nn.Linear(3 * 128, c), nn.ReLU(inplace=True), nn.Linear(c, c))
            self.proposal_feat = nn.Sequential(nn.Linear(c + cfg.num_classes + 1 + 7, c), nn.LayerNorm(c), nn.ReLU(inplace=True), nn.Linear(c, c), nn.LayerNorm(c))
        self.decoder = TrackDecoder(cfg.num_layers, c)
        self.cls_branches = nn.ModuleList([make_cls_branch(c, cfg.num_classes) for _ in range(cfg.num_layers)])
        self.reg_branches = nn.ModuleList([make_reg_branch(c, 10) for _ in range(cfg.num_layers)])
        bias = -math.log((1 - 0.01) / 0.01)
        for branch in self.cls_branches:
            nn.init.constant_(branch[-1].bias, bias)
        self.query_interact = QueryInteractionModule(c, 0.0, cfg.qim_update_query_pos, cfg.qim_fp_ratio, cfg.qim_random_drop)
        self.memory_bank = MemoryBank(c, cfg.memory_len)
        self.track_base = RuntimeTrackerBase(cfg.score_thresh, cfg.filter_score_thresh, cfg.miss_tolerance)
        self.pc = torch.tensor(cfg.pc_range, dtype=torch.float32)
        self._state: dict | None = None

    # ------------------------------------------------------------------ queries
    def _norm_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        pc = self.pc.to(xyz.device)
        return (xyz - pc[:3]) / (pc[3:] - pc[:3])

    def _denorm_xyz(self, n: torch.Tensor) -> torch.Tensor:
        pc = self.pc.to(n.device)
        return n * (pc[3:] - pc[:3]) + pc[:3]

    def _blank(self, query: torch.Tensor, ref_pts: torch.Tensor) -> Instances:
        n, dev, c = query.shape[0], query.device, self.cfg.embed_dims
        return Instances(
            query=query,
            ref_pts=ref_pts,
            output_embedding=torch.zeros(n, c, device=dev),
            obj_idxes=torch.full((n,), -1, dtype=torch.long, device=dev),
            matched_gt_idxes=torch.full((n,), -1, dtype=torch.long, device=dev),
            disappear_time=torch.zeros(n, dtype=torch.long, device=dev),
            iou=torch.zeros(n, device=dev),
            scores=torch.zeros(n, device=dev),
            pred_boxes=torch.zeros(n, 10, device=dev),
            pred_logits=torch.zeros(n, self.cfg.num_classes, device=dev),
            mem_bank=torch.zeros(n, self.memory_bank.max_his_length, c, device=dev),
            mem_padding_mask=torch.ones(n, self.memory_bank.max_his_length, dtype=torch.bool, device=dev),
            save_period=torch.zeros(n, device=dev),
            is_det_query=torch.ones(n, dtype=torch.bool, device=dev),  # True only in the frame the query is created
        )

    def detection_queries(self, bev_map: torch.Tensor, proposals: dict | None) -> Instances:
        parts = []
        dev = bev_map.device
        if self.cfg.query_mode in ("learned", "hybrid"):
            q = self.query_embedding.weight
            parts.append(self._blank(q, self.reference_points(q[:, : self.cfg.embed_dims])))
        if self.cfg.query_mode in ("proposal", "hybrid") and proposals is not None:
            boxes = torch.as_tensor(proposals["boxes"], dtype=torch.float32, device=dev).reshape(-1, 9)
            scores = torch.as_tensor(proposals["scores"], dtype=torch.float32, device=dev)
            labels = torch.as_tensor(proposals["labels"], dtype=torch.long, device=dev)
            pc = self.pc.to(dev)
            keep = (scores >= self.cfg.proposal_min_score) & (boxes[:, 0].abs() < pc[3]) & (boxes[:, 1].abs() < pc[4])
            order = torch.argsort(scores[keep], descending=True)[: self.cfg.max_proposals]
            boxes, scores, labels = boxes[keep][order], scores[keep][order], labels[keep][order]
            if len(boxes):
                norm = self._norm_xyz(boxes[:, :3]).clamp(0.0, 1.0)
                pos = self.proposal_pos(sine_embed(norm, 128))
                sampled = sample_at(bev_map, UNIAD_GRID, boxes[:, :2])[0].T
                code = boxes_to_code(boxes)
                extra = torch.cat([code[:, 2:4], code[:, 5:10]], dim=1)  # log l, log w, log h, sin, cos, vx, vy
                onehot = torch.nn.functional.one_hot(labels, self.cfg.num_classes).float()
                feat = self.proposal_feat(torch.cat([sampled, onehot, scores[:, None], extra], dim=1))
                parts.append(self._blank(torch.cat([pos, feat], dim=1), inverse_sigmoid(norm)))
        if not parts:
            return self._blank(torch.zeros(0, 2 * self.cfg.embed_dims, device=dev), torch.zeros(0, 3, device=dev))
        return Instances.cat(parts)

    # ------------------------------------------------------------------ decode
    def decode(self, inst: Instances, value: torch.Tensor) -> dict[str, torch.Tensor]:
        c = self.cfg.embed_dims
        query_pos, query = inst.query[:, :c], inst.query[:, c:]
        ref0 = inst.ref_pts.sigmoid()
        states, refs = self.decoder(query[:, None], query_pos[:, None], value, ref0[None], UNIAD_GRID.shape, self.reg_branches)
        pc = self.pc.to(value.device)
        all_cls, all_box = [], []
        for lvl in range(states.shape[0]):
            hs = states[lvl, :, 0]
            reference = inverse_sigmoid(ref0 if lvl == 0 else refs[lvl - 1, 0])
            tmp = self.reg_branches[lvl](hs)
            xy = (tmp[:, 0:2] + reference[:, 0:2]).sigmoid()
            z = (tmp[:, 4:5] + reference[:, 2:3]).sigmoid()
            last_ref = torch.cat([xy, z], dim=1)
            box = torch.cat([xy * (pc[3:5] - pc[0:2]) + pc[0:2], tmp[:, 2:4], z * (pc[5] - pc[2]) + pc[2], tmp[:, 5:10]], dim=1)
            all_cls.append(self.cls_branches[lvl](hs))
            all_box.append(box)
        return {"cls": torch.stack(all_cls), "box": torch.stack(all_box), "hs": states[:, :, 0], "last_ref": inverse_sigmoid(last_ref)}

    def velo_update(self, ref_pts: torch.Tensor, velocity: torch.Tensor, prev_l2g: np.ndarray, cur_l2g: np.ndarray, dt: float) -> torch.Tensor:
        """Propagate reference points (logit space) to the current LiDAR frame: ego motion + v * dt."""

        if len(ref_pts) == 0:
            return ref_pts
        xyz = self._denorm_xyz(ref_pts.sigmoid())
        xyz = torch.cat([xyz[:, :2] + velocity * float(dt), xyz[:, 2:3]], dim=1)
        t = torch.as_tensor(np.linalg.inv(cur_l2g) @ prev_l2g, dtype=torch.float32, device=ref_pts.device)
        xyz = xyz @ t[:3, :3].T + t[:3, 3]
        return inverse_sigmoid(self._norm_xyz(xyz))

    def _propagate(self, tracks: Instances, prev_l2g: np.ndarray, cur_l2g: np.ndarray, dt: float, detach_velocity: bool) -> None:
        velocity = tracks.pred_boxes[:, 8:10].detach() if detach_velocity else tracks.pred_boxes[:, 8:10]
        ref = self.velo_update(tracks.ref_pts, velocity, prev_l2g, cur_l2g, dt)
        if self.cfg.ref_z_from_query:
            z = self.reference_points(tracks.query[:, : self.cfg.embed_dims])[:, 2:3]
            ref = torch.cat([ref[:, :2], z], dim=1)
        tracks.ref_pts = ref

    def _mark_sdc(self, inst: Instances) -> None:
        if self.cfg.include_sdc_query and self.cfg.query_mode in ("learned", "hybrid"):
            inst.obj_idxes[self.cfg.num_learned_queries] = -2

    def _apply_outputs(self, inst: Instances, out: dict[str, torch.Tensor]) -> None:
        inst.pred_logits = out["cls"][-1]
        inst.pred_boxes = out["box"][-1]
        inst.output_embedding = out["hs"][-1]
        inst.ref_pts = out["last_ref"]
        inst.scores = out["cls"][-1].sigmoid().max(dim=-1).values.detach()

    # ------------------------------------------------------------------ inference
    def reset(self) -> None:
        self._state = None
        self.track_base.clear()

    @torch.no_grad()
    def forward_inference(
        self,
        bev: torch.Tensor | None,
        history: torch.Tensor | None,
        proposals: dict | None,
        lidar2global: np.ndarray,
        timestamp: float,
        scene: str,
        bev_value: torch.Tensor | None = None,
    ) -> dict:
        """One online step. ``bev_value`` ([H*W, 1, 256] on the UniAD grid) bypasses the adapter (port check)."""

        if self._state is None or self._state["scene"] != scene:
            self._state = None
            if self.cfg.reset_ids_per_scene:
                self.track_base.clear()
            tracks = None
        else:
            tracks = self._state["tracks"]
            if len(tracks):
                self._propagate(tracks, self._state["l2g"], lidar2global, timestamp - self._state["timestamp"], detach_velocity=True)
        if bev_value is not None:
            value = bev_value
            bev_map = value.permute(1, 2, 0).reshape(1, -1, *UNIAD_GRID.shape)
        else:
            value, bev_map = self.bev_adapter(bev, history)
        inst = self.detection_queries(bev_map, proposals)
        if tracks is not None and len(tracks):
            tracks.is_det_query = torch.zeros(len(tracks), dtype=torch.bool, device=tracks.query.device)
            inst = Instances.cat([inst, tracks])
        out = self.decode(inst, value)
        self._apply_outputs(inst, out)
        self._mark_sdc(inst)
        self.track_base.update(inst)
        active = (inst.obj_idxes >= 0) & (inst.scores >= self.cfg.filter_score_thresh)
        result = {
            "boxes": code_to_boxes(inst.pred_boxes[active]).cpu().numpy().astype(np.float64),
            "scores": inst.scores[active].cpu().numpy().astype(np.float64),
            "labels": inst.pred_logits[active].argmax(-1).cpu().numpy(),
            "ids": inst.obj_idxes[active].cpu().numpy(),
            "embeddings": inst.output_embedding[active].cpu().numpy().astype(np.float32),
            "from_track_query": (~inst.is_det_query[active]).cpu().numpy(),
        }
        det_scores = inst.pred_logits.sigmoid().max(-1).values
        top = torch.argsort(det_scores, descending=True)[:300]
        result["det_boxes"] = code_to_boxes(inst.pred_boxes[top]).cpu().numpy().astype(np.float64)
        result["det_scores"] = det_scores[top].cpu().numpy().astype(np.float64)
        result["det_labels"] = inst.pred_logits[top].argmax(-1).cpu().numpy()
        inst = self.memory_bank(inst)
        tracks = self.query_interact(inst)
        self._state = {"tracks": tracks, "timestamp": timestamp, "l2g": np.asarray(lidar2global), "scene": scene}
        return result

    # ------------------------------------------------------------------ training
    def forward_train_clip(self, frames: list[dict], criterion) -> dict[str, torch.Tensor]:
        """MOTR/UniAD clip training. ``frames``: dicts with bev, history, proposals, l2g, timestamp."""

        criterion.initialize_for_single_clip([f["gt"] for f in frames])
        tracks: Instances | None = None
        prev = None
        for i, frame in enumerate(frames):
            if tracks is not None and len(tracks):
                self._propagate(tracks, prev["l2g"], frame["l2g"], frame["timestamp"] - prev["timestamp"], detach_velocity=True)
            value, bev_map = self.bev_adapter(frame["bev"], frame.get("history"))
            inst = self.detection_queries(bev_map, frame.get("proposals"))
            if tracks is not None and len(tracks):
                tracks.is_det_query = torch.zeros(len(tracks), dtype=torch.bool, device=tracks.query.device)
                inst = Instances.cat([inst, tracks])
            out = self.decode(inst, value)
            self._apply_outputs(inst, out)
            self._mark_sdc(inst)
            criterion.match_for_single_frame(inst, out, i)
            inst = self.memory_bank(inst)
            tracks = self.query_interact(inst)
            prev = frame
        return criterion.losses()
