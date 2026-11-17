"""ClipMatcher (MOTR / UniAD ``losses/track_loss.py``) in plain PyTorch.

For every frame of a training clip and every decoder layer:

1. *Track queries* (``obj_idxes >= 0``) keep their identity: their target is
   the GT box with the same instance id, or background if it disappeared.
2. Remaining GT objects are Hungarian-matched to *new* queries
   (``obj_idxes == -1``) with cost ``2 * focal_cls + 0.25 * L1(code[:8])``.
3. Loss = 2.0 * sigmoid focal (all queries) + 0.25 * L1 (matched, code
   weights ``[1]*8 + [0.2, 0.2]``), averaged over matched targets.

Only the last layer's matching updates the persistent ``obj_idxes`` (the
identity that will be carried to the next frame). ``iou`` is set to 1 for
matched queries whose centre lies within ``active_dist_m`` of the GT; QIM uses
``iou > 0.5`` to decide which tracks survive during training. (UniAD uses 3D
IoU > 0.5 here; the centre-distance proxy is a documented deviation that keeps
small VRU boxes from being dropped early in training.)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .instances import Instances


def focal_cost(logits: torch.Tensor, labels: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-12) -> torch.Tensor:
    p = logits.sigmoid()
    neg = -(1 - p + eps).log() * (1 - alpha) * p.pow(gamma)
    pos = -(p + eps).log() * alpha * (1 - p).pow(gamma)
    return pos[:, labels] - neg[:, labels]


def sigmoid_focal_loss(logits: torch.Tensor, target_onehot: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target_onehot, reduction="none")
    p_t = p * target_onehot + (1 - p) * (1 - target_onehot)
    loss = ce * (1 - p_t).pow(gamma)
    alpha_t = alpha * target_onehot + (1 - alpha) * (1 - target_onehot)
    return (alpha_t * loss).sum()


class ClipMatcher:
    def __init__(
        self,
        num_classes: int = 10,
        code_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2),
        cls_weight: float = 2.0,
        bbox_weight: float = 0.25,
        match_cls_weight: float = 2.0,
        match_reg_weight: float = 0.25,
        active_dist_m: float = 1.0,
    ) -> None:
        self.num_classes = num_classes
        self.code_weights = torch.tensor(code_weights)
        self.cls_weight, self.bbox_weight = cls_weight, bbox_weight
        self.match_cls_weight, self.match_reg_weight = match_cls_weight, match_reg_weight
        self.active_dist_m = active_dist_m
        self.gts: list[dict] = []
        self._losses: list[dict[str, torch.Tensor]] = []
        self.stats: dict[str, float] = {}

    def initialize_for_single_clip(self, gts: list[dict]) -> None:
        self.gts = gts
        self._losses = []
        self.stats = {"matched_new": 0, "matched_track": 0, "disappeared": 0, "gt": 0}

    def _assign(self, inst: Instances, cls: torch.Tensor, box: torch.Tensor, gt: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Returns (src_idx, tgt_idx) with tgt=-1 for disappeared tracks, and #disappeared."""

        dev = cls.device
        obj_ids = gt["obj_ids"].tolist()
        id_to_gt = {oid: k for k, oid in enumerate(obj_ids)}
        tracked = (inst.obj_idxes >= 0).nonzero().flatten()
        tgt_tracked = torch.tensor([id_to_gt.get(int(inst.obj_idxes[j]), -1) for j in tracked.tolist()], dtype=torch.long, device=dev)
        disappeared = int((tgt_tracked == -1).sum())
        taken = set(tgt_tracked[tgt_tracked >= 0].tolist())
        untracked = torch.tensor([k for k in range(len(obj_ids)) if k not in taken], dtype=torch.long, device=dev)
        new_q = (inst.obj_idxes == -1).nonzero().flatten()
        src_new = torch.zeros(0, dtype=torch.long, device=dev)
        tgt_new = torch.zeros(0, dtype=torch.long, device=dev)
        if len(untracked) and len(new_q):
            with torch.no_grad():
                cost = self.match_cls_weight * focal_cost(cls[new_q].float(), gt["labels"][untracked])
                cost = cost + self.match_reg_weight * torch.cdist(box[new_q, :8].float(), gt["code"][untracked, :8], p=1)
            rows, cols = linear_sum_assignment(np.nan_to_num(cost.cpu().numpy(), nan=1e6, posinf=1e6))
            src_new = new_q[torch.as_tensor(rows, dtype=torch.long, device=dev)]
            tgt_new = untracked[torch.as_tensor(cols, dtype=torch.long, device=dev)]
        return torch.cat([src_new, tracked]), torch.cat([tgt_new, tgt_tracked]), disappeared

    def _loss(self, cls: torch.Tensor, box: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor, gt: dict) -> dict[str, torch.Tensor]:
        target = torch.zeros_like(cls)
        pos = tgt >= 0
        if pos.any():
            target[src[pos], gt["labels"][tgt[pos]]] = 1.0
        avg = max(float(len(src)), 1.0)
        loss_cls = sigmoid_focal_loss(cls.float(), target) / avg
        if pos.any():
            weights = self.code_weights.to(box.device)
            target_code = gt["code"][tgt[pos]]
            valid_v = gt["vel_valid"][tgt[pos]].float()[:, None]
            w = weights.expand_as(target_code).clone()
            w[:, 8:10] = w[:, 8:10] * valid_v
            loss_box = (torch.abs(box[src[pos]].float() - target_code) * w).sum() / max(float(pos.sum()), 1.0)
        else:
            loss_box = box.sum() * 0.0
        return {"cls": self.cls_weight * loss_cls, "box": self.bbox_weight * loss_box}

    def match_for_single_frame(self, inst: Instances, out: dict[str, torch.Tensor], frame_idx: int) -> None:
        gt = self.gts[frame_idx]
        n_layers = out["cls"].shape[0]
        frame_losses: dict[str, torch.Tensor] = {}
        base_obj = inst.obj_idxes.clone()
        for lvl in range(n_layers):
            probe = Instances(obj_idxes=base_obj.clone())
            src, tgt, disappeared = self._assign(probe, out["cls"][lvl], out["box"][lvl], gt)
            losses = self._loss(out["cls"][lvl], out["box"][lvl], src, tgt, gt)
            for key, value in losses.items():
                frame_losses[f"{key}_d{lvl}"] = value
            if lvl == n_layers - 1:
                obj_idxes = base_obj.clone()
                matched = torch.full_like(base_obj, -1)
                pos = tgt >= 0
                obj_idxes[src[pos]] = gt["obj_ids"][tgt[pos]]
                matched[src] = tgt
                inst.obj_idxes = obj_idxes
                inst.matched_gt_idxes = matched
                iou = torch.zeros_like(inst.iou)
                if pos.any():
                    dist = torch.linalg.norm(out["box"][lvl][src[pos], :2].detach() - gt["code"][tgt[pos], :2], dim=1)
                    iou[src[pos]] = (dist < self.active_dist_m).float()
                inst.iou = iou
                self.stats["matched_new"] += int((pos & (base_obj[src] == -1)).sum())
                self.stats["matched_track"] += int((pos & (base_obj[src] >= 0)).sum())
                self.stats["disappeared"] += disappeared
                self.stats["gt"] += len(gt["labels"])
        self._losses.append(frame_losses)

    def losses(self) -> dict[str, torch.Tensor]:
        total: dict[str, torch.Tensor] = {}
        for frame in self._losses:
            for key, value in frame.items():
                total[key] = total.get(key, 0.0) + value / len(self._losses)
        total["loss"] = sum(total.values())
        return total
