"""Frozen BEVFusion (camera + LiDAR) inference with fused-BEV capture.

The model, config and checkpoint are used unchanged. The runner only

* feeds the official mmdet3d test pipeline with correct per-frame inputs,
* captures the fused BEV map that the TransFusion head consumes
  (``pts_neck`` output, ``[1, 512, 180, 180]``), and
* can decode the same frozen head on a *supplied* BEV map (used by the
  temporal-BEV ablation in Phase 2).

BEVFusion BEV layout (verified in Phase 2): tensor dim -2 indexes LiDAR **x**,
dim -1 indexes LiDAR **y**, cell size 0.6 m over [-54, 54] m.

Box convention: the frozen checkpoint (``bevfusion_demo_mmdet3d.pth``) regresses
boxes in the *legacy* mmdet3d-0.x / MIT-BEVFusion parametrisation
(``dims = (w, l, h)``, ``yaw_old``). Verified against nuScenes GT in Phase 1
(reports/phase_01_detection.md): matched cars have pred ``(dx, dy) = (w, l)`` and
``-yaw_pred - pi/2 == yaw_gt``. ``legacy_box_convention=True`` converts outputs
to the mmdet3d>=1.0 convention (``dx = l, dy = w, yaw = -yaw_old - pi/2``); the
network itself is untouched.
"""

from __future__ import annotations

import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MMDET3D_ROOT = Path("/home/anhhd/GIT/mmdetection3d")
DEFAULT_CONFIG = DEFAULT_MMDET3D_ROOT / "projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
DEFAULT_CHECKPOINT = Path("/media/anhhd/New Volume/navsim_workspace/checkpoints/bevfusion_demo_mmdet3d.pth")


BOX_CONVENTION_TAG = "mmdet3d_v1_bottom_z (legacy dims/yaw + gravity-z corrected)"


def legacy_to_v1_boxes(boxes: np.ndarray, *, fix_dims_yaw: bool = True, fix_gravity_z: bool = True) -> np.ndarray:
    """Checkpoint output -> mmdet3d>=1.0 boxes ``(x, y, z_bottom, l, w, h, yaw, vx, vy)``.

    * legacy dims/yaw: ``(w, l, h, yaw_old)`` -> ``(l, w, h, -yaw_old - pi/2)``
    * gravity z: mmdet3d's BEVFusion coder returns the gravity-centre z (encode adds
      h/2 to an already gravity-centred target, decode subtracts it once), while
      ``LiDARInstance3DBoxes`` expects the bottom centre -> ``z -= h/2``.
      Measured on mini_val: median (z_pred - z_gt_bottom) / h = 0.496 (n = 2479).
    """

    boxes = np.asarray(boxes, dtype=np.float64)
    out = np.array(boxes, copy=True).reshape(-1, boxes.shape[-1] if len(boxes) else 9)
    if len(out):
        if fix_dims_yaw:
            out[:, 3], out[:, 4] = boxes[:, 4], boxes[:, 3]
            out[:, 6] = np.arctan2(np.sin(-boxes[:, 6] - np.pi / 2), np.cos(-boxes[:, 6] - np.pi / 2))
        if fix_gravity_z:
            out[:, 2] = boxes[:, 2] - boxes[:, 5] / 2.0
    return out


@dataclass(slots=True)
class FrameDetections:
    """Raw BEVFusion output for one keyframe (LiDAR frame)."""

    boxes: np.ndarray  # (N, 9) x, y, z_bottom, dx, dy, dz, yaw, vx, vy
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,) index into DETECTION_CLASSES
    bev: Any = None  # torch tensor [1, 512, 180, 180] (device), optional
    timings_ms: dict[str, float] = field(default_factory=dict)


def ensure_import_paths(mmdet3d_root: Path = DEFAULT_MMDET3D_ROOT) -> None:
    for path in (str(mmdet3d_root), str(mmdet3d_root / "projects/BEVFusion")):
        if path not in sys.path:
            sys.path.insert(0, path)


class BEVFusionRunner:
    def __init__(
        self,
        config: Path | str = DEFAULT_CONFIG,
        checkpoint: Path | str = DEFAULT_CHECKPOINT,
        *,
        device: str = "cuda:0",
        mmdet3d_root: Path = DEFAULT_MMDET3D_ROOT,
        legacy_box_convention: bool = True,
    ) -> None:
        ensure_import_paths(mmdet3d_root)
        self.legacy_box_convention = legacy_box_convention
        import torch
        from mmdet3d.apis import init_model
        from mmengine.config import Config
        from mmengine.dataset import Compose

        cfg = Config.fromfile(str(config))
        cfg.model.img_backbone.init_cfg = None  # weights come from the checkpoint
        self.model = init_model(cfg, str(checkpoint), device=device)
        self.model.eval()
        self.cfg = cfg
        self.device = device
        self.pipeline = Compose(deepcopy(cfg.test_dataloader.dataset.pipeline))
        self._captured: dict[str, Any] = {}
        self._hook = self.model.pts_neck.register_forward_hook(self._capture_neck)
        self.torch = torch

    def _capture_neck(self, _module: Any, _inputs: Any, output: Any) -> None:
        feature = output[0] if isinstance(output, (list, tuple)) else output
        self._captured["bev"] = feature.detach()

    def preprocess(self, pipeline_input: dict[str, Any]) -> dict[str, Any]:
        from mmengine.dataset import pseudo_collate

        return pseudo_collate([self.pipeline(deepcopy(pipeline_input))])

    def infer(self, pipeline_input: dict[str, Any], *, keep_bev: bool = True) -> FrameDetections:
        torch = self.torch
        sync = torch.cuda.synchronize if self.device.startswith("cuda") else (lambda: None)
        t0 = time.perf_counter()
        batch = self.preprocess(pipeline_input)
        sync()
        t1 = time.perf_counter()
        with torch.no_grad():
            result = self.model.test_step(batch)[0]
        sync()
        t2 = time.perf_counter()
        inst = result.pred_instances_3d
        boxes = inst.bboxes_3d.tensor.detach().cpu().numpy().astype(np.float64)
        if self.legacy_box_convention:
            boxes = legacy_to_v1_boxes(boxes)
        detections = FrameDetections(
            boxes=boxes,
            scores=inst.scores_3d.detach().cpu().numpy().astype(np.float64),
            labels=inst.labels_3d.detach().cpu().numpy().astype(np.int64),
            bev=self._captured.get("bev") if keep_bev else None,
            timings_ms={"preprocess_ms": (t1 - t0) * 1e3, "model_ms": (t2 - t1) * 1e3},
        )
        self._last_batch_metas = [item.metainfo for item in batch["data_samples"]]
        return detections

    def decode_bev(self, bev: Any) -> FrameDetections:
        """Run the frozen TransFusion head on a supplied BEV map (same metas as last ``infer``)."""

        torch = self.torch
        metas = getattr(self, "_last_batch_metas", None)
        if metas is None:
            from mmdet3d.structures import LiDARInstance3DBoxes

            metas = [{"box_type_3d": LiDARInstance3DBoxes}]  # the head only reads box_type_3d
        with torch.no_grad():
            outputs = self.model.bbox_head.predict([bev.to(next(self.model.parameters()).device).float()], metas)
        inst = outputs[0]
        boxes = inst.bboxes_3d.tensor.detach().cpu().numpy().astype(np.float64)
        return FrameDetections(
            boxes=legacy_to_v1_boxes(boxes) if self.legacy_box_convention else boxes,
            scores=inst.scores_3d.detach().cpu().numpy().astype(np.float64),
            labels=inst.labels_3d.detach().cpu().numpy().astype(np.int64),
        )

    def heatmap(self, bev: Any) -> Any:
        """Dense class heatmap (sigmoid) of the frozen head for a BEV map: [1, 10, 180, 180]."""

        head = self.model.bbox_head
        with self.torch.no_grad():
            dense = head.shared_conv(bev)
            return head.heatmap_head(dense).sigmoid()
