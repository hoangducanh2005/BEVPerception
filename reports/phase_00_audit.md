# Phase 0 — Repository and environment audit (checkpoint 2)

Status: **done**. Everything below was checked by running code or reading source
in this session (2026-11-17). Where a number comes from a later phase, that
phase is named.

## 1. What was audited

* The BEVPerception repo at commit `c1e0caa` (checkpoint 1), including the
  handoff `docs/handoff/checkpoint_1.md`.
* The external MMDetection3D checkout (BEVFusion project) and the frozen
  BEVFusion checkpoint.
* The UniAD v2.0 repo `~/GIT/UniAD`, with its checkpoints, its `uniad2.0` conda
  env and the earlier UniAD reproduction in `~/GIT/UniAD/research_outputs`.
* Datasets: local nuScenes-mini, and the partial nuScenes trainval on the data
  disk.
* The GPU, RAM, disk and Python packages.

## 2. Environment

| Item | Value |
|---|---|
| GPU | RTX 4050 Laptop, 6 GB (driver 580) |
| RAM / free disk | 15 GB / 298 GB on `/` |
| Env `ai_core` (BEVFusion + this repo) | Python 3.10, torch 2.1.0+cu121, mmcv 2.1.0, mmdet 3.2.0, mmdet3d 1.4.0, nuscenes-devkit 1.2.0, **no motmetrics** |
| Env `uniad2.0` (UniAD) | Python 3.9, torch 2.0.1+cu118, mmcv-full 1.7.2, mmdet 2.26, mmdet3d 1.0.0rc6, nuscenes-devkit 1.1.11, motmetrics 1.1.3 |
| BEVFusion checkpoint | `/media/anhhd/New Volume/navsim_workspace/checkpoints/bevfusion_demo_mmdet3d.pth` (unchanged, frozen) |
| BEVFusion config | `mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py` |
| UniAD checkpoints | `UniAD/ckpts/uniad_base_track_map.pth` (stage 1: track + map), `uniad_base_e2e.pth` (stage 2) |
| nuScenes-mini | `/home/anhhd/dataset/nuscenes-mini` (v1.0-mini, 10 scenes, 404 keyframes, sweeps present) |
| nuScenes trainval (partial) | `/media/anhhd/New Volume/nuscenes`. Full metadata, but only 3,376 keyframes with LiDAR sweeps on disk (about 85 scenes) |

The data disk had been renamed `/dev/nvme0n1p4` and was not mounted. It was
mounted **read-only** with `udisksctl mount -b /dev/nvme0n1p4 -o ro`, as the
handoff requires (the NTFS volume is marked dirty).

The tracking eval needs `motmetrics`, which is not in `ai_core`. Rather than
change that env, both detection and tracking eval run through the devkit, and
tracking eval runs in `uniad2.0`. That is the same devkit and motmetrics
version that produced the UniAD reference numbers, so the comparison is
like-for-like.

## 3. What already existed and what was reused

| Component | Before checkpoint 2 | Decision |
|---|---|---|
| BEVFusion inference | `scripts/phase12_nuscenes_infer.py` via `inference_multi_modality_detector` | **Replaced** by `perception/bevfusion_runner.py`, because of the input bugs below. The model and config are reused unchanged. |
| Fused-BEV capture | `pts_neck` forward hook (`tracking/temporal_query.py`) | Reused (same hook idea) |
| Temporal BEV warp | `tracking/temporal_query.py::warp_bev_feature` | **Wrong axis convention** (see 5.4). New module `temporal/bev_alignment.py`; the old function is kept only as a comparison baseline. |
| Kalman / Hungarian | `tracking/kalman.py`, `association.py`, `track_manager.py` | Kalman filter reused and extended. A new configurable `tracking/mot_baseline.py` was added; the old `TrackManager` is left untouched for the old scripts. |
| Camera projection, rigid transforms | `geometry/*` | Kept. The new renderer `visualization/render3d.py` follows the same conventions. |
| Metrics | Custom greedy 2 m matching (`evidence_metrics.py`) | **Replaced** by the official nuScenes devkit (DetectionEval, TrackingEval → AMOTA/AMOTP/IDS/FRAG) |
| Query tracker | None (only an EMA "query bank", no learned parts) | New: `src/bevdrive/trackformer/` (UniAD TrackFormer port + BEVFusion adapter) |

### UniAD TrackFormer reuse analysis

UniAD's tracker is `UniADTrack` + `BEVFormerTrackHead` + `track_head_plugin`
(QIM, MemoryBank, RuntimeTrackerBase) + `losses/track_loss.py` (ClipMatcher).
It is written against mmcv 1.x / mmdet 2.x registries, which cannot be
imported next to BEVFusion's mmcv 2.1. Reuse was therefore done by **porting**
the code to plain PyTorch with identical parameter names, so that UniAD's
stage-1 weights load directly. Phase 4 verifies the port numerically against
UniAD.

The two checkpoints are **not** directly compatible. The interface differences
are listed in `reports/phase_04_query_tracking.md` §2: 512 vs 256 channels,
180² @ 0.6 m vs 200² @ 0.512 m, `[C,x,y]` vs `[C,y,x]` layout, ±54 m vs
±51.2 m range, LiDAR+camera features vs BEVFormer features, and no
positional or temporal content in BEVFusion's BEV.

## 4. Data actually used

| Split | Scenes | Keyframes | nuScenes official split | Use |
|---|---|---|---|---|
| mini_train | 0061, 0553, 0655, 0757, 0796, 1077, 1094, 1100 | 323 | 6 train + 2 val (0553, 0796) | TrackFormer training only |
| mini_val | 0103, 0916 | 81 | both **val** | All evaluation; never used for training or model selection |

Checkpoint 1 evaluated on scene-0003 (val) and scene-0004. scene-0004 is in the
official **train** split, so the BEVFusion detector had seen it. Checkpoint 2
evaluates only on mini_val (val scenes). That is also the split of the
existing UniAD reference numbers.

## 5. Problems found (with evidence)

| # | Problem | Evidence | Impact | Fixed in |
|---|---|---|---|---|
| 5.1 | **LiDAR sweeps missing.** `make_info()` provides no `lidar_sweeps`. The config's `LoadPointsFromMultiSweeps(pad_empty_sweeps=True)` then silently repeats the keyframe 9× and the time channel is 0. | Source read: `scripts/phase12_nuscenes_infer.py:158`, `mmdet3d/datasets/transforms/loading.py:416` | Single sweep: mini_val VRU mAP 0.743 → 0.601, mAVE 0.43 → 1.01 | Phase 1 |
| 5.2 | **Stale camera images.** The script calls `inference_multi_modality_detector(model, one_lidar_path, ...)`, which reads `data_list[index]` with `index = 0`. Every frame was fused with the six images of the **first** frame. | Source read: `mmdet3d/apis/inference.py:233-238`; the script writes all 80 infos once (lines 177-180) | Stale images: VRU mAP 0.640. Both bugs (the checkpoint-1 setup): mAP 0.334, VRU mAP 0.348 (bicycle AP 0.095) | Phase 1 |
| 5.3 | **lidar2cam ignored the LiDAR/camera time offset.** `inv(cam2ego) @ lidar2ego` drops each camera's ego pose. | Source read | Up to about 0.4 m misalignment at 10 m/s | Phase 1 (full chain through both ego poses) |
| 5.4 | **Wrong BEV axis convention in checkpoint-1 warp and sampling.** They assumed rows = y, columns = x. The TransFusion head lays its BEV out as `[x, y]`. | Frozen heatmap at GT centres: 0.631 (`xy`) vs 0.007 (checkpoint-1 convention) vs 0.012 random (Phase 2) | The checkpoint-1 "temporal cosine 0.499" and query ROC-AUC were computed at wrong locations | Phase 2 |
| 5.5 | **Checkpoint box convention.** The demo checkpoint outputs legacy mmdet3d-0.x boxes (`dims = w, l, h`, `yaw_old`), and the mmdet3d coder returns gravity-centre z inside a bottom-centre box type. | Matched cars: pred `(dx, dy) = (w, l)` and `-yaw - π/2 = yaw_gt`; median `(z_pred - z_bottom)/h = 0.496` over 2,479 boxes | Before the fix: VRU ASE 0.80, AOE 1.5 rad, floating boxes in the camera views. AP and ATE were unaffected (2D centre matching). | Phase 1 (`legacy_to_v1_boxes`) |
| 5.6 | **Custom metrics.** Greedy 2 m matching, no GT point filtering, no class ranges, no AMOTA. | Source read | Not comparable to the literature | Official devkit used everywhere |
| 5.7 | `lidar2sensor` for sweeps: mmdet3d computes `p @ M[:3,:3] - M[:3,3]`. Its converter's encoding is exact only when R ≈ I. | `tests/test_sweep_encoding.py` | Up to about 0.2 m error on turning sweeps | Phase 1: encoded as `M = [Rᵀ, -t]` |

## 6. Plan, as executed

1. Phase 1: correct per-frame inputs, keep the frozen detector, decode all 200
   TransFusion proposals, cache the fused BEV (fp16), and use official
   detection metrics.
2. Phase 2: explicit-layout BEV grids, a pose-based warp, a history buffer,
   and motion-gated aggregation, validated on real features.
3. Phase 3: a Kalman + Hungarian baseline with detector velocity, per-class
   gates, a two-pass association and coasting, using official AMOTA.
4. Phase 4: port UniAD TrackFormer (decoder, QIM, MemoryBank, runtime
   lifecycle, ClipMatcher). Prove the port against UniAD on UniAD's own BEV,
   then add a BEVFusion→UniAD BEV adapter and detection queries from
   BEVFusion proposals. Train on mini_train with BEVFusion frozen.
5. Phase 5: compare A vs B on mini_val with the same detector outputs and the
   same devkit, plus latency, memory and visual evidence.

## 7. Files and commands

No code was changed in Phase 0. Commands used: `git status/log`, `nvidia-smi`,
`lsblk`, `udisksctl mount -b /dev/nvme0n1p4 -o ro`, `conda env list`,
package-version probes, a split-membership check with
`nuscenes.utils.splits.create_splits_scenes`, and a key/shape listing of the
UniAD checkpoint state dict.
