# BEVPerception — Checkpoint 2 Handoff

Context for the next agent: what exists, what actually ran, which numbers are
valid, and what is open. It supersedes `checkpoint_1.md`, whose
detection/tracking numbers came from buggy inputs; see §3.

## 1. Objective and scope

VRU (pedestrian, bicycle, motorcycle) 3D detection and multi-object tracking
from 6 cameras + LiDAR on nuScenes. The pipeline is:
BEVFusion → temporal BEV → (A) Kalman/Hungarian or (B) UniAD-style
TrackFormer. Prediction, occupancy and planning are out of scope.

## 2. Environment, data, checkpoints

| Item | Value |
|---|---|
| Env `ai_core` | BEVFusion / mmdet3d 1.4 / torch 2.1; all `cp2_*` scripts |
| Env `uniad2.0` | UniAD / mmcv 1.7.2; **tracking evaluation** (has motmetrics 1.1.3) and the UniAD dump script |
| BEVFusion checkpoint | `/media/anhhd/New Volume/navsim_workspace/checkpoints/bevfusion_demo_mmdet3d.pth`, frozen. Outputs the legacy box convention, converted by `legacy_to_v1_boxes`. |
| UniAD checkpoints | `~/GIT/UniAD/ckpts/uniad_base_track_map.pth` (stage-1 weights used to initialise B) |
| nuScenes-mini | `/home/anhhd/dataset/nuscenes-mini` |
| nuScenes trainval (partial) | `/media/anhhd/New Volume/nuscenes`. Mount read-only: `udisksctl mount -b /dev/nvme0n1p4 -o ro`; the device name changes between boots, so check `lsblk`. 84 complete scenes = 61 train + 23 val: `artifacts/checkpoint_2/phase_01/trainval_partial/scenes.json`. |
| Feature caches | `artifacts/checkpoint_2/bev_cache/v1.0-mini` (404 × 33 MB) and `.../v1.0-trainval` (3,337 × 33 MB, about 110 GB). fp16 `[512,180,180]` per sample token. |

## 3. Checkpoint-1 defects found and fixed

1. No LiDAR sweeps.
2. Every frame used the first frame's camera images (`inference_multi_modality_detector` index bug).
3. `lidar2cam` ignored per-camera ego pose.
4. Wrong BEV axis convention in the old warp and query sampling.
5. Legacy box dims/yaw and gravity-z in checkpoint outputs.
6. Custom metrics, and a training-split scene in the evaluation.

Quantified in `reports/phase_00_audit.md` and `reports/phase_01_detection.md`:
the checkpoint-1 input configuration scores VRU mAP 0.348; fixed, it scores
0.743.

## 4. Code map (new in checkpoint 2)

| Path | Purpose |
|---|---|
| `src/bevdrive/datasets/nuscenes_infos.py` | Per-keyframe records: 10 sweeps, exact calibration, GT with instance ids. `pipeline_input()` |
| `src/bevdrive/perception/bevfusion_runner.py` | Frozen BEVFusion; BEV capture; `decode_bev`; `heatmap`; box-convention fix |
| `src/bevdrive/temporal/bev_alignment.py` | `BEVGrid` (layout-aware), `resample`, `warp_to_current`, `TemporalBEVBuffer`, `aggregate` (none/mean/decay/gated) |
| `src/bevdrive/tracking/mot_baseline.py` | Baseline A tracker (KF + detector velocity, two-pass class-aware Hungarian, coasting) |
| `src/bevdrive/trackformer/` | UniAD TrackFormer port (`layers`, `query_modules`, `criterion`), `bev_adapter`, `model`, `weights` |
| `src/bevdrive/evaluation/nuscenes_submission.py` | LiDAR → global devkit submissions |
| `src/bevdrive/visualization/render3d.py` | Six-camera projection + BEV canvas |
| `scripts/cp2_*.py` | Phase drivers (detect, temporal, track baseline, train/infer query tracker, online pipeline, eval, render, failure analysis, compare) |
| `scripts/uniad_dump_track_io.py` | Runs in `uniad2.0` from the UniAD repo root; dumps BEV + tracker I/O for the port check |
| `tests/test_bev_alignment.py`, `test_sweep_encoding.py`, `test_trackformer.py` | 12 new tests (25 total pass) |

The old modules (`tracking/track_manager.py`, `temporal_query.py`,
`phase12_*` scripts) are untouched. `temporal_query.py` is kept only as the
"checkpoint-1 warp" comparison; do not use it for new work.

## 5. Valid results (official devkit)

* mini_val: `reports/phase_05_evaluation.md` §3, `results/phase_05/comparison_mini_val.md`
* 23 trainval-val scenes ("val-23", 914 keyframes): `reports/phase_05_evaluation.md` §6, `results/phase_05/comparison_val23.md`.
  Evaluate with `--version v1.0-val23-trainval --data-root artifacts/checkpoint_2/nuscenes_slim`, because full
  trainval metadata OOMs the devkit on this 15 GB machine.

Key mini_val numbers:

* BEVFusion VRU mAP **0.743** (pedestrian AP 0.926).
* **A**: AMOTA **0.741**, VRU AMOTA 0.676, pedestrian 0.856, IDS 26.
* **B** (mini-trained): AMOTA 0.530, overfit; 0.851 on its own training scenes.
* UniAD camera-only reference: AMOTA 0.486.
* The port reproduces UniAD (AMOTA 0.488 vs 0.486, 99.4 % same ids).

Key val-23 numbers:

* BEVFusion mAP 0.653 / NDS 0.693, VRU mAP 0.761.
* **A**: AMOTA 0.589, VRU 0.626; train-tuned 0.619 / 0.668.
* **B** (mini-trained): AMOTA 0.536, VRU 0.485.

## 6. Open items / recommended next steps

1. B's generalisation. **The B scale-up training was paused at the user's
   request (2026-11-17) and has not run.** `artifacts/checkpoint_2/run_scaleup.sh`
   trains on the 61 local train-split scenes + mini_train (4 epochs, about
   90 min), then infers and evaluates on mini_val and val-23. Its val-23
   evaluation lines already use the slim metadata (§5). Beyond that: only 61 of 700 train scenes are on disk, so
   download more blobs *with approval*. Then validate design
   changes on a held-out **train** scene (never on val). The most promising
   change is a detection-score prior: initialise the class logit of each
   detection query from `logit(BEVFusion score)` and learn a residual.
2. Baseline A: tune the gates and noise on train scenes. The per-class
   two-pass and coasting choices already help; see the ablations in Phase 3.
3. The temporal BEV is validated but adds nothing yet. Try flow-guided
   (per-object) warping, or history > 2 with a learned gate.
4. Evaluation: run the full 150-scene val when the data is available. All
   scripts take `--version v1.0-trainval --data-root ... --scenes val`.
5. Nothing was committed at the time of writing. `artifacts/` and `outputs/`
   are git-ignored; `results/` (small JSONs) and `reports/` are tracked.
