# Phase 1 — BEVFusion 3D detection (6 cameras + LiDAR)

Status: **done**. Frozen pretrained BEVFusion, corrected inputs, official
nuScenes detection metrics on mini_val.

## 1. What was implemented

* `datasets/nuscenes_infos.py` builds one record per keyframe with:
  * 9 previous LiDAR sweeps (10-sweep input, as in training), with the exact
    rigid transform encoded for mmdet3d's loader;
  * per-camera `lidar2cam` composed through the camera's own ego pose;
  * timestamps in seconds;
  * GT boxes in the LiDAR frame (mmdet3d ≥1.0 convention), with velocity,
    instance tokens, point counts and visibility.
* `perception/bevfusion_runner.py` runs the official test pipeline and
  `model.test_step` on each frame's own data. It captures the fused BEV
  (`pts_neck` output, `[1, 512, 180, 180]`) and can decode the frozen head on
  any supplied BEV (`decode_bev`, used in Phase 2) or return its dense heatmap.
* **Output convention fix** (`legacy_to_v1_boxes`). The demo checkpoint
  regresses legacy boxes (`w, l, h, yaw_old`), and the coder returns
  gravity-centre z. Outputs are converted to `(x, y, z_bottom, l, w, h, yaw,
  vx, vy)`. This only re-parametrises the outputs; the network is untouched.
* `evaluation/nuscenes_submission.py` converts LiDAR → ego → global and writes
  devkit submissions. The `attribute_name` heuristic matches mmdet3d's.
* `scripts/cp2_detect.py` is the driver. It also has ablation switches that
  reproduce the checkpoint-1 bugs (`--no-sweeps`, `--stale-images`).
* `scripts/cp2_eval_nuscenes.py` runs the official devkit evaluation
  (DetectionEval `detection_cvpr_2019`, TrackingEval `tracking_nips_2019`).

VRU filtering: the detector keeps all 10 classes, because the devkit needs
them and the tracker uses them as context. Pedestrian, bicycle and motorcycle
results are reported separately as "VRU".

## 2. Data flow

```text
nuScenes sample ─┬─ 6 camera images + intrinsics + lidar2cam(t_cam) ─┐
                 └─ LIDAR_TOP keyframe + 9 sweeps → keyframe frame ──┤
                                                                      ▼
   BEVFusion (frozen): Swin-T + LSS (camera BEV, 80 ch) ⊕ SECOND voxel (256 ch)
   → ConvFuser → SECOND backbone → SECONDFPN = fused BEV [512, 180(x), 180(y)], 0.6 m
                                                                      │  (cached fp16 per frame)
   TransFusion head (200 proposals, 1 decoder layer) → legacy→v1 box conversion
   → boxes (LiDAR) + score + class + velocity + timestamp → global → devkit
```

## 3. Files

Created: `src/bevdrive/datasets/nuscenes_infos.py`,
`src/bevdrive/perception/bevfusion_runner.py`,
`src/bevdrive/evaluation/nuscenes_submission.py`,
`src/bevdrive/visualization/render3d.py`, `scripts/cp2_detect.py`,
`scripts/cp2_eval_nuscenes.py`, `scripts/cp2_convert_legacy_dets.py`,
`tests/test_sweep_encoding.py`.

## 4. Commands

```bash
# all 10 mini scenes (404 frames); caches fused BEV for Phases 2-4
conda run -n ai_core python scripts/cp2_detect.py --scenes mini_train mini_val \
  --output-dir artifacts/checkpoint_2/phase_01/mini_all \
  --cache-bev artifacts/checkpoint_2/bev_cache/v1.0-mini --vis-dir outputs/phase_01/mini_all
# (this run predates the runner's convention fix; it was converted offline with)
conda run -n ai_core python scripts/cp2_convert_legacy_dets.py artifacts/checkpoint_2/phase_01/mini_all --vis-dir outputs/phase_01/mini_all
# official evaluation on mini_val
conda run -n ai_core python scripts/cp2_eval_nuscenes.py --task det \
  --result-json artifacts/checkpoint_2/phase_01/mini_all/submission_det.json \
  --output-dir results/phase_01/bevfusion_mini_val --eval-set mini_val
# checkpoint-1 bug reproduction (mini_val only)
python scripts/cp2_detect.py --scenes mini_val --output-dir .../ablation_nosweeps --no-sweeps
python scripts/cp2_detect.py --scenes mini_val --output-dir .../ablation_staleimg --stale-images
python scripts/cp2_detect.py --scenes mini_val --output-dir .../ablation_ckpt1bugs --no-sweeps --stale-images
```

## 5. Results (nuScenes devkit, mini_val = scene-0103 + scene-0916, 81 keyframes)

### 5.1 Main result and impact of the checkpoint-1 input bugs

| Input | mAP | NDS | mAVE | VRU mAP | Ped AP | Bicycle AP | Moto AP | Car AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Fixed** (10 sweeps, own images, exact calibration) | **0.587** | **0.589** | **0.429** | **0.743** | **0.926** | **0.575** | **0.728** | **0.929** |
| Single sweep | 0.507 | 0.454 | 1.012 | 0.601 | 0.876 | 0.346 | 0.582 | 0.883 |
| Stale (frame-0) camera images | 0.504 | 0.549 | 0.415 | 0.640 | 0.909 | 0.422 | 0.589 | 0.846 |
| Checkpoint-1 inputs (both bugs) | 0.334 | 0.370 | 1.048 | 0.348 | 0.759 | 0.095 | 0.189 | 0.627 |

mAP and NDS average all 10 classes. mini_val has no trailer,
construction-vehicle or barrier GT, so those classes score AP = 0 by absence;
absolute mAP is therefore not comparable to full-val numbers. VRU per-class AP
is the relevant figure.

### 5.2 VRU true-positive errors (fixed pipeline)

| Class | AP | ATE (m) | ASE | AOE (rad) | AVE (m/s) | AP@0.5 m | AP@1 m | AP@2 m | AP@4 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pedestrian | 0.926 | 0.113 | 0.257 | 0.383 | 0.222 | 0.909 | 0.919 | 0.927 | 0.950 |
| Bicycle | 0.575 | 0.166 | 0.241 | 0.356 | 0.537 | 0.568 | 0.568 | 0.568 | 0.593 |
| Motorcycle | 0.728 | 0.195 | 0.273 | 0.388 | 0.053 | 0.650 | 0.730 | 0.747 | 0.785 |

Before the box-convention fix, pedestrian ASE was 0.80 and AOE 1.62 rad. This
is what you'd expect when (w, l) are swapped and yaw is off by −yaw − π/2.
Fixing dims/yaw left AP unchanged (0.584). The gravity-z fix then raised
bicycle AP 0.552 → 0.575. Centre matching in the devkit is 2D, but its
bike-rack filter uses a **3D** point-in-box test. With z too high by h/2,
predicted bicycles inside racks escaped the filter that removes the matching
GT. All numbers here are after both fixes.

### 5.3 VRU recall by range (GT with ≥1 point, same class, ≤2 m centre distance)

| Class | score ≥ | 0–15 m | 15–30 m | 30–40 m | all (0–40 m) |
|---|---|---:|---:|---:|---:|
| Pedestrian (n = 1060) | 0.1 | 0.997 | 0.964 | 0.896 | 0.966 |
| Pedestrian | 0.3 | 0.933 | 0.790 | 0.747 | 0.834 |
| Bicycle (n = 36) | 0.1 | 0.833 | 0.579 | 0.400 | 0.639 |
| Motorcycle (n = 212) | 0.1 | 0.770 | 0.709 | 0.657 | 0.722 |

Checkpoint 1 reported pedestrian recall 0.037 at 30–50 m on its (buggy)
setup; with correct inputs, recall at 30–40 m is 0.90.

### 5.4 Runtime (RTX 4050 Laptop, fp32, batch 1, 404 frames, first 3 excluded)

| | Value |
|---|---|
| BEVFusion forward (`test_step`) | 635 ms mean (1.58 FPS) |
| Pre-processing (6 images + 10 sweeps from disk, pipeline) | 434 ms mean |
| End-to-end detector | 0.94 FPS |
| Peak GPU memory (allocated / reserved) | 2,295 MB / 2,588 MB |

## 6. Visualisations

* `outputs/phase_01/mini_all/*.jpg`: every 10th frame. The six cameras with
  projected 3D boxes (VRU coloured by class with scores, other classes grey)
  sit beside the BEV (ego frame, LiDAR points, GT green, predictions coloured,
  VRU velocity arrows).
* Example: `outputs/phase_01/mini_all/0380_scene-0916_f17.jpg` (motorcycle
  parking and a pedestrian group behind the ego).

## 7. Issues, limitations, next steps

* mini_val is small: 2 scenes, 36 bicycle GT boxes. Bicycle AP has a large
  variance, so compare VRU classes jointly or on more scenes.
* The demo checkpoint's provenance is unclear. Its outputs follow the legacy
  convention, and the conversion is verified empirically (§5.2). An official
  mmdet3d-1.x checkpoint would not need it.
* The cached BEV is fp16. Decoding the frozen head on the cache gives the
  same devkit result as the live run (mAP 0.587, VRU mAP 0.743, Phase 2).
* Full trainval: run with `--version v1.0-trainval --data-root
  "/media/anhhd/New Volume/nuscenes" --scenes train val --complete-only`.
  84 scenes are complete on disk; all 3,337 of their keyframes were
  processed (459 ms/frame model). On the 23 val-split scenes (914
  keyframes) BEVFusion scores **mAP 0.653, NDS 0.693, VRU mAP 0.761**
  (pedestrian 0.905, bicycle 0.682, motorcycle 0.696), in line with published
  full-val BEVFusion results (`results/phase_05/bevfusion_val23_det/`).
