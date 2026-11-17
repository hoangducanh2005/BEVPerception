# Phase 3 — Tracking-by-detection baseline (Baseline A)

Status: **done**. BEVFusion → Kalman filter + Hungarian, evaluated with the
official nuScenes tracking metrics.

Checkpoint 1 already had a Kalman/Hungarian tracker, but it consumed buggy
detections (Phase 0 §5), used a single 3 m gate, ignored the detector's
velocity, and was scored with custom metrics only. The existing
`ConstantVelocityKalman` is reused (with one added joint position+velocity
update). The tracker logic is a new configurable module, so the old
`TrackManager` keeps working for the checkpoint-1 scripts.

## 1. What was implemented (`src/bevdrive/tracking/mot_baseline.py`)

| Requirement | Implementation |
|---|---|
| Ego-motion compensation | Boxes are tracked in the **global** frame (LiDAR → ego → global per keyframe), so ego motion is removed by construction. Outputs are transformed back to the current LiDAR frame. |
| 3D state, position + velocity | Constant-velocity Kalman `[x, y, vx, vy]`. The measurement is the detected centre **and** the BEVFusion velocity (rotated to global); σ_pos = 0.5 m, σ_vel = 1 m/s. Process noise is per class (pedestrian 2, bicycle 3, motorcycle 4 m/s²). |
| Class-aware association | Hungarian on the centre distance between the KF prediction and the detection. Pairs of different classes are forbidden. Per-class gates: pedestrian 1.5, bicycle 3, motorcycle 4, car 4, truck 4, bus 5.5, trailer 3 m. |
| Two-pass association | Pass 1 uses detections with score ≥ 0.3; pass 2 matches the remaining tracks to low-score detections (0.1–0.3), ByteTrack-style. Recovers partially occluded VRUs. |
| Initialisation | A new track from each unmatched detection with score ≥ 0.3, with its velocity initialised from the detector |
| Ageing / termination / occlusion | Unmatched tracks coast on the KF prediction for up to `max_age = 3` frames (1.5 s), then are deleted |
| Persistent IDs | Unique per scene; reset at scene change |
| Output | Matched tracks only (nuScenes convention). tracking_score = detection score; velocity = filtered KF velocity. |

## 2. Data flow

```text
Phase-1 detections (LiDAR frame, 200 / frame) ─► class + score filter (≥0.1, 7 tracking classes)
   ─► global frame (centre, velocity) ─► KF predict (dt from timestamps)
   ─► Hungarian pass 1 (score≥0.3) ─► Hungarian pass 2 (0.1–0.3) ─► KF update (pos + vel)
   ─► births (unmatched, ≥0.3) / coasting / deletion ─► tracks (id, box, velocity, history)
   ─► nuScenes tracking submission (global) ─► devkit TrackingEval
```

## 3. Files

Created `src/bevdrive/tracking/mot_baseline.py`, `scripts/cp2_track_baseline.py`,
`scripts/cp2_render_tracks.py` and `scripts/cp2_failure_analysis.py`.
Modified `src/bevdrive/tracking/kalman.py` (added `update_position_velocity`).

## 4. Commands

```bash
conda run -n ai_core python scripts/cp2_track_baseline.py \
  --records artifacts/checkpoint_2/phase_01/mini_all/records.pkl \
  --detections artifacts/checkpoint_2/phase_01/mini_all/detections.pkl \
  --scenes scene-0103 scene-0916 --output-dir artifacts/checkpoint_2/phase_03/kf_default
conda run -n uniad2.0 python scripts/cp2_eval_nuscenes.py --task track \
  --result-json artifacts/checkpoint_2/phase_03/kf_default/submission_track.json \
  --output-dir results/phase_03/kf_default --eval-set mini_val
conda run -n ai_core python scripts/cp2_render_tracks.py --tracks artifacts/checkpoint_2/phase_03/kf_default/tracks.pkl \
  --out-dir outputs/phase_03/kf_default --label "A: BEVFusion + KF/Hungarian" --video
```

## 5. Results (nuScenes TrackingEval `tracking_nips_2019`, mini_val, 81 keyframes)

All hyper-parameters were fixed **before** looking at mini_val. The
ablations below are reported for insight and were not used to pick a
configuration.

### 5.1 Baseline A (default configuration)

| Class | AMOTA ↑ | AMOTP (m) ↓ | Recall | MOTA | IDS | FRAG | GT |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Pedestrian** | **0.856** | 0.378 | 0.824 | 0.789 | 4 | 0 | 1088 |
| **Bicycle** | **0.475** | 1.131 | 0.537 | 0.537 | 0 | 0 | 41 |
| **Motorcycle** | **0.697** | 0.746 | 0.728 | 0.719 | 1 | 2 | 224 |
| VRU mean | 0.676 | | | | 5 | 2 | |
| All 7 classes (devkit mean) | 0.741 | 0.626 | 0.767 | 0.749 | 26 | 26 | |

Tracker latency is 11.5 ms/frame (CPU, Python), on top of BEVFusion.

External reference, *not* the same detector: UniAD-B stage 1 (camera-only
BEVFormer + TrackFormer) on the same mini_val with the same devkit, from
`~/GIT/UniAD/research_outputs`, scores AMOTA 0.486. Per class: pedestrian
0.532 (IDS 28), bicycle 0.148, motorcycle 0.454 (IDS 5). The gap is mostly
the detector (LiDAR + camera vs camera only), not the tracker.

### 5.2 Design ablations (same detections)

| Variant | AMOTA (all) | IDS | Ped AMOTA / IDS | Bicycle | Moto AMOTA / IDS | VRU mean |
|---|---:|---:|---:|---:|---:|---:|
| **Default** | 0.741 | 26 | 0.856 / 4 | 0.475 | 0.697 / 1 | 0.676 |
| No detector velocity (KF position only) | 0.726 | 41 | 0.854 / 3 | 0.430 | 0.673 / 2 | 0.652 |
| No coasting (`max_age = 1`) | 0.723 | 38 | 0.856 / 4 | 0.475 | 0.622 / 2 | 0.651 |
| Also report coasted tracks (score × 0.5^misses) | 0.767 | 21 | 0.848 / 2 | 0.600 | 0.717 / 1 | 0.722 |
| Single-pass association (all ≥ 0.1 in one pass) | 0.770 | 49 | 0.871 / 15 | 0.509 | 0.767 / 1 | 0.716 |

Reading:

* The detector's velocity matters most for identity: IDS 41 → 26 without
  vs with it.
* Coasting matters for motorcycles, which are often occluded among parked
  cars: AMOTA 0.62 → 0.70.
* A single pass raises AMOTA but nearly doubles ID switches (49), mostly on
  pedestrians: low-score boxes steal tracks in crowds. The two-pass
  hierarchy trades a little AMOTA for identity stability.

### 5.3 Diagnostic failure counts (score ≥ 0.3, 2 m, VRU GT with ≥1 point within 40 m)

These are our own diagnostic counts, not official metrics. The official IDS
uses a per-recall-threshold matching.

| Class | GT | TP | ID switch (consecutive) | Re-ID failure after gap | Fragmentation | Miss (occluded*) | Miss (visible) | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pedestrian | 1060 | 884 | 27 | 9 | 41 | 130 | 46 | 168 |
| Bicycle | 36 | 18 | 0 | 0 | 0 | 15 | 3 | 8 |
| Motorcycle | 212 | 123 | 0 | 0 | 21 | 70 | 19 | 9 |

\*occluded = nuScenes visibility ≤ 40 % or < 5 LiDAR points. Most VRU misses
are occluded objects: 74 % of pedestrian and 79 % of motorcycle misses.

## 6. Visualisations

* `outputs/phase_03/kf_default/tracking.mp4`: all 81 frames at 2 fps. Six
  cameras with boxes coloured by track id (`ped #25`), plus a BEV with GT,
  ids, 1 s velocity arrows and 12-frame trails.
* Per-frame images in `outputs/phase_03/kf_default/scene-*_fNN.jpg`.
* Failure cases: `outputs/phase_05/failures_A/` (red = missed GT, yellow =
  ID switch, orange = false positive), summarised in
  `results/phase_05/failures_A.json`.

## 7. Limitations / next steps

* There is no appearance cue: identity is purely geometric. Crossing
  pedestrians in dense groups (scene-0916 bus stop) cause most consecutive ID
  switches.
* Gates and noise are hand-set per class, and nothing was tuned on train
  scenes. A tuning sweep on mini_train (or trainval) is a clean next step.
* mini_val has only 41 bicycle GT boxes (2–3 instances), so bicycle AMOTA is
  very noisy.
