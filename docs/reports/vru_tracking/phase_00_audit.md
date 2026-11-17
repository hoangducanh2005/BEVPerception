# Phase 0 — Repository and Evaluation Audit

## A. Objective

Audit the existing BEVFusion → tracking → evidence pipeline before changing an
algorithm. The audit covers data flow, transforms, timestamps, scene resets,
class mapping, identity metrics, and query lifecycle.

Acceptance requires a reproducible baseline and a clearly documented protocol.
The data mount was initially absent, then restored read-only from
`/dev/nvme1n1p4`; fresh bounded baselines were subsequently executed.

## B. Work completed

### Repository architecture map

```text
sensor inputs
  ├── scripts/phase12_nuscenes_infer.py
  │     ├── NuScenes metadata/calibration reader
  │     ├── MMDetection3D inference_multi_modality_detector()
  │     ├── pts_neck forward hook → [B, 512, 180, 180] BEV feature
  │     ├── world-frame DetectionMeasurement conversion
  │     ├── TrackManager Phase 1/2
  │     ├── evidence_metrics.py matching and aggregation
  │     └── camera/BEV images + tracks.json + metrics.json + evidence.md
  │
  └── scripts/phase12_navsim_infer.py
        ├── NavSim/OpenScene camera mapping and PCD conversion
        ├── same BEVFusion inference/checkpoint
        └── same bounded tracking/evidence path
```

Relevant implementation inventory:

| Concern | File/function | Observed behavior |
|---|---|---|
| Detector integration | `scripts/phase12_nuscenes_infer.py:183-221` | Loads external MMDetection3D BEVFusion config/checkpoint and captures `model.pts_neck`. |
| Sensor calibration | `scripts/phase12_nuscenes_infer.py:73-120` | Builds rigid transforms from nuScenes quaternions/calibration records. |
| LiDAR → global | `scripts/phase12_nuscenes_infer.py:103-105` | `ego_to_global @ lidar_to_ego`. |
| Prediction → tracker | `scripts/phase12_nuscenes_infer.py:223-229` | LiDAR box centers are transformed to global before tracking. |
| Tracker | `src/bevdrive/tracking/track_manager.py` | Constant-velocity Kalman state `[x,y,vx,vy]`, class-aware Hungarian center gating, 3 missed frames. |
| Association | `src/bevdrive/tracking/association.py` | Euclidean XY cost, same-class only, max distance 3m. |
| Temporal memory | `src/bevdrive/tracking/temporal_query.py` | Ego-pose BEV warp, cosine/delta/valid-fraction diagnostics, reset on scene boundary. |
| Query bank | `src/bevdrive/tracking/temporal_query.py` | BEV sample at track center and EMA update; no expiration policy. |
| Evaluation | `scripts/evidence_metrics.py:13-136` | Greedy score-ordered same-class XY matching at 2m. |
| Output | `scripts/phase12_*_infer.py` | Per-frame evidence plus aggregate JSON/Markdown and image overviews. |

### Commands/checks performed

- Inspected all tracking, temporal, geometry, dataset and evidence modules.
- Inspected previous local `metrics.json`, `tracks.json` and `evidence.md`.
- Verified repository was clean before this audit.
- Added lifecycle regression tests in `tests/test_tracking_lifecycle.py`.
- The supplied paths were checked with `ls`, `find`, `stat`, and the nuScenes
  SDK constructor.
- Fresh baseline commands were executed into new, non-overwriting directories:
  `artifacts/experiments/vru_tracking/phase_00/navsim_baseline` and
  `artifacts/experiments/vru_tracking/phase_00/nuscenes_baseline`.

## C. Experimental results

### Existing bounded artifact baseline

| Dataset | Frames | Tracking P/R/F1 | ID switches | Unique IDs |
|---|---:|---:|---:|---:|
| NavSim shard 5 | 10 | 0.833 / 0.133 / 0.229 | 0 | 22 |
| nuScenes scene-0003/0004 | 80 | 0.948 / 0.237 / 0.380 | 232 | 190 |

These values come from existing local artifacts, not a new run in this phase.

### Current data verification

The requested roots were temporarily absent at the start of the audit, then
mounted from the identified NTFS partition:

```text
/dev/nvme1n1p4 -> /media/anhhd/New Volume (NTFS, read-only)
/media/anhhd/New Volume/nuscenes          -> verified
/media/anhhd/New Volume/navsim_workspace  -> verified
```

The following data checks are now verified:

- nuScenes `v1.0-trainval` metadata: 850 scenes and 34,149 samples.
- `scene-0003`: 40 samples, all selected sensor files present, max sensor
  timestamp span 81.901ms.
- `scene-0004`: 40 samples, all selected sensor files present, max sensor
  timestamp span 81.215ms.
- NavSim sequence exists with 834 frames; first 10 are selected.
- Fresh frozen-checkpoint baselines completed for both datasets.

Raw-detection threshold sweep and GT-box oracle remain Phase 1/2 work and were
not claimed here.

### Confirmed metric behavior from existing JSON/code

For the old nuScenes artifact, the `232` reported identity changes decompose
as follows when comparing consecutive matched GT-instance assignments:

| Scene | Consecutive-frame ID changes | Changes after a gap | Total reported |
|---|---:|---:|---:|
| scene-0003 | 11 | 26 | 37 |
| scene-0004 | 133 | 62 | 195 |
| Total | 144 | 88 | 232 |

The 88 post-gap changes are not cleanly separable as true ID switches under a
standard MOT protocol; they are at least fragmentation candidates. The current
code increments `id_switches` whenever a matched instance later receives a
different track ID, regardless of whether the previous match was in the
immediately preceding frame.

The Phase 2 query counts are monotonic within each scene:

- scene-0003: 21 → 72
- scene-0004: 15 → 190

This confirms that `TemporalQueryBank._states` accumulates historical IDs and
does not represent the current active tracker set.

## D. Technical findings

### Verified behavior

1. BEVFusion predictions are in the LiDAR frame and are transformed to the
   global/odometry frame before Kalman tracking.
2. The Kalman `dt` is computed from integer microsecond timestamps as
   `(current - previous) / 1e6`, with a small lower bound of 1ms.
3. `TrackManager` resets all tracks and restarts IDs when `scene_id` changes.
4. Temporal BEV memory and query bank reset on scene changes in both inference
   scripts.
5. Ground-truth boxes returned by the nuScenes path and predictions are
   matched in the LiDAR coordinate frame before global tracking is used.
6. Hungarian association is one-to-one and class-aware, but uses only XY
   center distance.

### Confirmed protocol issues

1. **ID switch versus fragmentation is conflated.** The evaluator has no
   explicit consecutive-frame condition or gap counter.
2. **GT history is not explicitly reset at the scene boundary.** nuScenes
   instance tokens are normally scene-scoped, so collision is unlikely, but the
   protocol should reset it explicitly for correctness.
3. **Matching is greedy rather than optimal.** Predictions are processed in
   descending score order; this can differ from a global one-to-one minimum
   distance assignment.
4. **Native VRU classes are collapsed.** `PRED_TO_COARSE` maps both
   `bicycle` and `motorcycle` to `vehicle`, so current aggregate metrics cannot
   answer bicycle/motorcycle recall separately.
5. **Query state has no lifecycle.** It is not expired when a tracker deletes a
   track, so `max_active_queries` is accumulated historical state.
6. **Existing metrics do not record the full reproduction manifest.** The
   output JSON records the phase and data subset but not checkpoint hash,
   config hash, seed, or command line.

### Suspected but unverified due missing mount

- Camera image presence and calibration consistency for the selected frames.
- Exact LiDAR/camera timestamp spans in the current files.
- Whether false negatives are thresholded detections or absent after model
  top-K/NMS.
- Whether the detector's bicycle/motorcycle labels match GT categories on this
  mount.

## E. Changes relative to previous phase

No detector, tracker, class mapping, or evaluation algorithm was changed in
Phase 0. Only two regression tests were added for the existing tracker:

- scene reset clears state and restarts IDs;
- timestamp delta reaches the constant-velocity prediction path.

## F. Limitations

- The NTFS volume is dirty according to the kernel and was mounted read-only;
  Windows CHKDSK is recommended before a writable mount.
- The fresh baseline JSON does not yet include a full command/config/checkpoint
  manifest; the exact commands and checkpoint hash are recorded in this report
  and will be added to the experiment runner in Phase 1.
- No raw predictions below threshold were cached, so low-threshold recovery
  cannot be established from current artifacts.
- No official nuScenes tracking evaluator was run.

## G. Recommendation

Proceed to Phase 1 detection analysis using the restored read-only mount. Before
interpreting tracking improvements:

1. rerun the exact 80-frame/10-frame baseline;
2. fix or version the evaluation protocol for consecutive ID switches,
   fragmentation, native VRU classes and query lifecycle;
3. cache raw detector outputs before score filtering;
4. then run the threshold and distance sweeps.

## H. Completion status

**PASS**

The repository, current mounted dataset metadata, transforms, timestamps and
fresh bounded baseline are verified. Protocol issues are documented and will
be handled as explicit Phase 2/3 evaluation work rather than silently ignored.
