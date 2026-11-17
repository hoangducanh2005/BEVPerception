# BEVPerception — Checkpoint 1 Handoff

This file is the handoff context for the next agent (Claude). It describes
what is implemented, what was actually executed, which results are valid, and
what remains unfinished. Treat the bounded metrics below as experiment
evidence, not full nuScenes benchmark results.

## 1. Project objective

The project targets inference-only 3D VRU detection and multi-view/multi-frame
tracking while preserving the existing camera–LiDAR BEV perception pipeline.
The current scope is perception only. Planning, MotionFormer, OccFormer and
full UniAD are out of scope.

No neural network was trained or fine-tuned. The pretrained checkpoint was not
modified.

## 2. Runtime and checkpoint

### Environment

- Repository: `/home/anhhd/GIT/BEVPerception`
- Conda environment: `ai_core`
- Python 3.10.22
- PyTorch 2.1.0 + CUDA 12.1
- MMCV 2.1.0 with CUDA ops
- MMEngine 0.10.7
- MMDetection 3.2.0
- MMDetection3D 1.4.0
- nuScenes devkit 1.2.0
- GPU: RTX 4050 class device, approximately 6GB VRAM
- External MMDetection3D checkout: `/home/anhhd/GIT/mmdetection3d`

### Frozen checkpoint

```text
Path:
/media/anhhd/New Volume/navsim_workspace/checkpoints/bevfusion_demo_mmdet3d.pth

SHA-256:
bce1028e7baccad9a75e21b6ff888f3a0452a480cdfae040bee44898093ca25e

Config:
/home/anhhd/GIT/mmdetection3d/projects/BEVFusion/configs/
bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
```

The data disk is `/dev/nvme1n1p4`, NTFS label `New Volume`, mounted as:

```text
/media/anhhd/New Volume
```

It was mounted read-only because the NTFS volume is marked dirty. Do not force
read-write mounting from Linux; run Windows CHKDSK first if writable access is
needed.

## 3. Implemented architecture

```text
6 nuScenes cameras + top LiDAR
              │
              ▼
Frozen BEVFusion camera–LiDAR fusion
              │
              ▼
Fused BEV feature from model.pts_neck
[1, 512, 180, 180]
              │
              ▼
3D TransFusion detection head
              │
              ▼
3D boxes + class + confidence
              │
      ┌───────┴────────┐
      │                │
      ▼                ▼
Phase 1             Phase 2
Kalman +            Temporal BEV warp
Hungarian           + persistent query
tracking            state prototype
      │                │
      └───────┬────────┘
              ▼
track_id + velocity + metric history
```

### Phase 1

`src/bevdrive/tracking/track_manager.py` uses:

- constant-velocity Kalman state `[x, y, vx, vy]`;
- ego-motion compensation by tracking object centers in the global/odometry
  frame;
- class-aware Hungarian association;
- XY center-distance gate of 3m;
- track deletion after 3 missed frames;
- timestamps converted from microseconds to seconds for `dt`.

### Phase 2

`src/bevdrive/tracking/temporal_query.py` adds:

- capture of the final fused BEV feature from `pts_neck`;
- ego-pose warp of the previous BEV into the current LiDAR frame;
- BEV temporal cosine, mean absolute delta and valid-warp diagnostics;
- BEV sampling at track centers;
- EMA persistent query-state prototype.

This is not learned UniAD. There is no TrackFormer decoder, learned identity
head, UniAD checkpoint, training loss or neural query-to-detection feedback.
The old Phase 2 query bank was diagnostic only and did not affect Hungarian
association.

### Current optional experimental code

An unvalidated optional `QueryAssociationTrackManager` was added in the current
working tree at `src/bevdrive/tracking/query_association.py`. It combines
motion cost and BEV appearance cosine with class/motion gating. It has not yet
produced a completed ablation output and must not be presented as an improved
tracker.

## 4. Dataset scope actually used

### Native nuScenes

```text
Root: /media/anhhd/New Volume/nuscenes
Version: v1.0-trainval
Scenes: scene-0003, scene-0004
Frames: 40 + 40 = 80 complete keyframes
```

Verification performed:

- 850 scenes and 34,149 samples available in metadata;
- no missing camera/LiDAR files in the selected 80 frames;
- maximum six-camera/LiDAR sensor timestamp span approximately 82ms;
- LiDAR/sample timestamp mismatch did not exceed 100ms.

### NavSim/OpenScene

```text
Root: /media/anhhd/New Volume/navsim_workspace
Sequence:
dataset/navsim_logs/mini/2021.06.09.14.58.55_veh-35_01894_02311.pkl
Sequence length: 834 frames
Frames used: first 10 consecutive frames only
```

The NavSim runner maps six of eight available cameras into the six-camera
nuScenes BEVFusion input convention. NavSim results are pipeline/domain sanity
checks, not native nuScenes benchmark results.

## 5. Main bounded baseline results

Original local Phase 1 and Phase 2 evidence is under:

- `artifacts/phase_runs/nuscenes_shard1/phase1/`
- `artifacts/phase_runs/nuscenes_shard1/phase2/`

Both phases use the same frozen detector and same Kalman/Hungarian tracker.

| Metric | Phase 1 | Phase 2 |
|---|---:|---:|
| Vehicle F1 | 0.551 | 0.551 |
| Pedestrian F1 | 0.326 | 0.326 |
| Tracking precision | 0.948 | 0.948 |
| Tracking recall | 0.237 | 0.237 |
| Tracking F1 | 0.380 | 0.380 |
| Reported ID changes | 232 | 232 |
| Unique track IDs | 190 | 190 |
| Mean temporal BEV cosine | 0.4994 | 0.4994 |
| Mean query cosine | N/A | 0.6181 |

Why Phase 1 and Phase 2 detection/tracking numbers are identical: the old
Phase 2 query state was only recorded. It was not connected to the detector or
the primary association cost.

The old tracking metrics are custom diagnostics:

- score threshold: `0.20`;
- greedy same-class BEV center matching at `2m`;
- no official AMOTA/AMOTP/HOTA/AssA;
- the old `232` ID-change count includes some post-gap reappearances.

## 6. Controlled experiment results

The newer controlled experiment outputs are under:

```text
artifacts/experiments/vru_tracking/
```

### Phase 0 — audit and fresh baseline

Fresh bounded baseline outputs:

- `phase_00/navsim_baseline/`
- `phase_00/nuscenes_baseline/`

Fresh nuScenes baseline reproduced the same detector behavior on 80 frames.
The audit confirmed two protocol problems: old ID switches mixed with
fragmentation, and old query counts accumulated historical states.

Report: `docs/reports/vru_tracking/phase_00_audit.md`

### Phase 1 — detection recall analysis

Runner: `scripts/vru_detection_experiment.py`

Artifacts:

- `phase_01/nuscenes_detection/raw_detections.json`
- `phase_01/nuscenes_detection/threshold_sweep.csv`
- `phase_01/nuscenes_detection/distance_recall.csv`
- `phase_01/nuscenes_detection/false_negative_analysis.json`
- `phase_01/nuscenes_detection/diagnostics/`

Important measured result:

| Threshold | Pedestrian precision | Pedestrian recall | Pedestrian F1 |
|---:|---:|---:|---:|
| 0.05 | 0.660 | 0.358 | 0.464 |
| 0.20 | 0.913 | 0.198 | 0.326 |

At threshold 0.20, the 924 unmatched VRU GT boxes were categorized as:

- 446 candidates below threshold;
- 424 with no same-class raw candidate within 2m;
- 37 with a wrong-class raw candidate nearby;
- 17 candidate competition/matching conflicts.

Conclusion: lowering threshold helps, but nearly half of the misses have no
same-class candidate in the cached model outputs. Thresholding is not the only
bottleneck. Pedestrian recall also falls to `0.037` in the 30–50m bin.

Report: `docs/reports/vru_tracking/phase_01_detection.md`

### Phase 2 — corrected tracking audit and GT-box oracle

Runner: `scripts/vru_tracking_oracle.py`

Artifact: `phase_02/tracking_oracle/oracle_metrics.json`

| Mode | Tracking F1 | Consecutive ID switches | Gap fragmentations | Lost frames |
|---|---:|---:|---:|---:|
| Predicted boxes + tracker | 0.371 | 133 | 83 | 351 |
| GT boxes + same tracker | 1.000 | 36 | 0 | 0 |

Interpretation:

- Detector dropout dominates the difference between predicted-box and oracle
  tracking.
- The tracker still has an association ambiguity component: 36 consecutive ID
  switches remain even with GT boxes.
- Scene-0004 is the main source of identity failures.

Report: `docs/reports/vru_tracking/phase_02_tracking_oracle.md`

### Phase 3 — BEV query identity analysis

Runner: `scripts/vru_query_analysis.py`

The final embedding cache is:

```text
phase_03/query_analysis_v3_embeddings/
```

Identity-pair results:

| Feature | Positive mean cosine | Negative mean cosine | ROC-AUC |
|---|---:|---:|---:|
| Single BEV cell | 0.556 | 0.328 | 0.800 |
| 3×3 ROI pooled | 0.683 | 0.409 | 0.824 |

ROI pooling is better than single-cell sampling on this bounded subset, but
the result is not a learned identity embedding and is not a MOT score.

Lifecycle finding:

- maximum active tracker tracks: 48;
- maximum query states: 190;
- maximum query-state excess over active tracks: 179;
- query expiration in old prototype: not implemented.

Report: `docs/reports/vru_tracking/phase_03_query_analysis.md`

### Phase 4 — cross-camera analysis

Runner: `scripts/vru_cross_camera.py`

Artifact:

```text
phase_04/cross_camera/cross_camera_events.json
phase_04/cross_camera/visualizations/
```

Results:

- 18 geometry-based camera-transition candidates;
- 12/18 preserve the same predicted track ID;
- 6/18 have a missing track on one or both sides;
- 0 events fully manually validated.

Therefore `12/18` is only geometric-candidate continuity, not a validated
cross-camera handoff success rate. The phase status is PARTIAL.

Report: `docs/reports/vru_tracking/phase_04_cross_camera.md`

### Phase 5 — association improvement

Report status is BLOCKED because the first association ablation runner was
terminated before creating `association_ablation.json`. The following
experimental code exists but is not validated:

- `src/bevdrive/tracking/query_association.py`
- `scripts/vru_association_experiment.py`
- `scripts/cache_nuscenes_transforms.py`
- `tests/test_query_association.py`

Do not claim that query-assisted association improved tracking. No completed
E1/E2/E4/E5 comparison exists yet.

Report: `docs/reports/vru_tracking/phase_05_association.md`

## 7. Source-code map

### Existing inference paths

- `scripts/phase12_nuscenes_infer.py`: bounded native nuScenes Phase 1/2 run.
- `scripts/phase12_navsim_infer.py`: bounded NavSim Phase 1/2 run.
- `scripts/evidence_metrics.py`: old bounded detection/tracking metrics.

### Core modules

- `src/bevdrive/tracking/track_manager.py`: Kalman/Hungarian tracker.
- `src/bevdrive/tracking/association.py`: center-distance association and
  optional query-cost association.
- `src/bevdrive/tracking/kalman.py`: constant-velocity filter.
- `src/bevdrive/tracking/temporal_query.py`: temporal BEV memory and query bank.
- `src/bevdrive/tracking/query_association.py`: unvalidated optional query-cost
  tracker.
- `src/bevdrive/geometry/transforms.py`: rigid transform utilities.
- `src/bevdrive/geometry/projection.py`: camera projection utilities.

### Tests

Current local tests include geometry, temporal warp, scene reset, timestamp
handling and query-association smoke tests. The last verified state before this
handoff was:

```text
13 passed
ruff: All checks passed
```

## 8. Reproduction commands

Use `ai_core`, the frozen checkpoint and the read-only mounted dataset. Example
baseline command:

```bash
PYTHONPATH=src:scripts:/home/anhhd/GIT/mmdetection3d \
conda run --no-capture-output -n ai_core \
python scripts/phase12_nuscenes_infer.py \
  --data-root "/media/anhhd/New Volume/nuscenes" \
  --version v1.0-trainval \
  --scene-name scene-0003 --scene-name scene-0004 \
  --frame-limit 100 \
  --checkpoint "/media/anhhd/New Volume/navsim_workspace/checkpoints/bevfusion_demo_mmdet3d.pth" \
  --config /home/anhhd/GIT/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  --output-dir artifacts/experiments/reproduction/nuscenes_baseline \
  --score-thr 0.20 --device cuda:0
```

Detection analysis uses the cached output from:

```text
artifacts/experiments/vru_tracking/phase_01/nuscenes_detection/raw_detections.json
```

Do not rerun full datasets unless explicitly authorized. Do not overwrite
existing artifact directories; create a new versioned experiment directory.

## 9. Git/worktree handoff

Last known pushed baseline commit:

```text
acbab9a Add inference-only temporal query base
```

At the time this checkpoint was written, the following current experimental
work was uncommitted:

- `scripts/phase12_nuscenes_infer.py` native GT-name metadata addition;
- `src/bevdrive/tracking/association.py` query-cost association addition;
- `src/bevdrive/tracking/__init__.py` export;
- `src/bevdrive/tracking/query_association.py`;
- `scripts/vru_*.py` experiment runners;
- `scripts/cache_nuscenes_transforms.py`;
- `tests/test_query_association.py` and `tests/test_tracking_lifecycle.py`;
- `docs/reports/vru_tracking/` reports.

No commit or push was requested after these experimental changes. Inspect
`git status` before making further changes. Generated `artifacts/` are ignored
and should not be committed unless explicitly requested.

## 10. Handoff recommendation

The next agent should:

1. Diagnose why `scripts/vru_association_experiment.py` received SIGTERM and
   rerun it with the transform cache, without changing the detector.
2. Produce a completed association ablation before claiming any improvement.
3. Keep the classical tracker as the default control.
4. Add query expiration/track lifecycle handling before using query states in
   production association.
5. Keep cross-camera results labeled as geometric candidates until manual
   visibility validation is expanded.
6. Do not begin learned UniAD TrackFormer training based only on the current
   query AUC; first complete the inference-only association ablation and verify
   reproducible VRU gains.
