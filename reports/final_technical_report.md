# VRU 3D Detection and Tracking with BEVFusion and a UniAD-style TrackFormer — Technical Report (checkpoint 2)

*Repository:* `BEVPerception`. *Date:* 2026-11-17. *Hardware:* RTX 4050 Laptop
(6 GB), 15 GB RAM. *Scope:* 3D detection and multi-object tracking of
pedestrians, bicycles and motorcycles from 6 cameras + LiDAR on nuScenes.
Motion prediction, occupancy and planning are out of scope.

Every number below was produced in this session by the scripts named, with
the official nuScenes devkit. Nothing is estimated or copied from papers,
except where marked as an external reference.

## 1. Pipeline

```text
6 cameras + LIDAR_TOP (keyframe + 9 sweeps)          nuScenes calibrated_sensor + ego_pose
            │                                                     │
            ▼                                                     ▼
  BEVFusion (frozen, mmdet3d 1.4)                    lidar2global per keyframe, lidar2cam per camera
  Swin-T + LSS camera BEV ⊕ SECOND LiDAR BEV
            │ fused BEV [512, 180(x), 180(y)] @0.6 m
            ├────────────► TransFusion head → 200 proposals: box, class, score, velocity
            │                                              (legacy box convention converted)
            ▼
  Temporal BEV (Phase 2): history buffer (2) → ego-motion warp → motion-gated context
            │
  ┌─────────┴───────────────────────────┬──────────────────────────────────────────┐
  │ Baseline A (Phase 3)                │ Proposed B (Phase 4)                      │
  │ Kalman [x,y,vx,vy] in global frame  │ BEVAdapter: [BEV ; context] → 256 ch,     │
  │ + detector velocity                 │   metric resample to UniAD grid 200² yx   │
  │ class-aware two-pass Hungarian,     │ detection queries ← proposals             │
  │ per-class gates, coasting 3 frames  │ track queries ← t−1 (ego + v·Δt)          │
  │                                     │ 6-layer deformable decoder (UniAD)        │
  │                                     │ lifecycle: birth 0.4 / sleep 0.35 / die 5 │
  │                                     │ MemoryBank + QIM → next frame             │
  └─────────┬───────────────────────────┴───────────────────┬──────────────────────┘
            ▼                                               ▼
  VRU 3D boxes + track ids + velocity (+ 256-d learned track embeddings for B)
```

## 2. What was fixed relative to checkpoint 1 (evidence in `reports/phase_00_audit.md`)

| Defect in checkpoint 1 | Effect (mini_val, devkit) |
|---|---|
| LiDAR sweeps never passed (mmdet3d silently repeated the keyframe) | VRU mAP 0.743 → 0.601, mAVE 0.43 → 1.01 |
| `inference_multi_modality_detector` called per frame → always `data_list[0]`: every frame used the first frame's six images | VRU mAP 0.743 → 0.640 |
| Both together (the checkpoint-1 configuration) | **VRU mAP 0.348**, bicycle AP 0.095 (vs 0.575) |
| Demo checkpoint outputs legacy boxes (w/l swapped, yaw_old) with gravity-centre z | ASE 0.80, AOE 1.6 rad, floating boxes; bike-rack filter errors (bicycle AP 0.552 vs 0.575) |
| BEV axis convention wrong in the temporal warp / query sampling | Heatmap at GT 0.007 (≈ chance) vs 0.631 correct |
| One evaluation scene (scene-0004) was in BEVFusion's training split; custom metrics only | Replaced by mini_val (val-split scenes) and the official devkit |

## 3. Results on nuScenes-mini `mini_val` (scene-0103, scene-0916; 81 keyframes)

### Detection (BEVFusion, frozen, fixed inputs)

mAP 0.587, NDS 0.589.

| Class | AP | ATE (m) | ASE | AOE (rad) | AVE (m/s) |
|---|---:|---:|---:|---:|---:|
| Pedestrian | 0.926 | 0.11 | 0.26 | 0.38 | 0.22 |
| Bicycle | 0.575 | 0.17 | 0.24 | 0.36 | 0.54 |
| Motorcycle | 0.728 | 0.20 | 0.27 | 0.39 | 0.05 |

Pedestrian recall within 2 m: 0.97 at 0–40 m (score ≥ 0.1), and 0.90 even at
30–40 m.

### Tracking

| | AMOTA | AMOTP | IDS | Ped AMOTA | Bicycle AMOTA | Moto AMOTA | VRU AMOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A** BEVFusion + KF/Hungarian | **0.741** | **0.626** | **26** | **0.856** | **0.475** | **0.697** | **0.676** |
| **B** BEVFusion + temporal BEV + TrackFormer (trained on mini_train) | 0.530 | 0.987 | 32 | 0.673 | 0.000 | 0.289 | 0.321 |
| B without temporal BEV | 0.557 | 0.959 | 31 | 0.709 | 0.000 | 0.294 | 0.334 |
| UniAD-B stage 1, camera-only (external reference, same split/devkit) | 0.486 | 1.088 | 38 | 0.532 | 0.148 | 0.454 | 0.378 |

### Runtime

Live end-to-end, RTX 4050, fp32:

* **A:** about 615 ms/frame (1.63 FPS). BEVFusion 425 ms + KF 4.6 ms.
* **B:** 673 ms/frame (1.49 FPS). BEVFusion 424 ms + temporal 12 ms +
  TrackFormer 31 ms.
* Peak GPU memory is about 2.6 GB allocated / 3.2 GB reserved for both.

## 4. What the query tracker is, and evidence that it is a real TrackFormer

* A plain-PyTorch port of UniAD's tracking stack: deformable DETR3D decoder
  with box refinement, MOTR QIM, MemoryBank, runtime lifecycle and
  ClipMatcher loss. Parameter names are identical, so 271 UniAD stage-1
  tensors load directly.
* **Equivalence test.** Fed UniAD's own BEVFormer features and poses, the
  port reproduces UniAD's tracker:
  * 99.4 % identical output track ids;
  * median per-frame worst box error 6.5 mm;
  * AMOTA 0.488 vs UniAD's 0.486;
  * per class 0.535 / 0.148 / 0.454 vs 0.532 / 0.148 / 0.454.
* Identity is carried by the query. There is no Kalman filter or Hungarian
  matching at inference; Hungarian matching appears only in the training
  loss.
* The BEVFusion ↔ UniAD interface gap is bridged explicitly: channels,
  grid, layout, range, positional encoding, reference points, box code and
  temporal memory (table in `reports/phase_04_query_tracking.md` §2).
* Training was actually executed, with BEVFusion frozen: 12 epochs on
  mini_train in 32 min. The untrained smoke tests (`tests/test_trackformer.py`)
  are kept separate and are not used as results.

## 5. Interpretation

1. **The detector dominates VRU tracking quality.** Fixing the inputs more
   than doubled VRU detection mAP (0.348 → 0.743). A simple, well-configured
   KF tracker then reaches VRU AMOTA 0.676 and pedestrian AMOTA 0.856.
2. **The learned query tracker learns but overfits at mini scale** (confirmed
   on the 23-scene val set: VRU AMOTA 0.485 vs 0.626 for A). It reaches
   AMOTA 0.851 on its 8 training scenes, but only 0.530 on held-out scenes;
   the KF baseline goes 0.766 → 0.741. Its specific advantages over A, fewer
   fragmentations (motorcycles 21 → 5) and fewer occlusion misses, are
   outweighed by worse re-classification and re-scoring of proposals.
3. **The temporal BEV is correct but not yet useful.** The ego-motion
   alignment is verified (2.9 cm pose chain, static-object feature cosine
   0.41 → 0.87, heatmap overlays). Gating prevents the smearing that naive
   averaging causes (VRU mAP 0.540 with mean vs 0.746 gated). It gave no
   measurable tracking gain in the small-data regime.

## 6. Larger held-out evaluation: 23 trainval val-split scenes (914 keyframes)

BEVFusion was run on all 84 complete trainval scenes on disk (3,337
keyframes). The 23 val-split scenes form an evaluation set 11× larger than
mini_val: 5,774 pedestrian, 558 bicycle and 110 motorcycle GT. B was **not**
retrained on the 61 train-split scenes. That training was prepared
(`artifacts/checkpoint_2/run_scaleup.sh`) and paused at the user's request,
so B below is the mini-trained model.

| val-23 | AMOTA | IDS | Ped AMOTA | Bicycle AMOTA | Moto AMOTA | **VRU AMOTA** | Detection VRU mAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.589 | 123 | 0.752 | 0.519 | 0.606 | 0.626 | 0.761 (BEVFusion) |
| A, thresholds tuned on mini_train | **0.619** | 143 | **0.793** | **0.567** | **0.643** | **0.668** | — |
| B (mini-trained) | 0.536 | 131 | 0.604 | 0.332 | 0.519 | 0.485 | 0.556 (decoder) |

BEVFusion on val-23 scores mAP 0.653 / NDS 0.693, consistent with published
full-val results. The A > B ranking holds, but the gap is far smaller than
mini_val suggested (0.589 vs 0.536 overall). On cars, B ≈ A (0.775 vs 0.787)
with fewer car ID switches (39 vs 54).

Engineering note: the devkit evaluation on full trainval metadata was
OOM-killed on the 15 GB machine. A verbatim 23-scene metadata subset
(`scripts/cp2_make_slim_nuscenes.py`) fixes this.

## 7. Limitations

* **Evaluation size.** mini_val has 2 scenes. Bicycle results rest on 41 GT
  boxes, about 3 instances. The 23-scene trainval-val evaluation (§6)
  mitigates this, but it is still a subset of the 150-scene val split.
* **Detector checkpoint.** It is a demo conversion with a legacy box
  convention, corrected by a verified output re-parametrisation. No official
  mmdet3d-1.x BEVFusion checkpoint was used.
* **Training data for B.** B was trained on mini_train only (323 keyframes);
  UniAD used 700 scenes. Training on the 61 locally available train-split
  scenes is prepared but was paused at the user's request. No
  hyper-parameter search was done for B.
  The QIM survival rule uses a 1 m distance instead of 3D IoU > 0.5
  (deviation).
* **No appearance re-identification for A.** Long occlusions (> 1.5 s) end
  tracks.
* **Not evaluated.** Cross-camera handoff beyond what the BEV tracking
  implies; night/rain subsets; latency optimisations (fp16/TensorRT).

## 8. Reproduction (env `ai_core` unless stated)

```bash
python scripts/cp2_detect.py --scenes mini_train mini_val --output-dir artifacts/checkpoint_2/phase_01/mini_all \
    --cache-bev artifacts/checkpoint_2/bev_cache/v1.0-mini --vis-dir outputs/phase_01/mini_all          # Phase 1
python scripts/cp2_temporal_bev.py                                                                       # Phase 2
python scripts/cp2_track_baseline.py --records .../records.pkl --detections .../detections.pkl \
    --scenes scene-0103 scene-0916 --output-dir artifacts/checkpoint_2/phase_03/kf_default             # Phase 3 (A)
python scripts/cp2_validate_port.py --dump artifacts/checkpoint_2/phase_04/uniad_port_check_fp32       # Phase 4 port check
python scripts/cp2_train_trackformer.py --output-dir artifacts/checkpoint_2/phase_04/train_main_temporal --epochs 12
python scripts/cp2_track_query.py --checkpoint artifacts/checkpoint_2/phase_04/train_main_temporal/final.pth \
    --output-dir artifacts/checkpoint_2/phase_04/infer_main_temporal                                   # Phase 4/5 (B)
python scripts/cp2_online_pipeline.py --tracker query --output-dir artifacts/checkpoint_2/phase_05/online_B \
    --render outputs/phase_05/online_B                                                                 # live end-to-end
conda run -n uniad2.0 python scripts/cp2_eval_nuscenes.py --task track --result-json <submission_track.json> \
    --output-dir <dir> --eval-set mini_val                                                             # official metrics
```

Per-phase details: `reports/phase_00_audit.md` … `reports/phase_05_evaluation.md`.
